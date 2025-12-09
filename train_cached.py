"""Training script using pre-computed CLIP embeddings.

This is significantly faster than train.py because:
1. No CLIP model needs to be loaded (saves GPU memory)
2. No image encoding at each batch (saves ~60-70% forward time)

Usage:
    # First, precompute embeddings
    python precompute_embeddings.py --data-path dataset/screenspot_training.parquet
    
    # Then train with cached embeddings
    python train_cached.py --embeddings-path dataset/screenspot_training.embeddings.pt
"""

import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import yaml

from models.fusion import CLIPFusion
from models.trm_core import TRMController
from models.bbox_head import BBoxHead
from utils import (
    BBoxLoss, EMAHelper, compute_metrics, compute_iou,
    log_samples_to_wandb, save_sample_images, load_image_from_parquet_row,
)

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def get_device(requested: str = "auto") -> torch.device:
    """Get the best available device."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("Warning: CUDA not available, falling back to CPU")
            return torch.device("cpu")
    elif requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("Warning: MPS not available, falling back to CPU")
            return torch.device("cpu")
    else:
        return torch.device("cpu")


@dataclass
class TrainConfig:
    """Training configuration for cached mode."""
    # Data
    parquet_path: str = "dataset/screenspot_training.parquet"
    embeddings_path: str = "dataset/screenspot_training.embeddings.pt"
    val_split: float = 0.1
    
    # Training
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    
    # TRM specific
    use_ema: bool = True
    ema_rate: float = 0.999
    deep_supervision: bool = True
    deep_supervision_weight: float = 0.1
    
    # Model config
    trm_hidden_size: int = 256
    H_cycles: int = 3
    L_cycles: int = 4
    L_layers: int = 2
    expansion: float = 4.0
    bbox_hidden_dim: int = 128
    bbox_output_format: str = "xyxy"
    fusion_type: str = "concat_proj"
    
    # Loss
    loss_type: str = "smooth_l1"
    
    # Logging and checkpoints
    eval_interval: int = 1
    log_interval: int = 100
    checkpoint_dir: str = "checkpoints/"
    save_best: bool = True
    
    # Sample visualization
    gen_samples: int = 0           # Number of samples to visualize (0 = disabled)
    interval_samples: int = 500    # Generate samples every N steps
    samples_dir: str = "samples/"  # Directory for local sample storage
    
    # Wandb
    use_wandb: bool = True
    wandb_project: str = "screenspot-trm"
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    
    # Device
    device: str = "auto"
    num_workers: int = 4
    
    # Reproducibility
    seed: int = 42


class CachedEmbeddingDataset(Dataset):
    """Dataset using pre-computed CLIP embeddings."""
    
    def __init__(
        self,
        parquet_path: str,
        embeddings_path: str,
        split: str = "train",
        val_split: float = 0.1,
        seed: int = 42,
    ):
        self.df = pd.read_parquet(parquet_path)
        
        data = torch.load(embeddings_path)
        self.img_embeddings = data["img_embeddings"]
        self.txt_embeddings = data["txt_embeddings"]
        self.embed_dim = data["embed_dim"]
        
        n = len(self.df)
        indices = list(range(n))
        
        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=rng).tolist()
        indices = [indices[i] for i in perm]
        
        val_size = int(n * val_split)
        if split == "val":
            self.indices = indices[:val_size]
        else:
            self.indices = indices[val_size:]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        row_idx = self.indices[idx]
        row = self.df.iloc[row_idx]
        
        bbox = row["bbox"]
        if isinstance(bbox, (list, tuple)):
            bbox = torch.tensor(bbox, dtype=torch.float32)
        else:
            bbox = torch.tensor(list(bbox), dtype=torch.float32)
        
        return {
            "img_emb": self.img_embeddings[row_idx],
            "txt_emb": self.txt_embeddings[row_idx],
            "bbox": bbox,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Tensor]:
    return {
        "img_embs": torch.stack([b["img_emb"] for b in batch]),
        "txt_embs": torch.stack([b["txt_emb"] for b in batch]),
        "bboxes": torch.stack([b["bbox"] for b in batch]),
    }


class TRMOnlyModel(nn.Module):
    """Model without CLIP - uses pre-computed embeddings."""
    
    def __init__(self, config: TrainConfig, clip_dim: int = 768):
        super().__init__()
        
        self.fusion = CLIPFusion(
            clip_dim=clip_dim,
            trm_dim=config.trm_hidden_size,
            fusion_type=config.fusion_type,
        )
        
        self.trm = TRMController(
            hidden_size=config.trm_hidden_size,
            H_cycles=config.H_cycles,
            L_cycles=config.L_cycles,
            L_layers=config.L_layers,
            expansion=config.expansion,
        )
        
        self.bbox_head = BBoxHead(
            input_dim=config.trm_hidden_size,
            hidden_dim=config.bbox_hidden_dim,
            output_format=config.bbox_output_format,
        )
    
    def forward(
        self,
        img_emb: Tensor,
        txt_emb: Tensor,
        return_intermediates: bool = False,
    ):
        h_ctx = self.fusion(img_emb, txt_emb)
        y_final, intermediates = self.trm(h_ctx, return_intermediates)
        bbox_pred = self.bbox_head(y_final)
        
        bbox_intermediates = None
        if intermediates is not None:
            bbox_intermediates = [self.bbox_head(y) for y in intermediates]
        
        return bbox_pred, bbox_intermediates
    
    def count_parameters(self):
        def count_params(module):
            return sum(p.numel() for p in module.parameters())
        
        return {
            "fusion": count_params(self.fusion),
            "trm": count_params(self.trm),
            "bbox_head": count_params(self.bbox_head),
            "total": count_params(self),
        }


def cosine_schedule_with_warmup(
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr_ratio: float = 0.1,
) -> float:
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        img_embs = batch["img_embs"].to(device)
        txt_embs = batch["txt_embs"].to(device)
        bboxes = batch["bboxes"].to(device)
        
        pred_bbox, _ = model(img_embs, txt_embs, return_intermediates=False)
        
        loss = criterion(pred_bbox, bboxes)
        total_loss += loss.item() * len(img_embs)
        
        all_preds.append(pred_bbox.cpu())
        all_targets.append(bboxes.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    metrics = compute_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(dataloader.dataset)
    
    return metrics


@torch.no_grad()
def generate_sample_visualizations(
    model, df, embeddings, sample_indices, device, config, global_step, use_wandb,
):
    """Generate and log/save sample visualizations.
    
    Args:
        model: The model to use for inference
        df: DataFrame with image data
        embeddings: Dict with img_embeddings and txt_embeddings
        sample_indices: Indices of samples to visualize
        device: Device to run inference on
        config: Training config
        global_step: Current training step
        use_wandb: Whether to log to wandb
    """
    model.eval()
    
    images = []
    pred_bboxes = []
    gt_bboxes = []
    tasks = []
    
    img_embeddings = embeddings["img_embeddings"]
    txt_embeddings = embeddings["txt_embeddings"]
    
    for idx in sample_indices:
        # Load original image
        row = df.iloc[idx]
        img = load_image_from_parquet_row(row)
        images.append(img)
        
        # Get task
        tasks.append(str(row["task"]))
        
        # Get ground truth bbox
        bbox = row["bbox"]
        if isinstance(bbox, (list, tuple)):
            gt_bbox = torch.tensor(bbox, dtype=torch.float32)
        else:
            gt_bbox = torch.tensor(list(bbox), dtype=torch.float32)
        gt_bboxes.append(gt_bbox)
        
        # Get embeddings and run inference
        img_emb = img_embeddings[idx].unsqueeze(0).to(device)
        txt_emb = txt_embeddings[idx].unsqueeze(0).to(device)
        
        pred_bbox, _ = model(img_emb, txt_emb, return_intermediates=False)
        pred_bboxes.append(pred_bbox.squeeze(0).cpu())
    
    # Stack tensors
    pred_bboxes = torch.stack(pred_bboxes)
    gt_bboxes = torch.stack(gt_bboxes)
    
    # Compute IoUs
    ious = compute_iou(pred_bboxes, gt_bboxes)
    
    # Log to wandb or save locally
    if use_wandb:
        try:
            log_samples_to_wandb(
                images, pred_bboxes, gt_bboxes, tasks, ious, 
                step=global_step, prefix="samples"
            )
        except Exception as e:
            print(f"Failed to log samples to wandb: {e}")
    
    # Also save locally
    save_sample_images(
        images, pred_bboxes, gt_bboxes, tasks,
        output_dir=config.samples_dir,
        step=global_step,
    )
    
    print(f"  Generated {len(images)} sample visualizations (mean IoU: {ious.mean():.3f})")


def train_epoch(
    model, dataloader, optimizer, criterion, device,
    config, epoch, global_step, total_steps, ema=None,
    use_wandb=False, df=None, embeddings=None, sample_indices=None,
):
    model.train()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    running_loss = 0.0
    epoch_start_time = time.time()
    batch_start_time = time.time()
    samples_processed = 0
    
    for batch_idx, batch in enumerate(pbar):
        img_embs = batch["img_embs"].to(device)
        txt_embs = batch["txt_embs"].to(device)
        bboxes = batch["bboxes"].to(device)
        batch_size = len(img_embs)
        
        lr = cosine_schedule_with_warmup(
            global_step, total_steps, config.warmup_steps, config.lr
        )
        set_lr(optimizer, lr)
        
        pred_bbox, intermediates = model(
            img_embs, txt_embs,
            return_intermediates=config.deep_supervision
        )
        
        loss = criterion(pred_bbox, bboxes, intermediates)
        
        optimizer.zero_grad()
        loss.backward()
        
        if config.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        else:
            grad_norm = 0.0
        
        optimizer.step()
        
        if ema is not None:
            ema.update(model)
        
        running_loss += loss.item()
        samples_processed += batch_size
        global_step += 1
        
        # Logging
        if batch_idx % config.log_interval == 0:
            avg_loss = running_loss / (batch_idx + 1)
            elapsed = time.time() - batch_start_time
            samples_per_sec = samples_processed / elapsed if elapsed > 0 else 0
            
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lr": f"{lr:.2e}",
                "samples/s": f"{samples_per_sec:.1f}"
            })
            
            # Log to wandb
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/loss_avg": avg_loss,
                    "train/lr": lr,
                    "train/grad_norm": grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
                    "train/samples_per_sec": samples_per_sec,
                    "train/epoch": epoch,
                    "train/step": global_step,
                }, step=global_step)
        
        # Generate sample visualizations
        if (config.gen_samples > 0 and 
            global_step % config.interval_samples == 0 and
            df is not None and embeddings is not None and sample_indices is not None):
            generate_sample_visualizations(
                model, df, embeddings, sample_indices, device,
                config, global_step, use_wandb,
            )
            model.train()  # Restore training mode
    
    # Log epoch summary
    epoch_time = time.time() - epoch_start_time
    epoch_loss = running_loss / len(dataloader)
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "train/epoch_loss": epoch_loss,
            "train/epoch_time_sec": epoch_time,
            "train/epoch": epoch,
        }, step=global_step)
    
    return global_step


def train(config: TrainConfig):
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    device = get_device(config.device)
    print(f"Using device: {device}")
    
    # Initialize wandb
    use_wandb = config.use_wandb and WANDB_AVAILABLE
    if config.use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb not installed, logging disabled. Install with: pip install wandb")
    
    if use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name,
            config=asdict(config),
            tags=["screenspot", "trm", "bbox", "cached"],
        )
    
    # Load embeddings
    print(f"Loading embeddings: {config.embeddings_path}")
    emb_data = torch.load(config.embeddings_path)
    clip_dim = emb_data["embed_dim"]
    print(f"  CLIP dim: {clip_dim}")
    print(f"  Samples: {emb_data['n_samples']}")
    
    # Load parquet for sample visualization (if enabled)
    df = None
    sample_indices = None
    if config.gen_samples > 0:
        print(f"Loading parquet for sample visualization...")
        df = pd.read_parquet(config.parquet_path)
        # Select random sample indices for visualization
        n_total = len(df)
        torch.manual_seed(config.seed + 1)  # Different seed for sample selection
        sample_indices = torch.randperm(n_total)[:config.gen_samples].tolist()
        print(f"  Will visualize {config.gen_samples} samples every {config.interval_samples} steps")
    
    # Create model
    print("Creating TRM model (no CLIP)...")
    model = TRMOnlyModel(config, clip_dim=clip_dim)
    model = model.to(device)
    
    # Log parameter counts
    param_counts = model.count_parameters()
    print(f"Model parameters:")
    for k, v in param_counts.items():
        print(f"  {k}: {v:,}")
    
    if use_wandb:
        wandb.log({"model/params_" + k: v for k, v in param_counts.items()}, step=0)
    
    # Create datasets
    print("Loading cached datasets...")
    train_dataset = CachedEmbeddingDataset(
        config.parquet_path,
        config.embeddings_path,
        split="train",
        val_split=config.val_split,
        seed=config.seed,
    )
    val_dataset = CachedEmbeddingDataset(
        config.parquet_path,
        config.embeddings_path,
        split="val",
        val_split=config.val_split,
        seed=config.seed,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    if use_wandb:
        wandb.log({
            "data/train_samples": len(train_dataset),
            "data/val_samples": len(val_dataset),
            "data/clip_dim": clip_dim,
        }, step=0)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    
    criterion = BBoxLoss(
        loss_type=config.loss_type,
        deep_supervision_weight=config.deep_supervision_weight if config.deep_supervision else 0.0,
    )
    
    ema = None
    if config.use_ema:
        ema = EMAHelper(mu=config.ema_rate)
        ema.register(model)
    
    steps_per_epoch = len(train_loader)
    total_steps = config.epochs * steps_per_epoch
    print(f"Total training steps: {total_steps}")
    
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    with open(checkpoint_dir / "train_config.yaml", "w") as f:
        yaml.dump(asdict(config), f)
    
    global_step = 0
    best_iou = 0.0
    
    for epoch in range(config.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{config.epochs}")
        print(f"{'='*50}")
        
        global_step = train_epoch(
            model, train_loader, optimizer, criterion, device,
            config, epoch + 1, global_step, total_steps, ema,
            use_wandb=use_wandb, df=df, embeddings=emb_data, sample_indices=sample_indices,
        )
        
        if (epoch + 1) % config.eval_interval == 0:
            eval_model = ema.ema_copy(model) if ema else model
            eval_model = eval_model.to(device)
            
            metrics = evaluate(eval_model, val_loader, criterion, device)
            
            print(f"\nValidation metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
            
            # Log to wandb
            if use_wandb:
                val_metrics = {f"val/{k}": v for k, v in metrics.items()}
                val_metrics["val/best_iou"] = max(best_iou, metrics["iou_mean"])
                val_metrics["val/epoch"] = epoch + 1
                wandb.log(val_metrics, step=global_step)
            
            if config.save_best and metrics["iou_mean"] > best_iou:
                best_iou = metrics["iou_mean"]
                save_path = checkpoint_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": eval_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                    "config": asdict(config),
                    "clip_dim": clip_dim,
                }, save_path)
                print(f"Saved best model (IoU: {best_iou:.4f})")
                
                # Log model artifact
                if use_wandb:
                    artifact = wandb.Artifact(
                        name=f"best-model-{wandb.run.id}",
                        type="model",
                        metadata={"iou_mean": best_iou, "epoch": epoch + 1}
                    )
                    artifact.add_file(str(save_path))
                    wandb.log_artifact(artifact)
            
            if ema:
                del eval_model
    
    final_model = ema.ema_copy(model) if ema else model
    torch.save({
        "epoch": config.epochs,
        "model_state_dict": final_model.state_dict(),
        "config": asdict(config),
        "clip_dim": clip_dim,
    }, checkpoint_dir / "final_model.pt")
    
    # Log final summary
    if use_wandb:
        wandb.log({
            "final/best_iou": best_iou,
            "final/epochs_trained": config.epochs,
        }, step=global_step)
        wandb.finish()
    
    print(f"\nTraining complete!")
    print(f"Best validation IoU: {best_iou:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train with cached embeddings")
    
    parser.add_argument("--parquet-path", type=str, default=None)
    parser.add_argument("--embeddings-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    
    # Wandb options
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    
    # Sample visualization
    parser.add_argument("--gen-samples", type=int, default=None,
                        help="Number of samples to visualize during training")
    parser.add_argument("--interval-samples", type=int, default=None,
                        help="Generate sample visualizations every N steps")
    
    args = parser.parse_args()
    
    config = TrainConfig()
    
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
            for k, v in cfg.items():
                if hasattr(config, k):
                    setattr(config, k, v)
    
    if args.parquet_path:
        config.parquet_path = args.parquet_path
    if args.embeddings_path:
        config.embeddings_path = args.embeddings_path
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.epochs = args.epochs
    if args.lr:
        config.lr = args.lr
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.device:
        config.device = args.device
    
    # Wandb overrides
    if args.wandb:
        config.use_wandb = True
    if args.no_wandb:
        config.use_wandb = False
    if args.wandb_project:
        config.wandb_project = args.wandb_project
    if args.wandb_run_name:
        config.wandb_run_name = args.wandb_run_name
    
    # Sample visualization overrides
    if args.gen_samples is not None:
        config.gen_samples = args.gen_samples
    if args.interval_samples is not None:
        config.interval_samples = args.interval_samples
    
    train(config)


if __name__ == "__main__":
    main()
