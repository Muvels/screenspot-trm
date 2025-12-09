"""Training script V2 (On-the-fly): Computes CLIP embeddings during training.

This version does NOT require precomputed embeddings - it computes them on-the-fly.
This trades compute time for memory savings (no 30-60GB embeddings file needed).

Advantages:
- No precomputation step required
- No large embeddings file to store
- Lower RAM usage (only batch-sized embeddings in memory)

Disadvantages:
- Slightly slower training (CLIP forward pass each batch)
- Requires GPU memory for CLIP model during training

Usage:
    python train_v2.py --parquet-path dataset/screenspot_training.parquet --wandb --epochs 50
"""

import argparse
import io
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import yaml

from models.clip_backbone_patches import CLIPBackboneWithPatches
from models.fusion_attention import CrossAttentionFusion, SpatialAwareFusion
from models.trm_core import TRMController
from models.bbox_head import BBoxHead
from utils import BBoxLoss, EMAHelper, compute_metrics, compute_iou
from utils import log_samples_to_wandb, save_sample_images, load_image_from_parquet_row

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif requested == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    elif requested == "mps":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    else:
        return torch.device("cpu")


@dataclass
class TrainConfig:
    """Training configuration V2 (on-the-fly embeddings)."""
    # Data
    parquet_path: str = "dataset/screenspot_training.parquet"
    val_split: float = 0.1
    
    # CLIP model
    clip_model: str = "ViT-B-16"
    clip_pretrained: str = "openai"
    
    # Training - tuned for better convergence
    batch_size: int = 32  # Smaller due to CLIP overhead
    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 2000
    grad_clip: float = 1.0
    
    # TRM - larger capacity
    use_ema: bool = True
    ema_rate: float = 0.999
    deep_supervision: bool = True
    deep_supervision_weight: float = 0.2
    
    # Model - improved architecture
    trm_hidden_size: int = 384
    H_cycles: int = 4
    L_cycles: int = 6
    L_layers: int = 2
    expansion: float = 4.0
    bbox_hidden_dim: int = 256
    bbox_output_format: str = "cxcywh"
    fusion_type: str = "cross_attention"
    fusion_num_heads: int = 8
    fusion_num_layers: int = 2
    
    # Loss
    loss_type: str = "giou"
    use_combined_loss: bool = True
    coord_weight: float = 1.0
    giou_weight: float = 2.0
    
    # Logging
    eval_interval: int = 1
    log_interval: int = 50
    checkpoint_dir: str = "checkpoints_v2/"
    save_best: bool = True
    
    # Sample visualization
    gen_samples: int = 8
    interval_samples: int = 500
    samples_dir: str = "samples_v2/"
    
    # Wandb
    use_wandb: bool = True
    wandb_project: str = "screenspot-trm-v2"
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    
    # Device
    device: str = "auto"
    num_workers: int = 4
    seed: int = 42


class OnTheFlyDataset(Dataset):
    """Dataset that returns raw images and text for on-the-fly embedding."""
    
    def __init__(
        self,
        parquet_path: str,
        preprocess,  # CLIP preprocessing transform
        split: str = "train",
        val_split: float = 0.1,
        seed: int = 42,
    ):
        self.df = pd.read_parquet(parquet_path)
        self.preprocess = preprocess
        
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
    
    def _load_image(self, row):
        """Load and preprocess image from parquet row."""
        img_data = row["image"]
        if isinstance(img_data, dict):
            img_bytes = img_data["bytes"]
        else:
            img_bytes = img_data
        
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return self.preprocess(image)
    
    def __getitem__(self, idx):
        row_idx = self.indices[idx]
        row = self.df.iloc[row_idx]
        
        # Load and preprocess image
        image_tensor = self._load_image(row)
        
        # Get text
        task = str(row["task"])
        
        # Get bbox
        bbox = row["bbox"]
        if isinstance(bbox, (list, tuple)):
            bbox = torch.tensor(bbox, dtype=torch.float32)
        else:
            bbox = torch.tensor(list(bbox), dtype=torch.float32)
        
        return {
            "image": image_tensor,
            "task": task,
            "bbox": bbox,
            "row_idx": row_idx,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "tasks": [b["task"] for b in batch],
        "bboxes": torch.stack([b["bbox"] for b in batch]),
        "row_indices": [b["row_idx"] for b in batch],
    }


class TRMModelV2(nn.Module):
    """TRM model with cross-attention fusion (same as cached version)."""
    
    def __init__(self, config: TrainConfig, clip_dim: int = 512, patch_dim: int = 768):
        super().__init__()
        
        self.clip_dim = clip_dim
        self.patch_dim = patch_dim
        
        if config.fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(
                clip_dim=patch_dim,
                txt_dim=clip_dim,
                trm_dim=config.trm_hidden_size,
                num_heads=config.fusion_num_heads,
                num_layers=config.fusion_num_layers,
            )
        else:
            self.fusion = SpatialAwareFusion(
                clip_dim=patch_dim,
                txt_dim=clip_dim,
                trm_dim=config.trm_hidden_size,
                num_heads=config.fusion_num_heads,
                num_layers=config.fusion_num_layers,
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
    
    def forward(self, img_patches, txt_emb, img_pooled=None, return_intermediates=False):
        h_ctx = self.fusion(img_patches, txt_emb, img_pooled)
        y_final, intermediates = self.trm(h_ctx, return_intermediates)
        bbox_pred = self.bbox_head(y_final)
        
        bbox_intermediates = None
        if intermediates is not None:
            bbox_intermediates = [self.bbox_head(y) for y in intermediates]
        
        return bbox_pred, bbox_intermediates
    
    def count_parameters(self):
        def count(m):
            return sum(p.numel() for p in m.parameters())
        return {
            "fusion": count(self.fusion),
            "trm": count(self.trm),
            "bbox_head": count(self.bbox_head),
            "total": count(self),
        }


class CombinedLoss(nn.Module):
    """Combined coordinate + GIoU loss."""
    
    def __init__(self, coord_weight=1.0, giou_weight=2.0, ds_weight=0.2):
        super().__init__()
        self.coord_loss = nn.SmoothL1Loss()
        self.coord_weight = coord_weight
        self.giou_weight = giou_weight
        self.ds_weight = ds_weight
    
    def _giou_loss(self, pred, target):
        pred_x1 = torch.min(pred[..., 0], pred[..., 2])
        pred_y1 = torch.min(pred[..., 1], pred[..., 3])
        pred_x2 = torch.max(pred[..., 0], pred[..., 2])
        pred_y2 = torch.max(pred[..., 1], pred[..., 3])
        
        inter_x1 = torch.max(pred_x1, target[..., 0])
        inter_y1 = torch.max(pred_y1, target[..., 1])
        inter_x2 = torch.min(pred_x2, target[..., 2])
        inter_y2 = torch.min(pred_y2, target[..., 3])
        inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target[..., 2] - target[..., 0]) * (target[..., 3] - target[..., 1])
        union = pred_area + target_area - inter
        
        iou = inter / union.clamp(min=1e-6)
        
        enc_x1 = torch.min(pred_x1, target[..., 0])
        enc_y1 = torch.min(pred_y1, target[..., 1])
        enc_x2 = torch.max(pred_x2, target[..., 2])
        enc_y2 = torch.max(pred_y2, target[..., 3])
        enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)
        
        giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)
        return (1 - giou).mean()
    
    def forward(self, pred, target, intermediates=None):
        coord = self.coord_loss(pred, target)
        giou = self._giou_loss(pred, target)
        
        main_loss = self.coord_weight * coord + self.giou_weight * giou
        
        if intermediates and len(intermediates) > 1:
            ds_losses = []
            for p in intermediates[:-1]:
                ds_losses.append(
                    self.coord_weight * self.coord_loss(p, target) +
                    self.giou_weight * self._giou_loss(p, target)
                )
            ds_loss = sum(ds_losses) / len(ds_losses)
            return main_loss + self.ds_weight * ds_loss
        
        return main_loss


def cosine_schedule_with_warmup(step, total_steps, warmup_steps, base_lr, min_lr_ratio=0.01):
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


@torch.no_grad()
def evaluate(model, clip_encoder, dataloader, criterion, device):
    """Evaluate model with on-the-fly embeddings."""
    model.eval()
    clip_encoder.eval()
    
    total_loss = 0.0
    all_preds, all_targets = [], []
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        images = batch["images"].to(device)
        tasks = batch["tasks"]
        bboxes = batch["bboxes"].to(device)
        
        # Compute embeddings on-the-fly
        img_pooled, img_patches, txt_embs = clip_encoder(images, tasks)
        
        pred, _ = model(img_patches, txt_embs, return_intermediates=False)
        loss = criterion(pred, bboxes)
        total_loss += loss.item() * len(images)
        
        all_preds.append(pred.cpu())
        all_targets.append(bboxes.cpu())
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    metrics = compute_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(dataloader.dataset)
    return metrics


@torch.no_grad()
def generate_samples(model, clip_encoder, df, indices, device, config, step, use_wandb):
    """Generate sample predictions for visualization."""
    model.eval()
    clip_encoder.eval()
    
    images, preds, gts, tasks = [], [], [], []
    
    for idx in indices:
        row = df.iloc[idx]
        img = load_image_from_parquet_row(row)
        images.append(img)
        tasks.append(str(row["task"]))
        
        bbox = row["bbox"]
        gt = torch.tensor(bbox if isinstance(bbox, list) else list(bbox), dtype=torch.float32)
        gts.append(gt)
        
        # Preprocess and encode
        img_tensor = clip_encoder.preprocess(img).unsqueeze(0).to(device)
        _, img_patches, txt_emb = clip_encoder(img_tensor, [str(row["task"])])
        
        pred, _ = model(img_patches, txt_emb)
        preds.append(pred.squeeze(0).cpu())
    
    preds = torch.stack(preds)
    gts = torch.stack(gts)
    ious = compute_iou(preds, gts)
    
    if use_wandb:
        try:
            log_samples_to_wandb(images, preds, gts, tasks, ious, step, "samples")
        except Exception as e:
            print(f"wandb sample log failed: {e}")
    
    save_sample_images(images, preds, gts, tasks, config.samples_dir, step)
    print(f"  Samples: mean IoU = {ious.mean():.3f}")


def train_epoch(
    model, clip_encoder, loader, optimizer, criterion, device, config,
    epoch, step, total_steps, ema, use_wandb, df=None, sample_idx=None
):
    """Train for one epoch with on-the-fly embedding computation."""
    model.train()
    clip_encoder.eval()  # Keep CLIP frozen
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    running_loss = 0.0
    t0 = time.time()
    
    for i, batch in enumerate(pbar):
        images = batch["images"].to(device)
        tasks = batch["tasks"]
        bboxes = batch["bboxes"].to(device)
        
        # Compute embeddings on-the-fly (no gradient for CLIP)
        with torch.no_grad():
            img_pooled, img_patches, txt_embs = clip_encoder(images, tasks)
        
        lr = cosine_schedule_with_warmup(step, total_steps, config.warmup_steps, config.lr)
        set_lr(optimizer, lr)
        
        pred, intermediates = model(img_patches, txt_embs, return_intermediates=config.deep_supervision)
        loss = criterion(pred, bboxes, intermediates)
        
        optimizer.zero_grad()
        loss.backward()
        if config.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        
        if ema:
            ema.update(model)
        
        running_loss += loss.item()
        step += 1
        
        if i % config.log_interval == 0:
            avg_loss = running_loss / (i + 1)
            elapsed = time.time() - t0
            sps = (i + 1) * config.batch_size / elapsed
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}", "sps": f"{sps:.0f}"})
            
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({"train/loss": loss.item(), "train/lr": lr, "train/step": step}, step=step)
        
        if config.gen_samples > 0 and step % config.interval_samples == 0 and df is not None:
            generate_samples(model, clip_encoder, df, sample_idx, device, config, step, use_wandb)
            model.train()
    
    return step


def train(config: TrainConfig):
    torch.manual_seed(config.seed)
    device = get_device(config.device)
    print(f"Device: {device}")
    
    use_wandb = config.use_wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name,
            config=asdict(config),
            tags=["v2", "on-the-fly"]
        )
    
    # Load CLIP encoder (frozen, for on-the-fly embeddings)
    print(f"Loading CLIP model: {config.clip_model}")
    clip_encoder = CLIPBackboneWithPatches(
        model_name=config.clip_model,
        pretrained=config.clip_pretrained,
        device=str(device),
    )
    clip_encoder.freeze()
    clip_dim = clip_encoder.embed_dim  # 512
    patch_dim = clip_encoder.patch_dim  # 768
    print(f"  CLIP dim: {clip_dim}, patch dim: {patch_dim}")
    
    # Load parquet for sample visualization
    df = pd.read_parquet(config.parquet_path)
    sample_idx = torch.randperm(len(df))[:config.gen_samples].tolist() if config.gen_samples > 0 else None
    
    # Create model
    model = TRMModelV2(config, clip_dim=clip_dim, patch_dim=patch_dim).to(device)
    params = model.count_parameters()
    print(f"Model parameters: {params}")
    if use_wandb:
        wandb.log({f"model/{k}": v for k, v in params.items()}, step=0)
    
    # Create datasets with CLIP preprocessing
    train_ds = OnTheFlyDataset(
        config.parquet_path,
        clip_encoder.preprocess,
        "train",
        config.val_split,
        config.seed
    )
    val_ds = OnTheFlyDataset(
        config.parquet_path,
        clip_encoder.preprocess,
        "val",
        config.val_split,
        config.seed
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    train_loader = DataLoader(
        train_ds, config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # Optimizer & Loss (only for TRM model, not CLIP)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = CombinedLoss(config.coord_weight, config.giou_weight, config.deep_supervision_weight)
    
    ema = EMAHelper(config.ema_rate) if config.use_ema else None
    if ema:
        ema.register(model)
    
    total_steps = config.epochs * len(train_loader)
    print(f"Total steps: {total_steps}")
    
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    step, best_iou = 0, 0.0
    
    for epoch in range(config.epochs):
        print(f"\n{'='*50}\nEpoch {epoch+1}/{config.epochs}\n{'='*50}")
        
        step = train_epoch(
            model, clip_encoder, train_loader, optimizer, criterion,
            device, config, epoch+1, step, total_steps, ema, use_wandb,
            df, sample_idx
        )
        
        if (epoch + 1) % config.eval_interval == 0:
            eval_model = ema.ema_copy(model).to(device) if ema else model
            metrics = evaluate(eval_model, clip_encoder, val_loader, criterion, device)
            
            print("Validation:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
            
            if use_wandb:
                wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=step)
                wandb.log({"val/best_iou": max(best_iou, metrics["iou_mean"])}, step=step)
            
            if metrics["iou_mean"] > best_iou:
                best_iou = metrics["iou_mean"]
                torch.save({
                    "model": eval_model.state_dict(),
                    "config": asdict(config),
                    "metrics": metrics,
                    "epoch": epoch+1
                }, ckpt_dir / "best.pt")
                print(f"Saved best (IoU: {best_iou:.4f})")
            
            if ema:
                del eval_model
    
    if use_wandb:
        wandb.finish()
    print(f"\nDone! Best IoU: {best_iou:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train TRM V2 with on-the-fly embeddings")
    parser.add_argument("--parquet-path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--gen-samples", type=int, default=None)
    parser.add_argument("--interval-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--clip-model", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    
    args = parser.parse_args()
    config = TrainConfig()
    
    if args.parquet_path:
        config.parquet_path = args.parquet_path
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr
    if args.wandb:
        config.use_wandb = True
    if args.no_wandb:
        config.use_wandb = False
    if args.gen_samples is not None:
        config.gen_samples = args.gen_samples
    if args.interval_samples:
        config.interval_samples = args.interval_samples
    if args.device:
        config.device = args.device
    if args.clip_model:
        config.clip_model = args.clip_model
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    
    train(config)


if __name__ == "__main__":
    main()
