"""Training script for ScreenSpot TRM bounding box model."""

import argparse
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import yaml

from models import ScreenBBoxTRMModel
from models.screen_trm_model import ModelConfig
from data import ScreenSpotDataset, collate_fn
from utils import (
    BBoxLoss, EMAHelper, compute_metrics, compute_iou,
    log_samples_to_wandb, save_sample_images, draw_bbox_on_image,
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
    """Training configuration."""
    # Data
    train_path: str = "dataset/screenspot_training.parquet"
    val_split: float = 0.1
    
    # Training
    batch_size: int = 32
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


def cosine_schedule_with_warmup(
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr_ratio: float = 0.1,
) -> float:
    """Compute learning rate with cosine decay and linear warmup."""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set learning rate for all parameter groups."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_image_sizes = []
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        images = batch["images"].to(device)
        tasks = batch["tasks"]
        bboxes = batch["bboxes"].to(device)
        image_sizes = batch["image_sizes"]
        
        pred_bbox, _ = model(images, tasks, return_intermediates=False)
        
        loss = criterion(pred_bbox, bboxes)
        total_loss += loss.item() * len(images)
        
        all_preds.append(pred_bbox.cpu())
        all_targets.append(bboxes.cpu())
        all_image_sizes.extend(image_sizes)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    metrics = compute_metrics(all_preds, all_targets, all_image_sizes)
    metrics["loss"] = total_loss / len(dataloader.dataset)
    
    return metrics


@torch.no_grad()
def generate_sample_visualizations(
    model, val_dataset, sample_indices, device, config, global_step, use_wandb,
):
    """Generate and log/save sample visualizations.
    
    Args:
        model: The model to use for inference
        val_dataset: Validation dataset
        sample_indices: Indices of samples to visualize
        device: Device to run inference on
        config: Training config
        global_step: Current training step
        use_wandb: Whether to log to wandb
    """
    from PIL import Image
    import io
    import pandas as pd
    
    model.eval()
    
    images = []
    pred_bboxes = []
    gt_bboxes = []
    tasks = []
    
    for idx in sample_indices:
        sample = val_dataset[idx]
        
        # Get original image (before CLIP preprocessing)
        row_idx = val_dataset.indices[idx]
        row = val_dataset.df.iloc[row_idx]
        
        img_data = row["image"]
        if isinstance(img_data, dict):
            img_bytes = img_data["bytes"]
        else:
            img_bytes = img_data
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        images.append(img)
        
        # Get task and ground truth
        tasks.append(sample["task"])
        gt_bboxes.append(sample["bbox"])
        
        # Run inference
        img_tensor = sample["image"].unsqueeze(0).to(device)
        task = [sample["task"]]
        
        pred_bbox, _ = model(img_tensor, task, return_intermediates=False)
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
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: TrainConfig,
    epoch: int,
    global_step: int,
    total_steps: int,
    ema: Optional[EMAHelper] = None,
    use_wandb: bool = False,
    val_dataset = None,
    sample_indices = None,
) -> int:
    """Train for one epoch."""
    model.train()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    running_loss = 0.0
    epoch_start_time = time.time()
    batch_start_time = time.time()
    samples_processed = 0
    
    for batch_idx, batch in enumerate(pbar):
        images = batch["images"].to(device)
        tasks = batch["tasks"]
        bboxes = batch["bboxes"].to(device)
        batch_size = len(images)
        
        lr = cosine_schedule_with_warmup(
            global_step, total_steps, config.warmup_steps, config.lr
        )
        set_lr(optimizer, lr)
        
        pred_bbox, intermediates = model(
            images, tasks,
            return_intermediates=config.deep_supervision
        )
        
        loss = criterion(pred_bbox, bboxes, intermediates)
        
        optimizer.zero_grad()
        loss.backward()
        
        if config.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip
            )
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
            val_dataset is not None and sample_indices is not None):
            generate_sample_visualizations(
                model, val_dataset, sample_indices, device,
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


def train(config: TrainConfig, model_config: ModelConfig) -> None:
    """Main training function."""
    # Set seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Device
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
            config={
                "train": asdict(config),
                "model": asdict(model_config),
            },
            tags=["screenspot", "trm", "bbox"],
        )
    
    # Create model
    print("Creating model...")
    model = ScreenBBoxTRMModel(config=model_config, device=str(device))
    model = model.to(device)
    
    # Print and log parameter counts
    param_counts = model.count_parameters()
    print(f"Model parameters:")
    for k, v in param_counts.items():
        print(f"  {k}: {v:,}")
    
    if use_wandb:
        wandb.log({"model/params_" + k: v for k, v in param_counts.items()}, step=0)
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = ScreenSpotDataset(
        config.train_path,
        model.preprocess,
        split="train",
        val_split=config.val_split,
        seed=config.seed,
    )
    val_dataset = ScreenSpotDataset(
        config.train_path,
        model.preprocess,
        split="val",
        val_split=config.val_split,
        seed=config.seed,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Select sample indices for visualization
    sample_indices = None
    if config.gen_samples > 0:
        torch.manual_seed(config.seed + 1)
        sample_indices = torch.randperm(len(val_dataset))[:config.gen_samples].tolist()
        print(f"Will visualize {config.gen_samples} samples every {config.interval_samples} steps")
    
    if use_wandb:
        wandb.log({
            "data/train_samples": len(train_dataset),
            "data/val_samples": len(val_dataset),
        }, step=0)
    
    # Dataloaders
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
    
    # Optimizer
    trainable_params = model.get_trainable_params()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    
    # Loss
    criterion = BBoxLoss(
        loss_type=config.loss_type,
        deep_supervision_weight=config.deep_supervision_weight if config.deep_supervision else 0.0,
    )
    
    # EMA
    ema = None
    if config.use_ema:
        ema = EMAHelper(mu=config.ema_rate)
        ema.register(model)
    
    # Calculate total steps
    steps_per_epoch = len(train_loader)
    total_steps = config.epochs * steps_per_epoch
    print(f"Total training steps: {total_steps}")
    
    # Checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configs
    with open(checkpoint_dir / "train_config.yaml", "w") as f:
        yaml.dump(asdict(config), f)
    with open(checkpoint_dir / "model_config.yaml", "w") as f:
        yaml.dump(asdict(model_config), f)
    
    # Training loop
    global_step = 0
    best_iou = 0.0
    
    for epoch in range(config.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{config.epochs}")
        print(f"{'='*50}")
        
        # Train
        global_step = train_epoch(
            model, train_loader, optimizer, criterion, device,
            config, epoch + 1, global_step, total_steps, ema,
            use_wandb=use_wandb, val_dataset=val_dataset, sample_indices=sample_indices,
        )
        
        # Evaluate
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
            
            # Save best model
            if config.save_best and metrics["iou_mean"] > best_iou:
                best_iou = metrics["iou_mean"]
                save_path = checkpoint_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": eval_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                    "model_config": asdict(model_config),
                    "train_config": asdict(config),
                }, save_path)
                print(f"Saved best model (IoU: {best_iou:.4f})")
                
                # Log model artifact to wandb
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
    
    # Save final model
    final_model = ema.ema_copy(model) if ema else model
    torch.save({
        "epoch": config.epochs,
        "model_state_dict": final_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "train_config": asdict(config),
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
    print(f"Checkpoints saved to: {checkpoint_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train ScreenSpot TRM model")
    
    # Config files
    parser.add_argument("--train-config", type=str, default=None,
                        help="Path to training config YAML")
    parser.add_argument("--model-config", type=str, default=None,
                        help="Path to model config YAML")
    
    # Override common options
    parser.add_argument("--train-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    
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
    
    # Load configs
    train_config = TrainConfig()
    model_config = ModelConfig()
    
    if args.train_config:
        with open(args.train_config) as f:
            train_dict = yaml.safe_load(f)
            for k, v in train_dict.items():
                if hasattr(train_config, k):
                    setattr(train_config, k, v)
    
    if args.model_config:
        with open(args.model_config) as f:
            model_dict = yaml.safe_load(f)
            for k, v in model_dict.items():
                if hasattr(model_config, k):
                    setattr(model_config, k, v)
    
    # Apply command line overrides
    if args.train_path:
        train_config.train_path = args.train_path
    if args.batch_size:
        train_config.batch_size = args.batch_size
    if args.epochs:
        train_config.epochs = args.epochs
    if args.lr:
        train_config.lr = args.lr
    if args.checkpoint_dir:
        train_config.checkpoint_dir = args.checkpoint_dir
    if args.device:
        train_config.device = args.device
    
    # Wandb overrides
    if args.wandb:
        train_config.use_wandb = True
    if args.no_wandb:
        train_config.use_wandb = False
    if args.wandb_project:
        train_config.wandb_project = args.wandb_project
    if args.wandb_run_name:
        train_config.wandb_run_name = args.wandb_run_name
    
    # Sample visualization overrides
    if args.gen_samples is not None:
        train_config.gen_samples = args.gen_samples
    if args.interval_samples is not None:
        train_config.interval_samples = args.interval_samples
    
    # Run training
    train(train_config, model_config)


if __name__ == "__main__":
    main()
