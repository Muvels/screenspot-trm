"""Training script for ScreenSpot TRM bounding box model."""

import argparse
import json
import math
import os
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
from utils import BBoxLoss, EMAHelper, compute_metrics


def get_device(requested: str = "auto") -> torch.device:
    """Get the best available device.
    
    Args:
        requested: Device string ("auto", "cuda", "mps", "cpu")
        
    Returns:
        torch.device for the selected device
    """
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
    
    # Device
    device: str = "auto"  # auto, cuda, mps, cpu
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
    """Compute learning rate with cosine decay and linear warmup.
    
    Args:
        step: Current step
        total_steps: Total training steps
        warmup_steps: Number of warmup steps
        base_lr: Base learning rate
        min_lr_ratio: Minimum LR as fraction of base
        
    Returns:
        Learning rate for current step
    """
    if step < warmup_steps:
        # Linear warmup
        return base_lr * step / max(warmup_steps, 1)
    else:
        # Cosine decay
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
    """Evaluate model on validation set.
    
    Args:
        model: Model to evaluate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to use
        
    Returns:
        Dictionary of metric names to values
    """
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
        
        # Forward
        pred_bbox, _ = model(images, tasks, return_intermediates=False)
        
        # Loss
        loss = criterion(pred_bbox, bboxes)
        total_loss += loss.item() * len(images)
        
        # Collect predictions
        all_preds.append(pred_bbox.cpu())
        all_targets.append(bboxes.cpu())
        all_image_sizes.extend(image_sizes)
    
    # Aggregate
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    metrics = compute_metrics(all_preds, all_targets, all_image_sizes)
    metrics["loss"] = total_loss / len(dataloader.dataset)
    
    return metrics


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
) -> int:
    """Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        config: Training config
        epoch: Current epoch number
        global_step: Current global step
        total_steps: Total training steps
        ema: Optional EMA helper
        
    Returns:
        Updated global step
    """
    model.train()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    running_loss = 0.0
    
    for batch_idx, batch in enumerate(pbar):
        images = batch["images"].to(device)
        tasks = batch["tasks"]
        bboxes = batch["bboxes"].to(device)
        
        # Update learning rate
        lr = cosine_schedule_with_warmup(
            global_step, total_steps, config.warmup_steps, config.lr
        )
        set_lr(optimizer, lr)
        
        # Forward
        pred_bbox, intermediates = model(
            images, tasks,
            return_intermediates=config.deep_supervision
        )
        
        # Loss
        loss = criterion(pred_bbox, bboxes, intermediates)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip
            )
        
        optimizer.step()
        
        # EMA update
        if ema is not None:
            ema.update(model)
        
        # Logging
        running_loss += loss.item()
        global_step += 1
        
        if batch_idx % config.log_interval == 0:
            avg_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"})
    
    return global_step


def train(config: TrainConfig, model_config: ModelConfig) -> None:
    """Main training function.
    
    Args:
        config: Training configuration
        model_config: Model configuration
    """
    # Set seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Device
    device = get_device(config.device)
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    model = ScreenBBoxTRMModel(config=model_config, device=str(device))
    model = model.to(device)
    
    # Print parameter counts
    param_counts = model.count_parameters()
    print(f"Model parameters:")
    for k, v in param_counts.items():
        print(f"  {k}: {v:,}")
    
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
    
    # Optimizer (only trainable params)
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
            config, epoch + 1, global_step, total_steps, ema
        )
        
        # Evaluate
        if (epoch + 1) % config.eval_interval == 0:
            # Use EMA model for evaluation if available
            eval_model = ema.ema_copy(model) if ema else model
            eval_model = eval_model.to(device)
            
            metrics = evaluate(eval_model, val_loader, criterion, device)
            
            print(f"\nValidation metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
            
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
            
            # Clean up
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
    
    # Run training
    train(train_config, model_config)


if __name__ == "__main__":
    main()
