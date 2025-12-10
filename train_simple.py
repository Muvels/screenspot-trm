"""Simple baseline training script for debugging.

Uses a much simpler architecture:
- Concatenate mean-pooled patches + text
- Simple MLP to predict bbox
- No TRM complexity

This helps us understand if the issue is:
1. The embeddings (CLIP features don't contain enough info)
2. The architecture (TRM/cross-attention is too complex)
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

from utils import compute_metrics, compute_iou
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
        return torch.device("cpu")
    return torch.device(requested)


@dataclass
class SimpleConfig:
    parquet_path: str = "dataset/screenspot_training.parquet"
    embeddings_path: str = "dataset/screenspot_training_patch_embeddings.pt"
    val_split: float = 0.1
    
    batch_size: int = 128
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    
    hidden_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.1
    
    checkpoint_dir: str = "checkpoints_simple/"
    samples_dir: str = "samples_simple/"
    gen_samples: int = 5
    interval_samples: int = 500
    
    use_wandb: bool = True
    wandb_project: str = "screenspot-simple"
    device: str = "auto"
    num_workers: int = 4
    seed: int = 42


def load_embeddings(path: str):
    """Load embeddings (supports shards)."""
    import re
    from pathlib import Path
    
    p = Path(path)
    
    # Check for shards
    if not p.exists():
        parent = p.parent
        base_stem = re.sub(r'_shard_\d+$', '', p.stem)
        shards = sorted(parent.glob(f"{base_stem}_shard_*.pt"))
        
        if shards:
            print(f"Loading {len(shards)} shards...")
            all_pooled, all_patches, all_txt = [], [], []
            
            for shard in tqdm(shards, desc="Loading shards"):
                data = torch.load(shard, map_location='cpu', weights_only=False)
                all_pooled.append(data["img_pooled"].float())
                all_patches.append(data["img_patches"].float())
                all_txt.append(data["txt_embeddings"].float())
            
            return {
                "img_pooled": torch.cat(all_pooled),
                "img_patches": torch.cat(all_patches),
                "txt_embeddings": torch.cat(all_txt),
                "embed_dim": data["embed_dim"],
                "patch_dim": data.get("patch_dim", data["embed_dim"]),
                "num_patches": data["num_patches"],
            }
    
    print(f"Loading single file: {path}")
    data = torch.load(path, map_location='cpu', weights_only=False)
    for k in ['img_pooled', 'img_patches', 'txt_embeddings']:
        if data[k].dtype == torch.float16:
            data[k] = data[k].float()
    return data


class SimpleDataset(Dataset):
    def __init__(self, parquet_path, emb_data, split="train", val_split=0.1, seed=42):
        self.df = pd.read_parquet(parquet_path)
        self.img_patches = emb_data["img_patches"]
        self.txt_emb = emb_data["txt_embeddings"]
        
        n = min(len(self.df), len(self.img_patches))
        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=rng).tolist()
        
        val_size = int(n * val_split)
        self.indices = perm[:val_size] if split == "val" else perm[val_size:]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        i = self.indices[idx]
        bbox = self.df.iloc[i]["bbox"]
        bbox = torch.tensor(list(bbox), dtype=torch.float32)
        
        return {
            "patches": self.img_patches[i],
            "txt": self.txt_emb[i],
            "bbox": bbox,
            "idx": i,
        }


def collate_fn(batch):
    return {
        "patches": torch.stack([b["patches"] for b in batch]),
        "txt": torch.stack([b["txt"] for b in batch]),
        "bbox": torch.stack([b["bbox"] for b in batch]),
        "idx": [b["idx"] for b in batch],
    }


class SimpleBBoxModel(nn.Module):
    """Simple baseline: mean-pool patches, concat with text, MLP to bbox."""
    
    def __init__(self, patch_dim=768, txt_dim=512, hidden_dim=512, num_layers=3, dropout=0.1):
        super().__init__()
        
        input_dim = patch_dim + txt_dim
        
        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_d, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        
        self.encoder = nn.Sequential(*layers)
        self.bbox_head = nn.Linear(hidden_dim, 4)
        
        # Init bbox head for center prediction
        nn.init.zeros_(self.bbox_head.bias)
        nn.init.xavier_uniform_(self.bbox_head.weight, gain=0.1)
    
    def forward(self, patches, txt):
        # Mean pool patches: [B, 196, 768] -> [B, 768]
        pooled = patches.mean(dim=1)
        
        # Concat with text: [B, 768+512]
        x = torch.cat([pooled, txt], dim=-1)
        
        # Encode
        h = self.encoder(x)
        
        # Predict bbox as cxcywh, convert to xyxy
        raw = self.bbox_head(h)
        raw = torch.sigmoid(raw)
        
        cx, cy, w, h = raw[..., 0], raw[..., 1], raw[..., 2], raw[..., 3]
        
        x1 = (cx - w / 2).clamp(0, 1)
        y1 = (cy - h / 2).clamp(0, 1)
        x2 = (cx + w / 2).clamp(0, 1)
        y2 = (cy + h / 2).clamp(0, 1)
        
        return torch.stack([x1, y1, x2, y2], dim=-1)


class GIoULoss(nn.Module):
    def __init__(self, coord_weight=1.0, giou_weight=1.0):
        super().__init__()
        self.coord_weight = coord_weight
        self.giou_weight = giou_weight
    
    def forward(self, pred, target):
        # Coord loss
        coord_loss = nn.functional.smooth_l1_loss(pred, target)
        
        # GIoU loss
        x1 = torch.max(pred[..., 0], target[..., 0])
        y1 = torch.max(pred[..., 1], target[..., 1])
        x2 = torch.min(pred[..., 2], target[..., 2])
        y2 = torch.min(pred[..., 3], target[..., 3])
        inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        
        pred_area = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
        target_area = (target[..., 2] - target[..., 0]) * (target[..., 3] - target[..., 1])
        union = pred_area + target_area - inter
        
        iou = inter / union.clamp(min=1e-6)
        
        enc_x1 = torch.min(pred[..., 0], target[..., 0])
        enc_y1 = torch.min(pred[..., 1], target[..., 1])
        enc_x2 = torch.max(pred[..., 2], target[..., 2])
        enc_y2 = torch.max(pred[..., 3], target[..., 3])
        enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)
        
        giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)
        giou_loss = (1 - giou).mean()
        
        return self.coord_weight * coord_loss + self.giou_weight * giou_loss


def train(config: SimpleConfig):
    torch.manual_seed(config.seed)
    device = get_device(config.device)
    print(f"Device: {device}")
    
    use_wandb = config.use_wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(project=config.wandb_project, config=asdict(config), tags=["simple", "baseline"])
    
    # Load data
    emb = load_embeddings(config.embeddings_path)
    patch_dim = emb.get("patch_dim", emb["embed_dim"])
    txt_dim = emb["embed_dim"]
    print(f"patch_dim={patch_dim}, txt_dim={txt_dim}")
    
    df = pd.read_parquet(config.parquet_path)
    
    train_ds = SimpleDataset(config.parquet_path, emb, "train", config.val_split, config.seed)
    val_ds = SimpleDataset(config.parquet_path, emb, "val", config.val_split, config.seed)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, config.batch_size, shuffle=True, 
                              num_workers=config.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, config.batch_size, shuffle=False,
                            num_workers=config.num_workers, collate_fn=collate_fn, pin_memory=True)
    
    # Model
    model = SimpleBBoxModel(patch_dim, txt_dim, config.hidden_dim, config.num_layers, config.dropout).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = GIoULoss(coord_weight=1.0, giou_weight=2.0)
    
    total_steps = config.epochs * len(train_loader)
    warmup_steps = int(total_steps * config.warmup_ratio)
    print(f"Total steps: {total_steps}, warmup: {warmup_steps}")
    
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.samples_dir).mkdir(parents=True, exist_ok=True)
    
    step, best_iou = 0, 0.0
    sample_idx = torch.randperm(len(df))[:config.gen_samples].tolist()
    
    for epoch in range(config.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            patches = batch["patches"].to(device)
            txt = batch["txt"].to(device)
            bbox = batch["bbox"].to(device)
            
            # LR schedule
            if step < warmup_steps:
                lr = config.lr * step / max(warmup_steps, 1)
            else:
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                lr = config.lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))
            
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            
            pred = model(patches, txt)
            loss = criterion(pred, bbox)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}"})
            
            if use_wandb:
                wandb.log({"train/loss": loss.item(), "train/lr": lr}, step=step)
            
            # Generate samples
            if config.gen_samples > 0 and step % config.interval_samples == 0:
                model.eval()
                with torch.no_grad():
                    images, preds, gts, tasks = [], [], [], []
                    for i in sample_idx:
                        row = df.iloc[i]
                        img = load_image_from_parquet_row(row)
                        images.append(img)
                        tasks.append(str(row["task"]))
                        gts.append(torch.tensor(list(row["bbox"])))
                        
                        p = emb["img_patches"][i].unsqueeze(0).to(device)
                        t = emb["txt_embeddings"][i].unsqueeze(0).to(device)
                        pred_box = model(p, t).squeeze(0).cpu()
                        preds.append(pred_box)
                    
                    preds = torch.stack(preds)
                    gts = torch.stack(gts)
                    ious = compute_iou(preds, gts)
                    print(f"  Samples: mean IoU = {ious.mean():.3f}")
                    save_sample_images(images, preds, gts, tasks, config.samples_dir, step)
                model.train()
        
        # Validation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                patches = batch["patches"].to(device)
                txt = batch["txt"].to(device)
                bbox = batch["bbox"].to(device)
                
                pred = model(patches, txt)
                all_preds.append(pred.cpu())
                all_targets.append(bbox.cpu())
        
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        metrics = compute_metrics(all_preds, all_targets)
        
        print(f"Val: IoU={metrics['iou_mean']:.4f}, acc@0.5={metrics['acc@0.5']:.4f}")
        
        if use_wandb:
            wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=step)
        
        if metrics["iou_mean"] > best_iou:
            best_iou = metrics["iou_mean"]
            torch.save({"model": model.state_dict(), "config": asdict(config)}, 
                       Path(config.checkpoint_dir) / "best.pt")
            print(f"  Saved best (IoU: {best_iou:.4f})")
    
    if use_wandb:
        wandb.finish()
    print(f"Done! Best IoU: {best_iou:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    
    config = SimpleConfig()
    if args.embeddings_path:
        config.embeddings_path = args.embeddings_path
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr
    if args.no_wandb:
        config.use_wandb = False
    
    train(config)


if __name__ == "__main__":
    main()
