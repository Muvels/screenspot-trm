---
name: ScreenSpot TRM BBox Model
overview: Design a screen UI bounding-box prediction model combining frozen CLIP vision-text encoder, a Tiny Recursive Model (TRM) controller, and a regression head for normalized bbox output [x1, y1, x2, y2].
todos:
  - id: clip-backbone
    content: Implement CLIPBackbone wrapper with frozen ViT-B/16 using open_clip
    status: completed
  - id: fusion-layer
    content: Implement CLIPFusion with concat+project strategy
    status: completed
  - id: trm-core
    content: Implement TRMController with SwiGLU blocks and deep supervision (adapted from reference)
    status: completed
  - id: bbox-head
    content: Implement BBoxHead regression MLP with sigmoid output
    status: completed
  - id: full-model
    content: Implement ScreenBBoxTRMModel combining all components
    status: completed
  - id: dataset
    content: Implement ScreenSpotDataset reading from parquet with CLIP preprocessing
    status: completed
  - id: losses-metrics
    content: Implement BBoxLoss (SmoothL1 + deep supervision) and IoU/center metrics
    status: completed
  - id: train-script
    content: Implement training loop with AdamW, cosine LR, EMA, and validation
    status: completed
  - id: config-files
    content: Create YAML configs for model and training hyperparameters
    status: completed
  - id: pyproject-deps
    content: Update pyproject.toml with required dependencies (open_clip, torch, etc.)
    status: completed
---

# Screen UI Bounding-Box Prediction with TRM

## 1. Architecture Overview

```
Image + Task Instruction
         |
    [CLIP Encoder] (frozen)
    /           \
img_emb [B,768]  txt_emb [B,768]
         \       /
      [Fusion Layer]
            |
       h_ctx [B,d_trm]
            |
      [TRM Controller]
       (recursive)
            |
     y_final [B,d_trm]
            |
    [BBox Regression Head]
            |
    bbox [B,4] in [0,1]
```

## 2. Component Details

### 2.1 CLIP Backbone (`models/clip_backbone.py`)

```python
class CLIPBackbone(nn.Module):
    """Frozen CLIP ViT-B/16 encoder for images and text."""
    
    def __init__(self, model_name="ViT-B-16", pretrained="openai"):
        # Use open_clip for flexibility
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(...)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.freeze()  # requires_grad=False for all params
    
    def encode_image(self, images: Tensor) -> Tensor:  # [B,3,224,224] -> [B,768]
    def encode_text(self, texts: List[str]) -> Tensor:  # List[str] -> [B,768]
    def forward(self, images, texts) -> Tuple[Tensor, Tensor]:
        return self.encode_image(images), self.encode_text(texts)
```

- **Model**: OpenCLIP `ViT-B-16` (768-dim embeddings)
- **Frozen** in v1 (`requires_grad=False`)
- Returns pooled CLS embeddings (not patches) for simplicity

### 2.2 Fusion Layer (`models/fusion.py`)

```python
class CLIPFusion(nn.Module):
    """Fuse image and text embeddings into TRM context."""
    
    def __init__(self, clip_dim=768, trm_dim=256, fusion_type="concat_proj"):
        # Option 1 (default): Concatenate + project
        self.proj = nn.Sequential(
            nn.Linear(clip_dim * 2, trm_dim),
            nn.LayerNorm(trm_dim),
            nn.GELU()
        )
    
    def forward(self, img_emb, txt_emb) -> Tensor:  # -> [B, d_trm]
        return self.proj(torch.cat([img_emb, txt_emb], dim=-1))
```

### 2.3 TRM Controller (`models/trm_core.py`)

Adapted from `ExampleTinyRecursiveModels/models/recursive_reasoning/trm.py`:

**Key differences from reference:**

- No token sequence (seq_len=1, single vector per sample)
- No ACT halting or Q-learning
- No puzzle embeddings
- Fixed recursion steps
```python
class TRMBlock(nn.Module):
    """Single TRM block: LayerNorm + SwiGLU MLP (2 layers)."""
    
    def __init__(self, hidden_size=256, expansion=4.0):
        self.mlp = SwiGLU(hidden_size, expansion)  # From reference layers.py
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x + self.mlp(x))

class TRMReasoningModule(nn.Module):
    """Stack of TRM blocks for one reasoning level."""
    
    def __init__(self, num_layers=2, hidden_size=256, expansion=4.0):
        self.layers = nn.ModuleList([TRMBlock(...) for _ in range(num_layers)])
    
    def forward(self, hidden: Tensor, injection: Tensor) -> Tensor:
        hidden = hidden + injection
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden
```


**TRM Controller with full backprop:**

```python
class TRMController(nn.Module):
    """
    TRM recursive controller following paper algorithm.
    
    Variables:
      x: context embedding (h_ctx from fusion), constant across recursion
      y: answer embedding, refined each step
      z: latent reasoning state
    
    Recursion (per outer step):
      for _ in range(L_cycles):
          z = L_level(z, x + y)    # latent reasoning
      y = L_level(y, z)            # answer refinement
    """
    
    def __init__(self, hidden_size=256, H_cycles=3, L_cycles=4, L_layers=2):
        self.L_level = TRMReasoningModule(num_layers=L_layers, hidden_size=hidden_size)
        
        # Learned initial states (as in reference, line 153-154)
        self.y_init = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.z_init = nn.Parameter(torch.randn(hidden_size) * 0.02)
    
    def forward(self, h_ctx: Tensor, return_intermediates=False):
        """
        Args:
            h_ctx: [B, d_trm] fused context
        Returns:
            y_final: [B, d_trm] final answer embedding
            intermediates: Optional list of y at each H_cycle
        """
        B = h_ctx.shape[0]
        x = h_ctx  # context (constant)
        y = self.y_init.expand(B, -1)
        z = self.z_init.expand(B, -1)
        
        intermediates = []
        
        # Deep supervision: H_cycles-1 without grad, 1 with grad
        # But FULL backprop through final cycle's inner loops
        
        with torch.no_grad():
            for _ in range(self.H_cycles - 1):
                for _ in range(self.L_cycles):
                    z = self.L_level(z, x + y)
                y = self.L_level(y, z)
                intermediates.append(y.clone())
        
        # Final cycle WITH gradients (backprop through all L_cycles)
        for _ in range(self.L_cycles):
            z = self.L_level(z, x + y)
        y = self.L_level(y, z)
        intermediates.append(y)
        
        return y, intermediates if return_intermediates else None
```

### 2.4 BBox Regression Head (`models/bbox_head.py`)

```python
class BBoxHead(nn.Module):
    """Regress normalized bounding box from TRM output."""
    
    def __init__(self, input_dim=256, hidden_dim=128, output_format="xyxy"):
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4)
        )
        self.output_format = output_format  # "xyxy" or "cxcywh"
    
    def forward(self, y: Tensor) -> Tensor:
        raw = self.mlp(y)  # [B, 4]
        
        if self.output_format == "cxcywh":
            # Predict (cx, cy, w, h), convert to (x1, y1, x2, y2)
            cx, cy, w, h = torch.sigmoid(raw).chunk(4, dim=-1)
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            return torch.cat([x1, y1, x2, y2], dim=-1).clamp(0, 1)
        else:
            # Direct (x1, y1, x2, y2) with sigmoid
            return torch.sigmoid(raw)
```

### 2.5 Full Model (`models/screen_trm_model.py`)

```python
class ScreenBBoxTRMModel(nn.Module):
    def __init__(self, config: ModelConfig):
        self.clip = CLIPBackbone(model_name=config.clip_model)
        self.fusion = CLIPFusion(clip_dim=768, trm_dim=config.trm_hidden_size)
        self.trm = TRMController(
            hidden_size=config.trm_hidden_size,
            H_cycles=config.H_cycles,
            L_cycles=config.L_cycles,
            L_layers=config.L_layers
        )
        self.bbox_head = BBoxHead(input_dim=config.trm_hidden_size)
    
    def forward(self, images: Tensor, tasks: List[str], return_intermediates=False):
        # 1. CLIP encoding (frozen, no grad)
        with torch.no_grad():
            img_emb, txt_emb = self.clip(images, tasks)
        
        # 2. Fusion
        h_ctx = self.fusion(img_emb, txt_emb)
        
        # 3. TRM recursion
        y_final, intermediates = self.trm(h_ctx, return_intermediates)
        
        # 4. BBox prediction
        bbox_pred = self.bbox_head(y_final)
        
        # Optional: intermediate bbox predictions for deep supervision
        bbox_intermediates = None
        if intermediates:
            bbox_intermediates = [self.bbox_head(y) for y in intermediates]
        
        return bbox_pred, bbox_intermediates
```

## 3. Dataset (`data/screenspot_dataset.py`)

```python
class ScreenSpotDataset(Dataset):
    def __init__(self, parquet_path: str, clip_preprocess: Callable, split="train"):
        self.df = pd.read_parquet(parquet_path)
        self.clip_preprocess = clip_preprocess
        # Optional: train/val split (e.g., 90/10)
    
    def __getitem__(self, idx) -> Dict:
        row = self.df.iloc[idx]
        
        # Decode image from bytes
        img_bytes = row["image"]["bytes"]
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Apply CLIP preprocessing
        image_tensor = self.clip_preprocess(image)
        
        # BBox as tensor
        bbox = torch.tensor(row["bbox"], dtype=torch.float32)
        
        return {
            "image": image_tensor,       # [3, 224, 224]
            "task": row["task"],          # str
            "bbox": bbox,                 # [4]
            "image_size": (row["image_width"], row["image_height"])
        }
```

**DataLoader collate function** to handle variable-length text:

```python
def collate_fn(batch):
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "tasks": [b["task"] for b in batch],
        "bboxes": torch.stack([b["bbox"] for b in batch]),
        "image_sizes": [b["image_size"] for b in batch]
    }
```

## 4. Loss and Metrics (`utils/losses.py`, `utils/metrics.py`)

### Loss

```python
class BBoxLoss(nn.Module):
    def __init__(self, loss_type="smooth_l1", deep_supervision_weight=0.1):
        self.loss_fn = nn.SmoothL1Loss() if loss_type == "smooth_l1" else nn.L1Loss()
        self.ds_weight = deep_supervision_weight
    
    def forward(self, pred, target, intermediates=None):
        main_loss = self.loss_fn(pred, target)
        
        if intermediates:
            # Deep supervision on intermediate predictions
            ds_loss = sum(self.loss_fn(p, target) for p in intermediates[:-1])
            ds_loss = ds_loss / max(len(intermediates) - 1, 1)
            return main_loss + self.ds_weight * ds_loss
        
        return main_loss
```

### Metrics

```python
def compute_iou(pred_bbox, gt_bbox):
    """Compute IoU for normalized bboxes [x1, y1, x2, y2]."""
    # Intersection
    x1 = torch.max(pred_bbox[..., 0], gt_bbox[..., 0])
    y1 = torch.max(pred_bbox[..., 1], gt_bbox[..., 1])
    x2 = torch.min(pred_bbox[..., 2], gt_bbox[..., 2])
    y2 = torch.min(pred_bbox[..., 3], gt_bbox[..., 3])
    
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    # Union
    area_pred = (pred_bbox[..., 2] - pred_bbox[..., 0]) * (pred_bbox[..., 3] - pred_bbox[..., 1])
    area_gt = (gt_bbox[..., 2] - gt_bbox[..., 0]) * (gt_bbox[..., 3] - gt_bbox[..., 1])
    union = area_pred + area_gt - inter
    
    return inter / union.clamp(min=1e-6)

def compute_center_distance(pred_bbox, gt_bbox):
    """L2 distance between centers (normalized)."""
    pred_cx = (pred_bbox[..., 0] + pred_bbox[..., 2]) / 2
    pred_cy = (pred_bbox[..., 1] + pred_bbox[..., 3]) / 2
    gt_cx = (gt_bbox[..., 0] + gt_bbox[..., 2]) / 2
    gt_cy = (gt_bbox[..., 1] + gt_bbox[..., 3]) / 2
    return torch.sqrt((pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2)
```

## 5. Training Script (`train.py`)

```python
def train(config):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model
    model = ScreenBBoxTRMModel(config.model).to(device)
    
    # Dataset
    train_dataset = ScreenSpotDataset(
        config.data.train_path,
        model.clip.preprocess,
        split="train"
    )
    val_dataset = ScreenSpotDataset(...)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                               shuffle=True, collate_fn=collate_fn)
    
    # Optimizer (only trainable params: fusion + TRM + bbox_head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)
    
    # LR scheduler (cosine with warmup)
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, config.total_steps)
    
    # Loss
    criterion = BBoxLoss(deep_supervision_weight=0.1)
    
    # Optional: EMA (from reference implementation)
    ema = EMAHelper(mu=0.999) if config.use_ema else None
    if ema:
        ema.register(model)
    
    # Training loop
    for epoch in range(config.epochs):
        model.train()
        for batch in train_loader:
            images = batch["images"].to(device)
            tasks = batch["tasks"]
            bboxes = batch["bboxes"].to(device)
            
            # Forward
            pred_bbox, intermediates = model(images, tasks, return_intermediates=True)
            
            # Loss
            loss = criterion(pred_bbox, bboxes, intermediates)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            
            if ema:
                ema.update(model)
        
        # Validation
        if epoch % config.eval_interval == 0:
            eval_model = ema.ema_copy(model) if ema else model
            metrics = evaluate(eval_model, val_loader, device)
            log_metrics(metrics, epoch)
```

## 6. Configuration (`config/`)

### `config/model.yaml`

```yaml
clip_model: "ViT-B-16"
clip_pretrained: "openai"

trm_hidden_size: 256
H_cycles: 3
L_cycles: 4
L_layers: 2
expansion: 4.0

bbox_output_format: "xyxy"  # or "cxcywh"
```

### `config/train.yaml`

```yaml
# Data
train_path: "dataset/screenspot_training.parquet"
val_split: 0.1

# Training
batch_size: 32
epochs: 50
lr: 1e-4
weight_decay: 0.01
warmup_steps: 1000

# TRM specific
use_ema: true
ema_rate: 0.999
deep_supervision: true
deep_supervision_weight: 0.1

# Logging
eval_interval: 1
log_interval: 100
checkpoint_dir: "checkpoints/"
```

## 7. Project File Structure

```
screenspot-trm/
├── config/
│   ├── model.yaml
│   └── train.yaml
├── data/
│   └── screenspot_dataset.py
├── dataset/
│   └── screenspot_training.parquet  # (existing)
├── models/
│   ├── __init__.py
│   ├── clip_backbone.py
│   ├── fusion.py
│   ├── trm_core.py
│   ├── bbox_head.py
│   ├── screen_trm_model.py
│   └── layers.py          # SwiGLU, rms_norm (from reference)
├── utils/
│   ├── __init__.py
│   ├── losses.py
│   ├── metrics.py
│   └── ema.py             # EMAHelper (from reference)
├── train.py
├── evaluate.py
├── pyproject.toml
└── README.md
```

## 8. Key Implementation Notes

### From Reference TRM (`ExampleTinyRecursiveModels`):

1. **Reuse `SwiGLU` and `rms_norm`** from `models/layers.py` (lines 151-169)
2. **Reuse `EMAHelper`** from `models/ema.py`
3. **Reuse `trunc_normal_init_`** from `models/common.py` for init

### Adaptations from Reference:

| Reference TRM | Our Model |

|--------------|-----------|

| Token sequence [B, L, D] | Single vector [B, D] |

| Token embeddings | CLIP embeddings |

| LM head (vocab logits) | BBox regression head |

| ACT halting + Q-learning | Fixed recursion steps |

| Puzzle embeddings | Not needed |

| z_H, z_L (two latents) | Single z (simpler) |

### Deep Supervision Strategy:

Following reference (lines 207-216 in `trm.py`):

- Run H_cycles-1 recursions without gradients
- Run final cycle with full gradient backprop through all L_cycles
- This is NOT truncated BPTT - we backprop through the entire final cycle

### Future Enhancements (v2+):

1. **CLIP fine-tuning**: Unfreeze last N layers with lower LR
2. **Patch attention**: Use CLIP patch tokens + cross-attention in fusion
3. **GIoU/DIoU loss**: Replace SmoothL1 with IoU-based losses
4. **Multi-scale**: Support variable input resolutions