# ScreenSpot TRM: UI Bounding Box Prediction with Tiny Recursive Models

A screen UI bounding-box prediction model that combines:
- **Frozen CLIP** (ViT-B/16) for vision-text encoding
- **Tiny Recursive Model (TRM)** as the reasoning controller
- **Regression head** for normalized bounding box prediction

Based on the TRM architecture from ["Less is More: Recursive Reasoning with Tiny Networks"](https://arxiv.org/abs/2510.04871).

## Architecture

```
Image + Task Instruction
         |
    [CLIP Encoder] (frozen)
    /           \
img_emb [B,768]  txt_emb [B,768]
         \       /
      [Fusion Layer]
            |
       h_ctx [B,256]
            |
      [TRM Controller]
       (recursive)
            |
     y_final [B,256]
            |
    [BBox Regression Head]
            |
    bbox [B,4] in [0,1]
```

## Installation

Using `uv` (recommended):

```bash
# Create virtual environment and install
uv venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
uv pip install -e .

# With development dependencies
uv pip install -e ".[dev]"

# With wandb logging
uv pip install -e ".[wandb]"
```

Using pip:

```bash
pip install -e .
```

## Dataset

The model expects a Parquet file with the following schema:
- `image`: struct with `bytes` blob (PNG/JPEG)
- `task`: natural language instruction (varchar)
- `image_width`, `image_height`: original dimensions (int64)
- `bbox`: normalized `[x1, y1, x2, y2]` coordinates (float[4])

## Training

### Quick Start (Standard Mode)

```bash
# Train with default config
python train.py

# Train with custom config files
python train.py --train-config config/train.yaml --model-config config/model.yaml

# Override specific options
python train.py --batch-size 64 --epochs 100 --lr 5e-5
```

### Fast Training with Pre-computed Embeddings (Recommended)

Since CLIP is frozen, you can pre-compute embeddings once and reuse them:

```bash
# Step 1: Pre-compute CLIP embeddings (one-time, ~10-15 min for 108k samples)
python precompute_embeddings.py --data-path dataset/screenspot_training.parquet

# Step 2: Train with cached embeddings (much faster, no CLIP overhead)
python train_cached.py --embeddings-path dataset/screenspot_training.embeddings.pt
```

Benefits of cached training:
- **~60-70% faster** per epoch (no CLIP encoding)
- **Lower GPU memory** (no CLIP model loaded)
- **Larger batch sizes** possible (64+ instead of 32)
- Can train on smaller GPUs or even CPU

### Configuration

Edit `config/model.yaml` for model architecture:

```yaml
clip_model: "ViT-B-16"
trm_hidden_size: 256
H_cycles: 3      # Outer deep supervision cycles
L_cycles: 4      # Inner latent reasoning cycles
L_layers: 2      # Layers per reasoning module
```

Edit `config/train.yaml` for training hyperparameters:

```yaml
batch_size: 32
epochs: 50
lr: 1.0e-4
use_ema: true
deep_supervision: true
```

## Evaluation

```bash
# Evaluate checkpoint
python evaluate.py --checkpoint checkpoints/best_model.pt

# Evaluate on specific split
python evaluate.py --checkpoint checkpoints/best_model.pt --split val

# Save predictions
python evaluate.py --checkpoint checkpoints/best_model.pt --save-predictions --output predictions.json
```

## Project Structure

```
screenspot-trm/
├── config/
│   ├── model.yaml          # Model architecture config
│   └── train.yaml          # Training hyperparameters
├── data/
│   └── screenspot_dataset.py   # Dataset loading
├── dataset/
│   └── screenspot_training.parquet  # Training data
├── models/
│   ├── clip_backbone.py    # CLIP encoder wrapper
│   ├── fusion.py           # Image-text fusion
│   ├── trm_core.py         # TRM controller
│   ├── bbox_head.py        # BBox regression head
│   ├── screen_trm_model.py # Full model
│   └── layers.py           # SwiGLU, RMSNorm
├── utils/
│   ├── losses.py           # BBox loss functions
│   ├── metrics.py          # IoU, center distance
│   └── ema.py              # EMA helper
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── pyproject.toml          # Dependencies
└── README.md
```

## TRM Key Concepts

The Tiny Recursive Model maintains three state variables:
- **x**: Context embedding (from CLIP fusion), constant during recursion
- **y**: Answer embedding, refined at each step
- **z**: Latent reasoning state

### Recursion Structure

```
for H_cycle in range(H_cycles):      # Outer loop (deep supervision)
    for L_cycle in range(L_cycles):  # Inner loop (latent reasoning)
        z = L_level(z, x + y)        # Update latent state
    y = L_level(y, z)                # Refine answer
```

### Deep Supervision

- Run H_cycles-1 outer cycles without gradients
- Run final cycle with full gradient backprop through all inner loops
- This is NOT truncated BPTT - we backprop through the entire final cycle

## Metrics

- **IoU**: Intersection over Union
- **GIoU**: Generalized IoU (better for non-overlapping boxes)
- **Center Distance**: L2 distance between predicted and GT centers
- **Acc@k**: Percentage of predictions with IoU >= k

## Model Parameters

With default config (CLIP ViT-B/16 frozen):
- CLIP: ~150M (frozen)
- Fusion: ~0.4M
- TRM: ~2.5M
- BBox Head: ~0.04M
- **Trainable: ~3M parameters**

## References

- [TRM Paper: Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [ScreenSpot Dataset](https://huggingface.co/datasets/HyperCluster/OS-Atlas_ScreenSpot)

## License

MIT
