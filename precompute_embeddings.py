"""Pre-compute CLIP embeddings for the ScreenSpot dataset.

Since CLIP is frozen during training, we can compute embeddings once
and reuse them, saving ~60-70% of forward pass time per batch.

Usage:
    python precompute_embeddings.py --data-path dataset/screenspot_training.parquet
    python precompute_embeddings.py --data-path dataset/screenspot_training.parquet --batch-size 64
"""

import argparse
import io
from pathlib import Path
from typing import List

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from models.clip_backbone import CLIPBackbone


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


def load_and_preprocess_images(
    df: pd.DataFrame,
    indices: List[int],
    preprocess,
) -> torch.Tensor:
    """Load and preprocess a batch of images.
    
    Args:
        df: DataFrame with image data
        indices: Row indices to load
        preprocess: CLIP preprocessing transform
        
    Returns:
        Batch of preprocessed images [B, 3, 224, 224]
    """
    images = []
    for idx in indices:
        row = df.iloc[idx]
        img_data = row["image"]
        if isinstance(img_data, dict):
            img_bytes = img_data["bytes"]
        else:
            img_bytes = img_data
        
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image_tensor = preprocess(image)
        images.append(image_tensor)
    
    return torch.stack(images)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute CLIP embeddings for ScreenSpot dataset"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="dataset/screenspot_training.parquet",
        help="Path to parquet file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path for embeddings (default: <data-path>.embeddings.pt)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, mps, cpu)",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="ViT-B-16",
        help="CLIP model to use",
    )
    parser.add_argument(
        "--clip-pretrained",
        type=str,
        default="openai",
        help="CLIP pretrained weights",
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset: {args.data_path}")
    df = pd.read_parquet(args.data_path)
    n_samples = len(df)
    print(f"Total samples: {n_samples}")
    
    # Load CLIP model
    print(f"Loading CLIP model: {args.clip_model}")
    clip = CLIPBackbone(
        model_name=args.clip_model,
        pretrained=args.clip_pretrained,
        device=str(device),
    )
    clip.eval()
    
    # Pre-allocate tensors
    embed_dim = clip.embed_dim
    img_embeddings = torch.zeros(n_samples, embed_dim, dtype=torch.float32)
    txt_embeddings = torch.zeros(n_samples, embed_dim, dtype=torch.float32)
    
    # Process in batches
    print(f"Computing embeddings (batch_size={args.batch_size})...")
    
    with torch.no_grad():
        for start_idx in tqdm(range(0, n_samples, args.batch_size)):
            end_idx = min(start_idx + args.batch_size, n_samples)
            batch_indices = list(range(start_idx, end_idx))
            
            # Load and preprocess images
            images = load_and_preprocess_images(df, batch_indices, clip.preprocess)
            images = images.to(device)
            
            # Get text tasks
            tasks = [str(df.iloc[i]["task"]) for i in batch_indices]
            
            # Encode
            img_emb = clip.encode_image(images)
            txt_emb = clip.encode_text(tasks)
            
            # Store
            img_embeddings[start_idx:end_idx] = img_emb.cpu()
            txt_embeddings[start_idx:end_idx] = txt_emb.cpu()
    
    # Save embeddings
    output_path = args.output_path
    if output_path is None:
        data_path = Path(args.data_path)
        output_path = data_path.parent / f"{data_path.stem}.embeddings.pt"
    
    print(f"Saving embeddings to: {output_path}")
    torch.save({
        "img_embeddings": img_embeddings,
        "txt_embeddings": txt_embeddings,
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "embed_dim": embed_dim,
        "n_samples": n_samples,
        "source_file": str(args.data_path),
    }, output_path)
    
    # Print stats
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\nDone!")
    print(f"  Samples: {n_samples}")
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Image embeddings: {img_embeddings.shape}")
    print(f"  Text embeddings: {txt_embeddings.shape}")
    print(f"  File size: {file_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
