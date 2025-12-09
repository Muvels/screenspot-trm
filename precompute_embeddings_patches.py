"""Pre-compute CLIP embeddings WITH patch tokens for spatial features.

This version saves both pooled embeddings AND patch embeddings,
which are essential for good localization performance.

Usage:
    python precompute_embeddings_patches.py --data-path dataset/screenspot_training.parquet
"""

import argparse
import io
from pathlib import Path
from typing import List

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from models.clip_backbone_patches import CLIPBackboneWithPatches


def get_device(requested: str = "auto") -> torch.device:
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
            return torch.device("cpu")
    elif requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def load_and_preprocess_images(df, indices, preprocess):
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
        description="Pre-compute CLIP embeddings with patch tokens"
    )
    parser.add_argument("--data-path", type=str, 
                        default="dataset/screenspot_training.parquet")
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--clip-model", type=str, default="ViT-B-16")
    parser.add_argument("--clip-pretrained", type=str, default="openai")
    
    args = parser.parse_args()
    
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset: {args.data_path}")
    df = pd.read_parquet(args.data_path)
    n_samples = len(df)
    print(f"Total samples: {n_samples}")
    
    # Load CLIP with patch extraction
    print(f"Loading CLIP model with patch extraction: {args.clip_model}")
    clip = CLIPBackboneWithPatches(
        model_name=args.clip_model,
        pretrained=args.clip_pretrained,
        device=str(device),
    )
    clip.eval()
    
    embed_dim = clip.embed_dim  # 512 for pooled/text
    patch_dim = clip.patch_dim  # 768 for patches
    num_patches = clip.num_patches ** 2  # 14*14 = 196 for ViT-B/16
    
    print(f"  Embed dim (pooled/text): {embed_dim}")
    print(f"  Patch dim: {patch_dim}")
    print(f"  Num patches: {num_patches}")
    
    # Pre-allocate tensors
    img_pooled = torch.zeros(n_samples, embed_dim, dtype=torch.float32)
    img_patches = torch.zeros(n_samples, num_patches, patch_dim, dtype=torch.float32)
    txt_embeddings = torch.zeros(n_samples, embed_dim, dtype=torch.float32)
    
    # Process in batches
    print(f"Computing embeddings (batch_size={args.batch_size})...")
    
    with torch.no_grad():
        for start_idx in tqdm(range(0, n_samples, args.batch_size)):
            end_idx = min(start_idx + args.batch_size, n_samples)
            batch_indices = list(range(start_idx, end_idx))
            
            # Load images
            images = load_and_preprocess_images(df, batch_indices, clip.preprocess)
            images = images.to(device)
            
            # Get texts
            tasks = [str(df.iloc[i]["task"]) for i in batch_indices]
            
            # Encode
            pooled, patches, txt_emb = clip(images, tasks)
            
            # Store
            img_pooled[start_idx:end_idx] = pooled.cpu()
            img_patches[start_idx:end_idx] = patches.cpu()
            txt_embeddings[start_idx:end_idx] = txt_emb.cpu()
    
    # Save
    output_path = args.output_path
    if output_path is None:
        data_path = Path(args.data_path)
        output_path = data_path.parent / f"{data_path.stem}.embeddings_patches.pt"
    
    print(f"Saving embeddings to: {output_path}")
    torch.save({
        "img_pooled": img_pooled,
        "img_patches": img_patches,
        "txt_embeddings": txt_embeddings,
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "embed_dim": embed_dim,
        "patch_dim": patch_dim,
        "num_patches": num_patches,
        "grid_size": clip.num_patches,
        "n_samples": n_samples,
        "source_file": str(args.data_path),
    }, output_path)
    
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\nDone!")
    print(f"  Samples: {n_samples}")
    print(f"  Embed dim (pooled/text): {embed_dim}")
    print(f"  Patch dim: {patch_dim}")
    print(f"  Patches per image: {num_patches}")
    print(f"  File size: {file_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
