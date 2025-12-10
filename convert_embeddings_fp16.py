"""Convert embeddings from float32 to float16 to reduce memory usage.

Supports both:
- Single file format (legacy): converts entire file
- Sharded format (new): converts each shard individually (memory-efficient)

CLIP embeddings work fine in float16 - the precision loss is negligible.

Usage:
    # Single file
    python convert_embeddings_fp16.py --input dataset/embeddings.pt
    
    # Sharded (auto-detects pattern)
    python convert_embeddings_fp16.py --input dataset/embeddings_shard_00000.pt
    
    # Or specify base path for shards
    python convert_embeddings_fp16.py --input dataset/embeddings.pt --sharded
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import tqdm


def find_shard_files(input_path: str) -> List[Path]:
    """Find all shard files matching the pattern."""
    path = Path(input_path)
    parent = path.parent
    stem = path.stem
    
    # Remove _shard_XXXXX suffix if present to get base stem
    base_stem = re.sub(r'_shard_\d+$', '', stem)
    
    # Find all matching shards
    shard_pattern = f"{base_stem}_shard_*.pt"
    shards = sorted(parent.glob(shard_pattern))
    
    return shards


def convert_single_file(input_path: str, output_path: str = None) -> Tuple[float, float]:
    """Convert a single embeddings file to fp16."""
    print(f"Loading: {input_path}")
    data = torch.load(input_path, map_location='cpu', weights_only=False)
    
    # Check if already fp16
    if data.get('img_patches') is not None and data['img_patches'].dtype == torch.float16:
        print("  Already in float16, skipping...")
        return 0.0, 0.0
    
    # Calculate original size
    total_original = 0
    for key in ['img_pooled', 'img_patches', 'txt_embeddings']:
        if key in data:
            t = data[key]
            size_gb = t.numel() * t.element_size() / (1024**3)
            total_original += size_gb
    
    # Convert to float16
    for key in ['img_pooled', 'img_patches', 'txt_embeddings']:
        if key in data:
            data[key] = data[key].half()
    
    # Update dtype metadata if present
    if 'dtype' in data:
        data['dtype'] = 'float16'
    
    # Calculate new size
    total_new = 0
    for key in ['img_pooled', 'img_patches', 'txt_embeddings']:
        if key in data:
            t = data[key]
            size_gb = t.numel() * t.element_size() / (1024**3)
            total_new += size_gb
    
    # Determine output path
    if output_path is None:
        input_p = Path(input_path)
        # For shards, replace in place or add _fp16 before _shard
        if '_shard_' in input_p.stem:
            # Keep same name (overwrite) or could add _fp16
            output_path = input_p  # Overwrite in place for shards
        else:
            output_path = input_p.parent / f"{input_p.stem}_fp16.pt"
    
    # Save
    torch.save(data, output_path)
    
    return total_original, total_new


def convert_sharded(input_path: str, in_place: bool = True):
    """Convert sharded embeddings to fp16, one shard at a time."""
    shard_files = find_shard_files(input_path)
    
    if not shard_files:
        print(f"No shard files found matching pattern from: {input_path}")
        return
    
    print(f"Found {len(shard_files)} shards to convert")
    
    total_original = 0.0
    total_new = 0.0
    
    for shard_path in tqdm(shard_files, desc="Converting shards"):
        if in_place:
            output_path = str(shard_path)
        else:
            # Create new file with _fp16 in the base name
            stem = shard_path.stem
            base_stem = re.sub(r'_shard_(\d+)$', r'_fp16_shard_\1', stem)
            output_path = str(shard_path.parent / f"{base_stem}.pt")
        
        orig, new = convert_single_file(str(shard_path), output_path)
        total_original += orig
        total_new += new
    
    print(f"\nConversion complete!")
    print(f"  Shards processed: {len(shard_files)}")
    if total_original > 0:
        print(f"  Original total: {total_original:.2f} GB")
        print(f"  New total: {total_new:.2f} GB")
        print(f"  Reduction: {total_original - total_new:.2f} GB ({100 * (1 - total_new/total_original):.1f}%)")
    else:
        print("  All shards were already in float16")


def convert_to_fp16(input_path: str, output_path: str = None, force_sharded: bool = False):
    """Convert embeddings to fp16, auto-detecting format."""
    path = Path(input_path)
    
    # Check if this looks like sharded format
    is_sharded = force_sharded or '_shard_' in path.stem
    
    if is_sharded or not path.exists():
        # Try to find shards
        shard_files = find_shard_files(input_path)
        if shard_files:
            print(f"Detected sharded format with {len(shard_files)} shards")
            convert_sharded(input_path, in_place=(output_path is None))
            return
    
    # Single file format
    if not path.exists():
        print(f"Error: File not found: {input_path}")
        return
    
    print("Detected single file format")
    print(f"Loading embeddings from: {input_path}")
    
    data = torch.load(input_path, map_location='cpu', weights_only=False)
    
    # Check if already fp16
    if data.get('img_patches') is not None and data['img_patches'].dtype == torch.float16:
        print("Already in float16, nothing to do!")
        return
    
    # Show original info
    print("\nOriginal tensor info:")
    total_original = 0
    for key in ['img_pooled', 'img_patches', 'txt_embeddings']:
        if key in data:
            t = data[key]
            size_gb = t.numel() * t.element_size() / (1024**3)
            total_original += size_gb
            print(f"  {key}: shape={t.shape}, dtype={t.dtype}, size={size_gb:.2f} GB")
    print(f"Total: {total_original:.2f} GB")
    
    # Convert to float16
    print("\nConverting to float16...")
    for key in ['img_pooled', 'img_patches', 'txt_embeddings']:
        if key in data:
            data[key] = data[key].half()
    
    # Update dtype metadata if present
    if 'dtype' in data:
        data['dtype'] = 'float16'
    
    # Show new info
    print("\nConverted tensor info:")
    total_new = 0
    for key in ['img_pooled', 'img_patches', 'txt_embeddings']:
        if key in data:
            t = data[key]
            size_gb = t.numel() * t.element_size() / (1024**3)
            total_new += size_gb
            print(f"  {key}: shape={t.shape}, dtype={t.dtype}, size={size_gb:.2f} GB")
    print(f"Total: {total_new:.2f} GB")
    print(f"Reduction: {total_original - total_new:.2f} GB ({100 * (1 - total_new/total_original):.1f}%)")
    
    # Determine output path
    if output_path is None:
        input_p = Path(input_path)
        output_path = input_p.parent / f"{input_p.stem}_fp16.pt"
    
    # Save
    print(f"\nSaving to: {output_path}")
    torch.save(data, output_path)
    
    # Verify file size
    file_size_gb = Path(output_path).stat().st_size / (1024**3)
    print(f"File size on disk: {file_size_gb:.2f} GB")
    print("\nDone! Use the new file with --embeddings-path in training.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert embeddings to float16 (supports single file and sharded formats)"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to input embeddings file or any shard file"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path to output file (default: adds _fp16 suffix, or overwrites shards in-place)"
    )
    parser.add_argument(
        "--sharded", action="store_true",
        help="Force sharded mode (auto-detected from filename if contains _shard_)"
    )
    args = parser.parse_args()
    
    convert_to_fp16(args.input, args.output, args.sharded)


if __name__ == "__main__":
    main()
