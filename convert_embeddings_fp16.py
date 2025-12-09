"""Convert embeddings from float32 to float16 to reduce memory usage.

This converts the 61GB embeddings file to ~30GB by using half precision.
CLIP embeddings work fine in float16 - the precision loss is negligible.

Usage:
    python convert_embeddings_fp16.py --input dataset/screenspot_training.embeddings_patches.pt
    
This will create a new file with '_fp16' suffix.
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm


def convert_to_fp16(input_path: str, output_path: str = None):
    print(f"Loading embeddings from: {input_path}")
    data = torch.load(input_path, map_location='cpu', weights_only=False)
    
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
    parser = argparse.ArgumentParser(description="Convert embeddings to float16")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to input embeddings file")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Path to output file (default: adds _fp16 suffix)")
    args = parser.parse_args()
    
    convert_to_fp16(args.input, args.output)


if __name__ == "__main__":
    main()

