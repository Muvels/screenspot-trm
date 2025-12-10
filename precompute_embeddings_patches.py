"""Pre-compute CLIP embeddings WITH patch tokens for spatial features.

This version saves both pooled embeddings AND patch embeddings,
which are essential for good localization performance.

Updated version:
- Streams embeddings to disk in shards instead of keeping everything in RAM.
- Reads parquet in chunks to avoid loading all images into RAM.
- Much lower peak RAM usage for large datasets.

Usage:
    python precompute_embeddings_patches.py --data-path dataset/screenspot_training.parquet \
        --output-path dataset/screenspot_training_patches.pt \
        --batch-size 32 --shard-size 5000
"""

import argparse
import io
from pathlib import Path
from typing import List, Tuple, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
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


def load_and_preprocess_images_from_table(table, row_indices: List[int], preprocess) -> torch.Tensor:
    """Load images from a PyArrow table slice."""
    images = []
    for idx in row_indices:
        img_col = table.column("image")
        img_data = img_col[idx].as_py()
        
        if isinstance(img_data, dict):
            img_bytes = img_data["bytes"]
        else:
            img_bytes = img_data

        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image_tensor = preprocess(image)
        images.append(image_tensor)

    return torch.stack(images)


def get_tasks_from_table(table, row_indices: List[int]) -> List[str]:
    """Get task strings from a PyArrow table slice."""
    task_col = table.column("task")
    return [str(task_col[idx].as_py()) for idx in row_indices]


def iter_parquet_batches(parquet_path: str, batch_size: int) -> Iterator[Tuple[int, pa.Table]]:
    """
    Iterate over a parquet file in batches without loading everything into RAM.
    
    Yields (start_idx, table_slice) tuples.
    """
    parquet_file = pq.ParquetFile(parquet_path)
    
    # Read row groups and yield batches
    current_idx = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        table = pa.Table.from_batches([batch])
        yield current_idx, table
        current_idx += len(table)


def init_shard_tensors(
    capacity: int,
    embed_dim: int,
    patch_dim: int,
    num_patches: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate tensors for a single shard."""
    img_pooled = torch.zeros(capacity, embed_dim, dtype=dtype)
    img_patches = torch.zeros(capacity, num_patches, patch_dim, dtype=dtype)
    txt_embeddings = torch.zeros(capacity, embed_dim, dtype=dtype)
    return img_pooled, img_patches, txt_embeddings


def save_shard(
    shard_idx: int,
    base_path: Path,
    img_pooled: torch.Tensor,
    img_patches: torch.Tensor,
    txt_embeddings: torch.Tensor,
    valid_samples: int,
    *,
    clip,
    args,
    embed_dim: int,
    patch_dim: int,
    num_patches: int,
    use_fp16: bool,
    total_samples: int,
    global_start_idx: int,
) -> float:
    """Save one shard to disk and return its size in GB."""
    if valid_samples == 0:
        return 0.0

    suffix = base_path.suffix or ".pt"
    shard_path = base_path.with_name(
        f"{base_path.stem}_shard_{shard_idx:05d}{suffix}"
    )

    payload = {
        "img_pooled": img_pooled[:valid_samples],
        "img_patches": img_patches[:valid_samples],
        "txt_embeddings": txt_embeddings[:valid_samples],
        # metadata
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "embed_dim": embed_dim,
        "patch_dim": patch_dim,
        "num_patches": num_patches,
        "grid_size": clip.num_patches,
        "n_samples": valid_samples,            # samples in this shard
        "dataset_n_samples": total_samples,    # total samples across all shards
        "global_start_idx": global_start_idx,  # index of the first sample in this shard
        "source_file": str(args.data_path),
        "dtype": "float16" if use_fp16 else "float32",
    }

    torch.save(payload, shard_path)
    file_size_gb = shard_path.stat().st_size / (1024 ** 3)
    print(f"  Saved shard {shard_idx} with {valid_samples} samples to {shard_path} ({file_size_gb:.2f} GB)")
    return file_size_gb


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute CLIP embeddings with patch tokens (sharded to disk)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="dataset/screenspot_training.parquet",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help=(
            "Base path for output. Shards will be written as "
            "<stem>_shard_00000.pt, <stem>_shard_00001.pt, ... "
            "(default: derive from data-path)"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--shard-size",
        type=int,
        default=5000,
        help="Number of samples per shard written to disk.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--clip-model", type=str, default="ViT-B-16")
    parser.add_argument("--clip-pretrained", type=str, default="openai")
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Save embeddings in float16 (default: True, saves ~50% memory)",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Save embeddings in float32 (uses more memory)",
    )

    args = parser.parse_args()

    # Handle dtype flags
    use_fp16 = args.fp16 and not args.fp32
    storage_dtype = torch.float16 if use_fp16 else torch.float32

    # Resolve output base path
    if args.output_path is not None:
        base_path = Path(args.output_path)
    else:
        # <data_dir>/<data_stem>_patch_embeddings.pt
        data_path = Path(args.data_path)
        base_name = data_path.stem + "_patch_embeddings.pt"
        base_path = data_path.with_name(base_name)

    print("Configuration:")
    print(f"  Data path: {args.data_path}")
    print(f"  Output base path: {base_path}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Shard size: {args.shard_size}")
    print(f"  Device (requested): {args.device}")

    device = get_device(args.device)
    print(f"  Device (used): {device}")

    # Get dataset info WITHOUT loading into memory
    print("\nScanning dataset...")
    parquet_file = pq.ParquetFile(args.data_path)
    n_samples = parquet_file.metadata.num_rows
    if n_samples == 0:
        print("Dataset is empty, nothing to do.")
        return

    print(f"  Number of samples: {n_samples}")
    print(f"  Number of row groups: {parquet_file.metadata.num_row_groups}")
    print("  (Reading in streaming mode - NOT loading all into RAM)")

    # Load CLIP with patch extraction
    print("\nLoading CLIP model with patch extraction...")
    clip = CLIPBackboneWithPatches(
        model_name=args.clip_model,
        pretrained=args.clip_pretrained,
        device=str(device),
    )
    clip.eval()

    embed_dim = clip.embed_dim  # 512 for pooled/text
    patch_dim = clip.patch_dim  # 768 for patches
    num_patches = clip.num_patches ** 2  # 14*14 = 196 for ViT-B/16

    print("Model / embedding config:")
    print(f"  Embed dim (pooled/text): {embed_dim}")
    print(f"  Patch dim: {patch_dim}")
    print(f"  Num patches per image: {num_patches}")
    print(f"  Storage dtype: {storage_dtype}")

    # Initialise first shard
    shard_capacity = min(args.shard_size, n_samples)
    img_pooled_shard, img_patches_shard, txt_embeddings_shard = init_shard_tensors(
        shard_capacity, embed_dim, patch_dim, num_patches, storage_dtype
    )
    shard_pos = 0
    shard_idx = 0
    total_written = 0
    total_gb = 0.0

    print("\nComputing embeddings and writing shards to disk...")
    print("  (Streaming parquet - only batch_size rows in RAM at a time)")
    
    with torch.no_grad():
        # Stream through parquet file batch by batch
        pbar = tqdm(total=n_samples, desc="Processing")
        
        for batch_start, table_batch in iter_parquet_batches(args.data_path, args.batch_size):
            batch_len = len(table_batch)
            local_indices = list(range(batch_len))
            
            # Load images from this batch only (not entire dataset!)
            images = load_and_preprocess_images_from_table(
                table_batch, local_indices, clip.preprocess
            ).to(device)

            # Get texts from this batch
            tasks = get_tasks_from_table(table_batch, local_indices)

            # Encode
            pooled, patches, txt_emb = clip(images, tasks)

            # Move to CPU + cast to storage dtype
            pooled = pooled.to(dtype=storage_dtype).cpu()
            patches = patches.to(dtype=storage_dtype).cpu()
            txt_emb = txt_emb.to(dtype=storage_dtype).cpu()
            
            # Free table batch memory
            del table_batch, images
            
            remaining = batch_len
            offset = 0

            # May need to split this batch across multiple shards
            while remaining > 0:
                capacity_left = img_pooled_shard.shape[0] - shard_pos
                if capacity_left == 0:
                    # Current shard full → save to disk
                    total_gb += save_shard(
                        shard_idx,
                        base_path,
                        img_pooled_shard,
                        img_patches_shard,
                        txt_embeddings_shard,
                        shard_pos,
                        clip=clip,
                        args=args,
                        embed_dim=embed_dim,
                        patch_dim=patch_dim,
                        num_patches=num_patches,
                        use_fp16=use_fp16,
                        total_samples=n_samples,
                        global_start_idx=total_written,
                    )
                    total_written += shard_pos
                    shard_idx += 1

                    # Allocate a new shard (remaining samples might be less than shard_size)
                    remaining_overall = n_samples - total_written
                    if remaining_overall == 0:
                        break
                    shard_capacity = min(args.shard_size, remaining_overall)
                    img_pooled_shard, img_patches_shard, txt_embeddings_shard = init_shard_tensors(
                        shard_capacity, embed_dim, patch_dim, num_patches, storage_dtype
                    )
                    shard_pos = 0
                    capacity_left = shard_capacity

                take = min(remaining, capacity_left)

                img_pooled_shard[shard_pos : shard_pos + take] = pooled[offset : offset + take]
                img_patches_shard[shard_pos : shard_pos + take] = patches[offset : offset + take]
                txt_embeddings_shard[shard_pos : shard_pos + take] = txt_emb[offset : offset + take]

                shard_pos += take
                offset += take
                remaining -= take
            
            pbar.update(batch_len)
        
        pbar.close()

        # After loop: save any remaining samples in the last shard
        if shard_pos > 0:
            total_gb += save_shard(
                shard_idx,
                base_path,
                img_pooled_shard,
                img_patches_shard,
                txt_embeddings_shard,
                shard_pos,
                clip=clip,
                args=args,
                embed_dim=embed_dim,
                patch_dim=patch_dim,
                num_patches=num_patches,
                use_fp16=use_fp16,
                total_samples=n_samples,
                global_start_idx=total_written,
            )
            total_written += shard_pos
            shard_idx += 1

    print("\nDone!")
    print(f"  Total samples written: {total_written} (dataset has {n_samples})")
    print(f"  Number of shards: {shard_idx}")
    print(f"  Approx total size on disk: {total_gb:.2f} GB")
    print("  NOTE: Outputs are sharded. Look for files named like:")
    suffix = base_path.suffix or ".pt"
    example_name = base_path.with_name(f"{base_path.stem}_shard_00000{suffix}")
    print(f"        {example_name}")
    print("  Each shard contains its own 'n_samples' plus 'dataset_n_samples' and 'global_start_idx' metadata.")


if __name__ == "__main__":
    main()
