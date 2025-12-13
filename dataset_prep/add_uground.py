"""
Script to download and process osunlp/UGround-V1-Data-Box dataset 
and append it to the screenspot_training.parquet dataset.

The UGround dataset has conversations with human prompts and GPT bbox responses.
Each human/gpt pair becomes a separate row in our dataset.

Uses STREAMING mode to avoid downloading the full 300GB+ dataset.
Supports checkpointing for resumable processing.
Graceful stop with 's' key, disk space checking.

Usage:
    uv run python add_uground.py --output merged.parquet          # Stream full dataset
    uv run python add_uground.py --test --output test.parquet     # Use local shard for testing
    uv run python add_uground.py --output merged.parquet --resume # Resume from checkpoint

Press 's' + Enter during processing to gracefully stop after the current batch.
"""

import argparse
import json
import os
import re
import select
import shutil
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# Global flag for graceful stop
_stop_requested = False
_input_lock = threading.Lock()


def check_for_stop_input():
    """Background thread to check for 's' input."""
    global _stop_requested
    while True:
        try:
            line = sys.stdin.readline()
            if line.strip().lower() == 's':
                with _input_lock:
                    _stop_requested = True
                print("\n[!] Stop requested. Will save and exit after current batch...")
                break
        except:
            break


def is_stop_requested() -> bool:
    """Check if user requested stop."""
    with _input_lock:
        return _stop_requested


def check_disk_space(path: Path, min_gb: float = 5.0) -> tuple[bool, float]:
    """
    Check if there's enough disk space.
    Returns (has_enough_space, available_gb)
    """
    try:
        stat = shutil.disk_usage(path.parent if path.parent.exists() else Path.cwd())
        available_gb = stat.free / (1024 ** 3)
        return available_gb >= min_gb, available_gb
    except Exception:
        return True, -1  # Assume OK if we can't check


def parse_bbox_string(bbox_str: str) -> tuple[int, int, int, int] | None:
    """
    Parse bbox string like '(30, 56, 303, 91)' into tuple of ints.
    Returns None if parsing fails.
    """
    match = re.match(r'\(?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)?', bbox_str.strip())
    if match:
        return tuple(int(x) for x in match.groups())
    return None


def normalize_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> list[float]:
    """
    Normalize absolute pixel bbox to 0-1 range and clamp values.
    """
    x1, y1, x2, y2 = bbox
    return [
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
        max(0.0, min(1.0, x2 / width)),
        max(0.0, min(1.0, y2 / height)),
    ]


def parse_conversations(conversations_json: str) -> list[tuple[str, str]]:
    """Parse conversations JSON and extract (human_prompt, gpt_bbox) pairs."""
    try:
        conversations = json.loads(conversations_json) if isinstance(conversations_json, str) else conversations_json
    except json.JSONDecodeError:
        return []
    
    pairs = []
    i = 0
    while i < len(conversations) - 1:
        current = conversations[i]
        next_item = conversations[i + 1]
        
        if current.get('from') == 'human' and next_item.get('from') == 'gpt':
            pairs.append((current.get('value', ''), next_item.get('value', '')))
            i += 2
        else:
            i += 1
    
    return pairs


def process_uground_example(example: dict) -> list[dict]:
    """Process a single example from the UGround dataset."""
    width = int(example['width'])
    height = int(example['height'])
    image_bytes = example['image']
    conversations_json = example['conversations']
    
    pairs = parse_conversations(conversations_json)
    image_dict = {'bytes': image_bytes, 'path': None}
    
    results = []
    for human_prompt, gpt_bbox_str in pairs:
        bbox_tuple = parse_bbox_string(gpt_bbox_str)
        if bbox_tuple is None:
            continue
        
        normalized_bbox = normalize_bbox(bbox_tuple, width, height)
        results.append({
            'image': image_dict,
            'task': human_prompt,
            'image_width': width,
            'image_height': height,
            'bbox': normalized_bbox,
        })
    
    return results


def get_parquet_schema(existing_path: Path):
    """Read schema from existing parquet file."""
    pf = pq.ParquetFile(existing_path)
    return pf.schema_arrow


def batch_to_table(batch_rows: list[dict], schema: pa.Schema) -> pa.Table:
    """Convert a list of row dicts to an Arrow table with the correct schema."""
    if not batch_rows:
        return pa.Table.from_pylist([], schema=schema)
    
    images = [row['image'] for row in batch_rows]
    tasks = [row['task'] for row in batch_rows]
    widths = [row['image_width'] for row in batch_rows]
    heights = [row['image_height'] for row in batch_rows]
    bboxes = [row['bbox'] for row in batch_rows]
    
    bbox_type = schema.field('bbox').type
    
    image_array = pa.array(images, type=schema.field('image').type)
    task_array = pa.array(tasks, type=pa.string())
    width_array = pa.array(widths, type=pa.int64())
    height_array = pa.array(heights, type=pa.int64())
    
    if pa.types.is_fixed_size_list(bbox_type):
        flat_values = []
        for bbox in bboxes:
            flat_values.extend(bbox)
        value_array = pa.array(flat_values, type=pa.float32())
        bbox_array = pa.FixedSizeListArray.from_arrays(value_array, 4)
    else:
        bbox_array = pa.array(bboxes, type=bbox_type)
    
    return pa.Table.from_arrays(
        [image_array, task_array, width_array, height_array, bbox_array],
        names=['image', 'task', 'image_width', 'image_height', 'bbox']
    )


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load checkpoint data."""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint_path: Path, source_count: int, processed_count: int):
    """Save checkpoint with current progress."""
    with open(checkpoint_path, 'w') as f:
        json.dump({
            'source_rows_processed': source_count,
            'task_bbox_pairs_written': processed_count,
        }, f)


def stream_and_process_uground(
    output_path: Path, 
    existing_path: Path, 
    test_mode: bool, 
    limit: int | None,
    resume: bool = False,
    min_disk_gb: float = 5.0,
):
    """
    Stream the UGround dataset and write processed rows incrementally to parquet.
    Supports: checkpointing, graceful stop, disk space checking.
    """
    BATCH_SIZE = 500
    
    schema = get_parquet_schema(existing_path)
    checkpoint_path = output_path.with_suffix('.checkpoint.json')
    
    # Start input listener thread
    print("Press 's' + Enter at any time to gracefully stop and save progress.")
    input_thread = threading.Thread(target=check_for_stop_input, daemon=True)
    input_thread.start()
    
    # Check for resume
    skip_rows = 0
    checkpoint_data = load_checkpoint(checkpoint_path)
    if resume and checkpoint_data:
        skip_rows = checkpoint_data.get('source_rows_processed', 0)
        if skip_rows > 0:
            print(f"Resuming from checkpoint: skipping first {skip_rows} source rows")
            if not output_path.exists():
                print("ERROR: Checkpoint exists but output file not found.")
                print("Delete the checkpoint file to start fresh, or restore the output file.")
                return 0, 0
    
    # Load source data
    if test_mode:
        print("Test mode: Loading local shard_0000.parquet...")
        shard_path = Path(__file__).parent / "shard_0000.parquet"
        if not shard_path.exists():
            raise FileNotFoundError(f"Test shard not found: {shard_path}")
        
        shard_df = pd.read_parquet(shard_path)
        
        def row_iterator():
            for idx, row in shard_df.iterrows():
                yield row.to_dict()
        
        total_source_rows = len(shard_df)
        source_iter = row_iterator()
    else:
        print("Streaming osunlp/UGround-V1-Data-Box from HuggingFace...")
        from datasets import load_dataset
        ds = load_dataset("osunlp/UGround-V1-Data-Box", split="train", streaming=True)
        source_iter = iter(ds)
        total_source_rows = None
    
    processed_count = 0
    source_count = 0
    batch_rows = []
    existing_rows = 0
    stopped_early = False
    
    print("Processing UGround dataset (streaming)...")
    
    if resume and skip_rows > 0 and output_path.exists():
        # Resume mode: skip already processed rows, then append
        print(f"Skipping {skip_rows} already-processed rows...")
        for _ in range(skip_rows):
            try:
                next(source_iter)
                source_count += 1
            except StopIteration:
                print("Reached end of dataset. Nothing more to process.")
                return 0, 0
        
        print(f"Continuing from row {skip_rows}...")
        
        for example in source_iter:
            if limit and source_count >= limit:
                break
            if is_stop_requested():
                stopped_early = True
                break
            
            try:
                rows = process_uground_example(example)
                batch_rows.extend(rows)
            except Exception as e:
                print(f"Warning: Error processing row {source_count}: {e}")
            
            source_count += 1
            
            if len(batch_rows) >= BATCH_SIZE * 10:
                # Check disk space
                has_space, avail_gb = check_disk_space(output_path, min_disk_gb)
                if not has_space:
                    print(f"\n[!] LOW DISK SPACE: {avail_gb:.1f}GB available (need {min_disk_gb}GB)")
                    print("Saving current progress and stopping...")
                    stopped_early = True
                
                table = batch_to_table(batch_rows, schema)
                existing_table = pq.read_table(output_path)
                combined = pa.concat_tables([existing_table, table])
                pq.write_table(combined, output_path)
                
                processed_count += len(batch_rows)
                save_checkpoint(checkpoint_path, source_count, processed_count)
                batch_rows = []
                
                progress = f"{source_count}/{total_source_rows}" if total_source_rows else f"{source_count}"
                _, avail_gb = check_disk_space(output_path)
                print(f"  Processed {progress} rows, {processed_count} pairs | Disk: {avail_gb:.1f}GB free")
                
                if stopped_early:
                    break
        
        # Write remaining
        if batch_rows:
            table = batch_to_table(batch_rows, schema)
            existing_table = pq.read_table(output_path)
            combined = pa.concat_tables([existing_table, table])
            pq.write_table(combined, output_path)
            processed_count += len(batch_rows)
            save_checkpoint(checkpoint_path, source_count, processed_count)
    
    else:
        # Fresh start
        with pq.ParquetWriter(output_path, schema) as writer:
            print(f"Copying existing dataset from {existing_path}...")
            existing_pf = pq.ParquetFile(existing_path)
            for i in range(existing_pf.metadata.num_row_groups):
                table = existing_pf.read_row_group(i)
                writer.write_table(table)
                if (i + 1) % 10 == 0:
                    print(f"  Copied {i + 1}/{existing_pf.metadata.num_row_groups} row groups...")
            existing_rows = existing_pf.metadata.num_rows
            print(f"Copied {existing_rows} existing rows")
            
            for example in source_iter:
                if limit and source_count >= limit:
                    break
                if is_stop_requested():
                    stopped_early = True
                    break
                
                try:
                    rows = process_uground_example(example)
                    batch_rows.extend(rows)
                except Exception as e:
                    print(f"Warning: Error processing row {source_count}: {e}")
                
                source_count += 1
                
                if len(batch_rows) >= BATCH_SIZE * 10:
                    # Check disk space
                    has_space, avail_gb = check_disk_space(output_path, min_disk_gb)
                    if not has_space:
                        print(f"\n[!] LOW DISK SPACE: {avail_gb:.1f}GB available")
                        stopped_early = True
                    
                    table = batch_to_table(batch_rows, schema)
                    writer.write_table(table)
                    processed_count += len(batch_rows)
                    save_checkpoint(checkpoint_path, source_count, processed_count)
                    batch_rows = []
                    
                    progress = f"{source_count}/{total_source_rows}" if total_source_rows else f"{source_count}"
                    _, avail_gb = check_disk_space(output_path)
                    print(f"  Processed {progress} rows, {processed_count} pairs | Disk: {avail_gb:.1f}GB free")
                    
                    if stopped_early:
                        break
            
            if batch_rows:
                table = batch_to_table(batch_rows, schema)
                writer.write_table(table)
                processed_count += len(batch_rows)
                save_checkpoint(checkpoint_path, source_count, processed_count)
    
    # Cleanup or keep checkpoint
    if stopped_early:
        print(f"\n[!] Stopped early. Checkpoint saved at {checkpoint_path}")
        print(f"    Run with --resume to continue.")
    elif checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Processing complete. Checkpoint removed.")
    
    print(f"\nDone!")
    print(f"  Source rows processed: {source_count}")
    print(f"  Task/bbox pairs added: {processed_count}")
    print(f"  Output: {output_path}")
    
    return existing_rows, processed_count


def main():
    parser = argparse.ArgumentParser(
        description="Process UGround dataset (streaming) and append to screenspot_training.parquet"
    )
    parser.add_argument("--test", action="store_true", help="Use local shard for testing")
    parser.add_argument("--output", type=str, required=True, help="Output path for merged dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit source rows to process")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--min-disk-gb", type=float, default=5.0, help="Minimum disk space (GB) before stopping")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    dataset_dir = script_dir.parent / "dataset"
    screenspot_path = dataset_dir / "screenspot_training.parquet"
    output_path = Path(args.output)
    
    if not screenspot_path.exists():
        raise FileNotFoundError(f"screenspot_training.parquet not found at {screenspot_path}")
    
    # Initial disk check
    has_space, avail_gb = check_disk_space(output_path, args.min_disk_gb)
    print(f"Input: {screenspot_path}")
    print(f"Output: {output_path}")
    print(f"Disk space: {avail_gb:.1f}GB available (min: {args.min_disk_gb}GB)")
    if args.resume:
        print("Resume mode: ON")
    print()
    
    if not has_space:
        print(f"ERROR: Not enough disk space. Need at least {args.min_disk_gb}GB.")
        return
    
    existing_rows, new_rows = stream_and_process_uground(
        output_path=output_path,
        existing_path=screenspot_path,
        test_mode=args.test,
        limit=args.limit,
        resume=args.resume,
        min_disk_gb=args.min_disk_gb,
    )
    
    print(f"\nSummary:")
    print(f"  Original rows: {existing_rows}")
    print(f"  New rows added: {new_rows}")
    print(f"  Total rows: {existing_rows + new_rows}")


if __name__ == "__main__":
    main()
