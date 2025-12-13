"""
Script to download and process osunlp/UGround-V1-Data-Box dataset 
and append it to the screenspot_training.parquet dataset.

The UGround dataset has conversations with human prompts and GPT bbox responses.
Each human/gpt pair becomes a separate row in our dataset.

Uses STREAMING mode to avoid downloading the full 300GB+ dataset.

Usage:
    uv run python add_uground.py           # Stream full dataset and process
    uv run python add_uground.py --test    # Use local shard_0000.parquet for testing
    uv run python add_uground.py --limit 10000  # Process only first 10000 rows
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


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
    Input: (x1, y1, x2, y2) in pixels
    Output: list of floats [x1/width, y1/height, x2/width, y2/height], clamped to [0, 1]
    """
    x1, y1, x2, y2 = bbox
    normalized = [
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
        max(0.0, min(1.0, x2 / width)),
        max(0.0, min(1.0, y2 / height)),
    ]
    return normalized


def parse_conversations(conversations_json: str) -> list[tuple[str, str]]:
    """
    Parse conversations JSON and extract (human_prompt, gpt_bbox) pairs.
    Returns list of tuples.
    """
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
            human_value = current.get('value', '')
            gpt_value = next_item.get('value', '')
            pairs.append((human_value, gpt_value))
            i += 2
        else:
            i += 1
    
    return pairs


def process_uground_example(example: dict) -> list[dict]:
    """
    Process a single example from the UGround dataset.
    Returns a list of dicts, one for each human/gpt pair in the conversations.
    """
    width = int(example['width'])
    height = int(example['height'])
    image_bytes = example['image']
    conversations_json = example['conversations']
    
    # Parse conversations to get human/gpt pairs
    pairs = parse_conversations(conversations_json)
    
    # Convert image bytes once for all pairs
    image_dict = {'bytes': image_bytes, 'path': None}
    
    results = []
    for human_prompt, gpt_bbox_str in pairs:
        # Parse the bbox string
        bbox_tuple = parse_bbox_string(gpt_bbox_str)
        if bbox_tuple is None:
            continue
        
        # Normalize the bbox (clamped to [0, 1])
        normalized_bbox = normalize_bbox(bbox_tuple, width, height)
        
        result = {
            'image': image_dict,
            'task': human_prompt,
            'image_width': width,
            'image_height': height,
            'bbox': normalized_bbox,
        }
        results.append(result)
    
    return results


def get_parquet_schema(existing_path: Path):
    """Read schema from existing parquet file to ensure compatibility"""
    pf = pq.ParquetFile(existing_path)
    return pf.schema_arrow


def batch_to_table(batch_rows: list[dict], schema: pa.Schema) -> pa.Table:
    """
    Convert a list of row dicts to an Arrow table with the correct schema.
    Handles fixed_size_list for bbox column.
    """
    if not batch_rows:
        return pa.Table.from_pylist([], schema=schema)
    
    # Build each column separately to ensure correct types
    images = [row['image'] for row in batch_rows]
    tasks = [row['task'] for row in batch_rows]
    widths = [row['image_width'] for row in batch_rows]
    heights = [row['image_height'] for row in batch_rows]
    bboxes = [row['bbox'] for row in batch_rows]
    
    # Get the bbox type from schema
    bbox_type = schema.field('bbox').type
    
    # Create arrays
    image_array = pa.array(images, type=schema.field('image').type)
    task_array = pa.array(tasks, type=pa.string())
    width_array = pa.array(widths, type=pa.int64())
    height_array = pa.array(heights, type=pa.int64())
    
    # Handle bbox - need to create fixed_size_list if that's what schema expects
    if pa.types.is_fixed_size_list(bbox_type):
        # Create as fixed_size_list
        flat_values = []
        for bbox in bboxes:
            flat_values.extend(bbox)
        value_array = pa.array(flat_values, type=pa.float32())
        bbox_array = pa.FixedSizeListArray.from_arrays(value_array, 4)
    else:
        # Regular list
        bbox_array = pa.array(bboxes, type=bbox_type)
    
    return pa.Table.from_arrays(
        [image_array, task_array, width_array, height_array, bbox_array],
        names=['image', 'task', 'image_width', 'image_height', 'bbox']
    )

def stream_and_process_uground(output_path: Path, existing_path: Path, test_mode: bool, limit: int | None):
    """
    Stream the UGround dataset and write processed rows incrementally to parquet.
    Uses batched writing to avoid memory issues.
    """
    BATCH_SIZE = 500  # Process this many UGround rows before writing
    
    schema = get_parquet_schema(existing_path)
    
    # First, copy existing data to the new output file
    print(f"Copying existing dataset to output...")
    if existing_path != output_path:
        # Read and write in chunks to avoid memory issues
        existing_pf = pq.ParquetFile(existing_path)
        with pq.ParquetWriter(output_path, schema) as writer:
            for i in range(existing_pf.metadata.num_row_groups):
                table = existing_pf.read_row_group(i)
                writer.write_table(table)
                if (i + 1) % 10 == 0:
                    print(f"  Copied {i + 1}/{existing_pf.metadata.num_row_groups} row groups...")
        existing_rows = existing_pf.metadata.num_rows
        print(f"Copied {existing_rows} existing rows")
    else:
        # Output is same as input, we'll need to write to temp and rename
        raise ValueError("Output path cannot be the same as input when streaming. Use --output to specify a different path.")
    
    # Now stream and append UGround data
    if test_mode:
        print("Test mode: Loading local shard_0000.parquet...")
        shard_path = Path(__file__).parent / "shard_0000.parquet"
        if not shard_path.exists():
            raise FileNotFoundError(f"Test shard not found: {shard_path}")
        
        # Read the shard and iterate
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
        total_source_rows = None  # Unknown in streaming mode
    
    # Process in batches and append to parquet
    processed_count = 0
    source_count = 0
    batch_rows = []
    
    print("Processing UGround dataset (streaming)...")
    
    # Open the file in append mode
    with pq.ParquetWriter(output_path, schema, writer_engine_version='V2') as writer:
        # First write existing data
        existing_pf = pq.ParquetFile(existing_path)
        for i in range(existing_pf.metadata.num_row_groups):
            table = existing_pf.read_row_group(i)
            writer.write_table(table)
        existing_rows = existing_pf.metadata.num_rows
        print(f"Wrote {existing_rows} existing rows")
        
        # Now process streaming data
        for example in source_iter:
            if limit and source_count >= limit:
                break
            
            try:
                rows = process_uground_example(example)
                batch_rows.extend(rows)
            except Exception as e:
                print(f"Warning: Error processing row {source_count}: {e}")
            
            source_count += 1
            
            # Write batch when it's large enough
            if len(batch_rows) >= BATCH_SIZE * 10:  # ~5000 processed rows per write
                # Convert to Arrow table using proper schema handling
                table = batch_to_table(batch_rows, schema)
                writer.write_table(table)
                processed_count += len(batch_rows)
                batch_rows = []
                
                progress = f"{source_count}/{total_source_rows}" if total_source_rows else f"{source_count}"
                print(f"  Processed {progress} source rows, {processed_count} task/bbox pairs written...")
        
        # Write remaining rows
        if batch_rows:
            table = batch_to_table(batch_rows, schema)
            writer.write_table(table)
            processed_count += len(batch_rows)
    
    print(f"\nDone!")
    print(f"  Source rows processed: {source_count}")
    print(f"  Task/bbox pairs added: {processed_count}")
    print(f"  Output: {output_path}")
    
    return existing_rows, processed_count


def main():
    parser = argparse.ArgumentParser(
        description="Process UGround dataset (streaming) and append to screenspot_training.parquet"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use local shard_0000.parquet instead of streaming full dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the merged dataset (required to avoid overwriting during streaming)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of source rows to process (for testing)"
    )
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).parent
    dataset_dir = script_dir.parent / "dataset"
    screenspot_path = dataset_dir / "screenspot_training.parquet"
    output_path = Path(args.output)
    
    if not screenspot_path.exists():
        raise FileNotFoundError(f"screenspot_training.parquet not found at {screenspot_path}")
    
    print(f"Input: {screenspot_path}")
    print(f"Output: {output_path}")
    print()
    
    existing_rows, new_rows = stream_and_process_uground(
        output_path=output_path,
        existing_path=screenspot_path,
        test_mode=args.test,
        limit=args.limit,
    )
    
    print(f"\nSummary:")
    print(f"  Original rows: {existing_rows}")
    print(f"  New rows added: {new_rows}")
    print(f"  Total rows: {existing_rows + new_rows}")


if __name__ == "__main__":
    main()
