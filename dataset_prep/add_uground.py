"""
Script to download and process osunlp/UGround-V1-Data-Box dataset 
and append it to the screenspot_training.parquet dataset.

The UGround dataset has conversations with human prompts and GPT bbox responses.
Each human/gpt pair becomes a separate row in our dataset.

Usage:
    uv run python add_uground.py           # Download full dataset and process
    uv run python add_uground.py --test    # Use local shard_0000.parquet for testing
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_bbox_string(bbox_str: str) -> tuple[int, int, int, int] | None:
    """
    Parse bbox string like '(30, 56, 303, 91)' into tuple of ints.
    Returns None if parsing fails.
    """
    # Remove parentheses and split by comma
    match = re.match(r'\(?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)?', bbox_str.strip())
    if match:
        return tuple(int(x) for x in match.groups())
    return None


def normalize_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> np.ndarray:
    """
    Normalize absolute pixel bbox to 0-1 range and clamp values.
    Input: (x1, y1, x2, y2) in pixels
    Output: np.array([x1/width, y1/height, x2/width, y2/height]), clamped to [0, 1]
    """
    x1, y1, x2, y2 = bbox
    normalized = np.array([
        x1 / width,
        y1 / height,
        x2 / width,
        y2 / height,
    ], dtype=np.float32)
    # Clamp to [0, 1] range in case of annotation errors
    return np.clip(normalized, 0.0, 1.0)


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


def convert_image_bytes_to_dict(image_bytes: bytes) -> dict:
    """
    Convert raw image bytes to the format used by screenspot_training.parquet.
    The screenspot format stores images as dict with 'bytes' and 'path' keys.
    """
    return {'bytes': image_bytes, 'path': None}


def process_uground_row(row: pd.Series) -> list[dict]:
    """
    Process a single row from the UGround dataset.
    Returns a list of dicts, one for each human/gpt pair in the conversations.
    """
    width = int(row['width'])
    height = int(row['height'])
    image_bytes = row['image']
    conversations_json = row['conversations']
    
    # Parse conversations to get human/gpt pairs
    pairs = parse_conversations(conversations_json)
    
    # Convert image bytes once for all pairs
    image_dict = convert_image_bytes_to_dict(image_bytes)
    
    results = []
    for human_prompt, gpt_bbox_str in pairs:
        # Parse the bbox string
        bbox_tuple = parse_bbox_string(gpt_bbox_str)
        if bbox_tuple is None:
            # Skip this pair silently (don't spam warnings)
            continue
        
        # Normalize the bbox (returns numpy array, clamped to [0, 1])
        normalized_bbox = normalize_bbox(bbox_tuple, width, height)
        
        # Create the row in screenspot format
        result = {
            'image': image_dict,
            'task': human_prompt,
            'image_width': width,
            'image_height': height,
            'bbox': normalized_bbox,
        }
        results.append(result)
    
    return results


def load_uground_dataset(test_mode: bool) -> pd.DataFrame:
    """
    Load the UGround dataset.
    In test mode, uses local shard_0000.parquet.
    Otherwise, downloads from HuggingFace.
    """
    if test_mode:
        print("Test mode: Loading local shard_0000.parquet...")
        shard_path = Path(__file__).parent / "shard_0000.parquet"
        if not shard_path.exists():
            raise FileNotFoundError(f"Test shard not found: {shard_path}")
        return pd.read_parquet(shard_path)
    else:
        print("Downloading osunlp/UGround-V1-Data-Box from HuggingFace...")
        from datasets import load_dataset
        ds = load_dataset("osunlp/UGround-V1-Data-Box", split="train")
        return ds.to_pandas()


def main():
    parser = argparse.ArgumentParser(
        description="Process UGround dataset and append to screenspot_training.parquet"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use local shard_0000.parquet instead of downloading full dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the merged dataset (default: overwrite screenspot_training.parquet)"
    )
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).parent
    dataset_dir = script_dir.parent / "dataset"
    screenspot_path = dataset_dir / "screenspot_training.parquet"
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = screenspot_path
    
    # Load existing screenspot dataset
    print(f"Loading existing dataset from {screenspot_path}...")
    if not screenspot_path.exists():
        raise FileNotFoundError(f"screenspot_training.parquet not found at {screenspot_path}")
    
    existing_df = pd.read_parquet(screenspot_path)
    print(f"Existing dataset: {len(existing_df)} rows")
    
    # Load UGround dataset
    uground_df = load_uground_dataset(args.test)
    print(f"UGround dataset loaded: {len(uground_df)} rows")
    
    # Process UGround dataset
    print("Processing UGround dataset...")
    new_rows = []
    skipped = 0
    for idx, row in uground_df.iterrows():
        try:
            processed_rows = process_uground_row(row)
            new_rows.extend(processed_rows)
        except Exception as e:
            print(f"Warning: Error processing row {idx}: {e}")
            skipped += 1
        
        # Progress indicator
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(uground_df)} rows...")
    
    print(f"Processed {len(new_rows)} task/bbox pairs from UGround dataset")
    if skipped > 0:
        print(f"Skipped {skipped} rows due to errors")
    
    # Convert to DataFrame
    new_df = pd.DataFrame(new_rows)
    
    # Ensure column order matches
    new_df = new_df[['image', 'task', 'image_width', 'image_height', 'bbox']]
    
    # Append to existing dataset
    print("Merging datasets...")
    merged_df = pd.concat([existing_df, new_df], ignore_index=True)
    print(f"Merged dataset: {len(merged_df)} rows")
    
    # Save
    print(f"Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(output_path)
    
    print(f"\nDone!")
    print(f"  Original rows: {len(existing_df)}")
    print(f"  New rows added: {len(new_df)}")
    print(f"  Total rows: {len(merged_df)}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
