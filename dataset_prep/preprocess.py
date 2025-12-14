"""
Memory-Efficient Dataset Preprocessing

Processes large parquet files in streaming chunks to avoid OOM.
Tokenizes text and writes output incrementally.

Usage:
    python dataset_prep/preprocess.py \
        --input_path ../merged_v2.parquet \
        --output_path dataset/screenspot_tokenized.parquet \
        --model_name google/siglip-base-patch16-256-multilingual \
        --batch_size 10000
"""

import torch
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoProcessor
import argparse
from tqdm import tqdm
import logging
import os


def extract_bytes(x):
    """Extract bytes from image struct if needed."""
    if isinstance(x, dict) and 'bytes' in x:
        return x['bytes']
    return x


def preprocess_dataset_streaming(
    input_path: str, 
    output_path: str, 
    model_name: str, 
    batch_size: int = 10000
):
    """
    Processes a large parquet file in streaming chunks.
    
    Memory usage: ~8-16 GB regardless of file size.
    
    Args:
        input_path: Path to input parquet file (can be 100s of GB)
        output_path: Path to output parquet file
        model_name: HuggingFace model name for tokenizer
        batch_size: Rows to process at a time (higher = faster but more RAM)
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading processor for {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    
    logger.info(f"Opening {input_path} for streaming...")
    parquet_file = pq.ParquetFile(input_path)
    
    # Get metadata
    total_rows = parquet_file.metadata.num_rows
    num_row_groups = parquet_file.metadata.num_row_groups
    logger.info(f"Total rows: {total_rows:,}")
    logger.info(f"Row groups: {num_row_groups}")
    logger.info(f"Processing in batches of {batch_size:,}")
    
    # Get schema and add new columns
    original_schema = parquet_file.schema_arrow
    
    # Define new fields for tokenized data
    new_fields = [
        pa.field("input_ids", pa.list_(pa.int64())),
        pa.field("attention_mask", pa.list_(pa.int64()))
    ]
    
    # Build new schema (original + new fields)
    # First, check if image column needs to be converted to binary
    new_schema_fields = []
    for field in original_schema:
        if field.name == "image":
            # Convert image struct to binary
            new_schema_fields.append(pa.field("image", pa.binary()))
        else:
            new_schema_fields.append(field)
    new_schema_fields.extend(new_fields)
    new_schema = pa.schema(new_schema_fields)
    
    # Open writer
    writer = pq.ParquetWriter(output_path, new_schema, compression='snappy')
    
    # Progress bar
    pbar = tqdm(total=total_rows, desc="Processing", unit="rows")
    
    try:
        # Iterate through batches
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            # Convert to pandas for easier manipulation
            df_batch = batch.to_pandas()
            batch_len = len(df_batch)
            
            # Extract image bytes if needed
            if 'image' in df_batch.columns:
                df_batch['image'] = df_batch['image'].apply(extract_bytes)
            
            # Get tasks and tokenize
            tasks = df_batch['task'].fillna("").tolist()
            
            encoded = processor(
                text=tasks, 
                return_tensors="np", 
                padding="max_length", 
                truncation=True, 
                max_length=64
            )
            
            input_ids = encoded["input_ids"]
            if "attention_mask" in encoded:
                attn_mask = encoded["attention_mask"]
            else:
                # Generate mask for models that don't return it
                pad_id = getattr(processor.tokenizer, 'pad_token_id', 0) or 0
                attn_mask = (input_ids != pad_id).astype("int64")
            
            # Add new columns
            df_batch["input_ids"] = input_ids.tolist()
            df_batch["attention_mask"] = attn_mask.tolist()
            
            # Convert back to PyArrow table and write
            table_batch = pa.Table.from_pandas(df_batch, schema=new_schema, preserve_index=False)
            writer.write_table(table_batch)
            
            pbar.update(batch_len)
            
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        raise
    finally:
        writer.close()
        pbar.close()
    
    # Log output file size
    output_size = os.path.getsize(output_path) / (1024**3)
    logger.info(f"Done! Output file: {output_path} ({output_size:.2f} GB)")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-tokenize dataset with streaming (memory-efficient)"
    )
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Path to input parquet file"
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path to output parquet file"
    )
    parser.add_argument(
        "--model_name", type=str, 
        default="google/siglip-base-patch16-256-multilingual",
        help="HuggingFace model name for tokenizer"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10000,
        help="Rows to process at a time (default: 10000)"
    )
    args = parser.parse_args()
    
    preprocess_dataset_streaming(
        args.input_path, 
        args.output_path, 
        args.model_name,
        args.batch_size
    )


if __name__ == "__main__":
    main()
