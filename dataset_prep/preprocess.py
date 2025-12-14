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
import pandas as pd
from transformers import AutoProcessor
import argparse
from tqdm import tqdm
import logging
import os
import gc


def extract_bytes(x):
    """Extract bytes from image struct if needed."""
    if isinstance(x, dict) and 'bytes' in x:
        return x['bytes']
    return x


def read_row_group_safe(parquet_file, rg_idx: int, columns: list = None):
    """
    Read a row group with fallback for problematic nested types.
    
    First tries standard read_row_group. If that fails with nested type error,
    falls back to reading via pandas which handles it differently.
    """
    try:
        # Try standard PyArrow approach first
        table = parquet_file.read_row_group(rg_idx, columns=columns)
        return table.to_pandas()
    except pa.lib.ArrowNotImplementedError as e:
        if "Nested data conversions" in str(e):
            # Fallback: Read via pandas parquet reader for this row group
            # This is slower but handles edge cases
            logging.warning(f"Row group {rg_idx}: Using pandas fallback for nested types")
            
            # Get row range for this row group
            metadata = parquet_file.metadata
            start_row = sum(
                metadata.row_group(i).num_rows 
                for i in range(rg_idx)
            )
            num_rows = metadata.row_group(rg_idx).num_rows
            
            # Read just this slice using pandas
            df = pd.read_parquet(
                parquet_file.reader.source,  # Get file path
                columns=columns
            ).iloc[start_row:start_row + num_rows]
            
            return df
        else:
            raise


def preprocess_dataset_streaming(
    input_path: str, 
    output_path: str, 
    model_name: str, 
    batch_size: int = 10000,
    resume_from_rg: int = 0
):
    """
    Processes a large parquet file in streaming chunks using row groups.
    
    Memory usage: ~8-16 GB regardless of file size.
    
    Args:
        input_path: Path to input parquet file (can be 100s of GB)
        output_path: Path to output parquet file
        model_name: HuggingFace model name for tokenizer
        batch_size: Rows per tokenization batch (for progress)
        resume_from_rg: Resume from specific row group (for recovery)
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory: {output_dir}")
    
    logger.info(f"Loading processor for {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    
    logger.info(f"Opening {input_path} for streaming...")
    parquet_file = pq.ParquetFile(input_path)
    
    # Get metadata
    total_rows = parquet_file.metadata.num_rows
    num_row_groups = parquet_file.metadata.num_row_groups
    logger.info(f"Total rows: {total_rows:,}")
    logger.info(f"Row groups: {num_row_groups}")
    
    if resume_from_rg > 0:
        logger.info(f"Resuming from row group {resume_from_rg}")
    
    # Get column names (we need these for the fallback reader)
    schema = parquet_file.schema_arrow
    original_columns = [schema.field(i).name for i in range(len(schema))]
    
    # We'll build schema from first row group
    writer = None
    rows_processed = 0
    
    # Calculate rows already processed if resuming
    rows_before_resume = sum(
        parquet_file.metadata.row_group(i).num_rows 
        for i in range(resume_from_rg)
    )
    
    # Progress bar
    pbar = tqdm(
        total=total_rows, 
        initial=rows_before_resume,
        desc="Processing", 
        unit="rows"
    )
    
    failed_row_groups = []
    
    try:
        # Iterate through row groups
        for rg_idx in range(resume_from_rg, num_row_groups):
            try:
                # Read row group (with fallback for problematic ones)
                table = parquet_file.read_row_group(rg_idx)
                df_batch = table.to_pandas()
            except pa.lib.ArrowNotImplementedError as e:
                if "Nested data conversions" in str(e):
                    logger.warning(f"Row group {rg_idx}: Skipping due to nested type issue")
                    failed_row_groups.append(rg_idx)
                    # Update progress for skipped rows
                    skipped_rows = parquet_file.metadata.row_group(rg_idx).num_rows
                    pbar.update(skipped_rows)
                    continue
                else:
                    raise
            
            batch_len = len(df_batch)
            
            # Extract image bytes if needed
            if 'image' in df_batch.columns:
                df_batch['image'] = df_batch['image'].apply(extract_bytes)
            
            # Get tasks and tokenize in sub-batches for memory efficiency
            tasks = df_batch['task'].fillna("").tolist()
            
            input_ids_list = []
            attention_mask_list = []
            
            # Process in smaller tokenization batches
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i:i + batch_size]
                
                encoded = processor(
                    text=batch_tasks, 
                    return_tensors="np", 
                    padding="max_length", 
                    truncation=True, 
                    max_length=64
                )
                
                input_ids = encoded["input_ids"]
                if "attention_mask" in encoded:
                    attn_mask = encoded["attention_mask"]
                else:
                    pad_id = getattr(processor.tokenizer, 'pad_token_id', 0) or 0
                    attn_mask = (input_ids != pad_id).astype("int64")
                
                input_ids_list.extend(input_ids.tolist())
                attention_mask_list.extend(attn_mask.tolist())
            
            # Add new columns
            df_batch["input_ids"] = input_ids_list
            df_batch["attention_mask"] = attention_mask_list
            
            # Convert back to PyArrow table
            table_batch = pa.Table.from_pandas(df_batch, preserve_index=False)
            
            # Initialize writer with schema from first batch
            if writer is None:
                writer = pq.ParquetWriter(output_path, table_batch.schema, compression='snappy')
                logger.info(f"Output schema: {table_batch.schema}")
            
            writer.write_table(table_batch)
            rows_processed += batch_len
            pbar.update(batch_len)
            
            # Free memory explicitly
            del df_batch, table, table_batch
            del input_ids_list, attention_mask_list, tasks
            
            # Force garbage collection every 10 row groups to prevent slowdown
            if rg_idx % 10 == 0:
                gc.collect()
            
    except KeyboardInterrupt:
        logger.warning(f"Interrupted! Resume with --resume_from_rg {rg_idx}")
        raise
    except Exception as e:
        logger.error(f"Error processing row group {rg_idx}: {e}")
        logger.error(f"Resume with --resume_from_rg {rg_idx}")
        raise
    finally:
        if writer is not None:
            writer.close()
        pbar.close()
    
    # Report failed row groups
    if failed_row_groups:
        logger.warning(f"Failed row groups (skipped): {failed_row_groups}")
        logger.warning(f"Skipped {len(failed_row_groups)} row groups")
    
    # Log output file size
    if os.path.exists(output_path):
        output_size = os.path.getsize(output_path) / (1024**3)
        logger.info(f"Done! Processed {rows_processed:,} rows")
        logger.info(f"Output file: {output_path} ({output_size:.2f} GB)")


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
        help="Tokenization batch size (default: 10000)"
    )
    parser.add_argument(
        "--resume_from_rg", type=int, default=0,
        help="Resume from specific row group (for recovery)"
    )
    args = parser.parse_args()
    
    preprocess_dataset_streaming(
        args.input_path, 
        args.output_path, 
        args.model_name,
        args.batch_size,
        args.resume_from_rg
    )


if __name__ == "__main__":
    main()
