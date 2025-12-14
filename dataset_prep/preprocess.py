"""
Memory-Efficient Parallel Dataset Preprocessing

Processes large parquet files using multiple CPU cores.
Tokenizes text in parallel and writes output incrementally.

Usage:
    python dataset_prep/preprocess.py \
        --input_path ../merged_v2.parquet \
        --output_path dataset/screenspot_tokenized.parquet \
        --model_name google/siglip-base-patch16-256-multilingual \
        --num_workers 24
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from transformers import AutoProcessor
import argparse
from tqdm import tqdm
import logging
import os
import gc
from multiprocessing import Pool, cpu_count
from functools import partial


# Global processor (initialized per worker)
_processor = None
_model_name = None


def init_worker(model_name: str):
    """Initialize tokenizer in each worker process."""
    global _processor, _model_name
    _model_name = model_name
    _processor = AutoProcessor.from_pretrained(model_name)


def extract_bytes(x):
    """Extract bytes from image struct if needed."""
    if isinstance(x, dict) and 'bytes' in x:
        return x['bytes']
    return x


def process_row_group(args):
    """
    Process a single row group: read, tokenize, return as bytes.
    
    Returns (rg_idx, processed_df_bytes, num_rows) or (rg_idx, None, 0) on error.
    """
    rg_idx, input_path, batch_size = args
    
    global _processor
    
    try:
        # Open parquet file in this worker
        parquet_file = pq.ParquetFile(input_path)
        
        # Read row group
        table = parquet_file.read_row_group(rg_idx)
        df_batch = table.to_pandas()
        batch_len = len(df_batch)
        
        # Extract image bytes if needed
        if 'image' in df_batch.columns:
            df_batch['image'] = df_batch['image'].apply(extract_bytes)
        
        # Get tasks and tokenize
        tasks = df_batch['task'].fillna("").tolist()
        
        input_ids_list = []
        attention_mask_list = []
        
        # Process in smaller tokenization batches
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i + batch_size]
            
            encoded = _processor(
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
                pad_id = getattr(_processor.tokenizer, 'pad_token_id', 0) or 0
                attn_mask = (input_ids != pad_id).astype("int64")
            
            input_ids_list.extend(input_ids.tolist())
            attention_mask_list.extend(attn_mask.tolist())
        
        # Add new columns
        df_batch["input_ids"] = input_ids_list
        df_batch["attention_mask"] = attention_mask_list
        
        return (rg_idx, df_batch, batch_len)
        
    except pa.lib.ArrowNotImplementedError as e:
        if "Nested data conversions" in str(e):
            return (rg_idx, None, 0)  # Skip this row group
        raise
    except Exception as e:
        logging.error(f"Error in row group {rg_idx}: {e}")
        return (rg_idx, None, 0)


def preprocess_dataset_parallel(
    input_path: str, 
    output_path: str, 
    model_name: str, 
    batch_size: int = 10000,
    num_workers: int = None
):
    """
    Processes a large parquet file using multiple CPU cores.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        model_name: HuggingFace model name for tokenizer
        batch_size: Rows per tokenization batch
        num_workers: Number of parallel workers (default: cpu_count - 2)
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)
    
    logger.info(f"Using {num_workers} parallel workers")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory: {output_dir}")
    
    logger.info(f"Opening {input_path}...")
    parquet_file = pq.ParquetFile(input_path)
    
    # Get metadata
    total_rows = parquet_file.metadata.num_rows
    num_row_groups = parquet_file.metadata.num_row_groups
    logger.info(f"Total rows: {total_rows:,}")
    logger.info(f"Row groups: {num_row_groups}")
    
    # Prepare arguments for parallel processing
    args_list = [(rg_idx, input_path, batch_size) for rg_idx in range(num_row_groups)]
    
    # Initialize writer (will be set after first successful result)
    writer = None
    rows_processed = 0
    failed_row_groups = []
    
    # Progress bar
    pbar = tqdm(total=total_rows, desc="Processing", unit="rows")
    
    try:
        # Create worker pool with initialized tokenizers
        with Pool(num_workers, initializer=init_worker, initargs=(model_name,)) as pool:
            # Process row groups in parallel, collect results
            # Using imap_unordered for better throughput
            for rg_idx, df_batch, batch_len in pool.imap_unordered(process_row_group, args_list):
                if df_batch is None:
                    logger.warning(f"Row group {rg_idx}: Skipped")
                    failed_row_groups.append(rg_idx)
                    # Estimate skipped rows
                    skipped = parquet_file.metadata.row_group(rg_idx).num_rows
                    pbar.update(skipped)
                    continue
                
                # Convert to PyArrow table
                table_batch = pa.Table.from_pandas(df_batch, preserve_index=False)
                
                # Initialize writer with schema from first successful batch
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table_batch.schema, compression='snappy')
                    logger.info(f"Output schema initialized")
                
                writer.write_table(table_batch)
                rows_processed += batch_len
                pbar.update(batch_len)
                
                # Clean up
                del df_batch, table_batch
                
    except KeyboardInterrupt:
        logger.warning("Interrupted by user!")
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        if writer is not None:
            writer.close()
        pbar.close()
    
    # Report results
    if failed_row_groups:
        logger.warning(f"Failed row groups (skipped): {sorted(failed_row_groups)}")
        logger.warning(f"Skipped {len(failed_row_groups)} row groups")
    
    if os.path.exists(output_path):
        output_size = os.path.getsize(output_path) / (1024**3)
        logger.info(f"Done! Processed {rows_processed:,} rows")
        logger.info(f"Output file: {output_path} ({output_size:.2f} GB)")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-tokenize dataset with parallel processing"
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
        "--num_workers", type=int, default=None,
        help="Number of parallel workers (default: cpu_count - 2)"
    )
    args = parser.parse_args()
    
    preprocess_dataset_parallel(
        args.input_path, 
        args.output_path, 
        args.model_name,
        args.batch_size,
        args.num_workers
    )


if __name__ == "__main__":
    main()
