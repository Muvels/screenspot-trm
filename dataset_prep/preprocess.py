import argparse
import pandas as pd
import torch
from transformers import CLIPProcessor
import logging
from tqdm import tqdm
import os

def preprocess_dataset(input_path: str, output_path: str, model_name: str, batch_size: int = 1000):
    """
    Loads the parquet dataset, tokenizes the 'task' column using CLIPProcessor,
    and saves the new dataset with 'input_ids' and 'attention_mask' columns.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"Loading dataset from {input_path}...")
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    logger.info(f"Initializing processor: {model_name}")
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # We will process in chunks to show progress
    input_ids_list = []
    attention_mask_list = []
    
    logger.info("Tokenizing text...")
    
    tasks = df['task'].fillna("").tolist() # Handle NaNs
    
    # Flatten image column to bytes to avoid PyArrow nested struct issues
    logger.info("Flattening image struct to bytes...")
    def extract_bytes(x):
        if isinstance(x, dict) and 'bytes' in x:
            return x['bytes']
        return x
    
    # Check if 'image' exists
    if 'image' in df.columns:
        df['image'] = df['image'].apply(extract_bytes)
        # Rename to generic name or keep as image? 
        # Keeping as 'image' but ensuring it's bytes.
        # Verify it's bytes
        # df['image'] = df['image'].astype(bytes)
    
    # Batch processing for speed
    for i in tqdm(range(0, len(tasks), batch_size)):
        batch_tasks = tasks[i : i + batch_size]
        
        # Tokenize (padding="max_length" to ensure consistent columns in parquet? 
        # Actually standard parquet handles lists of varying length, but for ML training 
        # we usually want fixed max_len or we pad in COLLATE_FN.
        # However, CLIP expects max_length=77 usually.
        encoded = processor(text=batch_tasks, return_tensors="np", padding="max_length", truncation=True, max_length=77)
        
        input_ids_list.extend(encoded["input_ids"].tolist())
        attention_mask_list.extend(encoded["attention_mask"].tolist())
        
    # Add columns
    logger.info("Adding new columns to dataframe...")
    df["input_ids"] = input_ids_list
    df["attention_mask"] = attention_mask_list
    
    logger.info(f"Saving to {output_path}...")
    df.to_parquet(output_path)
    logger.info("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="dataset/screenspot_training.parquet")
    parser.add_argument("--output_path", type=str, default="dataset/screenspot_tokenized.parquet")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
    args = parser.parse_args()
    
    preprocess_dataset(args.input_path, args.output_path, args.model_name)
