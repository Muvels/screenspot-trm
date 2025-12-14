"""
Memory-Efficient Dataset Loader for Large Parquet Files

Uses memory-mapping and lazy loading to handle datasets larger than RAM.
Only loads individual rows when accessed via __getitem__.
"""

import pyarrow.parquet as pq
import pyarrow as pa
import torch
import numpy as np
import io
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Callable, Dict, Any
import logging
import os


class ScreenspotDataset(Dataset):
    """
    Memory-efficient dataset loader for the Screenspot Parquet dataset.
    
    Uses memory-mapped reading to handle files larger than RAM.
    Only loads individual rows when accessed.
    
    Expected Columns:
    - image: binary (or struct with bytes)
    - task: str
    - bbox: list[float] [x1, y1, x2, y2]
    - image_width: int
    - image_height: int
    
    Optional (if pre-tokenized):
    - input_ids: list[int]
    - attention_mask: list[int]
    """
    
    def __init__(self, parquet_path: str, transform: Optional[Callable] = None):
        """
        Args:
            parquet_path: Path to the .parquet file.
            transform: Preprocessing function for images (e.g., from CLIP/ViT).
                       Should accept a PIL Image and return a tensor.
        """
        self.parquet_path = parquet_path
        self.transform = transform
        
        if not os.path.exists(parquet_path):
            logging.error(f"Parquet file not found at {parquet_path}")
            raise FileNotFoundError(f"Parquet file not found at {parquet_path}")

        logging.info(f"Opening dataset from {parquet_path} (memory-mapped)...")
        
        # Open parquet file WITHOUT loading into memory
        # This just reads metadata and creates file handle
        self.parquet_file = pq.ParquetFile(parquet_path, memory_map=True)
        self._num_rows = self.parquet_file.metadata.num_rows
        self._num_row_groups = self.parquet_file.metadata.num_row_groups
        
        # Get column names from schema
        self._column_names = [
            self.parquet_file.schema_arrow.field(i).name 
            for i in range(len(self.parquet_file.schema_arrow))
        ]
        
        # Build row group index for fast random access
        # Maps global row index -> (row_group_idx, local_row_idx)
        self._row_group_offsets = []
        offset = 0
        for rg_idx in range(self._num_row_groups):
            rg_rows = self.parquet_file.metadata.row_group(rg_idx).num_rows
            self._row_group_offsets.append((offset, rg_rows))
            offset += rg_rows
        
        logging.info(f"Dataset ready: {self._num_rows:,} rows in {self._num_row_groups} row groups")
        logging.info(f"Columns: {self._column_names}")
        
        # Cache for current row group to avoid re-reading for sequential access
        self._cached_rg_idx = -1
        self._cached_table = None

    def __len__(self):
        return self._num_rows
    
    def _get_row_group_for_idx(self, idx: int) -> tuple:
        """Find which row group contains the given global index."""
        for rg_idx, (offset, num_rows) in enumerate(self._row_group_offsets):
            if offset <= idx < offset + num_rows:
                local_idx = idx - offset
                return rg_idx, local_idx
        raise IndexError(f"Index {idx} out of range for dataset with {self._num_rows} rows")
    
    def _get_row(self, idx: int) -> Dict[str, Any]:
        """Get a single row, using caching for sequential access patterns."""
        rg_idx, local_idx = self._get_row_group_for_idx(idx)
        
        # Check cache
        if rg_idx != self._cached_rg_idx:
            # Read new row group
            self._cached_table = self.parquet_file.read_row_group(rg_idx)
            self._cached_rg_idx = rg_idx
        
        # Extract row data from cached table
        row_data = {}
        for col_name in self._column_names:
            row_data[col_name] = self._cached_table[col_name][local_idx].as_py()
        
        return row_data

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single training sample."""
        if idx < 0:
            idx = self._num_rows + idx
        if idx < 0 or idx >= self._num_rows:
            raise IndexError(f"Index {idx} out of range")
        
        # Get row data (uses caching)
        row = self._get_row(idx)
        
        # 1. Decode Image
        try:
            image_val = row.get("image")
            
            if isinstance(image_val, dict) and 'bytes' in image_val:
                image_bytes = image_val['bytes']
            else:
                image_bytes = image_val
            
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            logging.warning(f"Error decoding image at idx {idx}: {e}")
            image = Image.new("RGB", (224, 224))
            
        original_size = image.size  # (W, H)

        # 2. Apply Transform
        if self.transform:
            pixel_values = self.transform(image)
        else:
            import torchvision.transforms as T
            pixel_values = T.ToTensor()(image)

        # 3. Get Instruction / Tokenized Features
        if "input_ids" in self._column_names and row.get("input_ids") is not None:
            input_ids_list = row["input_ids"]
            attn_mask_list = row["attention_mask"]
            
            input_ids = torch.tensor(np.array(input_ids_list), dtype=torch.long)
            attention_mask = torch.tensor(np.array(attn_mask_list), dtype=torch.long)
            instruction = ""
        else:
            instruction = row.get("task", "") or ""
            input_ids = None
            attention_mask = None

        # 4. Get Bounding Box
        bbox = row.get("bbox", [0.0, 0.0, 0.0, 0.0])
        if bbox is None:
            bbox = [0.0, 0.0, 0.0, 0.0]
        
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)
        bbox_tensor = torch.clamp(bbox_tensor, 0.0, 1.0)

        return {
            "pixel_values": pixel_values,
            "instruction": instruction,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ground_truth_bbox": bbox_tensor,
            "original_size": torch.tensor(original_size)
        }
    
    def __del__(self):
        """Clean up cached data."""
        self._cached_table = None


if __name__ == "__main__":
    # Verification
    import sys
    logging.basicConfig(level=logging.INFO)
    
    path = "dataset/screenspot_training.parquet"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    
    print(f"Testing loader with {path}")
    try:
        ds = ScreenspotDataset(path)
        print(f"Dataset length: {len(ds)}")
        sample = ds[0]
        print("Sample keys:", sample.keys())
        print("Instruction:", sample["instruction"])
        print("BBox:", sample["ground_truth_bbox"])
        print("Image shape:", sample["pixel_values"].shape)
        if sample["input_ids"] is not None:
            print("Input IDs shape:", sample["input_ids"].shape)
    except FileNotFoundError:
        print("Parquet file not found, skipping test.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An error occurred: {e}")
