"""
Memory-Efficient Dataset Loader for Large Parquet Files

Uses memory-mapping and row group-aware batching for optimal performance.
Includes a custom sampler that maintains data locality within row groups.
"""

import pyarrow.parquet as pq
import pyarrow as pa
import torch
import numpy as np
import io
from PIL import Image
from torch.utils.data import Dataset, Sampler
from typing import Optional, Callable, Dict, Any, Iterator, List
import logging
import os
import random


class RowGroupBatchSampler(Sampler):
    """
    A batch sampler that groups samples by row group for efficient disk access.
    
    Instead of random access across the entire dataset (which causes constant
    row group cache misses), this sampler:
    1. Shuffles row groups
    2. Yields all samples from each row group before moving to the next
    3. Optionally shuffles samples within each row group
    
    This maintains data locality while still providing randomization.
    """
    
    def __init__(
        self, 
        row_group_offsets: List[tuple],  # [(offset, num_rows), ...]
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        self.row_group_offsets = row_group_offsets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Calculate total samples
        self.total_samples = sum(num_rows for _, num_rows in row_group_offsets)
        
    def __iter__(self) -> Iterator[List[int]]:
        # Get row group order (shuffled or sequential)
        rg_indices = list(range(len(self.row_group_offsets)))
        if self.shuffle:
            random.shuffle(rg_indices)
        
        # For each row group, yield batches
        for rg_idx in rg_indices:
            offset, num_rows = self.row_group_offsets[rg_idx]
            
            # Get indices within this row group
            indices = list(range(offset, offset + num_rows))
            if self.shuffle:
                random.shuffle(indices)
            
            # Yield batches
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch
    
    def __len__(self) -> int:
        if self.drop_last:
            return self.total_samples // self.batch_size
        else:
            return (self.total_samples + self.batch_size - 1) // self.batch_size


class ScreenspotDataset(Dataset):
    """
    Memory-efficient dataset loader for the Screenspot Parquet dataset.
    
    Uses memory-mapped reading to handle files larger than RAM.
    Optimized for row group-aware batch sampling.
    
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
            transform: Preprocessing function for images.
        """
        self.parquet_path = parquet_path
        self.transform = transform
        
        if not os.path.exists(parquet_path):
            logging.error(f"Parquet file not found at {parquet_path}")
            raise FileNotFoundError(f"Parquet file not found at {parquet_path}")

        logging.info(f"Opening dataset from {parquet_path} (memory-mapped)...")
        
        # Open parquet file WITHOUT loading into memory
        self.parquet_file = pq.ParquetFile(parquet_path, memory_map=True)
        self._num_rows = self.parquet_file.metadata.num_rows
        self._num_row_groups = self.parquet_file.metadata.num_row_groups
        
        # Get column names from schema
        self._column_names = [
            self.parquet_file.schema_arrow.field(i).name 
            for i in range(len(self.parquet_file.schema_arrow))
        ]
        
        # Build row group index for fast random access
        self._row_group_offsets = []
        offset = 0
        for rg_idx in range(self._num_row_groups):
            rg_rows = self.parquet_file.metadata.row_group(rg_idx).num_rows
            self._row_group_offsets.append((offset, rg_rows))
            offset += rg_rows
        
        logging.info(f"Dataset ready: {self._num_rows:,} rows in {self._num_row_groups} row groups")
        logging.info(f"Columns: {self._column_names}")
        
        # Cache for current row group
        self._cached_rg_idx = -1
        self._cached_table = None

    def __len__(self):
        return self._num_rows
    
    def get_row_group_offsets(self) -> List[tuple]:
        """Return row group offsets for batch sampler."""
        return self._row_group_offsets.copy()
    
    def _get_row_group_for_idx(self, idx: int) -> tuple:
        """Find which row group contains the given global index."""
        for rg_idx, (offset, num_rows) in enumerate(self._row_group_offsets):
            if offset <= idx < offset + num_rows:
                local_idx = idx - offset
                return rg_idx, local_idx
        raise IndexError(f"Index {idx} out of range")
    
    def _get_row(self, idx: int) -> Dict[str, Any]:
        """Get a single row, using caching for sequential access."""
        rg_idx, local_idx = self._get_row_group_for_idx(idx)
        
        # Check cache
        if rg_idx != self._cached_rg_idx:
            self._cached_table = self.parquet_file.read_row_group(rg_idx)
            self._cached_rg_idx = rg_idx
        
        # Extract row data
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
            
        original_size = image.size

        # 2. Apply Transform
        if self.transform:
            pixel_values = self.transform(image)
        else:
            import torchvision.transforms as T
            pixel_values = T.ToTensor()(image)

        # 3. Get Instruction / Tokenized Features
        if "input_ids" in self._column_names and row.get("input_ids") is not None:
            input_ids = torch.tensor(np.array(row["input_ids"]), dtype=torch.long)
            attention_mask = torch.tensor(np.array(row["attention_mask"]), dtype=torch.long)
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
        self._cached_table = None


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    path = "dataset/screenspot_training.parquet"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    
    print(f"Testing loader with {path}")
    try:
        ds = ScreenspotDataset(path)
        print(f"Dataset length: {len(ds)}")
        print(f"Row groups: {len(ds.get_row_group_offsets())}")
        
        # Test sampling
        sampler = RowGroupBatchSampler(ds.get_row_group_offsets(), batch_size=32)
        print(f"Total batches: {len(sampler)}")
        
        # Get first batch
        for batch_indices in sampler:
            print(f"First batch indices: {batch_indices[:5]}...")
            break
            
    except FileNotFoundError:
        print("Parquet file not found, skipping test.")
    except Exception as e:
        import traceback
        traceback.print_exc()
