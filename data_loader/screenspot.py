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
    Dataset loader for the Screenspot Parquet dataset.
    
    Expected Columns:
    - image: struct(bytes, ...)
    - task: str
    - bbox: list[float] [x1, y1, x2, y2]
    - image_width: int
    - image_height: int
    """
    def __init__(self, parquet_path: str, transform: Optional[Callable] = None):
        """
        Args:
            parquet_path: Path to the .parquet file.
            transform: Analysis/Preprocessing function for images (e.g., from CLIP/ViT).
                       Should accept a PIL Image and return a tensor.
        """
        self.parquet_path = parquet_path
        self.transform = transform
        
        if not os.path.exists(parquet_path):
             logging.error(f"Parquet file not found at {parquet_path}")
             # We might want to raise, but for now let's just log. 
             # Actually, raising is better to fail fast.
             raise FileNotFoundError(f"Parquet file not found at {parquet_path}")

        logging.info(f"Loading dataset from {parquet_path}...")
        try:
            # We use PyArrow Table for potentially better handling of nested types
            # or simply load it; if pandas fails, we can stick to pyarrow.
            self.table = pq.read_table(parquet_path)
            logging.info(f"Loaded {self.table.num_rows} examples.")
        except Exception as e:
            logging.error(f"Failed to load parquet file: {e}")
            raise e

    def __len__(self):
        return self.table.num_rows

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Accessing PyArrow row is tricky efficiently, but for random access:
        # We can slice the table or convert just the row to py dict.
        # slicing: table.slice(idx, length=1)
        # But this might be slow if repeating every time.
        # Ideally we might convert columns to numpy arrays if they fit in memory (images might not).
        # Let's try converting the single row to a python dict which handles the conversion.
        
        # NOTE: self.table[col][idx] is reasonably fast for column-oriented access.
        
        # 1. Decode Image
        try:
            # We flattened it to bytes in preprocess.py, so it should be direct bytes now
            # OR it might still be the struct if using original file.
            # Let's handle both.
            image_val = self.table["image"][idx].as_py()
            
            if isinstance(image_val, dict) and 'bytes' in image_val:
                image_bytes = image_val['bytes']
            else:
                 image_bytes = image_val
            
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            logging.warning(f"Error decoding image at idx {idx}: {e}")
            image = Image.new("RGB", (224, 224))
            
        original_size = image.size # (W, H)

        # 2. Apply Transform
        if self.transform:
            pixel_values = self.transform(image)
        else:
            import torchvision.transforms as T
            pixel_values = T.ToTensor()(image)

        # 3. Get Instruction / Tokenized Features
        # Check if pre-tokenized columns exist in schema
        table_cols = self.table.column_names
        
        if "input_ids" in table_cols:
             input_ids_list = self.table["input_ids"][idx].as_py()
             attn_mask_list = self.table["attention_mask"][idx].as_py()
             
             import numpy as np
             input_ids = torch.tensor(np.array(input_ids_list), dtype=torch.long)
             attention_mask = torch.tensor(np.array(attn_mask_list), dtype=torch.long)
             instruction = "" 
        else:
             instruction = self.table["task"][idx].as_py()
             if instruction is None:
                 instruction = ""
             input_ids = None
             attention_mask = None

        # 4. Get Bounding Box
        bbox = self.table["bbox"][idx].as_py() # [x1, y1, x2, y2]
        if bbox is None: 
            bbox = [0.0, 0.0, 0.0, 0.0]
        
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)
        # Clamp to [0, 1] to avoid out-of-bounds errors found in dataset
        bbox_tensor = torch.clamp(bbox_tensor, 0.0, 1.0)

        return {
            "pixel_values": pixel_values,
            "instruction": instruction,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ground_truth_bbox": bbox_tensor,
            "original_size": torch.tensor(original_size)
        }

if __name__ == "__main__":
    # verification
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
    except FileNotFoundError:
        print("Parquet file not found, skipping test.")
    except Exception as e:
        print(f"An error occurred: {e}")
