import io
import torch
import pandas as pd
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
            self.df = pd.read_parquet(parquet_path)
            logging.info(f"Loaded {len(self.df)} examples.")
        except Exception as e:
            logging.error(f"Failed to load parquet file: {e}")
            raise e

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        
        # 1. Decode Image
        try:
            image_entry = row['image']
            if isinstance(image_entry, dict) and 'bytes' in image_entry:
                image_bytes = image_entry['bytes']
            else:
                 # Backup if format is strictly bytes
                 image_bytes = image_entry
            
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            logging.warning(f"Error decoding image at idx {idx}: {e}")
            image = Image.new("RGB", (224, 224))
            
        original_size = image.size # (W, H)

        # 2. Apply Transform (Vision Encoder Preprocessing)
        if self.transform:
            pixel_values = self.transform(image)
        else:
            # Default to tensor if no transform
            import torchvision.transforms as T
            pixel_values = T.ToTensor()(image)

        # 3. Get Instruction
        instruction = row['task']
        if instruction is None:
            instruction = ""

        # 4. Get Bounding Box
        # Ensure it's a tensor of float32
        bbox = row['bbox'] # [x1, y1, x2, y2] normalized
        if bbox is None: 
            bbox = [0.0, 0.0, 0.0, 0.0]
        
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)

        return {
            "pixel_values": pixel_values,
            "instruction": instruction,
            "ground_truth_bbox": bbox_tensor,
            "original_size": torch.tensor(original_size) # (W, H)
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
