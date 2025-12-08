"""ScreenSpot dataset for bounding box prediction."""

import io
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class ScreenSpotDataset(Dataset):
    """Dataset for ScreenSpot bounding box prediction.
    
    Loads data from a Parquet file containing:
    - image: struct with bytes blob
    - task: natural language instruction
    - image_width, image_height: original dimensions
    - bbox: normalized [x1, y1, x2, y2] coordinates
    
    Attributes:
        df: Pandas DataFrame with dataset
        clip_preprocess: CLIP image preprocessing transform
        indices: Indices for train/val split
    """
    
    def __init__(
        self,
        parquet_path: Union[str, Path],
        clip_preprocess: Callable,
        split: str = "train",
        val_split: float = 0.1,
        seed: int = 42,
    ):
        """Initialize dataset.
        
        Args:
            parquet_path: Path to parquet file
            clip_preprocess: CLIP image preprocessing transform
            split: Either "train" or "val"
            val_split: Fraction of data to use for validation
            seed: Random seed for reproducible splits
        """
        self.parquet_path = Path(parquet_path)
        self.clip_preprocess = clip_preprocess
        self.split = split
        
        # Load data
        self.df = pd.read_parquet(parquet_path)
        
        # Create train/val split
        n = len(self.df)
        indices = list(range(n))
        
        # Shuffle deterministically
        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=rng).tolist()
        indices = [indices[i] for i in perm]
        
        # Split
        val_size = int(n * val_split)
        if split == "val":
            self.indices = indices[:val_size]
        else:
            self.indices = indices[val_size:]
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, str, Tuple[int, int]]]:
        """Get a single sample.
        
        Args:
            idx: Index into the dataset
            
        Returns:
            Dictionary with:
            - image: Preprocessed image tensor [3, 224, 224]
            - task: Task instruction string
            - bbox: Bounding box tensor [4]
            - image_size: Tuple of (width, height)
        """
        # Get actual row index
        row_idx = self.indices[idx]
        row = self.df.iloc[row_idx]
        
        # Decode image from bytes
        img_data = row["image"]
        if isinstance(img_data, dict):
            img_bytes = img_data["bytes"]
        else:
            # Handle case where image is stored differently
            img_bytes = img_data
        
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Apply CLIP preprocessing
        image_tensor = self.clip_preprocess(image)
        
        # Get bbox as tensor
        bbox = row["bbox"]
        if isinstance(bbox, (list, tuple)):
            bbox = torch.tensor(bbox, dtype=torch.float32)
        else:
            bbox = torch.tensor(list(bbox), dtype=torch.float32)
        
        # Get original image size
        image_size = (int(row["image_width"]), int(row["image_height"]))
        
        return {
            "image": image_tensor,
            "task": str(row["task"]),
            "bbox": bbox,
            "image_size": image_size,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Union[Tensor, List]]:
    """Collate function for DataLoader.
    
    Stacks tensors and collects lists.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Collated batch dictionary
    """
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "tasks": [b["task"] for b in batch],
        "bboxes": torch.stack([b["bbox"] for b in batch]),
        "image_sizes": [b["image_size"] for b in batch],
    }


class ScreenSpotDatasetCached(Dataset):
    """Dataset with pre-computed CLIP embeddings for faster training.
    
    Useful when CLIP is frozen and embeddings don't change.
    """
    
    def __init__(
        self,
        parquet_path: Union[str, Path],
        embeddings_path: Union[str, Path],
        split: str = "train",
        val_split: float = 0.1,
        seed: int = 42,
    ):
        """Initialize cached dataset.
        
        Args:
            parquet_path: Path to parquet file
            embeddings_path: Path to pre-computed embeddings (.pt file)
            split: Either "train" or "val"
            val_split: Fraction of data for validation
            seed: Random seed
        """
        self.parquet_path = Path(parquet_path)
        self.embeddings_path = Path(embeddings_path)
        self.split = split
        
        # Load data
        self.df = pd.read_parquet(parquet_path)
        
        # Load embeddings
        embeddings = torch.load(embeddings_path)
        self.img_embeddings = embeddings["img_embeddings"]
        self.txt_embeddings = embeddings["txt_embeddings"]
        
        # Create split
        n = len(self.df)
        indices = list(range(n))
        
        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=rng).tolist()
        indices = [indices[i] for i in perm]
        
        val_size = int(n * val_split)
        if split == "val":
            self.indices = indices[:val_size]
        else:
            self.indices = indices[val_size:]
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        row_idx = self.indices[idx]
        row = self.df.iloc[row_idx]
        
        # Get cached embeddings
        img_emb = self.img_embeddings[row_idx]
        txt_emb = self.txt_embeddings[row_idx]
        
        # Get bbox
        bbox = row["bbox"]
        if isinstance(bbox, (list, tuple)):
            bbox = torch.tensor(bbox, dtype=torch.float32)
        else:
            bbox = torch.tensor(list(bbox), dtype=torch.float32)
        
        return {
            "img_emb": img_emb,
            "txt_emb": txt_emb,
            "bbox": bbox,
        }


def collate_fn_cached(batch: List[Dict]) -> Dict[str, Tensor]:
    """Collate function for cached dataset."""
    return {
        "img_embs": torch.stack([b["img_emb"] for b in batch]),
        "txt_embs": torch.stack([b["txt_emb"] for b in batch]),
        "bboxes": torch.stack([b["bbox"] for b in batch]),
    }
