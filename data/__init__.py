"""Data loading utilities for ScreenSpot TRM."""

from .screenspot_dataset import (
    ScreenSpotDataset,
    ScreenSpotDatasetCached,
    collate_fn,
    collate_fn_cached,
)

__all__ = [
    "ScreenSpotDataset",
    "ScreenSpotDatasetCached",
    "collate_fn",
    "collate_fn_cached",
]
