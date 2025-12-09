"""Visualization utilities for bounding box predictions."""

import io
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor


def draw_bbox_on_image(
    image: Image.Image,
    pred_bbox: Tensor,
    gt_bbox: Optional[Tensor] = None,
    pred_color: str = "red",
    gt_color: str = "green",
    line_width: int = 3,
    show_labels: bool = True,
) -> Image.Image:
    """Draw bounding boxes on an image.
    
    Args:
        image: PIL Image
        pred_bbox: Predicted bbox [4] in normalized [x1, y1, x2, y2]
        gt_bbox: Optional ground truth bbox [4] in normalized coords
        pred_color: Color for predicted bbox
        gt_color: Color for ground truth bbox
        line_width: Width of bbox lines
        show_labels: Whether to show "Pred" and "GT" labels
        
    Returns:
        Image with bboxes drawn
    """
    # Make a copy to avoid modifying original
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    width, height = img.size
    
    # Convert normalized coords to pixel coords
    def to_pixels(bbox):
        return [
            int(bbox[0] * width),
            int(bbox[1] * height),
            int(bbox[2] * width),
            int(bbox[3] * height),
        ]
    
    # Draw ground truth first (so prediction is on top)
    if gt_bbox is not None:
        gt_pixels = to_pixels(gt_bbox)
        draw.rectangle(gt_pixels, outline=gt_color, width=line_width)
        if show_labels:
            draw.text((gt_pixels[0], gt_pixels[1] - 15), "GT", fill=gt_color)
    
    # Draw prediction
    pred_pixels = to_pixels(pred_bbox)
    draw.rectangle(pred_pixels, outline=pred_color, width=line_width)
    if show_labels:
        draw.text((pred_pixels[0], pred_pixels[1] - 15), "Pred", fill=pred_color)
    
    return img


def create_sample_grid(
    images: List[Image.Image],
    pred_bboxes: Tensor,
    gt_bboxes: Optional[Tensor] = None,
    tasks: Optional[List[str]] = None,
    max_cols: int = 4,
    cell_size: Tuple[int, int] = (256, 256),
    pred_color: str = "red",
    gt_color: str = "green",
) -> Image.Image:
    """Create a grid of sample images with bboxes.
    
    Args:
        images: List of PIL Images
        pred_bboxes: Predicted bboxes [N, 4]
        gt_bboxes: Optional ground truth bboxes [N, 4]
        tasks: Optional list of task descriptions
        max_cols: Maximum columns in grid
        cell_size: Size to resize each cell to
        pred_color: Color for predicted bboxes
        gt_color: Color for ground truth bboxes
        
    Returns:
        Grid image with all samples
    """
    n_samples = len(images)
    n_cols = min(n_samples, max_cols)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    cell_w, cell_h = cell_size
    grid_w = n_cols * cell_w
    grid_h = n_rows * cell_h
    
    grid = Image.new("RGB", (grid_w, grid_h), color="white")
    
    for i, img in enumerate(images):
        # Get bboxes
        pred_bbox = pred_bboxes[i]
        gt_bbox = gt_bboxes[i] if gt_bboxes is not None else None
        
        # Draw bboxes on image
        img_with_bbox = draw_bbox_on_image(
            img, pred_bbox, gt_bbox,
            pred_color=pred_color, gt_color=gt_color,
        )
        
        # Resize to cell size
        img_resized = img_with_bbox.resize(cell_size, Image.Resampling.LANCZOS)
        
        # Place in grid
        row = i // n_cols
        col = i % n_cols
        grid.paste(img_resized, (col * cell_w, row * cell_h))
    
    return grid


def save_sample_images(
    images: List[Image.Image],
    pred_bboxes: Tensor,
    gt_bboxes: Optional[Tensor] = None,
    tasks: Optional[List[str]] = None,
    output_dir: Union[str, Path] = "samples",
    step: int = 0,
    prefix: str = "sample",
) -> List[Path]:
    """Save individual sample images with bboxes.
    
    Args:
        images: List of PIL Images
        pred_bboxes: Predicted bboxes [N, 4]
        gt_bboxes: Optional ground truth bboxes [N, 4]
        tasks: Optional task descriptions
        output_dir: Directory to save images
        step: Training step (for filename)
        prefix: Filename prefix
        
    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for i, img in enumerate(images):
        pred_bbox = pred_bboxes[i]
        gt_bbox = gt_bboxes[i] if gt_bboxes is not None else None
        
        img_with_bbox = draw_bbox_on_image(img, pred_bbox, gt_bbox)
        
        filename = f"{prefix}_step{step:06d}_sample{i:03d}.png"
        filepath = output_dir / filename
        img_with_bbox.save(filepath)
        saved_paths.append(filepath)
    
    return saved_paths


def log_samples_to_wandb(
    images: List[Image.Image],
    pred_bboxes: Tensor,
    gt_bboxes: Optional[Tensor] = None,
    tasks: Optional[List[str]] = None,
    ious: Optional[Tensor] = None,
    step: int = 0,
    prefix: str = "samples",
):
    """Log sample images with bboxes to wandb.
    
    Args:
        images: List of PIL Images
        pred_bboxes: Predicted bboxes [N, 4]
        gt_bboxes: Optional ground truth bboxes [N, 4]
        tasks: Optional task descriptions
        ious: Optional IoU values for each sample
        step: Training step
        prefix: wandb key prefix
    """
    try:
        import wandb
    except ImportError:
        print("wandb not available for logging samples")
        return
    
    wandb_images = []
    
    for i, img in enumerate(images):
        pred_bbox = pred_bboxes[i]
        gt_bbox = gt_bboxes[i] if gt_bboxes is not None else None
        
        img_with_bbox = draw_bbox_on_image(img, pred_bbox, gt_bbox)
        
        # Create caption
        caption_parts = []
        if tasks is not None and i < len(tasks):
            # Truncate long tasks
            task = tasks[i][:100] + "..." if len(tasks[i]) > 100 else tasks[i]
            caption_parts.append(f"Task: {task}")
        if ious is not None:
            caption_parts.append(f"IoU: {ious[i]:.3f}")
        
        caption = " | ".join(caption_parts) if caption_parts else None
        
        wandb_images.append(wandb.Image(img_with_bbox, caption=caption))
    
    wandb.log({f"{prefix}/predictions": wandb_images}, step=step)


def load_image_from_parquet_row(row) -> Image.Image:
    """Load image from a parquet row.
    
    Args:
        row: Pandas row with image data
        
    Returns:
        PIL Image
    """
    img_data = row["image"]
    if isinstance(img_data, dict):
        img_bytes = img_data["bytes"]
    else:
        img_bytes = img_data
    
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")
