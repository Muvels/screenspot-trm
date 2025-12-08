"""Metrics for bounding box prediction evaluation."""

from typing import Dict, List, Tuple, Union

import torch
from torch import Tensor


def compute_iou(pred_bbox: Tensor, gt_bbox: Tensor) -> Tensor:
    """Compute Intersection over Union (IoU) for bounding boxes.
    
    Args:
        pred_bbox: Predicted boxes [..., 4] in (x1, y1, x2, y2) format
        gt_bbox: Ground truth boxes [..., 4] in (x1, y1, x2, y2) format
        
    Returns:
        IoU values with same batch shape as inputs
    """
    # Intersection coordinates
    x1 = torch.max(pred_bbox[..., 0], gt_bbox[..., 0])
    y1 = torch.max(pred_bbox[..., 1], gt_bbox[..., 1])
    x2 = torch.min(pred_bbox[..., 2], gt_bbox[..., 2])
    y2 = torch.min(pred_bbox[..., 3], gt_bbox[..., 3])
    
    # Intersection area
    inter_width = (x2 - x1).clamp(min=0)
    inter_height = (y2 - y1).clamp(min=0)
    inter_area = inter_width * inter_height
    
    # Areas of each box
    pred_area = (pred_bbox[..., 2] - pred_bbox[..., 0]) * (pred_bbox[..., 3] - pred_bbox[..., 1])
    gt_area = (gt_bbox[..., 2] - gt_bbox[..., 0]) * (gt_bbox[..., 3] - gt_bbox[..., 1])
    
    # Union area
    union_area = pred_area + gt_area - inter_area
    
    # IoU
    iou = inter_area / union_area.clamp(min=1e-6)
    
    return iou


def compute_giou(pred_bbox: Tensor, gt_bbox: Tensor) -> Tensor:
    """Compute Generalized IoU (GIoU) for bounding boxes.
    
    GIoU = IoU - (C - U) / C
    where C is the area of the smallest enclosing box.
    
    Args:
        pred_bbox: Predicted boxes [..., 4]
        gt_bbox: Ground truth boxes [..., 4]
        
    Returns:
        GIoU values in range [-1, 1]
    """
    # Regular IoU
    iou = compute_iou(pred_bbox, gt_bbox)
    
    # Enclosing box
    enclose_x1 = torch.min(pred_bbox[..., 0], gt_bbox[..., 0])
    enclose_y1 = torch.min(pred_bbox[..., 1], gt_bbox[..., 1])
    enclose_x2 = torch.max(pred_bbox[..., 2], gt_bbox[..., 2])
    enclose_y2 = torch.max(pred_bbox[..., 3], gt_bbox[..., 3])
    
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    
    # Union area (recompute for GIoU)
    pred_area = (pred_bbox[..., 2] - pred_bbox[..., 0]) * (pred_bbox[..., 3] - pred_bbox[..., 1])
    gt_area = (gt_bbox[..., 2] - gt_bbox[..., 0]) * (gt_bbox[..., 3] - gt_bbox[..., 1])
    
    x1 = torch.max(pred_bbox[..., 0], gt_bbox[..., 0])
    y1 = torch.max(pred_bbox[..., 1], gt_bbox[..., 1])
    x2 = torch.min(pred_bbox[..., 2], gt_bbox[..., 2])
    y2 = torch.min(pred_bbox[..., 3], gt_bbox[..., 3])
    inter_area = ((x2 - x1).clamp(0) * (y2 - y1).clamp(0))
    union_area = pred_area + gt_area - inter_area
    
    # GIoU
    giou = iou - (enclose_area - union_area) / enclose_area.clamp(min=1e-6)
    
    return giou


def compute_center_distance(pred_bbox: Tensor, gt_bbox: Tensor) -> Tensor:
    """Compute L2 distance between box centers.
    
    Args:
        pred_bbox: Predicted boxes [..., 4]
        gt_bbox: Ground truth boxes [..., 4]
        
    Returns:
        L2 distance between centers (normalized coordinates)
    """
    # Compute centers
    pred_cx = (pred_bbox[..., 0] + pred_bbox[..., 2]) / 2
    pred_cy = (pred_bbox[..., 1] + pred_bbox[..., 3]) / 2
    gt_cx = (gt_bbox[..., 0] + gt_bbox[..., 2]) / 2
    gt_cy = (gt_bbox[..., 1] + gt_bbox[..., 3]) / 2
    
    # L2 distance
    distance = torch.sqrt((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2)
    
    return distance


def compute_center_distance_pixels(
    pred_bbox: Tensor,
    gt_bbox: Tensor,
    image_sizes: List[Tuple[int, int]],
) -> Tensor:
    """Compute center distance in pixels.
    
    Args:
        pred_bbox: Predicted boxes [B, 4] (normalized)
        gt_bbox: Ground truth boxes [B, 4] (normalized)
        image_sizes: List of (width, height) tuples
        
    Returns:
        Distance in pixels for each sample
    """
    # Convert to pixel coordinates
    device = pred_bbox.device
    widths = torch.tensor([s[0] for s in image_sizes], device=device, dtype=pred_bbox.dtype)
    heights = torch.tensor([s[1] for s in image_sizes], device=device, dtype=pred_bbox.dtype)
    
    # Centers in normalized coords
    pred_cx = (pred_bbox[..., 0] + pred_bbox[..., 2]) / 2
    pred_cy = (pred_bbox[..., 1] + pred_bbox[..., 3]) / 2
    gt_cx = (gt_bbox[..., 0] + gt_bbox[..., 2]) / 2
    gt_cy = (gt_bbox[..., 1] + gt_bbox[..., 3]) / 2
    
    # Convert to pixels
    pred_cx_px = pred_cx * widths
    pred_cy_px = pred_cy * heights
    gt_cx_px = gt_cx * widths
    gt_cy_px = gt_cy * heights
    
    # Distance
    distance = torch.sqrt((pred_cx_px - gt_cx_px) ** 2 + (pred_cy_px - gt_cy_px) ** 2)
    
    return distance


def compute_accuracy_at_threshold(
    pred_bbox: Tensor,
    gt_bbox: Tensor,
    threshold: float = 0.5,
) -> Tensor:
    """Compute accuracy at IoU threshold.
    
    Args:
        pred_bbox: Predicted boxes [B, 4]
        gt_bbox: Ground truth boxes [B, 4]
        threshold: IoU threshold
        
    Returns:
        Fraction of predictions with IoU >= threshold
    """
    iou = compute_iou(pred_bbox, gt_bbox)
    accuracy = (iou >= threshold).float().mean()
    return accuracy


def compute_metrics(
    pred_bbox: Tensor,
    gt_bbox: Tensor,
    image_sizes: List[Tuple[int, int]] = None,
    iou_thresholds: List[float] = [0.3, 0.5, 0.7],
) -> Dict[str, float]:
    """Compute comprehensive metrics for bounding box prediction.
    
    Args:
        pred_bbox: Predicted boxes [B, 4]
        gt_bbox: Ground truth boxes [B, 4]
        image_sizes: Optional list of (width, height) for pixel metrics
        iou_thresholds: IoU thresholds for accuracy computation
        
    Returns:
        Dictionary of metric names to values
    """
    metrics = {}
    
    # IoU
    iou = compute_iou(pred_bbox, gt_bbox)
    metrics["iou_mean"] = iou.mean().item()
    metrics["iou_median"] = iou.median().item()
    
    # GIoU
    giou = compute_giou(pred_bbox, gt_bbox)
    metrics["giou_mean"] = giou.mean().item()
    
    # Center distance (normalized)
    center_dist = compute_center_distance(pred_bbox, gt_bbox)
    metrics["center_dist_mean"] = center_dist.mean().item()
    metrics["center_dist_median"] = center_dist.median().item()
    
    # Center distance (pixels) if image sizes provided
    if image_sizes is not None:
        center_dist_px = compute_center_distance_pixels(pred_bbox, gt_bbox, image_sizes)
        metrics["center_dist_px_mean"] = center_dist_px.mean().item()
        metrics["center_dist_px_median"] = center_dist_px.median().item()
    
    # Accuracy at thresholds
    for thresh in iou_thresholds:
        acc = compute_accuracy_at_threshold(pred_bbox, gt_bbox, thresh)
        metrics[f"acc@{thresh}"] = acc.item()
    
    return metrics


def bbox_to_center_wh(bbox: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Convert (x1, y1, x2, y2) to (cx, cy, w, h).
    
    Args:
        bbox: Boxes [..., 4]
        
    Returns:
        Tuple of (cx, cy, w, h) tensors
    """
    cx = (bbox[..., 0] + bbox[..., 2]) / 2
    cy = (bbox[..., 1] + bbox[..., 3]) / 2
    w = bbox[..., 2] - bbox[..., 0]
    h = bbox[..., 3] - bbox[..., 1]
    return cx, cy, w, h


def center_wh_to_bbox(cx: Tensor, cy: Tensor, w: Tensor, h: Tensor) -> Tensor:
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2).
    
    Args:
        cx, cy, w, h: Center coordinates and dimensions
        
    Returns:
        Boxes [..., 4]
    """
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)
