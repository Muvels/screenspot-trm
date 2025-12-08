"""Loss functions for bounding box prediction."""

from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BBoxLoss(nn.Module):
    """Loss for bounding box prediction with optional deep supervision.
    
    Supports:
    - SmoothL1 loss (default, more robust to outliers)
    - L1 loss
    - MSE loss
    - GIoU loss (generalized IoU)
    
    Deep supervision applies the loss to intermediate predictions
    with a weighted contribution.
    
    Attributes:
        loss_type: Type of loss function
        deep_supervision_weight: Weight for intermediate losses
        reduction: How to reduce the loss
    """
    
    def __init__(
        self,
        loss_type: Literal["smooth_l1", "l1", "mse", "giou"] = "smooth_l1",
        deep_supervision_weight: float = 0.1,
        reduction: Literal["mean", "sum", "none"] = "mean",
        smooth_l1_beta: float = 1.0,
    ):
        """Initialize loss.
        
        Args:
            loss_type: Type of loss function to use
            deep_supervision_weight: Weight for intermediate predictions
            reduction: How to reduce the loss ("mean", "sum", "none")
            smooth_l1_beta: Beta parameter for SmoothL1Loss
        """
        super().__init__()
        
        self.loss_type = loss_type
        self.ds_weight = deep_supervision_weight
        self.reduction = reduction
        self.smooth_l1_beta = smooth_l1_beta
        
        # Setup base loss function
        if loss_type == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss(reduction=reduction, beta=smooth_l1_beta)
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss(reduction=reduction)
        elif loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction=reduction)
        elif loss_type == "giou":
            # GIoU doesn't use nn loss, handled separately
            self.loss_fn = None
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _compute_giou_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute Generalized IoU loss.
        
        Args:
            pred: Predicted boxes [B, 4] in (x1, y1, x2, y2)
            target: Target boxes [B, 4] in (x1, y1, x2, y2)
            
        Returns:
            GIoU loss (1 - GIoU)
        """
        # Ensure valid boxes (x1 < x2, y1 < y2)
        pred_x1 = torch.min(pred[..., 0], pred[..., 2])
        pred_y1 = torch.min(pred[..., 1], pred[..., 3])
        pred_x2 = torch.max(pred[..., 0], pred[..., 2])
        pred_y2 = torch.max(pred[..., 1], pred[..., 3])
        
        target_x1 = target[..., 0]
        target_y1 = target[..., 1]
        target_x2 = target[..., 2]
        target_y2 = target[..., 3]
        
        # Intersection
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        
        # Areas
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        
        # Union
        union_area = pred_area + target_area - inter_area
        
        # IoU
        iou = inter_area / union_area.clamp(min=1e-6)
        
        # Enclosing box
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)
        
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
        
        # GIoU
        giou = iou - (enclose_area - union_area) / enclose_area.clamp(min=1e-6)
        
        # Loss is 1 - GIoU (so that lower is better)
        loss = 1 - giou
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
    
    def _compute_single_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute loss for a single prediction.
        
        Args:
            pred: Predicted boxes [B, 4]
            target: Target boxes [B, 4]
            
        Returns:
            Loss value
        """
        if self.loss_type == "giou":
            return self._compute_giou_loss(pred, target)
        else:
            return self.loss_fn(pred, target)
    
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        intermediates: Optional[List[Tensor]] = None,
    ) -> Tensor:
        """Compute loss with optional deep supervision.
        
        Args:
            pred: Final predicted boxes [B, 4]
            target: Target boxes [B, 4]
            intermediates: Optional list of intermediate predictions
            
        Returns:
            Total loss value
        """
        # Main loss
        main_loss = self._compute_single_loss(pred, target)
        
        # Deep supervision loss
        if intermediates is not None and len(intermediates) > 1:
            # Apply loss to all but the last (which is same as pred)
            ds_losses = [
                self._compute_single_loss(p, target)
                for p in intermediates[:-1]
            ]
            ds_loss = sum(ds_losses) / len(ds_losses)
            return main_loss + self.ds_weight * ds_loss
        
        return main_loss


class CombinedBBoxLoss(nn.Module):
    """Combined loss with multiple components.
    
    Combines coordinate regression loss with IoU-based loss.
    """
    
    def __init__(
        self,
        coord_weight: float = 1.0,
        giou_weight: float = 1.0,
        deep_supervision_weight: float = 0.1,
    ):
        """Initialize combined loss.
        
        Args:
            coord_weight: Weight for coordinate loss
            giou_weight: Weight for GIoU loss
            deep_supervision_weight: Weight for intermediate predictions
        """
        super().__init__()
        
        self.coord_loss = BBoxLoss(
            loss_type="smooth_l1",
            deep_supervision_weight=deep_supervision_weight,
        )
        self.giou_loss = BBoxLoss(
            loss_type="giou",
            deep_supervision_weight=deep_supervision_weight,
        )
        self.coord_weight = coord_weight
        self.giou_weight = giou_weight
    
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        intermediates: Optional[List[Tensor]] = None,
    ) -> Tensor:
        """Compute combined loss.
        
        Args:
            pred: Predicted boxes [B, 4]
            target: Target boxes [B, 4]
            intermediates: Optional intermediate predictions
            
        Returns:
            Combined loss value
        """
        coord = self.coord_loss(pred, target, intermediates)
        giou = self.giou_loss(pred, target, intermediates)
        return self.coord_weight * coord + self.giou_weight * giou
