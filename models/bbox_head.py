"""Bounding box regression head."""

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor


class BBoxHead(nn.Module):
    """Regress normalized bounding box from TRM output.
    
    Supports two output formats:
    - xyxy: Direct prediction of (x1, y1, x2, y2) with sigmoid
    - cxcywh: Predict (cx, cy, w, h) and convert to (x1, y1, x2, y2)
    
    The cxcywh format can provide more stable training since width/height
    are always positive and center coordinates are independent.
    
    Attributes:
        input_dim: Dimension of input features
        hidden_dim: Dimension of hidden layers
        output_format: Format for internal prediction
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        output_format: Literal["xyxy", "cxcywh"] = "xyxy",
    ):
        """Initialize bbox head.
        
        Args:
            input_dim: Dimension of input features from TRM
            hidden_dim: Dimension of hidden layers
            output_format: How to parameterize the bbox prediction
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_format = output_format
        
        # MLP for regression
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )
        
        # Initialize final layer with small weights for stable training
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        # Initialize final layer with small weights
        # This ensures initial predictions are close to center of image
        final_layer = self.mlp[-1]
        nn.init.xavier_uniform_(final_layer.weight, gain=0.1)
        nn.init.zeros_(final_layer.bias)
    
    def forward(self, y: Tensor) -> Tensor:
        """Predict bounding box from TRM output.
        
        Args:
            y: TRM answer embedding [B, input_dim]
            
        Returns:
            Predicted bbox [B, 4] in normalized (x1, y1, x2, y2) format
        """
        raw = self.mlp(y)  # [B, 4]
        
        if self.output_format == "cxcywh":
            # Predict (cx, cy, w, h), convert to (x1, y1, x2, y2)
            # Sigmoid ensures all values in [0, 1]
            raw_sigmoid = torch.sigmoid(raw)
            cx = raw_sigmoid[..., 0:1]
            cy = raw_sigmoid[..., 1:2]
            w = raw_sigmoid[..., 2:3]
            h = raw_sigmoid[..., 3:4]
            
            # Convert to corners
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            
            # Clamp to [0, 1]
            bbox = torch.cat([x1, y1, x2, y2], dim=-1).clamp(0, 1)
        else:
            # Direct (x1, y1, x2, y2) with sigmoid
            bbox = torch.sigmoid(raw)
        
        return bbox
    
    def forward_with_uncertainty(self, y: Tensor) -> tuple[Tensor, Tensor]:
        """Predict bbox with uncertainty estimate.
        
        For future use - predict mean and variance.
        Currently just returns zeros for variance.
        
        Args:
            y: TRM answer embedding [B, input_dim]
            
        Returns:
            bbox: Predicted bbox [B, 4]
            uncertainty: Uncertainty estimate [B, 4]
        """
        bbox = self.forward(y)
        uncertainty = torch.zeros_like(bbox)
        return bbox, uncertainty
