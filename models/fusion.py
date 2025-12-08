"""Fusion layer for combining CLIP image and text embeddings."""

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor


class CLIPFusion(nn.Module):
    """Fuse image and text embeddings into TRM context.
    
    Supports multiple fusion strategies:
    - concat_proj: Concatenate embeddings and project to TRM dimension
    - add_proj: Add embeddings and project to TRM dimension
    - cross_attn: Cross-attention between image and text (future)
    
    Attributes:
        clip_dim: Input dimension from CLIP
        trm_dim: Output dimension for TRM
        fusion_type: Type of fusion strategy
    """
    
    def __init__(
        self,
        clip_dim: int = 768,
        trm_dim: int = 256,
        fusion_type: Literal["concat_proj", "add_proj"] = "concat_proj",
    ):
        """Initialize fusion layer.
        
        Args:
            clip_dim: Dimension of CLIP embeddings
            trm_dim: Dimension of TRM hidden state
            fusion_type: Fusion strategy to use
        """
        super().__init__()
        
        self.clip_dim = clip_dim
        self.trm_dim = trm_dim
        self.fusion_type = fusion_type
        
        if fusion_type == "concat_proj":
            # Concatenate image and text, then project
            self.proj = nn.Sequential(
                nn.Linear(clip_dim * 2, trm_dim),
                nn.LayerNorm(trm_dim),
                nn.GELU(),
            )
        elif fusion_type == "add_proj":
            # Add image and text, then project
            self.proj = nn.Sequential(
                nn.Linear(clip_dim, trm_dim),
                nn.LayerNorm(trm_dim),
                nn.GELU(),
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(self, img_emb: Tensor, txt_emb: Tensor) -> Tensor:
        """Fuse image and text embeddings.
        
        Args:
            img_emb: Image embeddings [B, clip_dim]
            txt_emb: Text embeddings [B, clip_dim]
            
        Returns:
            Fused context embedding [B, trm_dim]
        """
        if self.fusion_type == "concat_proj":
            combined = torch.cat([img_emb, txt_emb], dim=-1)
        elif self.fusion_type == "add_proj":
            combined = img_emb + txt_emb
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        
        return self.proj(combined)
