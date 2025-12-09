"""Advanced fusion layers with cross-attention for spatial features."""

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion between image patches and text.
    
    Uses text embedding to attend to image patches, producing
    a text-conditioned spatial representation.
    
    This is much more powerful than simple concat+project because
    it allows the model to focus on relevant image regions based
    on the task instruction.
    """
    
    def __init__(
        self,
        clip_dim: int = 768,
        txt_dim: Optional[int] = None,  # If different from clip_dim (patches)
        trm_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.clip_dim = clip_dim  # patch dim
        self.txt_dim = txt_dim if txt_dim is not None else clip_dim
        self.trm_dim = trm_dim
        self.num_heads = num_heads
        
        # Project to TRM dim - patches and text may have different input dims
        self.img_proj = nn.Linear(clip_dim, trm_dim)
        self.txt_proj = nn.Linear(self.txt_dim, trm_dim)
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=trm_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(trm_dim)
            for _ in range(num_layers)
        ])
        
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(trm_dim, trm_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(trm_dim * 4, trm_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(trm_dim)
            for _ in range(num_layers)
        ])
        
        # Final pooling and projection
        self.output_norm = nn.LayerNorm(trm_dim)
    
    def forward(
        self,
        img_patches: Tensor,
        txt_emb: Tensor,
        img_pooled: Optional[Tensor] = None,
    ) -> Tensor:
        """Fuse image patches with text via cross-attention.
        
        Args:
            img_patches: Image patch embeddings [B, num_patches, clip_dim]
            txt_emb: Text embedding [B, clip_dim]
            img_pooled: Optional pooled image embedding (unused, for compatibility)
            
        Returns:
            Fused context [B, trm_dim]
        """
        # Project to TRM dim
        patches = self.img_proj(img_patches)  # [B, num_patches, trm_dim]
        text = self.txt_proj(txt_emb).unsqueeze(1)  # [B, 1, trm_dim]
        
        # Cross-attention: text attends to image patches
        query = text
        for i, (attn, norm, ffn, ffn_norm) in enumerate(
            zip(self.cross_attn_layers, self.norms, self.ffns, self.ffn_norms)
        ):
            # Cross-attention
            attn_out, _ = attn(query, patches, patches)
            query = norm(query + attn_out)
            
            # FFN
            ffn_out = ffn(query)
            query = ffn_norm(query + ffn_out)
        
        # Pool to single vector
        output = self.output_norm(query.squeeze(1))
        
        return output


class SpatialAwareFusion(nn.Module):
    """Fusion that preserves spatial information for bbox prediction.
    
    Instead of just outputting a single vector, this produces
    a spatial feature map that can be used for better localization.
    """
    
    def __init__(
        self,
        clip_dim: int = 768,
        txt_dim: Optional[int] = None,  # If different from clip_dim (patches)
        trm_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        grid_size: int = 14,  # ViT-B/16 with 224x224 -> 14x14 patches
    ):
        super().__init__()
        
        self.clip_dim = clip_dim
        self.txt_dim = txt_dim if txt_dim is not None else clip_dim
        self.trm_dim = trm_dim
        self.grid_size = grid_size
        
        # Project dims - patches and text may have different input dims
        self.img_proj = nn.Linear(clip_dim, trm_dim)
        self.txt_proj = nn.Linear(self.txt_dim, trm_dim)
        
        # Learned 2D positional embeddings for spatial awareness
        self.pos_embed = nn.Parameter(
            torch.randn(1, grid_size * grid_size, trm_dim) * 0.02
        )
        
        # Cross-attention: text -> patches
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=trm_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(trm_dim)
        
        # Self-attention on patches
        self.self_attn = nn.MultiheadAttention(
            embed_dim=trm_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(trm_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(trm_dim, trm_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(trm_dim * 4, trm_dim),
        )
        self.ffn_norm = nn.LayerNorm(trm_dim)
        
        # Pool spatial features to single vector for TRM
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
        )
        self.output_proj = nn.Linear(trm_dim, trm_dim)
    
    def forward(
        self,
        img_patches: Tensor,
        txt_emb: Tensor,
        img_pooled: Optional[Tensor] = None,
    ) -> Tensor:
        """Fuse with spatial awareness.
        
        Args:
            img_patches: [B, num_patches, clip_dim]
            txt_emb: [B, clip_dim]
            img_pooled: Optional, unused
            
        Returns:
            Fused context [B, trm_dim]
        """
        B = img_patches.shape[0]
        
        # Project and add positional embeddings
        patches = self.img_proj(img_patches) + self.pos_embed
        text = self.txt_proj(txt_emb).unsqueeze(1)
        
        # Cross-attention: condition patches on text
        cross_out, attn_weights = self.cross_attn(patches, text, text)
        patches = self.cross_norm(patches + cross_out)
        
        # Self-attention on conditioned patches
        self_out, _ = self.self_attn(patches, patches, patches)
        patches = self.self_norm(patches + self_out)
        
        # FFN
        ffn_out = self.ffn(patches)
        patches = self.ffn_norm(patches + ffn_out)
        
        # Pool to single vector
        # [B, num_patches, trm_dim] -> [B, trm_dim, num_patches] -> pool -> [B, trm_dim]
        pooled = self.pool(patches.transpose(1, 2)).squeeze(-1)
        output = self.output_proj(pooled)
        
        return output
