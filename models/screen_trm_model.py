"""Full ScreenSpot TRM model for bounding box prediction."""

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .clip_backbone import CLIPBackbone
from .fusion import CLIPFusion
from .trm_core import TRMController
from .bbox_head import BBoxHead


@dataclass
class ModelConfig:
    """Configuration for ScreenBBoxTRMModel.
    
    Attributes:
        clip_model: CLIP model architecture name
        clip_pretrained: Pretrained weights to load
        trm_hidden_size: Dimension of TRM hidden state
        H_cycles: Number of outer deep supervision cycles
        L_cycles: Number of inner latent reasoning cycles
        L_layers: Number of layers per reasoning module
        expansion: Expansion factor for SwiGLU
        bbox_output_format: How to parameterize bbox prediction
        fusion_type: How to fuse image and text embeddings
    """
    # CLIP config
    clip_model: str = "ViT-B-16"
    clip_pretrained: str = "openai"
    
    # TRM config
    trm_hidden_size: int = 256
    H_cycles: int = 3
    L_cycles: int = 4
    L_layers: int = 2
    expansion: float = 4.0
    
    # BBox head config
    bbox_hidden_dim: int = 128
    bbox_output_format: Literal["xyxy", "cxcywh"] = "xyxy"
    
    # Fusion config
    fusion_type: Literal["concat_proj", "add_proj"] = "concat_proj"


class ScreenBBoxTRMModel(nn.Module):
    """Screen UI bounding box prediction model using TRM.
    
    Combines:
    1. Frozen CLIP encoder for image and text embeddings
    2. Fusion layer to combine embeddings
    3. TRM controller for recursive reasoning
    4. BBox regression head for final prediction
    
    Attributes:
        config: Model configuration
        clip: CLIP backbone (frozen)
        fusion: Fusion layer
        trm: TRM controller
        bbox_head: Bounding box regression head
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        device: str = "cuda",
    ):
        """Initialize the model.
        
        Args:
            config: Model configuration (uses defaults if None)
            device: Device to load CLIP model on
        """
        super().__init__()
        
        self.config = config or ModelConfig()
        
        # CLIP backbone (frozen)
        self.clip = CLIPBackbone(
            model_name=self.config.clip_model,
            pretrained=self.config.clip_pretrained,
            device=device,
        )
        
        # Fusion layer
        self.fusion = CLIPFusion(
            clip_dim=self.clip.embed_dim,
            trm_dim=self.config.trm_hidden_size,
            fusion_type=self.config.fusion_type,
        )
        
        # TRM controller
        self.trm = TRMController(
            hidden_size=self.config.trm_hidden_size,
            H_cycles=self.config.H_cycles,
            L_cycles=self.config.L_cycles,
            L_layers=self.config.L_layers,
            expansion=self.config.expansion,
        )
        
        # Bounding box head
        self.bbox_head = BBoxHead(
            input_dim=self.config.trm_hidden_size,
            hidden_dim=self.config.bbox_hidden_dim,
            output_format=self.config.bbox_output_format,
        )
    
    @property
    def preprocess(self):
        """Get CLIP image preprocessing transform."""
        return self.clip.preprocess
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters (excluding frozen CLIP).
        
        Returns:
            List of parameters that require gradients
        """
        params = []
        params.extend(self.fusion.parameters())
        params.extend(self.trm.parameters())
        params.extend(self.bbox_head.parameters())
        return list(params)
    
    def forward(
        self,
        images: Tensor,
        tasks: List[str],
        return_intermediates: bool = False,
    ) -> Tuple[Tensor, Optional[List[Tensor]]]:
        """Forward pass for bounding box prediction.
        
        Args:
            images: Preprocessed images [B, 3, 224, 224]
            tasks: List of task instruction strings
            return_intermediates: Whether to return intermediate bbox predictions
            
        Returns:
            bbox_pred: Predicted bounding boxes [B, 4] in normalized [x1,y1,x2,y2]
            bbox_intermediates: Optional list of intermediate bbox predictions
        """
        # 1. CLIP encoding (frozen, no grad)
        with torch.no_grad():
            img_emb, txt_emb = self.clip(images, tasks)
        
        # Ensure embeddings have gradients disabled from CLIP
        img_emb = img_emb.detach()
        txt_emb = txt_emb.detach()
        
        # 2. Fusion
        h_ctx = self.fusion(img_emb, txt_emb)
        
        # 3. TRM recursion
        y_final, intermediates = self.trm(h_ctx, return_intermediates)
        
        # 4. BBox prediction
        bbox_pred = self.bbox_head(y_final)
        
        # Optional: intermediate bbox predictions for deep supervision
        bbox_intermediates = None
        if intermediates is not None:
            bbox_intermediates = [self.bbox_head(y) for y in intermediates]
        
        return bbox_pred, bbox_intermediates
    
    def forward_with_embeddings(
        self,
        img_emb: Tensor,
        txt_emb: Tensor,
        return_intermediates: bool = False,
    ) -> Tuple[Tensor, Optional[List[Tensor]]]:
        """Forward pass with pre-computed CLIP embeddings.
        
        Useful for caching embeddings during training.
        
        Args:
            img_emb: Image embeddings [B, clip_dim]
            txt_emb: Text embeddings [B, clip_dim]
            return_intermediates: Whether to return intermediate predictions
            
        Returns:
            bbox_pred: Predicted bounding boxes [B, 4]
            bbox_intermediates: Optional list of intermediate predictions
        """
        h_ctx = self.fusion(img_emb, txt_emb)
        y_final, intermediates = self.trm(h_ctx, return_intermediates)
        bbox_pred = self.bbox_head(y_final)
        
        bbox_intermediates = None
        if intermediates is not None:
            bbox_intermediates = [self.bbox_head(y) for y in intermediates]
        
        return bbox_pred, bbox_intermediates
    
    def count_parameters(self) -> dict:
        """Count parameters by component.
        
        Returns:
            Dictionary with parameter counts
        """
        def count_params(module):
            return sum(p.numel() for p in module.parameters())
        
        def count_trainable(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        return {
            "clip_total": count_params(self.clip),
            "clip_trainable": count_trainable(self.clip),
            "fusion": count_params(self.fusion),
            "trm": count_params(self.trm),
            "bbox_head": count_params(self.bbox_head),
            "total": count_params(self),
            "trainable": count_trainable(self),
        }


def create_model(
    config: Optional[ModelConfig] = None,
    device: str = "cuda",
) -> ScreenBBoxTRMModel:
    """Create a ScreenBBoxTRMModel with the given config.
    
    Args:
        config: Model configuration
        device: Device to load model on
        
    Returns:
        Initialized model
    """
    model = ScreenBBoxTRMModel(config=config, device=device)
    return model
