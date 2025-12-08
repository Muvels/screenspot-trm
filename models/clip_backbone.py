"""CLIP backbone for image and text encoding."""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

import open_clip


class CLIPBackbone(nn.Module):
    """Frozen CLIP ViT-B/16 encoder for images and text.
    
    Uses OpenCLIP for flexibility and access to various pretrained weights.
    By default, all parameters are frozen (no gradient updates).
    
    Attributes:
        model: The OpenCLIP model
        preprocess: Image preprocessing transform
        tokenizer: Text tokenizer
        embed_dim: Embedding dimension (768 for ViT-B/16)
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        device: str = "cuda",
    ):
        """Initialize CLIP backbone.
        
        Args:
            model_name: CLIP model architecture name
            pretrained: Pretrained weights to load
            device: Device to load model on
        """
        super().__init__()
        
        self.model_name = model_name
        self.pretrained = pretrained
        
        # Load model and transforms
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # Get embedding dimension
        self.embed_dim = self.model.visual.output_dim
        
        # Freeze all parameters by default
        self.freeze()
    
    def freeze(self) -> None:
        """Freeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    def unfreeze_visual_layers(self, num_layers: int) -> None:
        """Unfreeze the last N layers of the visual encoder.
        
        Args:
            num_layers: Number of transformer layers to unfreeze from the end
        """
        # First freeze everything
        self.freeze()
        
        # Unfreeze specified layers
        visual = self.model.visual
        if hasattr(visual, 'transformer'):
            # OpenCLIP ViT structure
            blocks = visual.transformer.resblocks
            for block in blocks[-num_layers:]:
                for param in block.parameters():
                    param.requires_grad = True
        
        # Also unfreeze the projection layer if unfreezing any layers
        if num_layers > 0 and hasattr(visual, 'proj'):
            if visual.proj is not None:
                visual.proj.requires_grad = True
    
    def encode_image(self, images: Tensor) -> Tensor:
        """Encode images to embeddings.
        
        Args:
            images: Preprocessed images [B, 3, 224, 224]
            
        Returns:
            Image embeddings [B, embed_dim]
        """
        return self.model.encode_image(images)
    
    def encode_text(self, texts: List[str]) -> Tensor:
        """Encode text strings to embeddings.
        
        Args:
            texts: List of text strings
            
        Returns:
            Text embeddings [B, embed_dim]
        """
        # Tokenize on the same device as the model
        device = next(self.model.parameters()).device
        tokens = self.tokenizer(texts).to(device)
        return self.model.encode_text(tokens)
    
    def forward(
        self,
        images: Tensor,
        texts: List[str],
    ) -> Tuple[Tensor, Tensor]:
        """Encode both images and texts.
        
        Args:
            images: Preprocessed images [B, 3, 224, 224]
            texts: List of text strings
            
        Returns:
            Tuple of (image_embeddings, text_embeddings), each [B, embed_dim]
        """
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(texts)
        return img_emb, txt_emb
