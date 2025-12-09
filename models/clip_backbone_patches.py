"""CLIP backbone with patch token extraction for spatial features."""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

import open_clip


class CLIPBackboneWithPatches(nn.Module):
    """CLIP encoder that returns both pooled and patch-level features.
    
    For bounding box prediction, we need spatial information from patches,
    not just the pooled CLS token.
    
    Attributes:
        model: The OpenCLIP model
        preprocess: Image preprocessing transform
        tokenizer: Text tokenizer
        embed_dim: Embedding dimension
        patch_size: Size of each patch (e.g., 16 for ViT-B/16)
        num_patches: Number of patches per side (e.g., 14 for 224/16)
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        device: str = "cuda",
    ):
        super().__init__()
        
        self.model_name = model_name
        self.pretrained = pretrained
        
        # Load model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # Get model info
        self.embed_dim = self.model.visual.output_dim  # 512 for pooled (after proj)
        self.patch_dim = self.model.visual.transformer.width  # 768 for patches (internal)
        
        # For ViT-B/16 with 224x224 input: 14x14 = 196 patches
        if "16" in model_name:
            self.patch_size = 16
        elif "14" in model_name:
            self.patch_size = 14
        else:
            self.patch_size = 16  # default
        
        self.image_size = 224  # Standard CLIP input size
        self.num_patches = self.image_size // self.patch_size  # 14 for ViT-B/16
        
        self.freeze()
    
    def freeze(self) -> None:
        """Freeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    def encode_image_with_patches(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode images and return both pooled and patch features.
        
        Args:
            images: Preprocessed images [B, 3, 224, 224]
            
        Returns:
            pooled: Pooled CLS embedding [B, embed_dim]
            patches: Patch embeddings [B, num_patches^2, embed_dim]
        """
        visual = self.model.visual
        
        # Get patch embeddings before pooling
        x = visual.conv1(images)  # [B, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, width, grid^2]
        x = x.permute(0, 2, 1)  # [B, grid^2, width]
        
        # Add class token
        class_token = visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([class_token, x], dim=1)  # [B, 1 + grid^2, width]
        
        # Add positional embedding
        x = x + visual.positional_embedding.to(x.dtype)
        
        # Pre-norm
        x = visual.ln_pre(x)
        
        # Transformer
        x = x.permute(1, 0, 2)  # [L, B, D]
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)  # [B, L, D]
        
        # Post-norm
        x = visual.ln_post(x)
        
        # Split CLS and patches
        cls_token = x[:, 0]  # [B, D]
        patch_tokens = x[:, 1:]  # [B, num_patches^2, D]
        
        # Project CLS token
        if visual.proj is not None:
            cls_token = cls_token @ visual.proj
        
        return cls_token, patch_tokens
    
    def encode_text(self, texts: List[str]) -> Tensor:
        """Encode text strings."""
        device = next(self.model.parameters()).device
        tokens = self.tokenizer(texts).to(device)
        return self.model.encode_text(tokens)
    
    def forward(
        self,
        images: Tensor,
        texts: List[str],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Encode images and texts.
        
        Args:
            images: Preprocessed images [B, 3, 224, 224]
            texts: List of text strings
            
        Returns:
            img_pooled: Pooled image embedding [B, embed_dim] (512)
            img_patches: Patch embeddings [B, num_patches^2, patch_dim] (768)
            txt_emb: Text embedding [B, embed_dim] (512)
        """
        img_pooled, img_patches = self.encode_image_with_patches(images)
        txt_emb = self.encode_text(texts)
        return img_pooled, img_patches, txt_emb
