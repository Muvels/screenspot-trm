import torch
import torch.nn as nn
from typing import Optional, Dict

from models.encodings import VisionTextEncoder
from models.trm import TinyRecursiveModel
from models.policy import BBoxPolicyHead, ValueHead

class InfoMaxAgent(nn.Module):
    """
    Full Agent Model.
    Structure:
    1. Vision+Text Encoder -> Context
    2. TRM (Recursive Core) -> Final Latent
    3. Policy -> Action (BBox)
    4. Value -> Estimate
    """
    def __init__(self, 
                 vision_text_model: str = "openai/clip-vit-base-patch32", 
                 hidden_size: int = 512, 
                 trm_layers: int = 2,
                 trm_depth: int = 6,
                 freeze_backbone: bool = True):
        super().__init__()
        
        self.encoder = VisionTextEncoder(model_name=vision_text_model, freeze_backbone=freeze_backbone)
        # Note: encoding output dim is configurable in VisionTextEncoder, defaults to 512
        if self.encoder.out_dim != hidden_size:
            # We could add a projection here or change encoder config
            # For now assume they match
            pass
            
        self.core = TinyRecursiveModel(
            hidden_size=hidden_size, 
            num_heads=8, 
            num_layers=trm_layers, 
            recursion_depth=trm_depth
        )
        
        self.policy_head = BBoxPolicyHead(input_dim=hidden_size)
        self.value_head = ValueHead(input_dim=hidden_size)
        
    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Args:
            pixel_values: (B, C, H, W)
            input_ids: (B, L)
            attention_mask: (B, L)
            
        Returns:
            bbox: (B, Steps, 4)
            value: (B, Steps, 1)
            final_state: (B, Steps, Seq, H)
        """
        # 1. Encode
        context = self.encoder(pixel_values, input_ids, attention_mask) # (B, Seq, H)
        
        # 2. Recurse
        trm_outputs = self.core(context) # (B, Steps, Seq, H)
        
        # 3. Aggregate for Heads
        # We need to process each step.
        # Flatten steps into batch dimension for efficiency or loop.
        # Shape: (B, Steps, Seq, H) -> (B * Steps, Seq, H)
        B_size, Steps, Seq, H = trm_outputs.shape
        trm_flat = trm_outputs.view(B_size * Steps, Seq, H)
        
        # Aggregate (Mean Pool)
        state_vec_flat = trm_flat.mean(dim=1) # (B * Steps, H)
        
        # 4. Heads
        bbox_flat = self.policy_head(state_vec_flat) # (B * Steps, 4)
        value_flat = self.value_head(state_vec_flat) # (B * Steps, 1)
        
        # Reshape back to (B, Steps, ...)
        bbox = bbox_flat.view(B_size, Steps, 4)
        value = value_flat.view(B_size, Steps, 1)
        
        # For compatibility, if we just want "inference" output, we might take the last one.
        # But for training we return all.
        # Let the trainer handle selecting the last one for inference/metrics.
        
        return bbox, value, trm_outputs

if __name__ == "__main__":
    # Test
    agent = InfoMaxAgent(trm_depth=6, trm_layers=2)
    B = 2
    img = torch.randn(B, 3, 224, 224)
    txt = torch.randint(0, 1000, (B, 10))
    bbox, val, _ = agent(img, txt)
    print("BBox shape:", bbox.shape)
    print("Value shape:", val.shape)
