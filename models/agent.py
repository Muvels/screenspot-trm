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
                 trm_layers: int = 1,
                 trm_depth: int = 3):
        super().__init__()
        
        self.encoder = VisionTextEncoder(model_name=vision_text_model, freeze_backbone=True)
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
        
    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        """
        Args:
            pixel_values: (B, C, H, W) or specific format
            input_ids: (B, L)
            attention_mask: (B, L)
            kwargs: e.g. image_grid_thw
            
        Returns:
            bbox: (B, 4)
            value: (B, 1)
            final_state: (B, Seq, H)
        """
        # 1. Encode
        context = self.encoder(pixel_values, input_ids, attention_mask, **kwargs) # (B, Seq, H)
        
        # 2. Recurse
        final_context_state = self.core(context) # (B, Seq, H)
        
        # 3. Aggregate for Heads
        # We need a single vector to represent the state.
        # Options:
        # - Max Pool over sequence
        # - Mean Pool over sequence
        # - Specific Token (e.g. first token if we had a CLS)
        # Since fusion is concatenation of [Img, Txt], doing Mean Pool or Max Pool over the whole sequence is reasonable.
        # Let's do Mean Pool for now.
        
        state_vec = final_context_state.mean(dim=1) # (B, H)
        
        # 4. Heads
        bbox = self.policy_head(state_vec)
        value = self.value_head(state_vec)
        
        return bbox, value, final_context_state

if __name__ == "__main__":
    # Test
    agent = InfoMaxAgent(trm_depth=2)
    B = 2
    img = torch.randn(B, 3, 224, 224)
    txt = torch.randint(0, 1000, (B, 10))
    bbox, val, _ = agent(img, txt)
    print("BBox shape:", bbox.shape)
    print("Value shape:", val.shape)
