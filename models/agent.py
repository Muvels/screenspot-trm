"""
InfoMax Agent for UI Grounding

Enhanced agent architecture combining:
- VisionTextEncoder: CLIP/SigLIP based multimodal encoding
- TinyRecursiveModel: Dual-state recursive reasoning
- Policy/Value Heads: BBox prediction and value estimation

Supports:
- Deep supervision (outputs from all recursion steps)
- Optional adaptive halting (ACT)
- Perspective-biased initialization (UI-Ins inspired)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, NamedTuple
from dataclasses import dataclass

from models.encodings import VisionTextEncoder
from models.trm import TinyRecursiveModel, HaltingInfo
from models.policy import BBoxPolicyHead, ValueHead


@dataclass
class AgentOutput:
    """Structured output from InfoMaxAgent."""
    bbox: torch.Tensor           # (B, Steps, 4) - Predicted bounding boxes
    value: torch.Tensor          # (B, Steps, 1) - Value estimates
    final_state: torch.Tensor    # (B, Steps, Seq, H) - All TRM states
    halting_info: Optional[HaltingInfo] = None  # Halting information if ACT enabled


class InfoMaxAgent(nn.Module):
    """
    Full Agent Model for UI Grounding.
    
    Architecture:
    1. VisionTextEncoder -> Multimodal context embeddings
    2. TinyRecursiveModel -> Dual-state recursive reasoning
    3. BBoxPolicyHead -> Bounding box prediction
    4. ValueHead -> State value estimation
    
    Args:
        vision_text_model: CLIP/SigLIP model name
        hidden_size: Hidden dimension for TRM
        trm_layers: Number of layers in L_level
        H_cycles: Number of high-level reasoning cycles
        L_cycles: Number of low-level refinement cycles
        use_act: Enable adaptive halting
        max_steps: Maximum steps if ACT enabled
        freeze_backbone: Freeze vision/text encoder
    """
    def __init__(
        self, 
        vision_text_model: str = "openai/clip-vit-base-patch32", 
        hidden_size: int = 512, 
        trm_layers: int = 2,
        H_cycles: int = 3,
        L_cycles: int = 6,
        use_act: bool = False,
        max_steps: int = 10,
        freeze_backbone: bool = True
    ):
        super().__init__()
        
        # Vision + Text Encoder
        self.encoder = VisionTextEncoder(
            model_name=vision_text_model, 
            freeze_backbone=freeze_backbone
        )
        
        # Dimension alignment
        encoder_dim = self.encoder.out_dim
        self.needs_projection = encoder_dim != hidden_size
        
        if self.needs_projection:
            self.input_proj = nn.Linear(encoder_dim, hidden_size)
        
        # Enhanced Recursive Core
        self.core = TinyRecursiveModel(
            hidden_size=hidden_size, 
            num_heads=8, 
            L_layers=trm_layers,
            H_cycles=H_cycles, 
            L_cycles=L_cycles,
            use_act=use_act,
            max_steps=max_steps
        )
        
        # Store for tracking
        self.use_act = use_act
        
        # Output Heads
        self.policy_head = BBoxPolicyHead(input_dim=hidden_size)
        self.value_head = ValueHead(input_dim=hidden_size)
        
    def forward(
        self, 
        pixel_values: torch.Tensor, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        return_structured: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[HaltingInfo]]:
        """
        Forward pass through the agent.
        
        Args:
            pixel_values: (B, C, H, W) - Input images
            input_ids: (B, L) - Tokenized instructions
            attention_mask: (B, L) - Attention mask for text
            return_structured: If True, return AgentOutput dataclass
            
        Returns:
            bbox: (B, Steps, 4) - Predicted bounding boxes for each step
            value: (B, Steps, 1) - Value estimates for each step
            final_state: (B, Steps, Seq, H) - All TRM hidden states
            halting_info: Optional halting information
        """
        # =========================================================
        # 1. Encode Vision + Text
        # =========================================================
        # Get separate embeddings for perspective-biased init
        vision_embeds, text_embeds = self._encode_separate(
            pixel_values, input_ids, attention_mask
        )
        
        # Combined context
        context = torch.cat([vision_embeds, text_embeds], dim=1)
        
        # Project if needed
        if self.needs_projection:
            context = self.input_proj(context)
            vision_embeds = self.input_proj(vision_embeds)
            text_embeds = self.input_proj(text_embeds)
        
        # =========================================================
        # 2. Recursive Reasoning with Dual States
        # =========================================================
        trm_outputs, halting_info = self.core(
            context,
            vision_context=vision_embeds,
            text_context=text_embeds
        )
        # trm_outputs: (B, Steps, Seq, H)
        
        # =========================================================
        # 3. Apply Output Heads to Each Step
        # =========================================================
        B, Steps, Seq, H = trm_outputs.shape
        
        # Flatten for efficient head computation
        trm_flat = trm_outputs.view(B * Steps, Seq, H)
        
        # Mean pooling over sequence
        state_vec_flat = trm_flat.mean(dim=1)  # (B * Steps, H)
        
        # Compute outputs
        bbox_flat = self.policy_head(state_vec_flat)  # (B * Steps, 4)
        value_flat = self.value_head(state_vec_flat)  # (B * Steps, 1)
        
        # Reshape back
        bbox = bbox_flat.view(B, Steps, 4)
        value = value_flat.view(B, Steps, 1)
        
        if return_structured:
            return AgentOutput(
                bbox=bbox,
                value=value,
                final_state=trm_outputs,
                halting_info=halting_info
            )
        
        return bbox, value, trm_outputs, halting_info
    
    def _encode_separate(
        self, 
        pixel_values: torch.Tensor, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode vision and text separately for perspective-biased initialization.
        
        Returns:
            vision_embeds: (B, N_v, H) - Vision patch embeddings
            text_embeds: (B, N_t, H) - Text token embeddings
        """
        # Vision forward
        inputs = {"pixel_values": pixel_values}
        vision_outputs = self.encoder.backbone.vision_model(**inputs)
        vision_embeds = vision_outputs.last_hidden_state  # (B, N_v, D_vis)
        
        # Text forward
        text_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        text_outputs = self.encoder.backbone.text_model(**text_inputs)
        text_embeds = text_outputs.last_hidden_state  # (B, N_t, D_txt)
        
        # Project to shared space
        vision_embeds = self.encoder.vis_proj(vision_embeds)
        text_embeds = self.encoder.txt_proj(text_embeds)
        
        return vision_embeds, text_embeds
    
    def get_final_prediction(
        self, 
        pixel_values: torch.Tensor, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get only the final bounding box prediction (for inference).
        
        Returns:
            bbox: (B, 4) - Final predicted bounding boxes
        """
        bbox, _, _, halting_info = self.forward(pixel_values, input_ids, attention_mask)
        
        if halting_info is not None:
            # Use the step at which each sample halted
            B = bbox.shape[0]
            final_bbox = torch.zeros(B, 4, device=bbox.device)
            for i in range(B):
                step_idx = halting_info.steps_taken[i] - 1  # 0-indexed
                final_bbox[i] = bbox[i, step_idx]
            return final_bbox
        else:
            # Use the last step
            return bbox[:, -1, :]
    
    def predict_with_confidence(
        self, 
        pixel_values: torch.Tensor, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get bounding box prediction WITH confidence score (for inference).
        
        The Value Head is trained to predict uncertainty (1 - IoU) during RL,
        so confidence = 1 - value output.
        
        Returns:
            bbox: (B, 4) - Final predicted bounding boxes
            confidence: (B,) - Confidence scores (0-1, higher = more confident)
        """
        bbox, value, _, halting_info = self.forward(pixel_values, input_ids, attention_mask)
        
        if halting_info is not None:
            # Use the step at which each sample halted
            B = bbox.shape[0]
            final_bbox = torch.zeros(B, 4, device=bbox.device)
            final_value = torch.zeros(B, device=bbox.device)
            for i in range(B):
                step_idx = halting_info.steps_taken[i] - 1  # 0-indexed
                final_bbox[i] = bbox[i, step_idx]
                final_value[i] = value[i, step_idx, 0]
            
            # Convert uncertainty to confidence
            confidence = torch.clamp(1.0 - final_value, min=0.0, max=1.0)
            return final_bbox, confidence
        else:
            # Use the last step
            final_value = value[:, -1, 0]  # (B,)
            confidence = torch.clamp(1.0 - final_value, min=0.0, max=1.0)
            return bbox[:, -1, :], confidence


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing InfoMaxAgent...")
    
    # Test basic forward
    agent = InfoMaxAgent(
        H_cycles=3, 
        L_cycles=4,
        trm_layers=2,
        use_act=False
    )
    
    B = 2
    img = torch.randn(B, 3, 224, 224)
    txt = torch.randint(0, 1000, (B, 10))
    mask = torch.ones(B, 10)
    
    bbox, val, states, halt_info = agent(img, txt, mask)
    
    print(f"BBox shape: {bbox.shape}")      # Expected: (2, 3, 4)
    print(f"Value shape: {val.shape}")      # Expected: (2, 3, 1)
    print(f"States shape: {states.shape}")  # Expected: (2, 3, Seq, 512)
    print(f"Halting info: {halt_info}")     # Expected: None
    
    # Test final prediction
    final_bbox = agent.get_final_prediction(img, txt, mask)
    print(f"Final BBox shape: {final_bbox.shape}")  # Expected: (2, 4)
    
    # Test with ACT
    agent_act = InfoMaxAgent(
        H_cycles=5, 
        L_cycles=4,
        use_act=True,
        max_steps=5
    )
    
    bbox_act, val_act, states_act, halt_info_act = agent_act(img, txt, mask)
    print(f"\nWith ACT:")
    print(f"BBox shape: {bbox_act.shape}")
    print(f"Halt logits shape: {halt_info_act.halt_logits.shape}")
    print(f"Steps taken: {halt_info_act.steps_taken}")
    
    # Parameter count
    num_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {num_params:,}")
