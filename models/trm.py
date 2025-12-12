"""
Enhanced Tiny Recursive Model (TRM) for UI Grounding

Combines insights from:
- Samsung TRM: Dual-state hierarchy (z_H, z_L), input injection, nested loops
- UI-Ins Paper: Multi-perspective reasoning (appearance, functionality, location, intent)

Architecture:
- z_H: High-level semantic state (intent, functionality understanding)
- z_L: Low-level spatial state (appearance, location reasoning)
- Nested H×L loop structure with input injection at every L-cycle
- Optional adaptive halting (ACT) using Q-learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, NamedTuple
from dataclasses import dataclass


# =============================================================================
# Utility Functions
# =============================================================================

def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float = 1e-6) -> torch.Tensor:
    """RMSNorm without learnable parameters (used inline)."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


def _find_multiple(a: int, b: int) -> int:
    """Find next multiple of b >= a."""
    return (-(a // -b)) * b


# =============================================================================
# Building Blocks
# =============================================================================

class SwiGLU(nn.Module):
    """
    SwiGLU activation: gate * swish(input)
    More expressive than GELU for recursive reasoning.
    """
    def __init__(self, hidden_size: int, expansion: float = 2.67):
        super().__init__()
        # Expand to ~2.67x then project back (standard for SwiGLU)
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 64)
        
        self.gate_up_proj = nn.Linear(hidden_size, inter * 2, bias=False)
        self.down_proj = nn.Linear(inter, hidden_size, bias=False)
        
        # Initialize with smaller weights for stability
        nn.init.xavier_uniform_(self.gate_up_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.down_proj.weight, gain=0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class RecursiveBlock(nn.Module):
    """
    Enhanced recursive block with:
    - Post-norm RMSNorm (more stable for deep recursion)
    - SwiGLU MLP (more expressive)
    - Optional cross-attention for biased reasoning
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int, 
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.rms_norm_eps = rms_norm_eps
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # SwiGLU MLP
        self.mlp = SwiGLU(hidden_size)
        
    def forward(
        self, 
        x: torch.Tensor, 
        injection: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, SeqLen, Hidden) - Current state
            injection: (B, SeqLen, Hidden) - Optional input to inject (added before processing)
            mask: Optional attention mask
        """
        # Input injection (key TRM feature)
        if injection is not None:
            x = x + injection
        
        # Self-attention with post-norm (Samsung TRM style)
        residual = x
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = rms_norm(residual + attn_out, self.rms_norm_eps)
        
        # MLP with post-norm
        residual = x
        mlp_out = self.mlp(x)
        x = rms_norm(residual + mlp_out, self.rms_norm_eps)
        
        return x


class ReasoningLevel(nn.Module):
    """
    A reasoning level consisting of multiple stacked blocks.
    Corresponds to L_level in Samsung TRM.
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int, 
        num_layers: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            RecursiveBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        injection: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Current state
            injection: Input to inject (only applied to first layer)
        """
        for i, layer in enumerate(self.layers):
            # Only inject at first layer
            inj = injection if i == 0 else None
            hidden_states = layer(hidden_states, injection=inj)
        return hidden_states


# =============================================================================
# Adaptive Computation Time (ACT) Components
# =============================================================================

@dataclass
class HaltingInfo:
    """Information about halting decisions during forward pass."""
    halt_logits: torch.Tensor  # (B,) - Raw halt logits
    halted: torch.Tensor       # (B,) - Boolean mask of halted samples
    steps_taken: torch.Tensor  # (B,) - Number of steps taken per sample
    

class QHaltHead(nn.Module):
    """
    Q-learning based halting head.
    Learns when to stop recursion based on state quality.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.q_head = nn.Linear(hidden_size, 1, bias=True)
        
        # Initialize to discourage early halting
        nn.init.zeros_(self.q_head.weight)
        nn.init.constant_(self.q_head.bias, -3.0)  # Sigmoid(-3) ≈ 0.05
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, SeqLen, Hidden)
        Returns:
            halt_logits: (B,) - Positive = should halt
        """
        # Use mean-pooled state for halt decision
        pooled = state.mean(dim=1)  # (B, Hidden)
        return self.q_head(pooled).squeeze(-1)  # (B,)


# =============================================================================
# Main TRM Model
# =============================================================================

class TinyRecursiveModel(nn.Module):
    """
    Enhanced Tiny Recursive Model for UI Grounding.
    
    Key Features:
    1. Dual-State: z_H (semantic/intent) and z_L (spatial/appearance)
    2. Input Injection: Context injected at every L-cycle
    3. Nested Loops: H_cycles × L_cycles structure
    4. Deep Supervision: Returns states from all H-cycles
    5. Optional Adaptive Halting: Q-learning based early stopping
    
    Args:
        hidden_size: Dimension of hidden states
        num_heads: Number of attention heads
        L_layers: Number of layers in L_level network
        H_cycles: Number of high-level reasoning cycles (outer loop)
        L_cycles: Number of low-level refinement cycles (inner loop)
        use_act: Whether to use Adaptive Computation Time (halting)
        max_steps: Maximum steps if ACT is enabled
    """
    def __init__(
        self, 
        hidden_size: int = 512, 
        num_heads: int = 8, 
        L_layers: int = 2,
        H_cycles: int = 3, 
        L_cycles: int = 6,
        use_act: bool = False,
        max_steps: int = 10,
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.use_act = use_act
        self.max_steps = max_steps
        
        # Shared L-level network (parameter efficient)
        self.L_level = ReasoningLevel(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=L_layers,
            dropout=dropout
        )
        
        # =========================================================
        # Learnable Initial States (UI-Ins Inspired)
        # =========================================================
        # z_H: Semantic/Intent focused initialization
        # - Biased towards understanding "what" and "why"
        self.H_init = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.H_bias_proj = nn.Linear(hidden_size, hidden_size)
        
        # z_L: Spatial/Appearance focused initialization
        # - Biased towards understanding "where" and "how it looks"
        self.L_init = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.L_bias_proj = nn.Linear(hidden_size, hidden_size)
        
        # Initialize projections for different perspectives
        nn.init.orthogonal_(self.H_bias_proj.weight)
        nn.init.orthogonal_(self.L_bias_proj.weight)
        
        # Small random init for base states
        nn.init.normal_(self.H_init, std=0.02)
        nn.init.normal_(self.L_init, std=0.02)
        
        # =========================================================
        # Optional Adaptive Halting (ACT)
        # =========================================================
        if use_act:
            self.q_halt = QHaltHead(hidden_size)
            
    @property
    def recursion_depth(self) -> int:
        """Total number of reasoning steps (for compatibility)."""
        return self.H_cycles
        
    def _initialize_states(
        self, 
        context: torch.Tensor,
        vision_context: Optional[torch.Tensor] = None,
        text_context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize z_H and z_L with perspective-biased context.
        
        If vision_context and text_context are provided separately,
        we bias z_H towards text (semantic) and z_L towards vision (spatial)
        by pooling each modality and broadcasting the bias.
        
        Otherwise, we use the combined context for both.
        """
        B, Seq, H = context.shape
        
        # Expand learnable inits to batch size
        z_H = self.H_init.expand(B, Seq, -1).clone()
        z_L = self.L_init.expand(B, Seq, -1).clone()
        
        if vision_context is not None and text_context is not None:
            # Perspective-biased initialization (UI-Ins inspired)
            # Pool each modality to get a (B, H) bias vector, then broadcast
            
            # z_H gets text bias (semantic understanding)
            # Mean pool text context: (B, N_t, H) -> (B, H) -> (B, 1, H)
            text_bias = self.H_bias_proj(text_context.mean(dim=1, keepdim=True))
            z_H = z_H + text_bias  # Broadcasts to (B, Seq, H)
            
            # z_L gets vision bias (spatial understanding)
            # Mean pool vision context: (B, N_v, H) -> (B, H) -> (B, 1, H)
            vision_bias = self.L_bias_proj(vision_context.mean(dim=1, keepdim=True))
            z_L = z_L + vision_bias  # Broadcasts to (B, Seq, H)
        else:
            # Fall back to using combined context for both
            # This maintains per-position information
            z_H = z_H + self.H_bias_proj(context)
            z_L = z_L + self.L_bias_proj(context)
            
        return z_H, z_L
        
    def forward(
        self, 
        context: torch.Tensor,
        vision_context: Optional[torch.Tensor] = None,
        text_context: Optional[torch.Tensor] = None,
        steps: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[HaltingInfo]]:
        """
        Forward pass with dual-state recursive reasoning.
        
        Args:
            context: (B, SeqLen, Hidden) - Combined vision + text embeddings
            vision_context: Optional (B, Seq_v, Hidden) - Vision embeddings only
            text_context: Optional (B, Seq_t, Hidden) - Text embeddings only
            steps: Override number of H-cycles
            
        Returns:
            all_states: (B, Steps, SeqLen, Hidden) - States from each H-cycle
            halting_info: Optional halting information if ACT is enabled
        """
        B = context.shape[0]
        H_cycles = steps if steps is not None else self.H_cycles
        
        # Initialize dual states with perspective bias
        z_H, z_L = self._initialize_states(context, vision_context, text_context)
        
        all_states = []
        halt_logits_all = []
        
        # Track halting state for ACT
        if self.use_act:
            halted = torch.zeros(B, dtype=torch.bool, device=context.device)
            steps_taken = torch.zeros(B, dtype=torch.long, device=context.device)
        
        # =========================================================
        # Main Recursive Loop
        # =========================================================
        for h in range(H_cycles):
            
            # ---------------------------------------------------------
            # L-cycles: Refine z_L using z_H guidance + input injection
            # ---------------------------------------------------------
            for l in range(self.L_cycles):
                # KEY: Input injection at every L-cycle
                # z_L is refined using both high-level guidance (z_H) and 
                # grounding to the original input (context)
                injection = z_H + context
                z_L = self.L_level(z_L, injection=injection)
            
            # ---------------------------------------------------------
            # H-cycle: Update z_H using refined z_L
            # ---------------------------------------------------------
            # z_H incorporates the spatial/appearance findings from z_L
            z_H = self.L_level(z_H, injection=z_L)
            
            # Collect state for deep supervision
            all_states.append(z_H)
            
            # ---------------------------------------------------------
            # Adaptive Halting (if enabled)
            # ---------------------------------------------------------
            if self.use_act:
                halt_logits = self.q_halt(z_H)
                halt_logits_all.append(halt_logits)
                
                # Update halting state
                should_halt = (halt_logits > 0) | (h >= self.max_steps - 1)
                newly_halted = should_halt & ~halted
                steps_taken = torch.where(newly_halted, h + 1, steps_taken)
                halted = halted | should_halt
                
                # Early exit if all samples halted (inference time)
                if not self.training and halted.all():
                    break
        
        # Stack all states: (B, Steps, Seq, Hidden)
        all_states = torch.stack(all_states, dim=1)
        
        # Prepare halting info
        halting_info = None
        if self.use_act:
            # For samples that never halted, set steps to max
            steps_taken = torch.where(
                steps_taken == 0, 
                torch.tensor(len(halt_logits_all), device=context.device),
                steps_taken
            )
            halting_info = HaltingInfo(
                halt_logits=torch.stack(halt_logits_all, dim=1) if halt_logits_all else torch.zeros(B, 1, device=context.device),
                halted=halted,
                steps_taken=steps_taken
            )
        
        return all_states, halting_info


# =============================================================================
# Legacy-compatible wrapper
# =============================================================================

class TinyRecursiveModelLegacy(TinyRecursiveModel):
    """
    Backward-compatible wrapper that matches the old API.
    Returns only the states tensor (ignoring halting info).
    """
    def forward(
        self, 
        context: torch.Tensor, 
        steps: Optional[int] = None
    ) -> torch.Tensor:
        all_states, _ = super().forward(context, steps=steps)
        return all_states


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Enhanced TRM...")
    
    # Test basic forward
    model = TinyRecursiveModel(
        hidden_size=512, 
        num_heads=8, 
        H_cycles=3, 
        L_cycles=6,
        use_act=False
    )
    
    B, Seq, H = 2, 50, 512
    dummy_context = torch.randn(B, Seq, H)
    
    output, halt_info = model(dummy_context)
    print(f"Output shape: {output.shape}")  # Expected: (2, 3, 50, 512)
    print(f"Halting info: {halt_info}")     # Expected: None
    
    # Test with ACT
    model_act = TinyRecursiveModel(
        hidden_size=512,
        num_heads=8,
        H_cycles=5,
        L_cycles=4,
        use_act=True,
        max_steps=5
    )
    
    output_act, halt_info_act = model_act(dummy_context)
    print(f"\nWith ACT:")
    print(f"Output shape: {output_act.shape}")
    print(f"Halt logits shape: {halt_info_act.halt_logits.shape}")
    print(f"Steps taken: {halt_info_act.steps_taken}")
    
    # Test with separate vision/text context
    vision_ctx = torch.randn(B, 30, H)  # 30 vision tokens
    text_ctx = torch.randn(B, 20, H)    # 20 text tokens
    combined_ctx = torch.cat([vision_ctx, text_ctx], dim=1)  # 50 total
    
    output_biased, _ = model(combined_ctx, vision_context=vision_ctx, text_context=text_ctx)
    print(f"\nWith biased init:")
    print(f"Output shape: {output_biased.shape}")
    
    # Gradient check
    loss = output.sum()
    loss.backward()
    print("\n✅ Gradient check passed!")
    
    # Parameter count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {num_params:,}")
    print(f"  (~{num_params / 1e6:.2f}M)")
