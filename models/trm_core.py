"""Tiny Recursive Model (TRM) core controller.

Adapted from ExampleTinyRecursiveModels for bounding box prediction.
Implements the TRM algorithm with full backprop through recursion.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .layers import SwiGLU, rms_norm, trunc_normal_init_


class TRMBlock(nn.Module):
    """Single TRM block: residual SwiGLU MLP with RMS normalization.
    
    Implements: output = RMSNorm(x + SwiGLU(x))
    
    Attributes:
        hidden_size: Dimension of hidden state
        expansion: Expansion factor for SwiGLU
    """
    
    def __init__(
        self,
        hidden_size: int = 256,
        expansion: float = 4.0,
        norm_eps: float = 1e-5,
    ):
        """Initialize TRM block.
        
        Args:
            hidden_size: Dimension of hidden state
            expansion: Expansion factor for SwiGLU intermediate dim
            norm_eps: Epsilon for RMS normalization
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.norm_eps = norm_eps
        
        self.mlp = SwiGLU(hidden_size, expansion)
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply TRM block.
        
        Args:
            x: Input tensor [B, hidden_size] or [B, L, hidden_size]
            
        Returns:
            Output tensor with same shape as input
        """
        # Residual + post-norm (as in reference)
        out = self.mlp(x)
        return rms_norm(x + out, self.norm_eps)


class TRMReasoningModule(nn.Module):
    """Stack of TRM blocks for one reasoning level.
    
    Applies input injection and processes through multiple TRM blocks.
    
    Attributes:
        num_layers: Number of TRM blocks
        hidden_size: Dimension of hidden state
    """
    
    def __init__(
        self,
        num_layers: int = 2,
        hidden_size: int = 256,
        expansion: float = 4.0,
        norm_eps: float = 1e-5,
    ):
        """Initialize reasoning module.
        
        Args:
            num_layers: Number of TRM blocks to stack
            hidden_size: Dimension of hidden state
            expansion: Expansion factor for SwiGLU
            norm_eps: Epsilon for RMS normalization
        """
        super().__init__()
        
        self.layers = nn.ModuleList([
            TRMBlock(hidden_size, expansion, norm_eps)
            for _ in range(num_layers)
        ])
    
    def forward(self, hidden: Tensor, injection: Tensor) -> Tensor:
        """Apply reasoning module.
        
        Args:
            hidden: Current hidden state [B, hidden_size]
            injection: Context to inject [B, hidden_size]
            
        Returns:
            Updated hidden state [B, hidden_size]
        """
        # Add injection to hidden state
        hidden = hidden + injection
        
        # Process through all layers
        for layer in self.layers:
            hidden = layer(hidden)
        
        return hidden


class TRMController(nn.Module):
    """TRM recursive controller following the paper algorithm.
    
    Maintains three state variables:
    - x: context embedding (h_ctx from fusion), constant across recursion
    - y: answer embedding, refined each step
    - z: latent reasoning state
    
    Recursion structure (per outer H_cycle):
        for _ in range(L_cycles):
            z = L_level(z, x + y)    # latent reasoning
        y = L_level(y, z)            # answer refinement
    
    Deep supervision:
    - Run H_cycles-1 outer cycles without gradients
    - Run final cycle with full gradient backprop through all inner loops
    - This is NOT truncated BPTT - we backprop through entire final cycle
    
    Attributes:
        hidden_size: Dimension of hidden state
        H_cycles: Number of outer recursion cycles (deep supervision)
        L_cycles: Number of inner recursion cycles (latent reasoning)
        L_layers: Number of layers in reasoning module
    """
    
    def __init__(
        self,
        hidden_size: int = 256,
        H_cycles: int = 3,
        L_cycles: int = 4,
        L_layers: int = 2,
        expansion: float = 4.0,
        norm_eps: float = 1e-5,
    ):
        """Initialize TRM controller.
        
        Args:
            hidden_size: Dimension of hidden state
            H_cycles: Number of outer deep supervision cycles
            L_cycles: Number of inner latent reasoning cycles
            L_layers: Number of layers per reasoning module
            expansion: Expansion factor for SwiGLU
            norm_eps: Epsilon for RMS normalization
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        
        # Shared reasoning module (used for both z and y updates)
        self.L_level = TRMReasoningModule(
            num_layers=L_layers,
            hidden_size=hidden_size,
            expansion=expansion,
            norm_eps=norm_eps,
        )
        
        # Learned initial states
        # Use small initialization as in reference (line 153-154)
        self.y_init = nn.Parameter(
            trunc_normal_init_(torch.empty(hidden_size), std=1.0)
        )
        self.z_init = nn.Parameter(
            trunc_normal_init_(torch.empty(hidden_size), std=1.0)
        )
    
    def forward(
        self,
        h_ctx: Tensor,
        return_intermediates: bool = False,
    ) -> Tuple[Tensor, Optional[List[Tensor]]]:
        """Run TRM recursion.
        
        Args:
            h_ctx: Fused context embedding [B, hidden_size]
            return_intermediates: Whether to return intermediate y values
            
        Returns:
            y_final: Final answer embedding [B, hidden_size]
            intermediates: Optional list of y at each H_cycle (including no-grad cycles)
        """
        B = h_ctx.shape[0]
        device = h_ctx.device
        dtype = h_ctx.dtype
        
        # Context (constant throughout recursion)
        x = h_ctx
        
        # Initialize y and z by expanding learned initial states
        y = self.y_init.to(dtype).unsqueeze(0).expand(B, -1).clone()
        z = self.z_init.to(dtype).unsqueeze(0).expand(B, -1).clone()
        
        intermediates = [] if return_intermediates else None
        
        # Deep supervision: H_cycles-1 without grad, 1 with grad
        # But FULL backprop through final cycle's inner loops
        
        if self.H_cycles > 1:
            with torch.no_grad():
                for _ in range(self.H_cycles - 1):
                    # Inner loop: latent reasoning
                    for _ in range(self.L_cycles):
                        z = self.L_level(z, x + y)
                    # Answer refinement
                    y = self.L_level(y, z)
                    
                    if return_intermediates:
                        intermediates.append(y.clone())
        
        # Final cycle WITH gradients (backprop through all L_cycles)
        # Re-enable gradients for y and z from no-grad phase
        y = y.detach().requires_grad_(True) if self.training else y
        z = z.detach().requires_grad_(True) if self.training else z
        
        for _ in range(self.L_cycles):
            z = self.L_level(z, x + y)
        y = self.L_level(y, z)
        
        if return_intermediates:
            intermediates.append(y)
        
        return y, intermediates
    
    def forward_all_grad(
        self,
        h_ctx: Tensor,
        return_intermediates: bool = False,
    ) -> Tuple[Tensor, Optional[List[Tensor]]]:
        """Run TRM recursion with gradients through ALL cycles.
        
        Use this for debugging or when you want full gradient flow.
        
        Args:
            h_ctx: Fused context embedding [B, hidden_size]
            return_intermediates: Whether to return intermediate y values
            
        Returns:
            y_final: Final answer embedding [B, hidden_size]
            intermediates: Optional list of y at each H_cycle
        """
        B = h_ctx.shape[0]
        dtype = h_ctx.dtype
        
        x = h_ctx
        y = self.y_init.to(dtype).unsqueeze(0).expand(B, -1).clone()
        z = self.z_init.to(dtype).unsqueeze(0).expand(B, -1).clone()
        
        intermediates = [] if return_intermediates else None
        
        # All cycles with gradients
        for _ in range(self.H_cycles):
            for _ in range(self.L_cycles):
                z = self.L_level(z, x + y)
            y = self.L_level(y, z)
            
            if return_intermediates:
                intermediates.append(y)
        
        return y, intermediates
