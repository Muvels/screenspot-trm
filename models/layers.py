"""Common neural network layers for TRM.

Adapted from ExampleTinyRecursiveModels/models/layers.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _find_multiple(a: int, b: int) -> int:
    """Find smallest multiple of b >= a."""
    return (-(a // -b)) * b


def rms_norm(hidden_states: Tensor, variance_epsilon: float = 1e-5) -> Tensor:
    """Root Mean Square Layer Normalization.
    
    Args:
        hidden_states: Input tensor
        variance_epsilon: Small constant for numerical stability
        
    Returns:
        Normalized tensor
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    
    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


def trunc_normal_init_(tensor: Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0) -> Tensor:
    """Truncated normal initialization (JAX-style).
    
    Args:
        tensor: Tensor to initialize
        std: Standard deviation
        lower: Lower bound (in units of std)
        upper: Upper bound (in units of std)
        
    Returns:
        Initialized tensor
    """
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor


class SwiGLU(nn.Module):
    """SwiGLU activation with linear layers.
    
    SwiGLU(x) = (xW_gate * SiLU(xW_up)) @ W_down
    
    Attributes:
        hidden_size: Input/output dimension
        expansion: Expansion factor for intermediate dimension
    """
    
    def __init__(self, hidden_size: int, expansion: float = 4.0):
        """Initialize SwiGLU layer.
        
        Args:
            hidden_size: Input and output dimension
            expansion: Expansion factor for intermediate dimension
        """
        super().__init__()
        
        # Compute intermediate size (rounded to multiple of 256 for efficiency)
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        
        # Combined gate and up projection
        self.gate_up_proj = nn.Linear(hidden_size, inter * 2, bias=False)
        self.down_proj = nn.Linear(inter, hidden_size, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using truncated normal."""
        in_features = self.gate_up_proj.in_features
        std = 1.0 / math.sqrt(in_features)
        trunc_normal_init_(self.gate_up_proj.weight, std=std)
        
        inter = self.down_proj.in_features
        std = 1.0 / math.sqrt(inter)
        trunc_normal_init_(self.down_proj.weight, std=std)
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply SwiGLU transformation.
        
        Args:
            x: Input tensor [..., hidden_size]
            
        Returns:
            Output tensor [..., hidden_size]
        """
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with learnable scale.
    
    Attributes:
        hidden_size: Dimension to normalize
        eps: Small constant for numerical stability
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        """Initialize RMSNorm.
        
        Args:
            hidden_size: Dimension to normalize
            eps: Numerical stability constant
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalization.
        
        Args:
            x: Input tensor [..., hidden_size]
            
        Returns:
            Normalized tensor [..., hidden_size]
        """
        return rms_norm(x, self.eps) * self.weight
