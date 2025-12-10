import torch
import torch.nn as nn
import torch.nn.functional as F

class BBoxPolicyHead(nn.Module):
    """
    Predicts normalized bounding box [x1, y1, x2, y2].
    Can be deterministic (for now) or stochastic.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid() # Enforce [0, 1] range
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Hidden)
        Returns: (B, 4)
        """
        return self.net(x)

class ValueHead(nn.Module):
    """
    Estimates scalar value V(s).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Hidden)
        Returns: (B, 1)
        """
        return self.net(x)
