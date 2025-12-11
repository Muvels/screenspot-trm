import torch
import torch.nn as nn
from typing import Optional, Tuple

class RecursiveBlock(nn.Module):
    """
    A single block applied recursively. 
    Standard Transformer Decoder Layer structure (Self-Attn + Cross-Attn + MLP)
    or just Self-Attn + MLP if we treat context as part of the sequence.
    
    Here we implement a standard Pre-Norm Transformer Block.
    """
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        x: (B, SeqLen, Hidden)
        mask: Optional mask
        """
        residual = x
        x = self.norm1(x)
        x2, _ = self.self_attn(x, x, x, attn_mask=mask) # Self-attention
        x = residual + x2
        
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        return x

class TinyRecursiveModel(nn.Module):
    """
    Recursive Reasoning Core.
    
    Args:
        hidden_size: Dimension of the state (must match encoder output).
        num_layers: Number of LAYERS within one recursive Step (usually 1 for TRM).
        num_heads: Number of attention heads.
        recursion_depth: Default number of recursion steps T.
    """
    def __init__(self, hidden_size: int, num_heads: int, num_layers: int = 2, recursion_depth: int = 6):
        super().__init__()
        self.hidden_size = hidden_size
        self.recursion_depth = recursion_depth
        
        # The reusable block
        self.blocks = nn.ModuleList([
            RecursiveBlock(hidden_size, num_heads) for _ in range(num_layers)
        ])
        
        # State Initialization (Learnable query or similar)
        # In TRM, 'z' is often initialized from context 'x' or zero.
        # We'll initialize z_0 = x (context).
        
    def forward(self, context: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            context: (B, SeqLen, Hidden)
            steps: Override recursion depth
        
        Returns:
            final_state: (B, SeqLen, Hidden)
        """
        if steps is None:
            steps = self.recursion_depth
            
        z = context # Initialize state with context
        
        # Recursion Loop with FULL Gradient Flow
        # We collect all states for Deep Supervision
        all_states = []
        
        for t in range(steps):
            # Apply blocks
            for block in self.blocks:
                z = block(z)
            all_states.append(z)
                
        # Stack: (B, Steps, SeqLen, Hidden)
        return torch.stack(all_states, dim=1)

if __name__ == "__main__":
    # Test
    model = TinyRecursiveModel(hidden_size=512, num_heads=8, recursion_depth=6)
    dummy_input = torch.randn(2, 50, 512)
    output = model(dummy_input)
    print("Output shape:", output.shape)
    
    # Gradient Check
    loss = output.sum()
    loss.backward()
    print("Gradient check passed (no error).")
