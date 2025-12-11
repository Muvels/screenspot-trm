import torch
import torch.nn as nn
from typing import Optional

class StandardTransformer(nn.Module):
    """
    Standard Transformer Encoder Architecture.
    
    Args:
        hidden_size: Dimension of the state.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        dropout: Dropout probability.
    """
    def __init__(self, hidden_size: int, num_heads: int, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Standard PyTorch Transformer Encoder
        # batch_first=True makes it work with (B, Seq, H)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads, 
            dim_feedforward=hidden_size * 4, 
            dropout=dropout,
            batch_first=True,
            norm_first=True # Pre-Norm is generally more stable, matching TRM implementation which was also Pre-Norm
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, context: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            context: (B, SeqLen, Hidden)
            mask: Optional mask (B, SeqLen) or (B, SeqLen, SeqLen) - usually padding mask
        
        Returns:
            final_state: (B, SeqLen, Hidden)
        """
        # nn.TransformerEncoder takes mask as src_key_padding_mask if (B, L)
        # or mask if (L, L) or (B*H, L, L).
        # Assuming typical padding mask (B, L) where True is masked (ignored).
        # Check how TRM handles it. TRM had 'mask' in forward but didn't seem to propagate it fully in agent.py
        
        # In agent.py call: self.core(context)
        # It doesn't pass mask.
        
        return self.encoder(context, src_key_padding_mask=mask)

if __name__ == "__main__":
    # Test
    model = StandardTransformer(hidden_size=512, num_heads=8, num_layers=6)
    dummy_input = torch.randn(2, 50, 512)
    output = model(dummy_input)
    print("Output shape:", output.shape)
