"""Exponential Moving Average (EMA) helper for model parameters.

Adapted from ExampleTinyRecursiveModels/models/ema.py
"""

import copy
from typing import Dict

import torch.nn as nn
from torch import Tensor


class EMAHelper:
    """Exponential Moving Average helper for model parameters.
    
    Maintains shadow copies of model parameters that are updated
    with exponential moving average during training. Useful for
    stabilizing training and improving generalization.
    
    Usage:
        ema = EMAHelper(mu=0.999)
        ema.register(model)
        
        # During training
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            ema.update(model)  # Update EMA after each step
        
        # For evaluation
        eval_model = ema.ema_copy(model)
        eval_model.eval()
    
    Attributes:
        mu: EMA decay rate (higher = slower updates)
        shadow: Dictionary of shadow parameter copies
    """
    
    def __init__(self, mu: float = 0.999):
        """Initialize EMA helper.
        
        Args:
            mu: EMA decay rate (0 to 1, higher means slower updates)
        """
        self.mu = mu
        self.shadow: Dict[str, Tensor] = {}
    
    def register(self, module: nn.Module) -> None:
        """Register a module's parameters for EMA tracking.
        
        Args:
            module: PyTorch module to track
        """
        if isinstance(module, nn.DataParallel):
            module = module.module
        
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, module: nn.Module) -> None:
        """Update shadow parameters with current model parameters.
        
        Args:
            module: Module with updated parameters
        """
        if isinstance(module, nn.DataParallel):
            module = module.module
        
        for name, param in module.named_parameters():
            if param.requires_grad and name in self.shadow:
                # EMA update: shadow = mu * shadow + (1 - mu) * param
                self.shadow[name].data = (
                    self.mu * self.shadow[name].data +
                    (1.0 - self.mu) * param.data
                )
    
    def ema(self, module: nn.Module) -> None:
        """Copy shadow parameters to module (in-place).
        
        Args:
            module: Module to update
        """
        if isinstance(module, nn.DataParallel):
            module = module.module
        
        for name, param in module.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name].data)
    
    def ema_copy(self, module: nn.Module) -> nn.Module:
        """Create a copy of module with EMA parameters.
        
        Args:
            module: Module to copy
            
        Returns:
            Deep copy of module with EMA parameters
        """
        module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy
    
    def state_dict(self) -> Dict[str, Tensor]:
        """Get EMA state for saving.
        
        Returns:
            Dictionary of shadow parameters
        """
        return self.shadow.copy()
    
    def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
        """Load EMA state.
        
        Args:
            state_dict: Dictionary of shadow parameters
        """
        self.shadow = state_dict.copy()
