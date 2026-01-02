import torch
import torch.nn as nn
from typing import Callable

class DifferentiableLogic:
    """
    Library of differentiable fuzzy logic operators (T-Norms).
    Allows logical rules to be used as loss functions.
    """
    
    @staticmethod
    def not_op(x: torch.Tensor) -> torch.Tensor:
        """Standard Negation: 1 - x"""
        return 1.0 - x

    @staticmethod
    def and_op(x: torch.Tensor, y: torch.Tensor, method='product') -> torch.Tensor:
        """
        Logical AND (Conjunction).
        - 'product': x * y (Best gradients)
        - 'godel': min(x, y)
        """
        if method == 'product':
            return x * y
        elif method == 'godel':
            return torch.minimum(x, y)
        else:
            raise ValueError(f"Unknown logic method: {method}")

    @staticmethod
    def or_op(x: torch.Tensor, y: torch.Tensor, method='product') -> torch.Tensor:
        """
        Logical OR (Disjunction).
        Derived via De Morgan's: NOT (NOT x AND NOT y)
        """
        if method == 'product':
            # x + y - xy
            return x + y - (x * y)
        elif method == 'godel':
            return torch.maximum(x, y)
        else:
            raise ValueError(f"Unknown logic method: {method}")

    @staticmethod
    def implies(x: torch.Tensor, y: torch.Tensor, method='product') -> torch.Tensor:
        """
        Logical Implication (x => y).
        CRITICAL for fairness: "If Feature A (x), Then Feature B (y)"
        """
        if method == 'product':
            # Reichenbach Implication: 1 - x + xy
            return 1.0 - x + (x * y)
        elif method == 'godel':
             # 1 if x <= y else y
            return torch.where(x <= y, torch.ones_like(x), y)
        else:
            raise ValueError(f"Unknown logic method: {method}")

class LogicConstraint(nn.Module):
    """
    A trainable module that calculates how much a batch of data violates a specific rule.
    """
    def __init__(self, rule_function: Callable, weight: float = 1.0):
        super().__init__()
        self.rule_function = rule_function
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the weighted loss (violation cost).
        Input x: The batch of generated data (or latent features).
        """
        # 1. Calculate Truth Value of the rule (0 to 1)
        truth_value = self.rule_function(x)
        
        # 2. Convert Truth to Loss (Minimize Loss = Maximize Truth)
        # Loss = 1 - Truth
        violation_loss = torch.mean(1.0 - truth_value)
        
        return self.weight * violation_loss