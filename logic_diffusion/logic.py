import torch
import torch.nn as nn

class DifferentiableLogic(nn.Module):
    """
    Implements a differentiable logic layer.
    Uses Product T-Norms to relax boolean logic into continuous gradients.
    """
    def __init__(self):
        super().__init__()

    def and_op(self, x, y):
        return x * y

    def or_op(self, x, y):
        return x + y - (x * y)

    def not_op(self, x):
        return 1.0 - x

    def implies_op(self, x, y):
        # x -> y  is equivalent to  not(x) or y
        return self.or_op(self.not_op(x), y)

class AxiomaticConstraint(nn.Module):
    """
    Defines the 'Logical Manifold' constraints.
    Example: 'Fairness' axiom -> If Feature A is present, Feature B must be balanced.
    """
    def __init__(self):
        super().__init__()
        self.logic = DifferentiableLogic()

    def forward(self, x):
        """
        Input: x (The generated tensor during diffusion)
        Output: logic_loss (Scalar tensor, 0.0 means logic is perfectly satisfied)
        """
        # --- EXAMPLE AXIOM: SYMMETRY & BOUNDS ---
        # "The output must be symmetric and bounded between -1 and 1"
        
        # 1. Constraint: Values must be within [-1, 1] (Soft Logic)
        # We penalize values outside this range.
        lower_bound = torch.relu(-1.0 - x)
        upper_bound = torch.relu(x - 1.0)
        bound_violation = torch.mean(lower_bound + upper_bound)

        # 2. Constraint: Vertical Symmetry (Example of a 'structural' logic)
        # Left half of tensor roughly equals flipped right half
        # Assuming x is (B, C, H, W)
        if x.dim() == 4:
            x_flipped = torch.flip(x, dims=[3]) 
            symmetry_violation = torch.mean((x - x_flipped) ** 2)
        else:
            symmetry_violation = 0.0

        # Total Logic Loss (minimize this to steer towards manifold)
        total_violation = bound_violation + symmetry_violation
        
        return total_violation
