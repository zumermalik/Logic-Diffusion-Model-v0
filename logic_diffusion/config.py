from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """
    Configuration for training Logic Diffusion v0.
    """
    # Image/Data Params
    image_size: int = 32      # Small size for v0 (faster training on Colab)
    channels: int = 1         # 1 for grayscale/tabular heatmaps, 3 for RGB
    batch_size: int = 64

    # Optimization Params
    num_epochs: int = 50
    learning_rate: float = 1e-4
    grad_accumulation_steps: int = 1

    # Diffusion Params
    timesteps: int = 1000     # Standard DDPM steps
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # Logic Params (The Special Sauce)
    logic_weight: float = 0.5 # How much we punish bias (Lambda)
    logic_start_step: int = 500 # Only apply logic in the noisy half or clean half?
