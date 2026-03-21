import torch

class Diffusion:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Linear variance schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def add_noise(self, x_0, t):
        """Forward process: q(x_t | x_0)"""
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod[t])[:, None, None, None]
        
        noise = torch.randn_like(x_0).to(self.device)
        x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
        
        return x_t, noise

    def sample_timesteps(self, batch_size):
        return torch.randint(low=1, high=self.num_timesteps, size=(batch_size,)).to(self.device)