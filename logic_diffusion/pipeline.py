import torch
import numpy as np
from tqdm import tqdm
from .logic import DifferentiableLogic

class LogicGuidedPipeline:
    """
    Manages the diffusion sampling process and injects 
    Logical Guidance to steer generation.
    """
    def __init__(self, model, config, scheduler=None):
        self.model = model
        self.cfg = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # Simple Linear Noise Scheduler
        self.beta = torch.linspace(config.beta_start, config.beta_end, config.timesteps).to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def sample(self, n_samples, constraints=[]):
        """
        Generate samples with Logic Guidance.
        """
        self.model.eval()
        with torch.no_grad():
            # Start from random noise
            x = torch.randn((n_samples, self.cfg.channels, self.cfg.image_size, self.cfg.image_size)).to(self.device)
            
            for i in tqdm(reversed(range(1, self.cfg.timesteps)), position=0):
                t = (torch.ones(n_samples) * i).long().to(self.device)
                
                # 1. Predict noise
                predicted_noise = self.model(x, t)
                
                # 2. Logic Guidance Step (The Innovation)
                # If we have constraints and are in the guidance phase
                if constraints and i < self.cfg.logic_start_step:
                    # We must enable gradients momentarily for the logic loss
                    with torch.enable_grad():
                        x_in = x.detach().requires_grad_(True)
                        
                        # Calculate Logic Loss on the current noisy image
                        # (Note: In v1 we will project x_in to x_0 estimate first)
                        loss = 0
                        for constraint in constraints:
                            loss += constraint(x_in)
                        
                        # Calculate gradient of Logic Loss w.r.t pixels
                        grad = torch.autograd.grad(loss, x_in)[0]
                        
                        # Subtract gradient (Steer away from bias)
                        # This "pushes" the noise toward the fair manifold
                        predicted_noise = predicted_noise - (self.cfg.logic_weight * grad)

                # 3. Denoise (Standard DDPM Update)
                alpha = self.alpha[i]
                alpha_hat = self.alpha_hat[i]
                beta = self.beta[i]
                
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                
        return x.detach().cpu()