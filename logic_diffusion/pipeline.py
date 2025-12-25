import torch
from diffusers import DDPMPipeline
from tqdm import tqdm

class LogicGuidedPipeline(DDPMPipeline):
    def __init__(self, unet, scheduler, logic_module):
        super().__init__(unet, scheduler)
        self.logic_module = logic_module

    @torch.no_grad()
    def __call__(
        self, 
        batch_size=1, 
        generator=None, 
        num_inference_steps=1000, 
        logic_guidance_scale=10.0  # How hard we force the logic
    ):
        # 1. Sample initial noise
        image_shape = (batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size)
        image = torch.randn(image_shape, generator=generator, device=self.device)

        # 2. Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # 3. Denoising Loop
        for t in tqdm(self.scheduler.timesteps):
            # A. Predict noise residual (Standard Diffusion)
            with torch.no_grad():
                model_output = self.unet(image, t).sample

            # B. --- THE LOGIC STEP ---
            # We need gradients for the input 'image' to steer it.
            # We must enable grad temporarily even though we are in no_grad mode
            with torch.enable_grad():
                x_in = image.detach().requires_grad_(True)
                
                # Calculate how much we violate the logic
                logic_loss = self.logic_module(x_in)
                
                # Calculate gradient: "Which direction moves us closer to the Logical Manifold?"
                logic_grad = torch.autograd.grad(logic_loss, x_in)[0]

            # C. Apply Guidance
            # We modify the estimated noise to include the logic correction
            # Effectively: pred_noise = pred_noise + (scale * logic_gradient)
            guided_output = model_output - (logic_guidance_scale * logic_grad)

            # D. Compute previous noisy sample x_{t-1}
            image = self.scheduler.step(guided_output, t, image).prev_sample

        return image
