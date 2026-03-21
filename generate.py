import torch
import torchvision
import os
from logic_diffusion.model import SimpleUNet
from logic_diffusion.diffusion import Diffusion

@torch.no_grad()
def generate_images(model, diffusion, num_images=16, img_size=32):
    device = diffusion.device
    model.eval()
    
    # 1. Start with pure random noise (1 channel for MNIST)
    x = torch.randn(num_images, 1, img_size, img_size).to(device)
    print("Generating images step-by-step from pure noise...")
    
    # 2. Iterate backwards from T to 0
    for i in reversed(range(1, diffusion.num_timesteps)):
        t = torch.full((num_images,), i, dtype=torch.long, device=device)
        
        predicted_noise = model(x, t)
        
        alpha = diffusion.alphas[t][:, None, None, None]
        alpha_cumprod = diffusion.alpha_cumprod[t][:, None, None, None]
        beta = diffusion.betas[t][:, None, None, None]
        
        noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
        x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * predicted_noise) + torch.sqrt(beta) * noise
        
        if i % 200 == 0:
            print(f"Timestep {i} complete...")

    # Scale values back to [0, 1] for saving
    x = (x.clamp(-1, 1) + 1) / 2
    return x

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize (1 channel)
    model = SimpleUNet(in_channels=1, out_channels=1).to(device)
    diffusion = Diffusion(device=device)
    
    # Load the trained weights!
    weight_path = "weights/logic_diffusion_mnist.pth"
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print("Loaded trained weights successfully.")
    else:
        print("WARNING: No weights found. Run train.py first!")
    
    # Generate a 4x4 grid of images
    samples = generate_images(model, diffusion, num_images=16)
    
    os.makedirs("outputs", exist_ok=True)
    torchvision.utils.save_image(samples, "outputs/mnist_generated.png", nrow=4)
    print("Success! Open 'outputs/mnist_generated.png' to see the results.")