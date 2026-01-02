import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from logic_diffusion.config import TrainingConfig
from logic_diffusion.modeling import SimpleUNet
from logic_diffusion.logic import LogicConstraint

def train():
    # 1. Load Config
    config = TrainingConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting Logic Diffusion Training on {device}")

    # 2. Initialize Model
    model = SimpleUNet(image_channels=config.channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    mse = torch.nn.MSELoss()

    # 3. Create Dummy Data (Bias Simulation)
    # Creating 100 images of random noise for testing compilation
    dummy_data = torch.randn(100, config.channels, config.image_size, config.image_size)
    dataloader = DataLoader(TensorDataset(dummy_data), batch_size=config.batch_size, shuffle=True)

    # 4. Training Loop
    model.train()
    for epoch in range(config.num_epochs):
        for batch in dataloader:
            x_0 = batch[0].to(device)
            t = torch.randint(0, config.timesteps, (x_0.shape[0],)).to(device)
            
            # Add noise (Forward Diffusion)
            # Note: For brevity, we are just adding simple Gaussian noise here.
            # In production, use the alpha_hat schedule from pipeline.
            noise = torch.randn_like(x_0)
            x_t = x_0 + noise # Simplified for v0 test
            
            optimizer.zero_grad()
            
            # Predict noise
            predicted_noise = model(x_t, t)
            
            # Calculate Standard Loss
            loss_diff = mse(noise, predicted_noise)
            
            # Calculate Logic Loss (Optional during training, stronger during sampling)
            # Here we enforce that the mean pixel value should be close to 0 (Toy Rule)
            # Rule: Mean(x) == 0  => Loss: (Mean(x) - 0)^2
            loss_logic = torch.mean(predicted_noise.mean() ** 2)
            
            # Total Loss
            loss = loss_diff + (config.logic_weight * loss_logic)
            
            loss.backward()
            optimizer.step()
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f} (Logic Loss: {loss_logic.item():.4f})")

    print("✅ Training Complete. Model ready for logic-guided sampling.")
    torch.save(model.state_dict(), "logic_diffusion_v0.pt")

if __name__ == "__main__":
    train()
