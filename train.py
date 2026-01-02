import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os

# Import our custom modules
from logic_diffusion.config import TrainingConfig
from logic_diffusion.modeling import SimpleUNet
from logic_diffusion.logic import LogicConstraint, DifferentiableLogic

def train():
    # 1. Setup & Configuration
    config = TrainingConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Initializing Logic Diffusion v0 on {device}...")
    
    # Create output directory for saved models
    os.makedirs("checkpoints", exist_ok=True)

    # 2. Initialize the Neuro-Symbolic Model
    model = SimpleUNet(image_channels=config.channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    mse_criterion = nn.MSELoss()

    # 3. Define Logic Constraints
    # EXAMPLE: We want the model to learn that "Pixel Intensity" shouldn't be too high.
    # Rule: Intensity < 0.8  =>  Truth = 1.0 if x < 0.8 else decay
    # In a real scenario, this would be: "Prediction is independent of Gender"
    def intensity_rule(x):
        # We use a "Godel" implication for simplicity in v0
        # "It is NOT true that x is very large"
        return DifferentiableLogic.not_op(torch.relu(x - 0.8))

    # Wrap it in our module so it tracks gradients
    logic_supervisor = LogicConstraint(intensity_rule, weight=config.logic_weight)

    # 4. Mock Data Loader (Bias Simulation)
    # We generate noise that intentionally HAS the bias we want to remove
    print("   ... Synthesizing biased training data")
    # 1000 samples, 1 channel, 32x32
    dummy_data = torch.randn(1000, config.channels, config.image_size, config.image_size) + 0.5 
    dataset = TensorDataset(dummy_data)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # 5. The Training Loop
    model.train()
    print(f"   ... Starting training for {config.num_epochs} epochs")
    
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        logic_loss_total = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}", leave=False)
        
        for batch in pbar:
            x_0 = batch[0].to(device)
            batch_size = x_0.shape[0]
            
            # A. Sample Timesteps (t)
            t = torch.randint(0, config.timesteps, (batch_size,), device=device).long()
            
            # B. Add Noise (Forward Diffusion)
            noise = torch.randn_like(x_0)
            # Simple linear noise schedule for v0 prototype
            # (In v1, we use the alpha_cumprod from config)
            noisy_image = x_0 + (0.1 * t.view(-1, 1, 1, 1) * noise)
            
            # C. Model Prediction
            noise_pred = model(noisy_image, t)
            
            # D. Calculate Standard Diffusion Loss (Reconstruction)
            loss_diff = mse_criterion(noise_pred, noise)
            
            # E. Calculate Logic Loss (The Correction)
            # We check if the predicted denoising direction respects our rules
            loss_logic = logic_supervisor(noisy_image - noise_pred)
            
            # F. Joint Optimization
            total_loss = loss_diff + loss_logic
            
            # Backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            logic_loss_total += loss_logic.item()
            
            pbar.set_postfix({"Loss": f"{total_loss.item():.4f}", "Logic": f"{loss_logic.item():.4f}"})

        # Logging
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1}: Avg Loss {avg_loss:.4f}")
            
            # Save Checkpoint
            torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pt")

    print("\n✅ Training Complete. Model saved to checkpoints/")

if __name__ == "__main__":
    train()