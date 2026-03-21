import torch
import torch.nn as nn
from torch.optim import AdamW
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from logic_diffusion.model import SimpleUNet
from logic_diffusion.diffusion import Diffusion
import os

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")
    
    # 1. Sprint Hyperparameters
    batch_size = 64
    epochs = 5  # 5 epochs takes just a few minutes on a Codespace GPU
    learning_rate = 1e-3
    img_size = 32
    
    # 2. Load MNIST Dataset (Grayscale = 1 Channel)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Scale pixels to [-1, 1] for stable math
    ])
    
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Initialize Model & Diffusion (Note: 1 channel for grayscale)
    model = SimpleUNet(in_channels=1, out_channels=1).to(device)
    diffusion = Diffusion(device=device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # 4. Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x_0, _) in enumerate(dataloader):
            x_0 = x_0.to(device)
            current_batch_size = x_0.shape[0]
            
            # Sample random timesteps
            t = diffusion.sample_timesteps(current_batch_size)
            
            # Forward process (Add noise)
            x_t, true_noise = diffusion.add_noise(x_0, t)
            
            # Reverse process (Predict noise)
            predicted_noise = model(x_t, t)
            
            # Calculate Loss & Backpropagate
            loss = criterion(predicted_noise, true_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch+1} Average Loss: {total_loss/len(dataloader):.4f}")
    
    # 5. Save the trained weights
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/logic_diffusion_mnist.pth")
    print("Training complete! Model weights saved to weights/logic_diffusion_mnist.pth")

if __name__ == '__main__':
    train()