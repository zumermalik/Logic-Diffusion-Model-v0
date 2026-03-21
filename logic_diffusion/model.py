import torch
import torch.nn as nn

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # Simplified for sprint setup: Downsample -> Bottleneck -> Upsample
        self.inc = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(64, 128, kernel_size=3, padding=1))
        
        # Time embedding projection
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.GELU(),
            nn.Linear(128, 128)
        )
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, t):
        # Time embedding
        t = t.unsqueeze(-1).type(torch.float)
        t_emb = self.time_embed(t).unsqueeze(-1).unsqueeze(-1)
        
        # Pass through network
        x1 = self.inc(x)
        x2 = self.down1(x1)
        
        # Inject time embedding at the bottleneck
        x2 = x2 + t_emb 
        
        x = self.up1(x2)
        # Note: In a full UNet, we'd add skip connections here: x = x + x1
        output = self.outc(x)
        return output