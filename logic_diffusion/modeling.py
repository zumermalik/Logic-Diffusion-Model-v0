import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Encodes the 'time' step so the model knows if it's looking at
    pure noise (t=1000) or a nearly finished image (t=0).
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    """A basic ResNet-style block for the U-Net"""
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.transform = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        else:
            self.transform = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        
        self.conv1 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()

    def forward(self, x, t):
        # First convolution / upsample
        h = self.bnorm1(self.relu(self.transform(x)))
        # Inject Time Embedding
        time_emb = self.relu(self.time_mlp(t))
        # Add time axis to match image dimensions [(B,C) -> (B,C,1,1)]
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        # Second convolution
        return self.bnorm2(self.relu(self.conv2(h)))

class SimpleUNet(nn.Module):
    """
    The main Logic Diffusion backbone.
    """
    def __init__(self, image_channels=1, down_channels=(64, 128, 256), time_emb_dim=32):
        super().__init__()
        self.image_channels = image_channels
        
        # Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Initial Projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample Path
        self.downs = nn.ModuleList([
            Block(down_channels[i], down_channels[i+1], time_emb_dim) \
            for i in range(len(down_channels)-1)
        ])
        
        # Upsample Path
        self.ups = nn.ModuleList([
            Block(down_channels[i+1], down_channels[i], time_emb_dim, up=True) \
            for i in range(len(down_channels)-1, 0, -1)
        ])

        # Output
        self.output = nn.Conv2d(down_channels[0], image_channels, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        
        # Save residuals for skip connections
        residuals = []
        
        # Encoder (Down)
        for down in self.downs:
            x = down(x, t)
            residuals.append(x)
            x = F.max_pool2d(x, 2)

        # Decoder (Up)
        for up in self.ups:
            residual = residuals.pop()
            # Bilinear upsample to match residual size if needed
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            # If shapes mismatch due to odd padding, crop
            if x.shape != residual.shape:
                x = x[:, :, :residual.shape[2], :residual.shape[3]]
            x = up(x + residual, t) # Skip connection

        return self.output(x)
