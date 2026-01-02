# Logic Diffusion v0

**Logic Diffusion** is a neuro-symbolic generative model that enforces axiomatic constraints during the diffusion process. It is designed to solve the "Bias Distribution" problem in deep learning by steering the generative manifold toward logically valid subspaces.

## 🚧 Status: v0 (Prototype)

Current functionality is focused on:
- **Differentiable Logic:** A library of T-Norms for backpropagating truth values.
- **Constraint Guidance:** A pipeline that accepts logical rules as loss functions.
- **Lightweight U-Net:** A custom architecture for testing on CPU/Colab.

## ⚡ Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt