# 🧠 Logic Diffusion v0

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/python-3.8%2B-green)](https://www.python.org/downloads/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/) [![Status](https://img.shields.io/badge/status-research--prototype-yellow)]()
**Logic Diffusion** is a neuro-symbolic generative architecture designed to address bias in deep learning distributions.

Unlike standard diffusion models that blindly approximate the training data distribution $p(x)$ (inheriting all its biases), Logic Diffusion learns a conditional distribution $p(x | L)$ subject to a set of differentiable logical constraints $L$. This effectively "steers" the generative process toward a fair manifold, even when trained on biased data.

---

## 📂 Project Structure

```text
logic-diffusion-v0/
│
├── logic_diffusion/        # The Core Framework
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Hyperparameters & Configuration
│   ├── logic.py            # Differentiable Logic (T-Norms) & Constraints
│   ├── modeling.py         # Lightweight U-Net Architecture
│   └── pipeline.py         # Logic-Guided Diffusion Sampling Loop
│
├── train.py                # Main training script (Joint Optimization)
├── app.py                  # Interactive Gradio Web Demo
├── requirements.txt        # Project Dependencies
└── README.md               # Documentation

```

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the dependencies.

```bash
git clone [https://github.com/your-username/logic-diffusion-v0.git](https://github.com/your-username/logic-diffusion-v0.git)
cd logic-diffusion-v0
pip install -r requirements.txt

```

### 2. Training the Model

Run the training script to initialize the U-Net and train it on synthetic data. The script uses a joint loss function:
$$ \mathcal{L}*{total} = \mathcal{L}*{MSE} + \lambda \cdot \mathcal{L}_{Logic} $$

```bash
python train.py

```

* **Output:** The script will print loss metrics to the console and save the trained model weights to `logic_diffusion_v0.pt`.

### 3. Interactive Demo (Web UI)

Launch the Gradio interface to generate samples and tweak the "Logic Strictness" parameter in real-time.

```bash
python app.py

```

* Click the local URL (e.g., `http://127.0.0.1:7860`) to open the app in your browser.

---

## 🧠 How It Works

### The Core Innovation: Differentiable Logic

Standard Boolean logic (`True`/`False`) has no gradients, making it unusable for deep learning training. Logic Diffusion uses **Fuzzy Logic (T-Norms)** to relax these rules into continuous functions.

* **AND Operator:**  (Product T-Norm)
* **IMPLIES Operator:**  (Reichenbach Implication)

### The Constraint

In `logic.py`, we define constraints that calculate a "Truth Value" (0 to 1) for a generated batch. The model minimizes the violation of this truth value:

```python
# Pseudo-code example
violation = 1.0 - truth_value(generated_image)
loss.backward()  # Gradients update the image pixels to be "truer"

```

---

## 🗺 Roadmap

* [x] **v0:** Core implementation of Differentiable Logic and Simple U-Net.
* [ ] **v0.1:** Integration with Latent Diffusion (Stable Diffusion).
* [ ] **v0.2:** API for defining First-Order Logic rules via natural language.

---

## 🤝 Contributing

Contributions are welcome! We are looking for help with:

1. Implementing new T-Norms (Godel, Lukasiewicz).
2. Optimizing the logical gradient calculation.
3. Adding support for RGB datasets (CIFAR-10, CelebA).

## 📜 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

