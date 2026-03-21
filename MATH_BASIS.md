# The Mathematical Basis of LogicDiffusion

This document outlines the core probabilistic logic driving the LogicDiffusion model, based on the framework of Denoising Diffusion Probabilistic Models (DDPMs).

## 1. The Forward Process (Adding Noise)

The forward process is a Markov chain that gradually adds Gaussian noise to the data over $T$ timesteps. Given a real data sample $x_{0}$, we define a variance schedule $\beta_{1}, \ldots, \beta_{T}$. 

The transition from timestep $t-1$ to $t$ is defined as:

$$
q(x_{t} | x_{t-1}) = \mathcal{N}(x_{t}; \sqrt{1 - \beta_{t}} x_{t-1}, \beta_{t} I)
$$

A crucial property of this process is that we can sample $x_{t}$ at any arbitrary timestep directly from $x_{0}$ without iterating through intermediate steps. Let $\alpha_{t} = 1 - \beta_{t}$ and $\bar{\alpha}_{t} = \prod_{i=1}^{t} \alpha_{i}$. The marginal distribution is:

$$
q(x_{t} | x_{0}) = \mathcal{N}(x_{t}; \sqrt{\bar{\alpha}_{t}} x_{0}, (1 - \bar{\alpha}_{t}) I)
$$

Using the reparameterization trick, we can express $x_{t}$ as:

$$
x_{t} = \sqrt{\bar{\alpha}_{t}} x_{0} + \sqrt{1 - \bar{\alpha}_{t}} \epsilon
$$

where $\epsilon \sim \mathcal{N}(0, I)$.

## 2. The Reverse Process (Denoising)

If we can reverse the forward process and sample from $q(x_{t-1} | x_{t})$, we can start from pure Gaussian noise $x_{T} \sim \mathcal{N}(0, I)$ and iteratively denoise it to generate a real sample $x_{0}$.

Because the true reverse step $q(x_{t-1} | x_{t})$ depends on the entire data distribution, we approximate it using a neural network parameterizing $p_{\theta}$:

$$
p_{\theta}(x_{t-1} | x_{t}) = \mathcal{N}(x_{t-1}; \mu_{\theta}(x_{t}, t), \Sigma_{\theta}(x_{t}, t))
$$

In DDPMs, the variance $\Sigma_{\theta}(x_{t}, t)$ is fixed to untrained constants. The network is trained solely to predict the mean $\mu_{\theta}$.

## 3. The Objective Function

It is empirically more stable to parameterize the model to predict the noise $\epsilon$ that was added to $x_{0}$ to create $x_{t}$. Let our neural network be $\epsilon_{\theta}(x_{t}, t)$.

The simplified objective function ($L_{simple}$) minimizes the Mean Squared Error:

$$
L_{simple} = \mathbb{E}_{t, x_{0}, \epsilon} [ \| \epsilon - \epsilon_{\theta}(\sqrt{\bar{\alpha}_{t}}x_{0} + \sqrt{1 - \bar{\alpha}_{t}}\epsilon, t) \|^{2} ]
$$

## 4. Sampling / Inference

Once trained, we generate new data by initializing $x_{T} \sim \mathcal{N}(0, I)$ and applying the reverse step for $t = T, T-1, \ldots, 1$:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_{t}}} \left( x_{t} - \frac{1 - \alpha_{t}}{\sqrt{1 - \bar{\alpha}_{t}}} \epsilon_{\theta}(x_{t}, t) \right) + \sigma_{t} z
$$

Where $z \sim \mathcal{N}(0, I)$ for $t > 1$, and $z = 0$ for $t = 1$.
