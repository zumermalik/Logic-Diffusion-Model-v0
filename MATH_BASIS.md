# The Mathematical Basis of LogicDiffusion

This document outlines the core probabilistic logic driving the LogicDiffusion model, based on the framework of Denoising Diffusion Probabilistic Models (DDPMs).

## 1. The Forward Process (Adding Noise)

The forward process is a Markov chain that gradually adds Gaussian noise to the data over $T$ timesteps. Given a real data sample $x_0$, we define a variance schedule $\beta_1, \ldots, \beta_T$. 

The transition from timestep $t-1$ to $t$ is defined as:
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$$

A crucial property of this process is that we can sample $x_t$ at any arbitrary timestep directly from $x_0$ without iterating through intermediate steps. Let $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$. The marginal distribution is:
$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$$

Using the reparameterization trick, we can express $x_t$ as:
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$
where $\epsilon \sim \mathcal{N}(0, I)$.

## 2. The Reverse Process (Denoising)

If we can reverse the forward process and sample from $q(x_{t-1} | x_t)$, we can start from pure Gaussian noise $x_T \sim \mathcal{N}(0, I)$ and iteratively denoise it to generate a real sample $x_0$.

Because the true reverse step $q(x_{t-1} | x_t)$ depends on the entire data distribution, we approximate it using a neural network parameterizing $p_\theta$:
$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

In DDPMs, the variance $\Sigma_\theta(x_t, t)$ is fixed to untrained constants (e.g., $\beta_t I$ or $\tilde{\beta}_t I$). The network is trained solely to predict the mean $\mu_\theta$.

## 3. The Objective Function

Instead of predicting the mean $\mu_\theta$ directly, empirical results show it is much more stable to parameterize the model to predict the noise $\epsilon$ that was added to $x_0$ to create $x_t$. Let our neural network be $\epsilon_\theta(x_t, t)$.

The simplified objective function ($L_{simple}$) minimizes the Mean Squared Error between the true noise $\epsilon$ and the predicted noise:
$$L_{simple} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon, t) \|^2 \right]$$

## 4. Sampling / Inference

Once the network $\epsilon_\theta$ is trained, we generate new data by initializing $x_T \sim \mathcal{N}(0, I)$ and applying the reverse step for $t = T, T-1, \ldots, 1$:
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$

Where $z \sim \mathcal{N}(0, I)$ for $t > 1$, and $z = 0$ for $t = 1$. This stochasticity acts as Langevin dynamics, keeping the generated images sharp and stable.