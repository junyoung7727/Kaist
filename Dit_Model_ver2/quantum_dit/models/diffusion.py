"""
Diffusion Scheduler for Quantum Circuit Generation
Advanced noise scheduling with multiple schedule types
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional


class DiffusionScheduler:
    """
    Advanced diffusion scheduler with multiple noise schedule types
    Supports linear, cosine, and sigmoid schedules
    """
    
    def __init__(self, 
                 timesteps: int = 1000,
                 noise_schedule: str = "cosine",
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 clip_sample: bool = True):
        """
        Initialize diffusion scheduler
        
        Args:
            timesteps: Number of diffusion timesteps
            noise_schedule: Type of noise schedule ("linear", "cosine", "sigmoid")
            beta_start: Starting beta value for linear schedule
            beta_end: Ending beta value for linear schedule
            clip_sample: Whether to clip samples to [-1, 1]
        """
        self.timesteps = timesteps
        self.noise_schedule = noise_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.clip_sample = clip_sample
        
        # Compute noise schedule
        self.betas = self._get_noise_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Precompute values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Precompute values for reverse process
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _get_noise_schedule(self) -> torch.Tensor:
        """Get noise schedule based on specified type"""
        if self.noise_schedule == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.timesteps)
        
        elif self.noise_schedule == "cosine":
            # Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
            s = 0.008
            steps = self.timesteps + 1
            x = torch.linspace(0, self.timesteps, steps)
            alphas_cumprod = torch.cos(((x / self.timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        
        elif self.noise_schedule == "sigmoid":
            # Sigmoid schedule
            betas = torch.linspace(-6, 6, self.timesteps)
            return torch.sigmoid(betas) * (self.beta_end - self.beta_start) + self.beta_start
        
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")
    
    def add_noise(self, 
                  original_samples: torch.Tensor,
                  noise: torch.Tensor,
                  timesteps: torch.Tensor) -> torch.Tensor:
        """
        Add noise to samples according to the noise schedule
        
        Args:
            original_samples: Original clean samples
            noise: Random noise to add
            timesteps: Timesteps for each sample in the batch
            
        Returns:
            Noisy samples
        """
        # Make sure alphas_cumprod are on the same device
        alphas_cumprod = self.sqrt_alphas_cumprod.to(original_samples.device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(original_samples.device)
        
        # Extract the appropriate values for each timestep
        sqrt_alpha_prod = alphas_cumprod[timesteps].flatten()
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alphas_cumprod[timesteps].flatten()
        
        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        # Add noise according to the schedule
        noisy_samples = (
            sqrt_alpha_prod * original_samples + 
            sqrt_one_minus_alpha_prod * noise
        )
        
        return noisy_samples
    
    def reverse_step(self,
                    noisy_samples: torch.Tensor,
                    predicted_noise: torch.Tensor,
                    timesteps: torch.Tensor,
                    generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Reverse diffusion step (denoising)
        
        Args:
            noisy_samples: Current noisy samples
            predicted_noise: Noise predicted by the model
            timesteps: Current timesteps
            generator: Random number generator
            
        Returns:
            Denoised samples
        """
        device = noisy_samples.device
        
        # Move scheduler tensors to the right device
        alphas = self.alphas.to(device)
        alphas_cumprod = self.alphas_cumprod.to(device)
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        posterior_variance = self.posterior_variance.to(device)
        posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        
        # Extract values for current timesteps
        alpha_t = alphas[timesteps].flatten()
        alpha_cumprod_t = alphas_cumprod[timesteps].flatten()
        sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[timesteps].flatten()
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[timesteps].flatten()
        
        # Reshape for broadcasting
        while len(alpha_t.shape) < len(noisy_samples.shape):
            alpha_t = alpha_t.unsqueeze(-1)
            alpha_cumprod_t = alpha_cumprod_t.unsqueeze(-1)
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.unsqueeze(-1)
        
        # Predict original sample using the predicted noise
        pred_original_sample = (
            noisy_samples - sqrt_one_minus_alpha_cumprod_t * predicted_noise
        ) / sqrt_alpha_cumprod_t
        
        # Clip if specified
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # Compute posterior mean
        posterior_mean_coef1_t = posterior_mean_coef1[timesteps].flatten()
        posterior_mean_coef2_t = posterior_mean_coef2[timesteps].flatten()
        
        while len(posterior_mean_coef1_t.shape) < len(noisy_samples.shape):
            posterior_mean_coef1_t = posterior_mean_coef1_t.unsqueeze(-1)
            posterior_mean_coef2_t = posterior_mean_coef2_t.unsqueeze(-1)
        
        pred_prev_sample = (
            posterior_mean_coef1_t * pred_original_sample + 
            posterior_mean_coef2_t * noisy_samples
        )
        
        # Add noise for non-zero timesteps
        if torch.any(timesteps > 0):
            variance = posterior_variance[timesteps].flatten()
            while len(variance.shape) < len(noisy_samples.shape):
                variance = variance.unsqueeze(-1)
            
            noise = torch.randn_like(noisy_samples, generator=generator)
            pred_prev_sample = pred_prev_sample + torch.sqrt(variance) * noise
        
        return pred_prev_sample
    
    def get_velocity(self,
                    sample: torch.Tensor,
                    noise: torch.Tensor,
                    timesteps: torch.Tensor) -> torch.Tensor:
        """
        Get velocity parameterization (v-parameterization)
        
        Args:
            sample: Clean sample
            noise: Noise
            timesteps: Timesteps
            
        Returns:
            Velocity target
        """
        device = sample.device
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        
        sqrt_alpha_prod = sqrt_alphas_cumprod[timesteps].flatten()
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alphas_cumprod[timesteps].flatten()
        
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity
    
    def step_with_velocity(self,
                          model_output: torch.Tensor,
                          timestep: torch.Tensor,
                          sample: torch.Tensor) -> torch.Tensor:
        """
        Step using velocity parameterization
        
        Args:
            model_output: Model output (velocity)
            timestep: Current timestep
            sample: Current sample
            
        Returns:
            Previous sample
        """
        device = sample.device
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        
        sqrt_alpha_prod = sqrt_alphas_cumprod[timestep].flatten()
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alphas_cumprod[timestep].flatten()
        
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        # Convert velocity to noise prediction
        pred_noise = sqrt_alpha_prod * model_output + sqrt_one_minus_alpha_prod * sample
        
        # Use standard reverse step
        return self.reverse_step(sample, pred_noise, timestep)


class DDIMScheduler(DiffusionScheduler):
    """
    DDIM (Denoising Diffusion Implicit Models) Scheduler
    Faster sampling with deterministic process
    """
    
    def __init__(self, *args, eta: float = 0.0, **kwargs):
        """
        Initialize DDIM scheduler
        
        Args:
            eta: DDIM eta parameter (0 = deterministic, 1 = DDPM)
        """
        super().__init__(*args, **kwargs)
        self.eta = eta
    
    def ddim_step(self,
                  noisy_samples: torch.Tensor,
                  predicted_noise: torch.Tensor,
                  timestep: int,
                  prev_timestep: int,
                  generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        DDIM sampling step
        
        Args:
            noisy_samples: Current noisy samples
            predicted_noise: Predicted noise
            timestep: Current timestep
            prev_timestep: Previous timestep
            generator: Random number generator
            
        Returns:
            Previous sample
        """
        device = noisy_samples.device
        
        # Move to device
        alphas_cumprod = self.alphas_cumprod.to(device)
        
        alpha_prod_t = alphas_cumprod[timestep]
        alpha_prod_t_prev = alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # Predict original sample
        pred_original_sample = (
            noisy_samples - beta_prod_t.sqrt() * predicted_noise
        ) / alpha_prod_t.sqrt()
        
        # Clip if specified
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # Compute direction to x_t
        pred_sample_direction = beta_prod_t_prev.sqrt() * predicted_noise
        
        # Compute x_{t-1}
        prev_sample = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction
        
        # Add noise if eta > 0
        if self.eta > 0:
            variance = self.eta * (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
            noise = torch.randn_like(noisy_samples, generator=generator)
            prev_sample = prev_sample + variance.sqrt() * noise
        
        return prev_sample
