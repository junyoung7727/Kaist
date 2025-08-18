#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

class FocalLoss(nn.Module):
    """Focal Loss for regression - focuses on hard examples with large errors"""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', threshold=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.threshold = threshold  # Error threshold to define "hard" examples
    
    def forward(self, inputs, targets):
        # Calculate MSE loss
        mse_loss = F.mse_loss(inputs, targets, reduction='none')
        
        # Convert to "difficulty" score: higher error = more difficult
        # Normalize by threshold to get relative difficulty
        difficulty = torch.clamp(mse_loss / self.threshold, min=0.0, max=1.0)
        
        # Focal weight: focus more on difficult examples
        focal_weight = self.alpha * (difficulty ** self.gamma)
        
        # Apply focal weighting
        focal_loss = focal_weight * mse_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class LabelSmoothingMSE(nn.Module):
    """Label smoothing for regression tasks"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        # Add noise to targets for regularization
        noise = torch.randn_like(target) * self.smoothing
        smoothed_target = target + noise
        return F.mse_loss(pred, smoothed_target)

class MixupLoss(nn.Module):
    """Mixup augmentation for quantum circuits"""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def mixup_data(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, pred, y_a, y_b, lam):
        return lam * F.mse_loss(pred, y_a) + (1 - lam) * F.mse_loss(pred, y_b)

class AdaptiveLossWeighting(nn.Module):
    """Dynamically adjust loss weights based on training progress"""
    def __init__(self, initial_weights: Dict[str, float]):
        super().__init__()
        self.weights = nn.ParameterDict({
            key: nn.Parameter(torch.tensor(weight, dtype=torch.float32))
            for key, weight in initial_weights.items()
        })
        self.loss_history = {key: [] for key in initial_weights.keys()}
    
    def update_weights(self, losses: Dict[str, float]):
        """Update weights based on relative loss magnitudes"""
        for key, loss_val in losses.items():
            self.loss_history[key].append(loss_val)
            
            # Keep only recent history
            if len(self.loss_history[key]) > 100:
                self.loss_history[key] = self.loss_history[key][-100:]
        
        # Adjust weights inversely proportional to loss magnitude
        if len(self.loss_history[list(losses.keys())[0]]) > 10:
            for key in losses.keys():
                avg_loss = np.mean(self.loss_history[key][-10:])
                # Higher loss gets lower weight to prevent dominance
                self.weights[key].data = torch.clamp(
                    1.0 / (avg_loss + 1e-8), min=0.1, max=100.0
                )
    
    def get_weights(self) -> Dict[str, float]:
        return {key: weight.item() for key, weight in self.weights.items()}

class GradientClipping:
    """Advanced gradient clipping techniques"""
    @staticmethod
    def adaptive_clip_grad(parameters, clip_factor=0.01, eps=1e-8):
        """Adaptive gradient clipping based on gradient norms"""
        total_norm = 0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        # Adaptive clipping
        clip_coef = clip_factor / (total_norm + eps)
        if clip_coef < 1:
            for p in parameters:
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
        
        return total_norm

class CosineAnnealingWarmRestarts:
    """Cosine annealing with warm restarts"""
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        self.T_i = T_0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.T_cur + 1
        self.T_cur = epoch
        
        if self.T_cur >= self.T_i:
            self.T_cur = self.T_cur - self.T_i
            self.T_i = self.T_i * self.T_mult
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = self.eta_min + (base_lr - self.eta_min) * \
                (1 + np.cos(np.pi * self.T_cur / self.T_i)) / 2

class EarlyStopping:
    """Early stopping with patience"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

class ProgressiveResizing:
    """Progressive resizing for sequence length"""
    def __init__(self, initial_size=64, final_size=256, epochs=100):
        self.initial_size = initial_size
        self.final_size = final_size
        self.epochs = epochs
    
    def get_size(self, epoch):
        """Get current sequence size based on epoch"""
        progress = min(epoch / self.epochs, 1.0)
        size = self.initial_size + (self.final_size - self.initial_size) * progress
        return int(size)

class StochasticWeightAveraging:
    """Stochastic Weight Averaging implementation"""
    def __init__(self, model, swa_start=10, swa_freq=5, swa_lr=0.05):
        self.model = model
        self.swa_model = None
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_lr = swa_lr
        self.swa_n = 0
    
    def update(self, epoch):
        """Update SWA model"""
        if epoch >= self.swa_start and (epoch - self.swa_start) % self.swa_freq == 0:
            if self.swa_model is None:
                self.swa_model = {key: param.clone() for key, param in self.model.named_parameters()}
                self.swa_n = 1
            else:
                # Update running average
                for key, param in self.model.named_parameters():
                    self.swa_model[key] = (self.swa_model[key] * self.swa_n + param) / (self.swa_n + 1)
                self.swa_n += 1
    
    def swap_swa_sgd(self):
        """Swap SWA and SGD parameters"""
        if self.swa_model is not None:
            for key, param in self.model.named_parameters():
                param.data, self.swa_model[key] = self.swa_model[key], param.data

# Advanced training utilities
class TrainingEnhancer:
    """Collection of training enhancement techniques"""
    
    @staticmethod
    def create_enhanced_loss(config):
        """Create enhanced loss function with multiple techniques"""
        losses = {}
        
        # Focal loss for imbalanced fidelity
        losses['fidelity'] = FocalLoss(alpha=2.0, gamma=2.0)
        
        # Label smoothing for other properties
        losses['entanglement'] = LabelSmoothingMSE(smoothing=0.05)
        losses['expressibility'] = LabelSmoothingMSE(smoothing=0.1)
        
        # Adaptive weighting
        initial_weights = {
            'entanglement': config.entanglement_weight,
            'fidelity': config.fidelity_weight,
            'expressibility': config.expressibility_weight
        }
        adaptive_weighting = AdaptiveLossWeighting(initial_weights)
        
        return losses, adaptive_weighting
    
    @staticmethod
    def setup_advanced_optimizer(model, config):
        """Setup optimizer with advanced techniques"""
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            eps=config.eps
        )
        
        # Cosine annealing with warm restarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Early stopping
        early_stopping = EarlyStopping(patience=15, min_delta=0.001)
        
        # SWA
        swa = StochasticWeightAveraging(model, swa_start=50, swa_freq=5)
        
        return optimizer, scheduler, early_stopping, swa
    
    @staticmethod
    def apply_mixup(batch, mixup_loss, alpha=0.2):
        """Apply mixup augmentation to batch"""
        if np.random.random() < 0.5:  # 50% chance to apply mixup
            mixed_x, y_a, y_b, lam = mixup_loss.mixup_data(batch['input'], batch['target'])
            batch['input'] = mixed_x
            batch['mixup_targets'] = (y_a, y_b, lam)
        return batch
