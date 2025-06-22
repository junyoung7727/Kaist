import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda import amp # For mixed precision
import numpy as np

from .constants import (
    DEVICE, LEARNING_RATE, SCHEDULER_FACTOR, SCHEDULER_PATIENCE, MAX_GRAD_NORM,
    CLIP_EPSILON, RL_EPOCHS, GAMMA, GAE_LAMBDA, ENTROPY_COEF, CRITIC_COEF, ACTION_DIM
)

class PPOAgent:
    """Proximal Policy Optimization (PPO) 에이전트"""

    def __init__(self, obs_dim, act_dim=ACTION_DIM, hidden_dim=64):
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim * 2)  # 평균과 로그 표준편차
        ).to(DEVICE)
        if hasattr(torch, 'compile'): self.actor = torch.compile(self.actor)

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        ).to(DEVICE)
        if hasattr(torch, 'compile'): self.critic = torch.compile(self.critic)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=LEARNING_RATE
        )
        self.act_dim = act_dim
        self.scaler = amp.GradScaler(enabled=(DEVICE.type == 'cuda'))
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)

    def get_action(self, obs, deterministic=False):
        """정책에서 액션 샘플링하고 로그 확률 반환"""
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0) # Batch dim

            out = self.actor(obs_tensor)
            mean, log_std = out[:, :self.act_dim], out[:, self.act_dim:]

        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)

        if deterministic:
            action_tensor = mean
        else:
            action_tensor = dist.sample()

        log_prob = dist.log_prob(action_tensor).sum(dim=-1) # Sum over action dimensions if act_dim > 1

        # 액션 범위 제한 (RL 환경의 action_space에 맞춰야 함)
        action_tensor_clamped = torch.clamp(action_tensor, 0.1, 2.0) # TODO: Get bounds from env.action_space

        action_np = action_tensor_clamped.squeeze(0).cpu().numpy() # Remove batch dim
        log_prob_item = log_prob.squeeze(0).cpu().item() # Remove batch dim, convert to scalar

        return action_np, log_prob_item

    def get_log_probs_and_values(self, obs, actions):
        """주어진 상태와 행동에 대한 로그 확률, 가치, 엔트로피 계산"""
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
        actions_tensor = torch.as_tensor(actions, dtype=torch.float32, device=DEVICE)

        if obs_tensor.dim() == 1: obs_tensor = obs_tensor.unsqueeze(0)
        # Ensure actions_tensor has batch dim if it's a single action from a batch of obs
        if actions_tensor.dim() == 1 and self.act_dim > 1 : actions_tensor = actions_tensor.unsqueeze(0)
        elif actions_tensor.dim() == 0 and self.act_dim == 1: actions_tensor = actions_tensor.unsqueeze(0).unsqueeze(0)
        elif actions_tensor.dim() == 1 and self.act_dim == 1 and obs_tensor.shape[0] == actions_tensor.shape[0]: actions_tensor = actions_tensor.unsqueeze(1)
        
        out = self.actor(obs_tensor)
        mean, log_std = out[:, :self.act_dim], out[:, self.act_dim:]
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        
        log_probs = dist.log_prob(actions_tensor).sum(dim=-1)
        
        values = self.critic(obs_tensor).squeeze(-1) 
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, values, entropy

    def get_value(self, obs):
        """가치 함수에서 상태 가치 계산"""
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            value = self.critic(obs_tensor)
            return value.squeeze().cpu().item()

    def update(self, rollouts):
        """PPO 업데이트"""
        obs = torch.as_tensor(rollouts["obs"], dtype=torch.float32, device=DEVICE)
        actions = torch.as_tensor(rollouts["actions"], dtype=torch.float32, device=DEVICE)
        old_log_probs = torch.as_tensor(rollouts["log_probs"], dtype=torch.float32, device=DEVICE)
        advantages = torch.as_tensor(rollouts["advantages"], dtype=torch.float32, device=DEVICE)
        returns = torch.as_tensor(rollouts["returns"], dtype=torch.float32, device=DEVICE)

        for _ in range(RL_EPOCHS):
            self.optimizer.zero_grad(set_to_none=True)
            
            with amp.autocast(enabled=(DEVICE.type == 'cuda')):
                log_probs, values, entropy = self.get_log_probs_and_values(obs, actions)
                
                ratio = torch.exp(log_probs - old_log_probs)
                
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns)
                entropy_loss = -entropy.mean()
                
                loss = policy_loss + CRITIC_COEF * value_loss + ENTROPY_COEF * entropy_loss
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer) 
            nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), MAX_GRAD_NORM)
            self.scaler.step(self.optimizer)
            self.scaler.update()
