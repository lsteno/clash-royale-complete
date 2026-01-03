"""
DreamerV3 Training Algorithm.

This implements the core Dreamer training procedure:
1. Collect experience from environment
2. Train world model (encoder, RSSM, decoder, reward predictor)
3. Train actor/critic on imagined trajectories
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import numpy as np

from .dreamer_v3 import DreamerV3, RSSMState


@dataclass
class Experience:
    """Single experience tuple."""
    obs: np.ndarray      # (C, H, W)
    action: np.ndarray   # (action_size,) one-hot
    reward: float
    done: bool
    

class SequenceBuffer:
    """
    Buffer that stores sequences for RSSM training.
    Dreamer needs sequences to train the recurrent model.
    """
    def __init__(self, capacity: int = 100000, seq_length: int = 50):
        self.capacity = capacity
        self.seq_length = seq_length
        self.buffer: List[Experience] = []
        self.episode_starts: List[int] = [0]
        
    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, done: bool):
        """Add experience to buffer."""
        self.buffer.append(Experience(obs, action, reward, done))
        
        # Track episode boundaries
        if done:
            self.episode_starts.append(len(self.buffer))
        
        # Remove old data if over capacity
        while len(self.buffer) > self.capacity:
            self.buffer.pop(0)
            # Adjust episode starts
            self.episode_starts = [max(0, s - 1) for s in self.episode_starts]
            self.episode_starts = [s for s in self.episode_starts if s < len(self.buffer)]
            if not self.episode_starts or self.episode_starts[0] != 0:
                self.episode_starts.insert(0, 0)
    
    def sample_sequences(self, batch_size: int, device: torch.device) -> Tuple:
        """
        Sample batch of sequences for training.
        
        Returns:
            obs: (B, T, C, H, W)
            actions: (B, T, action_size)
            rewards: (B, T)
            dones: (B, T)
        """
        # Find valid sequence start points
        valid_starts = []
        for i, start in enumerate(self.episode_starts):
            end = self.episode_starts[i + 1] if i + 1 < len(self.episode_starts) else len(self.buffer)
            for j in range(start, min(end - self.seq_length, end)):
                valid_starts.append(j)
        
        if len(valid_starts) < batch_size:
            # Not enough data, return None
            return None
        
        # Sample random sequences
        indices = np.random.choice(valid_starts, batch_size, replace=False)
        
        obs_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []
        
        for start_idx in indices:
            obs_seq = []
            action_seq = []
            reward_seq = []
            done_seq = []
            
            for t in range(self.seq_length):
                exp = self.buffer[start_idx + t]
                obs_seq.append(exp.obs)
                action_seq.append(exp.action)
                reward_seq.append(exp.reward)
                done_seq.append(float(exp.done))
            
            obs_batch.append(np.stack(obs_seq))
            action_batch.append(np.stack(action_seq))
            reward_batch.append(np.array(reward_seq))
            done_batch.append(np.array(done_seq))
        
        # Convert to tensors
        obs = torch.tensor(np.stack(obs_batch), dtype=torch.float32, device=device)
        actions = torch.tensor(np.stack(action_batch), dtype=torch.float32, device=device)
        rewards = torch.tensor(np.stack(reward_batch), dtype=torch.float32, device=device)
        dones = torch.tensor(np.stack(done_batch), dtype=torch.float32, device=device)
        
        return obs, actions, rewards, dones
    
    def __len__(self):
        return len(self.buffer)


class DreamerTrainer:
    """
    Trainer for DreamerV3.
    
    Training consists of two phases:
    1. World model training: Learn to predict observations and rewards
    2. Behavior training: Learn actor/critic on imagined trajectories
    """
    def __init__(
        self,
        model: DreamerV3,
        device: torch.device,
        # World model hyperparameters
        model_lr: float = 3e-4,
        kl_scale: float = 1.0,
        kl_free: float = 1.0,
        # Behavior hyperparameters
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        imagine_horizon: int = 15,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        # Training hyperparameters
        batch_size: int = 16,
        seq_length: int = 50,
        grad_clip: float = 100.0,
    ):
        self.model = model
        self.device = device
        self.kl_scale = kl_scale
        self.kl_free = kl_free
        self.imagine_horizon = imagine_horizon
        self.gamma = gamma
        self.lambda_ = lambda_
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.grad_clip = grad_clip
        
        # Separate optimizers for world model and behavior
        self.model_params = list(model.encoder.parameters()) + \
                           list(model.decoder.parameters()) + \
                           list(model.transition.parameters()) + \
                           list(model.representation.parameters()) + \
                           list(model.reward_predictor.parameters()) + \
                           list(model.continue_predictor.parameters())
        
        self.model_optimizer = torch.optim.Adam(self.model_params, lr=model_lr)
        self.actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=critic_lr)
        
        # Buffer
        self.buffer = SequenceBuffer(capacity=100000, seq_length=seq_length)
        
        # Statistics
        self.train_steps = 0
        
    def add_experience(self, obs: np.ndarray, action: np.ndarray, 
                       reward: float, done: bool):
        """Add experience to buffer."""
        self.buffer.add(obs, action, reward, done)
    
    def train_step(self) -> Optional[Dict[str, float]]:
        """
        Single training step.
        
        Returns:
            Dictionary of losses or None if not enough data
        """
        # Sample sequences
        data = self.buffer.sample_sequences(self.batch_size, self.device)
        if data is None:
            return None
        
        obs, actions, rewards, dones = data
        
        # Train world model
        model_losses = self._train_world_model(obs, actions, rewards, dones)
        
        # Train actor/critic on imagined trajectories
        behavior_losses = self._train_behavior(obs, actions)
        
        self.train_steps += 1
        
        return {**model_losses, **behavior_losses}
    
    def _train_world_model(self, obs: torch.Tensor, actions: torch.Tensor,
                           rewards: torch.Tensor, dones: torch.Tensor) -> Dict[str, float]:
        """
        Train world model components.
        
        Loss = reconstruction_loss + reward_loss + KL_divergence
        """
        B, T = obs.shape[:2]
        
        # Initialize state
        state = self.model.initial_state(B, self.device)
        
        # Process sequence
        priors = []
        posteriors = []
        
        for t in range(T):
            prior, posterior = self.model.observe_step(obs[:, t], actions[:, t], state)
            priors.append(prior)
            posteriors.append(posterior)
            state = posterior.detach()  # Use posterior for next step
        
        # Stack states
        def stack_states(states):
            return RSSMState(
                mean=torch.stack([s.mean for s in states], dim=1),
                std=torch.stack([s.std for s in states], dim=1),
                stoch=torch.stack([s.stoch for s in states], dim=1),
                deter=torch.stack([s.deter for s in states], dim=1),
            )
        
        prior_states = stack_states(priors)
        posterior_states = stack_states(posteriors)
        
        # Features for prediction
        features = posterior_states.features  # (B, T, feature_size)
        
        # Reconstruction loss
        recon = self.model.decoder(features)
        recon_loss = F.mse_loss(recon, obs)
        
        # Reward prediction loss
        pred_rewards = self.model.reward_predictor(features)
        reward_loss = F.mse_loss(pred_rewards, rewards)
        
        # Continue prediction loss
        pred_continues = self.model.continue_predictor(features)
        continue_loss = F.binary_cross_entropy(pred_continues, 1 - dones)
        
        # KL divergence between prior and posterior
        prior_dist = td.Independent(td.Normal(prior_states.mean, prior_states.std), 1)
        post_dist = td.Independent(td.Normal(posterior_states.mean, posterior_states.std), 1)
        kl_div = td.kl_divergence(post_dist, prior_dist).mean()
        
        # Free bits: don't penalize KL below threshold
        kl_loss = torch.clamp(kl_div - self.kl_free, min=0) * self.kl_scale
        
        # Total loss
        model_loss = recon_loss + reward_loss + continue_loss + kl_loss
        
        # Optimize
        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(self.model_params, self.grad_clip)
        self.model_optimizer.step()
        
        return {
            'recon_loss': recon_loss.item(),
            'reward_loss': reward_loss.item(),
            'continue_loss': continue_loss.item(),
            'kl_div': kl_div.item(),
            'model_loss': model_loss.item(),
        }
    
    def _train_behavior(self, obs: torch.Tensor, actions: torch.Tensor) -> Dict[str, float]:
        """
        Train actor and critic on imagined trajectories.
        
        1. Start from real states (from encoder)
        2. Imagine future using learned world model
        3. Compute value targets using lambda returns
        4. Update critic to predict values
        5. Update actor to maximize values
        """
        B, T = obs.shape[:2]
        
        # Get initial states from real observations (detached from world model)
        with torch.no_grad():
            state = self.model.initial_state(B, self.device)
            # Process first few steps to get meaningful state
            for t in range(min(T, 5)):
                _, posterior = self.model.observe_step(obs[:, t], actions[:, t], state)
                state = posterior
        
        # Imagine trajectory
        imagined_features, imagined_rewards, imagined_continues = \
            self.model.imagine_trajectory(state, self.imagine_horizon)
        
        # Compute value targets using TD(lambda)
        with torch.no_grad():
            values = self.model.critic(imagined_features)
            
            # Lambda returns
            returns = torch.zeros_like(imagined_rewards)
            last_return = values[:, -1]
            
            for t in reversed(range(self.imagine_horizon)):
                returns[:, t] = imagined_rewards[:, t] + \
                               self.gamma * imagined_continues[:, t] * \
                               ((1 - self.lambda_) * values[:, t] + self.lambda_ * last_return)
                last_return = returns[:, t]
        
        # Critic loss: predict returns
        pred_values = self.model.critic(imagined_features.detach())
        critic_loss = F.mse_loss(pred_values, returns)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()
        
        # Actor loss: maximize values
        # Re-imagine to get gradients through actor
        state_detached = RSSMState(
            mean=state.mean.detach(),
            std=state.std.detach(),
            stoch=state.stoch.detach(),
            deter=state.deter.detach(),
        )
        
        actor_features = []
        current_state = state_detached
        
        for _ in range(self.imagine_horizon):
            card, pos = self.model.actor(current_state.features)
            action = torch.zeros(B, 45, device=self.device)
            action_idx = card * 9 + pos
            action.scatter_(1, action_idx.unsqueeze(1), 1)
            
            # Use straight-through estimator for discrete actions
            current_state = self.model.imagine_step(action, current_state)
            actor_features.append(current_state.features)
        
        actor_features = torch.stack(actor_features, dim=1)
        actor_values = self.model.critic(actor_features)
        
        # Policy gradient (maximize value)
        actor_loss = -actor_values.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.model.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'imagined_reward': imagined_rewards.mean().item(),
            'imagined_value': values.mean().item(),
        }
    
    def save(self, path: str):
        """Save model and optimizer states."""
        torch.save({
            'model': self.model.state_dict(),
            'model_optimizer': self.model_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'train_steps': self.train_steps,
        }, path)
    
    def load(self, path: str):
        """Load model and optimizer states."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model_optimizer.load_state_dict(checkpoint['model_optimizer'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.train_steps = checkpoint['train_steps']
