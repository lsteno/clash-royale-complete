"""
DreamerV3 Training Algorithm for Clash Royale.
Implements world model learning and actor-critic policy optimization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import time

from .dreamer_model import DreamerV3Model, DreamerConfig, RSSMState, get_feat
from ..utils.device import get_device


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Replay buffer
    buffer_size: int = 1_000_000
    batch_size: int = 16
    batch_length: int = 50  # Sequence length
    
    # Learning rates
    model_lr: float = 1e-4
    actor_lr: float = 3e-5
    critic_lr: float = 3e-5
    
    # Loss weights
    kl_scale: float = 1.0
    kl_free: float = 1.0  # Free nats
    reward_scale: float = 1.0
    continue_scale: float = 1.0
    
    # Imagination
    horizon: int = 15
    discount: float = 0.99
    lambda_: float = 0.95
    
    # Optimization
    grad_clip: float = 100.0
    warmup_steps: int = 0
    
    # Training schedule
    train_every: int = 16  # Train every N environment steps
    train_steps: int = 1   # Gradient steps per train call
    prefill: int = 1000    # Steps before training starts


class ReplayBuffer:
    """
    Sequential replay buffer for DreamerV3.
    Stores transitions and samples sequences.
    """
    
    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], action_size: int):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_size = action_size
        
        # Storage
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_size), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.idx = 0
        self.size = 0
        self.episode_starts = [0]
    
    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, done: bool):
        """Add single transition."""
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = float(done)
        
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
        if done:
            self.episode_starts.append(self.idx)
            # Keep only recent episode starts
            if len(self.episode_starts) > 1000:
                self.episode_starts = self.episode_starts[-500:]
    
    def sample(self, batch_size: int, seq_length: int) -> Dict[str, torch.Tensor]:
        """
        Sample batch of sequences.
        
        Returns:
            Dict with keys: obs, actions, rewards, dones
            Each has shape (seq_length, batch_size, ...)
        """
        # Sample valid starting indices
        valid_starts = []
        for _ in range(batch_size * 10):  # Oversample then filter
            idx = np.random.randint(0, self.size - seq_length)
            # Check no episode boundary in sequence
            if not np.any(self.dones[idx:idx + seq_length - 1]):
                valid_starts.append(idx)
            if len(valid_starts) >= batch_size:
                break
        
        if len(valid_starts) < batch_size:
            # Fall back to random sampling
            valid_starts = np.random.randint(0, max(1, self.size - seq_length), batch_size)
        else:
            valid_starts = valid_starts[:batch_size]
        
        # Gather sequences
        obs_batch = np.stack([self.obs[i:i+seq_length] for i in valid_starts], axis=1)
        action_batch = np.stack([self.actions[i:i+seq_length] for i in valid_starts], axis=1)
        reward_batch = np.stack([self.rewards[i:i+seq_length] for i in valid_starts], axis=1)
        done_batch = np.stack([self.dones[i:i+seq_length] for i in valid_starts], axis=1)
        
        return {
            "obs": torch.from_numpy(obs_batch),
            "actions": torch.from_numpy(action_batch),
            "rewards": torch.from_numpy(reward_batch),
            "dones": torch.from_numpy(done_batch),
        }
    
    def __len__(self):
        return self.size


class DreamerV3Agent:
    """
    DreamerV3 agent for Clash Royale.
    Handles training and inference.
    """
    
    def __init__(self, 
                 model_config: Optional[DreamerConfig] = None,
                 train_config: Optional[TrainingConfig] = None,
                 device: torch.device = None):
        
        self.model_config = model_config or DreamerConfig()
        self.train_config = train_config or TrainingConfig()
        self.device = device or get_device()
        
        # Model
        self.model = DreamerV3Model(self.model_config).to(self.device)
        
        # Optimizers
        self.model_optimizer = torch.optim.Adam(
            list(self.model.encoder.parameters()) +
            list(self.model.decoder.parameters()) +
            list(self.model.rssm.parameters()) +
            list(self.model.reward_model.parameters()) +
            list(self.model.continue_model.parameters()),
            lr=self.train_config.model_lr
        )
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor.parameters(),
            lr=self.train_config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.model.critic.parameters(),
            lr=self.train_config.critic_lr
        )
        
        # Replay buffer
        obs_shape = (
            self.model_config.obs_channels,
            self.model_config.obs_height,
            self.model_config.obs_width
        )
        action_size = self.model_config.num_cards + 2
        self.replay_buffer = ReplayBuffer(
            self.train_config.buffer_size, obs_shape, action_size
        )
        
        # State tracking
        self._state: Optional[RSSMState] = None
        self._prev_action: Optional[torch.Tensor] = None
        self.step_count = 0
        self.train_count = 0
        
        # Metrics
        self.metrics = {
            "model_loss": [],
            "actor_loss": [],
            "critic_loss": [],
            "kl_loss": [],
            "reward_loss": [],
        }
    
    def reset(self):
        """Reset agent state for new episode."""
        self._state = self.model.rssm.initial_state(1, self.device)
        self._prev_action = torch.zeros(1, self.model_config.num_cards + 2, device=self.device)
    
    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, Tuple[float, float]]:
        """
        Select action given observation.
        
        Args:
            obs: State grid (channels, height, width)
            deterministic: If True, take mode of distributions
            
        Returns:
            card: Card index (0-4)
            pos: (x, y) normalized position
        """
        if self._state is None:
            self.reset()
        
        # Encode observation
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        embed = self.model.encoder(obs_t)
        
        # Update state
        _, self._state = self.model.rssm.observe_step(
            self._state, self._prev_action, embed
        )
        
        # Sample action
        if deterministic:
            card_dist, pos_dist = self.model.actor(self._state)
            card = card_dist.probs.argmax(dim=-1)
            pos = pos_dist.mean
        else:
            card, pos, _ = self.model.actor.sample(self._state)
        
        # Update prev action
        self._prev_action = self.model.encode_action(card, pos)
        
        card_int = card.item()
        pos_tuple = (pos[0, 0].item(), pos[0, 1].item())
        
        return card_int, pos_tuple
    
    def observe(self, obs: np.ndarray, action: np.ndarray, reward: float, done: bool):
        """
        Store transition in replay buffer.
        
        Args:
            obs: State grid
            action: Encoded action array
            reward: Received reward
            done: Episode done flag
        """
        self.replay_buffer.add(obs, action, reward, done)
        self.step_count += 1
        
        if done:
            self.reset()
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step.
        
        Returns:
            Dict of training metrics
        """
        if len(self.replay_buffer) < self.train_config.prefill:
            return {}
        
        # Sample batch
        batch = self.replay_buffer.sample(
            self.train_config.batch_size,
            self.train_config.batch_length
        )
        
        # Move to device
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        dones = batch["dones"].to(self.device)
        
        # Train world model
        model_loss, model_metrics = self._train_world_model(obs, actions, rewards, dones)
        
        # Train actor-critic via imagination
        actor_loss, critic_loss, ac_metrics = self._train_actor_critic(obs, actions)
        
        self.train_count += 1
        
        metrics = {**model_metrics, **ac_metrics}
        for k, v in metrics.items():
            self.metrics.setdefault(k, []).append(v)
        
        return metrics
    
    def _train_world_model(self, obs, actions, rewards, dones) -> Tuple[torch.Tensor, Dict]:
        """Train encoder, RSSM, decoder, reward, and continue models."""
        seq_len, batch_size = obs.shape[:2]
        
        # Encode observations
        obs_flat = obs.view(-1, *obs.shape[2:])
        embed_flat = self.model.encoder(obs_flat)
        embed = embed_flat.view(seq_len, batch_size, -1)
        
        # Roll out RSSM
        state = self.model.rssm.initial_state(batch_size, self.device)
        priors, posteriors = [], []
        
        for t in range(seq_len):
            action_t = actions[t] if t > 0 else torch.zeros_like(actions[0])
            prior, posterior = self.model.rssm.observe_step(state, action_t, embed[t])
            priors.append(prior)
            posteriors.append(posterior)
            state = posterior
        
        # Stack states
        def stack_states(states):
            return RSSMState(
                stoch=torch.stack([s.stoch for s in states]),
                deter=torch.stack([s.deter for s in states]),
                mean=torch.stack([s.mean for s in states]),
                std=torch.stack([s.std for s in states]),
            )
        
        prior_states = stack_states(priors)
        posterior_states = stack_states(posteriors)
        
        # Compute losses
        # KL divergence
        prior_dist = td.Independent(td.Normal(prior_states.mean, prior_states.std), 1)
        posterior_dist = td.Independent(td.Normal(posterior_states.mean, posterior_states.std), 1)
        kl_loss = td.kl_divergence(posterior_dist, prior_dist)
        kl_loss = torch.clamp(kl_loss - self.train_config.kl_free, min=0).mean()
        
        # Reconstruction loss
        feat = get_feat(posterior_states)
        feat_flat = feat.view(-1, feat.shape[-1])
        
        # Create state for decoder
        decoder_state = RSSMState(
            stoch=posterior_states.stoch.view(-1, posterior_states.stoch.shape[-1]),
            deter=posterior_states.deter.view(-1, posterior_states.deter.shape[-1]),
            mean=posterior_states.mean.view(-1, posterior_states.mean.shape[-1]),
            std=posterior_states.std.view(-1, posterior_states.std.shape[-1]),
        )
        
        recon = self.model.decoder(decoder_state)
        recon_loss = F.mse_loss(recon, obs_flat)
        
        # Reward prediction loss
        reward_pred = self.model.reward_model(decoder_state)
        reward_loss = F.mse_loss(reward_pred, rewards.view(-1))
        
        # Continue prediction loss
        continue_pred = self.model.continue_model(decoder_state)
        continue_loss = F.binary_cross_entropy(continue_pred, 1 - dones.view(-1))
        
        # Total model loss
        model_loss = (
            self.train_config.kl_scale * kl_loss +
            recon_loss +
            self.train_config.reward_scale * reward_loss +
            self.train_config.continue_scale * continue_loss
        )
        
        # Optimize
        self.model_optimizer.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.train_config.grad_clip
        )
        self.model_optimizer.step()
        
        metrics = {
            "model_loss": model_loss.item(),
            "kl_loss": kl_loss.item(),
            "recon_loss": recon_loss.item(),
            "reward_loss": reward_loss.item(),
            "continue_loss": continue_loss.item(),
        }
        
        return model_loss, metrics
    
    def _train_actor_critic(self, obs, actions) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Train actor and critic via imagined rollouts."""
        batch_size = obs.shape[1]
        
        # Get starting states from real data
        with torch.no_grad():
            obs_first = obs[0]
            embed = self.model.encoder(obs_first)
            state = self.model.rssm.initial_state(batch_size, self.device)
            _, state = self.model.rssm.observe_step(state, actions[0], embed)
        
        # Imagine forward
        imagined_states = [state]
        imagined_actions = []
        imagined_log_probs = []
        
        for _ in range(self.train_config.horizon):
            card, pos, log_prob = self.model.actor.sample(state)
            action = self.model.encode_action(card, pos)
            state = self.model.rssm.imagine_step(state, action)
            
            imagined_states.append(state)
            imagined_actions.append(action)
            imagined_log_probs.append(log_prob)
        
        # Compute values and rewards
        values = [self.model.critic(s) for s in imagined_states]
        rewards = [self.model.reward_model(s) for s in imagined_states[1:]]
        continues = [self.model.continue_model(s) for s in imagined_states[1:]]
        
        # Compute lambda returns
        returns = self._compute_lambda_returns(
            rewards, values[1:], continues,
            self.train_config.discount, self.train_config.lambda_
        )
        
        # Critic loss
        critic_loss = sum(
            F.mse_loss(values[i], returns[i].detach())
            for i in range(len(returns))
        ) / len(returns)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(
            self.model.critic.parameters(), self.train_config.grad_clip
        )
        self.critic_optimizer.step()
        
        # Actor loss (maximize returns)
        actor_loss = -sum(
            (returns[i] - values[i].detach()) * imagined_log_probs[i]
            for i in range(len(imagined_log_probs))
        ).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.actor.parameters(), self.train_config.grad_clip
        )
        self.actor_optimizer.step()
        
        metrics = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "mean_return": sum(r.mean().item() for r in returns) / len(returns),
        }
        
        return actor_loss, critic_loss, metrics
    
    def _compute_lambda_returns(self, rewards, values, continues, discount, lambda_):
        """Compute GAE-style lambda returns."""
        returns = []
        last_value = values[-1]
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = (1 - lambda_) * values[t + 1] + lambda_ * returns[0]
            
            ret = rewards[t] + discount * continues[t] * next_value
            returns.insert(0, ret)
        
        return returns
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_optimizer": self.model_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "step_count": self.step_count,
            "train_count": self.train_count,
            "model_config": self.model_config,
            "train_config": self.train_config,
        }, path)
        print(f"Saved checkpoint to {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model_optimizer.load_state_dict(checkpoint["model_optimizer"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.step_count = checkpoint["step_count"]
        self.train_count = checkpoint["train_count"]
        print(f"Loaded checkpoint from {path}")


if __name__ == "__main__":
    # Test agent
    print("Testing DreamerV3 Agent...")
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Create agent
    agent = DreamerV3Agent(device=device)
    
    # Simulate some transitions
    print("\nSimulating transitions...")
    agent.reset()
    
    for i in range(100):
        obs = np.random.randn(8, 18, 32).astype(np.float32)
        card, pos = agent.act(obs)
        
        # Create action array
        action = np.zeros(7, dtype=np.float32)
        action[:5] = np.eye(5)[card]
        action[5:7] = pos
        
        reward = np.random.randn()
        done = i % 50 == 49
        
        agent.observe(obs, action, reward, done)
    
    print(f"Buffer size: {len(agent.replay_buffer)}")
    
    # Test training step
    print("\nTesting training step...")
    metrics = agent.train_step()
    print(f"Training metrics: {metrics}")
    
    print("\nAgent test complete!")
