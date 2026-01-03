#!/usr/bin/env python3
"""
Offline DreamerV3 Training for Clash Royale - Following KataCR Approach.

This follows the researchers' approach:
- Train offline on pre-collected expert replay data  
- Use KataCR's YOLOv8 perception (kept as-is)
- Replace StARformer with DreamerV3 world model + actor-critic
- Validate live on device

The key insight: Train a world model on offline data, then do behavior learning
in imagination. This is more sample efficient than their sequence modeling.

Usage:
    python train_offline.py --dataset golem_ai
    python train_offline.py --dataset fast_hog_2.6
"""
import argparse
import lzma
import sys
from pathlib import Path
from io import BytesIO
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agent.dreamer_v3 import DreamerV3, create_dreamer_model

# ============== Constants ==============
# From KataCR - grid representation
GRID_ROWS = 32  # Arena grid rows
GRID_COLS = 18  # Arena grid cols

# From KataCR - state feature dimensions
# zij ∈ R^15: category, sub-faction, health, additional features
STATE_FEATURES = 15

# Action space: 4 cards × 9 positions + no-op = 37 (we use 45 for buffer)
N_CARDS = 4
N_POSITIONS = 9
ACTION_DIM = N_CARDS * N_POSITIONS + 1  # +1 for no-op


class OfflineReplayDataset(Dataset):
    """
    Dataset that loads KataCR replay format and converts to DreamerV3 format.
    
    KataCR format per frame:
    - state: {time, unit_infos, cards, elixir}
    - action: {xy, card_id}  
    - reward: float
    
    DreamerV3 format:
    - observation: (C, H, W) visual or grid features
    - action: one-hot vector
    - reward: scalar
    - done: boolean
    """
    
    def __init__(
        self, 
        path_dataset: str, 
        seq_length: int = 50,
        use_grid_state: bool = True,  # True=grid features, False=visual
    ):
        self.seq_length = seq_length
        self.use_grid_state = use_grid_state
        
        self.episodes = []
        self.episode_lengths = []
        self.sample_indices = []  # (episode_idx, start_frame)
        
        self._load_all_replays(path_dataset)
        self._build_sample_indices()
        
    def _load_all_replays(self, path_dataset: str):
        """Load all .xz replay files from directory."""
        path = Path(path_dataset)
        if path.is_dir():
            files = sorted(path.glob("*.xz"))
        else:
            files = [path]
        
        print(f"Loading {len(files)} replay files...")
        for f in tqdm(files, desc="Loading replays"):
            episode = self._load_single_replay(f)
            if episode is not None and len(episode['reward']) >= self.seq_length:
                self.episodes.append(episode)
                self.episode_lengths.append(len(episode['reward']))
        
        total_frames = sum(self.episode_lengths)
        print(f"Loaded {len(self.episodes)} episodes, {total_frames} total frames")
        
    def _load_single_replay(self, path: Path) -> dict:
        """Load a single replay file and convert to our format."""
        try:
            with lzma.open(str(path), 'rb') as f:
                data = np.load(BytesIO(f.read()), allow_pickle=True).item()
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
        
        states = data['state']
        actions = data['action']
        rewards = data['reward']
        
        # Find last action frame (clip terminal padding)
        n = len(actions)
        for i in range(n - 1, -1, -1):
            if actions[i]['card_id'] != 0 or actions[i]['xy'] is not None:
                break
        n = i + 1
        
        if n < self.seq_length:
            return None
        
        # Convert to arrays
        episode = {
            'observation': [],  # Grid or visual features
            'action': [],       # One-hot action vectors
            'reward': rewards[:n].astype(np.float32),
            'done': np.zeros(n, dtype=np.float32),
        }
        episode['done'][-1] = 1.0
        
        for i in range(n):
            # Convert state to observation
            obs = self._state_to_observation(states[i])
            episode['observation'].append(obs)
            
            # Convert action to one-hot
            action = self._action_to_onehot(actions[i])
            episode['action'].append(action)
        
        episode['observation'] = np.stack(episode['observation'])
        episode['action'] = np.stack(episode['action'])
        
        return episode
    
    def _state_to_observation(self, state: dict) -> np.ndarray:
        """
        Convert KataCR state to observation tensor.
        
        If use_grid_state: Returns (15, 32, 18) grid features like KataCR
        Otherwise: Returns (3, 256, 128) pseudo-visual (for image-based model)
        """
        if self.use_grid_state:
            # Build grid representation like KataCR paper
            # zij ∈ R^15: category (one-hot 10), belong (2), health (1), extra (2)
            grid = np.zeros((STATE_FEATURES, GRID_ROWS, GRID_COLS), dtype=np.float32)
            
            unit_infos = state.get('unit_infos', [])
            if unit_infos is None:
                unit_infos = []
                
            for unit in unit_infos:
                if unit is None:
                    continue
                xy = unit.get('xy')
                if xy is None:
                    continue
                
                # xy is numpy array [x, y]
                if isinstance(xy, np.ndarray):
                    x, y = float(xy[0]), float(xy[1])
                else:
                    x, y = float(xy[0]), float(xy[1])
                    
                # Convert xy to grid position
                col = int(np.clip(x, 0, GRID_COLS - 1))
                row = int(np.clip(y, 0, GRID_ROWS - 1))
                
                # Set features
                cls = unit.get('cls')
                bel = unit.get('bel', 0)  # 0=enemy, 1=friendly
                
                if cls is None:
                    cls = 0
                
                # Category one-hot (simplified to 10 categories)
                cat_idx = min(int(cls), 9)
                grid[cat_idx, row, col] = 1.0
                
                # Belong feature
                if bel is None:
                    bel = 0
                grid[10, row, col] = 1.0 if bel == 0 else 0.0  # enemy
                grid[11, row, col] = 1.0 if bel == 1 else 0.0  # friendly
                
                # Health (normalized) - bar1 can be None or scalar
                bar1 = unit.get('bar1')
                if bar1 is not None:
                    try:
                        health = float(bar1)
                        grid[12, row, col] = np.clip(health, 0, 1)
                    except (TypeError, ValueError):
                        pass  # Skip if not convertible
                
            # Add elixir as global feature (broadcast to all cells)
            elixir = state.get('elixir')
            if elixir is not None:
                try:
                    grid[13, :, :] = float(elixir) / 10.0
                except (TypeError, ValueError):
                    pass
            
            # Add time as global feature
            time = state.get('time')
            if time is not None:
                try:
                    grid[14, :, :] = float(time) / 180.0  # Normalize by max game time
                except (TypeError, ValueError):
                    pass
            
            return grid
        else:
            # For visual model, return placeholder
            # In real use, you'd decode actual frames
            return np.zeros((3, 256, 128), dtype=np.float32)
    
    def _action_to_onehot(self, action: dict) -> np.ndarray:
        """Convert KataCR action format to one-hot vector."""
        onehot = np.zeros(ACTION_DIM, dtype=np.float32)
        
        card_id = action.get('card_id', 0)
        xy = action.get('xy')
        
        if card_id == 0 or xy is None:
            # No action
            onehot[0] = 1.0
        else:
            # Map card_id (1-4) and xy to discrete position
            card_idx = min(card_id - 1, N_CARDS - 1)  # 0-3
            
            # Discretize xy to 9 positions (3x3 grid over arena)
            x, y = xy
            # Arena coords: x in [0, 18], y in [0, 32]
            pos_x = int(np.clip(x / 6, 0, 2))  # 0, 1, 2
            pos_y = int(np.clip(y / 11, 0, 2))  # 0, 1, 2
            pos_idx = pos_y * 3 + pos_x  # 0-8
            
            action_idx = 1 + card_idx * N_POSITIONS + pos_idx
            onehot[action_idx] = 1.0
        
        return onehot
    
    def _build_sample_indices(self):
        """Build list of valid (episode_idx, start_frame) pairs."""
        self.sample_indices = []
        self.sample_weights = []
        
        for ep_idx, ep_len in enumerate(self.episode_lengths):
            episode = self.episodes[ep_idx]
            actions = episode['action']
            
            for start in range(ep_len - self.seq_length + 1):
                self.sample_indices.append((ep_idx, start))
                
                # Weight by whether sequence contains actions (like KataCR resampling)
                end = start + self.seq_length
                has_action = (actions[start:end, 0] < 0.5).any()  # action[0]=1 means no-op
                weight = 10.0 if has_action else 1.0
                self.sample_weights.append(weight)
        
        self.sample_weights = np.array(self.sample_weights)
        self.sample_weights /= self.sample_weights.sum()
        
        print(f"Built {len(self.sample_indices)} training sequences")
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        ep_idx, start = self.sample_indices[idx]
        episode = self.episodes[ep_idx]
        end = start + self.seq_length
        
        return {
            'observation': torch.from_numpy(episode['observation'][start:end]),
            'action': torch.from_numpy(episode['action'][start:end]),
            'reward': torch.from_numpy(episode['reward'][start:end]),
            'done': torch.from_numpy(episode['done'][start:end]),
        }


class DreamerV3Offline(nn.Module):
    """
    DreamerV3 adapted for offline training on grid-based state representation.
    
    Changes from visual DreamerV3:
    - Input: (15, 32, 18) grid features instead of (3, H, W) images
    - Encoder: Conv on grid instead of images
    - Otherwise same RSSM world model + actor-critic
    """
    
    def __init__(
        self,
        state_channels: int = STATE_FEATURES,
        state_height: int = GRID_ROWS,
        state_width: int = GRID_COLS,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 256,
        deter_dim: int = 256,
        stoch_dim: int = 32,
        stoch_classes: int = 32,
        device: torch.device = None,
    ):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        
        # Encoder: Grid features -> latent
        self.encoder = nn.Sequential(
            nn.Conv2d(state_channels, 64, 3, stride=2, padding=1),  # 32x18 -> 16x9
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 16x9 -> 8x5
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 8x5 -> 4x3
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 3, hidden_dim),
            nn.ReLU(),
        )
        
        # RSSM components
        embed_dim = hidden_dim
        
        # Sequence model (GRU)
        self.rnn = nn.GRUCell(stoch_dim * stoch_classes + action_dim, deter_dim)
        
        # Prior (predict stochastic from deterministic)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stoch_dim * stoch_classes),
        )
        
        # Posterior (predict stochastic from deterministic + observation)
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stoch_dim * stoch_classes),
        )
        
        # Decoder: latent -> grid reconstruction
        # Target: (15, 32, 18) - need to carefully compute transpose conv output sizes
        # Start from (256, 4, 3) -> (15, 32, 18)
        self.decoder = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim * stoch_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256 * 4 * 3),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 3)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # (4,3) -> (8,6)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # (8,6) -> (16,12)
            nn.ReLU(),
            nn.ConvTranspose2d(64, state_channels, (4, 3), stride=2, padding=1),  # (16,12) -> (32,23)
            # Then crop to (32, 18)
        )
        
        # We'll handle the crop in forward pass
        self._target_h = state_height
        self._target_w = state_width
        
        # Reward predictor
        self.reward_net = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim * stoch_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Continue predictor (done prediction)
        self.continue_net = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim * stoch_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim * stoch_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        # Critic (value function)
        self.critic = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim * stoch_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.to(self.device)
    
    def get_features(self, deter, stoch):
        """Concatenate deterministic and stochastic states."""
        stoch_flat = stoch.reshape(stoch.shape[0], -1)
        return torch.cat([deter, stoch_flat], dim=-1)
    
    def decode(self, features):
        """Decode features to observation space, handling size mismatch."""
        x = self.decoder(features)
        # Crop to target size
        return x[:, :, :self._target_h, :self._target_w]
    
    def sample_stochastic(self, logits):
        """Sample from categorical distribution with straight-through gradients."""
        B = logits.shape[0]
        logits = logits.reshape(B, self.stoch_dim, self.stoch_classes)
        
        # Gumbel-softmax for differentiable sampling
        dist = torch.distributions.OneHotCategorical(logits=logits)
        sample = dist.sample()
        
        # Straight-through gradient
        probs = F.softmax(logits, dim=-1)
        sample = sample + probs - probs.detach()
        
        return sample.reshape(B, self.stoch_dim * self.stoch_classes)
    
    def initial_state(self, batch_size):
        """Get initial RSSM state."""
        deter = torch.zeros(batch_size, self.deter_dim, device=self.device)
        stoch = torch.zeros(batch_size, self.stoch_dim * self.stoch_classes, device=self.device)
        return deter, stoch
    
    def observe_step(self, obs, action, prev_state):
        """
        One step of observation in RSSM.
        
        Returns (prior, posterior) distributions.
        """
        prev_deter, prev_stoch = prev_state
        
        # Encode observation
        embed = self.encoder(obs)
        
        # Sequence model step
        rnn_input = torch.cat([prev_stoch, action], dim=-1)
        deter = self.rnn(rnn_input, prev_deter)
        
        # Prior
        prior_logits = self.prior_net(deter)
        prior_stoch = self.sample_stochastic(prior_logits)
        
        # Posterior (uses observation)
        post_input = torch.cat([deter, embed], dim=-1)
        post_logits = self.posterior_net(post_input)
        post_stoch = self.sample_stochastic(post_logits)
        
        return (deter, prior_logits, prior_stoch), (deter, post_logits, post_stoch)
    
    def imagine_step(self, action, prev_state):
        """One step of imagination (no observation)."""
        prev_deter, prev_stoch = prev_state
        
        rnn_input = torch.cat([prev_stoch, action], dim=-1)
        deter = self.rnn(rnn_input, prev_deter)
        
        prior_logits = self.prior_net(deter)
        prior_stoch = self.sample_stochastic(prior_logits)
        
        return deter, prior_stoch


def train_world_model(model, batch, optimizer):
    """Train world model on a batch of sequences."""
    obs = batch['observation'].to(model.device)  # (B, T, C, H, W)
    actions = batch['action'].to(model.device)   # (B, T, A)
    rewards = batch['reward'].to(model.device)   # (B, T)
    dones = batch['done'].to(model.device)       # (B, T)
    
    B, T = obs.shape[:2]
    
    # Initialize state
    deter, stoch = model.initial_state(B)
    
    # Losses
    recon_loss = 0
    reward_loss = 0
    continue_loss = 0
    kl_loss = 0
    
    for t in range(T):
        prior, posterior = model.observe_step(obs[:, t], actions[:, t], (deter, stoch))
        
        prior_deter, prior_logits, prior_stoch = prior
        post_deter, post_logits, post_stoch = posterior
        
        # Use posterior for predictions
        features = model.get_features(post_deter, post_stoch)
        
        # Reconstruction loss
        recon = model.decode(features)
        recon_loss += F.mse_loss(recon, obs[:, t])
        
        # Reward prediction loss
        pred_reward = model.reward_net(features).squeeze(-1)
        reward_loss += F.mse_loss(pred_reward, rewards[:, t])
        
        # Continue prediction loss
        pred_continue = model.continue_net(features).squeeze(-1)
        continue_loss += F.binary_cross_entropy_with_logits(pred_continue, 1 - dones[:, t])
        
        # KL divergence (posterior vs prior)
        prior_logits_reshaped = prior_logits.reshape(B, model.stoch_dim, model.stoch_classes)
        post_logits_reshaped = post_logits.reshape(B, model.stoch_dim, model.stoch_classes)
        
        prior_dist = torch.distributions.Categorical(logits=prior_logits_reshaped)
        post_dist = torch.distributions.Categorical(logits=post_logits_reshaped)
        kl = torch.distributions.kl_divergence(post_dist, prior_dist).sum(dim=-1).mean()
        kl_loss += kl
        
        # Update state for next step
        deter = post_deter
        stoch = post_stoch
    
    # Average over time
    loss = (recon_loss + reward_loss + continue_loss + 0.1 * kl_loss) / T
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
    optimizer.step()
    
    return {
        'loss': loss.item(),
        'recon_loss': (recon_loss / T).item(),
        'reward_loss': (reward_loss / T).item(),
        'kl_loss': (kl_loss / T).item(),
    }


def train_behavior(model, batch, actor_optimizer, critic_optimizer, horizon=15, gamma=0.99, lam=0.95):
    """Train actor-critic on imagined trajectories."""
    obs = batch['observation'].to(model.device)
    actions = batch['action'].to(model.device)
    
    B, T = obs.shape[:2]
    
    # Get initial posterior states from real data
    with torch.no_grad():
        deter, stoch = model.initial_state(B)
        for t in range(min(T, 10)):  # Warm up on real data
            _, posterior = model.observe_step(obs[:, t], actions[:, t], (deter, stoch))
            deter, _, stoch = posterior
    
    # Imagine trajectories
    imagined_features = []
    imagined_rewards = []
    imagined_continues = []
    imagined_actions = []
    
    for h in range(horizon):
        features = model.get_features(deter, stoch)
        imagined_features.append(features)
        
        # Sample action from actor
        action_logits = model.actor(features)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        action = F.one_hot(action_dist.sample(), model.action_dim).float()
        imagined_actions.append(action)
        
        # Imagine next state
        deter, stoch = model.imagine_step(action, (deter, stoch))
        
        # Predict rewards and continues
        next_features = model.get_features(deter, stoch)
        reward = model.reward_net(next_features).squeeze(-1)
        cont = torch.sigmoid(model.continue_net(next_features).squeeze(-1))
        
        imagined_rewards.append(reward)
        imagined_continues.append(cont)
    
    # Stack
    features = torch.stack(imagined_features, dim=1)  # (B, H, F)
    rewards = torch.stack(imagined_rewards, dim=1)     # (B, H)
    continues = torch.stack(imagined_continues, dim=1) # (B, H)
    
    # Compute values
    values = model.critic(features).squeeze(-1)  # (B, H)
    
    # Compute lambda returns
    returns = torch.zeros_like(rewards)
    last_value = values[:, -1]
    
    for t in reversed(range(horizon)):
        if t == horizon - 1:
            next_value = last_value
        else:
            next_value = values[:, t + 1]
        
        delta = rewards[:, t] + gamma * continues[:, t] * next_value - values[:, t]
        returns[:, t] = values[:, t] + delta  # Simple TD target
    
    # Critic loss
    critic_loss = F.mse_loss(values, returns.detach())
    
    critic_optimizer.zero_grad()
    critic_loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 100.0)
    critic_optimizer.step()
    
    # Actor loss (policy gradient)
    advantages = (returns - values).detach()
    
    # Recompute action log probs
    actor_loss = 0
    for h in range(horizon):
        action_logits = model.actor(features[:, h])
        action_dist = torch.distributions.Categorical(logits=action_logits)
        action = imagined_actions[h].argmax(dim=-1)
        log_prob = action_dist.log_prob(action)
        actor_loss -= (log_prob * advantages[:, h]).mean()
    
    actor_loss /= horizon
    
    actor_optimizer.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 100.0)
    actor_optimizer.step()
    
    return {
        'actor_loss': actor_loss.item(),
        'critic_loss': critic_loss.item(),
        'mean_value': values.mean().item(),
        'mean_return': returns.mean().item(),
    }


def main(args):
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║           Offline DreamerV3 Training for Clash Royale                        ║
║                    Following KataCR Approach                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Train on pre-collected expert data (like KataCR)                            ║
║  But use DreamerV3 world model instead of StARformer                         ║
║                                                                              ║
║  Phase 1: Train world model on offline data                                  ║
║  Phase 2: Train actor-critic in imagination                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✓ Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Dataset
    dataset_path = Path(__file__).parent.parent / "Clash-Royale-Replay-Dataset" / args.dataset
    print(f"\nLoading dataset from: {dataset_path}")
    
    dataset = OfflineReplayDataset(
        str(dataset_path),
        seq_length=args.seq_length,
        use_grid_state=True,
    )
    
    # Use weighted sampling like KataCR
    sampler = WeightedRandomSampler(
        dataset.sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        drop_last=True,
    )
    
    # Model
    model = DreamerV3Offline(device=device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizers
    world_optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) +
        list(model.rnn.parameters()) +
        list(model.prior_net.parameters()) +
        list(model.posterior_net.parameters()) +
        list(model.decoder.parameters()) +
        list(model.reward_net.parameters()) +
        list(model.continue_net.parameters()),
        lr=args.lr,
    )
    actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=args.lr * 0.1)
    critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=args.lr)
    
    # Logging
    log_dir = Path("logs") / f"offline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logging to: {log_dir}")
    
    # Training loop
    global_step = 0
    
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        
        world_losses = defaultdict(list)
        behavior_losses = defaultdict(list)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        for batch in pbar:
            # Train world model
            wm_metrics = train_world_model(model, batch, world_optimizer)
            for k, v in wm_metrics.items():
                world_losses[k].append(v)
            
            # Train behavior (after some world model training)
            if global_step > args.prefill:
                bh_metrics = train_behavior(
                    model, batch, actor_optimizer, critic_optimizer,
                    horizon=args.horizon, gamma=args.gamma,
                )
                for k, v in bh_metrics.items():
                    behavior_losses[k].append(v)
            
            global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'wm_loss': f"{wm_metrics['loss']:.4f}",
                'kl': f"{wm_metrics['kl_loss']:.4f}",
            })
        
        # Epoch summary
        print(f"\nWorld Model - Loss: {np.mean(world_losses['loss']):.4f}, "
              f"Recon: {np.mean(world_losses['recon_loss']):.4f}, "
              f"KL: {np.mean(world_losses['kl_loss']):.4f}")
        
        if behavior_losses:
            print(f"Behavior - Actor: {np.mean(behavior_losses['actor_loss']):.4f}, "
                  f"Critic: {np.mean(behavior_losses['critic_loss']):.4f}, "
                  f"Value: {np.mean(behavior_losses['mean_value']):.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = log_dir / f"checkpoint_epoch{epoch + 1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'world_optimizer': world_optimizer.state_dict(),
                'actor_optimizer': actor_optimizer.state_dict(),
                'critic_optimizer': critic_optimizer.state_dict(),
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
    
    # Final save
    final_path = log_dir / "checkpoint_final.pt"
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
    }, final_path)
    print(f"\n✓ Training complete! Final model: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline DreamerV3 Training")
    parser.add_argument("--dataset", type=str, default="fast_hog_2.6",
                        help="Dataset folder name (fast_hog_2.6 or golem_ai)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-length", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--prefill", type=int, default=100,
                        help="Steps before behavior learning starts")
    parser.add_argument("--save-every", type=int, default=5)
    
    args = parser.parse_args()
    main(args)
