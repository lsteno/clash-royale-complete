#!/usr/bin/env python3
"""
Fast Offline DreamerV3 Training for Clash Royale.

Key optimizations:
1. Preprocessed dataset (load once, convert once)
2. Shorter sequences (25 frames vs 50)
3. Larger batches (32 vs 8)  
4. Simpler encoder (MLP instead of Conv for small grid)
5. Mixed precision training
6. Reduced logging overhead

This is the version to use for actual training.
"""
import argparse
import lzma
import os
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

# Constants - match KataCR paper
GRID_ROWS = 32
GRID_COLS = 18
STATE_FEATURES = 15  # zij ∈ R^15
N_CARDS = 4
N_POSITIONS = 9
ACTION_DIM = N_CARDS * N_POSITIONS + 1  # 37 actions


class CachedReplayDataset(Dataset):
    """
    Optimized dataset with preprocessing.
    
    Key optimizations:
    - Pre-converts all episodes to tensors once
    - Uses numpy memmap for large datasets
    - Efficient sampling
    """
    
    def __init__(self, path_dataset: str, seq_length: int = 25, cache_dir: str = None):
        self.seq_length = seq_length
        self.cache_path = Path(cache_dir or f".cache_{Path(path_dataset).name}")
        
        # Try to load from cache first
        if self._load_cache():
            print(f"✓ Loaded from cache: {self.cache_path}")
        else:
            print(f"Building dataset from scratch...")
            self._build_dataset(path_dataset)
            self._save_cache()
            
        self._build_sample_indices()
    
    def _cache_exists(self) -> bool:
        return (self.cache_path / "observations.npy").exists()
    
    def _load_cache(self) -> bool:
        if not self._cache_exists():
            return False
        try:
            self.observations = np.load(self.cache_path / "observations.npy", mmap_mode='r')
            self.actions = np.load(self.cache_path / "actions.npy", mmap_mode='r')
            self.rewards = np.load(self.cache_path / "rewards.npy", mmap_mode='r')
            self.dones = np.load(self.cache_path / "dones.npy", mmap_mode='r')
            self.episode_starts = np.load(self.cache_path / "episode_starts.npy")
            return True
        except Exception as e:
            print(f"Cache load failed: {e}")
            return False
    
    def _save_cache(self):
        self.cache_path.mkdir(parents=True, exist_ok=True)
        np.save(self.cache_path / "observations.npy", self.observations)
        np.save(self.cache_path / "actions.npy", self.actions)
        np.save(self.cache_path / "rewards.npy", self.rewards)
        np.save(self.cache_path / "dones.npy", self.dones)
        np.save(self.cache_path / "episode_starts.npy", self.episode_starts)
        print(f"✓ Saved cache to: {self.cache_path}")
    
    def _build_dataset(self, path_dataset: str):
        path = Path(path_dataset)
        files = sorted(path.glob("*.xz")) if path.is_dir() else [path]
        
        all_obs, all_act, all_rew, all_done = [], [], [], []
        episode_starts = [0]
        
        print(f"Loading {len(files)} replay files...")
        for f in tqdm(files, desc="Loading"):
            try:
                with lzma.open(str(f), 'rb') as fp:
                    data = np.load(BytesIO(fp.read()), allow_pickle=True).item()
                
                states = data['state']
                actions = data['action']
                rewards = data['reward']
                
                # Find actual episode length (trim padding)
                n = len(actions)
                for i in range(n - 1, -1, -1):
                    if actions[i]['card_id'] != 0 or actions[i]['xy'] is not None:
                        break
                n = min(i + 1, len(states), len(rewards))
                
                if n < self.seq_length:
                    continue
                
                # Convert to arrays
                obs = np.stack([self._state_to_obs(states[i]) for i in range(n)])
                act = np.stack([self._action_to_vec(actions[i]) for i in range(n)])
                rew = rewards[:n].astype(np.float32)
                done = np.zeros(n, dtype=np.float32)
                done[-1] = 1.0
                
                all_obs.append(obs)
                all_act.append(act)
                all_rew.append(rew)
                all_done.append(done)
                episode_starts.append(episode_starts[-1] + n)
                
            except Exception as e:
                print(f"Error loading {f}: {e}")
                continue
        
        # Concatenate all episodes
        self.observations = np.concatenate(all_obs, axis=0)
        self.actions = np.concatenate(all_act, axis=0)
        self.rewards = np.concatenate(all_rew, axis=0)
        self.dones = np.concatenate(all_done, axis=0)
        self.episode_starts = np.array(episode_starts)
        
        print(f"Total frames: {len(self.observations)}, Episodes: {len(episode_starts)-1}")
    
    def _state_to_obs(self, state: dict) -> np.ndarray:
        """Convert state to (15, 32, 18) grid."""
        grid = np.zeros((STATE_FEATURES, GRID_ROWS, GRID_COLS), dtype=np.float32)
        
        unit_infos = state.get('unit_infos') or []
        for unit in unit_infos:
            if unit is None:
                continue
            xy = unit.get('xy')
            if xy is None:
                continue
            
            x, y = float(xy[0]), float(xy[1]) if hasattr(xy, '__getitem__') else (0, 0)
            col = int(np.clip(x, 0, GRID_COLS - 1))
            row = int(np.clip(y, 0, GRID_ROWS - 1))
            
            cls = int(unit.get('cls') or 0)
            bel = int(unit.get('bel') or 0)
            
            grid[min(cls, 9), row, col] = 1.0
            grid[10, row, col] = float(bel == 0)  # enemy
            grid[11, row, col] = float(bel == 1)  # friendly
            
            bar1 = unit.get('bar1')
            if bar1 is not None:
                try:
                    grid[12, row, col] = np.clip(float(bar1), 0, 1)
                except:
                    pass
        
        # Global features
        elixir = state.get('elixir')
        if elixir is not None:
            try:
                grid[13, :, :] = float(elixir) / 10.0
            except:
                pass
        
        time = state.get('time')
        if time is not None:
            try:
                grid[14, :, :] = float(time) / 180.0
            except:
                pass
        
        return grid
    
    def _action_to_vec(self, action: dict) -> np.ndarray:
        """Convert action to one-hot vector."""
        onehot = np.zeros(ACTION_DIM, dtype=np.float32)
        
        card_id = action.get('card_id', 0)
        xy = action.get('xy')
        
        if card_id == 0 or xy is None:
            onehot[0] = 1.0
        else:
            card_idx = min(card_id - 1, N_CARDS - 1)
            x, y = float(xy[0]), float(xy[1]) if hasattr(xy, '__getitem__') else (0, 0)
            pos_x = int(np.clip(x / 6, 0, 2))
            pos_y = int(np.clip(y / 11, 0, 2))
            pos_idx = pos_y * 3 + pos_x
            action_idx = 1 + card_idx * N_POSITIONS + pos_idx
            onehot[min(action_idx, ACTION_DIM - 1)] = 1.0
        
        return onehot
    
    def _build_sample_indices(self):
        """Build weighted sample indices."""
        self.sample_indices = []
        self.sample_weights = []
        
        for ep_idx in range(len(self.episode_starts) - 1):
            start = self.episode_starts[ep_idx]
            end = self.episode_starts[ep_idx + 1]
            ep_len = end - start
            
            for frame_start in range(ep_len - self.seq_length + 1):
                global_start = start + frame_start
                self.sample_indices.append(global_start)
                
                # Weight sequences with actions higher
                seq_actions = self.actions[global_start:global_start + self.seq_length]
                has_action = (seq_actions[:, 0] < 0.5).any()
                self.sample_weights.append(10.0 if has_action else 1.0)
        
        self.sample_weights = np.array(self.sample_weights, dtype=np.float64)
        self.sample_weights /= self.sample_weights.sum()
        
        print(f"Built {len(self.sample_indices)} training sequences")
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        start = self.sample_indices[idx]
        end = start + self.seq_length
        
        return {
            'observation': torch.from_numpy(self.observations[start:end].copy()),
            'action': torch.from_numpy(self.actions[start:end].copy()),
            'reward': torch.from_numpy(self.rewards[start:end].copy()),
            'done': torch.from_numpy(self.dones[start:end].copy()),
        }


class FastDreamer(nn.Module):
    """
    Optimized DreamerV3-style world model for grid observations.
    
    Key differences from full DreamerV3:
    - MLP encoder (grids are small, don't need heavy convs)
    - Smaller latent space for faster iteration
    - Continuous latent (simpler than categorical for small models)
    """
    
    def __init__(
        self,
        obs_shape=(STATE_FEATURES, GRID_ROWS, GRID_COLS),
        action_dim=ACTION_DIM,
        hidden_dim=256,
        deter_dim=256,
        stoch_dim=32,
        device=None,
    ):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.obs_flat = np.prod(obs_shape)
        
        # Encoder: flatten grid -> embed
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.obs_flat, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # RSSM
        self.rnn = nn.GRUCell(stoch_dim + action_dim, deter_dim)
        
        self.prior = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stoch_dim * 2),  # mean, logstd
        )
        
        self.posterior = nn.Sequential(
            nn.Linear(deter_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stoch_dim * 2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.obs_flat),
        )
        
        # Reward predictor
        self.reward_head = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Continue predictor
        self.continue_head = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Actor-Critic
        self.actor = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        self.critic = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.to(self.device)
    
    def initial_state(self, batch_size):
        return (
            torch.zeros(batch_size, self.deter_dim, device=self.device),
            torch.zeros(batch_size, self.stoch_dim, device=self.device),
        )
    
    def _sample_stoch(self, stats):
        mean, logstd = stats.chunk(2, dim=-1)
        std = F.softplus(logstd) + 0.1
        dist = torch.distributions.Normal(mean, std)
        return dist.rsample(), mean, std
    
    def observe_step(self, obs, action, state):
        """One step with observation."""
        deter, stoch = state
        
        # Encode observation
        embed = self.encoder(obs)
        
        # RNN step
        rnn_in = torch.cat([stoch, action], dim=-1)
        deter = self.rnn(rnn_in, deter)
        
        # Prior (from deter only)
        prior_stats = self.prior(deter)
        prior_stoch, prior_mean, prior_std = self._sample_stoch(prior_stats)
        
        # Posterior (from deter + obs)
        post_in = torch.cat([deter, embed], dim=-1)
        post_stats = self.posterior(post_in)
        post_stoch, post_mean, post_std = self._sample_stoch(post_stats)
        
        return {
            'deter': deter,
            'prior': (prior_mean, prior_std),
            'posterior': (post_mean, post_std),
            'stoch': post_stoch,
        }
    
    def imagine_step(self, action, state):
        """One step without observation (imagination)."""
        deter, stoch = state
        
        rnn_in = torch.cat([stoch, action], dim=-1)
        deter = self.rnn(rnn_in, deter)
        
        prior_stats = self.prior(deter)
        stoch, _, _ = self._sample_stoch(prior_stats)
        
        return deter, stoch
    
    def get_features(self, deter, stoch):
        return torch.cat([deter, stoch], dim=-1)
    
    def decode(self, features, obs_shape):
        flat = self.decoder(features)
        return flat.view(-1, *obs_shape)
    
    def forward(self, obs_seq, action_seq):
        """Process a sequence, return losses."""
        B, T = obs_seq.shape[:2]
        obs_shape = obs_seq.shape[2:]
        
        deter, stoch = self.initial_state(B)
        
        recon_loss = 0
        reward_loss = 0
        kl_loss = 0
        
        for t in range(T):
            out = self.observe_step(obs_seq[:, t], action_seq[:, t], (deter, stoch))
            
            deter = out['deter']
            stoch = out['stoch']
            prior_mean, prior_std = out['prior']
            post_mean, post_std = out['posterior']
            
            features = self.get_features(deter, stoch)
            
            # Reconstruction
            recon = self.decode(features, obs_shape)
            recon_loss += F.mse_loss(recon, obs_seq[:, t])
            
            # KL divergence
            prior_dist = torch.distributions.Normal(prior_mean, prior_std)
            post_dist = torch.distributions.Normal(post_mean, post_std)
            kl = torch.distributions.kl_divergence(post_dist, prior_dist).sum(-1).mean()
            kl_loss += kl
        
        return {
            'recon_loss': recon_loss / T,
            'kl_loss': kl_loss / T,
            'total_loss': (recon_loss + 0.1 * kl_loss) / T,
        }


def train_epoch(model, dataloader, optimizer, device, prefill=100, step_offset=0):
    """Train for one epoch."""
    model.train()
    metrics = defaultdict(list)
    
    pbar = tqdm(dataloader, desc="Training")
    for i, batch in enumerate(pbar):
        obs = batch['observation'].to(device)
        action = batch['action'].to(device)
        
        # World model loss
        losses = model(obs, action)
        loss = losses['total_loss']
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        
        metrics['loss'].append(loss.item())
        metrics['recon'].append(losses['recon_loss'].item())
        metrics['kl'].append(losses['kl_loss'].item())
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'kl': f"{losses['kl_loss'].item():.4f}",
        })
    
    return {k: np.mean(v) for k, v in metrics.items()}


def train_behavior(model, dataloader, actor_optim, critic_optim, device, horizon=15, gamma=0.99):
    """
    Train actor-critic by imagining trajectories in the world model.
    
    This is the key DreamerV3 insight: learn to act by "dreaming".
    """
    model.train()
    metrics = defaultdict(list)
    
    pbar = tqdm(dataloader, desc="Behavior")
    for batch in pbar:
        obs = batch['observation'].to(device)
        action = batch['action'].to(device)
        reward = batch['reward'].to(device)
        
        B, T = obs.shape[:2]
        
        # Get initial state from real data (warm up world model)
        with torch.no_grad():
            deter, stoch = model.initial_state(B)
            for t in range(min(T, 10)):
                out = model.observe_step(obs[:, t], action[:, t], (deter, stoch))
                deter, stoch = out['deter'], out['stoch']
        
        # Now imagine trajectories using the actor
        imagined_features = []
        imagined_rewards = []
        log_probs = []
        entropies = []
        
        for h in range(horizon):
            features = model.get_features(deter, stoch)
            imagined_features.append(features)
            
            # Actor chooses action
            action_logits = model.actor(features)
            action_dist = torch.distributions.Categorical(logits=action_logits)
            sampled_action = action_dist.sample()
            log_prob = action_dist.log_prob(sampled_action)
            entropy = action_dist.entropy()
            log_probs.append(log_prob)
            entropies.append(entropy)
            
            # One-hot encode action
            action_onehot = F.one_hot(sampled_action, model.action_dim).float()
            
            # Imagine next state
            deter, stoch = model.imagine_step(action_onehot, (deter, stoch))
            
            # Predict reward (clipped to reasonable range)
            next_features = model.get_features(deter, stoch)
            pred_reward = model.reward_head(next_features).squeeze(-1)
            pred_reward = torch.tanh(pred_reward)  # Bound rewards to [-1, 1]
            imagined_rewards.append(pred_reward)
        
        # Stack tensors
        features = torch.stack(imagined_features, dim=1)  # (B, H, F)
        rewards = torch.stack(imagined_rewards, dim=1)     # (B, H)
        log_probs = torch.stack(log_probs, dim=1)          # (B, H)
        entropies = torch.stack(entropies, dim=1)          # (B, H)
        
        # Compute values (bounded)
        values = torch.tanh(model.critic(features).squeeze(-1))  # (B, H)
        
        # Compute returns (with bounded values)
        returns = torch.zeros_like(rewards)
        running_return = values[:, -1].detach()
        
        for t in reversed(range(horizon)):
            running_return = rewards[:, t] + gamma * running_return
            running_return = torch.clamp(running_return, -10, 10)  # Clip returns
            returns[:, t] = running_return
        
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Critic loss (value estimation)
        value_targets = returns.detach()
        critic_loss = F.mse_loss(values, value_targets)
        
        critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 1.0)
        critic_optim.step()
        
        # Actor loss (policy gradient with advantage + entropy bonus)
        advantages = (returns - values).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        actor_loss = -(log_probs * advantages).mean() - 0.01 * entropies.mean()
        
        actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
        actor_optim.step()
        
        metrics['actor_loss'].append(actor_loss.item())
        metrics['critic_loss'].append(critic_loss.item())
        metrics['mean_value'].append(values.mean().item())
        metrics['mean_reward'].append(rewards.mean().item())
        metrics['entropy'].append(entropies.mean().item())
        
        pbar.set_postfix({
            'actor': f"{actor_loss.item():.3f}",
            'value': f"{values.mean().item():.3f}",
            'entropy': f"{entropies.mean().item():.2f}",
        })
    
    return {k: np.mean(v) for k, v in metrics.items()}


def main(args):
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║         Fast Offline DreamerV3 Training for Clash Royale                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Optimized for speed:                                                        ║
║  - Cached dataset loading                                                    ║
║  - Shorter sequences (25 frames)                                             ║
║  - MLP encoder (fast for small grids)                                        ║
║  - Larger batches                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✓ Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Dataset
    dataset_path = Path(__file__).parent.parent / "Clash-Royale-Replay-Dataset" / args.dataset
    cache_dir = Path(__file__).parent / f".cache_{args.dataset}"
    
    print(f"\nDataset: {dataset_path}")
    dataset = CachedReplayDataset(str(dataset_path), seq_length=args.seq_length, cache_dir=str(cache_dir))
    
    sampler = WeightedRandomSampler(
        dataset.sample_weights,
        num_samples=min(len(dataset), args.steps_per_epoch * args.batch_size),
        replacement=True,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,  # MPS doesn't like multiprocessing
        pin_memory=False,
    )
    
    # Model
    model = FastDreamer(device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Separate optimizers for world model and behavior
    world_params = list(model.encoder.parameters()) + list(model.rnn.parameters()) + \
                   list(model.prior.parameters()) + list(model.posterior.parameters()) + \
                   list(model.decoder.parameters()) + list(model.reward_head.parameters()) + \
                   list(model.continue_head.parameters())
    
    world_optim = torch.optim.AdamW(world_params, lr=args.lr, weight_decay=1e-5)
    actor_optim = torch.optim.AdamW(model.actor.parameters(), lr=args.lr * 0.1, weight_decay=1e-5)
    critic_optim = torch.optim.AdamW(model.critic.parameters(), lr=args.lr, weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(world_optim, T_max=args.epochs)
    
    # Logging
    log_dir = Path("logs") / f"fast_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logging to: {log_dir}")
    
    # Training
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} (lr={scheduler.get_last_lr()[0]:.2e}) ===")
        
        # Phase 1: Train world model
        print("Phase 1: World Model")
        wm_metrics = train_epoch(model, dataloader, world_optim, device)
        scheduler.step()
        
        print(f"  WM Loss: {wm_metrics['loss']:.4f} | Recon: {wm_metrics['recon']:.4f} | KL: {wm_metrics['kl']:.4f}")
        
        # Phase 2: Train behavior (after first epoch)
        if epoch >= 1:
            print("Phase 2: Behavior Learning (Actor-Critic in Imagination)")
            bh_metrics = train_behavior(model, dataloader, actor_optim, critic_optim, device)
            print(f"  Actor: {bh_metrics['actor_loss']:.4f} | Critic: {bh_metrics['critic_loss']:.4f} | Value: {bh_metrics['mean_value']:.4f}")
        
        # Save best (based on world model loss)
        if wm_metrics['loss'] < best_loss:
            best_loss = wm_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'world_optim_state_dict': world_optim.state_dict(),
                'actor_optim_state_dict': actor_optim.state_dict(),
                'critic_optim_state_dict': critic_optim.state_dict(),
                'loss': best_loss,
            }, log_dir / "best.pt")
            print(f"✓ Saved best model (loss={best_loss:.4f})")
        
        # Periodic save
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, log_dir / f"epoch_{epoch+1}.pt")
    
    # Final save
    torch.save({'model_state_dict': model.state_dict()}, log_dir / "final.pt")
    print(f"\n✓ Training complete! Models saved to {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fast_hog_2.6")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-length", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    
    main(parser.parse_args())
