#!/usr/bin/env python3
"""
Training script matching the KataCR paper approach:
- Continuous action space: (delay, pos_x, pos_y, card_select)
- Action frame resampling (~25x for action frames)
- Delay prediction: model learns WHEN to act
- Gaussian output for continuous, categorical for card

Paper: "Playing Non-Embedded Card-Based Games with Reinforcement Learning"
"""
import argparse
import lzma
import os
import sys
from pathlib import Path
from io import BytesIO
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

# Constants from paper
GRID_ROWS = 32
GRID_COLS = 18
STATE_FEATURES = 15
N_CARDS = 4
MAX_DELAY = 20  # Maximum frames until next action (paper: Tdelay = 20)


class PaperDataset(Dataset):
    """
    Dataset matching paper's approach:
    - Computes adelay for each frame (frames until next action)
    - Stores continuous (pos_x, pos_y) and card_select
    - Resamples action frames ~25x more than non-action frames
    """
    
    def __init__(self, path_dataset: str, seq_length: int = 50, cache_dir: str = None):
        self.seq_length = seq_length
        self.cache_path = Path(cache_dir or f".cache_paper_{Path(path_dataset).name}")
        
        if self._load_cache():
            print(f"✓ Loaded from cache: {self.cache_path}")
        else:
            print(f"Building dataset from scratch...")
            self._build_dataset(path_dataset)
            self._save_cache()
        
        self._compute_sampling_weights()
        print(f"Total frames: {len(self.observations)}, Action frames: {self.n_action_frames}")
    
    def _load_cache(self) -> bool:
        if not (self.cache_path / "observations.npy").exists():
            return False
        try:
            self.observations = np.load(self.cache_path / "observations.npy", mmap_mode='r')
            self.delays = np.load(self.cache_path / "delays.npy", mmap_mode='r')
            self.positions = np.load(self.cache_path / "positions.npy", mmap_mode='r')
            self.cards = np.load(self.cache_path / "cards.npy", mmap_mode='r')
            self.is_action = np.load(self.cache_path / "is_action.npy", mmap_mode='r')
            self.rewards = np.load(self.cache_path / "rewards.npy", mmap_mode='r')
            self.dones = np.load(self.cache_path / "dones.npy", mmap_mode='r')
            self.episode_starts = np.load(self.cache_path / "episode_starts.npy")
            return True
        except:
            return False
    
    def _save_cache(self):
        self.cache_path.mkdir(parents=True, exist_ok=True)
        np.save(self.cache_path / "observations.npy", self.observations)
        np.save(self.cache_path / "delays.npy", self.delays)
        np.save(self.cache_path / "positions.npy", self.positions)
        np.save(self.cache_path / "cards.npy", self.cards)
        np.save(self.cache_path / "is_action.npy", self.is_action)
        np.save(self.cache_path / "rewards.npy", self.rewards)
        np.save(self.cache_path / "dones.npy", self.dones)
        np.save(self.cache_path / "episode_starts.npy", self.episode_starts)
        print(f"✓ Saved cache to: {self.cache_path}")
    
    def _build_dataset(self, path_dataset: str):
        path = Path(path_dataset)
        files = sorted(path.glob("*.xz")) if path.is_dir() else [path]
        
        all_obs, all_delay, all_pos, all_card = [], [], [], []
        all_is_action, all_rew, all_done = [], [], []
        episode_starts = [0]
        
        print(f"Loading {len(files)} replay files...")
        for f in tqdm(files, desc="Loading"):
            try:
                with lzma.open(str(f), 'rb') as fp:
                    data = np.load(BytesIO(fp.read()), allow_pickle=True).item()
                
                states = data['state']
                actions = data['action']
                rewards = data['reward']
                
                # Find actual episode length
                n = len(actions)
                for i in range(n - 1, -1, -1):
                    if actions[i]['card_id'] != 0 or actions[i]['xy'] is not None:
                        break
                n = min(i + 1, len(states), len(rewards))
                
                if n < self.seq_length:
                    continue
                
                # Process episode
                obs = np.stack([self._state_to_obs(states[i]) for i in range(n)])
                
                # Find action frames and compute delays
                action_frames = []
                positions = np.zeros((n, 2), dtype=np.float32)
                cards = np.zeros(n, dtype=np.int64)
                is_action = np.zeros(n, dtype=np.float32)
                
                for i in range(n):
                    act = actions[i]
                    card_id = act.get('card_id', 0)
                    xy = act.get('xy')
                    
                    if card_id > 0 and xy is not None:
                        action_frames.append(i)
                        is_action[i] = 1.0
                        # Normalize position to [0, 1]
                        positions[i, 0] = float(xy[0]) / GRID_COLS
                        positions[i, 1] = float(xy[1]) / GRID_ROWS
                        cards[i] = min(card_id, N_CARDS)  # 1-4
                
                # Compute delay for each frame (frames until next action)
                delays = np.full(n, MAX_DELAY, dtype=np.float32)
                action_idx = 0
                for i in range(n):
                    if action_idx < len(action_frames):
                        next_action = action_frames[action_idx]
                        if i == next_action:
                            delays[i] = 0
                            action_idx += 1
                        else:
                            delays[i] = min(next_action - i, MAX_DELAY)
                    else:
                        delays[i] = MAX_DELAY
                
                rew = rewards[:n].astype(np.float32)
                done = np.zeros(n, dtype=np.float32)
                done[-1] = 1.0
                
                all_obs.append(obs)
                all_delay.append(delays)
                all_pos.append(positions)
                all_card.append(cards)
                all_is_action.append(is_action)
                all_rew.append(rew)
                all_done.append(done)
                episode_starts.append(episode_starts[-1] + n)
                
            except Exception as e:
                print(f"Error loading {f}: {e}")
                continue
        
        self.observations = np.concatenate(all_obs, axis=0)
        self.delays = np.concatenate(all_delay, axis=0)
        self.positions = np.concatenate(all_pos, axis=0)
        self.cards = np.concatenate(all_card, axis=0)
        self.is_action = np.concatenate(all_is_action, axis=0)
        self.rewards = np.concatenate(all_rew, axis=0)
        self.dones = np.concatenate(all_done, axis=0)
        self.episode_starts = np.array(episode_starts)
    
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
            
            x, y = float(xy[0]), float(xy[1])
            col = int(np.clip(x, 0, GRID_COLS - 1))
            row = int(np.clip(y, 0, GRID_ROWS - 1))
            
            cls = int(unit.get('cls') or 0)
            bel = int(unit.get('bel') or 0)
            
            grid[min(cls, 9), row, col] = 1.0
            grid[10, row, col] = float(bel == 0)
            grid[11, row, col] = float(bel == 1)
            
            bar1 = unit.get('bar1')
            if bar1 is not None:
                try:
                    grid[12, row, col] = np.clip(float(bar1), 0, 1)
                except:
                    pass
        
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
    
    def _compute_sampling_weights(self):
        """
        Paper Eq (8): Resample action frames more frequently.
        Action frames get ~25x higher weight than non-action frames.
        """
        self.n_action_frames = int(self.is_action.sum())
        n_total = len(self.is_action)
        ra = self.n_action_frames / n_total  # action frame ratio
        
        print(f"Action frame ratio: {ra*100:.2f}%")
        
        # Build sample indices and weights
        self.sample_indices = []
        self.sample_weights = []
        
        for ep_idx in range(len(self.episode_starts) - 1):
            start = self.episode_starts[ep_idx]
            end = self.episode_starts[ep_idx + 1]
            ep_len = end - start
            
            # Find action frames in this episode
            ep_is_action = self.is_action[start:end]
            action_indices = np.where(ep_is_action > 0.5)[0]
            
            for frame_start in range(ep_len - self.seq_length + 1):
                global_start = start + frame_start
                global_end = global_start + self.seq_length
                
                # Paper resampling: weight based on proximity to action frame
                seq_is_action = self.is_action[global_start:global_end]
                has_action = seq_is_action.any()
                
                # Higher weight for sequences containing action frames
                if has_action:
                    # Even higher weight if sequence ENDS with action
                    if seq_is_action[-5:].any():
                        weight = 25.0
                    else:
                        weight = 10.0
                else:
                    weight = 1.0
                
                self.sample_indices.append(global_start)
                self.sample_weights.append(weight)
        
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
            'delay': torch.from_numpy(self.delays[start:end].copy()),
            'position': torch.from_numpy(self.positions[start:end].copy()),
            'card': torch.from_numpy(self.cards[start:end].copy()),
            'is_action': torch.from_numpy(self.is_action[start:end].copy()),
            'reward': torch.from_numpy(self.rewards[start:end].copy()),
            'done': torch.from_numpy(self.dones[start:end].copy()),
        }


class PaperDreamer(nn.Module):
    """
    DreamerV3 adapted to paper's action space:
    - Predicts: delay (continuous), position (continuous x,y), card (categorical)
    - Uses Gaussian distributions for continuous outputs
    - Learns WHEN to act via delay prediction
    """
    
    def __init__(
        self,
        obs_shape=(STATE_FEATURES, GRID_ROWS, GRID_COLS),
        hidden_dim=256,
        deter_dim=256,
        stoch_dim=32,
        device=None,
    ):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.obs_flat = np.prod(obs_shape)
        
        # For action embedding (delay + pos_x + pos_y + card_onehot)
        self.action_embed_dim = 3 + N_CARDS + 1  # delay, x, y, card(5)
        
        # Encoder
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
        self.rnn = nn.GRUCell(stoch_dim + self.action_embed_dim, deter_dim)
        
        self.prior = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stoch_dim * 2),
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
        
        # === Actor heads (paper approach) ===
        feature_dim = deter_dim + stoch_dim
        
        # Delay head: predicts frames until action (0 = act now)
        self.delay_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # mean, logstd
        )
        
        # Position head: predicts (x, y) normalized to [0,1]
        self.position_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # mean_x, mean_y, logstd_x, logstd_y
        )
        
        # Card head: categorical over 5 options (0=no card, 1-4=cards)
        self.card_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N_CARDS + 1),
        )
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
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
    
    def embed_action(self, delay, position, card):
        """Embed action into vector for RNN input."""
        B = delay.shape[0]
        
        # Normalize delay to [0, 1]
        delay_norm = delay.float() / MAX_DELAY
        
        # Card one-hot (0-4)
        card_onehot = F.one_hot(card.long(), N_CARDS + 1).float()
        
        # Concatenate: [delay, pos_x, pos_y, card_onehot]
        action_embed = torch.cat([
            delay_norm.unsqueeze(-1),
            position,
            card_onehot,
        ], dim=-1)
        
        return action_embed
    
    def _sample_stoch(self, stats):
        mean, logstd = stats.chunk(2, dim=-1)
        std = F.softplus(logstd) + 0.1
        dist = torch.distributions.Normal(mean, std)
        return dist.rsample(), mean, std
    
    def observe_step(self, obs, prev_action_embed, state):
        """One step with observation."""
        deter, stoch = state
        
        embed = self.encoder(obs)
        rnn_in = torch.cat([stoch, prev_action_embed], dim=-1)
        deter = self.rnn(rnn_in, deter)
        
        prior_stats = self.prior(deter)
        prior_stoch, prior_mean, prior_std = self._sample_stoch(prior_stats)
        
        post_in = torch.cat([deter, embed], dim=-1)
        post_stats = self.posterior(post_in)
        post_stoch, post_mean, post_std = self._sample_stoch(post_stats)
        
        return {
            'deter': deter,
            'prior': (prior_mean, prior_std),
            'posterior': (post_mean, post_std),
            'stoch': post_stoch,
        }
    
    def get_features(self, deter, stoch):
        return torch.cat([deter, stoch], dim=-1)
    
    def predict_action(self, features, sample=True):
        """
        Predict action from features.
        Returns: delay, position (x,y), card, log_probs
        """
        # Delay prediction
        delay_stats = self.delay_head(features)
        delay_mean, delay_logstd = delay_stats.chunk(2, dim=-1)
        delay_std = F.softplus(delay_logstd) + 0.1
        
        if sample:
            delay_dist = torch.distributions.Normal(delay_mean.squeeze(-1), delay_std.squeeze(-1))
            delay = delay_dist.rsample()
            delay_logprob = delay_dist.log_prob(delay)
        else:
            delay = delay_mean.squeeze(-1)
            delay_logprob = torch.zeros_like(delay)
        
        # Clamp delay to valid range
        delay = delay.clamp(0, MAX_DELAY)
        
        # Position prediction
        pos_stats = self.position_head(features)
        pos_mean, pos_logstd = pos_stats[:, :2], pos_stats[:, 2:]
        pos_std = F.softplus(pos_logstd) + 0.05
        
        if sample:
            pos_dist = torch.distributions.Normal(pos_mean, pos_std)
            position = pos_dist.rsample()
            pos_logprob = pos_dist.log_prob(position).sum(dim=-1)
        else:
            position = pos_mean
            pos_logprob = torch.zeros(features.shape[0], device=features.device)
        
        # Clamp position to [0, 1]
        position = position.clamp(0, 1)
        
        # Card prediction (categorical)
        card_logits = self.card_head(features)
        
        if sample:
            card_dist = torch.distributions.Categorical(logits=card_logits)
            card = card_dist.sample()
            card_logprob = card_dist.log_prob(card)
        else:
            card = card_logits.argmax(dim=-1)
            card_logprob = torch.zeros(features.shape[0], device=features.device)
        
        total_logprob = delay_logprob + pos_logprob + card_logprob
        
        return delay, position, card, total_logprob


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train one epoch with paper's approach."""
    model.train()
    metrics = {'wm_loss': 0, 'delay_loss': 0, 'pos_loss': 0, 'card_loss': 0}
    n_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        obs = batch['observation'].to(device)
        delay = batch['delay'].to(device)
        position = batch['position'].to(device)
        card = batch['card'].to(device)
        is_action = batch['is_action'].to(device)
        reward = batch['reward'].to(device)
        
        B, T = obs.shape[:2]
        
        # Initialize state
        deter, stoch = model.initial_state(B)
        prev_action_embed = torch.zeros(B, model.action_embed_dim, device=device)
        
        total_loss = 0
        wm_loss_sum = 0
        delay_loss_sum = 0
        pos_loss_sum = 0
        card_loss_sum = 0
        
        for t in range(T):
            # Observe step
            out = model.observe_step(obs[:, t], prev_action_embed, (deter, stoch))
            deter = out['deter']
            stoch = out['stoch']
            
            # World model losses
            features = model.get_features(deter, stoch)
            
            # Reconstruction loss
            obs_pred = model.decoder(features).view(B, -1)
            obs_target = obs[:, t].view(B, -1)
            recon_loss = F.mse_loss(obs_pred, obs_target)
            
            # KL loss
            prior_mean, prior_std = out['prior']
            post_mean, post_std = out['posterior']
            kl = torch.distributions.kl_divergence(
                torch.distributions.Normal(post_mean, post_std),
                torch.distributions.Normal(prior_mean, prior_std)
            ).sum(dim=-1).mean()
            
            # Reward prediction
            reward_pred = model.reward_head(features).squeeze(-1)
            reward_loss = F.mse_loss(reward_pred, reward[:, t])
            
            wm_loss = recon_loss + 0.1 * kl + reward_loss
            wm_loss_sum += wm_loss.item()
            
            # === Action prediction losses ===
            # Delay loss (all frames)
            delay_stats = model.delay_head(features)
            delay_mean = delay_stats[:, 0]
            delay_loss = F.mse_loss(delay_mean, delay[:, t])
            delay_loss_sum += delay_loss.item()
            
            # Position and card loss (only on action frames, weighted)
            action_mask = is_action[:, t]
            if action_mask.sum() > 0:
                # Position loss
                pos_stats = model.position_head(features)
                pos_mean = pos_stats[:, :2]
                pos_loss = F.mse_loss(pos_mean[action_mask > 0], position[:, t][action_mask > 0])
                pos_loss_sum += pos_loss.item()
                
                # Card loss
                card_logits = model.card_head(features)
                card_loss = F.cross_entropy(card_logits[action_mask > 0], card[:, t][action_mask > 0].long())
                card_loss_sum += card_loss.item()
            else:
                pos_loss = torch.tensor(0.0, device=device)
                card_loss = torch.tensor(0.0, device=device)
            
            total_loss += wm_loss + delay_loss + pos_loss + card_loss
            
            # Update action embedding for next step
            prev_action_embed = model.embed_action(delay[:, t], position[:, t], card[:, t])
        
        # Backprop
        optimizer.zero_grad()
        (total_loss / T).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
        optimizer.step()
        
        metrics['wm_loss'] += wm_loss_sum / T
        metrics['delay_loss'] += delay_loss_sum / T
        metrics['pos_loss'] += pos_loss_sum / T
        metrics['card_loss'] += card_loss_sum / T
        n_batches += 1
        
        pbar.set_postfix({
            'WM': f"{wm_loss_sum/T:.4f}",
            'Delay': f"{delay_loss_sum/T:.3f}",
            'Pos': f"{pos_loss_sum/T:.3f}",
            'Card': f"{card_loss_sum/T:.3f}",
        })
    
    return {k: v / n_batches for k, v in metrics.items()}


def main(args):
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║        Paper-Style Training: Continuous Actions + Delay Prediction           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Key differences from previous approach:                                     ║
║  • Model predicts DELAY (frames until action) - learns WHEN to act           ║
║  • Continuous position output (x, y) instead of discrete grid                ║
║  • Categorical card selection (0-4)                                          ║
║  • Action frames resampled ~25x more frequently                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    
    # Dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        dataset_path = Path(__file__).parent.parent / "Clash-Royale-Replay-Dataset" / args.dataset
    
    print(f"Loading dataset: {dataset_path}")
    dataset = PaperDataset(str(dataset_path), seq_length=args.seq_length)
    
    # Weighted sampler
    sampler = WeightedRandomSampler(
        weights=dataset.sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
    )
    
    # Model
    model = PaperDreamer(device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training
    log_dir = Path("logs") / f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(model, dataloader, optimizer, device, epoch)
        
        print(f"\nEpoch {epoch}: WM={metrics['wm_loss']:.4f}, "
              f"Delay={metrics['delay_loss']:.3f}, "
              f"Pos={metrics['pos_loss']:.3f}, "
              f"Card={metrics['card_loss']:.3f}")
        
        # Save checkpoint
        total_loss = sum(metrics.values())
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }
        
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(checkpoint, log_dir / "best.pt")
            print(f"  ✓ Saved best model")
        
        if epoch % 5 == 0:
            torch.save(checkpoint, log_dir / f"epoch_{epoch}.pt")
    
    torch.save(checkpoint, log_dir / "final.pt")
    print(f"\n✓ Training complete! Models saved to: {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fast_hog_2.6", help="Dataset name or path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-length", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    args = parser.parse_args()
    main(args)
