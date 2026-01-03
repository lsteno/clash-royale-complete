#!/usr/bin/env python3
"""
Online DreamerV3 Training with Multiple Parallel Emulators.

DreamerV3 approach:
1. Workers collect experience into shared replay buffer
2. Learner trains WORLD MODEL on replay sequences (reconstruction + KL)
3. Actor-critic trained by IMAGINING trajectories in the world model

This is more sample-efficient than PPO because:
- World model learns from ALL data
- Actor trains on imagined trajectories (free simulation)
- No need for on-policy data

Setup:
    emulator -avd Pixel_6 -port 5554 &
    emulator -avd Pixel_6_2 -port 5556 &
    emulator -avd Pixel_6_3 -port 5558 &
    
    python train_live.py --n-envs 3
"""
import argparse
import subprocess
import time
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from multiprocessing import Process, Queue, Event, Manager
from queue import Empty
import threading
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import cv2

# Constants
GRID_ROWS = 32
GRID_COLS = 18
STATE_FEATURES = 15
N_CARDS = 4
MAX_DELAY = 20

# Screen coordinates for 1080x2400
CARD_POSITIONS = [(270, 2220), (460, 2220), (650, 2220), (840, 2220)]
ARENA_LEFT, ARENA_RIGHT = 60, 1020
DEPLOY_TOP, DEPLOY_BOTTOM = 1200, 1800
ELIXIR_ROI = (200, 2300, 1050, 2400)


# =============================================================================
# Replay Buffer
# =============================================================================

class ReplayBuffer:
    """
    Stores experience for DreamerV3 training.
    Supports sampling sequences for world model training.
    """
    
    def __init__(self, capacity: int = 100000, seq_length: int = 50):
        self.capacity = capacity
        self.seq_length = seq_length
        
        # Storage
        self.observations = None
        self.actions = None  # (delay, pos_x, pos_y, card)
        self.rewards = None
        self.dones = None
        
        self.position = 0
        self.size = 0
        self.episode_starts = [0]
    
    def _init_storage(self):
        """Initialize storage arrays."""
        self.observations = np.zeros(
            (self.capacity, STATE_FEATURES, GRID_ROWS, GRID_COLS), 
            dtype=np.float32
        )
        self.actions = np.zeros((self.capacity, 4), dtype=np.float32)  # delay, x, y, card
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
    
    def add_episode(self, episode: List[Dict]):
        """Add an episode to the buffer."""
        if self.observations is None:
            self._init_storage()
        
        for step in episode:
            idx = self.position % self.capacity
            
            self.observations[idx] = step['obs']
            self.actions[idx] = [
                step['action']['delay'] / MAX_DELAY,
                step['action']['pos_x'],
                step['action']['pos_y'],
                step['action']['card'] / N_CARDS,
            ]
            self.rewards[idx] = step['reward']
            self.dones[idx] = float(step['done'])
            
            self.position += 1
            self.size = min(self.size + 1, self.capacity)
        
        # Track episode boundary
        self.episode_starts.append(self.position % self.capacity)
        if len(self.episode_starts) > 1000:
            self.episode_starts = self.episode_starts[-500:]
    
    def sample_sequences(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample sequences for world model training."""
        if self.size < self.seq_length + 1:
            return None
        
        # Sample random starting points
        batch_obs = []
        batch_act = []
        batch_rew = []
        batch_done = []
        
        for _ in range(batch_size):
            # Find valid starting point (not crossing episode boundary)
            max_start = self.size - self.seq_length
            if max_start <= 0:
                continue
            
            start = random.randint(0, max_start - 1)
            end = start + self.seq_length
            
            batch_obs.append(self.observations[start:end])
            batch_act.append(self.actions[start:end])
            batch_rew.append(self.rewards[start:end])
            batch_done.append(self.dones[start:end])
        
        if not batch_obs:
            return None
        
        return {
            'observation': torch.from_numpy(np.stack(batch_obs)),
            'action': torch.from_numpy(np.stack(batch_act)),
            'reward': torch.from_numpy(np.stack(batch_rew)),
            'done': torch.from_numpy(np.stack(batch_done)),
        }
    
    def __len__(self):
        return self.size


# =============================================================================
# DreamerV3 Model
# =============================================================================

class DreamerV3(nn.Module):
    """
    DreamerV3 world model with actor-critic.
    
    Components:
    - Encoder: obs -> embedding
    - RSSM: recurrent state-space model (deter + stoch states)
    - Decoder: latent -> obs reconstruction
    - Reward head: latent -> reward prediction
    - Actor: latent -> action distribution
    - Critic: latent -> value estimate
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
        self.hidden_dim = hidden_dim
        self.obs_flat = int(np.prod(obs_shape))
        self.action_dim = 4  # delay, x, y, card (continuous)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.obs_flat, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        
        # RSSM
        self.rnn = nn.GRUCell(stoch_dim + self.action_dim, deter_dim)
        
        self.prior = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stoch_dim * 2),
        )
        
        self.posterior = nn.Sequential(
            nn.Linear(deter_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stoch_dim * 2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.obs_flat),
        )
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Continue head (predicts episode continuation)
        self.continue_head = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Actor heads (continuous outputs)
        feature_dim = deter_dim + stoch_dim
        
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.action_dim * 2),  # mean + logstd
        )
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.to(self.device)
    
    def initial_state(self, batch_size):
        return (
            torch.zeros(batch_size, self.deter_dim, device=self.device),
            torch.zeros(batch_size, self.stoch_dim, device=self.device),
        )
    
    def get_features(self, deter, stoch):
        return torch.cat([deter, stoch], dim=-1)
    
    def _sample_stoch(self, stats):
        mean, logstd = stats.chunk(2, dim=-1)
        std = F.softplus(logstd) + 0.1
        dist = Normal(mean, std)
        sample = dist.rsample()
        return sample, mean, std
    
    def observe_step(self, obs, action, state):
        """Single step with observation (for posterior)."""
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
            'stoch': post_stoch,
            'prior': (prior_mean, prior_std),
            'posterior': (post_mean, post_std),
        }
    
    def imagine_step(self, action, state):
        """Single step WITHOUT observation (for imagination)."""
        deter, stoch = state
        
        # RNN step
        rnn_in = torch.cat([stoch, action], dim=-1)
        deter = self.rnn(rnn_in, deter)
        
        # Prior only (no observation)
        prior_stats = self.prior(deter)
        prior_stoch, _, _ = self._sample_stoch(prior_stats)
        
        return deter, prior_stoch
    
    def get_action(self, features, sample=True):
        """Get action from actor."""
        action_stats = self.actor(features)
        mean, logstd = action_stats.chunk(2, dim=-1)
        std = F.softplus(logstd) + 0.1
        
        if sample:
            dist = Normal(mean, std)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        else:
            action = mean
            log_prob = torch.zeros(features.shape[0], device=features.device)
        
        # Clamp actions to valid ranges
        action = torch.sigmoid(action)  # All in [0, 1]
        
        return action, log_prob, (mean, std)
    
    def get_value(self, features):
        return self.critic(features)


# =============================================================================
# Environment
# =============================================================================

class ClashEnv:
    """Single Clash Royale environment connected to one emulator."""
    
    def __init__(self, device_id: str = "emulator-5554"):
        self.device_id = device_id
        self.adb_prefix = ['adb', '-s', device_id]
        
        self.episode_reward = 0
        self.steps = 0
        self.in_battle = False
    
    def _adb_command(self, cmd: List[str], timeout: float = 2.0) -> Optional[bytes]:
        try:
            result = subprocess.run(
                self.adb_prefix + cmd,
                capture_output=True,
                timeout=timeout
            )
            return result.stdout if result.returncode == 0 else None
        except:
            return None
    
    def capture_screen(self) -> Optional[np.ndarray]:
        data = self._adb_command(['exec-out', 'screencap', '-p'])
        if data is None:
            return None
        try:
            img_array = np.frombuffer(data, dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except:
            return None
    
    def detect_elixir(self, img: np.ndarray) -> int:
        if img is None:
            return 0
        
        x1, y1, x2, y2 = ELIXIR_ROI
        if y2 > img.shape[0]:
            return 5
        
        elixir_bar = img[y1:y2, x1:x2]
        hsv = cv2.cvtColor(elixir_bar, cv2.COLOR_BGR2HSV)
        
        pink_lower = np.array([140, 50, 50])
        pink_upper = np.array([170, 255, 255])
        mask = cv2.inRange(hsv, pink_lower, pink_upper)
        
        fill_ratio = np.sum(mask > 0) / mask.size
        return min(int(fill_ratio * 10), 10)
    
    def detect_battle_state(self, img: np.ndarray) -> str:
        if img is None:
            return "menu"
        
        x1, y1, x2, y2 = ELIXIR_ROI
        if y2 > img.shape[0]:
            return "menu"
        
        elixir_bar = img[y1:y2, x1:x2]
        hsv = cv2.cvtColor(elixir_bar, cv2.COLOR_BGR2HSV)
        
        pink_lower = np.array([140, 50, 50])
        pink_upper = np.array([170, 255, 255])
        mask = cv2.inRange(hsv, pink_lower, pink_upper)
        
        if np.sum(mask > 0) / mask.size > 0.03:
            return "battle"
        
        # Check for end screen
        roi = img[700:950, 250:830]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        gold_lower = np.array([15, 100, 100])
        gold_upper = np.array([35, 255, 255])
        gold_mask = cv2.inRange(hsv_roi, gold_lower, gold_upper)
        
        if np.sum(gold_mask > 0) / gold_mask.size > 0.2:
            return "end"
        
        return "menu"
    
    def get_observation(self, img: np.ndarray) -> np.ndarray:
        grid = np.zeros((STATE_FEATURES, GRID_ROWS, GRID_COLS), dtype=np.float32)
        
        elixir = self.detect_elixir(img)
        grid[13, :, :] = float(elixir) / 10.0
        grid[14, :, :] = 0.5
        
        return grid
    
    def execute_action(self, action: np.ndarray):
        """Execute action: [delay, pos_x, pos_y, card]"""
        delay = action[0] * MAX_DELAY
        pos_x = action[1]
        pos_y = action[2]
        card = int(action[3] * N_CARDS)
        
        # Only act if delay is low
        if delay > 3:
            return
        
        if card < 0 or card >= N_CARDS:
            return
        
        card_x, card_y = CARD_POSITIONS[card]
        screen_x = int(ARENA_LEFT + pos_x * (ARENA_RIGHT - ARENA_LEFT))
        screen_y = int(DEPLOY_TOP + pos_y * (DEPLOY_BOTTOM - DEPLOY_TOP))
        
        self._adb_command(['shell', 'input', 'tap', str(card_x), str(card_y)])
        time.sleep(0.1)
        self._adb_command(['shell', 'input', 'tap', str(screen_x), str(screen_y)])
    
    def reset(self) -> np.ndarray:
        self.episode_reward = 0
        self.steps = 0
        self.in_battle = False
        
        for _ in range(60):
            img = self.capture_screen()
            if img is None:
                time.sleep(0.5)
                continue
            
            state = self.detect_battle_state(img)
            
            if state == "battle":
                self.in_battle = True
                return self.get_observation(img)
            elif state == "end":
                self._adb_command(['shell', 'input', 'tap', '540', '1800'])
                time.sleep(1)
            
            time.sleep(0.5)
        
        return np.zeros((STATE_FEATURES, GRID_ROWS, GRID_COLS), dtype=np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        self.steps += 1
        
        self.execute_action(action)
        time.sleep(0.2)
        
        img = self.capture_screen()
        state = self.detect_battle_state(img) if img is not None else "menu"
        done = state != "battle"
        
        reward = -0.001 if not done else 0.0
        self.episode_reward += reward
        
        obs = self.get_observation(img) if img is not None else np.zeros(
            (STATE_FEATURES, GRID_ROWS, GRID_COLS), dtype=np.float32
        )
        
        return obs, reward, done, {'episode_reward': self.episode_reward, 'steps': self.steps}


# =============================================================================
# Worker Process
# =============================================================================

def env_worker(
    device_id: str,
    model_queue: Queue,
    experience_queue: Queue,
    stop_event: Event,
    worker_id: int,
):
    """Worker that collects experience."""
    print(f"[Worker {worker_id}] Starting with {device_id}")
    
    env = ClashEnv(device_id)
    device = torch.device("cpu")
    model = DreamerV3(device=device)
    
    episode_count = 0
    
    while not stop_event.is_set():
        # Check for model updates
        try:
            weights = model_queue.get_nowait()
            model.load_state_dict(weights)
            print(f"[Worker {worker_id}] Updated weights")
        except Empty:
            pass
        
        # Reset
        obs = env.reset()
        if not env.in_battle:
            print(f"[Worker {worker_id}] Waiting for battle...")
            time.sleep(2)
            continue
        
        episode_count += 1
        print(f"[Worker {worker_id}] Episode {episode_count}")
        
        # Collect episode
        deter, stoch = model.initial_state(1)
        prev_action = torch.zeros(1, 4, device=device)
        
        episode_data = []
        done = False
        
        while not done and not stop_event.is_set():
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                out = model.observe_step(obs_t, prev_action, (deter, stoch))
                deter = out['deter']
                stoch = out['stoch']
                
                features = model.get_features(deter, stoch)
                action, _, _ = model.get_action(features, sample=True)
            
            action_np = action[0].cpu().numpy()
            next_obs, reward, done, info = env.step(action_np)
            
            episode_data.append({
                'obs': obs,
                'action': {
                    'delay': action_np[0] * MAX_DELAY,
                    'pos_x': action_np[1],
                    'pos_y': action_np[2],
                    'card': int(action_np[3] * N_CARDS),
                },
                'reward': reward,
                'done': done,
            })
            
            obs = next_obs
            prev_action = action
        
        if episode_data:
            experience_queue.put((worker_id, episode_data))
            print(f"[Worker {worker_id}] Done: reward={info['episode_reward']:.3f}, steps={info['steps']}")


# =============================================================================
# DreamerV3 Training
# =============================================================================

def train_world_model(
    model: DreamerV3,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Train world model on batch of sequences."""
    model.train()
    
    obs = batch['observation'].to(device)
    action = batch['action'].to(device)
    reward = batch['reward'].to(device)
    done = batch['done'].to(device)
    
    B, T = obs.shape[:2]
    
    # Initialize state
    deter, stoch = model.initial_state(B)
    prev_action = torch.zeros(B, 4, device=device)
    
    total_recon_loss = 0
    total_kl_loss = 0
    total_reward_loss = 0
    
    for t in range(T):
        out = model.observe_step(obs[:, t], prev_action, (deter, stoch))
        deter = out['deter']
        stoch = out['stoch']
        prior_mean, prior_std = out['prior']
        post_mean, post_std = out['posterior']
        
        features = model.get_features(deter, stoch)
        
        # Reconstruction loss
        obs_pred = model.decoder(features)
        obs_target = obs[:, t].reshape(B, -1)
        recon_loss = F.mse_loss(obs_pred, obs_target)
        
        # KL divergence
        prior_dist = Normal(prior_mean, prior_std)
        post_dist = Normal(post_mean, post_std)
        kl = torch.distributions.kl_divergence(post_dist, prior_dist).sum(dim=-1).mean()
        
        # Reward prediction
        reward_pred = model.reward_head(features).squeeze(-1)
        reward_loss = F.mse_loss(reward_pred, reward[:, t])
        
        total_recon_loss += recon_loss
        total_kl_loss += kl
        total_reward_loss += reward_loss
        
        prev_action = action[:, t]
    
    # Total loss
    loss = (total_recon_loss + 0.1 * total_kl_loss + total_reward_loss) / T
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
    optimizer.step()
    
    return {
        'recon_loss': (total_recon_loss / T).item(),
        'kl_loss': (total_kl_loss / T).item(),
        'reward_loss': (total_reward_loss / T).item(),
    }


def train_actor_critic(
    model: DreamerV3,
    batch: Dict[str, torch.Tensor],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    device: torch.device,
    horizon: int = 15,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> Dict[str, float]:
    """Train actor-critic by imagining in the world model."""
    model.train()
    
    obs = batch['observation'].to(device)
    action = batch['action'].to(device)
    
    B, T = obs.shape[:2]
    
    # Get initial states from real data
    with torch.no_grad():
        deter, stoch = model.initial_state(B)
        prev_action = torch.zeros(B, 4, device=device)
        
        # Run through first few steps to get good starting state
        for t in range(min(5, T)):
            out = model.observe_step(obs[:, t], prev_action, (deter, stoch))
            deter = out['deter']
            stoch = out['stoch']
            prev_action = action[:, t]
    
    # Imagine forward
    imagined_features = []
    imagined_actions = []
    imagined_rewards = []
    imagined_values = []
    imagined_log_probs = []
    
    for h in range(horizon):
        features = model.get_features(deter, stoch)
        imagined_features.append(features)
        
        # Get action from actor
        act, log_prob, _ = model.get_action(features, sample=True)
        imagined_actions.append(act)
        imagined_log_probs.append(log_prob)
        
        # Predict reward and value
        reward = model.reward_head(features).squeeze(-1)
        value = model.critic(features).squeeze(-1)
        imagined_rewards.append(reward)
        imagined_values.append(value)
        
        # Imagine next state
        deter, stoch = model.imagine_step(act, (deter, stoch))
    
    # Final value for bootstrap
    final_features = model.get_features(deter, stoch)
    final_value = model.critic(final_features).squeeze(-1)
    
    # Compute lambda returns
    rewards = torch.stack(imagined_rewards, dim=1)  # (B, H)
    values = torch.stack(imagined_values, dim=1)    # (B, H)
    log_probs = torch.stack(imagined_log_probs, dim=1)  # (B, H)
    
    # GAE-style returns
    returns = torch.zeros_like(rewards)
    gae = 0
    for h in reversed(range(horizon)):
        next_value = final_value if h == horizon - 1 else values[:, h + 1]
        delta = rewards[:, h] + gamma * next_value - values[:, h]
        gae = delta + gamma * lambda_ * gae
        returns[:, h] = gae + values[:, h]
    
    # Normalize returns
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    # Critic loss
    critic_loss = F.mse_loss(values, returns.detach())
    
    critic_optimizer.zero_grad()
    critic_loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 100)
    critic_optimizer.step()
    
    # Actor loss (maximize returns)
    advantages = (returns - values).detach()
    actor_loss = -(log_probs * advantages).mean()
    
    # Entropy bonus
    features_stack = torch.stack(imagined_features, dim=1)
    action_stats = model.actor(features_stack)
    mean, logstd = action_stats.chunk(2, dim=-1)
    std = F.softplus(logstd) + 0.1
    entropy = Normal(mean, std).entropy().mean()
    
    actor_loss = actor_loss - 0.001 * entropy
    
    actor_optimizer.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 100)
    actor_optimizer.step()
    
    return {
        'actor_loss': actor_loss.item(),
        'critic_loss': critic_loss.item(),
        'entropy': entropy.item(),
        'mean_return': returns.mean().item(),
    }


# =============================================================================
# Main
# =============================================================================

def main(args):
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          Online DreamerV3 with {args.n_envs} Parallel Emulators                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  World model trained on replay buffer                                        ║
║  Actor-critic trained by imagining trajectories                              ║
║                                                                              ║
║  Start battles manually in Training Camp on each emulator                    ║
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
    
    # Check emulators
    result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
    print(f"ADB devices:\n{result.stdout}")
    
    emulator_ports = [5554 + i * 2 for i in range(args.n_envs)]
    device_ids = [f"emulator-{port}" for port in emulator_ports]
    
    # Model
    model = DreamerV3(device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Optimizers
    world_model_params = list(model.encoder.parameters()) + \
                         list(model.rnn.parameters()) + \
                         list(model.prior.parameters()) + \
                         list(model.posterior.parameters()) + \
                         list(model.decoder.parameters()) + \
                         list(model.reward_head.parameters())
    
    wm_optimizer = torch.optim.AdamW(world_model_params, lr=args.lr)
    actor_optimizer = torch.optim.AdamW(model.actor.parameters(), lr=args.lr * 0.1)
    critic_optimizer = torch.optim.AdamW(model.critic.parameters(), lr=args.lr)
    
    # Replay buffer
    replay_buffer = ReplayBuffer(capacity=args.buffer_size, seq_length=args.seq_length)
    
    # Multiprocessing
    model_queues = [Queue() for _ in range(args.n_envs)]
    experience_queue = Queue()
    stop_event = Event()
    
    # Start workers
    workers = []
    for i, device_id in enumerate(device_ids):
        p = Process(
            target=env_worker,
            args=(device_id, model_queues[i], experience_queue, stop_event, i)
        )
        p.start()
        workers.append(p)
        print(f"Started worker {i} for {device_id}")
    
    # Send initial weights
    weights = {k: v.cpu() for k, v in model.state_dict().items()}
    for q in model_queues:
        q.put(weights)
    
    # Training
    log_dir = Path("logs") / f"dreamer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    total_episodes = 0
    total_steps = 0
    train_steps = 0
    
    try:
        while total_episodes < args.max_episodes:
            # Collect experience
            try:
                worker_id, episode_data = experience_queue.get(timeout=30)
                total_episodes += 1
                total_steps += len(episode_data)
                
                replay_buffer.add_episode(episode_data)
                print(f"\n[Learner] Ep {total_episodes} from worker {worker_id}, "
                      f"buffer={len(replay_buffer)}, steps={total_steps}")
                
            except Empty:
                print("[Learner] Waiting for experience...")
                continue
            
            # Train if enough data
            if len(replay_buffer) >= args.min_buffer:
                for _ in range(args.train_steps):
                    batch = replay_buffer.sample_sequences(args.batch_size)
                    if batch is None:
                        continue
                    
                    # Train world model
                    wm_metrics = train_world_model(model, batch, wm_optimizer, device)
                    
                    # Train actor-critic
                    ac_metrics = train_actor_critic(
                        model, batch, actor_optimizer, critic_optimizer, device,
                        horizon=args.horizon
                    )
                    
                    train_steps += 1
                
                print(f"  WM: recon={wm_metrics['recon_loss']:.4f}, kl={wm_metrics['kl_loss']:.4f}")
                print(f"  AC: actor={ac_metrics['actor_loss']:.4f}, critic={ac_metrics['critic_loss']:.4f}")
            
            # Broadcast weights
            if total_episodes % args.update_freq == 0:
                weights = {k: v.cpu() for k, v in model.state_dict().items()}
                for q in model_queues:
                    q.put(weights)
                print(f"[Learner] Broadcast weights")
            
            # Save
            if total_episodes % args.save_freq == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'episodes': total_episodes,
                    'steps': total_steps,
                }, log_dir / f"checkpoint_{total_episodes}.pt")
                print(f"[Learner] Saved checkpoint")
    
    except KeyboardInterrupt:
        print("\n[Learner] Stopping...")
    
    finally:
        stop_event.set()
        for p in workers:
            p.terminate()
            p.join(timeout=5)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'episodes': total_episodes,
            'steps': total_steps,
        }, log_dir / "final.pt")
        print(f"\n✓ Saved to {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-episodes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-length", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--min-buffer", type=int, default=1000)
    parser.add_argument("--train-steps", type=int, default=10)
    parser.add_argument("--update-freq", type=int, default=5)
    parser.add_argument("--save-freq", type=int, default=20)
    
    args = parser.parse_args()
    main(args)
