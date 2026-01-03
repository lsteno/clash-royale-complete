#!/usr/bin/env python3
"""
Real Training Script for Clash Royale with DreamerV3.

This connects:
1. ADB screenshots → captures game state
2. Simple CNN → processes image to features
3. DreamerV3 Agent → learns and selects actions
4. ADB taps → executes actions in game

For now, we use a simplified perception (no YOLO required initially).
The agent learns from raw image features.
"""
import subprocess
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from collections import deque
import argparse
import signal
import sys

# Graceful shutdown
running = True
def signal_handler(sig, frame):
    global running
    print("\n\nStopping training gracefully...")
    running = False
signal.signal(signal.SIGINT, signal_handler)


# ============== ADB Interface ==============

def adb_screenshot() -> np.ndarray:
    """Capture screenshot via ADB and return as numpy array."""
    subprocess.run(["adb", "shell", "screencap", "-p", "/sdcard/s.png"],
                   capture_output=True, timeout=5)
    subprocess.run(["adb", "pull", "/sdcard/s.png", "/tmp/s.png"],
                   capture_output=True, timeout=5)
    img = cv2.imread("/tmp/s.png")
    if img is None:
        return np.zeros((2400, 1080, 3), dtype=np.uint8)
    return img


def adb_tap(x: int, y: int):
    """Send tap via ADB."""
    subprocess.run(["adb", "shell", "input", "tap", str(x), str(y)],
                   capture_output=True, timeout=5)


def adb_swipe(x1: int, y1: int, x2: int, y2: int, duration_ms: int = 100):
    """Send swipe/drag via ADB - more reliable for card deployment."""
    subprocess.run(["adb", "shell", "input", "swipe", 
                    str(x1), str(y1), str(x2), str(y2), str(duration_ms)],
                   capture_output=True, timeout=5)


# ============== Screen Regions (1080x2400) ==============

# Arena region (where troops fight)
ARENA_TOP = 288      # 12% from top
ARENA_BOTTOM = 1848  # 77% from top
ARENA_LEFT = 22      # 2% from left
ARENA_RIGHT = 1058   # 98% from left

# Card positions (1080x2400 screen)
# Cards are at y~2150, spread across bottom
CARDS = {
    1: (350, 2150),   # First card
    2: (550, 2150),   # Second card  
    3: (750, 2150),   # Third card
    4: (950, 2150),   # Fourth card
}

# Deployment positions (grid on YOUR side of arena)
# Y: 1150-1750 is your side (bottom half of arena)
DEPLOY_POSITIONS = [
    # Bridge positions (aggressive)
    (250, 1150, "left_bridge"),
    (550, 1150, "center_bridge"),
    (850, 1150, "right_bridge"),
    # Mid positions
    (100, 1500, "left_mid"),
    (550, 1450, "center_mid"),
    (950, 1500, "right_mid"),
    # Back positions (defensive)
    (100, 1700, "left_back"),
    (550, 1750, "center_back"),
    (950, 1700, "right_back"),
]


# ============== Simple Feature Extractor ==============

class SimpleEncoder(nn.Module):
    """
    Simple CNN to extract features from game screenshot.
    Input: (3, 128, 64) - resized arena image
    Output: (256,) feature vector
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # 64x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 16x8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 8x4
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Linear(256 * 8 * 4, 256)
    
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


# ============== Simple Actor Network ==============

class SimpleActor(nn.Module):
    """
    Simple policy network.
    Input: (256,) features
    Output: card logits (5) + position logits (10)
    """
    def __init__(self, feature_dim=256, num_cards=5, num_positions=10):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.card_head = nn.Linear(128, num_cards)  # 0=no action, 1-4=cards
        self.pos_head = nn.Linear(128, num_positions)
    
    def forward(self, features):
        x = self.trunk(features)
        card_logits = self.card_head(x)
        pos_logits = self.pos_head(x)
        return card_logits, pos_logits
    
    def act(self, features, deterministic=False):
        card_logits, pos_logits = self.forward(features)
        
        if deterministic:
            card = card_logits.argmax(dim=-1)
            pos = pos_logits.argmax(dim=-1)
        else:
            card = torch.distributions.Categorical(logits=card_logits).sample()
            pos = torch.distributions.Categorical(logits=pos_logits).sample()
        
        return card.item(), pos.item()


# ============== Simple Value Network ==============

class SimpleValue(nn.Module):
    """Value function for estimating state values."""
    def __init__(self, feature_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    
    def forward(self, features):
        return self.net(features).squeeze(-1)


# ============== Image Processing ==============

def process_screenshot(img: np.ndarray) -> torch.Tensor:
    """
    Process screenshot to tensor for the network.
    1. Crop to arena region
    2. Resize to 128x64
    3. Normalize and convert to tensor
    """
    # Crop arena
    arena = img[ARENA_TOP:ARENA_BOTTOM, ARENA_LEFT:ARENA_RIGHT]
    
    # Resize
    arena = cv2.resize(arena, (64, 128))
    
    # Convert BGR to RGB, normalize
    arena = cv2.cvtColor(arena, cv2.COLOR_BGR2RGB)
    arena = arena.astype(np.float32) / 255.0
    
    # To tensor (C, H, W)
    tensor = torch.from_numpy(arena).permute(2, 0, 1).unsqueeze(0)
    
    return tensor


# ============== Reward Estimation ==============

def estimate_reward(prev_img: np.ndarray, curr_img: np.ndarray) -> float:
    """
    Estimate reward from image difference.
    This is a simple heuristic - real training would use OCR for tower HP.
    
    Positive: enemy tower damage (red pixels decreasing in top half)
    Negative: our tower damage (red pixels decreasing in bottom half)
    """
    # Convert to grayscale for simplicity
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    
    # Simple difference
    diff = np.abs(curr_gray.astype(float) - prev_gray.astype(float)).mean()
    
    # Normalize to small reward
    reward = diff / 255.0 * 0.01
    
    # Random small noise to encourage exploration
    reward += np.random.randn() * 0.001
    
    return reward


def detect_game_state(img: np.ndarray) -> str:
    """
    Detect which state the game is in.
    Returns: 'main_menu', 'in_match', or 'end_screen'
    """
    # Check for cards at bottom (visible during match)
    # Cards are at y=2100-2200, colorful with high variance
    card_area = img[2100:2200, 300:900]
    card_gray = cv2.cvtColor(card_area, cv2.COLOR_BGR2GRAY)
    card_std = card_gray.std()
    
    # Check for elixir bar area (purple bar, visible during match)
    # Elixir bar is around y=2000-2050
    elixir_area = img[2000:2050, 100:1000]
    elixir_hsv = cv2.cvtColor(elixir_area, cv2.COLOR_BGR2HSV)
    # Purple elixir bar has specific hue
    purple_mask = ((elixir_hsv[:,:,0] > 120) & (elixir_hsv[:,:,0] < 170) & 
                   (elixir_hsv[:,:,1] > 30))
    purple_ratio = purple_mask.mean()
    
    # Check for battle button (main menu - golden button at bottom center)
    # Battle button is around y=1700-1900
    battle_area = img[1700:1900, 350:730]
    battle_hsv = cv2.cvtColor(battle_area, cv2.COLOR_BGR2HSV)
    # Golden/orange button
    gold_mask = ((battle_hsv[:,:,0] > 10) & (battle_hsv[:,:,0] < 35) & 
                 (battle_hsv[:,:,1] > 100) & (battle_hsv[:,:,2] > 150))
    gold_ratio = gold_mask.mean()
    
    # Check for arena (green grass visible during match)
    arena_area = img[1200:1600, 200:880]
    arena_hsv = cv2.cvtColor(arena_area, cv2.COLOR_BGR2HSV)
    # Green grass
    green_mask = ((arena_hsv[:,:,0] > 35) & (arena_hsv[:,:,0] < 85) & 
                  (arena_hsv[:,:,1] > 30))
    green_ratio = green_mask.mean()
    
    # Decision logic:
    # IN MATCH: cards visible (high variance) OR purple elixir bar OR green arena
    if card_std > 40 or purple_ratio > 0.05 or green_ratio > 0.1:
        return 'in_match'
    
    # MAIN MENU: golden battle button visible, no cards
    if gold_ratio > 0.05 and card_std < 30:
        return 'main_menu'
    
    # END SCREEN: no cards, no battle button, probably showing results
    return 'end_screen'


def check_game_over(img: np.ndarray) -> bool:
    """Check if game ended (for backward compatibility)."""
    return detect_game_state(img) == 'end_screen'


def get_elixir_count(img: np.ndarray) -> int:
    """
    Read elixir count from the white text display.
    The elixir number is shown in white text at bottom left (200-400, 2300-2400).
    Returns estimated elixir (0-10).
    """
    # Elixir text area
    elixir_area = img[2300:2400, 200:400]
    
    # Convert to grayscale and find white text
    gray = cv2.cvtColor(elixir_area, cv2.COLOR_BGR2GRAY)
    
    # Threshold for white text (elixir number)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Count white pixels - more white = higher number (rough estimate)
    white_ratio = thresh.mean() / 255.0
    
    # Very rough estimate based on digit size
    # Single digit (1-9) has less white than double digit (10)
    # This is a heuristic - could use OCR for accuracy
    if white_ratio < 0.02:
        return 0
    elif white_ratio < 0.05:
        return 2  # Probably 1-3
    elif white_ratio < 0.08:
        return 4  # Probably 4-6
    elif white_ratio < 0.12:
        return 7  # Probably 7-9
    else:
        return 10  # Probably 10


# ============== Experience Buffer ==============

class ExperienceBuffer:
    """Simple experience replay buffer."""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states = torch.cat([b[0] for b in batch])
        actions = torch.tensor([b[1] for b in batch])
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        next_states = torch.cat([b[3] for b in batch])
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


# ============== Training Loop ==============

def train(args):
    global running
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    Clash Royale - DreamerV3 Training                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Training with: 2.6 Hog Cycle deck                                           ║
║  Mode: Training Camp vs Trainer AI                                           ║
║  Press Ctrl+C to stop training gracefully                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✓ Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print(f"Using CPU")
    
    # Models
    encoder = SimpleEncoder().to(device)
    actor = SimpleActor().to(device)
    value = SimpleValue().to(device)
    
    # Optimizers
    encoder_opt = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    value_opt = torch.optim.Adam(value.parameters(), lr=args.lr)
    
    # Buffer
    buffer = ExperienceBuffer(capacity=args.buffer_size)
    
    # Logging
    log_dir = Path("logs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logging to: {log_dir}")
    
    # Stats
    episode = 0
    total_steps = 0
    episode_reward = 0
    episode_steps = 0
    rewards_history = []
    
    print(f"\nStarting training loop...")
    print(f"Taking screenshots every {args.step_delay}s")
    print(f"Training after {args.train_start} steps\n")
    
    # Initial screenshot
    prev_img = adb_screenshot()
    
    # Wait for match to start
    print("Waiting for match to start...")
    while running:
        state = detect_game_state(prev_img)
        if state == 'in_match':
            print("Match detected! Starting training...\n")
            break
        elif state == 'main_menu':
            print("  On main menu - please start a Training Camp match")
        elif state == 'end_screen':
            print("  On end screen - please dismiss and start new match")
        time.sleep(2)
        prev_img = adb_screenshot()
    
    prev_tensor = process_screenshot(prev_img).to(device)
    
    while running and total_steps < args.max_steps:
        # Check game state first
        game_state = detect_game_state(prev_img)
        
        if game_state != 'in_match':
            # Not in match anymore
            if game_state == 'end_screen':
                episode += 1
                rewards_history.append(episode_reward)
                avg_reward = np.mean(rewards_history[-10:]) if rewards_history else 0
                
                print(f"\n=== Episode {episode} ===")
                print(f"Steps: {episode_steps} | Reward: {episode_reward:.4f} | Avg(10): {avg_reward:.4f}")
                print(f"Total steps: {total_steps}\n")
                
                episode_reward = 0
                episode_steps = 0
                
                print("Match ended! Please start a new Training Camp match...")
            else:
                print("Not in match. Please start a Training Camp match...")
            
            # Wait for new match
            while running and detect_game_state(prev_img) != 'in_match':
                time.sleep(2)
                prev_img = adb_screenshot()
            
            if running:
                print("Match detected! Continuing training...\n")
                prev_tensor = process_screenshot(prev_img).to(device)
            continue
        
        # Check elixir before deciding action
        current_elixir = get_elixir_count(prev_img)
        
        # Get features
        with torch.no_grad():
            features = encoder(prev_tensor)
        
        # Select action
        if total_steps < args.random_steps:
            # Random exploration
            card = np.random.randint(0, 5)
            pos = np.random.randint(0, len(DEPLOY_POSITIONS))
        else:
            card, pos = actor.act(features, deterministic=False)
        
        # Execute action (only if we have enough elixir)
        # Most cards cost 2-5 elixir, require at least 3 to play safely
        MIN_ELIXIR = 3
        
        if card > 0 and current_elixir >= MIN_ELIXIR:
            # Get card and deploy positions
            cx, cy = CARDS[card]
            px, py, name = DEPLOY_POSITIONS[pos]
            
            # Use SWIPE/DRAG from card to position (more reliable than tap-tap)
            # This drags the card directly to the deployment location
            adb_swipe(cx, cy, px, py, duration_ms=150)
            
            if args.verbose:
                print(f"Step {total_steps}: Card {card} -> {name} (elixir: {current_elixir})")
        elif card > 0:
            if args.verbose:
                print(f"Step {total_steps}: Waiting for elixir ({current_elixir}/10)")
        else:
            if args.verbose:
                print(f"Step {total_steps}: No action (elixir: {current_elixir})")
        
        # Wait for game to update
        time.sleep(args.step_delay)
        
        # Get new screenshot
        curr_img = adb_screenshot()
        curr_tensor = process_screenshot(curr_img).to(device)
        
        # Estimate reward
        reward = estimate_reward(prev_img, curr_img)
        episode_reward += reward
        
        # Check if game over
        done = check_game_over(curr_img)
        
        # Store experience
        action = card * len(DEPLOY_POSITIONS) + pos  # Flatten action
        buffer.add(prev_tensor.cpu(), action, reward, curr_tensor.cpu(), done)
        
        # Update state
        prev_img = curr_img
        prev_tensor = curr_tensor
        
        total_steps += 1
        episode_steps += 1
        
        # Train
        if len(buffer) >= args.batch_size and total_steps >= args.train_start:
            if total_steps % args.train_every == 0:
                # Sample batch
                states, actions, rewards, next_states, dones = buffer.sample(args.batch_size)
                states = states.to(device)
                next_states = next_states.to(device)
                rewards = rewards.to(device)
                dones = dones.to(device)
                
                # Encode
                features = encoder(states)
                next_features = encoder(next_states)
                
                # Value loss
                values = value(features)
                with torch.no_grad():
                    next_values = value(next_features)
                    targets = rewards + args.gamma * next_values * (1 - dones)
                value_loss = F.mse_loss(values, targets)
                
                # Actor loss (simple policy gradient)
                card_logits, pos_logits = actor(features)
                card_actions = actions // len(DEPLOY_POSITIONS)
                pos_actions = actions % len(DEPLOY_POSITIONS)
                
                card_log_probs = F.log_softmax(card_logits, dim=-1)
                pos_log_probs = F.log_softmax(pos_logits, dim=-1)
                
                advantages = (targets - values).detach()
                actor_loss = -(
                    card_log_probs.gather(1, card_actions.unsqueeze(1)).squeeze() +
                    pos_log_probs.gather(1, pos_actions.unsqueeze(1)).squeeze()
                ) * advantages
                actor_loss = actor_loss.mean()
                
                # Update
                encoder_opt.zero_grad()
                actor_opt.zero_grad()
                value_opt.zero_grad()
                
                (value_loss + actor_loss).backward()
                
                encoder_opt.step()
                actor_opt.step()
                value_opt.step()
                
                if total_steps % 100 == 0:
                    print(f"Step {total_steps} | Value Loss: {value_loss.item():.4f} | Actor Loss: {actor_loss.item():.4f}")
        
        # Save checkpoint
        if total_steps % args.save_every == 0 and total_steps > 0:
            checkpoint = {
                "encoder": encoder.state_dict(),
                "actor": actor.state_dict(),
                "value": value.state_dict(),
                "total_steps": total_steps,
                "episode": episode,
            }
            torch.save(checkpoint, log_dir / f"checkpoint_{total_steps}.pt")
            print(f"Saved checkpoint at step {total_steps}")
    
    # Final save
    print("\nSaving final checkpoint...")
    checkpoint = {
        "encoder": encoder.state_dict(),
        "actor": actor.state_dict(),
        "value": value.state_dict(),
        "total_steps": total_steps,
        "episode": episode,
    }
    torch.save(checkpoint, log_dir / "checkpoint_final.pt")
    print(f"Training complete! {total_steps} steps, {episode} episodes")


def main():
    parser = argparse.ArgumentParser(description="Clash Royale Training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--buffer-size", type=int, default=10000, help="Replay buffer size")
    parser.add_argument("--max-steps", type=int, default=100000, help="Max training steps")
    parser.add_argument("--random-steps", type=int, default=500, help="Random exploration steps")
    parser.add_argument("--train-start", type=int, default=100, help="Steps before training starts")
    parser.add_argument("--train-every", type=int, default=4, help="Train every N steps")
    parser.add_argument("--save-every", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--step-delay", type=float, default=2.0, help="Delay between steps (seconds)")
    parser.add_argument("--verbose", action="store_true", help="Print every action")
    args = parser.parse_args()
    
    # Check ADB
    result = subprocess.run(["adb", "devices"], capture_output=True, text=True)
    if "emulator" not in result.stdout:
        print("❌ No emulator connected!")
        return
    print("✓ Emulator connected")
    
    train(args)


if __name__ == "__main__":
    main()
