#!/usr/bin/env python3
"""
DreamerV3 Training for Clash Royale - macOS Edition.

This integrates:
- KataCR-style perception (adapted for ADB on macOS)
- DreamerV3 world model and policy
- Live gameplay via Android emulator

Usage:
    python train_dreamer.py --verbose
    python train_dreamer.py --load-checkpoint logs/xxx/checkpoint.pt
"""
import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agent.dreamer_v3 import DreamerV3, create_dreamer_model
from src.agent.dreamer_trainer import DreamerTrainer
from src.perception.katacr_adapter import ADBScreenCapture, SimplifiedStateBuilder, create_perception_pipeline

# Global flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    global running
    print("\n\nStopping training gracefully...")
    running = False

signal.signal(signal.SIGINT, signal_handler)


# ============== Screen Coordinates (1080x2400) ==============
CARDS = {
    1: (350, 2150),
    2: (550, 2150),
    3: (750, 2150),
    4: (950, 2150),
}

DEPLOY_POSITIONS = [
    # Bridges (aggressive)
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

# Card elixir costs for 2.6 Hog Cycle
CARD_COSTS = {
    'hog_rider': 4,
    'musketeer': 4,
    'ice_golem': 2,
    'ice_spirit': 1,
    'skeletons': 1,
    'cannon': 3,
    'fireball': 4,
    'the_log': 2,
}


# ============== ADB Interface ==============
import subprocess

def adb_tap(x: int, y: int):
    """Send tap via ADB."""
    subprocess.run(["adb", "shell", "input", "tap", str(x), str(y)],
                   capture_output=True, timeout=5)

def adb_swipe(x1: int, y1: int, x2: int, y2: int, duration_ms: int = 150):
    """Send swipe/drag via ADB."""
    subprocess.run(["adb", "shell", "input", "swipe",
                    str(x1), str(y1), str(x2), str(y2), str(duration_ms)],
                   capture_output=True, timeout=5)


def detect_game_state(img: np.ndarray) -> str:
    """Detect game state: 'main_menu', 'in_match', or 'end_screen'."""
    # Check for cards at bottom (visible during match)
    card_area = img[2100:2200, 300:900]
    card_gray = cv2.cvtColor(card_area, cv2.COLOR_BGR2GRAY)
    card_std = card_gray.std()
    
    # Check for elixir bar (purple, visible during match)
    elixir_area = img[2000:2050, 100:1000]
    elixir_hsv = cv2.cvtColor(elixir_area, cv2.COLOR_BGR2HSV)
    purple_mask = ((elixir_hsv[:,:,0] > 120) & (elixir_hsv[:,:,0] < 170) &
                   (elixir_hsv[:,:,1] > 30))
    purple_ratio = purple_mask.mean()
    
    # Check for battle button (main menu)
    battle_area = img[1700:1900, 350:730]
    battle_hsv = cv2.cvtColor(battle_area, cv2.COLOR_BGR2HSV)
    gold_mask = ((battle_hsv[:,:,0] > 10) & (battle_hsv[:,:,0] < 35) &
                 (battle_hsv[:,:,1] > 100) & (battle_hsv[:,:,2] > 150))
    gold_ratio = gold_mask.mean()
    
    # Check for arena (green grass)
    arena_area = img[1200:1600, 200:880]
    arena_hsv = cv2.cvtColor(arena_area, cv2.COLOR_BGR2HSV)
    green_mask = ((arena_hsv[:,:,0] > 35) & (arena_hsv[:,:,0] < 85) &
                  (arena_hsv[:,:,1] > 30))
    green_ratio = green_mask.mean()
    
    # Decision logic
    if card_std > 40 or purple_ratio > 0.05 or green_ratio > 0.1:
        return 'in_match'
    if gold_ratio > 0.05 and card_std < 30:
        return 'main_menu'
    return 'end_screen'


def get_elixir(img: np.ndarray) -> int:
    """Get current elixir from pink/purple bar."""
    # Updated coordinates: (200, 2300, 1050, 2400)
    elixir_bar = img[2300:2400, 200:1050]
    hsv = cv2.cvtColor(elixir_bar, cv2.COLOR_BGR2HSV)
    # Pink/purple in HSV: H=140-180 or H=0-15 (wraps), with saturation
    mask = ((hsv[:,:,0] > 140) | (hsv[:,:,0] < 15)) & (hsv[:,:,1] > 50) & (hsv[:,:,2] > 50)
    cols = mask.any(axis=0)
    if cols.any():
        filled = np.where(cols)[0].max()
        return min(10, max(0, int((filled / mask.shape[1]) * 10)))
    return 0


def process_observation(img: np.ndarray) -> np.ndarray:
    """Process image for DreamerV3 input."""
    # Extract arena region
    arena = img[580:1850, 22:1058]  # Arena bounds
    # Resize to model input size
    arena = cv2.resize(arena, (128, 256))
    # Convert BGR to RGB and normalize
    arena = cv2.cvtColor(arena, cv2.COLOR_BGR2RGB)
    arena = arena.astype(np.float32) / 255.0
    # Transpose to (C, H, W)
    arena = arena.transpose(2, 0, 1)
    return arena


def estimate_reward(prev_img: np.ndarray, curr_img: np.ndarray) -> float:
    """
    Estimate reward from visual change.
    Positive for enemy damage, negative for our damage.
    """
    # Check tower HP areas for changes
    def get_hp_indicator(img, region):
        x1, y1, x2, y2 = region
        area = img[y1:y2, x1:x2]
        # Red indicates damage
        hsv = cv2.cvtColor(area, cv2.COLOR_BGR2HSV)
        red_mask = ((hsv[:,:,0] < 10) | (hsv[:,:,0] > 170)) & (hsv[:,:,1] > 100)
        return red_mask.mean()
    
    # Our towers (bottom)
    our_towers = [(100, 1650, 250, 1750), (830, 1650, 980, 1750)]
    # Enemy towers (top)
    enemy_towers = [(100, 650, 250, 750), (830, 650, 980, 750)]
    
    reward = 0.0
    
    for region in our_towers:
        prev_hp = get_hp_indicator(prev_img, region)
        curr_hp = get_hp_indicator(curr_img, region)
        if curr_hp > prev_hp + 0.05:  # Our tower took damage
            reward -= 0.1
    
    for region in enemy_towers:
        prev_hp = get_hp_indicator(prev_img, region)
        curr_hp = get_hp_indicator(curr_img, region)
        if curr_hp > prev_hp + 0.05:  # Enemy tower took damage
            reward += 0.1
    
    return reward


def train(args):
    global running
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    Clash Royale - DreamerV3 Training                          ║
║                           macOS + Apple Silicon                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  This is REAL DreamerV3 with:                                                ║
║    • RSSM World Model (learns to predict future)                             ║
║    • Imagination-based planning                                              ║
║    • Actor-Critic on imagined trajectories                                   ║
║                                                                              ║
║  Press Ctrl+C to stop training gracefully                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Create model
    model = create_dreamer_model(device)
    trainer = DreamerTrainer(
        model=model,
        device=device,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        imagine_horizon=args.horizon,
    )
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        print(f"Loading checkpoint: {args.load_checkpoint}")
        trainer.load(args.load_checkpoint)
    
    # Setup logging
    log_dir = Path("logs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logging to: {log_dir}")
    
    # Screen capture and perception
    if args.use_katacr:
        print("Initializing KataCR perception...")
        capture, perception = create_perception_pipeline(use_full_katacr=True)
        print("✓ KataCR YOLOv8 unit detection enabled")
    else:
        capture = ADBScreenCapture()
        perception = SimplifiedStateBuilder()
    
    if not capture.isOpened():
        print("ERROR: Cannot connect to emulator via ADB")
        return
    print("✓ Emulator connected")
    
    # Stats
    total_steps = 0
    episode = 0
    episode_reward = 0
    episode_steps = 0
    rewards_history = deque(maxlen=100)
    losses_history = deque(maxlen=100)
    
    print(f"\nStarting training...")
    print(f"Step delay: {args.step_delay}s")
    print(f"Training starts after {args.prefill} steps\n")
    
    # Wait for match
    print("Waiting for match to start...")
    while running:
        success, img = capture.read()
        if not success:
            time.sleep(0.5)
            continue
        
        state = detect_game_state(img)
        if state == 'in_match':
            print("Match detected! Starting...\n")
            break
        elif state == 'main_menu':
            print("  On main menu - please start a Training Camp match")
        else:
            print("  On end screen - please dismiss and start new match")
        time.sleep(2)
    
    # Get initial observation
    success, prev_img = capture.read()
    prev_obs = process_observation(prev_img)
    
    # For action tracking
    prev_action = np.zeros(45)  # One-hot: 5 cards * 9 positions
    prev_action[0] = 1  # No action initially
    
    while running:
        # Check game state
        success, curr_img = capture.read()
        if not success:
            time.sleep(0.5)
            continue
        
        game_state = detect_game_state(curr_img)
        
        if game_state != 'in_match':
            # Episode ended
            if game_state == 'end_screen':
                episode += 1
                rewards_history.append(episode_reward)
                avg_reward = np.mean(list(rewards_history)) if rewards_history else 0
                
                print(f"\n=== Episode {episode} ===")
                print(f"Steps: {episode_steps} | Reward: {episode_reward:.4f} | Avg: {avg_reward:.4f}")
                
                episode_reward = 0
                episode_steps = 0
            
            print("Waiting for next match...")
            while running and detect_game_state(capture.read()[1] if capture.read()[0] else curr_img) != 'in_match':
                time.sleep(2)
            
            if running:
                print("Match detected! Continuing...\n")
                success, prev_img = capture.read()
                prev_obs = process_observation(prev_img)
            continue
        
        # Get elixir
        elixir = get_elixir(curr_img)
        
        # Select action using model
        curr_obs = process_observation(curr_img)
        obs_tensor = torch.tensor(curr_obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            # Get RSSM state
            if not hasattr(train, 'rssm_state') or train.rssm_state is None:
                train.rssm_state = model.initial_state(1, device)
            
            action_tensor = torch.tensor(prev_action, dtype=torch.float32, device=device).unsqueeze(0)
            _, train.rssm_state = model.observe_step(obs_tensor, action_tensor, train.rssm_state)
            
            # Get action from actor
            if total_steps < args.random_steps:
                # Random exploration
                card = np.random.randint(0, 5)
                pos = np.random.randint(0, 9)
            else:
                card, pos = model.actor(train.rssm_state.features, deterministic=False)
                card = card.item()
                pos = pos.item()
        
        # Execute action if we have enough elixir
        MIN_ELIXIR = 2
        
        if card > 0 and elixir >= MIN_ELIXIR:
            cx, cy = CARDS[card]
            px, py, name = DEPLOY_POSITIONS[pos]
            
            # Drag card to position
            adb_swipe(cx, cy, px, py, duration_ms=150)
            
            if args.verbose:
                print(f"Step {total_steps}: Card {card} -> {name} (elixir: {elixir})")
        elif args.verbose:
            if card > 0:
                print(f"Step {total_steps}: Waiting for elixir ({elixir}/10)")
            else:
                print(f"Step {total_steps}: No action (elixir: {elixir})")
        
        # Wait for game to update
        time.sleep(args.step_delay)
        
        # Get new observation
        success, new_img = capture.read()
        if not success:
            continue
        
        new_obs = process_observation(new_img)
        
        # Calculate reward
        reward = estimate_reward(curr_img, new_img)
        done = detect_game_state(new_img) != 'in_match'
        
        # Create one-hot action
        action = np.zeros(45)
        if card > 0:
            action[card * 9 + pos] = 1
        else:
            action[0] = 1  # No action
        
        # Add to buffer
        trainer.add_experience(prev_obs, prev_action, reward, done)
        
        episode_reward += reward
        episode_steps += 1
        total_steps += 1
        
        # Train
        if len(trainer.buffer) >= args.prefill and total_steps % args.train_every == 0:
            losses = trainer.train_step()
            if losses:
                losses_history.append(losses)
                
                if total_steps % 100 == 0:
                    avg_model = np.mean([l['model_loss'] for l in losses_history])
                    avg_actor = np.mean([l['actor_loss'] for l in losses_history])
                    print(f"Step {total_steps} | Model: {avg_model:.4f} | Actor: {avg_actor:.4f}")
        
        # Update state
        prev_obs = new_obs
        prev_action = action
        prev_img = new_img
        
        # Save checkpoint
        if total_steps % args.save_every == 0 and total_steps > 0:
            checkpoint_path = log_dir / f"checkpoint_{total_steps}.pt"
            trainer.save(str(checkpoint_path))
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Final save
    print("\nSaving final checkpoint...")
    trainer.save(str(log_dir / "checkpoint_final.pt"))
    print(f"Training complete! {total_steps} steps, {episode} episodes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DreamerV3 Training for Clash Royale")
    parser.add_argument("--step-delay", type=float, default=1.5, help="Delay between steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--seq-length", type=int, default=30, help="Sequence length for RSSM")
    parser.add_argument("--horizon", type=int, default=15, help="Imagination horizon")
    parser.add_argument("--prefill", type=int, default=500, help="Steps before training")
    parser.add_argument("--random-steps", type=int, default=200, help="Random exploration steps")
    parser.add_argument("--train-every", type=int, default=5, help="Train every N steps")
    parser.add_argument("--save-every", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--load-checkpoint", type=str, help="Load from checkpoint")
    parser.add_argument("--verbose", action="store_true", help="Print every action")
    parser.add_argument("--use-katacr", action="store_true", help="Use KataCR YOLOv8 unit detection")
    
    args = parser.parse_args()
    train(args)
