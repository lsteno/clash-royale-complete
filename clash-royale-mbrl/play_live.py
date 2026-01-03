#!/usr/bin/env python3
"""
Live Play with Trained DreamerV3 Agent.

Features:
- Waits until a battle starts (detects arena)
- Only plays during active battle
- Detects game end (victory/defeat screen)
- Resets state between games

Usage:
    python play_live.py --checkpoint logs/fast_XXXXX/best.pt
    python play_live.py --checkpoint logs/fast_XXXXX/best.pt --debug
"""
import argparse
import time
import subprocess
import sys
from pathlib import Path
from enum import Enum

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from train_offline_fast import FastDreamer, GRID_ROWS, GRID_COLS, STATE_FEATURES, ACTION_DIM, N_CARDS, N_POSITIONS


class GameState(Enum):
    MENU = "menu"
    BATTLE = "battle"
    END_SCREEN = "end_screen"


# Screen coordinates for 1080x2400 display
CARD_POSITIONS = [
    (270, 2220),   # Card 1
    (460, 2220),   # Card 2
    (650, 2220),   # Card 3
    (840, 2220),   # Card 4
]

# Deploy positions (3x3 grid over arena)
DEPLOY_GRID = [
    # Row 0 (enemy side - top)

# Minimum elixir to attempt an action (prevents wasting taps)
MIN_ELIXIR_TO_ACT = 2
    [(270, 800), (540, 800), (810, 800)],
    # Row 1 (middle)
    [(270, 1100), (540, 1100), (810, 1100)],
    # Row 2 (our side - bottom)
    [(270, 1400), (540, 1400), (810, 1400)],
]

# Detection regions
ELIXIR_ROI = (200, 2300, 1050, 2400)  # Elixir bar
TIMER_ROI = (450, 100, 630, 180)      # Battle timer region
ARENA_ROI = (100, 400, 980, 1600)     # Arena play area
END_SCREEN_ROI = (300, 1000, 780, 1200)  # Victory/Defeat text area


class GameDetector:
    """Detects game state from screenshots."""
    
    def __init__(self):
        self.last_state = GameState.MENU
        self.battle_start_time = None
        self.battle_frame_count = 0
        
    def detect_state(self, img: np.ndarray) -> GameState:
        """Detect current game state from screenshot."""
        if img is None:
            return GameState.MENU
        
        # Check for active battle FIRST (more reliable)
        is_battle = self._detect_battle(img)
        
        if is_battle:
            self.battle_frame_count += 1
            return GameState.BATTLE
        
        # Only check for end screen if we were in battle for a while
        # This prevents false positives during loading screens
        if self.battle_frame_count > 10:  # At least 5 seconds of battle
            if self._detect_end_screen(img):
                self.battle_frame_count = 0
                return GameState.END_SCREEN
        
        # Reset battle counter if not in battle
        if not is_battle:
            self.battle_frame_count = 0
        
        return GameState.MENU
    
    def _detect_battle(self, img: np.ndarray) -> bool:
        """Detect if we're in an active battle."""
        # Method 1: Check for elixir bar (pink/purple color)
        x1, y1, x2, y2 = ELIXIR_ROI
        if y2 <= img.shape[0] and x2 <= img.shape[1]:
            elixir_region = img[y1:y2, x1:x2]
            hsv = cv2.cvtColor(elixir_region, cv2.COLOR_BGR2HSV)
            
            # Pink/magenta elixir bar
            pink_lower = np.array([140, 50, 50])
            pink_upper = np.array([170, 255, 255])
            mask = cv2.inRange(hsv, pink_lower, pink_upper)
            pink_ratio = np.sum(mask > 0) / mask.size
            
            if pink_ratio > 0.05:  # At least 5% pink pixels
                return True
        
        # Method 2: Check for timer at top (white numbers on dark)
        x1, y1, x2, y2 = TIMER_ROI
        if y2 <= img.shape[0] and x2 <= img.shape[1]:
            timer_region = img[y1:y2, x1:x2]
            gray = cv2.cvtColor(timer_region, cv2.COLOR_BGR2GRAY)
            
            # Timer has high contrast white text
            white_pixels = np.sum(gray > 200)
            total_pixels = gray.size
            white_ratio = white_pixels / total_pixels
            
            if white_ratio > 0.1 and white_ratio < 0.5:  # Some white but not all
                return True
        
        # Method 3: Check arena area for game elements (green grass)
        x1, y1, x2, y2 = ARENA_ROI
        if y2 <= img.shape[0] and x2 <= img.shape[1]:
            arena_region = img[y1:y2, x1:x2]
            hsv = cv2.cvtColor(arena_region, cv2.COLOR_BGR2HSV)
            
            # Green grass/arena floor
            green_lower = np.array([35, 40, 40])
            green_upper = np.array([85, 255, 255])
            mask = cv2.inRange(hsv, green_lower, green_upper)
            green_ratio = np.sum(mask > 0) / mask.size
            
            if green_ratio > 0.1:  # Significant green
                return True
        
        return False
    
    def _detect_end_screen(self, img: np.ndarray) -> bool:
        """Detect victory/defeat screen - VERY conservative to avoid false positives."""
        # The end screen has very distinct characteristics:
        # 1. No elixir bar visible
        # 2. Large centered text (Victory/Defeat)
        # 3. Darker overall image
        
        # First check: elixir bar should NOT be visible on end screen
        x1, y1, x2, y2 = ELIXIR_ROI
        if y2 <= img.shape[0] and x2 <= img.shape[1]:
            elixir_region = img[y1:y2, x1:x2]
            hsv = cv2.cvtColor(elixir_region, cv2.COLOR_BGR2HSV)
            pink_lower = np.array([140, 50, 50])
            pink_upper = np.array([170, 255, 255])
            mask = cv2.inRange(hsv, pink_lower, pink_upper)
            pink_ratio = np.sum(mask > 0) / mask.size
            
            # If elixir bar is visible, we're still in battle
            if pink_ratio > 0.03:
                return False
        
        # Check for victory screen (large gold banner)
        # Victory banner appears around y=800-1000, centered
        victory_roi = (250, 700, 830, 950)
        x1, y1, x2, y2 = victory_roi
        if y2 <= img.shape[0] and x2 <= img.shape[1]:
            region = img[y1:y2, x1:x2]
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            
            # Victory gold/yellow
            gold_lower = np.array([15, 100, 150])
            gold_upper = np.array([35, 255, 255])
            gold_mask = cv2.inRange(hsv, gold_lower, gold_upper)
            gold_ratio = np.sum(gold_mask > 0) / gold_mask.size
            
            if gold_ratio > 0.25:  # Need significant gold
                return True
        
        # Check for defeat screen (blue/gray tint)
        # More conservative - need multiple indicators
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        overall_brightness = np.mean(gray)
        
        # End screens are typically darker
        if overall_brightness < 60:
            # Additional check for "OK" button or similar
            return True
        
        return False


class LiveAgent:
    """Agent that plays Clash Royale using trained DreamerV3 model."""
    
    def __init__(self, checkpoint_path: str, device: str = None):
        # Device
        if device:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = FastDreamer(device=self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"✓ Loaded model from {checkpoint_path}")
        
        # State tracking
        self.reset_state()
        
        # Game detector
        self.game_detector = GameDetector()
        
        # Try to load KataCR perception
        self.perception = None
        try:
            from src.perception.katacr_adapter import KataCRPerception
            self.perception = KataCRPerception()
            print("✓ Loaded KataCR perception")
        except Exception as e:
            print(f"⚠ KataCR not available: {e}")
            print("  Using simple perception instead")
    
    def reset_state(self):
        """Reset RSSM state for new game."""
        self.deter, self.stoch = self.model.initial_state(1)
        self.prev_action = torch.zeros(1, ACTION_DIM, device=self.device)
        print("  [State reset for new game]")
    
    def capture_screen(self) -> np.ndarray:
        """Capture screenshot via ADB."""
        try:
            result = subprocess.run(
                ['adb', 'exec-out', 'screencap', '-p'],
                capture_output=True, timeout=2
            )
            if result.returncode != 0:
                return None
            
            img_array = np.frombuffer(result.stdout, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            return None
    
    def detect_elixir(self, img: np.ndarray) -> int:
        """Detect elixir from screenshot."""
        if img is None:
            return 0
        
        x1, y1, x2, y2 = ELIXIR_ROI
        if y2 > img.shape[0] or x2 > img.shape[1]:
            return 5
        
        elixir_bar = img[y1:y2, x1:x2]
        hsv = cv2.cvtColor(elixir_bar, cv2.COLOR_BGR2HSV)
        
        # Pink elixir
        pink_lower = np.array([140, 50, 50])
        pink_upper = np.array([170, 255, 255])
        mask = cv2.inRange(hsv, pink_lower, pink_upper)
        
        fill_ratio = np.sum(mask > 0) / mask.size
        elixir = int(fill_ratio * 10)
        return min(max(elixir, 0), 10)
    
    def get_game_state(self, img: np.ndarray) -> dict:
        """Get current game state from image."""
        if self.perception and img is not None:
            try:
                return self.perception.get_state(img)
            except:
                pass
        
        # Fallback simple state
        elixir = self.detect_elixir(img)
        return {
            'unit_infos': [],
            'elixir': elixir,
            'time': 90,
            'cards': [0, 0, 0, 0, 0],
        }
    
    def state_to_grid(self, state: dict) -> torch.Tensor:
        """Convert game state to grid observation."""
        grid = np.zeros((STATE_FEATURES, GRID_ROWS, GRID_COLS), dtype=np.float32)
        
        # Units
        for unit in state.get('unit_infos', []):
            if unit is None:
                continue
            
            xy = unit.get('xy')
            if xy is None:
                continue
            
            x = float(xy[0]) if hasattr(xy, '__getitem__') else 0
            y = float(xy[1]) if hasattr(xy, '__getitem__') else 0
            
            col = int(np.clip(x * GRID_COLS / 18, 0, GRID_COLS - 1))
            row = int(np.clip(y * GRID_ROWS / 32, 0, GRID_ROWS - 1))
            
            cls = int(unit.get('cls') or 0)
            bel = int(unit.get('bel') or 0)
            
            grid[min(cls, 9), row, col] = 1.0
            grid[10, row, col] = float(bel == 0)
            grid[11, row, col] = float(bel == 1)
        
        # Elixir
        elixir = state.get('elixir') or 5
        grid[13, :, :] = float(elixir) / 10.0
        
        # Time
        game_time = state.get('time') or 90
        grid[14, :, :] = float(game_time) / 180.0
        
        return torch.from_numpy(grid).unsqueeze(0).to(self.device)
    
    def select_action(self, state: dict, temperature: float = 1.0) -> tuple:
        """Select action using the trained actor."""
        # Check elixir - don't act if too low
        elixir = state.get('elixir') or 0
        if elixir < MIN_ELIXIR_TO_ACT:
            return (0, None, None, None)  # Wait for more elixir
        
        with torch.no_grad():
            obs = self.state_to_grid(state)
            
            out = self.model.observe_step(obs, self.prev_action, (self.deter, self.stoch))
            self.deter = out['deter']
            self.stoch = out['stoch']
            
            features = self.model.get_features(self.deter, self.stoch)
            action_logits = self.model.actor(features)
            
            if temperature != 1.0:
                action_logits = action_logits / temperature
            
            probs = F.softmax(action_logits, dim=-1)
            action_idx = torch.multinomial(probs, 1).item()
            
            self.prev_action = F.one_hot(
                torch.tensor([action_idx], device=self.device), 
                ACTION_DIM
            ).float()
            
            if action_idx == 0:
                return (0, None, None, None)
            
            action_idx -= 1
            card_idx = action_idx // N_POSITIONS
            pos_idx = action_idx % N_POSITIONS
            pos_y = pos_idx // 3
            pos_x = pos_idx % 3
            
            return (action_idx + 1, card_idx, pos_x, pos_y)
    
    def execute_action(self, card_idx: int, pos_x: int, pos_y: int):
        """Execute action via ADB taps."""
        if card_idx is None:
            return
        
        card_x, card_y = CARD_POSITIONS[card_idx]
        deploy_x, deploy_y = DEPLOY_GRID[pos_y][pos_x]
        
        # Tap card
        subprocess.run(['adb', 'shell', 'input', 'tap', str(card_x), str(card_y)],
                      capture_output=True, timeout=1)
        time.sleep(0.1)
        
        # Tap deploy position
        subprocess.run(['adb', 'shell', 'input', 'tap', str(deploy_x), str(deploy_y)],
                      capture_output=True, timeout=1)


def main(args):
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              Live Play - DreamerV3 Clash Royale Agent                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  The agent will:                                                             ║
║  • WAIT until a battle starts                                                ║
║  • Play during active battle                                                 ║
║  • STOP when game ends (Victory/Defeat)                                      ║
║  • Reset and wait for next game                                              ║
║                                                                              ║
║  Press Ctrl+C to stop                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Check ADB
    result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
    if 'emulator' not in result.stdout and 'device' not in result.stdout.split('\n')[1]:
        print("❌ No ADB device found!")
        return
    print("✓ ADB connected")
    
    # Load agent
    agent = LiveAgent(args.checkpoint)
    
    print("\n⏳ Waiting for battle to start...")
    print("   Start a Training Camp or Friendly Battle\n")
    
    current_state = GameState.MENU
    battle_step = 0
    total_actions = 0
    games_played = 0
    
    try:
        while True:
            # Capture screen
            img = agent.capture_screen()
            if img is None:
                time.sleep(0.5)
                continue
            
            # Detect game state
            detected_state = agent.game_detector.detect_state(img)
            
            # State transitions
            if detected_state != current_state:
                if detected_state == GameState.BATTLE:
                    print(f"\n🎮 BATTLE STARTED! (Game #{games_played + 1})")
                    agent.reset_state()
                    battle_step = 0
                elif detected_state == GameState.END_SCREEN:
                    print(f"\n🏁 GAME ENDED! Actions this game: {battle_step}")
                    games_played += 1
                    print(f"   Total games: {games_played}, Total actions: {total_actions}")
                    print("\n⏳ Waiting for next battle...")
                elif detected_state == GameState.MENU:
                    if current_state == GameState.END_SCREEN:
                        print("   Back to menu")
                
                current_state = detected_state
            
            # Only play during battle
            if current_state == GameState.BATTLE:
                # Get state
                state = agent.get_game_state(img)
                elixir = state.get('elixir', 0)
                n_units = len(state.get('unit_infos', []))
                
                # Select action
                action_idx, card_idx, pos_x, pos_y = agent.select_action(
                    state, 
                    temperature=args.temperature
                )
                
                battle_step += 1
                
                # Debug output
                if args.debug:
                    action_str = f"Card {card_idx+1}@({pos_x},{pos_y})" if card_idx is not None else "Wait"
                    print(f"  [{battle_step:3d}] Elixir={elixir:2d} Units={n_units:2d} → {action_str}")
                
                # Execute if not no-op
                if card_idx is not None:
                    agent.execute_action(card_idx, pos_x, pos_y)
                    total_actions += 1
                
                time.sleep(args.interval)
            else:
                # Slower polling when not in battle
                time.sleep(1.0)
    
    except KeyboardInterrupt:
        print(f"\n\n✓ Stopped")
        print(f"  Games played: {games_played}")
        print(f"  Total actions: {total_actions}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--debug", action="store_true")
    
    main(parser.parse_args())
