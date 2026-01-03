#!/usr/bin/env python3
"""
Live play using paper-style model (delay + continuous position + card).

Key difference: Model predicts WHEN to act via delay prediction.
- delay ≈ 0: act now
- delay > threshold: wait

Usage:
    python play_paper.py --checkpoint logs/paper_XXXXX/best.pt
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

sys.path.insert(0, str(Path(__file__).parent))
from train_paper import PaperDreamer, GRID_ROWS, GRID_COLS, STATE_FEATURES, N_CARDS, MAX_DELAY


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

# Arena bounds (pixel coordinates for 1080x2400)
ARENA_LEFT = 60
ARENA_RIGHT = 1020
ARENA_TOP = 450    # Enemy side (top)
ARENA_BOTTOM = 1850  # Our side (deployable area starts around y=1200)

# Deployable area (our side only)
DEPLOY_TOP = 1200
DEPLOY_BOTTOM = 1800

ELIXIR_ROI = (200, 2300, 1050, 2400)
DELAY_THRESHOLD = 3  # If predicted delay < this, execute action


class GameDetector:
    """Detect game state (menu/battle/end)."""
    
    def __init__(self):
        self.battle_frame_count = 0
    
    def detect_state(self, img: np.ndarray) -> GameState:
        if img is None:
            return GameState.MENU
        
        # Check for battle (elixir bar visible)
        if self._detect_battle(img):
            self.battle_frame_count += 1
            
            # Only check for end screen after some battle frames
            if self.battle_frame_count > 10:
                if self._detect_end_screen(img):
                    return GameState.END_SCREEN
            
            return GameState.BATTLE
        
        # Reset battle counter if not in battle
        if self.battle_frame_count > 0:
            if self._detect_end_screen(img):
                self.battle_frame_count = 0
                return GameState.END_SCREEN
        
        self.battle_frame_count = 0
        return GameState.MENU
    
    def _detect_battle(self, img: np.ndarray) -> bool:
        x1, y1, x2, y2 = ELIXIR_ROI
        if y2 > img.shape[0] or x2 > img.shape[1]:
            return False
        
        elixir_bar = img[y1:y2, x1:x2]
        hsv = cv2.cvtColor(elixir_bar, cv2.COLOR_BGR2HSV)
        
        pink_lower = np.array([140, 50, 50])
        pink_upper = np.array([170, 255, 255])
        mask = cv2.inRange(hsv, pink_lower, pink_upper)
        
        pink_ratio = np.sum(mask > 0) / mask.size
        return pink_ratio > 0.03
    
    def _detect_end_screen(self, img: np.ndarray) -> bool:
        x1, y1, x2, y2 = ELIXIR_ROI
        if y2 > img.shape[0]:
            return False
        
        elixir_bar = img[y1:y2, x1:x2]
        hsv = cv2.cvtColor(elixir_bar, cv2.COLOR_BGR2HSV)
        
        pink_lower = np.array([140, 50, 50])
        pink_upper = np.array([170, 255, 255])
        mask = cv2.inRange(hsv, pink_lower, pink_upper)
        
        if np.sum(mask > 0) / mask.size > 0.03:
            return False
        
        roi = img[700:950, 250:830]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        gold_lower = np.array([15, 100, 100])
        gold_upper = np.array([35, 255, 255])
        gold_mask = cv2.inRange(hsv_roi, gold_lower, gold_upper)
        
        blue_lower = np.array([100, 100, 100])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv_roi, blue_lower, blue_upper)
        
        gold_ratio = np.sum(gold_mask > 0) / gold_mask.size
        blue_ratio = np.sum(blue_mask > 0) / blue_mask.size
        
        return gold_ratio > 0.25 or blue_ratio > 0.15


class LiveAgent:
    """Agent using paper-style model."""
    
    def __init__(self, checkpoint_path: str, device: str = None):
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
        self.model = PaperDreamer(device=self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"✓ Loaded model from {checkpoint_path}")
        
        self.reset_state()
        self.game_detector = GameDetector()
    
    def reset_state(self):
        self.deter, self.stoch = self.model.initial_state(1)
        self.prev_action_embed = torch.zeros(1, self.model.action_embed_dim, device=self.device)
        print("  [State reset for new game]")
    
    def capture_screen(self) -> np.ndarray:
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
        except:
            return None
    
    def detect_elixir(self, img: np.ndarray) -> int:
        if img is None:
            return 0
        
        x1, y1, x2, y2 = ELIXIR_ROI
        if y2 > img.shape[0] or x2 > img.shape[1]:
            return 5
        
        elixir_bar = img[y1:y2, x1:x2]
        hsv = cv2.cvtColor(elixir_bar, cv2.COLOR_BGR2HSV)
        
        pink_lower = np.array([140, 50, 50])
        pink_upper = np.array([170, 255, 255])
        mask = cv2.inRange(hsv, pink_lower, pink_upper)
        
        fill_ratio = np.sum(mask > 0) / mask.size
        elixir = int(fill_ratio * 10)
        return min(max(elixir, 0), 10)
    
    def get_game_state(self, img: np.ndarray) -> dict:
        state = {
            'elixir': self.detect_elixir(img),
            'time': 90,
            'unit_infos': [],
        }
        return state
    
    def state_to_grid(self, state: dict) -> torch.Tensor:
        grid = np.zeros((STATE_FEATURES, GRID_ROWS, GRID_COLS), dtype=np.float32)
        
        for unit in state.get('unit_infos', []):
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
        
        elixir = state.get('elixir') or 5
        grid[13, :, :] = float(elixir) / 10.0
        
        game_time = state.get('time') or 90
        grid[14, :, :] = float(game_time) / 180.0
        
        return torch.from_numpy(grid).unsqueeze(0).to(self.device)
    
    def select_action(self, state: dict) -> tuple:
        """
        Select action using paper-style model.
        Returns: (should_act, card_idx, screen_x, screen_y, delay)
        """
        with torch.no_grad():
            obs = self.state_to_grid(state)
            
            out = self.model.observe_step(obs, self.prev_action_embed, (self.deter, self.stoch))
            self.deter = out['deter']
            self.stoch = out['stoch']
            
            features = self.model.get_features(self.deter, self.stoch)
            
            # Predict action
            delay, position, card, _ = self.model.predict_action(features, sample=True)
            
            delay_val = delay.item()
            pos_x = position[0, 0].item()  # Normalized [0,1]
            pos_y = position[0, 1].item()
            card_idx = card.item()  # 0-4 (0 = no card)
            
            # Update action embedding
            self.prev_action_embed = self.model.embed_action(
                delay.unsqueeze(0),
                position,
                card.unsqueeze(0)
            )
            
            # Decide whether to act based on delay
            should_act = delay_val < DELAY_THRESHOLD and card_idx > 0
            
            if should_act:
                # Convert normalized position to screen coordinates
                # Only deploy in our half (DEPLOY_TOP to DEPLOY_BOTTOM)
                screen_x = int(ARENA_LEFT + pos_x * (ARENA_RIGHT - ARENA_LEFT))
                screen_y = int(DEPLOY_TOP + pos_y * (DEPLOY_BOTTOM - DEPLOY_TOP))
                
                return (True, card_idx - 1, screen_x, screen_y, delay_val)
            else:
                return (False, None, None, None, delay_val)
    
    def execute_action(self, card_idx: int, screen_x: int, screen_y: int):
        """Execute action via ADB taps."""
        card_x, card_y = CARD_POSITIONS[card_idx]
        
        # Tap card
        subprocess.run(['adb', 'shell', 'input', 'tap', str(card_x), str(card_y)],
                      capture_output=True, timeout=1)
        time.sleep(0.1)
        
        # Tap deploy position
        subprocess.run(['adb', 'shell', 'input', 'tap', str(screen_x), str(screen_y)],
                      capture_output=True, timeout=1)


def main(args):
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║          Paper-Style Live Play: Delay + Continuous Position                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Model predicts:                                                             ║
║  • delay: frames until action (< 3 = act now)                                ║
║  • position: continuous (x, y) in arena                                      ║
║  • card: which card to play (0=wait, 1-4=cards)                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Check ADB
    result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
    if 'device' not in result.stdout or result.stdout.count('\n') < 3:
        print("❌ No ADB device found!")
        print("   Connect your Android device/emulator first")
        return
    print("✓ ADB connected")
    
    # Load agent
    agent = LiveAgent(args.checkpoint)
    
    # Main loop
    current_state = GameState.MENU
    games_played = 0
    total_actions = 0
    battle_step = 0
    
    print("\n⏳ Waiting for battle to start...")
    
    try:
        while True:
            img = agent.capture_screen()
            if img is None:
                time.sleep(0.5)
                continue
            
            detected_state = agent.game_detector.detect_state(img)
            
            if detected_state != current_state:
                if detected_state == GameState.BATTLE:
                    print(f"\n🎮 BATTLE STARTED! (Game #{games_played + 1})")
                    agent.reset_state()
                    battle_step = 0
                elif detected_state == GameState.END_SCREEN:
                    print(f"\n🏁 GAME ENDED! Actions: {total_actions}")
                    games_played += 1
                    print(f"   Total games: {games_played}")
                    print("\n⏳ Waiting for next battle...")
                
                current_state = detected_state
            
            if current_state == GameState.BATTLE:
                state = agent.get_game_state(img)
                elixir = state.get('elixir', 0)
                
                should_act, card_idx, screen_x, screen_y, delay = agent.select_action(state)
                
                battle_step += 1
                
                if args.debug:
                    action_str = f"Card {card_idx+1}@({screen_x},{screen_y})" if should_act else "Wait"
                    print(f"  [{battle_step:3d}] Elixir={elixir:2d} Delay={delay:.1f} → {action_str}")
                
                if should_act:
                    agent.execute_action(card_idx, screen_x, screen_y)
                    total_actions += 1
                
                time.sleep(args.interval)
            else:
                time.sleep(1.0)
    
    except KeyboardInterrupt:
        print(f"\n\n✓ Stopped")
        print(f"  Games: {games_played}, Actions: {total_actions}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--interval", type=float, default=0.3)
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    main(args)
