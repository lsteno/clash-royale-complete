"""
Android Emulator Environment for Clash Royale on macOS.
Uses scrcpy + mss for screen capture and ADB for action injection.
"""
import subprocess
import time
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import cv2
import mss
import mss.tools


@dataclass
class EmulatorConfig:
    """Configuration for Android emulator connection."""
    adb_path: str = "adb"
    device_serial: Optional[str] = None  # None for default device
    screen_width: int = 1080
    screen_height: int = 2400
    capture_region: Optional[dict] = None  # {"top": 0, "left": 0, "width": 1080, "height": 2400}
    scrcpy_window_title: str = "Android"  # Window title to capture from


class ADBController:
    """
    Android Debug Bridge controller for action injection.
    Sends tap and swipe commands to the emulator.
    """
    
    def __init__(self, config: EmulatorConfig):
        self.config = config
        self._check_adb()
    
    def _check_adb(self):
        """Verify ADB is available and device is connected."""
        try:
            result = subprocess.run(
                [self.config.adb_path, "devices"],
                capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.strip().split('\n')
            devices = [l for l in lines[1:] if l.strip() and 'device' in l]
            if not devices:
                raise RuntimeError("No Android devices/emulators connected. Start Android Studio Emulator first.")
            print(f"Connected devices: {devices}")
        except FileNotFoundError:
            raise RuntimeError(f"ADB not found at {self.config.adb_path}. Install Android SDK or set correct path.")
    
    def _adb_cmd(self, *args) -> str:
        """Execute ADB command."""
        cmd = [self.config.adb_path]
        if self.config.device_serial:
            cmd.extend(["-s", self.config.device_serial])
        cmd.extend(args)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.stdout
    
    def tap(self, x: int, y: int, duration_ms: int = 50):
        """
        Tap at screen coordinates.
        
        Args:
            x: X coordinate (0 = left)
            y: Y coordinate (0 = top)
            duration_ms: Tap duration in milliseconds
        """
        self._adb_cmd("shell", "input", "tap", str(x), str(y))
    
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300):
        """
        Swipe from (x1, y1) to (x2, y2).
        
        Args:
            x1, y1: Start coordinates
            x2, y2: End coordinates  
            duration_ms: Swipe duration
        """
        self._adb_cmd("shell", "input", "swipe", 
                      str(x1), str(y1), str(x2), str(y2), str(duration_ms))
    
    def long_press(self, x: int, y: int, duration_ms: int = 500):
        """Long press at coordinates."""
        self.swipe(x, y, x, y, duration_ms)
    
    def key_event(self, keycode: int):
        """Send key event (e.g., KEYCODE_BACK = 4)."""
        self._adb_cmd("shell", "input", "keyevent", str(keycode))


class ScreenCapture:
    """
    High-performance screen capture using mss.
    Captures the scrcpy mirror window.
    """
    
    def __init__(self, config: EmulatorConfig):
        self.config = config
        self.sct = mss.mss()
        self._monitor = None
    
    def _find_scrcpy_window(self) -> Optional[dict]:
        """
        Find the scrcpy window region.
        On macOS, we can use the full screen or a specific monitor.
        """
        # For simplicity, use full primary monitor or specified region
        if self.config.capture_region:
            return self.config.capture_region
        
        # Default to primary monitor
        return self.sct.monitors[1]  # monitors[0] is "all monitors"
    
    def capture(self) -> np.ndarray:
        """
        Capture current frame from emulator screen.
        
        Returns:
            np.ndarray: BGR image array (H, W, 3)
        """
        if self._monitor is None:
            self._monitor = self._find_scrcpy_window()
        
        screenshot = self.sct.grab(self._monitor)
        frame = np.array(screenshot)
        
        # Convert BGRA to BGR (mss captures with alpha channel)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # Resize to expected dimensions if needed
        if frame.shape[:2] != (self.config.screen_height, self.config.screen_width):
            frame = cv2.resize(frame, (self.config.screen_width, self.config.screen_height))
        
        return frame
    
    def capture_rgb(self) -> np.ndarray:
        """Capture frame in RGB format."""
        frame = self.capture()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class ClashRoyaleEmulatorEnv:
    """
    Gymnasium-compatible environment for Clash Royale via Android emulator.
    """
    
    # Arena grid dimensions (32 tiles wide, 18 tiles tall)
    GRID_WIDTH = 32
    GRID_HEIGHT = 18
    
    # Action space: (card_index, grid_x, grid_y)
    # card_index: 0 = no action, 1-4 = select card
    NUM_CARDS = 4
    
    def __init__(self, config: Optional[EmulatorConfig] = None):
        self.config = config or EmulatorConfig()
        self.adb = ADBController(self.config)
        self.screen = ScreenCapture(self.config)
        
        # Screen regions (normalized 0-1)
        self.arena_region = {
            "top": 0.15,      # Start after top bar
            "bottom": 0.75,   # End before cards
            "left": 0.0,
            "right": 1.0
        }
        self.cards_region = {
            "top": 0.82,
            "bottom": 0.95,
            "left": 0.15,
            "right": 0.85
        }
        
    def get_observation(self) -> np.ndarray:
        """Get current screen frame."""
        return self.screen.capture_rgb()
    
    def get_arena_frame(self) -> np.ndarray:
        """Get only the arena portion of the screen."""
        frame = self.get_observation()
        h, w = frame.shape[:2]
        
        top = int(h * self.arena_region["top"])
        bottom = int(h * self.arena_region["bottom"])
        left = int(w * self.arena_region["left"])
        right = int(w * self.arena_region["right"])
        
        return frame[top:bottom, left:right]
    
    def _grid_to_screen(self, grid_x: int, grid_y: int) -> Tuple[int, int]:
        """Convert arena grid coordinates to screen pixel coordinates."""
        h, w = self.config.screen_height, self.config.screen_width
        
        arena_top = int(h * self.arena_region["top"])
        arena_bottom = int(h * self.arena_region["bottom"])
        arena_left = int(w * self.arena_region["left"])
        arena_right = int(w * self.arena_region["right"])
        
        arena_h = arena_bottom - arena_top
        arena_w = arena_right - arena_left
        
        # Map grid to pixel
        pixel_x = arena_left + int((grid_x / self.GRID_WIDTH) * arena_w)
        pixel_y = arena_top + int((grid_y / self.GRID_HEIGHT) * arena_h)
        
        return pixel_x, pixel_y
    
    def _card_to_screen(self, card_index: int) -> Tuple[int, int]:
        """Get screen coordinates for card selection (1-4)."""
        h, w = self.config.screen_height, self.config.screen_width
        
        cards_top = int(h * self.cards_region["top"])
        cards_bottom = int(h * self.cards_region["bottom"])
        cards_left = int(w * self.cards_region["left"])
        cards_right = int(w * self.cards_region["right"])
        
        cards_w = cards_right - cards_left
        card_width = cards_w // 4
        
        pixel_x = cards_left + int((card_index - 0.5) * card_width)
        pixel_y = (cards_top + cards_bottom) // 2
        
        return pixel_x, pixel_y
    
    def step(self, action: Tuple[int, int, int]) -> np.ndarray:
        """
        Execute action in environment.
        
        Args:
            action: (card_index, grid_x, grid_y)
                - card_index: 0 = no action, 1-4 = deploy card
                - grid_x: 0-31 arena x position
                - grid_y: 0-17 arena y position
        
        Returns:
            observation: Current screen frame after action
        """
        card_idx, grid_x, grid_y = action
        
        if card_idx > 0:
            # First tap: select card
            card_x, card_y = self._card_to_screen(card_idx)
            self.adb.tap(card_x, card_y)
            time.sleep(0.05)
            
            # Second tap: deploy at position
            deploy_x, deploy_y = self._grid_to_screen(grid_x, grid_y)
            self.adb.tap(deploy_x, deploy_y)
        
        # Small delay for game to process
        time.sleep(0.1)
        
        return self.get_observation()
    
    def reset(self):
        """Reset environment (start new battle if needed)."""
        # This would need game-specific logic
        pass


if __name__ == "__main__":
    # Test the emulator connection
    print("Testing Android Emulator Environment...")
    
    config = EmulatorConfig()
    
    try:
        env = ClashRoyaleEmulatorEnv(config)
        print("✓ ADB connection successful")
        
        frame = env.get_observation()
        print(f"✓ Screen capture working: {frame.shape}")
        
        # Save test frame
        cv2.imwrite("/tmp/test_capture.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print("✓ Saved test frame to /tmp/test_capture.png")
        
    except Exception as e:
        print(f"✗ Error: {e}")
