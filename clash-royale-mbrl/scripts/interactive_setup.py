"""
Interactive training setup for Clash Royale on macOS.
This script helps configure and run training with the Android emulator.

REQUIREMENTS:
1. Android Emulator running with Clash Royale installed
2. ADB installed (brew install android-platform-tools)
3. scrcpy installed (brew install scrcpy) - OPTIONAL but recommended

GAME MODE: Training Camp (vs Trainer AI)
DECK: Golem Beat-down (what KataCR was trained on)

Golem Deck (8 cards):
- Golem (8 elixir)
- Baby Dragon (4 elixir)  
- Mega Minion (3 elixir)
- Lumberjack (4 elixir)
- Night Witch (4 elixir)
- Lightning (6 elixir)
- Tornado (3 elixir)
- Barbarian Barrel (2 elixir)

Alternative: 2.6 Hog Cycle (card classifier trained on this):
- Hog Rider (4 elixir)
- Musketeer (4 elixir)
- Ice Golem (2 elixir)
- Ice Spirit (1 elixir)
- Skeletons (1 elixir)
- Cannon (3 elixir)
- Fireball (4 elixir)
- The Log (2 elixir)
"""
import subprocess
import time
import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

# Try imports
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: opencv-python not installed")

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    print("Warning: mss not installed")


@dataclass
class EmulatorConfig:
    """Configuration for your specific emulator setup."""
    # Screen capture region (where your emulator window is)
    # Set these based on your emulator position on screen
    capture_left: int = 0       # X position of emulator window
    capture_top: int = 0        # Y position of emulator window  
    capture_width: int = 540    # Width of emulator window (half of 1080)
    capture_height: int = 1200  # Height of emulator window (half of 2400)
    
    # Emulator's internal resolution
    emulator_width: int = 1080
    emulator_height: int = 2400
    
    # ADB device (None = auto-detect)
    adb_device: Optional[str] = None


class ADBHelper:
    """Helper for ADB commands."""
    
    def __init__(self, device: Optional[str] = None):
        self.device = device
        self._check_adb()
    
    def _check_adb(self) -> bool:
        """Check if ADB is available and devices are connected."""
        try:
            result = subprocess.run(
                ["adb", "devices"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            lines = result.stdout.strip().split('\n')
            devices = [l.split('\t')[0] for l in lines[1:] if 'device' in l]
            
            if not devices:
                print("❌ No Android devices/emulators connected!")
                print("   Start your Android emulator first.")
                return False
            
            print(f"✓ Found devices: {devices}")
            if self.device is None and devices:
                self.device = devices[0]
                print(f"  Using: {self.device}")
            return True
            
        except FileNotFoundError:
            print("❌ ADB not found!")
            print("   Install with: brew install android-platform-tools")
            return False
    
    def _cmd(self, *args) -> subprocess.CompletedProcess:
        """Run ADB command."""
        cmd = ["adb"]
        if self.device:
            cmd.extend(["-s", self.device])
        cmd.extend(args)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    
    def tap(self, x: int, y: int):
        """Tap at absolute coordinates."""
        self._cmd("shell", "input", "tap", str(x), str(y))
        print(f"  TAP: ({x}, {y})")
    
    def tap_relative(self, rx: float, ry: float, width: int = 1080, height: int = 2400):
        """Tap at relative coordinates (0-1)."""
        x = int(rx * width)
        y = int(ry * height)
        self.tap(x, y)
    
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300):
        """Swipe gesture."""
        self._cmd("shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms))
    
    def screenshot(self, output_path: str = "/tmp/screen.png") -> Optional[np.ndarray]:
        """Take screenshot via ADB."""
        self._cmd("shell", "screencap", "-p", "/sdcard/screen.png")
        self._cmd("pull", "/sdcard/screen.png", output_path)
        
        if CV2_AVAILABLE and os.path.exists(output_path):
            return cv2.imread(output_path)
        return None


class ScreenCapture:
    """Capture emulator screen from desktop window."""
    
    def __init__(self, config: EmulatorConfig):
        self.config = config
        self.sct = mss.mss() if MSS_AVAILABLE else None
    
    def get_region(self) -> dict:
        """Get the capture region for mss."""
        return {
            "left": self.config.capture_left,
            "top": self.config.capture_top,
            "width": self.config.capture_width,
            "height": self.config.capture_height,
        }
    
    def capture(self) -> Optional[np.ndarray]:
        """Capture the emulator screen."""
        if not MSS_AVAILABLE:
            return None
        
        screenshot = self.sct.grab(self.get_region())
        frame = np.array(screenshot)
        # Convert BGRA to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame
    
    def find_emulator_window(self):
        """
        Helper to find emulator window position.
        Captures full screen and lets you identify the region.
        """
        if not MSS_AVAILABLE:
            print("mss not available")
            return
        
        # Capture primary monitor
        monitor = self.sct.monitors[1]
        screenshot = self.sct.grab(monitor)
        frame = np.array(screenshot)
        
        # Save for inspection
        cv2.imwrite("/tmp/full_screen.png", frame)
        print(f"Full screen captured to /tmp/full_screen.png")
        print(f"Monitor size: {monitor['width']}x{monitor['height']}")
        print("\nOpen the image and note your emulator window coordinates.")
        print("Update EmulatorConfig with: capture_left, capture_top, capture_width, capture_height")


class ClashRoyaleNavigator:
    """Navigate Clash Royale menus via ADB taps."""
    
    # Screen positions (relative to 1080x2400 resolution)
    POSITIONS = {
        # Main menu
        "battle_button": (0.5, 0.75),      # Battle button
        "cards_button": (0.2, 0.95),       # Cards deck button
        "shop_button": (0.8, 0.95),        # Shop button
        
        # Battle menu
        "training_camp": (0.6, 0.31),      # Training camp option in menu
        "party_mode": (0.5, 0.55),         # Party mode
        "1v1_battle": (0.5, 0.45),         # 1v1 ranked
        
        # Training camp specific
        "start_training": (0.68, 0.58),    # Start training button
        "ok_button": (0.5, 0.81),          # OK/Continue button
        
        # In-game
        "card1": (0.28, 0.92),             # First card slot
        "card2": (0.41, 0.92),             # Second card slot
        "card3": (0.59, 0.92),             # Third card slot
        "card4": (0.72, 0.92),             # Fourth card slot
        
        # Post-game
        "ok_end_game": (0.5, 0.81),        # OK after game ends
        "exit_menu": (0.91, 0.13),         # X button / exit
        
        # Menu navigation
        "hamburger_menu": (0.91, 0.13),    # Three lines menu
    }
    
    def __init__(self, adb: ADBHelper, width: int = 1080, height: int = 2400):
        self.adb = adb
        self.width = width
        self.height = height
    
    def tap_position(self, name: str, delay: float = 0.5):
        """Tap a named position."""
        if name not in self.POSITIONS:
            print(f"Unknown position: {name}")
            return
        
        rx, ry = self.POSITIONS[name]
        print(f"Tapping: {name}")
        self.adb.tap_relative(rx, ry, self.width, self.height)
        time.sleep(delay)
    
    def tap_arena(self, grid_x: int, grid_y: int):
        """
        Tap a position in the arena.
        Arena is roughly the middle 70% of screen vertically.
        grid_x: 0-17 (left to right)
        grid_y: 0-31 (top to bottom of arena)
        
        Your side (bottom): grid_y 16-31
        Enemy side (top): grid_y 0-15
        """
        # Arena bounds (relative)
        arena_left = 0.02
        arena_right = 0.98
        arena_top = 0.12    # Top of arena
        arena_bottom = 0.7292  # Bottom of arena (before cards) (~1750px on 2400px height)
        
        # Map grid to screen
        rx = arena_left + (grid_x / 17) * (arena_right - arena_left)
        ry = arena_top + (grid_y / 31) * (arena_bottom - arena_top)
        
        self.adb.tap_relative(rx, ry, self.width, self.height)
    
    def play_card(self, card_slot: int, grid_x: int, grid_y: int):
        """
        Play a card from slot (1-4) at arena position.
        """
        if card_slot < 1 or card_slot > 4:
            print(f"Invalid card slot: {card_slot}")
            return
        
        # Tap card
        self.tap_position(f"card{card_slot}", delay=0.1)
        # Tap arena position
        self.tap_arena(grid_x, grid_y)
    
    def start_training_match(self):
        """Navigate to and start a training camp match."""
        print("\n=== Starting Training Match ===")
        
        # From main menu
        print("1. Opening battle menu...")
        self.tap_position("battle_button", delay=1.0)
        
        print("2. Selecting training camp...")
        # Tap training camp (might need to scroll or tap menu first)
        self.tap_position("hamburger_menu", delay=0.5)
        self.tap_position("training_camp", delay=1.0)
        
        print("3. Starting match...")
        self.tap_position("start_training", delay=2.0)
        
        print("Match should be starting...")
    
    def end_match_and_restart(self):
        """Handle end of match and start new one."""
        print("\n=== Handling Match End ===")
        
        # Wait for OK button
        time.sleep(2.0)
        self.tap_position("ok_end_game", delay=1.0)
        
        # Tap to dismiss any rewards
        self.adb.tap_relative(0.5, 0.3, self.width, self.height)
        time.sleep(0.5)
        
        # Start new match
        self.start_training_match()


def calibrate_screen_capture():
    """Interactive tool to calibrate screen capture region."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         Screen Capture Calibration                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  We need to know where your Android emulator window is on screen.            ║
║                                                                              ║
║  Your emulator is on the LEFT side of the screen.                            ║
║                                                                              ║
║  Please provide the following measurements (in pixels):                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Get screen info
    if MSS_AVAILABLE:
        sct = mss.mss()
        monitor = sct.monitors[1]
        print(f"Your screen resolution: {monitor['width']}x{monitor['height']}")
    
    print("\nFor Android Studio Emulator, typical window sizes:")
    print("  - Phone (1080x2400 scaled): ~400x890 or ~540x1200")
    print("  - Tablet: varies")
    
    try:
        left = int(input("\nEmulator window LEFT edge (X position, e.g., 0): ") or "0")
        top = int(input("Emulator window TOP edge (Y position, e.g., 25): ") or "25")
        width = int(input("Emulator window WIDTH (e.g., 540): ") or "540")
        height = int(input("Emulator window HEIGHT (e.g., 1200): ") or "1200")
        
        config = EmulatorConfig(
            capture_left=left,
            capture_top=top,
            capture_width=width,
            capture_height=height,
        )
        
        print(f"\nConfiguration:")
        print(f"  capture_left={left}")
        print(f"  capture_top={top}")
        print(f"  capture_width={width}")
        print(f"  capture_height={height}")
        
        # Test capture
        if MSS_AVAILABLE and CV2_AVAILABLE:
            print("\nTesting capture...")
            capture = ScreenCapture(config)
            frame = capture.capture()
            if frame is not None:
                cv2.imwrite("/tmp/emulator_capture.png", frame)
                print(f"✓ Test capture saved to /tmp/emulator_capture.png")
                print("  Please verify this shows your emulator screen!")
        
        return config
        
    except ValueError:
        print("Invalid input. Using defaults.")
        return EmulatorConfig()


def main():
    """Main interactive setup and training."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              Clash Royale Visual-MBRL - Interactive Setup                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  BEFORE RUNNING:                                                             ║
║                                                                              ║
║  1. Start Android Emulator (Android Studio)                                  ║
║  2. Open Clash Royale in the emulator                                        ║
║  3. SET YOUR DECK to Golem Beat-down or 2.6 Hog Cycle                       ║
║  4. Navigate to the main menu                                                ║
║                                                                              ║
║  RECOMMENDED DECK (Golem):                                                   ║
║  Golem, Baby Dragon, Mega Minion, Lumberjack,                               ║
║  Night Witch, Lightning, Tornado, Barbarian Barrel                          ║
║                                                                              ║
║  GAME MODE: Training Camp (vs Trainer AI)                                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Check ADB
    print("\n[1/4] Checking ADB connection...")
    adb = ADBHelper()
    
    # Ask what to do
    print("\nOptions:")
    print("  1. Test ADB taps (verify connection)")
    print("  2. Calibrate screen capture")
    print("  3. Start training match")
    print("  4. Run full training loop")
    print("  5. Take screenshot via ADB")
    print("  q. Quit")
    
    while True:
        choice = input("\nChoice (1-5, q): ").strip().lower()
        
        if choice == 'q':
            break
            
        elif choice == '1':
            print("\nTesting ADB - will tap center of screen in 3 seconds...")
            print("Watch your emulator!")
            time.sleep(3)
            adb.tap(540, 1200)  # Center of 1080x2400
            print("Did you see the tap?")
            
        elif choice == '2':
            config = calibrate_screen_capture()
            
        elif choice == '3':
            print("\nStarting training match in 3 seconds...")
            print("Make sure you're on the Clash Royale main menu!")
            time.sleep(3)
            
            navigator = ClashRoyaleNavigator(adb)
            navigator.start_training_match()
            
        elif choice == '4':
            print("\nTraining loop not yet implemented.")
            print("This would run the DreamerV3 agent with the emulator.")
            
        elif choice == '5':
            print("\nTaking screenshot...")
            frame = adb.screenshot()
            if frame is not None:
                print(f"Screenshot saved: /tmp/screen.png ({frame.shape})")
            else:
                print("Screenshot saved to /tmp/screen.png (install opencv to view)")


if __name__ == "__main__":
    main()
