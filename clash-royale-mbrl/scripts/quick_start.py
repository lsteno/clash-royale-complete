#!/usr/bin/env python3
"""
Quick Start Script for Clash Royale Training.

This script:
1. Takes screenshots via ADB
2. Sends tap commands to play cards
3. Automates training camp matches

USAGE:
    python scripts/quick_start.py --test-tap
    python scripts/quick_start.py --start-match
    python scripts/quick_start.py --train
"""
import subprocess
import time
import argparse
import os
from pathlib import Path

# Screen coordinates for 1080x2400 resolution
COORDS = {
    # Main menu
    "battle_button": (540, 1800),          # Main battle button
    "menu_hamburger": (980, 320),          # Top-right menu (3 lines)
    
    # Battle selection
    "training_camp": (648, 746),           # Training camp in dropdown
    "party_button": (540, 1320),           # Party mode
    
    # Training camp
    "start_training": (730, 1400),         # Start training button
    
    # Cards (bottom of screen during battle)
    "card1": (302, 2208),
    "card2": (443, 2208),
    "card3": (637, 2208),
    "card4": (778, 2208),
    
    # Arena positions (your side - bottom half)
    "left_bridge": (270, 1260),            # Left bridge
    "right_bridge": (810, 1260),           # Right bridge
    "king_tower_front": (540, 1440),       # In front of king tower
    "left_princess": (270, 1560),          # Left princess tower area
    "right_princess": (810, 1560),         # Right princess tower area
    "back_left": (200, 1750),              # Back left
    "back_right": (880, 1750),             # Back right
    "back_center": (540, 1750),            # Back center
    
    # Post-game
    "ok_button": (540, 1940),              # OK button after game
    "tap_continue": (540, 800),            # Tap to continue
}


def adb_tap(x: int, y: int, delay: float = 0.3):
    """Send tap command via ADB."""
    subprocess.run(["adb", "shell", "input", "tap", str(x), str(y)], 
                   capture_output=True, timeout=5)
    print(f"  TAP ({x}, {y})")
    time.sleep(delay)


def adb_screenshot(output_path: str = "/tmp/cr_screen.png"):
    """Take screenshot via ADB."""
    subprocess.run(["adb", "shell", "screencap", "-p", "/sdcard/screen.png"],
                   capture_output=True, timeout=10)
    subprocess.run(["adb", "pull", "/sdcard/screen.png", output_path],
                   capture_output=True, timeout=10)
    print(f"Screenshot saved: {output_path}")


def tap_named(name: str, delay: float = 0.5):
    """Tap a named position."""
    if name not in COORDS:
        print(f"Unknown position: {name}")
        return
    x, y = COORDS[name]
    print(f"Tapping: {name}")
    adb_tap(x, y, delay)


def play_card(card_slot: int, position: str):
    """
    Play a card at a position.
    card_slot: 1-4
    position: name from COORDS (e.g., "left_bridge", "back_center")
    """
    if card_slot < 1 or card_slot > 4:
        print(f"Invalid card slot: {card_slot}")
        return
    
    if position not in COORDS:
        print(f"Unknown position: {position}")
        return
    
    # Tap card
    tap_named(f"card{card_slot}", delay=0.1)
    # Tap position
    tap_named(position, delay=0.3)


def navigate_to_training_camp():
    """Navigate from main menu to training camp."""
    print("\n=== Navigating to Training Camp ===")
    print("Make sure you're on the main menu!")
    time.sleep(1)
    
    # Tap battle button
    tap_named("battle_button", delay=1.0)
    
    # Tap hamburger menu
    tap_named("menu_hamburger", delay=0.5)
    
    # Tap training camp
    tap_named("training_camp", delay=1.0)
    
    # Start training
    tap_named("start_training", delay=2.0)
    
    print("Training match should be starting...")


def end_match_and_restart():
    """End current match and start new one."""
    print("\n=== Handling Match End ===")
    
    # Wait for OK button
    time.sleep(3.0)
    tap_named("ok_button", delay=2.0)
    
    # Tap to dismiss any screens
    tap_named("tap_continue", delay=0.5)
    
    # Navigate back to training
    navigate_to_training_camp()


def simple_play_loop(duration_seconds: int = 180):
    """
    Simple play loop that randomly plays cards.
    This is a basic demo - the real agent uses the DreamerV3 model.
    """
    import random
    
    print(f"\n=== Playing for {duration_seconds} seconds ===")
    print("This is a DEMO - plays cards randomly")
    print("The real agent uses visual perception + DreamerV3")
    
    positions = ["left_bridge", "right_bridge", "back_left", "back_right", "back_center"]
    start_time = time.time()
    
    while time.time() - start_time < duration_seconds:
        # Random delay (simulating elixir regen)
        time.sleep(random.uniform(2.0, 5.0))
        
        # Pick random card and position
        card = random.randint(1, 4)
        pos = random.choice(positions)
        
        print(f"\nPlaying card {card} at {pos}")
        play_card(card, pos)
        
        # Take screenshot for monitoring
        if random.random() < 0.2:  # 20% chance
            adb_screenshot()


def test_all_taps():
    """Test tap positions one by one."""
    print("\n=== Testing Tap Positions ===")
    print("Watch the emulator!")
    
    for name, (x, y) in COORDS.items():
        input(f"\nPress Enter to tap: {name} ({x}, {y})")
        adb_tap(x, y, delay=0.5)


def main():
    parser = argparse.ArgumentParser(description="Clash Royale Training Quick Start")
    parser.add_argument("--test-tap", action="store_true", 
                        help="Test tap positions interactively")
    parser.add_argument("--start-match", action="store_true",
                        help="Navigate to and start a training match")
    parser.add_argument("--screenshot", action="store_true",
                        help="Take a screenshot")
    parser.add_argument("--play-demo", action="store_true",
                        help="Run demo play loop (random cards)")
    parser.add_argument("--duration", type=int, default=180,
                        help="Duration for demo play in seconds")
    args = parser.parse_args()
    
    # Check ADB connection
    result = subprocess.run(["adb", "devices"], capture_output=True, text=True)
    if "device" not in result.stdout or "emulator" not in result.stdout:
        print("❌ No emulator connected!")
        print("Start Android Studio Emulator first.")
        return
    print("✓ Emulator connected")
    
    if args.test_tap:
        test_all_taps()
    elif args.start_match:
        navigate_to_training_camp()
    elif args.screenshot:
        adb_screenshot()
    elif args.play_demo:
        simple_play_loop(args.duration)
    else:
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    Clash Royale Training - Quick Start                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  STEP 1: Set your deck to Golem Beat-down or 2.6 Hog Cycle                  ║
║                                                                              ║
║  GOLEM DECK (Recommended):                                                   ║
║  • Golem            • Baby Dragon      • Mega Minion                        ║
║  • Lumberjack       • Night Witch      • Lightning                          ║
║  • Tornado          • Barbarian Barrel                                       ║
║                                                                              ║
║  2.6 HOG CYCLE (Alternative):                                                ║
║  • Hog Rider        • Musketeer        • Ice Golem                          ║
║  • Ice Spirit       • Skeletons        • Cannon                             ║
║  • Fireball         • The Log                                                ║
║                                                                              ║
║  STEP 2: Navigate to main menu in Clash Royale                              ║
║                                                                              ║
║  STEP 3: Run one of these commands:                                          ║
║                                                                              ║
║    python scripts/quick_start.py --screenshot      # Take screenshot        ║
║    python scripts/quick_start.py --test-tap        # Test tap positions     ║
║    python scripts/quick_start.py --start-match     # Start training match   ║
║    python scripts/quick_start.py --play-demo       # Run random demo        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
