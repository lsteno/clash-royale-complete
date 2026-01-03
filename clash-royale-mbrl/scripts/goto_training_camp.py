#!/usr/bin/env python3
"""
Navigate to Training Camp step by step.
Run this from main menu to test the navigation flow.

Usage:
    python scripts/goto_training_camp.py
    python scripts/goto_training_camp.py --step  # Wait between steps
"""
import subprocess
import time
import argparse

def adb_tap(x: int, y: int):
    """Send tap via ADB."""
    print(f"  Tapping ({x}, {y})...")
    subprocess.run(["adb", "shell", "input", "tap", str(x), str(y)], 
                   capture_output=True, timeout=5)

def adb_screenshot(path: str = "/tmp/cr_nav.png"):
    """Take and save screenshot."""
    subprocess.run(["adb", "shell", "screencap", "-p", "/sdcard/screen.png"], 
                   capture_output=True, timeout=5)
    subprocess.run(["adb", "pull", "/sdcard/screen.png", path], 
                   capture_output=True, timeout=5)
    return path

def wait_or_confirm(step_mode: bool, msg: str):
    """Wait for delay or user confirmation."""
    if step_mode:
        input(f"\n>>> {msg}\nPress Enter to continue...")
    else:
        time.sleep(1.5)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", action="store_true", help="Wait for confirmation between steps")
    args = parser.parse_args()
    
    print("""
╔═══════════════════════════════════════════════════════════╗
║         Navigate to Training Camp                         ║
║  Make sure you're on the MAIN MENU before running!        ║
╚═══════════════════════════════════════════════════════════╝
""")
    
    if args.step:
        input("Press Enter when ready on main menu...")
    else:
        time.sleep(2)
    
    # Step 1: Battle button
    print("\n[Step 1] Tapping BATTLE button (bottom center)...")
    adb_tap(540, 1800)
    wait_or_confirm(args.step, "Battle menu should open. You should see mode options.")
    
    # Take screenshot to verify
    if args.step:
        path = adb_screenshot()
        subprocess.run(["open", path])
    
    # Step 2: Hamburger menu (top right)
    print("\n[Step 2] Tapping hamburger menu (top right, ≡ icon)...")
    adb_tap(980, 320)
    wait_or_confirm(args.step, "Dropdown menu should appear with Training Camp option.")
    
    if args.step:
        path = adb_screenshot()
        subprocess.run(["open", path])
    
    # Step 3: Training Camp option
    print("\n[Step 3] Tapping 'Training Camp' in dropdown...")
    adb_tap(648, 746)
    wait_or_confirm(args.step, "Training Camp screen should show with START button.")
    
    if args.step:
        path = adb_screenshot()
        subprocess.run(["open", path])
    
    # Step 4: Start Training button
    print("\n[Step 4] Tapping 'Start Training' button...")
    adb_tap(730, 1400)
    
    print("\n✓ Navigation complete! Match should be loading...")
    print("If this didn't work, run with --step to debug each tap.")

if __name__ == "__main__":
    main()
