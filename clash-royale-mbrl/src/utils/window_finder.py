#!/usr/bin/env python
"""Utility to find and capture the scrcpy/Android window on macOS."""
from __future__ import annotations

from typing import Optional, Dict


def find_window_bounds(window_title: str = "Android") -> Optional[Dict[str, int]]:
    """
    Find window bounds by title on macOS using Quartz.
    
    Returns:
        Dict with keys: left, top, width, height (mss format)
        None if window not found
    """
    try:
        import Quartz
        
        # Get all windows
        window_list = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID
        )
        
        for window in window_list:
            name = window.get('kCGWindowName', '')
            owner = window.get('kCGWindowOwnerName', '')
            
            # Check if window title matches
            if window_title.lower() in (name or '').lower() or window_title.lower() in (owner or '').lower():
                bounds = window.get('kCGWindowBounds', {})
                if bounds:
                    return {
                        "left": int(bounds.get('X', 0)),
                        "top": int(bounds.get('Y', 0)),
                        "width": int(bounds.get('Width', 0)),
                        "height": int(bounds.get('Height', 0))
                    }
        
        # Also try matching "scrcpy" directly
        for window in window_list:
            owner = window.get('kCGWindowOwnerName', '')
            if 'scrcpy' in (owner or '').lower():
                bounds = window.get('kCGWindowBounds', {})
                if bounds and bounds.get('Width', 0) > 100:  # Skip tiny windows
                    return {
                        "left": int(bounds.get('X', 0)),
                        "top": int(bounds.get('Y', 0)),
                        "width": int(bounds.get('Width', 0)),
                        "height": int(bounds.get('Height', 0))
                    }
                    
    except ImportError:
        print("Quartz not available. Install pyobjc: pip install pyobjc-framework-Quartz")
    except Exception as e:
        print(f"Error finding window: {e}")
    
    return None


def list_all_windows():
    """List all visible windows for debugging."""
    try:
        import Quartz
        
        window_list = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID
        )
        
        print("Visible windows:")
        for window in window_list:
            name = window.get('kCGWindowName', '')
            owner = window.get('kCGWindowOwnerName', '')
            bounds = window.get('kCGWindowBounds', {})
            if bounds.get('Width', 0) > 50 and bounds.get('Height', 0) > 50:
                print(f"  [{owner}] '{name}' - {int(bounds.get('Width', 0))}x{int(bounds.get('Height', 0))} at ({int(bounds.get('X', 0))}, {int(bounds.get('Y', 0))})")
    except ImportError:
        print("Quartz not available")


def print_window_info(window_title: str = "Android"):
    """Print window info for debugging."""
    bounds = find_window_bounds(window_title)
    if bounds:
        print(f"Found window '{window_title}':")
        print(f"  Position: ({bounds['left']}, {bounds['top']})")
        print(f"  Size: {bounds['width']}x{bounds['height']}")
        print(f"\nFor EmulatorConfig, use:")
        print(f'  capture_region={{"left": {bounds["left"]}, "top": {bounds["top"]}, "width": {bounds["width"]}, "height": {bounds["height"]}}}')
    else:
        print(f"Window '{window_title}' not found.")
        print("\nListing all visible windows:")
        list_all_windows()
        print("\nMake sure scrcpy is running: scrcpy --window-title 'Android' --stay-awake")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Find Android/scrcpy window position")
    parser.add_argument("--title", default="Android", help="Window title to search for")
    parser.add_argument("--list", action="store_true", help="List all windows")
    args = parser.parse_args()
    
    if args.list:
        list_all_windows()
    else:
        print_window_info(args.title)
