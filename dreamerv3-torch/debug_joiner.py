
import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add the project root to path
sys.path.append(str(Path(__file__).parent))

from clash_env.pixel_env import PixelClashEnv

def debug_device(device_id):
    print(f"\n--- Debugging Device: {device_id} ---")
    env = PixelClashEnv(device_id)
    img = env.capture_screen()
    if img is None:
        print("Failed to capture screen.")
        return

    # Check State
    state = env.detect_state(img)
    print(f"Current Detected State: {state}")
    
    # Run OCR manually
    reader = env._get_reader()
    results = reader.readtext(img)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    print("\nDetected Text Blocks:")
    print("-" * 60)
    for (bbox, text, prob) in results:
        t_low = text.lower()
        if prob < 0.2: continue # Ignore noise
        
        # Calculate color in ROI
        h_h, w_w = hsv.shape[:2]
        y1, y2 = max(0, int(bbox[0][1] - 5)), min(h_h, int(bbox[2][1] + 5))
        x1, x2 = max(0, int(bbox[0][0] - 5)), min(w_w, int(bbox[1][0] + 5))
        
        if y2 > y1 and x2 > x1:
            roi = hsv[y1:y2, x1:x2]
            pixel_h = np.percentile(roi[:,:,0], 90)
            pixel_s = np.percentile(roi[:,:,1], 90)
            pixel_v = np.percentile(roi[:,:,2], 90)
        else:
            pixel_h, pixel_s, pixel_v = 0, 0, 0
            
        is_yellow = (5 < pixel_h < 50 and pixel_s > 40)
        is_purple = (115 < pixel_h < 165 and pixel_s > 40)
        
        marker = ""
        if is_yellow: marker = "[YELLOW]"
        elif is_purple: marker = "[PURPLE]"
        
        print(f"[{prob:.2f}] '{text}' at {bbox[0]} {marker}")
        print(f"      H={pixel_h:.1f}, S={pixel_s:.1f}, V={pixel_v:.1f}")

if __name__ == "__main__":
    # Test Joiner 1 (Device 2)
    device_id = "127.0.0.1:26656"
    debug_device(device_id)
