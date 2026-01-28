import sys
import subprocess
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from clash_env.pixel_env import PixelClashEnv


def analyze_region(name, img, roi, color_range=None):
    x1, y1, x2, y2 = roi
    h, w = img.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        print(f"[{name}] Empty crop! ROI: {roi}, Img: {img.shape}")
        return

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_mean = hsv[:, :, 0].mean()
    s_mean = hsv[:, :, 1].mean()
    v_mean = hsv[:, :, 2].mean()

    print(f"[{name}] ROI: {roi}")
    print(f"  Mean HSV: H={h_mean:.1f}, S={s_mean:.1f}, V={v_mean:.1f}")

    if color_range:
        h_min, h_max, s_min = color_range
        mask = ((hsv[:, :, 0] > h_min) & (hsv[:, :, 0] < h_max) & (hsv[:, :, 1] > s_min))
        ratio = mask.mean()
        print(f"  Match Ratio: {ratio:.4f}")


def draw_roi(img, roi, color, label):
    x1, y1, x2, y2 = roi
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    res = subprocess.run(["adb", "devices"], capture_output=True, text=True)
    lines = res.stdout.strip().split("\n")[1:]
    devices = [l.split("\t")[0] for l in lines if l.strip() and "offline" not in l]
    if not devices:
        print("No devices found via ADB.")
        return

    device = devices[0]
    print(f"Connecting to {device}...")
    env = PixelClashEnv(device)

    print("Capturing screen...")
    img = env.capture_screen()
    if img is None:
        print("Failed to capture screen.")
        return

    print(f"Image shape: {img.shape}")
    print(f"Active UI height: {env._actual_ui_h}")

    rois = env._state_rois()
    analyze_region("Elixir", img, rois["elixir"], (130, 170, 40))
    analyze_region("OK Button", img, rois["ok_button"], (90, 120, 150))
    analyze_region("Friendly Button", img, rois["friendly_button"], (125, 170, 80))
    analyze_region("Cancel Button", img, rois["cancel_button"], (0, 10, 100))
    analyze_region("Menu Battle", img, rois["menu_battle_button"], (20, 45, 100))

    annotated = img.copy()
    draw_roi(annotated, rois["elixir"], (255, 0, 255), "elixir")
    draw_roi(annotated, rois["ok_button"], (255, 255, 0), "ok")
    draw_roi(annotated, rois["friendly_button"], (200, 0, 200), "friendly")
    draw_roi(annotated, rois["cancel_button"], (0, 0, 255), "cancel")
    draw_roi(annotated, rois["menu_battle_button"], (0, 255, 255), "menu")
    cv2.imwrite("debug_state_rois.png", annotated)
    print("Saved 'debug_state_rois.png' with ROI overlays.")

    env._load_templates()
    scores = {}
    for key in ("ok_button", "cancel_button", "friendly_button", "menu_battle_button", "elixir"):
        scores[key] = env._match_template(img, rois[key], key)
    print("\nTemplate match scores:")
    for key, val in scores.items():
        print(f"  {key}: {val:.3f}")

    current_state = env.detect_state(img)
    print(f"\nDETECTED STATE: {current_state}")


if __name__ == "__main__":
    main()
