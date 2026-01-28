import argparse
import subprocess
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from clash_env.pixel_env import PixelClashEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "names",
        nargs="+",
        help="Template names to capture (ok_button, cancel_button, friendly_button, menu_battle_button, elixir)",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent.parent / "clash_env" / "templates"),
        help="Output directory for templates",
    )
    args = parser.parse_args()

    res = subprocess.run(["adb", "devices"], capture_output=True, text=True)
    lines = res.stdout.strip().split("\n")[1:]
    devices = [l.split("\t")[0] for l in lines if l.strip() and "offline" not in l]
    if not devices:
        print("No devices found via ADB.")
        return

    device = devices[0]
    print(f"Connecting to {device}...")
    env = PixelClashEnv(device)
    img = env.capture_screen()
    if img is None:
        print("Failed to capture screen.")
        return

    rois = env._state_rois()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.names:
        if name not in rois:
            print(f"Unknown template name: {name}")
            continue
        x1, y1, x2, y2 = rois[name]
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"Empty crop for {name}")
            continue
        out_path = out_dir / f"{name}.png"
        cv2.imwrite(str(out_path), crop)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
