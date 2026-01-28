import argparse
from pathlib import Path

import cv2
import numpy as np

from .pixel_env import PixelClashEnv


def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    # PixelClashEnv ROI math assumes 1080-wide logic coordinates.
    if img.shape[1] != 1080:
        scale = 1080 / img.shape[1]
        new_h = int(round(img.shape[0] * scale))
        img = cv2.resize(img, (1080, new_h))
    return img


def main():
    parser = argparse.ArgumentParser(description="Debug PixelClashEnv state/battle detection on image files.")
    parser.add_argument("images", nargs="+", help="Image paths (png/jpg) to evaluate.")
    args = parser.parse_args()

    env = PixelClashEnv("debug")
    for img_path in args.images:
        path = Path(img_path)
        img = _load_image(path)

        # Make ROI math follow the provided image.
        env._actual_ui_h = int(img.shape[0])

        rois = env._battle_rois()
        pink = env._pink_column_coverage(img, rois["elixir_band"])
        green = env._green_ratio(img, rois["arena_mid"])
        ok = env._match_template(img, env._state_rois()["ok_button"], "ok_button")
        state = env.detect_state(img)

        print(
            f"{path.name}: state={state} is_battle={env.is_battle(img)} "
            f"pink_cov={pink:.3f} green={green:.3f} ok_tpl={ok:.3f}"
        )


if __name__ == "__main__":
    main()
