#!/usr/bin/env python3
"""Capture KataCR perception snapshots from the emulator."""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "logs" / "perception_checks"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment.emulator_env import ClashRoyaleKataCREnv, EmulatorConfig
from src.perception.katacr_pipeline import KataCRPerceptionResult, KataCRVisionConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture YOLO/OCR perception snapshots.")
    parser.add_argument("--num-snapshots", type=int, default=10,
                        help="Number of frames to capture.")
    parser.add_argument("--duration-seconds", type=float, default=60.0,
                        help="Total duration to span when collecting frames.")
    parser.add_argument("--interval-seconds", type=float, default=None,
                        help="Override spacing between captures. Defaults to duration/(n-1).")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory to store screenshots and logs.")
    parser.add_argument("--adb-path", type=str, default="adb",
                        help="Path to the adb binary.")
    parser.add_argument("--device-serial", type=str, default=None,
                        help="Specific adb device serial (optional).")
    parser.add_argument("--use-window-capture", action="store_true",
                        help="Use scrcpy window capture instead of adb screencap.")
    parser.add_argument("--detector-paths", type=str, nargs="+", default=None,
                        help="Override KataCR detector weights.")
    parser.add_argument("--classifier-path", type=str, default=None,
                        help="Override KataCR card classifier checkpoint directory.")
    parser.add_argument("--canonical-width", type=int, default=576,
                        help="Width to resize captures for KataCR.")
    parser.add_argument("--canonical-height", type=int, default=1280,
                        help="Height to resize captures for KataCR.")
    parser.add_argument("--sleep-before", type=float, default=0.0,
                        help="Optional delay before the first capture (seconds).")
    return parser.parse_args()


def resolve_output_dir(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def build_env(args: argparse.Namespace) -> ClashRoyaleKataCREnv:
    env_cfg = EmulatorConfig(
        adb_path=args.adb_path,
        device_serial=args.device_serial,
        canonical_width=args.canonical_width,
        canonical_height=args.canonical_height,
        use_adb_capture_only=not args.use_window_capture,
    )
    katacr_cfg = KataCRVisionConfig()
    if args.detector_paths:
        katacr_cfg.detector_paths = [Path(p).expanduser() for p in args.detector_paths]
    if args.classifier_path:
        katacr_cfg.classifier_path = Path(args.classifier_path).expanduser()
    env = ClashRoyaleKataCREnv(env_cfg, katacr_cfg)
    env.reset()
    return env


def card_names_from_indices(indices: List[int], idx2card: Optional[dict]) -> List[str]:
    if idx2card is None:
        return [str(idx) for idx in indices]

    resolved = []
    for idx in indices:
        name = None
        if isinstance(idx2card, dict):
            name = idx2card.get(idx)
            if name is None:
                name = idx2card.get(str(idx))
        resolved.append(name if name is not None else str(idx))
    return resolved


def record_snapshot(
    result: KataCRPerceptionResult,
    env: ClashRoyaleKataCREnv,
    run_dir: Path,
    snap_idx: int,
    fps: Optional[float],
    frame_bgr,
    log_handle,
) -> None:
    info = result.info
    state = result.state
    snapshot_name = f"snapshot_{snap_idx:02d}.png"
    snapshot_path = run_dir / snapshot_name
    info["arena"].show_box(show_conf=True, save_path=str(snapshot_path))

    fullframe_name = f"frame_{snap_idx:02d}.png"
    fullframe_path = run_dir / fullframe_name
    cv2.imwrite(str(fullframe_path), frame_bgr)

    idx2card = info.get("idx2card")
    cards_numeric = state.get("cards", [])
    cards_named = card_names_from_indices(cards_numeric, idx2card)
    if info.get("cards") and all(name.isdigit() for name in cards_named):
        cards_named = info["cards"]
    reward_builder = env.katacr.reward_builder
    record = {
        "index": snap_idx,
        "wall_time": datetime.now().isoformat(),
        "game_time": info.get("time"),
        "fps": fps,
        "reward": result.reward,
        "elixir": state.get("elixir"),
        "cards_numeric": cards_numeric,
        "cards_named": cards_named,
        "raw_card_names": info.get("cards"),
        "tower_hp": reward_builder.hp_tower.tolist(),
        "king_tower_hp": reward_builder.hp_king_tower.tolist(),
        "screenshot": str(snapshot_path.relative_to(REPO_ROOT)),
        "raw_frame": str(fullframe_path.relative_to(REPO_ROOT)),
    }
    log_handle.write(json.dumps(record) + "\n")
    log_handle.flush()
    print(f"Saved snapshot {snap_idx:02d} -> {snapshot_path}")


def main():
    args = parse_args()
    if args.num_snapshots <= 0:
        raise ValueError("--num-snapshots must be positive")

    interval = args.interval_seconds
    if interval is None:
        interval = 0.0 if args.num_snapshots == 1 else args.duration_seconds / max(1, args.num_snapshots - 1)

    output_root = resolve_output_dir(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "perception_log.jsonl"

    print(f"Output directory: {run_dir}")
    print(f"Snapshots: {args.num_snapshots}, interval: {interval:.2f}s")

    env = build_env(args)
    if args.sleep_before > 0:
        time.sleep(args.sleep_before)

    with log_path.open("w", encoding="utf-8") as log_handle:
        for idx in range(args.num_snapshots):
            loop_start = time.time()
            result, fps, frame = env.capture_state()
            record_snapshot(result, env, run_dir, idx, fps, frame, log_handle)

            if idx == args.num_snapshots - 1:
                break
            elapsed = time.time() - loop_start
            sleep_time = max(0.0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    print(f"Snapshots and log saved under {run_dir}")


if __name__ == "__main__":
    main()
