#!/usr/bin/env python
"""Local loop that captures emulator frames and sends them to the remote FrameService.

- Captures BGR frames via ClashRoyaleEmulatorEnv (ADB/scrcpy screenshot path).
- Sends raw frames over gRPC (want_action=True by default).
- Applies returned action via ADB if provided.

Example:
    python scripts/remote_client_loop.py --target localhost:50051 --want-action
"""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import cv2
import numpy as np

from cr.rpc.v1.client import FrameServiceClient, RpcClientConfig
from cr.rpc.v1 import frame_service_pb2 as pb2
from src.environment.emulator_env import ClashRoyaleEmulatorEnv, EmulatorConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send emulator frames to remote FrameService")
    parser.add_argument("--target", type=str, required=True, help="FrameService address host:port")
    parser.add_argument("--deadline-ms", type=int, default=500, help="Per-RPC deadline in ms")
    parser.add_argument("--max-inflight", type=int, default=2, help="Max in-flight RPCs")
    parser.add_argument("--want-action", action="store_true", help="Request action from server")
    parser.add_argument("--fps", type=float, default=5.0, help="Capture/send rate")
    parser.add_argument("--scrcpy-title", type=str, default="Android", help="scrcpy window title to capture")
    parser.add_argument("--capture-region", type=str, default=None, help="Override capture region as left,top,width,height (pixels)")
    parser.add_argument("--no-adb-fallback", action="store_true", help="Disable adb screencap fallback (fail if scrcpy capture fails)")
    parser.add_argument("--ui-probe-save", action="store_true", help="Save annotated UI probe frames")
    parser.add_argument("--ui-probe-dir", type=str, default=None, help="Directory to save UI probe frames")
    parser.add_argument("--ui-probe-log-every", type=float, default=None, help="Seconds between UI probe logs")
    parser.add_argument("--ui-probe-every-frame", action="store_true", help="Force UI probe log on every frame")
    parser.add_argument("--ok-screen", type=str, default=None, help="Override OK button screen coords as x,y (1080x2400 ref)")
    parser.add_argument("--ok-color-bgr", type=str, default=None, help="Override OK button BGR color as b,g,r")
    parser.add_argument("--ok-tol", type=int, default=None, help="Override OK button color tolerance")
    parser.add_argument("--end-screen-dir", type=str, default=None, help="Directory to save end-screen screenshots before clicking OK")
    return parser.parse_args()


def _parse_pair(text: str) -> tuple[int, int]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError("expected two comma-separated integers")
    return int(parts[0]), int(parts[1])


def _parse_triplet(text: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError("expected three comma-separated integers")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _parse_quad(text: str) -> dict:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("expected four comma-separated integers for left,top,width,height")
    l, t, w, h = map(int, parts)
    return {"left": l, "top": t, "width": w, "height": h}


def _frame_to_bytes(frame: np.ndarray) -> tuple[bytes, int, int, int]:
    h, w, c = frame.shape
    return frame.tobytes(), w, h, c


async def main_async(args: argparse.Namespace) -> None:
    cfg_kwargs = {}
    if args.ui_probe_dir:
        cfg_kwargs["ui_probe_dir"] = Path(args.ui_probe_dir)
    if args.ui_probe_every_frame:
        cfg_kwargs["ui_probe_log_every"] = 0.0
    elif args.ui_probe_log_every is not None:
        cfg_kwargs["ui_probe_log_every"] = args.ui_probe_log_every
    cfg_kwargs["scrcpy_window_title"] = args.scrcpy_title
    if args.capture_region:
        cfg_kwargs["capture_region"] = _parse_quad(args.capture_region)
    if args.no_adb_fallback:
        cfg_kwargs["enable_adb_fallback"] = False
    if args.ok_screen:
        cfg_kwargs["ok_button_screen"] = _parse_pair(args.ok_screen)
    if args.ok_color_bgr:
        cfg_kwargs["ok_button_color_bgr"] = _parse_triplet(args.ok_color_bgr)
    if args.ok_tol is not None:
        cfg_kwargs["ok_button_tol"] = args.ok_tol
    cfg_kwargs["ui_probe_save_frames"] = args.ui_probe_save
    cfg = EmulatorConfig(**cfg_kwargs)
    env = ClashRoyaleEmulatorEnv(cfg)
    cfg = RpcClientConfig(target=args.target, deadline_ms=args.deadline_ms, max_inflight=args.max_inflight)
    client = FrameServiceClient(cfg)
    await client.connect()

    # Navigate to training camp on startup
    print("[remote_client] Navigating to training camp...")
    nav = getattr(env, "navigator", None)
    if nav is not None:
        try:
            nav.start_training_match()
            print("[remote_client] In training camp, starting capture loop")
        except Exception as exc:
            print(f"[remote_client] Navigation failed: {exc}")
    await asyncio.sleep(2.0)  # Wait for match to load

    frame_id = 0
    interval = 1.0 / max(0.1, args.fps)
    action_interval = interval  # Align action cadence with capture rate
    in_battle = False  # Track if we're actually in a battle
    last_action_time = 0.0  # Track last action timestamp

    try:
        while True:
            loop_start = time.time()
            frame = env.get_observation_bgr()
            if args.ui_probe_every_frame:
                try:
                    env.is_match_over(frame)
                except Exception as exc:
                    print(f"[UIProbe] error while probing OK button: {exc}")
            frame_bytes, w, h, c = _frame_to_bytes(frame)
            ts = time.time()
            try:
                resp: pb2.ProcessFrameResponse = await client.process_frame(
                    frame_id=frame_id,
                    timestamp=ts,
                    width=w,
                    height=h,
                    channels=c,
                    frame_bgr=frame_bytes,
                    want_action=args.want_action,
                )
            except Exception as exc:  # network or deadline
                print(f"RPC failed for frame {frame_id}: {exc}")
                await asyncio.sleep(interval)
                frame_id += 1
                continue

            match_over = bool(resp.done or resp.info_num.get("match_over", 0.0) > 0.5)

            # Detect if we're in battle by checking game_time > 0
            # (OCR may fail but extrapolation still works)
            game_time = resp.info_num.get("game_time", 0.0)
            ocr_failed = resp.ocr_failed

            # Debug: print OCR status on first few frames
            if frame_id < 10:
                print(f"  [debug] ocr_failed={ocr_failed} game_time={game_time} info_num keys={list(resp.info_num.keys())}")

            # Battle is active if game_time > 0, even if OCR failed (extrapolation works)
            if game_time > 0:
                in_battle = True
            elif match_over:
                in_battle = False

            # Apply action ONLY if we're in battle and aligned to capture cadence
            current_time = time.time()
            can_act = (current_time - last_action_time) >= action_interval

            if args.want_action and in_battle and can_act and resp.HasField("action"):
                card_idx, gx, gy = resp.action.card_idx, resp.action.grid_x, resp.action.grid_y
                elixir_val = resp.info_num.get("elixir", -1)
                if card_idx > 0:
                    print(
                        f"[remote_client] Applying action card={card_idx} grid=({gx},{gy}) in_battle={in_battle} elixir={elixir_val}"
                    )
                    env.step((card_idx, gx, gy))
                else:
                    print(f"[remote_client] Action has card_idx<=0, skipping tap (elixir={elixir_val})")
                last_action_time = current_time

            # Log minimal diagnostics
            print(
                f"frame={resp.frame_id} latency_ms={resp.latency_ms:.1f} reward={resp.reward:.3f}"
                + (f" in_battle={'YES' if in_battle else 'NO'}")
                + (f" action=({resp.action.card_idx},{resp.action.grid_x},{resp.action.grid_y})" if (resp.HasField("action") and in_battle) else "")
                + (" match_over=1" if match_over else "")
            )

            if match_over:
                # Save end-screen screenshot before clicking OK
                if args.end_screen_dir:
                    end_screen_path = Path(args.end_screen_dir)
                    end_screen_path.mkdir(parents=True, exist_ok=True)
                    timestamp = int(time.time() * 1000)
                    screenshot_path = end_screen_path / f"end_screen_{timestamp}.png"
                    cv2.imwrite(str(screenshot_path), frame)
                    print(f"[remote_client] Saved end-screen screenshot to {screenshot_path}")

                # Let the client-side navigator clear dialogs and start a new Training Camp match.
                nav = getattr(env, "navigator", None)
                if nav is not None:
                    try:
                        nav.dismiss_post_match()
                        nav.start_training_match()
                    except Exception as exc:
                        print(f"Navigator error after match end: {exc}")
                # Give the emulator a moment to load the next match before resuming streaming.
                await asyncio.sleep(2.0)
                frame_id += 1
                continue

            frame_id += 1
            elapsed = time.time() - loop_start
            sleep_for = interval - elapsed
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
    finally:
        await client.close()


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
