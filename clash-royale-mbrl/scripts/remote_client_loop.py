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
from typing import Optional

import numpy as np

from cr.rpc.v1.client import FrameServiceClient, RpcClientConfig
from cr.rpc.v1 import frame_service_pb2 as pb2
from src.environment.emulator_env import ClashRoyaleEmulatorEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send emulator frames to remote FrameService")
    parser.add_argument("--target", type=str, required=True, help="FrameService address host:port")
    parser.add_argument("--deadline-ms", type=int, default=500, help="Per-RPC deadline in ms")
    parser.add_argument("--max-inflight", type=int, default=2, help="Max in-flight RPCs")
    parser.add_argument("--want-action", action="store_true", help="Request action from server")
    parser.add_argument("--fps", type=float, default=5.0, help="Capture/send rate")
    return parser.parse_args()


def _frame_to_bytes(frame: np.ndarray) -> tuple[bytes, int, int, int]:
    h, w, c = frame.shape
    return frame.tobytes(), w, h, c


async def main_async(args: argparse.Namespace) -> None:
    env = ClashRoyaleEmulatorEnv()
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
    in_battle = False  # Track if we're actually in a battle
    last_action_time = 0.0  # Throttle actions to ~1 per second max
    
    try:
        while True:
            loop_start = time.time()
            frame = env.get_observation_bgr()
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
            
            # Detect if we're in battle by checking if OCR succeeded (game time detected)
            game_time = resp.info_num.get("game_time", 0.0)
            ocr_failed = resp.ocr_failed
            
            # Debug: print OCR status on first few frames
            if frame_id < 10:
                print(f"  [debug] ocr_failed={ocr_failed} game_time={game_time} info_num keys={list(resp.info_num.keys())}")
            
            if not ocr_failed and game_time > 0:
                in_battle = True
            elif match_over:
                in_battle = False

            # Apply action ONLY if we're in battle AND throttle to max 1 action/sec
            # (Higher FPS is good for perception but we don't want to spam actions)
            current_time = time.time()
            can_act = (current_time - last_action_time) >= 1.0
            
            if args.want_action and in_battle and can_act and resp.HasField("action") and resp.action.card_idx > 0:
                env.step((resp.action.card_idx, resp.action.grid_x, resp.action.grid_y))
                last_action_time = current_time

            # Log minimal diagnostics
            print(
                f"frame={resp.frame_id} latency_ms={resp.latency_ms:.1f} reward={resp.reward:.3f}"
                + (f" in_battle={'YES' if in_battle else 'NO'}")
                + (f" action=({resp.action.card_idx},{resp.action.grid_x},{resp.action.grid_y})" if (resp.HasField("action") and in_battle) else "")
                + (" match_over=1" if match_over else "")
            )

            if match_over:
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
