#!/usr/bin/env python3
"""Synthetic gRPC client to smoke-test the remote FrameService without an emulator.

This helps validate:
- gRPC connectivity and message sizing
- JPEG decode path
- action cadence vs. stream FPS behavior
- perception stride behavior on the server

Example (same machine as server):
  python3 scripts/rpc_smoketest.py --target 127.0.0.1:50051 --fps 30 --action-hz 4 --jpeg
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Ensure repo imports work when running as a script (without installing the package).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cr.rpc.v1.client import FrameServiceClient, RpcClientConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke-test client for FrameService")
    p.add_argument("--target", type=str, required=True, help="host:port")
    p.add_argument("--fps", type=float, default=30.0, help="Send frame rate")
    p.add_argument("--action-hz", type=float, default=4.0, help="Request actions at this rate")
    p.add_argument("--duration-s", type=float, default=10.0, help="How long to run")
    p.add_argument("--deadline-ms", type=int, default=10000, help="Per-RPC deadline")
    p.add_argument("--max-inflight", type=int, default=1, help="Max in-flight RPCs")
    p.add_argument("--jpeg", action="store_true", help="Send JPEG frames")
    p.add_argument("--jpeg-quality", type=int, default=70, help="JPEG quality")
    p.add_argument("--send-width", type=int, default=576, help="Resize width before send")
    p.add_argument("--send-height", type=int, default=1280, help="Resize height before send")
    p.add_argument("--image", type=Path, default=None, help="Optional image to loop (BGR)")
    p.add_argument("--want-action", action="store_true", help="Actually request actions")
    return p.parse_args()


def _load_or_make_frame(args: argparse.Namespace) -> np.ndarray:
    if args.image is not None:
        img = cv2.imread(str(args.image))
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {args.image}")
        frame = img
    else:
        frame = np.zeros((args.send_height, args.send_width, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "FrameService smoke test",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if frame.shape[1] != args.send_width or frame.shape[0] != args.send_height:
        frame = cv2.resize(frame, (int(args.send_width), int(args.send_height)), interpolation=cv2.INTER_AREA)
    return frame


def _encode_frame(frame_bgr: np.ndarray, jpeg: bool, jpeg_quality: int) -> tuple[bytes, str]:
    if not jpeg:
        return frame_bgr.tobytes(), "BGR"
    q = int(np.clip(jpeg_quality, 1, 100))
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError("cv2.imencode(.jpg) failed")
    return buf.tobytes(), "JPEG"


async def main_async(args: argparse.Namespace) -> None:
    cfg = RpcClientConfig(target=args.target, deadline_ms=args.deadline_ms, max_inflight=args.max_inflight)
    client = FrameServiceClient(cfg)
    await client.connect()

    frame = _load_or_make_frame(args)
    interval = 1.0 / max(0.1, float(args.fps))
    action_interval = 1.0 / max(0.1, float(args.action_hz))

    start = time.time()
    frame_id = 0
    last_action = 0.0
    latencies = []

    try:
        while True:
            now = time.time()
            if (now - start) >= float(args.duration_s):
                break

            want_action_now = bool(args.want_action and ((now - last_action) >= action_interval or frame_id == 0))
            payload, color_model = _encode_frame(frame, bool(args.jpeg), int(args.jpeg_quality))

            t0 = time.time()
            resp = await client.process_frame(
                frame_id=frame_id,
                timestamp=now,
                width=int(frame.shape[1]),
                height=int(frame.shape[0]),
                channels=3,
                frame_bgr=payload,
                want_action=want_action_now,
                color_model=color_model,
            )
            dt_ms = (time.time() - t0) * 1000.0
            latencies.append(dt_ms)

            if want_action_now:
                last_action = now

            if frame_id % max(1, int(args.fps)) == 0:
                p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
                p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
                print(
                    f"frame={resp.frame_id} want_action={int(want_action_now)} "
                    f"resp_ms={resp.latency_ms:.1f} client_ms={dt_ms:.1f} "
                    f"p50={p50:.1f} p95={p95:.1f} done={int(resp.done)} reward={resp.reward:.3f}"
                )

            frame_id += 1
            sleep_for = interval - (time.time() - now)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
    finally:
        await client.close()


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
