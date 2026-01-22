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
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import grpc

# Ensure repo imports work when running as a script (without installing the package).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cr.rpc.v1.client import FrameServiceClient, RpcClientConfig
from cr.rpc.v1 import frame_service_pb2 as pb2
from src.environment.emulator_env import ClashRoyaleEmulatorEnv, EmulatorConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send emulator frames to remote FrameService")
    parser.add_argument("--target", type=str, required=True, help="FrameService address host:port")
    # Perception can take >1s per frame (YOLO + OCR + classification + network).
    # Keep a generous default to avoid repeated timeouts; override as needed.
    parser.add_argument("--deadline-ms", type=int, default=10000, help="Per-RPC deadline in ms")
    parser.add_argument("--max-inflight", type=int, default=1, help="Max in-flight RPCs")
    parser.add_argument("--want-action", action="store_true", help="Request action from server")
    parser.add_argument("--fps", type=float, default=15.0, help="Capture/send rate")
    parser.add_argument("--action-hz", type=float, default=2.0, help="How often to request/apply actions")
    parser.add_argument("--send-width", type=int, default=None, help="Resize frames before sending (width px)")
    parser.add_argument("--send-height", type=int, default=None, help="Resize frames before sending (height px)")
    parser.add_argument("--jpeg", action="store_true", help="Send JPEG-compressed frames (recommended over WAN)")
    parser.add_argument("--jpeg-quality", type=int, default=70, help="JPEG quality (lower=faster/smaller)")
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
    parser.add_argument("--capture-debug-dir", type=str, default=None,
                        help="Save capture debug images (full monitor + ROI) to this directory and continue")
    parser.add_argument("--nav-cooldown-s", type=float, default=12.0, help="Minimum seconds between navigation attempts")
    parser.add_argument("--no-battle-recover-s", type=float, default=25.0, help="If not in battle for this long, try to recover")
    parser.add_argument("--recover-back-count", type=int, default=3, help="How many KEYCODE_BACK presses before re-navigation")
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


def _encode_frame(
    frame_bgr: np.ndarray,
    *,
    send_width: Optional[int],
    send_height: Optional[int],
    jpeg: bool,
    jpeg_quality: int,
) -> tuple[bytes, int, int, int, str]:
    if (send_width is None) ^ (send_height is None):
        raise ValueError("Provide both --send-width and --send-height, or neither")
    if send_width is not None and send_height is not None:
        frame_bgr = cv2.resize(frame_bgr, (int(send_width), int(send_height)), interpolation=cv2.INTER_AREA)

    h, w, c = frame_bgr.shape
    if jpeg:
        q = int(np.clip(jpeg_quality, 1, 100))
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            raise RuntimeError("cv2.imencode(.jpg) failed")
        return buf.tobytes(), w, h, 3, "JPEG"
    return frame_bgr.tobytes(), w, h, c, "BGR"


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

    # Capture debugging: dump a full-monitor screenshot with ROI bounds and
    # the actual ROI frame that will be sent to the server. This helps
    # calibrate --capture-region against the scrcpy window on macOS.
    if args.capture_debug_dir:
        try:
            out = Path(args.capture_debug_dir)
            out.mkdir(parents=True, exist_ok=True)
            full = getattr(getattr(env, "screen", None), "capture_full_bgr", lambda: None)()
            bounds = getattr(getattr(env, "screen", None), "get_capture_bounds", lambda: None)()
            roi = env.get_observation_bgr()
            ts = int(time.time() * 1000)
            if full is not None and bounds is not None:
                annotated = full.copy()
                l, t, w, h = int(bounds.get("left", 0)), int(bounds.get("top", 0)), int(bounds.get("width", 0)), int(bounds.get("height", 0))
                cv2.rectangle(annotated, (l, t), (l + w, t + h), (0, 0, 255), 3)
                cv2.putText(
                    annotated,
                    f"capture_region left={l} top={t} width={w} height={h}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imwrite(str(out / f"capture_full_{ts}.png"), annotated)
            cv2.imwrite(str(out / f"capture_roi_{ts}.png"), roi)
            print(f"[remote_client] Saved capture debug images to {out}")
        except Exception as exc:
            print(f"[remote_client] Failed to save capture debug images: {exc}")

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
    action_interval = 1.0 / max(0.1, float(args.action_hz))
    in_battle = False  # Track if we're actually in a battle
    last_confirmed_battle = 0.0
    no_battle_since = time.time()
    last_action_time = 0.0  # Track last action request/apply timestamp
    last_nav_time = 0.0
    consecutive_rpc_failures = 0

    def _can_navigate(now: float) -> bool:
        return (now - last_nav_time) >= float(args.nav_cooldown_s)

    def _recover_to_training(reason: str) -> None:
        nonlocal last_nav_time, no_battle_since, in_battle
        now = time.time()
        nav = getattr(env, "navigator", None)
        if nav is None:
            return
        if not _can_navigate(now):
            return
        print(f"[remote_client] Recovery triggered ({reason}); attempting to return to Training Camp...")
        try:
            # Try to escape any modal screen with a few BACK presses.
            back_n = max(0, int(args.recover_back_count))
            for _ in range(back_n):
                try:
                    env.adb.key_event(4)  # KEYCODE_BACK
                except Exception:
                    pass
                time.sleep(0.4)
            nav.dismiss_post_match()
            nav.start_training_match()
            last_nav_time = now
            no_battle_since = now
            in_battle = False
        except Exception as exc:
            print(f"[remote_client] Recovery navigation failed: {exc}")

    try:
        while True:
            loop_start = time.time()
            frame = env.get_observation_bgr()
            if args.ui_probe_every_frame:
                try:
                    env.is_match_over(frame)
                except Exception as exc:
                    print(f"[UIProbe] error while probing OK button: {exc}")
            frame_bytes, w, h, c, color_model = _encode_frame(
                frame,
                send_width=args.send_width,
                send_height=args.send_height,
                jpeg=bool(args.jpeg),
                jpeg_quality=int(args.jpeg_quality),
            )
            ts = time.time()
            current_time = time.time()
            can_act = (current_time - last_action_time) >= action_interval
            want_action_now = bool(args.want_action and (can_act or frame_id == 0))
            try:
                resp: pb2.ProcessFrameResponse = await client.process_frame(
                    frame_id=frame_id,
                    timestamp=ts,
                    width=w,
                    height=h,
                    channels=c,
                    frame_bgr=frame_bytes,
                    want_action=want_action_now,
                    color_model=color_model,
                )
            except Exception as exc:  # network or deadline
                consecutive_rpc_failures += 1
                if isinstance(exc, grpc.aio.AioRpcError):
                    code = exc.code()
                    details = exc.details()
                    print(f"RPC failed for frame {frame_id}: code={code} details={details}")
                else:
                    print(f"RPC failed for frame {frame_id}: {exc}")

                # If you're using an SSH tunnel and it drops, gRPC will start
                # returning UNAVAILABLE/connection refused. Attempt a reconnect
                # after a few failures to recover automatically.
                if consecutive_rpc_failures >= 3:
                    try:
                        await client.close()
                    except Exception:
                        pass
                    await asyncio.sleep(0.5)
                    try:
                        await client.connect()
                        print("[remote_client] Reconnected gRPC channel")
                        consecutive_rpc_failures = 0
                    except Exception as conn_exc:
                        print(f"[remote_client] Reconnect failed: {conn_exc}")
                await asyncio.sleep(interval)
                frame_id += 1
                continue
            consecutive_rpc_failures = 0

            if resp is None:
                print(f"RPC returned no response for frame {frame_id}")
                await asyncio.sleep(interval)
                frame_id += 1
                continue

            # Combine remote and local end-screen probes. Local probing prevents
            # stale server-side perception states from keeping us "in battle".
            match_over_local = False
            try:
                match_over_local = bool(env.is_match_over(frame))
            except Exception:
                match_over_local = False
            match_over = bool(match_over_local or resp.done or resp.info_num.get("match_over", 0.0) > 0.5)

            # Detect if we're in battle. Time OCR can extrapolate even after the
            # match ends (or on menus), so treat it as *confirming* battle only
            # when OCR succeeds. We keep a short "recently in battle" grace
            # window to tolerate transient OCR failures without tapping menus.
            game_time = resp.info_num.get("game_time", 0.0)
            elixir_val = resp.info_num.get("elixir", -1.0)
            elixir_failed = (resp.info_str.get("elixir_status") == "ocr_failed")
            ocr_failed = resp.ocr_failed

            # Debug: print OCR status on first few frames
            if frame_id < 10:
                print(f"  [debug] ocr_failed={ocr_failed} game_time={game_time} info_num keys={list(resp.info_num.keys())}")

            now = time.time()
            confirmed_battle = False
            if not match_over:
                if (not ocr_failed) and float(game_time) > 0.0:
                    confirmed_battle = True
                elif (not elixir_failed) and float(elixir_val) >= 0.0:
                    confirmed_battle = True
                elif float(resp.reward) != 0.0:
                    confirmed_battle = True

            if confirmed_battle:
                last_confirmed_battle = now
                no_battle_since = now

            grace_s = 5.0
            in_battle = bool((not match_over) and (now - last_confirmed_battle) <= grace_s)

            if not in_battle and (now - no_battle_since) >= float(args.no_battle_recover_s):
                _recover_to_training(f"no_battle_for_{args.no_battle_recover_s}s")

            # Apply action ONLY if we're in battle and aligned to capture cadence
            if want_action_now and in_battle and resp.HasField("action"):
                card_idx, gx, gy = resp.action.card_idx, resp.action.grid_x, resp.action.grid_y
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
                if nav is not None and _can_navigate(time.time()):
                    try:
                        nav.dismiss_post_match()
                        nav.start_training_match()
                        last_nav_time = time.time()
                        no_battle_since = time.time()
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
