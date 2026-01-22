"""gRPC FrameService processor that runs KataCR perception and encodes the state grid.

This bridges raw BGR frames to the RPC response contract. It keeps the action
computation pluggable so Dreamer can be invoked remotely later.
"""
from __future__ import annotations

import time
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import cv2

from cr.rpc.v1 import frame_service_pb2 as pb2
from src.environment.state_encoder import CHANNEL_NAMES, StateTensorEncoder
from src.specs import OBS_SPEC

if TYPE_CHECKING:
    from src.perception.katacr_pipeline import KataCRPerceptionEngine, KataCRVisionConfig
    from src.environment.embodied_env import RemoteBridgeV3

ActionFn = Callable[[np.ndarray, dict, float, dict], Optional[pb2.Action]]


@dataclass
class ProcessorConfig:
    """Configuration for the FrameService processor."""

    schema_version: str = "v1"
    model_version: str = "dev"
    max_game_seconds: int = 360
    # Run full KataCR perception every N frames and reuse the last observation
    # in between. This is the main knob for throughput.
    perception_stride: int = 1
    # Returning the full encoded observation over RPC is expensive (protobuf
    # serializes thousands of floats). The trainer doesn't need it; only the
    # remote client does for debugging.
    return_state_grid: bool = False
    # Optional debug dumps to disk (only on perception frames).
    debug_dump_dir: Optional[str] = None
    debug_dump_every: int = 0
    debug_dump_max: int = 200
    debug_dump_annotated: bool = False


class FrameServiceProcessor:
    """Implements the FrameProcessor protocol used by the gRPC server scaffold."""

    def __init__(
        self,
        cfg: ProcessorConfig = ProcessorConfig(),
        vision_cfg: Optional["KataCRVisionConfig"] = None,
        action_fn: Optional[ActionFn] = None,
        bridge: Optional["RemoteBridgeV3"] = None,
        use_pixels: bool = False,
        pixel_height: int = 192,
        pixel_width: int = 256,
    ):
        self._cfg = cfg
        # Always run perception for reward calculation (tower HP tracking)
        # Even in pixel mode, we need KataCR to compute dense rewards
        from src.perception.katacr_pipeline import KataCRPerceptionEngine

        self._perception: "KataCRPerceptionEngine" = KataCRPerceptionEngine(vision_cfg)
        self._encoder = StateTensorEncoder(cfg.max_game_seconds)
        self._action_fn = action_fn
        self._bridge = bridge
        self._frame_count = 0
        self._use_pixels = use_pixels
        self._pixel_height = int(pixel_height)
        self._pixel_width = int(pixel_width)
        self._perception_stride = max(1, int(getattr(cfg, "perception_stride", 1)))
        self._return_state_grid = bool(getattr(cfg, "return_state_grid", False))
        self._last_perception = None
        self._last_obs = None
        self._last_action: Optional[pb2.Action] = None
        dump_dir = getattr(cfg, "debug_dump_dir", None)
        self._dump_dir: Optional[Path] = Path(dump_dir) if dump_dir else None
        self._dump_every = int(getattr(cfg, "debug_dump_every", 0) or 0)
        if self._dump_dir is not None and self._dump_every <= 0:
            self._dump_every = 1
        self._dump_max = int(getattr(cfg, "debug_dump_max", 200) or 200)
        self._dump_annotated = bool(getattr(cfg, "debug_dump_annotated", False))
        self._dump_count = 0

    async def process_frame(self, request: pb2.ProcessFrameRequest) -> pb2.ProcessFrameResponse:
        t0 = time.time()
        frame_bgr = self._decode_frame(request)

        self._frame_count += 1
        # Only run expensive perception when the remote client is actually
        # stepping the environment (i.e., requesting actions). This lets the
        # client stream at 30 FPS while the trainer steps at ~action_hz.
        #
        # Always run once to initialize caches, even if want_action=False.
        run_perception = (self._last_perception is None) or (
            bool(request.want_action) and (self._frame_count % self._perception_stride == 0)
        )
        if run_perception:
            perception_result = self._perception.process(frame_bgr, deploy_cards=None)
            self._last_perception = perception_result
        else:
            perception_result = self._last_perception
        info = perception_result.info if perception_result is not None else {}
        state = perception_result.state if perception_result is not None else {}

        # Detect end-of-match via UI color probe on the received frame + OCR center texts.
        match_over = _detect_match_over(frame_bgr)
        center_flag = None if perception_result is None else perception_result.info.get("center_flag")
        center_says_end = bool(center_flag == 1)

        # Log tower HP every 5 frames
        if self._frame_count % 5 == 0 and run_perception:
            rb = self._perception.reward_builder
            hp_tower = rb.hp_tower  # shape (2, 2): [ally/enemy][left/right]
            hp_king = rb.hp_king_tower  # shape (2,): [ally, enemy]
            print(
                f"[TowerHP] frame={self._frame_count} "
                f"ally_left={hp_tower[0,0]} ally_right={hp_tower[0,1]} ally_king={hp_king[0]} | "
                f"enemy_left={hp_tower[1,0]} enemy_right={hp_tower[1,1]} enemy_king={hp_king[1]}"
            )

        # Encode observation: grid (default) or raw pixels (optional)
        if not run_perception and self._last_obs is not None:
            obs = self._last_obs
        elif self._use_pixels:
            # Resize to requested spatial dimensions and convert BGR->RGB for CNN encoder
            resized = self._resize_frame(frame_bgr, self._pixel_width, self._pixel_height)
            # Convert BGR to RGB (DreamerV3 CNN expects RGB channels-last)
            obs = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.uint8)
            # obs shape is (H, W, 3) - channels-last for DreamerV3 CNN
        else:
            obs = self._encoder.encode(
                perception_result.state,
                self._perception.reward_builder,
                perception_result.info,
            )
        self._last_obs = obs

        state_grid: Optional[pb2.StateGrid] = None
        if self._return_state_grid:
            # Build state grid message. Note: For pixels, obs is (H, W, 3)
            # channels-last; for grid, obs is (C, H, W) channels-first.
            if self._use_pixels:
                h, w, c = self._pixel_height, self._pixel_width, 3
            else:
                c, h, w = int(obs.shape[0]), int(obs.shape[1]), int(obs.shape[2])

            state_grid = pb2.StateGrid(
                channels=int(c),
                height=int(h),
                width=int(w),
                dtype="uint8" if self._use_pixels else "float32",
            )
            # Warning: converting to Python lists is expensive. Only enable when
            # you actively need the observation on the client side.
            state_grid.values.extend(obs.astype(np.float32).ravel().tolist())

        # Reward is always computed from perception (even in pixel mode).
        reward_value = float(perception_result.reward if (perception_result is not None and perception_result.reward is not None) else 0.0)
        if not run_perception:
            reward_value = 0.0
        done_value = bool(match_over or center_says_end)
        
        def _safe_float(value, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        time_value = _safe_float(state.get("time"), 0.0)
        elixir_value = _safe_float(state.get("elixir"), -1.0)
        action_msg = None
        if request.want_action:
            # Only advance the trainer on fresh perception frames (or when done).
            # This makes the environment step rate ~= perception FPS, even if the
            # client streams at 30 FPS.
            should_step = bool(run_perception or done_value)
            if not should_step:
                action_msg = self._last_action
            elif self._bridge is not None:
                act_tuple = self._bridge.publish(obs, reward_value, done_value, info)
                if act_tuple is not None:
                    action_msg = pb2.Action(card_idx=int(act_tuple[0]), grid_x=int(act_tuple[1]), grid_y=int(act_tuple[2]))
                    self._last_action = action_msg
            elif self._action_fn is not None:
                act = self._action_fn(
                    obs,
                    perception_result.state,
                    reward_value,
                    info,
                )
                if act is not None:
                    print(f"[processor] action_fn returned {act}")
                    action_msg = act

        if run_perception and self._dump_dir is not None and self._dump_count < self._dump_max:
            if self._dump_every > 0 and (self._frame_count % self._dump_every == 0):
                try:
                    self._dump_debug(request, frame_bgr, perception_result, obs, reward_value, done_value)
                    self._dump_count += 1
                except Exception as exc:
                    print(f"[processor] Warning: debug dump failed: {exc}")

        latency_ms = (time.time() - t0) * 1000.0
        resp = pb2.ProcessFrameResponse(
            frame_id=request.frame_id,
            reward=reward_value,
            done=done_value,
            latency_ms=latency_ms,
            ocr_failed=bool(info.get("ocr_time_failed", False)),
            model_version=self._cfg.model_version,
            schema_version=self._cfg.schema_version,
        )
        if state_grid is not None:
            resp.state_grid.CopyFrom(state_grid)

        if action_msg is not None:
            resp.action.CopyFrom(action_msg)

        # Populate info maps with lightweight diagnostics
        resp.info_str.update({
            "color_model": request.format.color_model or "BGR",
        })
        if info.get("elixir_failed"):
            resp.info_str["elixir_status"] = "ocr_failed"
        resp.info_num.update({
            "server_ms": latency_ms,
            "timestamp": request.timestamp,
            "match_over": 1.0 if done_value else 0.0,
            "game_time": time_value,
            "elixir": elixir_value,
        })

        if done_value:
            # Clear KataCR state/reward history before the next match begins.
            self._perception.reset()
        return resp

    def _dump_debug(
        self,
        request: pb2.ProcessFrameRequest,
        frame_bgr: np.ndarray,
        perception_result,
        obs: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        out_dir = Path(self._dump_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"frame_{int(request.frame_id):06d}_fc{int(self._frame_count):06d}"
        base = out_dir / stem
        base.mkdir(parents=True, exist_ok=True)

        def _jsonable(x):
            if x is None or isinstance(x, (bool, int, float, str)):
                return x
            if isinstance(x, (np.generic,)):
                return x.item()
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, (list, tuple)):
                return [_jsonable(v) for v in x]
            if isinstance(x, dict):
                return {str(k): _jsonable(v) for k, v in x.items()}
            return repr(x)

        # Save raw input frame (as received; may include ROI crop).
        cv2.imwrite(str(base / "frame_bgr.png"), frame_bgr)

        # Save detection overlay if available.
        try:
            info = getattr(perception_result, "info", {}) or {}
            arena = info.get("arena")
            if self._dump_annotated and arena is not None and hasattr(arena, "show_box"):
                annotated = arena.show_box(verbose=False, show_conf=True)
                if annotated is not None and getattr(annotated, "size", 0) != 0:
                    cv2.imwrite(str(base / "arena_boxes.png"), annotated)
        except Exception:
            pass

        # Save a KataCR renderer view (arena + grid + card/elixir text).
        try:
            sb = getattr(self._perception, "state_builder", None)
            if sb is not None and hasattr(sb, "render"):
                rendered = sb.render(action=None)
                if rendered is not None and getattr(rendered, "size", 0) != 0:
                    cv2.imwrite(str(base / "katacr_render.png"), rendered)
        except Exception:
            pass

        # Save raw detector boxes with decoded labels for offline verification.
        try:
            info = getattr(perception_result, "info", {}) or {}
            arena = info.get("arena")
            if arena is not None and hasattr(arena, "get_data"):
                from katacr.constants.label_list import idx2unit  # type: ignore
                from katacr.constants.state_list import idx2state  # type: ignore

                data = np.asarray(arena.get_data())
                rows = []
                for row in data.tolist():
                    # row is xyxy + (track_id) + conf + cls + bel
                    if len(row) not in (7, 8):
                        continue
                    xyxy = row[:4]
                    if len(row) == 8:
                        track_id = int(row[4])
                        conf, cls, bel = float(row[5]), int(row[6]), int(row[7])
                    else:
                        track_id = None
                        conf, cls, bel = float(row[4]), int(row[5]), int(row[6])
                    label = f"{idx2unit.get(cls, str(cls))}{idx2state.get(bel, str(bel))}"
                    rows.append({
                        "xyxy": xyxy,
                        "track_id": track_id,
                        "conf": conf,
                        "cls": cls,
                        "bel": bel,
                        "label": label,
                    })
                (base / "arena_boxes.json").write_text(json.dumps(rows, indent=2))
        except Exception:
            pass

        # Save observation that gets published to the trainer.
        np.save(str(base / "obs.npy"), obs)
        if isinstance(obs, np.ndarray) and obs.ndim >= 1:
            np.save(str(base / "obs_flat.npy"), obs.reshape(-1))

        # For semantic-grid mode, dump a readable montage of channels.
        if (not self._use_pixels) and isinstance(obs, np.ndarray) and obs.ndim == 3:
            try:
                c, h, w = obs.shape
                scale = 16
                pad = 8
                cols = 5
                rows = int(np.ceil(c / cols))
                tile_w = w * scale
                tile_h = h * scale
                canvas = np.zeros(
                    (rows * tile_h + (rows + 1) * pad, cols * tile_w + (cols + 1) * pad, 3),
                    dtype=np.uint8,
                )
                for ch in range(c):
                    r = ch // cols
                    col = ch % cols
                    y0 = pad + r * (tile_h + pad)
                    x0 = pad + col * (tile_w + pad)
                    img = obs[ch]
                    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
                    img = np.clip(img, 0.0, 1.0)
                    gray = (img * 255).astype(np.uint8)
                    up = cv2.resize(gray, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
                    heat = cv2.applyColorMap(up, cv2.COLORMAP_VIRIDIS)
                    canvas[y0:y0 + tile_h, x0:x0 + tile_w] = heat
                    name = CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f"ch_{ch}"
                    cv2.putText(
                        canvas,
                        f"{ch}:{name}",
                        (x0, max(0, y0 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.imwrite(str(base / "obs_channels.png"), canvas)
                (base / "obs_channels.json").write_text(json.dumps({
                    "channels": int(c),
                    "height": int(h),
                    "width": int(w),
                    "channel_names": list(CHANNEL_NAMES[:c]),
                }, indent=2))
            except Exception:
                pass

        # Save KataCR state + info (sanitized).
        state = getattr(perception_result, "state", {}) or {}
        info = getattr(perception_result, "info", {}) or {}
        info_slim = {k: v for k, v in info.items() if k not in ("arena",)}

        # Include action mask diagnostics (what the env will compute).
        try:
            from src.environment.action_mask import compute_action_mask

            cards = info_slim.get("cards") or []
            elixir = info_slim.get("elixir", 0)
            mask = compute_action_mask(cards, elixir)
            np.save(str(base / "action_mask.npy"), mask.astype(np.float32))
            info_slim["legal_action_count"] = int(np.sum(mask >= 0))
        except Exception:
            pass

        (base / "state.json").write_text(json.dumps(_jsonable(state), indent=2))
        (base / "info.json").write_text(json.dumps(_jsonable(info_slim), indent=2))
        (base / "meta.json").write_text(json.dumps({
            "frame_id": int(request.frame_id),
            "frame_count": int(self._frame_count),
            "timestamp": float(request.timestamp),
            "want_action": bool(request.want_action),
            "run_perception": True,
            "reward": float(reward),
            "done": bool(done),
            "use_pixels": bool(self._use_pixels),
            "obs_shape": list(obs.shape) if isinstance(obs, np.ndarray) else None,
            "obs_dtype": str(getattr(obs, "dtype", None)),
        }, indent=2))

    def _resize_frame(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        import cv2

        if frame.shape[1] == width and frame.shape[0] == height:
            return frame
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    async def heartbeat(self, request: pb2.HeartbeatRequest) -> pb2.HeartbeatResponse:
        return pb2.HeartbeatResponse(
            schema_version=self._cfg.schema_version,
            model_version=self._cfg.model_version,
        )

    def _decode_frame(self, request: pb2.ProcessFrameRequest) -> np.ndarray:
        fmt = request.format
        color_model = (fmt.color_model or "BGR").upper()
        buffer = memoryview(request.frame_bgr)
        if color_model in ("JPEG", "JPG"):
            arr = np.frombuffer(buffer, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Failed to decode JPEG frame")
        else:
            h, w, c = int(fmt.height), int(fmt.width), int(fmt.channels)
            expected = h * w * c
            if buffer.nbytes != expected:
                raise ValueError(
                    f"Frame byte size mismatch: got {buffer.nbytes}, expected {expected} for shape (H={h}, W={w}, C={c})"
                )
            frame = np.frombuffer(buffer, dtype=np.uint8).reshape((h, w, c))
        if request.HasField("roi"):
            frame = self._crop_roi(frame, request.roi)
        return frame

    def _crop_roi(self, frame: np.ndarray, roi: pb2.RegionOfInterest) -> np.ndarray:
        h, w = frame.shape[:2]
        x0 = int(max(0, min(w, roi.x)))
        y0 = int(max(0, min(h, roi.y)))
        x1 = int(max(x0, min(w, roi.x + roi.width)))
        y1 = int(max(y0, min(h, roi.y + roi.height)))
        return frame[y0:y1, x0:x1, :]


def _detect_match_over(frame_bgr: np.ndarray) -> bool:
    """Lightweight OK-button color probe on the canonical frame received over RPC."""
    if frame_bgr.size == 0:
        return False

    # Reference coordinates from emulator_env: screen (1080x2400) -> canonical resize.
    ref_w, ref_h = 1080.0, 2400.0
    ok_screen = (613.0, 2021.0)
    target_color = np.array([255, 187, 104], dtype=np.float32)
    tol = 40.0

    h, w = frame_bgr.shape[:2]
    fx = int(round(ok_screen[0] * w / ref_w))
    fy = int(round(ok_screen[1] * h / ref_h))
    fx = int(np.clip(fx, 0, w - 1))
    fy = int(np.clip(fy, 0, h - 1))

    # Patch average to reduce resize noise.
    r = 2
    x0, x1 = max(0, fx - r), min(w, fx + r + 1)
    y0, y1 = max(0, fy - r), min(h, fy + r + 1)
    patch = frame_bgr[y0:y1, x0:x1]
    mean = patch.mean(axis=(0, 1)) if patch.size else frame_bgr[fy, fx]

    dist = np.max(np.abs(mean.astype(np.float32) - target_color))
    return bool(dist <= tol)
