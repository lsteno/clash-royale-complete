"""gRPC FrameService processor that runs KataCR perception and encodes the state grid.

This bridges raw BGR frames to the RPC response contract. It keeps the action
computation pluggable so Dreamer can be invoked remotely later.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import cv2

from cr.rpc.v1 import frame_service_pb2 as pb2
from src.environment.state_encoder import StateTensorEncoder
from src.specs import OBS_SPEC
from src.environment.embodied_env import RemoteBridgeV3

if TYPE_CHECKING:
    from src.perception.katacr_pipeline import KataCRPerceptionEngine, KataCRVisionConfig

ActionFn = Callable[[np.ndarray, dict, float, dict], Optional[pb2.Action]]


@dataclass
class ProcessorConfig:
    """Configuration for the FrameService processor."""

    schema_version: str = "v1"
    model_version: str = "dev"
    max_game_seconds: int = 360


class FrameServiceProcessor:
    """Implements the FrameProcessor protocol used by the gRPC server scaffold."""

    def __init__(
        self,
        cfg: ProcessorConfig = ProcessorConfig(),
        vision_cfg: Optional["KataCRVisionConfig"] = None,
        action_fn: Optional[ActionFn] = None,
        bridge: Optional[RemoteBridgeV3] = None,
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

    async def process_frame(self, request: pb2.ProcessFrameRequest) -> pb2.ProcessFrameResponse:
        t0 = time.time()
        frame_bgr = self._decode_frame(request)

        # Always run perception for reward calculation (tower HP tracking)
        perception_result = self._perception.process(frame_bgr, deploy_cards=None)
        info = perception_result.info

        # Detect end-of-match via UI color probe on the received frame.
        match_over = _detect_match_over(frame_bgr)

        # Log tower HP every 5 frames
        self._frame_count += 1
        if self._frame_count % 5 == 0:
            rb = self._perception.reward_builder
            hp_tower = rb.hp_tower  # shape (2, 2): [ally/enemy][left/right]
            hp_king = rb.hp_king_tower  # shape (2,): [ally, enemy]
            print(
                f"[TowerHP] frame={self._frame_count} "
                f"ally_left={hp_tower[0,0]} ally_right={hp_tower[0,1]} ally_king={hp_king[0]} | "
                f"enemy_left={hp_tower[1,0]} enemy_right={hp_tower[1,1]} enemy_king={hp_king[1]}"
            )

        # Encode observation: grid (default) or raw pixels (optional)
        if self._use_pixels:
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

        # Build state grid message
        # Note: For pixels, obs is (H, W, 3) channels-last; for grid, obs is (C, H, W) channels-first
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
        state_grid.values.extend(obs.astype(np.float32).ravel().tolist())

        latency_ms = (time.time() - t0) * 1000.0
        # Reward is always computed from perception (even in pixel mode)
        reward_value = float(perception_result.reward if perception_result.reward is not None else 0.0)
        done_value = bool(match_over)
        
        def _safe_float(value, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        time_value = _safe_float(perception_result.state.get("time"), 0.0)
        elixir_value = _safe_float(perception_result.state.get("elixir"), -1.0)
        action_msg = None
        if request.want_action:
            if self._bridge is not None:
                # Hand the observation to the local trainer and wait for its action.
                act_tuple = self._bridge.publish(obs, reward_value, done_value, info)
                if act_tuple is not None:
                    print(f"[processor] bridge action={act_tuple}")
                    action_msg = pb2.Action(card_idx=int(act_tuple[0]), grid_x=int(act_tuple[1]), grid_y=int(act_tuple[2]))
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

        latency_ms = (time.time() - t0) * 1000.0
        resp = pb2.ProcessFrameResponse(
            frame_id=request.frame_id,
            state_grid=state_grid,
            reward=reward_value,
            done=done_value,
            latency_ms=latency_ms,
            ocr_failed=bool(info.get("ocr_time_failed", False)),
            model_version=self._cfg.model_version,
            schema_version=self._cfg.schema_version,
        )

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
        h, w, c = int(fmt.height), int(fmt.width), int(fmt.channels)
        expected = h * w * c
        buffer = memoryview(request.frame_bgr)
        if buffer.nbytes != expected:
            raise ValueError(
                f"Frame byte size mismatch: got {buffer.nbytes}, expected {expected} for shape (H={h}, W={w}, C={c})"
            )
        frame = np.frombuffer(buffer, dtype=np.uint8)
        frame = frame.reshape((h, w, c))
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
