"""gRPC FrameService processor that runs KataCR perception and encodes the state grid.

This bridges raw BGR frames to the RPC response contract. It keeps the action
computation pluggable so Dreamer can be invoked remotely later.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from cr.rpc.v1 import frame_service_pb2 as pb2
from src.environment.online_env import StateTensorEncoder
from src.perception.katacr_pipeline import KataCRPerceptionEngine, KataCRVisionConfig
from src.specs import OBS_SPEC

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
        vision_cfg: Optional[KataCRVisionConfig] = None,
        action_fn: Optional[ActionFn] = None,
    ):
        self._cfg = cfg
        self._perception = KataCRPerceptionEngine(vision_cfg)
        self._encoder = StateTensorEncoder(cfg.max_game_seconds)
        self._action_fn = action_fn

    async def process_frame(self, request: pb2.ProcessFrameRequest) -> pb2.ProcessFrameResponse:
        t0 = time.time()
        frame_bgr = self._decode_frame(request)

        # Run perception (KataCR)
        perception_result = self._perception.process(frame_bgr, deploy_cards=None)

        # Encode grid to float32 CxHxW row-major
        grid = self._encoder.encode(
            perception_result.state,
            self._perception.reward_builder,
            perception_result.info,
        )

        # Build state grid message
        state_grid = pb2.StateGrid(
            channels=int(grid.shape[0]),
            height=int(grid.shape[1]),
            width=int(grid.shape[2]),
            dtype="float32",
        )
        state_grid.values.extend(grid.astype(np.float32).ravel().tolist())

        action_msg = None
        if request.want_action and self._action_fn is not None:
            act = self._action_fn(grid, perception_result.state, perception_result.reward, perception_result.info)
            if act is not None:
                action_msg = act

        latency_ms = (time.time() - t0) * 1000.0
        resp = pb2.ProcessFrameResponse(
            frame_id=request.frame_id,
            state_grid=state_grid,
            reward=float(perception_result.reward if perception_result.reward is not None else 0.0),
            done=False,
            latency_ms=latency_ms,
            ocr_failed=bool(perception_result.info.get("ocr_time_failed", False)),
            model_version=self._cfg.model_version,
            schema_version=self._cfg.schema_version,
        )

        if action_msg is not None:
            resp.action.CopyFrom(action_msg)

        # Populate info maps with lightweight diagnostics
        resp.info_str.update({
            "color_model": request.format.color_model or "BGR",
        })
        resp.info_num.update({
            "server_ms": latency_ms,
            "timestamp": request.timestamp,
        })
        return resp

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
