"""Async gRPC client for FrameService.

Handles connection setup, deadlines, and bounded in-flight RPCs.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional, Sequence

import grpc

from cr.rpc.v1 import frame_service_pb2 as pb2
from cr.rpc.v1 import frame_service_pb2_grpc as pb2_grpc


@dataclass
class RpcClientConfig:
    """Client knobs for FrameService."""

    target: str  # host:port
    # Perception can easily exceed 500ms per frame (YOLO + OCR + classification).
    # Keep a safer default; callers can still override.
    deadline_ms: int = 2000
    max_inflight: int = 2
    insecure: bool = True
    root_cert: Optional[bytes] = None
    client_cert: Optional[bytes] = None
    client_key: Optional[bytes] = None


class FrameServiceClient:
    """Bounded-concurrency aio client for the remote perception/Dreamer service."""

    def __init__(self, config: RpcClientConfig):
        self._cfg = config
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub: Optional[pb2_grpc.FrameServiceStub] = None
        self._sem = asyncio.Semaphore(max(1, config.max_inflight))

    async def __aenter__(self) -> "FrameServiceClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def connect(self) -> None:
        if self._channel is not None:
            return
        self._channel = self._build_channel()
        self._stub = pb2_grpc.FrameServiceStub(self._channel)

    async def close(self) -> None:
        if self._channel is None:
            return
        await self._channel.close()
        self._channel = None
        self._stub = None

    def _build_channel(self) -> grpc.aio.Channel:
        if self._cfg.insecure:
            return grpc.aio.insecure_channel(self._cfg.target, options=self._channel_options())
        credentials = self._build_credentials()
        return grpc.aio.secure_channel(self._cfg.target, credentials, options=self._channel_options())

    def _channel_options(self):
        return [
            ("grpc.max_send_message_length", 32 * 1024 * 1024),
            ("grpc.max_receive_message_length", 32 * 1024 * 1024),
            ("grpc.keepalive_time_ms", 30_000),
            ("grpc.keepalive_timeout_ms", 10_000),
            ("grpc.http2.min_time_between_pings_ms", 10_000),
            ("grpc.keepalive_permit_without_calls", 1),
        ]

    def _build_credentials(self) -> grpc.ChannelCredentials:
        root = self._cfg.root_cert if self._cfg.root_cert else None
        private_key = self._cfg.client_key if self._cfg.client_key else None
        certificate_chain = self._cfg.client_cert if self._cfg.client_cert else None
        return grpc.ssl_channel_credentials(root_certificates=root, private_key=private_key, certificate_chain=certificate_chain)

    async def heartbeat(self, schema_version: str, model_version: str) -> pb2.HeartbeatResponse:
        if self._stub is None:
            raise RuntimeError("Client not connected")
        req = pb2.HeartbeatRequest(schema_version=schema_version, model_version=model_version)
        return await self._stub.Heartbeat(req, timeout=self._deadline_s())

    async def process_frame(
        self,
        *,
        frame_id: int,
        timestamp: float,
        width: int,
        height: int,
        channels: int,
        frame_bgr: bytes,
        card_hints: Optional[Sequence[str]] = None,
        want_action: bool = False,
        schema_version: str = "v1",
        roi: Optional[pb2.RegionOfInterest] = None,
        color_model: str = "BGR",
    ) -> pb2.ProcessFrameResponse:
        if self._stub is None:
            raise RuntimeError("Client not connected")
        async with self._sem:
            req = pb2.ProcessFrameRequest(
                frame_id=frame_id,
                timestamp=timestamp,
                format=pb2.FrameFormat(width=width, height=height, channels=channels, color_model=color_model),
                frame_bgr=frame_bgr,
                card_hints=list(card_hints) if card_hints else [],
                want_action=want_action,
                schema_version=schema_version,
            )
            if roi is not None:
                req.roi.CopyFrom(roi)
            return await self._stub.ProcessFrame(req, timeout=self._deadline_s())

    def _deadline_s(self) -> float:
        return max(0.001, self._cfg.deadline_ms / 1000.0)
