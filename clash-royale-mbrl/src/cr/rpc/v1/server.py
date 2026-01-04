"""Async gRPC server scaffold for FrameService."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional, Protocol

import grpc

from cr.rpc.v1 import frame_service_pb2 as pb2
from cr.rpc.v1 import frame_service_pb2_grpc as pb2_grpc


@dataclass
class RpcServerConfig:
    host: str = "0.0.0.0"
    port: int = 50051
    max_message_mb: int = 32
    schema_version: str = "v1"
    model_version: str = "dev"

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    def grpc_options(self):
        size = self.max_message_mb * 1024 * 1024
        return [
            ("grpc.max_send_message_length", size),
            ("grpc.max_receive_message_length", size),
            ("grpc.keepalive_time_ms", 30_000),
            ("grpc.keepalive_timeout_ms", 10_000),
            ("grpc.http2.min_time_between_pings_ms", 10_000),
            ("grpc.keepalive_permit_without_calls", 1),
        ]


class FrameProcessor(Protocol):
    async def process_frame(self, request: pb2.ProcessFrameRequest) -> pb2.ProcessFrameResponse:
        ...

    async def heartbeat(self, request: pb2.HeartbeatRequest) -> pb2.HeartbeatResponse:
        ...


class _Servicer(pb2_grpc.FrameServiceServicer):
    def __init__(self, processor: FrameProcessor, cfg: RpcServerConfig):
        self._processor = processor
        self._cfg = cfg

    async def ProcessFrame(self, request, context):  # noqa: N802 grpc naming
        return await self._processor.process_frame(request)

    async def Heartbeat(self, request, context):  # noqa: N802 grpc naming
        if hasattr(self._processor, "heartbeat"):
            return await self._processor.heartbeat(request)
        return pb2.HeartbeatResponse(schema_version=self._cfg.schema_version, model_version=self._cfg.model_version)


def build_server(processor: FrameProcessor, cfg: Optional[RpcServerConfig] = None) -> grpc.aio.Server:
    cfg = cfg or RpcServerConfig()
    server = grpc.aio.server(options=cfg.grpc_options())
    pb2_grpc.add_FrameServiceServicer_to_server(_Servicer(processor, cfg), server)
    server.add_insecure_port(cfg.address)
    return server


async def serve_forever(processor: FrameProcessor, cfg: Optional[RpcServerConfig] = None):
    server = build_server(processor, cfg)
    await server.start()
    await server.wait_for_termination()


async def run_until_cancelled(server: grpc.aio.Server):
    await server.start()
    await server.wait_for_termination()
