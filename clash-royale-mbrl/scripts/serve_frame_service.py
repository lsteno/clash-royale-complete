#!/usr/bin/env python
"""Run the gRPC FrameService server with KataCR perception.

Example:
    python scripts/serve_frame_service.py --host 0.0.0.0 --port 50051 --model-version a10-build-1
"""
from __future__ import annotations

import argparse
import asyncio

from cr.rpc.v1.processor import FrameServiceProcessor, ProcessorConfig
from cr.rpc.v1.server import RpcServerConfig, serve_forever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FrameService gRPC server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Listen host")
    parser.add_argument("--port", type=int, default=50051, help="Listen port")
    parser.add_argument("--schema-version", type=str, default="v1", help="Schema version string")
    parser.add_argument("--model-version", type=str, default="dev", help="Model/version identifier")
    parser.add_argument("--max-message-mb", type=int, default=32, help="Max gRPC message size MB")
    parser.add_argument("--perception-stride", type=int, default=2, help="Run perception every N frames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    proc_cfg = ProcessorConfig(
        schema_version=args.schema_version,
        model_version=args.model_version,
        perception_stride=args.perception_stride,
    )
    server_cfg = RpcServerConfig(
        host=args.host,
        port=args.port,
        max_message_mb=args.max_message_mb,
        schema_version=args.schema_version,
        model_version=args.model_version,
    )
    processor = FrameServiceProcessor(proc_cfg)
    asyncio.run(serve_forever(processor, server_cfg))


if __name__ == "__main__":
    main()
