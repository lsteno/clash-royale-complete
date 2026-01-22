#!/usr/bin/env python
"""Run the gRPC FrameService server with KataCR perception.

Example:
    python scripts/serve_frame_service.py --host 0.0.0.0 --port 50051 --model-version a10-build-1
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cr.rpc.v1.processor import FrameServiceProcessor, ProcessorConfig
from cr.rpc.v1.server import RpcServerConfig, serve_forever
from src.perception.katacr_pipeline import KataCRVisionConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FrameService gRPC server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Listen host")
    parser.add_argument("--port", type=int, default=50051, help="Listen port")
    parser.add_argument("--schema-version", type=str, default="v1", help="Schema version string")
    parser.add_argument("--model-version", type=str, default="dev", help="Model/version identifier")
    parser.add_argument("--max-message-mb", type=int, default=32, help="Max gRPC message size MB")
    parser.add_argument("--perception-stride", type=int, default=1, help="Run full perception every N stepped frames")
    parser.add_argument("--return-state-grid", action="store_true", help="Return full state_grid tensors (slow; debug)")
    parser.add_argument("--detector-count", type=int, default=2, help="Use first N YOLO detectors (1 = faster)")
    parser.add_argument("--disable-center-ocr", action="store_true", help="Skip center-screen OCR")
    parser.add_argument("--disable-card-classifier", action="store_true", help="Skip card classifier (faster; uses fallback)")
    parser.add_argument("--ocr-gpu", action="store_true", help="Use GPU for OCR (requires Paddle GPU wheels)")
    parser.add_argument("--ocr-cpu", action="store_true", help="Force CPU for OCR")
    parser.add_argument("--save-perception-crops", action="store_true", help="Save perception debug crops")
    parser.add_argument("--perception-crops-dir", type=Path, default=None, help="Dir for perception crops")
    parser.add_argument("--debug-dump-dir", type=Path, default=None, help="Dump perception + obs tensors to this directory")
    parser.add_argument("--debug-dump-every", type=int, default=0, help="Dump every N perception frames (0 disables)")
    parser.add_argument("--debug-dump-max", type=int, default=200, help="Maximum number of dumps to write")
    parser.add_argument("--debug-dump-annotated", action="store_true", help="Also dump YOLO overlay images (slower)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ocr_gpu and args.ocr_cpu:
        raise SystemExit("Choose only one of --ocr-gpu or --ocr-cpu")
    if args.ocr_cpu:
        ocr_gpu = False
    elif args.ocr_gpu:
        ocr_gpu = True
    else:
        ocr_gpu = bool(int(os.environ.get("CR_OCR_GPU", "0")))

    proc_cfg = ProcessorConfig(
        schema_version=args.schema_version,
        model_version=args.model_version,
        perception_stride=max(1, int(args.perception_stride)),
        return_state_grid=bool(args.return_state_grid),
        debug_dump_dir=str(args.debug_dump_dir) if args.debug_dump_dir else None,
        debug_dump_every=int(args.debug_dump_every),
        debug_dump_max=int(args.debug_dump_max),
        debug_dump_annotated=bool(args.debug_dump_annotated),
    )
    server_cfg = RpcServerConfig(
        host=args.host,
        port=args.port,
        max_message_mb=args.max_message_mb,
        schema_version=args.schema_version,
        model_version=args.model_version,
    )
    vision_cfg = KataCRVisionConfig(
        debug_save_parts=bool(args.save_perception_crops),
        debug_parts_dir=args.perception_crops_dir or Path("logs/perception_crops"),
        ocr_gpu=ocr_gpu,
        detector_count=int(args.detector_count),
        enable_center_ocr=not bool(args.disable_center_ocr),
        enable_card_classifier=not bool(args.disable_card_classifier),
    )
    processor = FrameServiceProcessor(proc_cfg, vision_cfg=vision_cfg)
    asyncio.run(serve_forever(processor, server_cfg))


if __name__ == "__main__":
    main()
