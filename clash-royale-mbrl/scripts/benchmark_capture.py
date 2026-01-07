#!/usr/bin/env python
"""Benchmark ADB screen capture and optionally YOLO/KataCR inference speed.

Usage (Mac - ADB only):
    PYTHONPATH=./src:. python scripts/benchmark_capture.py --adb-only

Usage (VM - full pipeline):
    PYTHONPATH=./src:. python scripts/benchmark_capture.py
"""
from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

# Add KataCR to path
KATACR_ROOT = Path(__file__).resolve().parents[2] / "KataCR"
if str(KATACR_ROOT) not in sys.path:
    sys.path.insert(0, str(KATACR_ROOT))


def benchmark_adb(n_frames: int = 20) -> None:
    """Benchmark raw ADB screen capture speed."""
    from src.environment.emulator_env import ADBScreenshotter, EmulatorConfig
    
    config = EmulatorConfig()
    screenshotter = ADBScreenshotter(config)
    
    # Warmup
    print("Warming up ADB...")
    for _ in range(3):
        screenshotter.capture()
    
    # Benchmark
    print(f"\nBenchmarking {n_frames} ADB screencaps...")
    times = []
    for i in range(n_frames):
        t0 = time.perf_counter()
        img = screenshotter.capture()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  Frame {i+1}: {elapsed*1000:.1f}ms  (shape: {img.shape})")
    
    avg = sum(times) / len(times)
    fps = 1.0 / avg
    print(f"\n=== ADB Results ===")
    print(f"  Average: {avg*1000:.1f}ms per frame")
    print(f"  Max FPS: {fps:.1f}")
    print(f"  Min: {min(times)*1000:.1f}ms, Max: {max(times)*1000:.1f}ms")


def benchmark_mss(n_frames: int = 20) -> None:
    """Benchmark mss screen capture (requires scrcpy window visible)."""
    import mss
    
    sct = mss.mss()
    # Use primary monitor - you may need to adjust this
    monitor = sct.monitors[1]  # monitors[0] is all monitors combined
    print(f"Capturing from monitor: {monitor}")
    
    # Warmup
    print("Warming up mss...")
    for _ in range(3):
        sct.grab(monitor)
    
    # Benchmark
    print(f"\nBenchmarking {n_frames} mss screen captures...")
    times = []
    for i in range(n_frames):
        t0 = time.perf_counter()
        img = sct.grab(monitor)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  Frame {i+1}: {elapsed*1000:.1f}ms  (size: {img.width}x{img.height})")
    
    avg = sum(times) / len(times)
    fps = 1.0 / avg
    print(f"\n=== MSS Results ===")
    print(f"  Average: {avg*1000:.1f}ms per frame")
    print(f"  Max FPS: {fps:.1f}")
    print(f"  Min: {min(times)*1000:.1f}ms, Max: {max(times)*1000:.1f}ms")
    print(f"\nNote: For scrcpy capture, set capture_region to the scrcpy window coordinates.")


def benchmark_yolo(n_frames: int = 20) -> None:
    """Benchmark YOLO/KataCR perception pipeline."""
    from src.perception.katacr_pipeline import KataCRPerceptionEngine, KataCRVisionConfig
    from src.environment.emulator_env import ADBScreenshotter, EmulatorConfig
    
    config = EmulatorConfig()
    screenshotter = ADBScreenshotter(config)
    
    # Capture one frame for YOLO testing
    print("Capturing test frame...")
    frame_bgr = screenshotter.capture()
    print(f"Frame shape: {frame_bgr.shape}")
    
    # Initialize perception
    print("\nInitializing KataCR perception (this loads YOLO weights)...")
    t0 = time.perf_counter()
    engine = KataCRPerceptionEngine(KataCRVisionConfig())
    load_time = time.perf_counter() - t0
    print(f"Model load time: {load_time:.2f}s")
    
    # Warmup
    print("\nWarming up inference...")
    for _ in range(3):
        engine.process(frame_bgr, deploy_cards=None)
    
    # Benchmark YOLO inference
    print(f"\nBenchmarking {n_frames} YOLO inferences...")
    times = []
    for i in range(n_frames):
        t0 = time.perf_counter()
        result = engine.process(frame_bgr, deploy_cards=None)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        n_units = len(result.state.get("unit_infos", []))
        print(f"  Frame {i+1}: {elapsed*1000:.1f}ms  (detected {n_units} units)")
    
    avg = sum(times) / len(times)
    fps = 1.0 / avg
    print(f"\n=== YOLO Results ===")
    print(f"  Average: {avg*1000:.1f}ms per frame")
    print(f"  Max FPS: {fps:.1f}")
    print(f"  Min: {min(times)*1000:.1f}ms, Max: {max(times)*1000:.1f}ms")


def benchmark_full_pipeline(n_frames: int = 20) -> None:
    """Benchmark ADB capture + YOLO combined."""
    from src.perception.katacr_pipeline import KataCRPerceptionEngine, KataCRVisionConfig
    from src.environment.emulator_env import ADBScreenshotter, EmulatorConfig
    
    config = EmulatorConfig()
    screenshotter = ADBScreenshotter(config)
    
    # Initialize perception
    print("Initializing KataCR perception...")
    engine = KataCRPerceptionEngine(KataCRVisionConfig())
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        frame_bgr = screenshotter.capture()
        engine.process(frame_bgr, deploy_cards=None)
    
    # Benchmark full pipeline
    print(f"\nBenchmarking {n_frames} full captures (ADB + YOLO)...")
    adb_times = []
    yolo_times = []
    for i in range(n_frames):
        # ADB
        t0 = time.perf_counter()
        frame_bgr = screenshotter.capture()
        adb_elapsed = time.perf_counter() - t0
        adb_times.append(adb_elapsed)
        
        # YOLO
        t0 = time.perf_counter()
        result = engine.process(frame_bgr, deploy_cards=None)
        yolo_elapsed = time.perf_counter() - t0
        yolo_times.append(yolo_elapsed)
        
        total = adb_elapsed + yolo_elapsed
        print(f"  Frame {i+1}: ADB={adb_elapsed*1000:.1f}ms, YOLO={yolo_elapsed*1000:.1f}ms, Total={total*1000:.1f}ms")
    
    adb_avg = sum(adb_times) / len(adb_times)
    yolo_avg = sum(yolo_times) / len(yolo_times)
    total_avg = adb_avg + yolo_avg
    
    print(f"\n=== Full Pipeline Results ===")
    print(f"  ADB average:  {adb_avg*1000:.1f}ms")
    print(f"  YOLO average: {yolo_avg*1000:.1f}ms")
    print(f"  Total average: {total_avg*1000:.1f}ms")
    print(f"  Max FPS: {1.0/total_avg:.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adb-only", action="store_true", help="Only benchmark ADB capture (no YOLO)")
    parser.add_argument("--mss-only", action="store_true", help="Only benchmark mss screen capture")
    parser.add_argument("--yolo-only", action="store_true", help="Only benchmark YOLO inference")
    parser.add_argument("-n", "--frames", type=int, default=20, help="Number of frames to benchmark")
    args = parser.parse_args()
    
    if args.adb_only:
        benchmark_adb(args.frames)
    elif args.mss_only:
        benchmark_mss(args.frames)
    elif args.yolo_only:
        benchmark_yolo(args.frames)
    else:
        benchmark_full_pipeline(args.frames)


if __name__ == "__main__":
    main()
