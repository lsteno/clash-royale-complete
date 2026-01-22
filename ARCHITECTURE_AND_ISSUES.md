# Clash Royale MBRL System Architecture & Current Issues

This document summarizes the current single-VM "Hive" architecture (what runs where, how data flows) and the known issues across host, containers, perception, and training. Legacy remote gRPC is still supported but no longer the default.

---

## Architecture (High-Level)

### Components
- **Single VM (Azure NVads A10 v5, Ubuntu 22.04)**
  - Docker + NVIDIA Container Toolkit.
  - Redroid kernel modules (binder/ashmem).
- **Redroid Hive (5-10 Android containers)**
  - `docker/hive/docker-compose.yml`
  - Each container exposes ADB on localhost (5555, 5556, ...).
  - libhoudini enabled for ARM -> x86 translation.
- **DreamerV3 Brain (single process)**
  - `clash-royale-mbrl/train_dreamerv3.py --env-mode hive`
  - Runs KataCR perception locally (YOLO + OCR + state builder).
  - Steps multiple environments in a batch.

### Data Flow
1. DreamerV3 steps env i (localhost ADB).
2. Env captures frame via `adb exec-out screencap -p`.
3. KataCR builds state + reward, encodes semantic grid or pixels.
4. Agent returns action, applied via ADB taps.
5. Repeat across N envs per batch.

### Debug / Validation Artifacts
- `logs_dreamerv3_*/metrics.jsonl` and `scores.jsonl`
- `model_summary.json` and `config.yaml` per run
- Optional perception crops: `logs_dreamerv3_*/perception_crops/`

### Legacy Remote (Optional)
- gRPC FrameService + remote client loop (Machine A/B) remains supported in `train_dreamerv3.py --env-mode grpc`.

---

## Current Issues (Grouped)

### 1) Host / Redroid
- **Kernel modules**
  - binder/ashmem must be installed; missing modules prevent containers from booting.
- **ARM translation**
  - libhoudini instability can cause crashes or slow startup.

### 2) ADB Capture / Automation
- **ADB screencap throughput**
  - `adb exec-out screencap -p` can be a bottleneck with many containers.
- **Coordinate calibration**
  - UI tap coords + OK button probe need recalibration for 720x1280.

### 3) Perception
- **OCR noise / state drift**
  - Elixir OCR failures and tower HP jumps still affect action masks.
- **Slow inference**
  - YOLO + OCR remains heavy; FPS drops with N envs.

### 4) Training / DreamerV3
- **Parallel env contention**
  - Many envs increase GPU/CPU load; policy/update steps slow down.
- **Large model sizes**
  - `size400m` can stress A10 memory with multiple envs.

### 5) Docker / Compose
- **Port collisions**
  - Multiple Redroid stacks on same host collide on ADB ports.
- **Volume growth**
  - `./dataN` volumes grow; need cleanup between runs.

---

## Known Constraints / Risks
- **Compute limits**
  - A10 24GB VRAM is still tight for large models + perception.
- **No API access**
  - UI-based CV/OCR remains brittle to UI changes.
- **Wall-clock latency**
  - Multi-env contention can inflate step latency and reduce effective FPS.

---

## Suggested Next Stabilizations (Short-Term)
- Calibrate tap coords + OK button probe for Redroid resolution.
- Start with 2-4 envs; scale up after stable throughput.
- Reduce `--pixel-width/--pixel-height` if pixel mode slows down.
- Enable TensorBoard outputs (`--logger.outputs tensorboard jsonl scope`).
- Consider reducing perception load (single detector, OCR CPU/GPU toggle).
