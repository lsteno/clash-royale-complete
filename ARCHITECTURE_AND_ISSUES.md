# Clash Royale MBRL System Architecture & Current Issues

This document summarizes the current end-to-end architecture (what runs where, how data flows) and the known issues across the environment, game client, VM, and tooling.

---

## Architecture (High-Level)

### Components
- **Machine B (Mac + Emulator)**
  - Android emulator (e.g., MEmu or similar), controlled via ADB + UI automation.
  - `clash-royale-mbrl/scripts/remote_client_loop.py` captures frames, navigates UI, and sends gRPC requests.
  - Screen capture via scrcpy window + ROI (manual `capture_region`).

- **Machine A (Azure VM + GPU)**
  - Docker container (`docker/machineA/Dockerfile`, `docker/machineA/docker-compose.vm.yml`).
  - Runs **DreamerV3** training (`clash-royale-mbrl/train_dreamerv3.py`).
  - Runs **gRPC FrameService** (`cr.rpc.v1.processor.FrameServiceProcessor`) for perception + state encoding.
  - **KataCR** perception stack (YOLO + OCR + state builder).

### Data Flow
1. **Mac client captures frame** from emulator (scrcpy ROI) and sends to VM via gRPC.
2. **VM FrameService** decodes frame, runs perception (YOLO/OCR), encodes semantic grid or pixels.
3. **RemoteBridge** posts observation/reward to Dreamer training loop.
4. **DreamerV3** returns action, VM replies to client.
5. **Mac client executes action** via ADB (tap card + placement).

### Debug / Validation Artifacts
- **VM debug dumps** (server-side):
  - `/mnt/azureuser/dumps_*/frame_*` → `frame_bgr.png`, `arena_boxes.png`, `katacr_render.png`, `state.json`, `info.json`, `obs.npy`, `obs_channels.png`, `action_mask.npy`.
- **Mac capture debugging** (client-side):
  - `--capture-debug-dir` saves `capture_full_*.png` and `capture_roi_*.png`.
- **Training logs**
  - `/mnt/azureuser/logs_dreamerv3_*/metrics.jsonl`
  - `/mnt/azureuser/logs_dreamerv3_*/scores.jsonl`
  - `model_summary.json` and `config.yaml` per run.

---

## Current Issues (Grouped)

### 1) Environment / Perception
- **OCR noise / state drift**
  - Elixir OCR occasionally fails → wrong action masks.
  - Tower HP OCR can jump (warnings like `ocr_hp > old hp`).
- **YOLO/Tracking dependencies**
  - Ultralytics tracker requires `lap` (fixed in Dockerfile).
- **Slow perception**
  - YOLO + OCR + rendering makes inference heavy; low end-to-end FPS.

### 2) Client (Mac) + Capture
- **ROI calibration fragile**
  - Incorrect `capture_region` crops out HUD/cards → model sees wrong state.
- **Menu navigation failure modes**
  - UI taps can get stuck on menus or end screens without recovery.

### 3) Training / DreamerV3
- **Stalls when frames stop**
  - Trainer used to crash on 600s timeout (fixed to wait indefinitely).
- **Large model sizes**
  - `size400m` fits but may be slow; compile time high; memory pressure on A10.

### 4) Networking / Tunneling
- **FD shutdown / UNAVAILABLE**
  - Happens when SSH tunnel dies or connects to wrong local port.
  - Multiple tunnels on same port cause conflicts.

### 5) Docker / Build Workflow
- **Rebuilds too often**
  - No `.dockerignore` originally; copying whole repo caused cache invalidation.
  - Added `.dockerignore` and reordered Dockerfile to improve cache.
- **Permission errors**
  - OCR caches attempted to write to `/.paddleocr` → fixed via `HOME=/tmp` and `PADDLEOCR_BASE_DIR`.

### 6) Logging / TensorBoard
- **TensorBoard empty**
  - `logger.outputs` defaults to `[jsonl, scope]`; TensorBoard needs `tensorboard`.
  - Must pass `--logger.outputs tensorboard jsonl scope` (requires restart).

---

## Known Constraints / Risks
- **Compute limits**
  - A10 24GB VRAM is tight for `size400m` + perception; throughput is low.
- **No API access**
  - Reliance on CV/OCR over UI makes the pipeline brittle to UI changes.
- **Wall-clock latency**
  - Slow perception + gRPC + ADB = delayed feedback for policy learning.

---

## Suggested Next Stabilizations (Short-Term)
- Lock ROI using capture debug images before long runs.
- Reduce `fps` and `action_hz` until RPC deadlines are stable.
- Increase `--deadline-ms` for large models.
- Enable TensorBoard outputs to monitor actual learning curves.
- Add a “dump current state now” endpoint (optional) for inspection without restarting.

