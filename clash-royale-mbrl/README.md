# Clash Royale Model-Based RL

An AI agent that learns to play Clash Royale using a **Dreamer** world model. It supports two execution modes:
- **Hive (recommended):** single Linux GPU VM running multiple Redroid Android containers over localhost ADB.
- **Legacy Remote:** Mac emulator streaming frames to a GPU server over gRPC.

Based on [KataCR](https://github.com/wty-yy/KataCR) perception and the paper ["KataCR: A Non-Embedded AI Agent for Clash Royale"](https://arxiv.org/abs/2406.17998).

## Features

- 🧠 **Dreamer World Model** - Learns environment dynamics from observations
- 🐝 **Hive Parallelism** - Multiple Redroid instances per GPU VM
- 🎮 **Local or Remote** - Hive (localhost ADB) or legacy gRPC streaming
- 📊 **KataCR Perception** - YOLOv8 object detection + card classification
- 🖥️ **GPU Accelerated** - JAX for classification, PyTorch for Dreamer

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Hive (Single VM, Recommended)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  Ubuntu 22.04 + NVIDIA GPU                                                   │
│                                                                              │
│  DreamerV3 (single process)                                                  │
│   ├─ KataCR perception + reward                                              │
│   ├─ Batch env stepper (N Redroid containers)                                │
│   └─ Localhost ADB actions (5555, 5556, ...)                                 │
│                                                                              │
│  Docker: redroid/redroid containers (ARM -> x86 via libhoudini)              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Legacy Remote (gRPC)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Distributed Training Setup                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Machine B (Mac/Local)              Machine A (GPU Server)                  │
│  ┌─────────────────────┐            ┌──────────────────────────────┐        │
│  │  Android Emulator   │            │      FrameService (gRPC)     │        │
│  │  Clash Royale Game  │───BGR───▶  │                              │        │
│  │                     │  frames    │  ┌────────────────────────┐  │        │
│  │  remote_client_loop │            │  │  KataCR Perception      │  │        │
│  │  ◀──────────────────│◀─actions── │  │  - State Encoder        │  │        │
│  └─────────────────────┘            │  │  - DreamerV3 Training   │  │        │
│                                     │  └────────────────────────┘  │        │
│                                     └──────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Observation Space

The agent supports two observation modes:

**Mode 1: State Grid (default)** - KataCR perception extracts game state
```
State Grid: (15, 32, 18) spatial tensor, flattened to (8640,) for MLP
├── Channel 0:   Friendly ground units (count per cell)
├── Channel 1:   Friendly air units
├── Channel 2:   Enemy ground units
├── Channel 3:   Enemy air units
├── Channel 4:   Friendly spells
├── Channel 5:   Enemy spells
├── Channel 6:   Friendly structures (HP ratio)
├── Channel 7:   Enemy structures (HP ratio)
├── Channel 8:   Elixir (0-10 normalized, broadcast)
├── Channel 9:   Game time (0-360s normalized, broadcast)
├── Channel 10:  Next card in queue (card index normalized)
└── Channels 11-14: Current hand cards 1-4 (card index normalized)
```

**Mode 2: Pixel (--pixels)** - Raw RGB frames for CNN encoder
```
Pixel Observation: (H, W, 3) uint8 RGB, default (192, 256, 3)
- Resized emulator frame (channels-last for DreamerV3 CNN encoder)
- Perception still runs for reward/action masking; observation is pixels
```

### Action Space

Discrete action space with **37 actions**:
- 1 no-op + 4 cards × 9 deploy cells (3×3 grid on friendly side)

---

## Quick Start

### Hive (Single VM)

```bash
# 1. Clone repositories
git clone <this-repo> clash-royale-complete
cd clash-royale-complete

# 2. Host setup (Ubuntu 22.04)
./clash-royale-mbrl/scripts/vm_hive_setup.sh

# 3. Start Redroid containers
./clash-royale-mbrl/scripts/hive_up.sh

# 4. Install Clash Royale APK on containers
./clash-royale-mbrl/scripts/hive_install_apk.sh ./clash.apk 5555 5556

# 5. Setup Python environment
python3.11 -m venv .venv
source .venv/bin/activate
cd clash-royale-mbrl
pip install -e .
python scripts/download_weights.py

# 6. Start DreamerV3 in hive mode
python train_dreamerv3.py --env-mode hive --adb-start-port 5555 --adb-count 2
```

Note: Keep `--redroid-width/--redroid-height` in sync with `docker/hive/docker-compose.yml`.

### Legacy Remote (Machine A/B)

**Machine A (GPU Server - Training)**

```bash
cd clash-royale-complete/clash-royale-mbrl
python train_dreamerv3.py --rpc-host 0.0.0.0 --rpc-port 50051
```

**Machine B (Mac - Emulator)**

```bash
python scripts/remote_client_loop.py --target <MACHINE_A_IP>:50051 --want-action
```

---

## Installation

### System Requirements

**Hive (Single VM):**
- Ubuntu 22.04 with NVIDIA GPU (A10-class recommended)
- Docker + NVIDIA Container Toolkit
- Redroid kernel modules (binder/ashmem)
- ADB (android-platform-tools)

**Legacy Remote:**
- Machine A (Training Server): Linux with NVIDIA GPU, CUDA 12.x, Python 3.11+
- Machine B (Emulator Host): macOS/Linux + Android Studio emulator (1080x2400) + ADB

### Verified Package Versions (Linux CUDA 12)

| Package | Version | Notes |
|---------|---------|-------|
| jax | 0.4.26 | Must init before PyTorch! |
| jaxlib | 0.4.26+cuda12.cudnn89 | CUDA 12 build |
| torch | 2.1.2+cu121 | CUDA 12.1 |
| flax | 0.8.1 | JAX neural networks |
| orbax-checkpoint | 0.4.4 | JAX checkpointing |
| ultralytics | 8.1.24 | YOLOv8 for detection |
| paddlepaddle-gpu | 2.6.1 | OCR support |
| grpcio | 1.76.0 | Frame streaming |
| numpy | 1.26.4 | |

See `requirements-frozen.txt` for complete list.

### ⚠️ Critical: JAX/PyTorch CUDA Conflict

JAX must initialize CUDA **before** PyTorch is imported. The training script handles this automatically, but if you're writing custom code:

```python
# CORRECT ORDER - JAX first!
import jax
jax.devices()  # Force CUDA initialization

import torch  # Now safe to import

# WRONG - will cause "cuSOLVER not found" error
import torch
import jax  # Too late - CUDA already broken
```

---

## Project Structure

```
clash-royale-mbrl/
├── train_dreamerv3.py          # Main training entry (hive or gRPC)
├── train_online.py             # Legacy DreamerV1/rlpyt entry (deprecated)
├── requirements-frozen.txt     # Pinned package versions (working)
├── requirements-apple-silicon.txt  # macOS deps
│
├── scripts/
│   ├── hive_up.sh              # Launch Redroid hive containers
│   ├── hive_install_apk.sh     # Install APK across Redroid instances
│   ├── vm_hive_setup.sh        # Host setup for Redroid + Docker
│   ├── remote_client_loop.py   # Legacy: streams frames to server
│   ├── serve_frame_service.py  # Standalone gRPC server (legacy)
│   ├── download_weights.py     # Download KataCR model weights
│   └── interactive_setup.py    # Guided setup wizard
│
├── src/
│   ├── cr/rpc/v1/
│   │   ├── processor.py        # Frame processing (perception → grid)
│   │   └── server.py           # gRPC server scaffold
│   │
│   ├── environment/
│   │   ├── online_env.py       # Gym env wrapper
│   │   ├── remote_bridge.py    # Connects gRPC to Dreamer
│   │   ├── hive_env.py         # Local Redroid hive env (embodied)
│   │   └── emulator_env.py     # ADB interaction
│   │
│   ├── perception/
│   │   └── katacr_pipeline.py  # KataCR integration
│   │
│   └── specs.py                # Observation/action specs
│
├── proto/
│   └── frame_service.proto     # gRPC service definition
│
└── logs_online/                # Training checkpoints & TensorBoard
```

---

## Key Files

| File | Description |
|------|-------------|
| `train_dreamerv3.py` | Entry point: hive or gRPC DreamerV3 training |
| `train_online.py` | Legacy DreamerV1/rlpyt entry (deprecated) |
| `src/cr/rpc/v1/processor.py` | Processes frames: KataCR perception → state grid |
| `src/environment/remote_bridge.py` | Bridges gRPC observations to Dreamer sampler |
| `src/environment/hive_env.py` | Local Redroid hive environment (embodied) |
| `scripts/hive_up.sh` | Bring up Redroid containers |
| `scripts/hive_install_apk.sh` | Install APK on all containers |
| `scripts/remote_client_loop.py` | Legacy client: captures emulator frames, sends to server |

---

## Training Flow

### Hive Mode (local, default)

1. **DreamerV3** steps N Redroid environments in a batch.
2. Each env captures a frame via local ADB screencap.
3. KataCR perception builds state + reward locally.
4. Observations are either state grids or pixel frames.
5. Actions are applied via localhost ADB taps.

### Legacy Remote (gRPC) - State Grid Mode

1. **Machine B** captures BGR frame from Android emulator.
2. **Machine B** sends frame via gRPC to Machine A.
3. **Machine A** runs KataCR perception.
4. **Machine A** encodes state into (15, 32, 18) grid.
5. **Machine A** feeds grid to Dreamer MLP encoder, gets action.
6. **Machine A** returns action via gRPC response.
7. **Machine B** executes tap on emulator.

### Legacy Remote (gRPC) - Pixel Mode (--pixels)

1. **Machine B** captures BGR frame from Android emulator.
2. **Machine B** sends frame via gRPC to Machine A.
3. **Machine A** runs KataCR perception for reward.
4. **Machine A** resizes frame to (H, W, 3) RGB for observation.
5. **Machine A** feeds pixels to Dreamer CNN encoder, gets action.
6. **Machine A** returns action via gRPC response.
7. **Machine B** executes tap on emulator.

> **Note:** Pixel mode differences:
> - **Observation:** Raw RGB pixels (learned visual features via CNN)
> - **Rewards:** Still computed via KataCR perception (tower HP, enemy elimination)
> - **Action masking:** Still uses cards/elixir info from perception
> - **Higher compute:** CNN encoder + perception pipeline both run

---

## Configuration

### gRPC Server Settings (DreamerV3)

```bash
python train_dreamerv3.py \
  --rpc-host 0.0.0.0 \    # Listen on all interfaces
  --rpc-port 50051 \      # gRPC port
  --logdir ./logs_dreamerv3

# Optional: train directly from emulator pixels (channels-last RGB)
python train_dreamerv3.py --pixels --pixel-height 180 --pixel-width 320
```

### Firewall (GCP)

Ensure port 50051 is open:
```bash
gcloud compute firewall-rules create allow-grpc \
  --allow tcp:50051 \
  --source-ranges 0.0.0.0/0
```

---

## Troubleshooting

### "cuSOLVER not found" / JAX falls back to CPU

**Cause:** PyTorch initialized CUDA before JAX.

**Fix:** The training script now forces JAX initialization first. If using custom code, import JAX and call `jax.devices()` before any PyTorch imports.

### "CardClassifier fell back to dummy mode"

**Cause:** JAX couldn't load checkpoint (CUDA not available or checkpoint was saved with different device).

**Fix:** Ensure JAX has CUDA access (`jax.devices()` should show `cuda(id=0)`).

### "No emulator found"

```bash
adb kill-server && adb start-server
adb devices  # Should show "emulator-5554 device"
```

### gRPC connection refused

1. Check server is running: `netstat -tlnp | grep 50051`
2. Check firewall allows port 50051
3. Verify IP address in client matches server

---

## References

- [KataCR: A Non-Embedded AI Agent for Clash Royale](https://github.com/wty-yy/KataCR)
- [Dreamer: Dream to Control](https://arxiv.org/abs/1912.01603)
- [DreamerV3: Mastering Diverse Domains](https://arxiv.org/abs/2301.04104)
- [Clash Royale Replay Dataset](https://github.com/wty-yy/Clash-Royale-Replay-Dataset)

---

## License

MIT
