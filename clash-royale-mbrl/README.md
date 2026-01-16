# Clash Royale Model-Based RL

An AI agent that learns to play Clash Royale using **Dreamer** world model with distributed training support. Machine A (GPU server) runs perception and training, Machine B (local) runs the Android emulator and streams frames.

Based on [KataCR](https://github.com/wty-yy/KataCR) perception and the paper ["KataCR: A Non-Embedded AI Agent for Clash Royale"](https://arxiv.org/abs/2406.17998).

## Features

- 🧠 **Dreamer World Model** - Learns environment dynamics from observations
- 🎮 **Remote Training** - Distributed setup with gRPC frame streaming
- 📊 **KataCR Perception** - YOLOv8 object detection + card classification
- 🖥️ **GPU Accelerated** - JAX for classification, PyTorch for Dreamer
- 🍎 **Multi-Platform** - Linux (CUDA) for training, macOS for emulator

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Distributed Training Setup                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Machine B (Mac/Local)              Machine A (GCP/GPU Server)              │
│  ┌─────────────────────┐            ┌──────────────────────────────┐        │
│  │  Android Emulator   │            │      FrameService (gRPC)     │        │
│  │  Clash Royale Game  │───BGR───▶  │                              │        │
│  │                     │  frames    │  ┌────────────────────────┐  │        │
│  │  remote_client_loop │            │  │  State Grid Mode       │  │        │
│  │  ◀──────────────────│◀─actions── │  │  - KataCR Perception   │  │        │
│  │                     │            │  │  - State Encoder       │  │        │
│  └─────────────────────┘            │  │  - MLP Encoder         │  │        │
│                                     │  └────────────────────────┘  │        │
│                                     │             OR                │        │
│                                     │  ┌────────────────────────┐  │        │
│                                     │  │  Pixel Mode (--pixels) │  │        │
│                                     │  │  - Resize to (H,W,3)   │  │        │
│                                     │  │  - CNN Encoder         │  │        │
│                                     │  └────────────────────────┘  │        │
│                                     │             │                 │        │
│                                     │             ▼                 │        │
│                                     │  ┌────────────────────────┐  │        │
│                                     │  │   DreamerV3 Training   │  │        │
│                                     │  │   - RSSM World Model   │  │        │
│                                     │  │   - Actor-Critic       │  │        │
│                                     │  │   - JAX/Ninjax         │  │        │
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
- No KataCR perception required on training server
```

### Action Space

Discrete action space with **37 actions**:
- 1 no-op + 4 cards × 9 deploy cells (3×3 grid on friendly side)

---

## Quick Start

### Machine A (GPU Server - Training)

```bash
# 1. Clone repositories
git clone <this-repo> clash-royale-complete
cd clash-royale-complete

# 2. Setup Python environment  
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies (DreamerV3 default)
cd clash-royale-mbrl
pip install -e .
# For NVIDIA GPUs, install matching jaxlib:
# pip install --upgrade "jaxlib[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 4. Download KataCR weights
python scripts/download_weights.py

# 5. Start DreamerV3 training server (listens for frames from Machine B)
python train_dreamerv3.py --rpc-host 0.0.0.0 --rpc-port 50051
```

### Machine B (Mac - Emulator)

```bash
# 1. Start Android emulator with Clash Royale
# 2. Navigate to Training Camp
# 3. Run the frame streaming client
python scripts/remote_client_loop.py --server-host <MACHINE_A_IP> --server-port 50051
```

---

## Installation

### System Requirements

**Machine A (Training Server):**
- Linux with NVIDIA GPU (tested on GCP n1-standard-4 + Tesla T4)
- CUDA 12.x compatible driver
- Python 3.11+

**Machine B (Emulator Host):**
- macOS or Linux
- Android Studio with emulator (1080×2400 resolution)
- ADB (Android Debug Bridge)

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
├── train_dreamerv3.py          # Main training entry (gRPC server + DreamerV3)
├── train_online.py             # Legacy DreamerV1/rlpyt entry (deprecated)
├── requirements-frozen.txt     # Pinned package versions (working)
├── requirements-apple-silicon.txt  # macOS deps
│
├── scripts/
│   ├── remote_client_loop.py   # Machine B: streams frames to server
│   ├── serve_frame_service.py  # Standalone gRPC server (no training)
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
| `train_dreamerv3.py` | Entry point: gRPC server + DreamerV3 training loop |
| `train_online.py` | Legacy DreamerV1/rlpyt entry (deprecated) |
| `src/cr/rpc/v1/processor.py` | Processes frames: KataCR perception → state grid |
| `src/environment/remote_bridge.py` | Bridges gRPC observations to Dreamer sampler |
| `scripts/remote_client_loop.py` | Client: captures emulator frames, sends to server |

---

## Training Flow

### State Grid Mode (default)

1. **Machine B** captures BGR frame from Android emulator
2. **Machine B** sends frame via gRPC to Machine A
3. **Machine A** runs KataCR perception:
   - YOLOv8 detects units, buildings, spells
   - CardClassifier (JAX) identifies hand cards
   - PaddleOCR reads game time
4. **Machine A** encodes state into (15, 32, 18) grid
5. **Machine A** feeds grid to Dreamer MLP encoder, gets action
6. **Machine A** returns action via gRPC response
7. **Machine B** executes tap on emulator
8. Repeat until match ends (detected by OK button color)

### Pixel Mode (--pixels)

1. **Machine B** captures BGR frame from Android emulator
2. **Machine B** sends frame via gRPC to Machine A
3. **Machine A** runs KataCR perception for **reward calculation** (tower HP tracking)
4. **Machine A** resizes frame to (H, W, 3) RGB for **observation**
5. **Machine A** feeds pixels to Dreamer CNN encoder, gets action
6. **Machine A** returns action via gRPC response
7. **Machine B** executes tap on emulator

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
