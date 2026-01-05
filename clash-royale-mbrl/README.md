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
│  │  Clash Royale Game  │───BGR───▶  │  ┌────────────────────────┐  │        │
│  │                     │  frames    │  │   KataCR Perception    │  │        │
│  │  remote_client_loop │            │  │  - YOLOv8 Detection    │  │        │
│  │  ◀──────────────────│◀─actions── │  │  - Card Classification │  │        │
│  │                     │            │  │  - OCR (PaddleOCR)     │  │        │
│  └─────────────────────┘            │  └────────────────────────┘  │        │
│                                     │             │                 │        │
│                                     │             ▼                 │        │
│                                     │  ┌────────────────────────┐  │        │
│                                     │  │   State Grid Encoder   │  │        │
│                                     │  │   (15, 32, 18) tensor  │  │        │
│                                     │  └────────────────────────┘  │        │
│                                     │             │                 │        │
│                                     │             ▼                 │        │
│                                     │  ┌────────────────────────┐  │        │
│                                     │  │   Dreamer Training     │  │        │
│                                     │  │   - RSSM World Model   │  │        │
│                                     │  │   - Actor-Critic       │  │        │
│                                     │  └────────────────────────┘  │        │
│                                     └──────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Observation Space

```
State Grid: (15, 32, 18) spatial tensor
├── Channels 0-9:   Unit category one-hot encoding
├── Channel 10-11:  Enemy/Friendly flag
├── Channel 12:     Unit health (normalized)
├── Channel 13:     Elixir (0-10 normalized)
└── Channel 14:     Game time (normalized)
```

### Action Space

Discrete action space with 324 actions:
- 4 cards × 81 grid positions (9×9 deployment grid)

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

# 3. Install dependencies
cd clash-royale-mbrl
pip install -r requirements-frozen.txt

# 4. Download KataCR weights
python scripts/download_weights.py

# 5. Start training server (listens for frames from Machine B)
python train_online.py --use-remote-frames --rpc-host 0.0.0.0 --rpc-port 50051
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
├── train_online.py             # Main training entry (gRPC server + Dreamer)
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
| `train_online.py` | Entry point: starts gRPC server + Dreamer training loop |
| `src/cr/rpc/v1/processor.py` | Processes frames: KataCR perception → state grid |
| `src/environment/remote_bridge.py` | Bridges gRPC observations to Dreamer sampler |
| `scripts/remote_client_loop.py` | Client: captures emulator frames, sends to server |

---

## Training Flow

1. **Machine B** captures BGR frame from Android emulator
2. **Machine B** sends frame via gRPC to Machine A
3. **Machine A** runs KataCR perception:
   - YOLOv8 detects units, buildings, spells
   - CardClassifier (JAX) identifies hand cards
   - PaddleOCR reads game time
4. **Machine A** encodes state into (15, 32, 18) grid
5. **Machine A** feeds grid to Dreamer, gets action
6. **Machine A** returns action via gRPC response
7. **Machine B** executes tap on emulator
8. Repeat until match ends (detected by OK button color)

---

## Configuration

### gRPC Server Settings

```bash
python train_online.py \
  --use-remote-frames \
  --rpc-host 0.0.0.0 \    # Listen on all interfaces
  --rpc-port 50051 \      # gRPC port
  --num-envs 1 \          # Number of parallel environments
  --batch-T 8             # Trajectory length per batch
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
