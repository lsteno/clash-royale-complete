# Clash Royale Model-Based RL (DreamerV3)

An AI agent that learns to play Clash Royale using **DreamerV3** world model. Supports both offline training on expert replays and online training with parallel emulators.

Based on [KataCR](https://github.com/wty-yy/KataCR) perception and the paper ["KataCR: A Non-Embedded AI Agent for Clash Royale"](https://arxiv.org/abs/2406.17998).

## Features

- 🧠 **DreamerV3 World Model** - Learns environment dynamics from data
- 🎮 **Live Play** - Plays via ADB on Android emulators
- 📊 **Offline Training** - Learn from expert replay dataset
- 🚀 **Online Training** - Train with 3+ parallel emulators
- 🍎 **Apple Silicon** - Optimized for MPS (M1/M2/M3)

---

## Quick Start

```bash
# 1. Clone this repo and dependencies
git clone <this-repo> clash-royale-mbrl
git clone https://github.com/wty-yy/KataCR
git clone https://github.com/wty-yy/Clash-Royale-Replay-Dataset

# 2. Setup Python environment
cd clash-royale-mbrl
python -m venv ../.venv
source ../.venv/bin/activate
pip install -r requirements-apple-silicon.txt

# 3. Download KataCR detector weights (see below)

# 4. Train offline (~15 min)
python train_offline_fast.py --epochs 5 --steps-per-epoch 300

# 5. Play live
python play_live.py --checkpoint logs/fast_*/best.pt --debug
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DreamerV3 World Model                       │
├─────────────────────────────────────────────────────────────────┤
│  Observation: (15, 32, 18) spatial grid                         │
│  - Channels 0-9: Unit category one-hot                          │
│  - Channel 10-11: Enemy/Friendly flag                           │
│  - Channel 12: Unit health                                      │
│  - Channel 13: Elixir (0-10 normalized)                         │
│  - Channel 14: Game time (normalized)                           │
├─────────────────────────────────────────────────────────────────┤
│  RSSM (Recurrent State-Space Model):                            │
│  - Encoder: MLP (obs → 256-dim embedding)                       │
│  - GRU: 256 deterministic state                                 │
│  - Stochastic: 32-dim Gaussian latent                           │
│  - Decoder: reconstruct observation                             │
│  - Reward head: predict rewards                                 │
├─────────────────────────────────────────────────────────────────┤
│  Actor-Critic (trained via imagination):                        │
│  - Actor: latent → action distribution                          │
│  - Critic: latent → value estimate                              │
│  Parameters: ~5.5M                                              │
└─────────────────────────────────────────────────────────────────┘
```

### Action Space (Paper Style)

Following the KataCR paper, actions are continuous:
- `delay` ∈ [0, 20]: frames to wait before acting
- `pos_x` ∈ [0, 1]: normalized x position
- `pos_y` ∈ [0, 1]: normalized y position  
- `card` ∈ {0, 1, 2, 3}: which card to play

The model learns **when to act** (delay prediction) rather than acting every frame.

---

## Installation

### Requirements

- macOS with Apple Silicon (M1/M2/M3) OR Linux with CUDA
- Python 3.11+
- Android Studio with emulator (1080x2400 resolution)
- ADB (Android Debug Bridge)

### Dependencies

```bash
# Create virtual environment
python -m venv ../.venv
source ../.venv/bin/activate

# Install packages
pip install torch torchvision torchaudio  # PyTorch with MPS
pip install numpy opencv-python tqdm
pip install ultralytics==8.1.24  # For KataCR YOLOv8

# Or use requirements file
pip install -r requirements-apple-silicon.txt
```

### KataCR Detector Weights

Download from [KataCR Releases](https://github.com/wty-yy/KataCR/releases):
- `detector1_v0.7.13.pt` → `KataCR/runs/detector1_v0.7.13/weights/best.pt`
- `detector2_v0.7.13.pt` → `KataCR/runs/detector2_v0.7.13/weights/best.pt`

### Expert Replay Dataset

```bash
git clone https://github.com/wty-yy/Clash-Royale-Replay-Dataset
# Uses: Clash-Royale-Replay-Dataset/fast_hog_2.6/
```

---

## Training

### Offline Training (Recommended Start)

Train on expert replays without needing emulators:

```bash
source ../.venv/bin/activate

# Quick test (~15 min)
python train_offline_fast.py --epochs 5 --steps-per-epoch 300

# Full training (~1 hour)  
python train_offline_fast.py --epochs 30 --steps-per-epoch 1000

# Paper-style with delay prediction
python train_paper.py --epochs 30
```

Checkpoints saved to `logs/fast_YYYYMMDD_HHMMSS/`

### Online Training (3 Parallel Emulators)

For faster learning with real gameplay:

```bash
# 1. Create 3 emulator AVDs in Android Studio
#    Names: Pixel_6, Pixel_6_2, Pixel_6_3
#    Resolution: 1080x2400

# 2. Start emulators on different ports
emulator -avd Pixel_6 -port 5554 &
emulator -avd Pixel_6_2 -port 5556 &
emulator -avd Pixel_6_3 -port 5558 &

# 3. Install Clash Royale and login with DIFFERENT accounts
#    (Supercell only allows 1 login per account)
#    Use Guest accounts for testing

# 4. Go to Training Camp on each emulator

# 5. Run online training
python train_live.py --n-envs 3
```

Workers collect experience → Replay Buffer → World Model Training → Imagination → Actor-Critic

---

## Live Play

```bash
# Start emulator with Clash Royale open
adb devices  # Verify: "emulator-5554 device"

# Run agent
python play_live.py --checkpoint logs/fast_*/best.pt --debug

# Paper-style model
python play_paper.py --checkpoint logs/paper_*/best.pt
```

The agent will:
1. Wait for battle to start (detects elixir bar)
2. Play cards during battle
3. Wait for next battle when game ends

---

## Project Structure

```
clash-royale-mbrl/
├── train_offline_fast.py   # Offline DreamerV3 training (discrete actions)
├── train_paper.py          # Paper-style training (delay + continuous)
├── train_live.py           # Online training with parallel emulators
├── play_live.py            # Live gameplay agent
├── play_paper.py           # Paper-style live agent
├── src/
│   ├── agent/              # DreamerV3 model components
│   ├── environment/        # Clash Royale env wrapper
│   ├── perception/         # KataCR YOLOv8 integration
│   └── utils/              # Helpers
├── logs/                   # Training checkpoints
└── scripts/                # Setup utilities
```

---

## Key Files

| File | Description |
|------|-------------|
| `train_offline_fast.py` | Offline training on expert replays, discrete 37-action space |
| `train_paper.py` | Paper-style training with delay prediction, continuous actions |
| `train_live.py` | Online DreamerV3 with 3 parallel emulators |
| `play_live.py` | Live agent using trained model |
| `play_paper.py` | Paper-style live agent with delay threshold |

---

## Configuration

### Screen Coordinates (1080x2400)

```python
CARD_POSITIONS = [(270, 2220), (460, 2220), (650, 2220), (840, 2220)]
ARENA_LEFT, ARENA_RIGHT = 60, 1020
DEPLOY_TOP, DEPLOY_BOTTOM = 1200, 1800
ELIXIR_ROI = (200, 2300, 1050, 2400)
```

### Model Hyperparameters

```python
HIDDEN_DIM = 256
DETER_DIM = 256      # GRU hidden state
STOCH_DIM = 32       # Stochastic latent
SEQ_LENGTH = 50      # Training sequence length
HORIZON = 15         # Imagination horizon
```

---

## Troubleshooting

### "No emulator found"
```bash
adb kill-server && adb start-server
adb devices
```

### "Model keeps losing"
- Ensure training data is from same deck
- Try paper-style training with delay prediction
- Increase training epochs

### "KataCR not found"
```bash
# Check KataCR is cloned at correct path
ls ../KataCR/runs/detector1_v0.7.13/weights/best.pt
```

---

## References

- [KataCR: A Non-Embedded AI Agent for Clash Royale](https://github.com/wty-yy/KataCR)
- [DreamerV3: Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)
- [Clash Royale Replay Dataset](https://github.com/wty-yy/Clash-Royale-Replay-Dataset)

---

## License

MIT
