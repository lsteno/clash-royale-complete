# Dreamer in Clash Royale (Repo Overview)

This document explains how Dreamer works in this repo and describes the three
top-level folders:

- `clash-royale-mbrl`
- `KataCR`
- `dreamer-pytorch`

It also includes an end-to-end frame-to-action trace and the exact observation
and action specs used by Dreamer.

---

## 1) End-to-End Flow (Frame -> Dreamer -> Action)

There are two modes: local emulator training (single machine) and remote
training (two machines). Both end up in the same Dreamer loop.

### Remote training (Machine B -> Machine A)

1) **Capture frame on Machine B**
   - `clash-royale-mbrl/scripts/remote_client_loop.py` reads emulator frames
     via `ClashRoyaleEmulatorEnv` in `clash-royale-mbrl/src/environment/emulator_env.py`.
   - Frames are raw BGR bytes.

2) **Send frame over gRPC**
   - `clash-royale-mbrl/src/cr/rpc/v1/client.py` sends `ProcessFrameRequest`
     defined in `clash-royale-mbrl/proto/frame_service.proto`.

3) **Perception + state encoding on Machine A**
   - `clash-royale-mbrl/src/cr/rpc/v1/processor.py` receives the frame and runs
     `KataCRPerceptionEngine` in `clash-royale-mbrl/src/perception/katacr_pipeline.py`.
   - KataCR produces state + reward; `StateTensorEncoder` in
     `clash-royale-mbrl/src/environment/online_env.py` converts it to a
     `(15, 32, 18)` float tensor.

4) **Bridge to Dreamer**
   - The processor publishes the state to `RemoteBridge` in
     `clash-royale-mbrl/src/environment/remote_bridge.py`.
   - `RemoteClashRoyaleEnv` exposes a gym-style env that Dreamer can step.

5) **Dreamer selects an action**
   - `clash-royale-mbrl/train_online.py` constructs `Dreamer` and
     `AtariDreamerAgent` from `dreamer-pytorch`.
   - Actions are discrete one-hot over 37 actions (1 no-op + 4 cards * 9 cells).

6) **Return action to Machine B**
   - The action tuple `(card_idx, grid_x, grid_y)` goes back through gRPC
     to `remote_client_loop.py`, which taps the emulator via ADB.

### Local training (single machine)

1) `ClashRoyaleDreamerEnv` in `clash-royale-mbrl/src/environment/online_env.py`
   calls `ClashRoyaleKataCREnv` (emulator + KataCR).
2) The same `StateTensorEncoder` produces the observation tensor.
3) Dreamer runs inside `train_online.py` and taps the emulator directly.

---

## 2) Observation and Action Specs

Defined in `clash-royale-mbrl/src/specs.py`:

- **Observation**: `(15, 32, 18)` float grid.
- **Action space**: 37 discrete actions.
  - `0` = no-op
  - `1..36` = 4 cards * 9 deploy cells

`ActionMapper` in `clash-royale-mbrl/src/environment/online_env.py` decodes the
index into `(card_slot, grid_x, grid_y)`.

---

## 3) Action Masking (Legal Actions Only)

To prevent illegal moves (not enough elixir, missing cards), this repo adds an
action mask:

- `clash-royale-mbrl/src/environment/action_mask.py` builds a per-action mask
  from the current cards + elixir.
- The mask is stored in a contextvar and consumed by Dreamer:
  - `dreamer-pytorch/dreamer/models/action.py` applies it to logits.
  - `dreamer-pytorch/dreamer/agents/dreamer_agent.py` uses it for
    epsilon-greedy random actions.
  - `dreamer-pytorch/dreamer/algos/dreamer_algo.py` clears it for imagination
    rollouts so the world model is not constrained.

---

## 4) Folder-by-Folder Details

### `clash-royale-mbrl`

Purpose: Dreamer + gRPC + emulator integration, using KataCR for perception.

Key folders and files:

- `clash-royale-mbrl/train_online.py`
  - Main entrypoint for Dreamer training.
  - Adds `dreamer-pytorch` and `rlpyt` to `sys.path`.
  - Supports local emulator or remote gRPC frames.

- `clash-royale-mbrl/src/environment`
  - `online_env.py`: Dreamer-facing env wrapper and state encoder.
  - `remote_bridge.py`: Thread-safe handoff between gRPC processor and trainer.
  - `emulator_env.py`: ADB taps, screen capture, match navigation.
  - `action_mask.py`: Legal action masking logic.

- `clash-royale-mbrl/src/perception`
  - `katacr_pipeline.py`: Full KataCR perception (YOLO + OCR + classifier).
  - `detection.py` / `katacr_adapter.py`: lighter alternatives.

- `clash-royale-mbrl/src/cr/rpc/v1`
  - `processor.py`: gRPC processor; runs perception and optional Dreamer action.
  - `server.py`: gRPC server scaffold.
  - `client.py`: gRPC async client.

- `clash-royale-mbrl/proto/frame_service.proto`
  - Protocol for sending frames and receiving state/action.

- `clash-royale-mbrl/scripts`
  - `remote_client_loop.py`: Machine B streaming loop.
  - `serve_frame_service.py`: perception-only server.
  - `capture_perception_snapshot.py`: debug snapshots and logs.

---

### `KataCR`

Purpose: Perception, OCR, and (offline) policy models for Clash Royale.

Key folders:

- `KataCR/katacr/yolov8`: custom YOLOv8 training/inference.
- `KataCR/katacr/classification`: ResNet card/elixir classifiers.
- `KataCR/katacr/ocr_text`: PaddleOCR wrappers and OCR utilities.
- `KataCR/katacr/policy/perceptron`: `StateBuilder` and `RewardBuilder`
  used by `clash-royale-mbrl`.
- `KataCR/katacr/build_dataset`: dataset generation and video processing.
- `KataCR/katacr/constants`: card names, labels, elixir costs.
- `KataCR/katacr/utils`: shared image/video helpers.

In this repo, KataCR is the perception backbone feeding Dreamer.

---

### `dreamer-pytorch`

Purpose: The Dreamer algorithm implementation used for training.

Key folders:

- `dreamer-pytorch/dreamer/models`: RSSM, observation, reward, value, action.
- `dreamer-pytorch/dreamer/algos`: Dreamer training loop + replay.
- `dreamer-pytorch/dreamer/agents`: Dreamer agents and sampling.
- `dreamer-pytorch/dreamer/envs`: wrappers like `one_hot.py`.
- `dreamer-pytorch/rlpyt`: sampler/runner for online RL.

In this repo, `clash-royale-mbrl/train_online.py` is the main driver that uses
these components.

---

## 5) Quick Reference: Key Files

- `clash-royale-mbrl/train_online.py`
- `clash-royale-mbrl/src/environment/online_env.py`
- `clash-royale-mbrl/src/environment/remote_bridge.py`
- `clash-royale-mbrl/src/perception/katacr_pipeline.py`
- `clash-royale-mbrl/src/cr/rpc/v1/processor.py`
- `clash-royale-mbrl/scripts/remote_client_loop.py`
- `dreamer-pytorch/dreamer/algos/dreamer_algo.py`
- `dreamer-pytorch/dreamer/models/action.py`
- `KataCR/katacr/policy/perceptron/reward_builder.py`

