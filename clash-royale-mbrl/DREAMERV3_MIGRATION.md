# Clash Royale MBRL: DreamerV3 Migration Summary

## Project Scope

### What This Project Is

A **Model-Based Reinforcement Learning (MBRL)** system that trains an AI agent to play Clash Royale using the **Dreamer** world model algorithm. The system uses:

- **KataCR** - Computer vision pipeline (YOLOv8 detection + card classification + OCR)
- **Dreamer** - World model that learns environment dynamics and plans actions
- **Distributed Architecture** - Remote frame streaming from emulator to GPU server

### Architecture Overview

```
Machine B (Mac/Local)                    Machine A (GPU Server)
┌─────────────────────┐                  ┌───────────────────────────────┐
│  Android Emulator   │                  │     FrameService (gRPC)       │
│  Clash Royale Game  │───BGR frames───▶ │  ┌─────────────────────────┐  │
│                     │                  │  │  KataCR Perception      │  │
│  remote_client_loop │◀────actions───── │  │  (YOLOv8 + OCR)         │  │
└─────────────────────┘                  │  └───────────┬─────────────┘  │
                                         │              ▼                 │
                                         │  ┌─────────────────────────┐  │
                                         │  │  State Encoder          │  │
                                         │  │  (15, 32, 18) tensor    │  │
                                         │  └───────────┬─────────────┘  │
                                         │              ▼                 │
                                         │  ┌─────────────────────────┐  │
                                         │  │  Dreamer Agent          │  │
                                         │  │  (World Model + Policy) │  │
                                         │  └─────────────────────────┘  │
                                         └───────────────────────────────┘
```

### Observation & Action Spaces

| Space | Format | Description |
|-------|--------|-------------|
| **Observation** | `(15, 32, 18)` float32 tensor | 15 channels encoding units, elixir, time, cards over 32×18 grid |
| **Action** | Discrete `[0, 37)` | 1 no-op + 4 cards × 9 deploy positions |

---

## Current Goal: Migrate from DreamerV1 to DreamerV3

### Why DreamerV3?

| Aspect | DreamerV1 (dreamer-pytorch) | DreamerV3 (dreamerv3-main) |
|--------|-----------------------------|-----------------------------|
| Framework | PyTorch | JAX (faster, better GPU utilization) |
| Algorithm | Original Dreamer | Latest improvements (symlog, twohot, etc.) |
| Architecture | Tied to rlpyt sampler | Flexible embodied.Env interface |
| Scaling | Limited | Multi-GPU, distributed training |
| Maintenance | Unmaintained | Active development |

### Key Differences

1. **Environment Interface**
   - V1: `gym.Env` with `reset()` returning obs, `step()` returning `(obs, reward, done, info)`
   - V3: `embodied.Env` with single `step(action_dict)` returning obs dict with `is_first/is_last/is_terminal`

2. **Action Handling**
   - V1: Action is a numpy array, separate `reset()` method
   - V3: Action is a dict with `{'action': ..., 'reset': bool}`, reset triggered via `action['reset']=True`

3. **Observation Space**
   - V1: `gym.spaces.Box` or rlpyt `FloatBox`
   - V3: `dict[str, elements.Space]` with required keys `is_first`, `is_last`, `is_terminal`, `reward`

4. **Training Loop**
   - V1: rlpyt's `MinibatchRl` runner
   - V3: `embodied.run.train()` with Driver-based collection

---

## Completed Work

- DreamerV3 environment wrapper: [src/environment/embodied_env.py](src/environment/embodied_env.py) with `ClashRoyaleEmbodiedEnv`, `RemoteBridgeV3`, `RemoteStepV3`, `make_clash_royale_env()`; obs flattened to `(8640,)`, discrete 37-action space, reset via `action['reset']`.
- Shared utilities: [src/environment/action_utils.py](src/environment/action_utils.py) with `ActionMapper`, `DeployCell`, `DEFAULT_DEPLOY_CELLS`.
- Package exports: [src/environment/__init__.py](src/environment/__init__.py) now exports DreamerV3 classes and MaskedAgent helpers.
- Dependencies installed: `elements>=3.19.1`, `portal>=3.5.0`, `ninjax>=3.5.1`, `granular>=0.20.3` (in venv).
- gRPC processor updated: [src/cr/rpc/v1/processor.py](src/cr/rpc/v1/processor.py) now uses `RemoteBridgeV3` for DreamerV3 integration.
- DreamerV3 training script added: [train_dreamerv3.py](train_dreamerv3.py) (loads YAML configs, spins up gRPC server, runs `embodied.run.train` with ClashRoyale env).
- Clash Royale config preset added: [dreamerv3-main/dreamerv3/configs.yaml](../dreamerv3-main/dreamerv3/configs.yaml) (`clash_royale` preset, size12m backbone, vector MLP enc/dec, single-env run).
- Action masking implemented: observation includes `action_mask`; `MaskedAgent` wrapper re-samples illegal actions using the mask (0.0 legal, -1e9 illegal).
- Remote client verified: [scripts/remote_client_loop.py](scripts/remote_client_loop.py) works unchanged with the updated server (protobuf contract unchanged).

## Current Architecture

- Machine B (emulator) streams BGR frames over gRPC `FrameService.ProcessFrame` with `want_action` when training.
- Machine A (GPU): KataCR perception → state encoder → `RemoteBridgeV3` → `ClashRoyaleEmbodiedEnv` → DreamerV3 `Agent` wrapped by `MaskedAgent`.
- Observations to agent: flattened state `(8640,)`, `action_mask` additive logits, `reward`, `is_first/is_last/is_terminal`.
- Actions: discrete 37 (1 no-op + 4 cards × 9 cells); masking enforced in policy sampling via `MaskedAgent`.
- Training entrypoint: `train_dreamerv3.py` (starts gRPC server, builds env/agent/replay/logger, runs `embodied.run.train`).

## File Inventory

```
clash-royale-mbrl/
├── train_online.py              # Legacy DreamerV1 (kept for now)
├── train_dreamerv3.py           # DreamerV3 training entrypoint
├── src/
│   ├── specs.py                 # OBS_SPEC, ACTION_SPEC
│   ├── environment/
│   │   ├── __init__.py          # Exports V3 env + MaskedAgent helpers
│   │   ├── embodied_env.py      # DreamerV3 env, action_mask, RemoteBridgeV3
│   │   ├── action_utils.py      # Action mapping utilities
│   │   ├── action_mask.py       # Mask computation utilities
│   │   ├── remote_bridge.py     # Legacy V1 bridge
│   │   ├── online_env.py        # Legacy V1 env
│   │   └── emulator_env.py      # Local/emulator env
│   ├── cr/rpc/v1/
│   │   ├── processor.py         # Uses RemoteBridgeV3
│   │   └── server.py            # gRPC server scaffold
│   └── perception/              # KataCR integration
└── scripts/
    └── remote_client_loop.py    # Remote client (unchanged, works with V3 server)
```


## Dependencies Summary

### Required Packages (DreamerV3)

```
elements>=3.19.1
portal>=3.5.0
ninjax>=3.5.1
granular>=0.20.3
jax[cuda12] or jax[cpu]  # Platform dependent
numpy<2
```

### Packages to Remove (after migration)

```
rlpyt
torch (if not needed elsewhere)
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| [src/environment/embodied_env.py](src/environment/embodied_env.py) | DreamerV3 environment wrapper |
| [src/environment/action_utils.py](src/environment/action_utils.py) | Action mapping utilities |
| [src/specs.py](src/specs.py) | Observation and action space specs |
| [dreamerv3-main/dreamerv3/configs.yaml](../dreamerv3-main/dreamerv3/configs.yaml) | DreamerV3 configuration |
| [dreamerv3-main/dreamerv3/main.py](../dreamerv3-main/dreamerv3/main.py) | DreamerV3 entry point (reference) |
| [dreamerv3-main/embodied/core/base.py](../dreamerv3-main/embodied/core/base.py) | `embodied.Env` interface |
