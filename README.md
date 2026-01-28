# Clash Royale Complete

This repository contains the Clash Royale environments and multiple DreamerV3
implementations, with a focus on the **PyTorch DreamerV3** in
`clash-royale-complete/dreamerv3-torch`.

## Quick start (PyTorch DreamerV3)

### 1) Create a Python env

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r clash-royale-complete/dreamerv3-torch/requirements.txt
```

> Default config uses `device: mps` (Apple Silicon). For CUDA, update
> `dreamerv3-torch/configs.yaml` or pass overrides in your run command.

### 3) Ensure Clash environment is available
DreamerV3 Clash uses ADB devices (real or emulator). Verify:

```bash
adb devices
```

At least one device must be listed and `device_ids` should be online.

### 4) Train (flagship)

```bash
cd clash-royale-complete/dreamerv3-torch
python dreamer.py --configs clash_royale_12m --logdir ./logdir/clash_v3_flagship --envs 2
```

### 5) Evaluate

```bash
cd clash-royale-complete/dreamerv3-torch
python dreamer.py --configs clash_royale_12m --logdir ./logdir/clash_v3_flagship \
  --envs 2 --eval_only True --eval_episode_num 20 --parallel False
```

## Logs and monitoring

- Training logs: `clash-royale-complete/dreamerv3-torch/logdir/`
- TensorBoard:

  ```bash
  tensorboard --logdir clash-royale-complete/dreamerv3-torch/logdir
  ```

## Where outputs live

Artifacts generated during analysis (tables, plots, videos) are stored under:

```
clash-royale-complete/dreamerv3-torch/logdir/summary_outputs/
```

## Common issues

- **No ADB devices**: make sure emulators are running and `adb devices` shows
  them as `device` (not `offline`).
- **No video logging**: `moviepy` is required for video summaries; if missing,
  the logger falls back to image strips.
