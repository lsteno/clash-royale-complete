# Machine A Docker (GPU training + perception)

This container is meant for the **GPU server** (Machine A): it runs the gRPC `FrameService` and DreamerV3 training (`clash-royale-mbrl/train_dreamerv3.py`).

Machine B (Android emulator + `remote_client_loop.py`) is usually **not** containerized on macOS.

## Build

From repo root:

```bash
docker build -f docker/machineA/Dockerfile -t clash-royale-machinea:latest .
```

## Run (Docker)

```bash
docker run --rm -it --gpus all \
  -p 50051:50051 \
  -v "$PWD/logs_dreamerv3:/app/logs_dreamerv3" \
  -v "$PWD/KataCR:/app/KataCR" \
  clash-royale-machinea:latest
```

## Run (Compose)

```bash
docker compose -f docker/machineA/docker-compose.yml up --build
```

## Run on Azure VM (docker-compose)

This repo includes a VM compose file that:
- binds gRPC to `127.0.0.1:50051` (use an SSH tunnel from your laptop)
- writes logs and debug dumps to `/mnt/azureuser`
- starts the trainer with debug-dump flags enabled

```bash
sudo mkdir -p /mnt/azureuser/logs_dreamerv3 /mnt/azureuser/dumps
sudo chown -R $USER:$USER /mnt/azureuser/logs_dreamerv3 /mnt/azureuser/dumps

sudo docker-compose -f docker/machineA/docker-compose.vm.yml up --build
```

### Override defaults (port, logdir, model size)

The VM compose supports env var overrides so you can run multiple configs without editing YAML:

```bash
export HOST_RPC_PORT=50052
export HOST_LOGDIR=/mnt/azureuser/logs_dreamerv3_100m
export HOST_DUMPDIR=/mnt/azureuser/dumps_100m
export CR_SIZE=size100m
sudo mkdir -p "$HOST_LOGDIR" "$HOST_DUMPDIR"
sudo chown -R $USER:$USER "$HOST_LOGDIR" "$HOST_DUMPDIR"

sudo docker-compose -f docker/machineA/docker-compose.vm.yml up --build -d
```

## Notes

- KataCR weights are **not** baked into the image; mount `KataCR/` after you download weights.
- If you hit CUDA/JAX issues, try `--configs debug` or switch platform with `--jax.platform cpu` for quick sanity checks.
