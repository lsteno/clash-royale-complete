"""Dreamer integration entrypoint for the Clash Royale online pipeline."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
DREAMER_ROOT = REPO_ROOT / "dreamer-pytorch"
if str(DREAMER_ROOT) not in sys.path:
    sys.path.insert(0, str(DREAMER_ROOT))

# Ensure project src package is importable when running from source checkout.
SRC_ROOT = CURRENT_DIR / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import jax

# CRITICAL: Force JAX CUDA initialization BEFORE any torch imports!
# PyTorch and JAX conflict if torch initializes CUDA first - JAX's cuSOLVER check fails.
# Calling jax.devices() triggers the backend initialization.
_jax_devices = jax.devices()
print(f"[train_online] JAX initialized with devices: {_jax_devices}")

# Newer JAX drops define_bool_state; flax still calls it. Prefer the upstream helper
# from jax._src.config when missing, otherwise fall back to a minimal stub compatible
# with the current add_option signature (name, holder, opt_type, meta_args, meta_kwargs).
if not hasattr(jax.config, "define_bool_state"):
    try:
        from jax._src import config as _jax_config_mod  # type: ignore

        _define_bool_state = getattr(_jax_config_mod, "define_bool_state", None)
    except Exception:  # pragma: no cover - safety net for exotic installs
        _define_bool_state = None

    if _define_bool_state is None:
        class _FlagHolder:
            def __init__(self, value: bool):
                self.value = bool(value)

            def _set(self, value: bool) -> None:
                self.value = bool(value)

        def _define_bool_state(name: str, default: bool, help: str):
            holder = _FlagHolder(default)
            jax.config.add_option(name, holder, bool, [], {"help": help})
            # Mirror the property jax uses so config.<name> returns the holder value.
            setattr(type(jax.config), name, property(lambda self, _n=name: self._read(_n)))
            return holder

    jax.config.define_bool_state = _define_bool_state

import torch
import gym
import numpy as np
try:
    from rlpyt.runners.minibatch_rl import MinibatchRl
    from rlpyt.samplers.serial.sampler import SerialSampler
    from rlpyt.utils.logging.context import logger_context
    from rlpyt.utils.logging import logger
except ModuleNotFoundError as exc:  # pragma: no cover - guides users to install deps.
    raise ModuleNotFoundError(
        "rlpyt is required for online training. Install it via 'pip install rlpyt'."
    ) from exc

from dreamer.agents.atari_dreamer_agent import AtariDreamerAgent
from dreamer.algos.dreamer_algo import Dreamer
from dreamer.envs.one_hot import OneHotAction
from dreamer.envs.wrapper import make_wapper

from src.environment.emulator_env import EmulatorConfig
from src.environment.online_env import ClashRoyaleDreamerEnv, OnlineEnvConfig
from src.environment.remote_bridge import RemoteBridge, RemoteClashRoyaleEnv

# Some downstream deps mutate sys.path; force our in-repo packages to the front.
sys.path.insert(0, str(SRC_ROOT))

# Debug path to ensure cr/* is importable at runtime.
print("[train_online] sys.path head", sys.path[:5])

from cr.rpc.v1.processor import FrameServiceProcessor, ProcessorConfig
from cr.rpc.v1.server import RpcServerConfig, serve_forever
from src.specs import ACTION_SPEC, OBS_SPEC
from rlpyt.spaces.float_box import FloatBox
import asyncio
import threading


@dataclass
class TrainConfig:
    logdir: Path = Path("logs_online")
    total_steps: int = 200_000
    seed: int = 0
    device: str = "auto"  # "cpu", "auto", or CUDA index string
    wandb_project: Optional[str] = None
    wandb_run: Optional[str] = None
    num_envs: int = 1
    batch_T: int = 8
    log_interval: int = 1_000
    print_interval: int = 100
    ui_probe_save_frames: bool = False
    use_remote_frames: bool = False
    rpc_host: str = "0.0.0.0"
    rpc_port: int = 50051


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Online Dreamer training for Clash Royale")
    parser.add_argument("--logdir", type=Path, default=TrainConfig.logdir)
    parser.add_argument("--total-steps", type=int, default=TrainConfig.total_steps)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=TrainConfig.num_envs)
    parser.add_argument("--batch-T", type=int, default=TrainConfig.batch_T)
    parser.add_argument("--log-interval", type=int, default=TrainConfig.log_interval)
    parser.add_argument("--print-interval", type=int, default=TrainConfig.print_interval, help="Console log every N env steps")
    parser.add_argument("--ui-probe-save-frames", action="store_true", help="Save annotated UI probe frames for debugging")
    parser.add_argument("--use-remote-frames", action="store_true", help="Consume observations from remote FrameService (Machine B emulator)")
    parser.add_argument("--rpc-host", type=str, default=TrainConfig.rpc_host, help="FrameService listen host (Machine A)")
    parser.add_argument("--rpc-port", type=int, default=TrainConfig.rpc_port, help="FrameService listen port (Machine A)")
    args = parser.parse_args()
    return TrainConfig(
        logdir=args.logdir,
        total_steps=args.total_steps,
        seed=args.seed,
        device=args.device,
        wandb_project=args.wandb_project,
        wandb_run=args.wandb_run,
        num_envs=args.num_envs,
        batch_T=args.batch_T,
        log_interval=args.log_interval,
        print_interval=args.print_interval,
        ui_probe_save_frames=args.ui_probe_save_frames,
        use_remote_frames=args.use_remote_frames,
        rpc_host=args.rpc_host,
        rpc_port=args.rpc_port,
    )


def build_sampler(cfg: TrainConfig) -> SerialSampler:
    if cfg.use_remote_frames:
        # Remote frames are pushed by Machine B via gRPC. Training consumes them here.
        bridge = RemoteBridge()
        obs_space = FloatBox(low=0.0, high=1.0, shape=OBS_SPEC.shape, dtype=np.float32)
        act_space = gym.spaces.Discrete(ACTION_SPEC.size)

        def env_factory():
            # Create the remote env and wrap with OneHotAction for Dreamer compatibility.
            base_env = RemoteClashRoyaleEnv(bridge, obs_space, act_space)
            return OneHotAction(base_env)

        env_kwargs = {}
        sampler = SerialSampler(
            EnvCls=env_factory,
            env_kwargs=env_kwargs,
            eval_env_kwargs=env_kwargs,
            batch_T=cfg.batch_T,
            batch_B=cfg.num_envs,
            max_decorrelation_steps=0,
        )
        return sampler, bridge

    env_factory = make_wapper(ClashRoyaleDreamerEnv, [OneHotAction], [dict()])
    env_kwargs = dict(
        config=OnlineEnvConfig(
            emulator=EmulatorConfig(ui_probe_save_frames=cfg.ui_probe_save_frames)
        )
    )
    sampler = SerialSampler(
        EnvCls=env_factory,
        env_kwargs=env_kwargs,
        eval_env_kwargs=env_kwargs,
        batch_T=cfg.batch_T,
        batch_B=cfg.num_envs,
        max_decorrelation_steps=0,
    )
    return sampler, None


def _resolve_affinity(device: str) -> dict:
    dev = device.lower()
    if dev == "cpu":
        return dict(cuda_idx=None)
    if dev == "auto":
        return dict(cuda_idx=0 if torch.cuda.is_available() else None)
    return dict(cuda_idx=int(device))


def main() -> None:
    cfg = parse_args()
    cfg.logdir.mkdir(parents=True, exist_ok=True)
    sampler, bridge = build_sampler(cfg)

    server_thread = None
    if cfg.use_remote_frames:
        if bridge is None:
            raise RuntimeError("Remote frames requested but bridge missing")

        def _run_server():
            proc_cfg = ProcessorConfig()
            processor = FrameServiceProcessor(proc_cfg, bridge=bridge)
            server_cfg = RpcServerConfig(host=cfg.rpc_host, port=cfg.rpc_port)
            print(f"[train_online] gRPC FrameService listening on {cfg.rpc_host}:{cfg.rpc_port}")
            asyncio.run(serve_forever(processor, server_cfg))

        server_thread = threading.Thread(target=_run_server, daemon=True)
        server_thread.start()
        import time as _time
        _time.sleep(2)  # Give server thread time to initialize

    algo = Dreamer(horizon=10, kl_scale=0.1, use_pcont=True)
    agent = AtariDreamerAgent(
        train_noise=0.4,
        eval_noise=0.0,
        expl_type="epsilon_greedy",
        expl_min=0.1,
        expl_decay=2000 / 0.3,
        model_kwargs=dict(use_pcont=True),
    )
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=cfg.total_steps,
        log_interval_steps=min(cfg.log_interval, cfg.print_interval),
        affinity=_resolve_affinity(cfg.device),
    )
    config_dict = dict(
        obs_shape=OBS_SPEC.shape,
        action_size=ACTION_SPEC.size,
        total_steps=cfg.total_steps,
        num_envs=cfg.num_envs,
        batch_T=cfg.batch_T,
        seed=cfg.seed,
        wandb_project=cfg.wandb_project,
        wandb_run=cfg.wandb_run,
    )
    try:
        logger.set_snapshot_gap(2000)
    except AttributeError:
        pass
    with logger_context(
        str(cfg.logdir),
        cfg.seed,
        "clash_royale_dreamer",
        config_dict,
        use_summary_writer=True,
        snapshot_mode="last",
        override_prefix=True,
    ):
        runner.train()


if __name__ == "__main__":
    main()
