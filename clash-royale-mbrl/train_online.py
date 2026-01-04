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

import torch
try:
    from rlpyt.runners.minibatch_rl import MinibatchRl
    from rlpyt.samplers.serial.sampler import SerialSampler
    from rlpyt.utils.logging.context import logger_context
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
from src.specs import ACTION_SPEC, OBS_SPEC


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
    ui_probe_save_frames: bool = False


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
    parser.add_argument("--ui-probe-save-frames", action="store_true", help="Save annotated UI probe frames for debugging")
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
        ui_probe_save_frames=args.ui_probe_save_frames,
    )


def build_sampler(cfg: TrainConfig) -> SerialSampler:
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
    return sampler


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
    sampler = build_sampler(cfg)
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
        log_interval_steps=cfg.log_interval,
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
    with logger_context(
        str(cfg.logdir),
        cfg.seed,
        "clash_royale_dreamer",
        config_dict,
        use_summary_writer=True,
    ):
        runner.train()


if __name__ == "__main__":
    main()
