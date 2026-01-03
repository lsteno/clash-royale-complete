"""
Online-only training entrypoint using dreamer-pytorch.

This is a scaffold that will be wired to the unified OnlineEnv and perception pipeline.
It locks the observation/action specs to:
- Observation: KataCR grid (15, 32, 18) float32
- Action: Discrete(37) = no-op + 4 cards * 9 cells
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.specs import ACTION_SPEC, OBS_SPEC


@dataclass
class TrainConfig:
    logdir: Path = Path("logs_online")
    total_steps: int = 1_000_000
    seed: int = 0
    device: str = "auto"
    wandb_project: Optional[str] = None
    wandb_run: Optional[str] = None
    # Parallel envs are optional; start with 1 for stability
    num_envs: int = 1


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Online Dreamer training for Clash Royale")
    parser.add_argument("--logdir", type=Path, default=TrainConfig.logdir)
    parser.add_argument("--total-steps", type=int, default=TrainConfig.total_steps)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=TrainConfig.num_envs)
    args = parser.parse_args()
    return TrainConfig(
        logdir=args.logdir,
        total_steps=args.total_steps,
        seed=args.seed,
        device=args.device,
        wandb_project=args.wandb_project,
        wandb_run=args.wandb_run,
        num_envs=args.num_envs,
    )


def main() -> None:
    cfg = parse_args()
    print("Specs locked:")
    print(f"  Observation shape: {OBS_SPEC.shape}")
    print(f"  Action size: {ACTION_SPEC.size} (no-op + 4 cards * 9 cells)")
    print("TODO: wire OnlineEnv, perception, reward/done, and dreamer-pytorch trainer.")
    print(f"Logdir: {cfg.logdir} | steps: {cfg.total_steps} | envs: {cfg.num_envs}")


if __name__ == "__main__":
    main()
