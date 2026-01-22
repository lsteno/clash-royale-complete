"""DreamerV3 training script for Clash Royale.

This script trains a DreamerV3 agent on Clash Royale using remote frames
streamed from Machine B via gRPC. It integrates:
1. The gRPC FrameService for receiving frames and sending actions
2. RemoteBridgeV3 for thread-safe communication between gRPC and training
3. ClashRoyaleEmbodiedEnv implementing embodied.Env interface
4. DreamerV3 agent with appropriate configuration for vector observations

Usage:
    python train_dreamerv3.py --logdir ./logs_dreamerv3 --rpc-port 50051
    
For debugging on CPU:
    python train_dreamerv3.py --configs debug --logdir ./logs_debug
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading
from dataclasses import dataclass
from functools import partial as bind
from pathlib import Path
from typing import Optional

# Setup paths before imports
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
DREAMERV3_ROOT = REPO_ROOT / "dreamerv3-main"
PROJECT_SRC = CURRENT_DIR / "src"

# Ensure dreamerv3 and embodied are importable
if str(DREAMERV3_ROOT) not in sys.path:
    sys.path.insert(0, str(DREAMERV3_ROOT))

# Ensure project root is importable (enables `import src.*`)
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Ensure `import cr.*` resolves to `clash-royale-mbrl/src/cr/*` when running this
# file directly (i.e. without `pip install -e`).
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

# Ensure XLA can fall back to lower-memory conv algorithms if autotuning OOMs.
# Only set if user hasn't specified XLA_FLAGS.
_xla_flag = "--xla_gpu_strict_conv_algorithm_picker=false"
if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = _xla_flag
elif _xla_flag not in os.environ["XLA_FLAGS"]:
    os.environ["XLA_FLAGS"] = f"{os.environ['XLA_FLAGS']} {_xla_flag}"

# Import JAX first to ensure proper CUDA initialization
import jax
_jax_devices = jax.devices()
print(f"[train_dreamerv3] JAX initialized with devices: {_jax_devices}")

import elements
import embodied
import numpy as np
import ruamel.yaml as yaml

from dreamerv3.agent import Agent
from src.environment.embodied_env import (
    ClashRoyaleEmbodiedEnv,
    RemoteBridgeV3,
    MaskedAgent,
)
from src.specs import ACTION_SPEC, OBS_SPEC


# ============================================================================
# Configuration
# ============================================================================

# Clash Royale specific defaults that override DreamerV3 defaults
CLASH_ROYALE_CONFIG = {
    # Task identifier
    'task': 'clash_royale',

    # Optional pixel-mode overrides (defaults for CLI toggles)
    'pixels': False,
    'pixel_height': 192,
    'pixel_width': 256,
    
    # Use smaller batch for single-env training
    'batch_size': 8,
    'batch_length': 32,
    'report_length': 16,
    
    # Replay buffer sized for our observation size (8640 floats per obs)
    'replay': {
        'size': 500_000,  # ~500K transitions, ~17GB at full capacity
        'online': True,
        'chunksize': 512,
    },
    
    # Training parameters for single-environment setup
    'run': {
        'steps': 1_000_000,
        'envs': 1,  # Single environment (remote gRPC frames)
        'train_ratio': 64.0,  # Train more aggressively with limited data
        'log_every': 60,
        'report_every': 300,
        'save_every': 600,
        'episode_timeout': 600,  # 10 min timeout for long matches
        'debug': True,  # Enable detailed logging
    },
    
    # JAX settings - will be overridden by CLI for CPU debugging
    'jax': {
        'platform': 'cuda',
        'prealloc': True,
    },
    
    # Agent architecture for vector observations (not images)
    'agent': {
        # Encoder for our flattened (8640,) vector input
        'enc': {
            'typ': 'simple',
            'simple': {
                'layers': 3,
                'units': 512,
                'symlog': True,  # Symlog normalization helps with varied scales
            },
        },
        # Decoder matching encoder
        'dec': {
            'typ': 'simple', 
            'simple': {
                'layers': 3,
                'units': 512,
            },
        },
        # RSSM sized ~50M parameter preset
        'dyn': {
            'typ': 'rssm',
            'rssm': {
                'deter': 4096,
                'hidden': 512,
                'stoch': 32,
                'classes': 32,
            },
        },
        # Imagination horizon
        'imag_length': 15,
        'horizon': 333,
    },
}


@dataclass
class TrainConfig:
    """Training configuration with CLI-friendly defaults."""
    logdir: Path = Path("logs_dreamerv3")
    rpc_host: str = "0.0.0.0"
    rpc_port: int = 50051
    configs: tuple = ("defaults",)  # Config presets to load
    save_perception_crops: bool = False
    seed: int = 0
    pixels: bool = False  # If True, train directly from emulator RGB frames (no encoder)
    pixel_height: int = 192
    pixel_width: int = 256
    perception_stride: int = 1
    return_state_grid: bool = False
    detector_count: int = 2
    disable_center_ocr: bool = False
    disable_card_classifier: bool = False
    debug_dump_dir: Optional[Path] = None
    debug_dump_every: int = 0
    debug_dump_max: int = 200
    debug_dump_annotated: bool = False
    
    # Override specific settings
    steps: Optional[int] = None
    batch_size: Optional[int] = None
    train_ratio: Optional[float] = None


def parse_args() -> tuple[TrainConfig, list[str]]:
    """Parse command line arguments, returning config and remaining args."""
    parser = argparse.ArgumentParser(
        description="DreamerV3 training for Clash Royale",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--logdir", type=Path, default=TrainConfig.logdir)
    parser.add_argument("--rpc-host", type=str, default=TrainConfig.rpc_host)
    parser.add_argument("--rpc-port", type=int, default=TrainConfig.rpc_port)
    parser.add_argument("--configs", nargs="*", default=["defaults"],
                        help="Config presets to load (e.g., 'defaults', 'debug', 'size12m')")
    parser.add_argument("--save-perception-crops", action="store_true")
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--pixels", action="store_true",
                        help="Train directly from raw emulator RGB frames (channels-last)")
    parser.add_argument("--pixel-height", type=int, default=TrainConfig.pixel_height,
                        help="Target pixel observation height (when --pixels)")
    parser.add_argument("--pixel-width", type=int, default=TrainConfig.pixel_width,
                        help="Target pixel observation width (when --pixels)")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--perception-stride", type=int, default=TrainConfig.perception_stride,
                        help="Run full perception every N stepped frames (throughput knob)")
    parser.add_argument("--return-state-grid", action="store_true",
                        help="Include full observation tensors in RPC responses (debug; slower)")
    parser.add_argument("--detector-count", type=int, default=TrainConfig.detector_count,
                        help="Use only the first N YOLO detectors (1 = faster, 2 = default)")
    parser.add_argument("--disable-center-ocr", action="store_true",
                        help="Skip center-screen OCR (faster); rely on OK-button probe for match end")
    parser.add_argument("--disable-card-classifier", action="store_true",
                        help="Skip card classifier (faster; uses fallback cards)")
    parser.add_argument("--debug-dump-dir", type=Path, default=None,
                        help="Dump perception + obs tensors to this directory (debug)")
    parser.add_argument("--debug-dump-every", type=int, default=0,
                        help="Dump every N perception frames (0 disables; default 0)")
    parser.add_argument("--debug-dump-max", type=int, default=200,
                        help="Maximum number of debug dumps to write")
    parser.add_argument("--debug-dump-annotated", action="store_true",
                        help="Also dump YOLO overlay images (slower)")
    
    args, remaining = parser.parse_known_args()
    
    cfg = TrainConfig(
        logdir=args.logdir,
        rpc_host=args.rpc_host,
        rpc_port=args.rpc_port,
        configs=tuple(args.configs),
        save_perception_crops=args.save_perception_crops,
        seed=args.seed,
        pixels=args.pixels,
        pixel_height=args.pixel_height,
        pixel_width=args.pixel_width,
        perception_stride=args.perception_stride,
        return_state_grid=args.return_state_grid,
        detector_count=args.detector_count,
        disable_center_ocr=args.disable_center_ocr,
        disable_card_classifier=args.disable_card_classifier,
        debug_dump_dir=args.debug_dump_dir,
        debug_dump_every=args.debug_dump_every,
        debug_dump_max=args.debug_dump_max,
        debug_dump_annotated=args.debug_dump_annotated,
        steps=args.steps,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
    )
    return cfg, remaining


def load_config(train_cfg: TrainConfig, remaining_args: list[str]) -> elements.Config:
    """Load and merge DreamerV3 configuration."""
    # Load base configs from DreamerV3
    configs_path = DREAMERV3_ROOT / "dreamerv3" / "configs.yaml"
    configs = yaml.YAML(typ='safe').load(configs_path.read_text())
    
    # Start with DreamerV3 defaults.
    config = elements.Config(configs['defaults'])

    # Apply Clash Royale specific defaults next (so user presets can override them).
    config = config.update(CLASH_ROYALE_CONFIG)

    # Apply requested config presets (excluding defaults, already loaded).
    for name in train_cfg.configs:
        if name == 'defaults':
            continue
        if name in configs:
            config = config.update(configs[name])
        else:
            print(f"[train_dreamerv3] Warning: Config preset '{name}' not found, skipping")
    
    # Apply CLI overrides
    cli_overrides = {}
    if train_cfg.steps is not None:
        cli_overrides['run.steps'] = train_cfg.steps
    if train_cfg.batch_size is not None:
        cli_overrides['batch_size'] = train_cfg.batch_size
    if train_cfg.train_ratio is not None:
        cli_overrides['run.train_ratio'] = train_cfg.train_ratio

    # Pixel mode toggles RGB observations instead of encoded grids
    cli_overrides['pixels'] = train_cfg.pixels
    cli_overrides['pixel_height'] = train_cfg.pixel_height
    cli_overrides['pixel_width'] = train_cfg.pixel_width
    
    cli_overrides['logdir'] = str(train_cfg.logdir)
    cli_overrides['seed'] = train_cfg.seed
    
    config = config.update(cli_overrides)
    
    # Parse any remaining command-line flags as config overrides
    if remaining_args:
        config = elements.Flags(config).parse(remaining_args)

    # If pixel observations are requested, switch encoder/decoder to CNN mode
    if train_cfg.pixels:
        config = config.update({
            'agent.enc.typ': 'cnn',
            'agent.dec.typ': 'cnn',
        })

    # Add timestamp to logdir if not already present
    config = config.update(logdir=(
        config.logdir.format(timestamp=elements.timestamp())
        if '{timestamp}' in config.logdir else config.logdir
    ))
    
    return config


# ============================================================================
# Factory Functions (following DreamerV3 patterns)
# ============================================================================

def make_agent(config: elements.Config, bridge: RemoteBridgeV3):
    """Create the DreamerV3 agent."""
    # Create a temporary env to get obs/act spaces
    obs_shape_override = None
    flatten_obs = True
    obs_dtype = np.float32
    if getattr(config, 'pixels', False):
        # Use channels-last uint8 images for CNN encoder
        height = int(getattr(config, 'pixel_height', 180))
        width = int(getattr(config, 'pixel_width', 320))
        obs_shape_override = (height, width, 3)
        flatten_obs = False
        obs_dtype = np.uint8

    env = ClashRoyaleEmbodiedEnv(
        bridge,
        flatten_obs=flatten_obs,
        obs_shape_override=obs_shape_override,
        obs_dtype=obs_dtype,
    )
    
    # Filter spaces for agent
    notlog = lambda k: not k.startswith('log/')
    obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
    act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
    
    # Check if this is a random agent run
    if config.random_agent:
        return embodied.RandomAgent(obs_space, act_space)
    
    # Create the DreamerV3 agent
    agent = Agent(obs_space, act_space, elements.Config(
        **config.agent,
        logdir=config.logdir,
        seed=config.seed,
        jax=config.jax,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=config.replica,
        replicas=config.replicas,
    ))

    # Wrap with MaskedAgent for action masking
    agent = MaskedAgent(agent, action_key='action')

    # Print model size summary for reproducibility and debugging.
    try:
        params = getattr(agent, "params", None)
        if params is not None:
            total = 0
            for v in params.values():
                shape = getattr(v, "shape", None)
                if not shape:
                    continue
                total += int(np.prod(shape))
            million = total / 1e6
            deter = int(getattr(config.agent.dyn.rssm, "deter", -1))
            hidden = int(getattr(config.agent.dyn.rssm, "hidden", -1))
            classes = int(getattr(config.agent.dyn.rssm, "classes", -1))
            enc_typ = str(getattr(config.agent.enc, "typ", "unknown"))
            dec_typ = str(getattr(config.agent.dec, "typ", "unknown"))
            obs_mode = "pixels" if bool(getattr(config, "pixels", False)) else "semantic_grid"
            print(
                f"[train_dreamerv3] Model params: {million:.1f}M "
                f"(rssm deter={deter} hidden={hidden} classes={classes}, enc={enc_typ}, dec={dec_typ}, obs={obs_mode})"
            )
            try:
                import json

                out = elements.Path(config.logdir) / "model_summary.json"
                out.write_text(json.dumps({
                    "param_count": int(total),
                    "param_count_millions": float(million),
                    "rssm": {"deter": deter, "hidden": hidden, "classes": classes},
                    "encoder": enc_typ,
                    "decoder": dec_typ,
                    "obs_mode": obs_mode,
                }, indent=2))
            except Exception:
                pass
    except Exception as exc:
        print(f"[train_dreamerv3] Warning: failed to compute model size: {exc}")
    
    return agent


def make_env(config: elements.Config, bridge: RemoteBridgeV3, index: int = 0):
    """Create the Clash Royale environment."""
    obs_shape_override = None
    flatten_obs = True  # Default: flattened state grid
    obs_dtype = np.float32

    if getattr(config, 'pixels', False):
        # Use channels-last uint8 images for CNN encoder
        height = int(getattr(config, 'pixel_height', 180))
        width = int(getattr(config, 'pixel_width', 320))
        obs_shape_override = (height, width, 3)
        flatten_obs = False
        obs_dtype = np.uint8

    env = ClashRoyaleEmbodiedEnv(
        bridge=bridge,
        step_timeout=config.run.episode_timeout,
        flatten_obs=flatten_obs,
        obs_shape_override=obs_shape_override,
        obs_dtype=obs_dtype,
    )
    # Apply standard wrappers
    env = wrap_env(env, config)
    return env


def wrap_env(env: embodied.Env, config: elements.Config) -> embodied.Env:
    """Apply standard DreamerV3 wrappers to the environment."""
    # Normalize continuous actions (Clash Royale uses discrete, so this is a no-op)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.NormalizeAction(env, name)
    
    # Unify dtypes across observation space
    env = embodied.wrappers.UnifyDtypes(env)
    
    # Validate spaces
    env = embodied.wrappers.CheckSpaces(env)
    
    # Clip continuous actions
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    
    return env


def make_replay(config: elements.Config, folder: str = 'replay', mode: str = 'train'):
    """Create the replay buffer."""
    batlen = config.batch_length if mode == 'train' else config.report_length
    consec = config.consec_train if mode == 'train' else config.consec_report
    capacity = config.replay.size if mode == 'train' else config.replay.size / 10
    length = consec * batlen + config.replay_context
    
    directory = elements.Path(config.logdir) / folder
    if config.replicas > 1:
        directory /= f'{config.replica:05}'
    
    kwargs = dict(
        length=length,
        capacity=int(capacity),
        online=config.replay.online,
        chunksize=config.replay.chunksize,
        directory=directory,
    )
    
    return embodied.replay.Replay(**kwargs)


def make_stream(config: elements.Config, replay, mode: str):
    """Create a data stream from the replay buffer."""
    fn = bind(replay.sample, config.batch_size, mode)
    stream = embodied.streams.Stateless(fn)
    stream = embodied.streams.Consec(
        stream,
        length=config.batch_length if mode == 'train' else config.report_length,
        consec=config.consec_train if mode == 'train' else config.consec_report,
        prefix=config.replay_context,
        strict=(mode == 'train'),
        contiguous=True,
    )
    return stream


def make_logger(config: elements.Config):
    """Create the logger."""
    step = elements.Counter()
    logdir = config.logdir
    
    outputs = []
    outputs.append(elements.logger.TerminalOutput(config.logger.filter, 'Agent'))
    
    for output in config.logger.outputs:
        if output == 'jsonl':
            outputs.append(elements.logger.JSONLOutput(logdir, 'metrics.jsonl'))
            outputs.append(elements.logger.JSONLOutput(
                logdir, 'scores.jsonl', 'episode/score'))
        elif output == 'tensorboard':
            outputs.append(elements.logger.TensorBoardOutput(
                logdir, config.logger.fps))
        elif output == 'scope':
            outputs.append(elements.logger.ScopeOutput(elements.Path(logdir)))
        # Skip unsupported outputs for now (wandb, expa)
    
    logger = elements.Logger(step, outputs, multiplier=1)
    return logger


# ============================================================================
# gRPC Server Management
# ============================================================================

def start_grpc_server(
    bridge: RemoteBridgeV3,
    train_cfg: TrainConfig,
    config: elements.Config,
) -> threading.Thread:
    """Start the gRPC FrameService in a background thread."""
    from cr.rpc.v1.processor import FrameServiceProcessor, ProcessorConfig
    from cr.rpc.v1.server import RpcServerConfig, serve_forever
    
    def _run_server():
        proc_cfg = ProcessorConfig(
            perception_stride=max(1, int(train_cfg.perception_stride)),
            return_state_grid=bool(train_cfg.return_state_grid),
            debug_dump_dir=str(train_cfg.debug_dump_dir) if train_cfg.debug_dump_dir else None,
            debug_dump_every=int(train_cfg.debug_dump_every),
            debug_dump_max=int(train_cfg.debug_dump_max),
            debug_dump_annotated=bool(train_cfg.debug_dump_annotated),
        )
        # Always configure perception - needed for reward calculation even in pixel mode
        from src.perception.katacr_pipeline import KataCRVisionConfig
        # Default to CPU OCR for portability. Paddle GPU wheels are often the
        # most fragile dependency in containerized setups.
        # Override by setting `CR_OCR_GPU=1` in the environment.
        ocr_gpu = bool(int(os.environ.get("CR_OCR_GPU", "0")))

        vision_cfg = KataCRVisionConfig(
            debug_save_parts=train_cfg.save_perception_crops,
            debug_parts_dir=Path(config.logdir) / "perception_crops",
            ocr_gpu=ocr_gpu,
            detector_count=int(train_cfg.detector_count),
            enable_center_ocr=not bool(train_cfg.disable_center_ocr),
            enable_card_classifier=not bool(train_cfg.disable_card_classifier),
        )
        processor = FrameServiceProcessor(
            proc_cfg,
            vision_cfg=vision_cfg,
            bridge=bridge,
            use_pixels=bool(config.pixels),
            pixel_height=int(getattr(config, 'pixel_height', 180)),
            pixel_width=int(getattr(config, 'pixel_width', 320)),
        )
        server_cfg = RpcServerConfig(
            host=train_cfg.rpc_host,
            port=train_cfg.rpc_port,
        )
        print(f"[train_dreamerv3] gRPC FrameService listening on "
              f"{train_cfg.rpc_host}:{train_cfg.rpc_port}")
        asyncio.run(serve_forever(processor, server_cfg))
    
    thread = threading.Thread(target=_run_server, daemon=True, name="grpc-server")
    thread.start()
    
    # Give server time to initialize
    import time
    time.sleep(2)
    
    return thread


# ============================================================================
# Main Training Loop
# ============================================================================

def main() -> None:
    """Main entry point for DreamerV3 training."""
    train_cfg, remaining_args = parse_args()
    config = load_config(train_cfg, remaining_args)
    
    # Setup logdir
    logdir = elements.Path(config.logdir)
    print(f"[train_dreamerv3] Logdir: {logdir}")
    logdir.mkdir()
    config.save(logdir / 'config.yaml')
    
    # Print config summary
    print(f"[train_dreamerv3] Task: {config.task}")
    print(f"[train_dreamerv3] Steps: {config.run.steps}")
    print(f"[train_dreamerv3] Batch: {config.batch_size}x{config.batch_length}")
    print(f"[train_dreamerv3] Train ratio: {config.run.train_ratio}")
    print(f"[train_dreamerv3] JAX platform: {config.jax.platform}")
    
    # Initialize timer
    elements.timer.global_timer.enabled = config.logger.timer
    
    # Create the remote bridge for gRPC <-> training communication
    bridge = RemoteBridgeV3(action_timeout=30.0)
    
    # Start gRPC server
    _server_thread = start_grpc_server(bridge, train_cfg, config)

    # Do not block on connection here: initialize training immediately.
    # The training loop will start once the first frame arrives (env reset blocks
    # until a frame is received), which keeps setup work off the critical path.
    print("[train_dreamerv3] Starting setup. Training will begin after first frame.")

    def _log_connection() -> None:
        print("[train_dreamerv3] Waiting for remote client to send frames...")
        while not bridge.wait_for_connection(timeout=10.0):
            print("[train_dreamerv3] Still waiting for remote client...")
        print("[train_dreamerv3] Remote client detected.")

    threading.Thread(target=_log_connection, name="wait-for-client", daemon=True).start()
    
    # Create training components using DreamerV3's patterns
    args = elements.Config(
        **config.run,
        replica=config.replica,
        replicas=config.replicas,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        report_length=config.report_length,
        consec_train=config.consec_train,
        consec_report=config.consec_report,
        replay_context=config.replay_context,
    )
    
    # Run training using embodied.run.train
    embodied.run.train(
        bind(make_agent, config, bridge),
        bind(make_replay, config, 'replay'),
        bind(make_env, config, bridge),
        bind(make_stream, config),
        bind(make_logger, config),
        args,
    )
    
    print("[train_dreamerv3] Training complete")


if __name__ == "__main__":
    main()
