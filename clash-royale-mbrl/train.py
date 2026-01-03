"""
Main training script for Clash Royale Visual-MBRL.
Orchestrates the full pipeline: Environment -> Perception -> Agent.
"""
import argparse
import time
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from tqdm import tqdm

from src.utils.device import get_device, check_mps_capabilities
from src.environment.emulator_env import ClashRoyaleEmulatorEnv, EmulatorConfig
from src.perception.detection import PerceptionPipeline, DetectionConfig
from src.agent.dreamer_agent import DreamerV3Agent, TrainingConfig
from src.agent.dreamer_model import DreamerConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Clash Royale Visual-MBRL Training")
    
    # Environment
    parser.add_argument("--no-emulator", action="store_true",
                        help="Run without emulator (synthetic data)")
    parser.add_argument("--screen-width", type=int, default=1080)
    parser.add_argument("--screen-height", type=int, default=2400)
    
    # Perception
    parser.add_argument("--yolo-weights", type=str, default=None,
                        help="Path to custom YOLOv8 weights for CR")
    parser.add_argument("--detection-conf", type=float, default=0.25,
                        help="Detection confidence threshold")
    
    # Agent
    parser.add_argument("--stoch-size", type=int, default=32)
    parser.add_argument("--deter-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=512)
    
    # Training
    parser.add_argument("--total-steps", type=int, default=1_000_000,
                        help="Total environment steps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-length", type=int, default=50)
    parser.add_argument("--model-lr", type=float, default=1e-4)
    parser.add_argument("--actor-lr", type=float, default=3e-5)
    parser.add_argument("--critic-lr", type=float, default=3e-5)
    parser.add_argument("--train-every", type=int, default=16)
    parser.add_argument("--prefill", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=15)
    
    # Checkpointing
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--load-checkpoint", type=str, default=None)
    
    # Device
    parser.add_argument("--device", type=str, default="mps",
                        choices=["mps", "cuda", "cpu"])
    
    # Wandb
    parser.add_argument("--wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="clash-royale-mbrl")
    parser.add_argument("--run-name", type=str, default=None)
    
    return parser.parse_args()


class SyntheticEnv:
    """
    Synthetic environment for testing without emulator.
    Generates random observations and rewards.
    """
    
    def __init__(self, config: DreamerConfig):
        self.config = config
        self.step_count = 0
        self.episode_length = np.random.randint(100, 300)
    
    def reset(self):
        self.step_count = 0
        self.episode_length = np.random.randint(100, 300)
        return self._generate_obs()
    
    def _generate_obs(self):
        """Generate synthetic state grid."""
        obs = np.zeros((self.config.obs_channels, self.config.obs_height, self.config.obs_width), 
                       dtype=np.float32)
        
        # Add some random "units"
        for _ in range(np.random.randint(5, 15)):
            ch = np.random.randint(0, 4)  # Unit channels
            y = np.random.randint(0, self.config.obs_height)
            x = np.random.randint(0, self.config.obs_width)
            obs[ch, y, x] = np.random.uniform(0.5, 1.0)
        
        # Elixir
        obs[5, :, :] = np.random.uniform(0.3, 1.0)
        
        return obs
    
    def step(self, action):
        """
        Args:
            action: (card_index, x, y)
        """
        self.step_count += 1
        
        # Generate reward based on action
        card, x, y = action
        if card > 0:
            # Reward for playing cards
            reward = np.random.uniform(-0.1, 0.3)
            # Bonus for good positioning
            if 0.4 < y < 0.6:  # Mid field
                reward += 0.1
        else:
            reward = 0.0
        
        done = self.step_count >= self.episode_length
        
        if done:
            # End of episode reward
            reward += np.random.choice([-1.0, 0.0, 1.0], p=[0.3, 0.4, 0.3])
        
        obs = self._generate_obs()
        
        return obs, reward, done


def main():
    args = parse_args()
    
    # Print MPS capabilities
    print("\n" + "="*60)
    print("Clash Royale Visual-MBRL Training")
    print("="*60)
    
    caps = check_mps_capabilities()
    for k, v in caps.items():
        print(f"  {k}: {v}")
    
    device = get_device(args.device)
    print(f"\nUsing device: {device}")
    
    # Initialize wandb
    if args.wandb:
        try:
            import wandb
            run_name = args.run_name or f"cr-mbrl-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        except ImportError:
            print("Warning: wandb not installed, skipping logging")
            args.wandb = False
    
    # Create configs
    model_config = DreamerConfig(
        stoch_size=args.stoch_size,
        deter_size=args.deter_size,
        hidden_size=args.hidden_size,
        horizon=args.horizon,
    )
    
    train_config = TrainingConfig(
        batch_size=args.batch_size,
        batch_length=args.seq_length,
        model_lr=args.model_lr,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        train_every=args.train_every,
        prefill=args.prefill,
        horizon=args.horizon,
    )
    
    # Create agent
    agent = DreamerV3Agent(model_config, train_config, device)
    
    if args.load_checkpoint:
        agent.load(args.load_checkpoint)
    
    # Create environment
    if args.no_emulator:
        print("\nUsing synthetic environment (no emulator)")
        env = SyntheticEnv(model_config)
        perception = None
    else:
        print("\nConnecting to Android emulator...")
        emulator_config = EmulatorConfig(
            screen_width=args.screen_width,
            screen_height=args.screen_height,
        )
        env = ClashRoyaleEmulatorEnv(emulator_config)
        
        detection_config = DetectionConfig(
            model_path=args.yolo_weights,
            confidence_threshold=args.detection_conf,
            device=str(device),
        )
        perception = PerceptionPipeline(detection_config)
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\nStarting training...")
    print(f"  Total steps: {args.total_steps:,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Train every: {args.train_every} steps")
    print()
    
    agent.reset()
    obs = env.reset()
    
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    
    metrics_buffer = []
    start_time = time.time()
    
    pbar = tqdm(range(args.total_steps), desc="Training")
    for step in pbar:
        # Get action
        if perception is not None:
            frame = env.get_observation()
            state_grid, _ = perception.process(frame)
            obs = state_grid
        
        card, pos = agent.act(obs)
        
        # Step environment
        if perception is not None:
            action_tuple = (card, int(pos[0] * 31), int(pos[1] * 17))
            obs_frame = env.step(action_tuple)
            state_grid, _ = perception.process(obs_frame)
            next_obs = state_grid
            reward = 0.0  # Would need reward estimation
            done = False  # Would need episode detection
        else:
            next_obs, reward, done = env.step((card, pos[0], pos[1]))
        
        # Create action array for buffer
        action_arr = np.zeros(7, dtype=np.float32)
        action_arr[:5] = np.eye(5)[card]
        action_arr[5:7] = pos
        
        # Store transition
        agent.observe(obs, action_arr, reward, done)
        
        episode_reward += reward
        episode_length += 1
        
        obs = next_obs
        
        # Train
        if step % args.train_every == 0 and step > 0:
            metrics = agent.train_step()
            if metrics:
                metrics_buffer.append(metrics)
        
        # Episode end
        if done:
            episode_count += 1
            
            if args.wandb:
                wandb.log({
                    "episode_reward": episode_reward,
                    "episode_length": episode_length,
                    "episode": episode_count,
                    "step": step,
                })
            
            obs = env.reset()
            agent.reset()
            episode_reward = 0
            episode_length = 0
        
        # Logging
        if step % args.log_every == 0 and metrics_buffer:
            avg_metrics = {}
            for key in metrics_buffer[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in metrics_buffer])
            
            elapsed = time.time() - start_time
            sps = step / elapsed if elapsed > 0 else 0
            
            pbar.set_postfix({
                "loss": f"{avg_metrics.get('model_loss', 0):.3f}",
                "ep": episode_count,
                "sps": f"{sps:.1f}",
            })
            
            if args.wandb:
                wandb.log({**avg_metrics, "step": step, "steps_per_sec": sps})
            
            metrics_buffer = []
        
        # Checkpointing
        if step % args.save_every == 0 and step > 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{step}.pt"
            agent.save(str(checkpoint_path))
    
    # Final save
    final_path = checkpoint_dir / "checkpoint_final.pt"
    agent.save(str(final_path))
    
    print(f"\nTraining complete!")
    print(f"  Total steps: {args.total_steps:,}")
    print(f"  Total episodes: {episode_count}")
    print(f"  Time elapsed: {time.time() - start_time:.1f}s")
    
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
