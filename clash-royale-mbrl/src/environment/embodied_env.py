"""DreamerV3-compatible environment wrapper for Clash Royale remote frames.

This module provides an environment wrapper that implements the `embodied.Env`
interface required by DreamerV3. It consumes frames pushed by the remote gRPC
FrameService and exposes them through the dictionary-based observation/action
interface that DreamerV3 expects.

Key differences from the rlpyt-based wrapper:
1. obs_space/act_space are dictionaries of elements.Space, not gym.spaces
2. step() takes a dict with 'reset' key and returns a dict with is_first/is_last/is_terminal
3. No separate reset() method - reset is triggered via action['reset']
4. Observations include reward, is_first, is_last, is_terminal in the returned dict
"""
from __future__ import annotations

import functools
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import elements
import numpy as np

# Local imports - we import embodied from the dreamerv3-main package
import sys
from pathlib import Path

# Ensure dreamerv3 is importable
REPO_ROOT = Path(__file__).resolve().parents[3]
DREAMERV3_ROOT = REPO_ROOT / "dreamerv3-main"
if str(DREAMERV3_ROOT) not in sys.path:
    sys.path.insert(0, str(DREAMERV3_ROOT))

import embodied

from src.specs import ACTION_SPEC, OBS_SPEC
from src.environment.action_mask import compute_action_mask, set_action_mask
from src.environment.action_utils import ActionMapper, DEFAULT_DEPLOY_CELLS


@dataclass
class RemoteStepV3:
    """Represents a single step received from the remote gRPC service.
    
    This is the communication unit between the gRPC processor thread and
    the environment's step() method. The processor creates this object,
    puts it in the queue, and waits for the action to be set by the trainer.
    """
    obs: np.ndarray  # (C, H, W) state tensor
    reward: float
    done: bool
    info: Dict[str, Any]
    _action_ready: threading.Event = field(default_factory=threading.Event)
    _action: Optional[Tuple[int, int, int]] = None

    def set_action(self, action: Optional[Tuple[int, int, int]]) -> None:
        """Called by the trainer to provide the action for this step."""
        self._action = action
        self._action_ready.set()

    def wait_action(self, timeout: Optional[float] = None) -> Optional[Tuple[int, int, int]]:
        """Called by the gRPC processor to wait for the trainer's action."""
        self._action_ready.wait(timeout=timeout)
        return self._action


class RemoteBridgeV3:
    """Thread-safe conduit between the gRPC processor and the DreamerV3 environment.
    
    This bridge connects two asynchronous processes:
    1. The gRPC server thread that receives frames from Machine B
    2. The DreamerV3 driver that calls env.step() to get observations
    
    The flow is:
    1. gRPC processor calls publish() with a new observation
    2. publish() puts the step in a queue and blocks waiting for action
    3. DreamerV3 driver calls env.step() which consumes from the queue
    4. env.step() processes the action and calls step.set_action()
    5. publish() returns with the action to send back to Machine B
    """

    def __init__(self, action_timeout: float = 30.0):
        """Initialize the bridge.
        
        Args:
            action_timeout: Maximum seconds to wait for the trainer to provide
                an action before returning a no-op. Increased from 5s to 30s
                to accommodate DreamerV3's potentially longer training steps.
        """
        self._steps: "queue.Queue[RemoteStepV3]" = queue.Queue()
        self._action_timeout = action_timeout
        self._ready = threading.Event()
        self._connected = threading.Event()
        self._episode_count = 0
        self._step_count = 0

    def set_ready(self) -> None:
        """Signal that the trainer is ready to consume frames."""
        self._ready.set()

    def is_ready(self) -> bool:
        """Check if the trainer is ready."""
        return self._ready.is_set()

    def wait_for_connection(self, timeout: Optional[float] = None) -> bool:
        """Block until the remote client sends at least one frame."""
        return self._connected.wait(timeout=timeout)

    def publish(
        self,
        obs: np.ndarray,
        reward: float,
        done: bool,
        info: Dict[str, Any]
    ) -> Optional[Tuple[int, int, int]]:
        """Publish a new observation and wait for the corresponding action.
        
        Called by the gRPC processor when a new frame arrives from Machine B.
        
        Args:
            obs: State tensor. Shape depends on mode:
                 - State grid mode: (C, H, W) float32 channels-first
                 - Pixel mode: (H, W, 3) uint8 channels-last RGB
            reward: Reward signal from the game
            done: Whether the episode has ended
            info: Additional info dict (cards, elixir, time, etc.)
            
        Returns:
            The action tuple (card_slot, grid_x, grid_y) or None for no-op.
        """
        self._connected.set()
        # If trainer isn't ready yet, return no-op immediately
        if not self._ready.is_set():
            return None
            
        step = RemoteStepV3(
            obs=obs,
            reward=reward,
            done=done,
            info=info,
            _action_ready=threading.Event()
        )
        self._steps.put(step)
        self._step_count += 1
        
        if done:
            self._episode_count += 1
        
        # Wait for trainer to provide action (with timeout to prevent deadlock)
        action = step.wait_action(timeout=self._action_timeout)
        return action

    def next_step(self, timeout: Optional[float] = None) -> RemoteStepV3:
        """Get the next step from the queue.
        
        Called by the environment's step() method.
        
        Args:
            timeout: Maximum seconds to wait for a step.
            
        Returns:
            The next RemoteStepV3 from the queue.
            
        Raises:
            queue.Empty: If no step is available within the timeout.
        """
        return self._steps.get(timeout=timeout)

    @property
    def stats(self) -> Dict[str, int]:
        """Return statistics about the bridge."""
        return {
            "episode_count": self._episode_count,
            "step_count": self._step_count,
            "queue_size": self._steps.qsize(),
        }


class ClashRoyaleEmbodiedEnv(embodied.Env):
    """DreamerV3-compatible environment for Clash Royale remote frames.
    
    This environment implements the embodied.Env interface required by DreamerV3.
    It consumes observations pushed by the remote gRPC FrameService and exposes
    them through DreamerV3's dictionary-based interface.
    
    Key characteristics:
    - Observations are flattened state vectors (not images) for MLP encoding
    - Actions are discrete (37 choices: 1 no-op + 4 cards × 9 deploy cells)
    - Reset is triggered via action['reset'] = True, not a separate method
    - Action masking info is stored in context variables for policy access
    
    Observation space:
    - 'state': Flattened state tensor of shape (C*H*W,) = (8640,)
    - 'reward': Scalar float32
    - 'is_first': Boolean (True on episode start)
    - 'is_last': Boolean (True on episode end)
    - 'is_terminal': Boolean (True if episode ended terminally)
    
    Action space:
    - 'action': Discrete int32 in [0, 37)
    - 'reset': Boolean
    """

    def __init__(
        self,
        bridge: RemoteBridgeV3,
        step_timeout: float = 60.0,
        flatten_obs: bool = True,
        obs_shape_override: Optional[tuple[int, ...]] = None,
        obs_dtype: np.dtype = np.float32,
    ):
        """Initialize the environment.
        
        Args:
            bridge: The RemoteBridgeV3 instance connecting to the gRPC processor.
            step_timeout: Maximum seconds to wait for a new frame in step().
            flatten_obs: If True, flatten the (C,H,W) observation to (C*H*W,).
                         If False, keep as (C,H,W) for potential CNN encoding.
        """
        self._bridge = bridge
        self._step_timeout = step_timeout
        self._flatten_obs = flatten_obs
        self._mapper = ActionMapper(DEFAULT_DEPLOY_CELLS)
        
        # Observation dtype
        self._obs_dtype = np.dtype(obs_dtype)

        # Episode state
        self._current_step: Optional[RemoteStepV3] = None
        self._episode_return = 0.0
        self._episode_length = 0
        self._is_first = True  # Will be True on first step after reset
        
        # Compute observation shape (allow override for pixel training)
        if obs_shape_override is not None:
            self._obs_shape = obs_shape_override
        elif flatten_obs:
            self._obs_shape = (OBS_SPEC.channels * OBS_SPEC.height * OBS_SPEC.width,)
        else:
            self._obs_shape = OBS_SPEC.shape

    @functools.cached_property
    def obs_space(self) -> Dict[str, elements.Space]:
        """Return the observation space dictionary.
        
        DreamerV3 requires specific keys: is_first, is_last, is_terminal, reward.
        We add 'state' for our observation tensor and 'action_mask' for masking.
        """
        spaces = {
            # Main observation - flattened state vector or spatial tensor
            'state': elements.Space(self._obs_dtype, self._obs_shape),
            # Action mask: additive logits (0.0 = legal, -1e9 = illegal)
            'action_mask': elements.Space(np.float32, (ACTION_SPEC.size,)),
            # Required by DreamerV3
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
            # Reward breakdown logs
            'log/reward_tower': elements.Space(np.float32),
            'log/reward_king_tower': elements.Space(np.float32),
            'log/reward_tower_misc': elements.Space(np.float32),
            'log/reward_tower_damage': elements.Space(np.float32),
            'log/reward_elixir_waste': elements.Space(np.float32),
            'log/reward_enemy_eliminated': elements.Space(np.float32),
        }
        return spaces

    @functools.cached_property  
    def act_space(self) -> Dict[str, elements.Space]:
        """Return the action space dictionary.
        
        DreamerV3 requires a 'reset' key. We add 'action' for our discrete action.
        """
        return {
            'action': elements.Space(np.int32, (), 0, ACTION_SPEC.size),
            'reset': elements.Space(bool),
        }

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one environment step.
        
        Unlike gym-style envs, DreamerV3 environments handle reset within step()
        by checking action['reset']. This method:
        1. Checks if reset is requested
        2. Sends the previous action to the gRPC processor (unblocking it)
        3. Waits for the next frame from the remote source
        4. Returns the observation dictionary
        
        Args:
            action: Dictionary with 'action' (int) and 'reset' (bool) keys.
            
        Returns:
            Observation dictionary with state, reward, is_first, is_last, is_terminal.
        """
        import sys
        should_reset = action.get('reset', False) or self._is_first
        
        if should_reset:
            return self._handle_reset()
        
        return self._handle_step(action)

    def _handle_reset(self) -> Dict[str, Any]:
        """Handle episode reset/start.
        
        On reset:
        1. Signal that we're ready to receive frames (if not already)
        2. If there's a pending step, send no-op action to unblock the gRPC handler
        3. Wait for the first frame of the new episode
        4. Return observation with is_first=True
        """
        # Signal readiness to the bridge
        self._bridge.set_ready()
        
        # If there's a pending step from a previous episode, release it with no-op
        if self._current_step is not None:
            self._current_step.set_action(None)
            self._current_step = None
        
        # Wait for the first frame
        try:
            self._current_step = self._bridge.next_step(timeout=self._step_timeout)
        except queue.Empty:
            raise TimeoutError(
                f"No frame received within {self._step_timeout}s. "
                "Is the remote client connected and sending frames?"
            )
        
        # Apply action mask for the new state
        self._apply_action_mask()
        
        # Reset episode tracking
        self._episode_return = 0.0
        self._episode_length = 0
        self._is_first = False
        
        # NOTE: We do NOT set an action here. The gRPC handler for this frame
        # is blocked waiting for an action. When the policy returns and
        # _handle_step() is called, it will set the action on this frame
        # before fetching the next one. This ensures the first action after
        # reset is actually executed on the emulator.
        
        return self._make_obs(
            reward=0.0,
            is_first=True,
            is_last=False,
            is_terminal=False,
        )

    def _handle_step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a regular (non-reset) step.
        
        1. Decode the action and send it to the current frame's gRPC handler
        2. Wait for the next frame
        3. Update episode statistics
        4. Return the observation
        """
        if self._current_step is None:
            raise RuntimeError(
                "No current step available. This should not happen - "
                "did reset complete successfully?"
            )
        
        # Decode and send the action
        action_idx = int(action.get('action', 0))
        decoded_action = self._decode_action(action_idx)
        self._current_step.set_action(decoded_action)
        
        # Wait for the next frame
        try:
            self._current_step = self._bridge.next_step(timeout=self._step_timeout)
        except queue.Empty:
            raise TimeoutError(
                f"No frame received within {self._step_timeout}s during step. "
                "The remote client may have disconnected."
            )
        
        # Apply action mask for the new state
        self._apply_action_mask()
        
        # Extract step info
        reward = float(self._current_step.reward)
        done = bool(self._current_step.done)
        
        # Update episode stats
        self._episode_return += reward
        self._episode_length += 1
        
        # Handle episode end
        if done:
            self._is_first = True  # Next step will trigger reset
        
        return self._make_obs(
            reward=reward,
            is_first=False,
            is_last=done,
            is_terminal=done,  # For now, all episode ends are terminal
        )

    def _make_obs(
        self,
        reward: float,
        is_first: bool,
        is_last: bool,
        is_terminal: bool,
    ) -> Dict[str, Any]:
        """Construct the observation dictionary from the current step."""
        reward_breakdown = {}
        if self._current_step is None:
            # Return zeros if no step available (shouldn't happen in normal operation)
            state = np.zeros(self._obs_shape, dtype=np.float32)
            action_mask = np.zeros(ACTION_SPEC.size, dtype=np.float32)
        else:
            state = self._current_step.obs
            if self._flatten_obs:
                state = state.reshape(-1)
            if state.dtype != self._obs_dtype:
                if self._obs_dtype == np.uint8:
                    state = np.clip(state, 0, 255).astype(np.uint8)
                else:
                    state = state.astype(self._obs_dtype)
            # Compute action mask from current step info
            action_mask = self._compute_action_mask()
            info = self._current_step.info
            if isinstance(info, dict):
                reward_breakdown = info.get("reward_breakdown") or {}
                if reward_breakdown is None:
                    reward_breakdown = {}
        
        tower_reward = float(reward_breakdown.get("tower", 0.0) or 0.0)
        king_tower_reward = float(reward_breakdown.get("king-tower", 0.0) or 0.0)
        tower_misc_reward = float(reward_breakdown.get("r_", 0.0) or 0.0)
        elixir_penalty = float(reward_breakdown.get("elixir", 0.0) or 0.0)
        enemy_elim_reward = float(reward_breakdown.get("enemy_eliminated", 0.0) or 0.0)
        tower_damage_total = tower_reward + king_tower_reward + tower_misc_reward

        return {
            'state': state,
            'action_mask': action_mask,
            'reward': np.float32(reward),
            'is_first': is_first,
            'is_last': is_last,
            'is_terminal': is_terminal,
            'log/reward_tower': np.float32(tower_reward),
            'log/reward_king_tower': np.float32(king_tower_reward),
            'log/reward_tower_misc': np.float32(tower_misc_reward),
            'log/reward_tower_damage': np.float32(tower_damage_total),
            'log/reward_elixir_waste': np.float32(elixir_penalty),
            'log/reward_enemy_eliminated': np.float32(enemy_elim_reward),
        }

    def _compute_action_mask(self) -> np.ndarray:
        """Compute the action mask for the current state.
        
        Returns:
            Action mask of shape (action_size,) with 0.0 for legal actions
            and -1e9 for illegal actions. This is an additive mask for logits.
            
        Note: In pixel mode (no KataCR perception), cards/elixir info is
        unavailable. In that case, returns all-zeros (no masking) to allow
        the agent to learn action validity from experience.
        """
        if self._current_step is None:
            return np.zeros(ACTION_SPEC.size, dtype=np.float32)
            
        info = self._current_step.info
        if not isinstance(info, dict):
            return np.zeros(ACTION_SPEC.size, dtype=np.float32)
            
        cards = info.get("cards")
        elixir = info.get("elixir", 0)
        
        # If cards info is empty/missing (e.g., pixel mode), disable masking
        # to let the agent learn action validity from experience
        if not cards or len(cards) == 0:
            return np.zeros(ACTION_SPEC.size, dtype=np.float32)
            
        mask = compute_action_mask(cards, elixir)
        return mask.astype(np.float32)

    def _apply_action_mask(self) -> None:
        """Compute and store the action mask in context variable.
        
        This is kept for backward compatibility with the context-variable
        approach. The preferred method is now observation augmentation.
        """
        mask = self._compute_action_mask()
        set_action_mask(mask)

    def _decode_action(self, action_idx: int) -> Optional[Tuple[int, int, int]]:
        """Decode a discrete action index to (card_slot, grid_x, grid_y) tuple."""
        decoded = self._mapper.decode(action_idx)
        if decoded is None:
            return None
        card_slot, gx, gy = decoded
        return (card_slot, gx, gy)

    def close(self) -> None:
        """Clean up resources."""
        if self._current_step is not None:
            # Send final no-op to unblock any waiting gRPC handler
            self._current_step.set_action(None)
            self._current_step = None

    @property
    def episode_return(self) -> float:
        """Return the cumulative reward for the current episode."""
        return self._episode_return

    @property
    def episode_length(self) -> int:
        """Return the number of steps in the current episode."""
        return self._episode_length


def make_clash_royale_env(bridge: RemoteBridgeV3, **kwargs) -> ClashRoyaleEmbodiedEnv:
    """Factory function to create the Clash Royale environment.
    
    This is the function that should be passed to DreamerV3's make_env.
    
    Args:
        bridge: The RemoteBridgeV3 instance.
        **kwargs: Additional arguments passed to ClashRoyaleEmbodiedEnv.
        
    Returns:
        A ClashRoyaleEmbodiedEnv instance.
    """
    return ClashRoyaleEmbodiedEnv(bridge=bridge, **kwargs)


class MaskedAgent:
    """Wrapper that applies action masking to a DreamerV3 agent's policy.
    
    This wrapper intercepts the policy() call and applies the action mask
    from the observation to the agent's action logits before sampling.
    The mask is expected in obs['action_mask'] as additive logits
    (0.0 = legal, -1e9 = illegal).
    
    The masking is done in NumPy after the JAX policy returns to avoid
    host-to-device transfer issues in JIT-compiled code.
    
    Usage:
        agent = Agent(obs_space, act_space, config)
        masked_agent = MaskedAgent(agent, action_key='action')
    """
    
    def __init__(self, agent, action_key: str = 'action'):
        """Initialize the masked agent wrapper.
        
        Args:
            agent: The underlying DreamerV3 agent.
            action_key: The key in act_space for the discrete action to mask.
        """
        self._agent = agent
        self._action_key = action_key
    
    def __getattr__(self, name: str):
        """Delegate all attributes to the underlying agent."""
        return getattr(self._agent, name)
    
    def policy(self, carry, obs, mode='train'):
        """Policy with action masking applied.
        
        This method:
        1. Calls the underlying agent's policy
        2. Extracts the action mask from observations
        3. Re-samples illegal actions in NumPy (outside JAX JIT)
        """
        # Get the action mask from observations before calling policy
        action_mask = obs.get('action_mask')
        
        # Call underlying policy (this is JIT-compiled by JAX)
        carry, act, out = self._agent.policy(carry, obs, mode)
        
        if action_mask is None:
            return carry, act, out
        
        # Apply mask in NumPy (outside JAX) to avoid host-to-device issues
        if self._action_key in act:
            # Convert to NumPy for masking
            sampled_action = np.asarray(act[self._action_key])
            mask = np.asarray(action_mask)
            
            # Check if sampled action is legal (mask value is 0.0 for legal, very negative for illegal)
            if mask.ndim == 2:
                # Batch case: gather mask values for each sampled action
                batch_indices = np.arange(mask.shape[0])
                mask_values = mask[batch_indices, sampled_action]
            else:
                mask_values = mask[sampled_action]
            
            # If action is illegal (mask < 0), resample from legal actions
            is_illegal = mask_values < 0
            
            if np.any(is_illegal):
                # Resample illegal actions uniformly from legal actions
                if mask.ndim == 2:
                    new_actions = sampled_action.copy()
                    for i in range(mask.shape[0]):
                        if is_illegal[i]:
                            legal_indices = np.where(mask[i] >= 0)[0]
                            if len(legal_indices) > 0:
                                new_actions[i] = np.random.choice(legal_indices)
                            else:
                                new_actions[i] = 0
                else:
                    if is_illegal:
                        legal_indices = np.where(mask >= 0)[0]
                        if len(legal_indices) > 0:
                            new_actions = np.random.choice(legal_indices)
                        else:
                            new_actions = 0
                    else:
                        new_actions = sampled_action
                
                act = {**act, self._action_key: new_actions}
        
        return carry, act, out


def make_masked_agent(agent, action_key: str = 'action') -> MaskedAgent:
    """Factory function to wrap an agent with action masking.
    
    Args:
        agent: The underlying DreamerV3 agent.
        action_key: The key in act_space for the discrete action to mask.
        
    Returns:
        A MaskedAgent wrapper.
    """
    return MaskedAgent(agent, action_key=action_key)
