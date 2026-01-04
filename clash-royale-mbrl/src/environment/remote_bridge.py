"""Bridges gRPC frame arrivals to a local gym-style environment.

Machine B (emulator) calls the FrameService on Machine A. The processor
publishes each perceived observation into this bridge. The training loop on
Machine A consumes observations via a gym-like env and supplies actions; the
processor then returns that action to Machine B in the same RPC.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from src.environment.online_env import ActionMapper, DEFAULT_DEPLOY_CELLS


@dataclass
class RemoteStep:
    obs: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]
    _action_ready: threading.Event = threading.Event()
    _action: Optional[Tuple[int, int, int]] = None

    def set_action(self, action: Optional[Tuple[int, int, int]]) -> None:
        self._action = action
        self._action_ready.set()

    def wait_action(self, timeout: Optional[float] = None) -> Optional[Tuple[int, int, int]]:
        self._action_ready.wait(timeout=timeout)
        return self._action


class RemoteBridge:
    """Thread-safe conduit between the gRPC processor and the trainer env."""

    def __init__(self):
        self._steps: "queue.Queue[RemoteStep]" = queue.Queue()

    def publish(self, obs: np.ndarray, reward: float, done: bool, info: Dict[str, Any]) -> Optional[Tuple[int, int, int]]:
        step = RemoteStep(obs=obs, reward=reward, done=done, info=info, _action_ready=threading.Event())
        self._steps.put(step)
        return step.wait_action()

    def next_step(self, timeout: Optional[float] = None) -> RemoteStep:
        return self._steps.get(timeout=timeout)


class RemoteClashRoyaleEnv:
    """Minimal gym-like env that consumes frames pushed by RemoteBridge."""

    def __init__(self, bridge: RemoteBridge, obs_space, act_space):
        self._bridge = bridge
        self.observation_space = obs_space
        self.action_space = act_space
        self._current: Optional[RemoteStep] = None
        self._mapper = ActionMapper(DEFAULT_DEPLOY_CELLS)

    def reset(self):
        # Block until the first frame arrives from Machine B.
        self._current = self._bridge.next_step()
        return self._current.obs

    def step(self, action):
        if self._current is None:
            raise RuntimeError("Call reset() before step().")
        # Supply action for the current frame so the RPC can return.
        self._current.set_action(self._decode_action(action))
        # Block until next frame arrives.
        self._current = self._bridge.next_step()
        return self._current.obs, float(self._current.reward), bool(self._current.done), self._current.info

    def _decode_action(self, action) -> Optional[Tuple[int, int, int]]:
        # action can be tuple or discrete index depending on sampler
        if isinstance(action, (tuple, list)) and len(action) == 3:
            return int(action[0]), int(action[1]), int(action[2])
        try:
            a = int(action)
        except Exception:
            return None
        decoded = self._mapper.decode(a)
        if decoded is None:
            return None
        card_slot, gx, gy = decoded
        return card_slot, gx, gy

    def close(self):
        self._current = None
