"""Embodied DreamerV3 environment for local Redroid/ADB "hive" setups."""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import elements
import numpy as np

import embodied

from src.environment.action_mask import compute_action_mask, set_action_mask
from src.environment.action_utils import ActionMapper, DEFAULT_DEPLOY_CELLS, DEFAULT_SPELL_CELLS
from src.environment.emulator_env import ClashRoyaleKataCREnv, EmulatorConfig
from src.environment.state_encoder import StateTensorEncoder
from src.perception.katacr_pipeline import KataCRVisionConfig
from src.specs import ACTION_SPEC, OBS_SPEC


@dataclass
class HiveEnvConfig:
    """Configuration for a single Redroid-backed environment."""

    device_serial: str
    adb_path: str = "adb"
    screen_width: int = 720
    screen_height: int = 1280
    tap_reference_width: int = 1080
    tap_reference_height: int = 2400
    canonical_width: int = 576
    canonical_height: int = 1280
    use_adb_capture_only: bool = True
    auto_restart: bool = True


class ClashRoyaleHiveEmbodiedEnv(embodied.Env):
    """Local ADB-backed DreamerV3 environment for Redroid containers."""

    def __init__(
        self,
        config: HiveEnvConfig,
        *,
        vision_cfg: Optional[KataCRVisionConfig] = None,
        step_timeout: Optional[float] = None,
        flatten_obs: bool = True,
        pixels: bool = False,
        pixel_height: int = 192,
        pixel_width: int = 256,
        max_game_seconds: int = 360,
    ):
        self._config = config
        self._step_timeout = step_timeout
        self._flatten_obs = flatten_obs
        self._pixels = bool(pixels)
        self._pixel_height = int(pixel_height)
        self._pixel_width = int(pixel_width)
        self._obs_dtype = np.uint8 if self._pixels else np.float32

        emu_cfg = EmulatorConfig(
            adb_path=config.adb_path,
            device_serial=config.device_serial,
            screen_width=config.screen_width,
            screen_height=config.screen_height,
            tap_reference_width=config.tap_reference_width,
            tap_reference_height=config.tap_reference_height,
            canonical_width=config.canonical_width,
            canonical_height=config.canonical_height,
            use_adb_capture_only=config.use_adb_capture_only,
            auto_restart=config.auto_restart,
        )
        self._base = ClashRoyaleKataCREnv(emu_cfg, vision_cfg)
        self._encoder = StateTensorEncoder(max_game_seconds)
        self._mapper = ActionMapper(DEFAULT_DEPLOY_CELLS, DEFAULT_SPELL_CELLS)

        if self._pixels:
            self._obs_shape = (self._pixel_height, self._pixel_width, 3)
        elif self._flatten_obs:
            self._obs_shape = (OBS_SPEC.channels * OBS_SPEC.height * OBS_SPEC.width,)
        else:
            self._obs_shape = OBS_SPEC.shape

        self._latest_obs: Optional[np.ndarray] = None
        self._latest_info: Dict[str, Any] = {}
        self._latest_state: Dict[str, Any] = {}
        self._latest_frame_bgr: Optional[np.ndarray] = None

        self._episode_return = 0.0
        self._episode_length = 0
        self._is_first = True

    @functools.cached_property
    def obs_space(self) -> Dict[str, elements.Space]:
        return {
            "state": elements.Space(self._obs_dtype, self._obs_shape),
            "action_mask": elements.Space(np.float32, (ACTION_SPEC.size,)),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
        }

    @functools.cached_property
    def act_space(self) -> Dict[str, elements.Space]:
        return {
            "action": elements.Space(np.int32, (), 0, ACTION_SPEC.size),
            "reset": elements.Space(bool),
        }

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        should_reset = action.get("reset", False) or self._is_first
        if should_reset:
            return self._handle_reset()
        return self._handle_step(action)

    def _handle_reset(self) -> Dict[str, Any]:
        self._base.reset()
        self._capture()
        self._episode_return = 0.0
        self._episode_length = 0
        self._is_first = False
        return self._make_obs(
            reward=0.0,
            is_first=True,
            is_last=False,
            is_terminal=False,
        )

    def _handle_step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        action_idx = int(action.get("action", 0))
        decoded = self._decode_action(action_idx)
        if decoded is not None:
            self._base.step(decoded)

        self._capture()
        reward = float(self._latest_info.get("reward", 0.0))
        done = self._is_terminal(self._latest_info) or self._perception_detects_match_end()

        self._episode_return += reward
        self._episode_length += 1
        if done:
            self._is_first = True
            if hasattr(self._base, "mark_match_finished"):
                self._base.mark_match_finished()
            if getattr(self._base, "katacr", None) is not None:
                try:
                    self._base.katacr.reset()
                except Exception:
                    pass

        return self._make_obs(
            reward=reward,
            is_first=False,
            is_last=done,
            is_terminal=done,
        )

    def _capture(self) -> None:
        result, _, frame_bgr = self._base.capture_state()
        self._latest_frame_bgr = frame_bgr
        self._latest_info = result.info if isinstance(result.info, dict) else {}
        self._latest_info["reward"] = float(result.reward or 0.0)
        self._latest_state = result.state if isinstance(result.state, dict) else {}

        if self._pixels:
            obs = self._frame_to_obs(frame_bgr)
        else:
            obs = self._encoder.encode(
                self._latest_state,
                self._base.katacr.reward_builder,
                self._latest_info,
            )
        self._latest_obs = obs

    def _frame_to_obs(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if (frame_rgb.shape[0], frame_rgb.shape[1]) != (self._pixel_height, self._pixel_width):
            frame_rgb = cv2.resize(
                frame_rgb,
                (self._pixel_width, self._pixel_height),
                interpolation=cv2.INTER_AREA,
            )
        return frame_rgb

    def _make_obs(
        self,
        reward: float,
        is_first: bool,
        is_last: bool,
        is_terminal: bool,
    ) -> Dict[str, Any]:
        if self._latest_obs is None:
            state = np.zeros(self._obs_shape, dtype=self._obs_dtype)
        else:
            state = self._latest_obs
            if self._flatten_obs and state.ndim > 1:
                state = state.reshape(-1)
            if state.dtype != self._obs_dtype:
                if self._obs_dtype == np.uint8:
                    state = np.clip(state, 0, 255).astype(np.uint8)
                else:
                    state = state.astype(self._obs_dtype)

        action_mask = self._compute_action_mask()
        set_action_mask(action_mask)

        return {
            "state": state,
            "action_mask": action_mask,
            "reward": np.float32(reward),
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        }

    def _compute_action_mask(self) -> np.ndarray:
        info = self._latest_info if isinstance(self._latest_info, dict) else {}
        cards = info.get("cards")
        elixir = info.get("elixir", self._latest_state.get("elixir", 0))
        if bool(info.get("elixir_failed", False)):
            elixir = -1
        try:
            if elixir is None or float(elixir) < 0:
                elixir = -1
        except Exception:
            elixir = -1
        if not cards or len(cards) == 0:
            return np.zeros(ACTION_SPEC.size, dtype=np.float32)
        mask = compute_action_mask(cards, elixir)
        return mask.astype(np.float32)

    def _decode_action(self, action_idx: int) -> Optional[Tuple[int, int, int]]:
        cards = self._latest_info.get("cards") if isinstance(self._latest_info, dict) else None
        decoded = self._mapper.decode(action_idx, cards=cards)
        if decoded is None:
            return None
        card_slot, gx, gy = decoded
        return (card_slot, gx, gy)

    def _perception_detects_match_end(self) -> bool:
        if hasattr(self._base, "is_match_over"):
            try:
                return bool(self._base.is_match_over(self._latest_frame_bgr))
            except Exception:
                return False
        return False

    @staticmethod
    def _is_terminal(info: Dict[str, Any]) -> bool:
        try:
            return bool(info.get("center_flag") == 1)
        except Exception:
            return False

    def close(self) -> None:
        self._latest_obs = None
        self._latest_info = {}
        self._latest_state = {}
        self._base = None  # type: ignore[assignment]

    @property
    def episode_return(self) -> float:
        return self._episode_return

    @property
    def episode_length(self) -> int:
        return self._episode_length
