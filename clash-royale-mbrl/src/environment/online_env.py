from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import gym
import numpy as np
from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.float_box import FloatBox

from dreamer.envs.env import EnvInfo

# Ensure KataCR package is importable when running from this repo checkout.
KATACR_ROOT = Path(__file__).resolve().parents[3] / "KataCR"
if str(KATACR_ROOT) not in sys.path:
    sys.path.insert(0, str(KATACR_ROOT))

from katacr.constants.card_list import card2elixir, card_list
from katacr.constants.label_list import (
    flying_unit_list,
    ground_unit_list,
    idx2unit,
    spell_unit_list,
    tower_unit_list,
    unit2idx,
)

from src.environment.emulator_env import ClashRoyaleKataCREnv, EmulatorConfig
from src.perception.katacr_pipeline import KataCRVisionConfig
from src.specs import ACTION_SPEC, OBS_SPEC

_UNIT_FRIENDLY = 1
_UNIT_ENEMY = 0

_GROUND = set(ground_unit_list)
_FLYING = set(flying_unit_list)
_SPELL = set(spell_unit_list)
_TOWER = set(tower_unit_list)
_CARD_COUNT = len(card_list)

CH_FRIENDLY_GROUND = 0
CH_FRIENDLY_AIR = 1
CH_ENEMY_GROUND = 2
CH_ENEMY_AIR = 3
CH_FRIENDLY_SPELL = 4
CH_ENEMY_SPELL = 5
CH_FRIENDLY_STRUCT = 6
CH_ENEMY_STRUCT = 7
CH_ELIXIR = 8
CH_TIME = 9
CH_NEXT_CARD = 10
CH_CARD_1 = 11
CH_CARD_2 = 12
CH_CARD_3 = 13
CH_CARD_4 = 14


@dataclass(frozen=True)
class DeployCell:
    """Canonical deploy cells on the friendly side of the arena grid."""

    grid_x: int
    grid_y: int


DEFAULT_DEPLOY_CELLS: Tuple[DeployCell, ...] = (
    DeployCell(4, 18),
    DeployCell(9, 18),
    DeployCell(14, 18),
    DeployCell(4, 22),
    DeployCell(9, 22),
    DeployCell(14, 22),
    DeployCell(4, 26),
    DeployCell(9, 26),
    DeployCell(14, 26),
)


@dataclass
class OnlineEnvConfig:
    """High-level knobs for the Dreamer-facing wrapper."""

    emulator: EmulatorConfig = field(default_factory=EmulatorConfig)
    vision: KataCRVisionConfig = field(default_factory=KataCRVisionConfig)
    deploy_cells: Sequence[DeployCell] = DEFAULT_DEPLOY_CELLS
    max_game_seconds: int = 360


class ActionMapper:
    """Translate discrete Dreamer actions to emulator taps."""

    def __init__(self, deploy_cells: Sequence[DeployCell]):
        if len(deploy_cells) != ACTION_SPEC.cells_per_card:
            raise ValueError(
                f"Expected {ACTION_SPEC.cells_per_card} deploy cells, got {len(deploy_cells)}"
            )
        self._cells = list(deploy_cells)

    def decode(self, action_index: int) -> Optional[Tuple[int, int, int]]:
        """Return (card_slot, grid_x, grid_y) or None for no-op."""

        if action_index == 0:
            return None
        action_index -= 1
        card_slot = action_index // len(self._cells)
        cell_idx = action_index % len(self._cells)
        card_slot += 1  # emulator expects 1-4 for the current hand
        if card_slot > ACTION_SPEC.num_cards:
            return None
        cell = self._cells[cell_idx]
        return (card_slot, cell.grid_x, cell.grid_y)


class StateTensorEncoder:
    """Project KataCR state dictionaries into Dreamer's grid tensor."""

    def __init__(self, max_game_seconds: int):
        self._max_game_seconds = max_game_seconds

    def encode(self, state: dict, reward_builder, info: dict) -> np.ndarray:
        grid = np.zeros(OBS_SPEC.shape, dtype=np.float32)
        structure_candidates = []
        for unit in state.get("unit_infos", []):
            xy = unit.get("xy")
            cls_idx = unit.get("cls")
            bel = unit.get("bel")
            if xy is None or cls_idx is None or bel is None:
                continue
            unit_name = idx2unit.get(int(cls_idx))
            if unit_name is None:
                continue
            channel = self._channel_for_unit(unit_name, bel)
            if channel is not None:
                gx, gy = self._cell_indices(xy)
                grid[channel, gy, gx] = min(1.0, grid[channel, gy, gx] + 1.0)
            if unit_name in _TOWER:
                structure_candidates.append((bel, unit_name, xy))
        self._paint_structures(grid, structure_candidates, reward_builder)
        self._paint_scalar_channels(grid, state, info)
        self._paint_cards(grid, state.get("cards"))
        return grid

    def _channel_for_unit(self, name: str, bel: int) -> Optional[int]:
        friendly = bel == _UNIT_FRIENDLY
        if name in _TOWER:
            return None
        if name in _SPELL:
            return CH_FRIENDLY_SPELL if friendly else CH_ENEMY_SPELL
        if name in _FLYING:
            return CH_FRIENDLY_AIR if friendly else CH_ENEMY_AIR
        if name in _GROUND:
            return CH_FRIENDLY_GROUND if friendly else CH_ENEMY_GROUND
        return None

    def _paint_structures(self, grid: np.ndarray, structures, reward_builder) -> None:
        for bel, name, xy in structures:
            gx, gy = self._cell_indices(xy)
            ratio = self._structure_hp_ratio(bel, name, xy, reward_builder)
            channel = CH_FRIENDLY_STRUCT if bel == _UNIT_FRIENDLY else CH_ENEMY_STRUCT
            grid[channel, gy, gx] = max(grid[channel, gy, gx], ratio)

    def _structure_hp_ratio(self, bel: int, name: str, xy, reward_builder) -> float:
        bel_idx = 1 if bel == _UNIT_FRIENDLY else 0
        if name == "king-tower":
            hp = reward_builder.hp_king_tower[bel_idx]
            full = reward_builder.full_hp["king-tower"][bel_idx]
            return self._safe_ratio(hp, full)
        hp_grid = reward_builder.hp_tower
        full = reward_builder.full_hp["tower"][bel_idx]
        side = 1 if xy[0] >= OBS_SPEC.width / 2 else 0
        hp = hp_grid[bel_idx, side]
        return self._safe_ratio(hp, full)

    def _paint_scalar_channels(self, grid: np.ndarray, state: dict, info: dict) -> None:
        elixir_value = self._safe_numeric(state.get("elixir"), default=0.0)
        grid[CH_ELIXIR, :, :] = np.clip(elixir_value / 10.0, 0.0, 1.0)
        time_value = self._safe_numeric(info.get("time"), default=np.nan)
        if not np.isfinite(time_value):
            time_norm = 0.0
        else:
            time_norm = np.clip(time_value / self._max_game_seconds, 0.0, 1.0)
        grid[CH_TIME, :, :] = time_norm

    def _paint_cards(self, grid: np.ndarray, cards: Optional[List[int]]) -> None:
        if cards is None:
            cards = []
        elif not isinstance(cards, list):
            cards = list(cards)
        slots = (cards + [-1] * 5)[:5]
        channels = [
            CH_NEXT_CARD,
            CH_CARD_1,
            CH_CARD_2,
            CH_CARD_3,
            CH_CARD_4,
        ]
        for channel, idx in zip(channels, slots):
            grid[channel, :, :] = self._card_scalar(idx)

    @staticmethod
    def _card_scalar(card_idx: Optional[int]) -> float:
        if card_idx is None:
            return 0.0
        idx = int(card_idx)
        if idx < 0:
            return 0.0
        if _CARD_COUNT <= 1:
            return 0.0
        return np.clip(idx / (_CARD_COUNT - 1), 0.0, 1.0)

    @staticmethod
    def _safe_ratio(value: float, full: float) -> float:
        if value is None or value <= 0 or full <= 0:
            return 0.0
        return np.clip(value / full, 0.0, 1.0)

    @staticmethod
    def _safe_numeric(value, default: float = 0.0) -> float:
        try:
            if value is None:
                return float(default)
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _cell_indices(xy: Iterable[float]) -> Tuple[int, int]:
        x, y = float(xy[0]), float(xy[1])
        gx = int(np.clip(round(x), 0, OBS_SPEC.width - 1))
        gy = int(np.clip(round(y), 0, OBS_SPEC.height - 1))
        return gx, gy


class ClashRoyaleDreamerEnv(Env):
    """rlpyt-compatible environment exposing the emulator to Dreamer."""

    def __init__(self, config: OnlineEnvConfig = OnlineEnvConfig()):
        self._config = config
        self._base = ClashRoyaleKataCREnv(config.emulator, config.vision)
        self._encoder = StateTensorEncoder(config.max_game_seconds)
        self._action_mapper = ActionMapper(config.deploy_cells)
        self._obs_space = FloatBox(low=0.0, high=1.0, shape=OBS_SPEC.shape, dtype=np.float32)
        self._act_space = gym.spaces.Discrete(ACTION_SPEC.size)
        self.random = np.random.RandomState()
        self._latest_cards: List[str] = []
        self._latest_elixir: int = 0
        self._latest_time: float = 0.0
        self._episode_return = 0.0
        self._episode_steps = 0

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._act_space

    def reset(self):
        self._base.reset()
        obs, _ = self._capture()
        self._episode_return = 0.0
        self._episode_steps = 0
        return obs

    def step(self, action):
        decoded = self._action_mapper.decode(int(action))
        if decoded is not None:
            card_slot, gx, gy = decoded
            print(f"[Env] Agent deploying card {card_slot} at grid ({gx}, {gy})")
            self._attempt_deploy(*decoded)
        obs, result = self._capture()
        reward_value = result.reward if result.reward is not None else 0.0
        reward = float(reward_value)
        perception_terminal = self._perception_detects_match_end(result)
        done = self._is_terminal(result.info) or perception_terminal
        if done and hasattr(self._base, "mark_match_finished"):
            self._base.mark_match_finished()
        self._episode_return += reward
        self._episode_steps += 1
        discount = np.array(0.0 if done else 1.0, dtype=np.float32)
        env_info = EnvInfo(discount, self._episode_return, done)
        return EnvStep(obs, reward, done, env_info)

    def close(self):
        self._base = None  # type: ignore[assignment]

    def _attempt_deploy(self, card_slot: int, grid_x: int, grid_y: int) -> None:
        card_names = self._latest_cards
        if card_slot <= 0 or card_slot >= len(card_names):
            return
        card_name = card_names[card_slot]
        elixir_cost = card2elixir.get(card_name, 0)
        if elixir_cost is None or elixir_cost < 0:
            elixir_cost = 0
        if self._latest_elixir < elixir_cost:
            return
        self._base.step((card_slot, grid_x, grid_y))

    def _capture(self):
        result, _, _ = self._base.capture_state()
        self._latest_cards = result.info.get("cards", []) or []
        elixir_value = self._coerce_number(result.state.get("elixir"), default=0.0)
        self._latest_elixir = int(max(0.0, elixir_value))
        self._latest_time = self._coerce_number(result.info.get("time"), default=0.0)
        obs = self._encoder.encode(result.state, self._base.katacr.reward_builder, result.info)
        return obs, result

    @staticmethod
    def _coerce_number(value, default: float = 0.0) -> float:
        if value is None:
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _perception_detects_match_end(self, result) -> bool:
        if hasattr(self._base, "is_match_over") and self._base.is_match_over():
            print("[Env] Ending episode due to ui_buttons")
            return True
        return False

    def _is_terminal(self, info: dict) -> bool:
        # this didn't work well in practice
        return False
