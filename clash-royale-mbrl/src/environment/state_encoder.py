from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Ensure KataCR package is importable when running from this repo checkout.
KATACR_ROOT = Path(__file__).resolve().parents[3] / "KataCR"
if str(KATACR_ROOT) not in sys.path:
    sys.path.insert(0, str(KATACR_ROOT))

from katacr.constants.card_list import card_list
from katacr.constants.label_list import (
    flying_unit_list,
    ground_unit_list,
    idx2unit,
    spell_unit_list,
    tower_unit_list,
)

from src.specs import OBS_SPEC

# KataCR uses `bel` (belonging) as a side indicator. In the upstream codebase,
# `bel == 0` corresponds to the bottom player (friendly from our perspective)
# and `bel == 1` to the top player (enemy).
_UNIT_FRIENDLY = 0
_UNIT_ENEMY = 1

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

CHANNEL_NAMES: Tuple[str, ...] = (
    "friendly_ground",
    "friendly_air",
    "enemy_ground",
    "enemy_air",
    "friendly_spell",
    "enemy_spell",
    "friendly_struct_hp",
    "enemy_struct_hp",
    "elixir",
    "time",
    "next_card",
    "card_1",
    "card_2",
    "card_3",
    "card_4",
)


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
        bel_idx = 0 if bel == _UNIT_FRIENDLY else 1
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
