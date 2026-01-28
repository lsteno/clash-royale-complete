"""Lightweight action-mask utilities shared between env and policy.

The mask is stored in a contextvar so the Dreamer policy can pick it up
when sampling actions. Values are additive to logits: 0.0 for legal, a
large negative for illegal.
"""
from __future__ import annotations

import contextvars
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

KATACR_ROOT = Path(__file__).resolve().parents[3] / "KataCR"
if str(KATACR_ROOT) not in sys.path:
    sys.path.insert(0, str(KATACR_ROOT))

from katacr.constants.card_list import card2elixir, card_list
from src.specs import ACTION_SPEC


_MASK_CTX: contextvars.ContextVar | None = None


def _ctx() -> contextvars.ContextVar:
    global _MASK_CTX
    if _MASK_CTX is None:
        _MASK_CTX = contextvars.ContextVar("action_mask", default=None)
    return _MASK_CTX


def compute_action_mask(cards: Sequence | None, elixir: float) -> np.ndarray:
    """Return additive logits mask of shape (action_size,).

    cards is expected to be a sequence where deployable slots live at
    positions 1..4 (matching ActionMapper). Missing/None/unknown cards are
    treated as illegal. Illegal actions receive -1e9.
    """

    mask = np.zeros(ACTION_SPEC.size, dtype=np.float32)
    if cards is None:
        cards = []
    cells = ACTION_SPEC.cells_per_card
    negative = -1e9

    def _card_affordable(slot: int) -> bool:
        if slot <= 0:
            return True  # no-op already guarded by slot > 0 elsewhere
        if slot >= len(cards):
            return False
        name = _resolve_card_name(cards[slot])
        if name is None:
            return False
        cost = card2elixir.get(name)
        if cost is None:
            return False
        try:
            return float(elixir) >= float(cost)
        except Exception:
            return False

    for action_idx in range(1, ACTION_SPEC.size):
        slot = (action_idx - 1) // cells + 1
        if not _card_affordable(slot):
            mask[action_idx] = negative
    # Ensure the no-op action is always legal
    mask[0] = 0.0
    return mask


def _resolve_card_name(card) -> str | None:
    if card is None:
        return None
    if isinstance(card, str):
        return card
    try:
        idx = int(card)
    except Exception:
        return None
    if 0 <= idx < len(card_list):
        return card_list[idx]
    return None


def set_action_mask(mask: Iterable | None) -> None:
    _ctx().set(mask)


def get_action_mask():
    return _ctx().get()


def clear_action_mask() -> None:
    _ctx().set(None)
