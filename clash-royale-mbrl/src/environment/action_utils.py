"""Action mapping utilities for Clash Royale environments.

This module contains action-related utilities that are independent of
the environment implementation (local vs remote) and don't require cv2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from src.specs import ACTION_SPEC


@dataclass(frozen=True)
class DeployCell:
    """Canonical deploy cells on the friendly side of the arena grid.
    
    The arena grid is 18 wide × 32 tall. These are the 9 discrete positions
    where cards can be deployed on the friendly side.
    """
    grid_x: int
    grid_y: int


# Default 9 deploy positions arranged in a 3x3 grid on the friendly side
DEFAULT_DEPLOY_CELLS: Tuple[DeployCell, ...] = (
    # Row 1 (closest to river)
    DeployCell(4, 18),
    DeployCell(9, 18),
    DeployCell(14, 18),
    # Row 2 (middle)
    DeployCell(4, 22),
    DeployCell(9, 22),
    DeployCell(14, 22),
    # Row 3 (closest to king tower)
    DeployCell(4, 26),
    DeployCell(9, 26),
    DeployCell(14, 26),
)

DEFAULT_SPELL_CELLS: Tuple[DeployCell, ...] = (
    # Top (enemy) half: approximate tower/behind-tower targets.
    DeployCell(4, 6),
    DeployCell(9, 6),
    DeployCell(14, 6),
    # River line / mid: common engagement locations.
    DeployCell(4, 14),
    DeployCell(9, 14),
    DeployCell(14, 14),
    # Bottom (friendly) half: defensive spell placements.
    DeployCell(4, 22),
    DeployCell(9, 22),
    DeployCell(14, 22),
)


class ActionMapper:
    """Translate discrete Dreamer actions to emulator tap commands.
    
    Action space layout:
    - Action 0: No-op
    - Actions 1-9: Card 1 at deploy cells 1-9
    - Actions 10-18: Card 2 at deploy cells 1-9
    - Actions 19-27: Card 3 at deploy cells 1-9
    - Actions 28-36: Card 4 at deploy cells 1-9
    
    Total: 37 actions = 1 no-op + 4 cards × 9 cells
    """

    def __init__(
        self,
        deploy_cells: Sequence[DeployCell] = DEFAULT_DEPLOY_CELLS,
        spell_cells: Sequence[DeployCell] = DEFAULT_SPELL_CELLS,
    ):
        """Initialize the action mapper.
        
        Args:
            deploy_cells: Sequence of DeployCell defining valid deploy positions.
                Must have exactly cells_per_card entries (default 9).
        """
        if len(deploy_cells) != ACTION_SPEC.cells_per_card:
            raise ValueError(
                f"Expected {ACTION_SPEC.cells_per_card} deploy cells, got {len(deploy_cells)}"
            )
        if len(spell_cells) != ACTION_SPEC.cells_per_card:
            raise ValueError(
                f"Expected {ACTION_SPEC.cells_per_card} spell cells, got {len(spell_cells)}"
            )
        self._deploy_cells = list(deploy_cells)
        self._spell_cells = list(spell_cells)

    def decode(self, action_index: int, *, cards: Optional[Sequence] = None) -> Optional[Tuple[int, int, int]]:
        """Decode a discrete action index to (card_slot, grid_x, grid_y).
        
        Args:
            action_index: Integer in [0, ACTION_SPEC.size)
            
        Returns:
            Tuple of (card_slot, grid_x, grid_y) where card_slot is 1-4,
            or None if action_index is 0 (no-op) or invalid.
        """
        if action_index == 0:
            return None
            
        action_index -= 1
        card_slot = action_index // len(self._deploy_cells)
        cell_idx = action_index % len(self._deploy_cells)
        card_slot += 1  # emulator expects 1-4 for the current hand
        
        if card_slot > ACTION_SPEC.num_cards:
            return None
            
        cells = self._cells_for_slot(card_slot, cards)
        cell = cells[cell_idx]
        return (card_slot, cell.grid_x, cell.grid_y)

    def encode(self, card_slot: int, grid_x: int, grid_y: int) -> Optional[int]:
        """Encode (card_slot, grid_x, grid_y) to a discrete action index.
        
        Args:
            card_slot: Card slot 1-4
            grid_x: Grid X coordinate
            grid_y: Grid Y coordinate
            
        Returns:
            Action index in [1, ACTION_SPEC.size) or None if no matching cell.
        """
        if card_slot < 1 or card_slot > ACTION_SPEC.num_cards:
            return None
            
        # Find matching cell
        for cell_idx, cell in enumerate(self._deploy_cells):
            if cell.grid_x == grid_x and cell.grid_y == grid_y:
                return 1 + (card_slot - 1) * len(self._deploy_cells) + cell_idx
                
        return None

    @property
    def num_actions(self) -> int:
        """Return the total number of discrete actions."""
        return ACTION_SPEC.size

    @property
    def cells(self) -> Tuple[DeployCell, ...]:
        """Return the deploy cells."""
        return tuple(self._deploy_cells)

    def _cells_for_slot(self, card_slot: int, cards: Optional[Sequence]) -> list[DeployCell]:
        if not cards or card_slot >= len(cards):
            return self._deploy_cells
        name = _resolve_card_name(cards[card_slot])
        if name is not None and _is_spell(name):
            return self._spell_cells
        return self._deploy_cells


_SPELL_NAMES: Optional[set[str]] = None


def _is_spell(card_name: str) -> bool:
    global _SPELL_NAMES
    if _SPELL_NAMES is None:
        try:
            from katacr.constants.label_list import spell_unit_list  # type: ignore
        except Exception:
            _SPELL_NAMES = set()
        else:
            _SPELL_NAMES = set(spell_unit_list)
    return card_name in _SPELL_NAMES


def _resolve_card_name(card) -> Optional[str]:
    if card is None:
        return None
    if isinstance(card, str):
        return card
    try:
        idx = int(card)
    except Exception:
        return None
    try:
        from katacr.constants.card_list import card_list  # type: ignore
    except Exception:
        return None
    if 0 <= idx < len(card_list):
        return card_list[idx]
    return None
