"""
Specification constants for the online-only Clash Royale pipeline.
- Observation: KataCR grid (channels, height, width) = (15, 32, 18), float32
- Action space: 37 discrete actions = 1 no-op + 4 cards * 9 deploy cells
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ObsSpec:
    channels: int = 15
    height: int = 32
    width: int = 18

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.channels, self.height, self.width)


@dataclass(frozen=True)
class ActionSpec:
    # 1 no-op + 4 cards * 9 cells
    num_cards: int = 4
    cells_per_card: int = 9

    @property
    def size(self) -> int:
        return 1 + self.num_cards * self.cells_per_card


OBS_SPEC = ObsSpec()
ACTION_SPEC = ActionSpec()
