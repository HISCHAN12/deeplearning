"""Manual TFT game-state features and short-term flow tracking."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class GameState:
    stage: float = 3.2
    health: int = 80
    gold: int = 30
    level: int = 6
    streak: int = 0
    board_strength: int = 6
    unspent_items: int = 1
    health_delta: int = 0
    gold_delta: int = 0
    level_delta: int = 0
    board_strength_delta: int = 0

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                np.clip(self.stage / 7.0, 0.0, 1.0),
                np.clip(self.health / 100.0, 0.0, 1.0),
                np.clip(self.gold / 100.0, 0.0, 1.0),
                np.clip(self.level / 10.0, 0.0, 1.0),
                np.clip(self.streak / 5.0, -1.0, 1.0),
                np.clip(self.board_strength / 10.0, 0.0, 1.0),
                np.clip(self.unspent_items / 6.0, 0.0, 1.0),
                np.clip(self.health_delta / 30.0, -1.0, 1.0),
                np.clip(self.gold_delta / 50.0, -1.0, 1.0),
                np.clip(self.level_delta / 2.0, -1.0, 1.0),
                np.clip(self.board_strength_delta / 5.0, -1.0, 1.0),
            ],
            dtype=np.float64,
        )


def with_flow(current: GameState, previous: GameState | None) -> GameState:
    if previous is None:
        return current
    return replace(
        current,
        health_delta=current.health - previous.health,
        gold_delta=current.gold - previous.gold,
        level_delta=current.level - previous.level,
        board_strength_delta=current.board_strength - previous.board_strength,
    )


def neutral_state_vector() -> np.ndarray:
    return GameState().to_vector()
