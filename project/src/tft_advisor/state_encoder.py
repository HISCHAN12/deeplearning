"""Helpers for turning selected TFT state into model features."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .synthetic_data import ARCHETYPES, AUGMENTS, CHAMPIONS, ITEMS, TRAITS


def deck_vector_from_names(
    champions: Iterable[str],
    traits: Iterable[str],
    items: Iterable[str],
    augments: Iterable[str],
) -> np.ndarray:
    parts = [
        _multi_hot(champions, CHAMPIONS),
        _multi_hot(traits, TRAITS),
        _multi_hot(items, ITEMS),
        _multi_hot(augments, AUGMENTS),
    ]
    return np.concatenate(parts)


def archetype_deck_vector(archetype: str) -> np.ndarray:
    spec = ARCHETYPES[archetype]
    return deck_vector_from_names(spec["champions"], spec["traits"], spec["items"], spec["augments"])


def board_from_style(style: str) -> np.ndarray:
    board = np.zeros((4, 7), dtype=np.float64)
    if style == "frontline":
        board[0, [1, 3, 5]] = [0.85, 1.0, 0.8]
        board[3, [2, 4]] = [0.75, 0.65]
    elif style == "backline":
        board[1, [2, 4]] = [0.55, 0.6]
        board[3, [1, 3, 5]] = [0.8, 1.0, 0.75]
    elif style == "corner_carry":
        board[0, [2, 3, 4]] = [0.7, 0.85, 0.7]
        board[3, [0, 1, 6]] = [1.0, 0.65, 0.55]
    else:
        board[0, [2, 4]] = [0.75, 0.75]
        board[2, [1, 3, 5]] = [0.55, 0.8, 0.55]
        board[3, [2, 4]] = [0.7, 0.7]
    return board


def _multi_hot(selected: Iterable[str], universe: list[str]) -> np.ndarray:
    vector = np.zeros(len(universe), dtype=np.float64)
    for value in selected:
        if value in universe:
            vector[universe.index(value)] = 1.0
    return vector

