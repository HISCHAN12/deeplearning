"""Synthetic TFT match-log generation for reproducible course experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


CHAMPIONS = [
    "Ahri",
    "Annie",
    "Aphelios",
    "Ashe",
    "Azir",
    "Bard",
    "Darius",
    "Ekko",
    "Ezreal",
    "Garen",
    "Jinx",
    "Kaisa",
    "Karma",
    "Kayle",
    "LeeSin",
    "Lux",
    "Mordekaiser",
    "Neeko",
    "Riven",
    "Sett",
    "Sona",
    "Syndra",
    "Taric",
    "Vayne",
]

TRAITS = [
    "Arcanist",
    "Bruiser",
    "Duelist",
    "Invoker",
    "Sniper",
    "Sentinel",
    "Warden",
    "Trickshot",
]

ITEMS = [
    "Guinsoo",
    "InfinityEdge",
    "JeweledGauntlet",
    "LastWhisper",
    "Morello",
    "RedBuff",
    "SpearOfShojin",
    "Sunfire",
    "TitanResolve",
    "Warmog",
]

AUGMENTS = [
    "Combat",
    "Economy",
    "Reroll",
    "Item",
    "Trait",
    "Leveling",
]

ACTIONS = ["level_up", "roll_down", "hold_economy", "slam_item", "reposition_carry"]


@dataclass(frozen=True)
class MatchRecord:
    deck_vector: np.ndarray
    board_grid: np.ndarray
    action: str
    placement: int
    archetype: str


ARCHETYPES: Dict[str, Dict[str, List[str]]] = {
    "fast_8_carry": {
        "champions": ["Ahri", "Azir", "Bard", "Ekko", "Karma", "Sett", "Sona", "Taric"],
        "traits": ["Invoker", "Sentinel", "Warden"],
        "items": ["SpearOfShojin", "JeweledGauntlet", "Morello"],
        "augments": ["Leveling", "Combat"],
    },
    "reroll_duelist": {
        "champions": ["Ashe", "Darius", "Jinx", "Kayle", "LeeSin", "Riven", "Vayne"],
        "traits": ["Duelist", "Trickshot"],
        "items": ["Guinsoo", "InfinityEdge", "LastWhisper"],
        "augments": ["Reroll", "Combat"],
    },
    "bruiser_frontline": {
        "champions": ["Annie", "Darius", "Garen", "Mordekaiser", "Neeko", "Sett", "Taric"],
        "traits": ["Bruiser", "Warden"],
        "items": ["Sunfire", "TitanResolve", "Warmog"],
        "augments": ["Economy", "Trait"],
    },
    "sniper_item_slam": {
        "champions": ["Aphelios", "Ashe", "Ezreal", "Jinx", "Kaisa", "Lux", "Vayne"],
        "traits": ["Sniper", "Sentinel"],
        "items": ["InfinityEdge", "LastWhisper", "RedBuff"],
        "augments": ["Item", "Combat"],
    },
}

ACTION_EFFECTS = {
    "fast_8_carry": {"level_up": 1.0, "hold_economy": 0.35, "roll_down": -0.45, "slam_item": 0.2, "reposition_carry": 0.2},
    "reroll_duelist": {"roll_down": 0.95, "slam_item": 0.4, "level_up": -0.5, "hold_economy": -0.1, "reposition_carry": 0.25},
    "bruiser_frontline": {"hold_economy": 0.6, "reposition_carry": 0.35, "slam_item": 0.2, "level_up": 0.1, "roll_down": -0.2},
    "sniper_item_slam": {"slam_item": 0.9, "reposition_carry": 0.55, "level_up": 0.05, "roll_down": 0.05, "hold_economy": -0.35},
}


def feature_names() -> List[str]:
    return [f"champion:{x}" for x in CHAMPIONS] + [f"trait:{x}" for x in TRAITS] + [f"item:{x}" for x in ITEMS] + [f"augment:{x}" for x in AUGMENTS]


def _multi_hot(selected: List[str], universe: List[str], rng: np.random.Generator, noise_count: int) -> np.ndarray:
    values = np.zeros(len(universe), dtype=np.float64)
    for name in selected:
        if name in universe:
            values[universe.index(name)] = 1.0
    if noise_count > 0:
        noise = rng.choice(len(universe), size=min(noise_count, len(universe)), replace=False)
        values[noise] = np.maximum(values[noise], rng.uniform(0.15, 0.55, size=len(noise)))
    return values


def _board_grid(archetype: str, rng: np.random.Generator) -> np.ndarray:
    grid = np.zeros((4, 7), dtype=np.float64)
    if archetype in {"bruiser_frontline", "fast_8_carry"}:
        frontline = rng.choice(7, size=3, replace=False)
        backline = rng.choice(7, size=2, replace=False)
        grid[0, frontline] = rng.uniform(0.7, 1.0, size=3)
        grid[3, backline] = rng.uniform(0.55, 0.95, size=2)
    else:
        backline = rng.choice(7, size=3, replace=False)
        midline = rng.choice(7, size=2, replace=False)
        grid[3, backline] = rng.uniform(0.7, 1.0, size=3)
        grid[1, midline] = rng.uniform(0.4, 0.75, size=2)
    if rng.random() < 0.25:
        grid[rng.integers(0, 4), rng.integers(0, 7)] = rng.uniform(0.3, 0.8)
    return grid


def _placement_from_score(score: float, rng: np.random.Generator) -> int:
    noisy = score + rng.normal(0, 0.75)
    if noisy >= 2.2:
        return int(rng.choice([1, 2], p=[0.55, 0.45]))
    if noisy >= 1.1:
        return int(rng.choice([2, 3, 4], p=[0.25, 0.45, 0.30]))
    if noisy >= 0.25:
        return int(rng.choice([3, 4, 5], p=[0.25, 0.45, 0.30]))
    if noisy >= -0.7:
        return int(rng.choice([5, 6, 7], p=[0.35, 0.40, 0.25]))
    return int(rng.choice([7, 8], p=[0.45, 0.55]))


def generate_matches(n_matches: int = 1600, seed: int = 7) -> List[MatchRecord]:
    rng = np.random.default_rng(seed)
    names = list(ARCHETYPES)
    records: List[MatchRecord] = []

    for _ in range(n_matches):
        archetype = str(rng.choice(names))
        spec = ARCHETYPES[archetype]
        action_scores = np.array([ACTION_EFFECTS[archetype][action_name] for action_name in ACTIONS])
        action_probs = np.exp(action_scores * 1.35)
        action_probs = action_probs / action_probs.sum()
        action = str(rng.choice(ACTIONS, p=action_probs))

        champion_vec = _multi_hot(spec["champions"], CHAMPIONS, rng, noise_count=2)
        trait_vec = _multi_hot(spec["traits"], TRAITS, rng, noise_count=1)
        item_vec = _multi_hot(spec["items"], ITEMS, rng, noise_count=1)
        augment_vec = _multi_hot(spec["augments"], AUGMENTS, rng, noise_count=1)
        deck_vector = np.concatenate([champion_vec, trait_vec, item_vec, augment_vec])

        board_grid = _board_grid(archetype, rng)
        board_bonus = 0.25 if board_grid[3].max() > 0.75 and archetype in {"reroll_duelist", "sniper_item_slam"} else 0.0
        board_bonus += 0.2 if board_grid[0].sum() > 1.8 and archetype in {"bruiser_frontline", "fast_8_carry"} else 0.0
        archetype_base = {
            "fast_8_carry": 0.65,
            "reroll_duelist": 0.45,
            "bruiser_frontline": 0.15,
            "sniper_item_slam": 0.55,
        }[archetype]
        score = archetype_base + ACTION_EFFECTS[archetype][action] + board_bonus + rng.normal(0, 0.2)
        placement = _placement_from_score(score, rng)

        records.append(MatchRecord(deck_vector, board_grid, action, placement, archetype))

    return records
