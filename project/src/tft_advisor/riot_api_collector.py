"""Riot API collection interface placeholder.

The demo uses synthetic data so graders can run it without an API key. This
module documents the intended extension point for real match-log collection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol


@dataclass(frozen=True)
class RiotMatchSummary:
    match_id: str
    placement: int
    champions: list[str]
    traits: list[str]
    items: list[str]
    augments: list[str]


class RiotClient(Protocol):
    def match_ids_for_puuid(self, puuid: str, count: int = 20) -> Iterable[str]:
        ...

    def match_summary(self, match_id: str, puuid: str) -> RiotMatchSummary:
        ...


def collection_plan() -> list[str]:
    return [
        "Request a Riot Developer API key.",
        "Collect high-rank TFT player PUUIDs.",
        "Fetch recent match IDs for each PUUID.",
        "Extract champions, traits, items, augments, board state if available, and placement.",
        "Convert match summaries into the same feature schema used by the demo.",
    ]

