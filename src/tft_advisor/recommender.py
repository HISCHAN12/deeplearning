"""Meta-deck clustering and action recommendation logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from .models import Autoencoder, BoardCNNEncoder, FeedForwardPlacementNet
from .synthetic_data import ACTIONS, MatchRecord


def kmeans(x: np.ndarray, n_clusters: int = 4, seed: int = 31, iterations: int = 45) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), size=n_clusters, replace=False)].copy()
    labels = np.zeros(len(x), dtype=int)

    for _ in range(iterations):
        distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        for cluster_id in range(n_clusters):
            members = x[labels == cluster_id]
            if len(members):
                centers[cluster_id] = members.mean(axis=0)
    return labels, centers


@dataclass
class Recommendation:
    cluster_id: int
    action: str
    expected_placement: float
    top4_rate: float
    predicted_placement: int
    placement_probabilities: List[float]


class TFTAdvisor:
    def __init__(self, n_clusters: int = 4) -> None:
        self.n_clusters = n_clusters
        self.autoencoder: Autoencoder | None = None
        self.board_encoder = BoardCNNEncoder()
        self.predictor: FeedForwardPlacementNet | None = None
        self.cluster_centers: np.ndarray | None = None
        self.action_stats: Dict[int, Dict[str, Dict[str, float]]] = {}

    def fit(self, records: Iterable[MatchRecord]) -> Dict[str, float]:
        records = list(records)
        deck_x = np.vstack([record.deck_vector for record in records])
        boards = np.stack([record.board_grid for record in records])
        placements = np.array([record.placement - 1 for record in records], dtype=int)

        self.autoencoder = Autoencoder(input_dim=deck_x.shape[1])
        reconstruction_loss = self.autoencoder.fit(deck_x)
        embeddings = self.autoencoder.encode(deck_x)
        labels, centers = kmeans(embeddings, n_clusters=self.n_clusters)
        self.cluster_centers = centers

        board_features = self.board_encoder.transform(boards)
        predictor_x = np.hstack([embeddings, board_features])
        self.predictor = FeedForwardPlacementNet(input_dim=predictor_x.shape[1])
        classifier_loss = self.predictor.fit(predictor_x, placements)

        probs = self.predictor.predict_proba(predictor_x)
        predicted = probs.argmax(axis=1)
        placement_accuracy = float(np.mean(predicted == placements))
        top4_accuracy = float(np.mean((predicted <= 3) == (placements <= 3)))

        self.action_stats = self._build_action_stats(records, labels)
        return {
            "reconstruction_loss": reconstruction_loss,
            "classifier_loss": classifier_loss,
            "placement_accuracy": placement_accuracy,
            "top4_accuracy": top4_accuracy,
        }

    def recommend(self, deck_vector: np.ndarray, board_grid: np.ndarray) -> Recommendation:
        if self.autoencoder is None or self.predictor is None or self.cluster_centers is None:
            raise RuntimeError("The advisor must be fitted before calling recommend().")

        embedding = self.autoencoder.encode(deck_vector.reshape(1, -1))
        board_features = self.board_encoder.transform(board_grid.reshape(1, 4, 7))
        probs = self.predictor.predict_proba(np.hstack([embedding, board_features]))[0]
        predicted_placement = int(probs.argmax() + 1)

        distances = ((self.cluster_centers - embedding[0]) ** 2).sum(axis=1)
        cluster_id = int(distances.argmin())
        cluster_stats = self.action_stats[cluster_id]
        best_action = min(ACTIONS, key=lambda action: cluster_stats[action]["expected_placement"])
        stats = cluster_stats[best_action]

        return Recommendation(
            cluster_id=cluster_id,
            action=best_action,
            expected_placement=stats["expected_placement"],
            top4_rate=stats["top4_rate"],
            predicted_placement=predicted_placement,
            placement_probabilities=[float(value) for value in probs],
        )

    def _build_action_stats(self, records: List[MatchRecord], labels: np.ndarray) -> Dict[int, Dict[str, Dict[str, float]]]:
        stats: Dict[int, Dict[str, Dict[str, float]]] = {}
        global_placements = np.array([record.placement for record in records], dtype=np.float64)
        global_default = {
            "expected_placement": float(global_placements.mean()),
            "top4_rate": float(np.mean(global_placements <= 4)),
        }

        for cluster_id in range(self.n_clusters):
            stats[cluster_id] = {}
            cluster_records = [record for record, label in zip(records, labels) if label == cluster_id]
            for action in ACTIONS:
                placements = np.array([record.placement for record in cluster_records if record.action == action], dtype=np.float64)
                if len(placements) == 0:
                    stats[cluster_id][action] = dict(global_default)
                else:
                    stats[cluster_id][action] = {
                        "expected_placement": float(placements.mean()),
                        "top4_rate": float(np.mean(placements <= 4)),
                    }
        return stats

