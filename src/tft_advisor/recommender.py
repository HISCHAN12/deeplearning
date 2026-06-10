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
    expected_placement_improvement: float
    top4_rate: float
    predicted_top4_probability: float
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

    def fit(self, records: Iterable[MatchRecord], test_ratio: float = 0.2, seed: int = 23) -> Dict[str, float]:
        records = list(records)
        if len(records) < 2:
            raise ValueError("At least two records are required to train and evaluate the advisor.")

        rng = np.random.default_rng(seed)
        order = rng.permutation(len(records))
        test_size = max(1, int(len(records) * test_ratio))
        test_indices = order[:test_size]
        train_indices = order[test_size:]
        if len(train_indices) == 0:
            raise ValueError("test_ratio leaves no training records.")

        train_records = [records[index] for index in train_indices]
        test_records = [records[index] for index in test_indices]

        train_decks = np.vstack([record.deck_vector for record in train_records])
        train_boards = np.stack([record.board_grid for record in train_records])
        train_placements = np.array([record.placement - 1 for record in train_records], dtype=int)
        test_decks = np.vstack([record.deck_vector for record in test_records])
        test_boards = np.stack([record.board_grid for record in test_records])
        test_placements = np.array([record.placement - 1 for record in test_records], dtype=int)

        self.autoencoder = Autoencoder(input_dim=train_decks.shape[1])
        reconstruction_loss = self.autoencoder.fit(train_decks)
        train_embeddings = self.autoencoder.encode(train_decks)
        labels, centers = kmeans(train_embeddings, n_clusters=self.n_clusters)
        self.cluster_centers = centers

        train_board_features = self.board_encoder.transform(train_boards)
        train_x = np.hstack([train_embeddings, train_board_features])
        self.predictor = FeedForwardPlacementNet(input_dim=train_x.shape[1])
        classifier_loss = self.predictor.fit(train_x, train_placements)

        test_embeddings = self.autoencoder.encode(test_decks)
        test_board_features = self.board_encoder.transform(test_boards)
        test_x = np.hstack([test_embeddings, test_board_features])
        probs = self.predictor.predict_proba(test_x)
        predicted = probs.argmax(axis=1)
        placement_accuracy = float(np.mean(predicted == test_placements))
        top4_probabilities = probs[:, :4].sum(axis=1)
        top4_accuracy = float(np.mean((top4_probabilities >= 0.5) == (test_placements <= 3)))

        self.action_stats = self._build_action_stats(train_records, labels)
        return {
            "reconstruction_loss": reconstruction_loss,
            "classifier_loss": classifier_loss,
            "placement_accuracy": placement_accuracy,
            "top4_accuracy": top4_accuracy,
            "train_samples": float(len(train_records)),
            "test_samples": float(len(test_records)),
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
            expected_placement_improvement=stats["expected_placement_improvement"],
            top4_rate=stats["top4_rate"],
            predicted_top4_probability=float(probs[:4].sum()),
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
            cluster_placements = np.array([record.placement for record in cluster_records], dtype=np.float64)
            cluster_baseline = float(cluster_placements.mean()) if len(cluster_placements) else global_default["expected_placement"]
            for action in ACTIONS:
                placements = np.array([record.placement for record in cluster_records if record.action == action], dtype=np.float64)
                if len(placements) == 0:
                    action_stats = dict(global_default)
                else:
                    action_stats = {
                        "expected_placement": float(placements.mean()),
                        "top4_rate": float(np.mean(placements <= 4)),
                    }
                action_stats["expected_placement_improvement"] = cluster_baseline - action_stats["expected_placement"]
                action_stats["support"] = float(len(placements))
                stats[cluster_id][action] = action_stats
        return stats

