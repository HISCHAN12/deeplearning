"""Meta-deck clustering and action recommendation logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from .models import Autoencoder, BoardCNNEncoder, FeedForwardPlacementNet
from .game_state import neutral_state_vector
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
    reason: str


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
        train_states = np.vstack([record.state_vector for record in train_records])
        train_placements = np.array([record.placement - 1 for record in train_records], dtype=int)
        test_decks = np.vstack([record.deck_vector for record in test_records])
        test_boards = np.stack([record.board_grid for record in test_records])
        test_states = np.vstack([record.state_vector for record in test_records])
        test_placements = np.array([record.placement - 1 for record in test_records], dtype=int)

        self.autoencoder = Autoencoder(input_dim=train_decks.shape[1])
        reconstruction_loss = self.autoencoder.fit(train_decks)
        train_embeddings = self.autoencoder.encode(train_decks)
        labels, centers = kmeans(train_embeddings, n_clusters=self.n_clusters)
        self.cluster_centers = centers

        train_board_features = self.board_encoder.transform(train_boards)
        train_x = np.hstack([train_embeddings, train_board_features, train_states])
        self.predictor = FeedForwardPlacementNet(input_dim=train_x.shape[1])
        classifier_loss = self.predictor.fit(train_x, train_placements)

        test_embeddings = self.autoencoder.encode(test_decks)
        test_board_features = self.board_encoder.transform(test_boards)
        test_x = np.hstack([test_embeddings, test_board_features, test_states])
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

    def recommend(
        self,
        deck_vector: np.ndarray,
        board_grid: np.ndarray,
        state_vector: np.ndarray | None = None,
    ) -> Recommendation:
        if self.autoencoder is None or self.predictor is None or self.cluster_centers is None:
            raise RuntimeError("The advisor must be fitted before calling recommend().")

        embedding = self.autoencoder.encode(deck_vector.reshape(1, -1))
        board_features = self.board_encoder.transform(board_grid.reshape(1, 4, 7))
        state = neutral_state_vector() if state_vector is None else state_vector
        probs = self.predictor.predict_proba(
            np.hstack([embedding, board_features, state.reshape(1, -1)])
        )[0]
        predicted_placement = int(probs.argmax() + 1)

        distances = ((self.cluster_centers - embedding[0]) ** 2).sum(axis=1)
        cluster_id = int(distances.argmin())
        cluster_stats = self.action_stats[cluster_id]
        action_adjustments, reasons = self._state_action_adjustments(state)
        best_action = min(
            ACTIONS,
            key=lambda action: cluster_stats[action]["expected_placement"] - action_adjustments[action],
        )
        stats = cluster_stats[best_action]
        adjusted_placement = max(1.0, stats["expected_placement"] - action_adjustments[best_action])

        return Recommendation(
            cluster_id=cluster_id,
            action=best_action,
            expected_placement=adjusted_placement,
            expected_placement_improvement=stats["expected_placement_improvement"] + action_adjustments[best_action],
            top4_rate=stats["top4_rate"],
            predicted_top4_probability=float(probs[:4].sum()),
            predicted_placement=predicted_placement,
            placement_probabilities=[float(value) for value in probs],
            reason=reasons[best_action],
        )

    def _state_action_adjustments(self, state: np.ndarray) -> tuple[Dict[str, float], Dict[str, str]]:
        health = float(state[1] * 100)
        gold = float(state[2] * 100)
        level = float(state[3] * 10)
        streak = float(state[4] * 5)
        unspent_items = float(state[6] * 6)
        health_delta = float(state[7] * 30)
        board_delta = float(state[10] * 5)

        adjustments = {action: 0.0 for action in ACTIONS}
        reasons = {
            "level_up": "현재 군집에서 레벨 업의 평균 성적이 가장 좋습니다.",
            "roll_down": "현재 군집에서 리롤의 평균 성적이 가장 좋습니다.",
            "hold_economy": "현재 군집에서 골드 유지의 평균 성적이 가장 좋습니다.",
            "slam_item": "현재 군집에서 아이템 장착의 평균 성적이 가장 좋습니다.",
            "reposition_carry": "현재 군집에서 캐리 위치 변경의 평균 성적이 가장 좋습니다.",
        }

        if health <= 35 or health_delta <= -15:
            adjustments["roll_down"] += 0.95
            adjustments["hold_economy"] -= 0.55
            reasons["roll_down"] = "체력이 낮거나 최근 체력 손실이 커서 즉시 보드 강화가 필요합니다."
        if gold >= 50 and health >= 55:
            adjustments["hold_economy"] += 0.65
            reasons["hold_economy"] = "체력이 안정적이고 50골드 이상이라 이자 운영 가치가 높습니다."
        if gold >= 30 and level <= 7:
            adjustments["level_up"] += 0.55
            reasons["level_up"] = "레벨이 낮고 사용할 골드가 있어 인구수 확장이 유리합니다."
        if unspent_items >= 3:
            adjustments["slam_item"] += 0.8
            reasons["slam_item"] = "대기 중인 아이템이 많아 즉시 전투력으로 전환할 가치가 높습니다."
        if streak <= -2 or board_delta < 0:
            adjustments["reposition_carry"] += 0.5
            reasons["reposition_carry"] = "연패 중이거나 보드 강도가 하락해 배치 조정이 필요합니다."

        return adjustments, reasons

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

