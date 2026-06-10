"""Small NumPy neural models used by the TFT advisor prototype."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass
class Autoencoder:
    input_dim: int
    latent_dim: int = 8
    hidden_dim: int = 24
    lr: float = 0.03
    weight_decay: float = 1e-4
    seed: int = 11

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.w1 = rng.normal(0, 0.12, size=(self.input_dim, self.hidden_dim))
        self.b1 = np.zeros(self.hidden_dim)
        self.w2 = rng.normal(0, 0.12, size=(self.hidden_dim, self.latent_dim))
        self.b2 = np.zeros(self.latent_dim)
        self.w3 = rng.normal(0, 0.12, size=(self.latent_dim, self.hidden_dim))
        self.b3 = np.zeros(self.hidden_dim)
        self.w4 = rng.normal(0, 0.12, size=(self.hidden_dim, self.input_dim))
        self.b4 = np.zeros(self.input_dim)

    def encode(self, x: np.ndarray) -> np.ndarray:
        h1 = relu(x @ self.w1 + self.b1)
        return h1 @ self.w2 + self.b2

    def fit(self, x: np.ndarray, epochs: int = 90, batch_size: int = 64) -> float:
        rng = np.random.default_rng(self.seed + 1)
        n = len(x)
        last_loss = 0.0
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                batch = x[order[start : start + batch_size]]
                m = len(batch)

                z1 = batch @ self.w1 + self.b1
                h1 = relu(z1)
                z2 = h1 @ self.w2 + self.b2
                z3 = z2 @ self.w3 + self.b3
                h3 = relu(z3)
                out = h3 @ self.w4 + self.b4

                diff = out - batch
                last_loss = float(np.mean(diff * diff))
                grad_out = (2.0 / m) * diff / self.input_dim

                gw4 = h3.T @ grad_out + self.weight_decay * self.w4
                gb4 = grad_out.sum(axis=0)
                gh3 = grad_out @ self.w4.T
                gz3 = gh3 * (z3 > 0)
                gw3 = z2.T @ gz3 + self.weight_decay * self.w3
                gb3 = gz3.sum(axis=0)
                gz2 = gz3 @ self.w3.T
                gw2 = h1.T @ gz2 + self.weight_decay * self.w2
                gb2 = gz2.sum(axis=0)
                gh1 = gz2 @ self.w2.T
                gz1 = gh1 * (z1 > 0)
                gw1 = batch.T @ gz1 + self.weight_decay * self.w1
                gb1 = gz1.sum(axis=0)

                self.w4 -= self.lr * gw4
                self.b4 -= self.lr * gb4
                self.w3 -= self.lr * gw3
                self.b3 -= self.lr * gb3
                self.w2 -= self.lr * gw2
                self.b2 -= self.lr * gb2
                self.w1 -= self.lr * gw1
                self.b1 -= self.lr * gb1
        return last_loss


class BoardCNNEncoder:
    """Convolution-style fixed feature extractor for a 4x7 TFT board."""

    def __init__(self) -> None:
        self.kernels = np.array(
            [
                [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
                [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
                [[0, 1, 0], [1, 2, 1], [0, 1, 0]],
            ],
            dtype=np.float64,
        )

    def transform(self, boards: np.ndarray) -> np.ndarray:
        features = []
        padded = np.pad(boards, ((0, 0), (1, 1), (1, 1)), mode="constant")
        for board_index in range(len(boards)):
            board_features = []
            for kernel in self.kernels:
                responses = []
                for row in range(4):
                    for col in range(7):
                        patch = padded[board_index, row : row + 3, col : col + 3]
                        responses.append(float(np.sum(patch * kernel)))
                responses_array = np.array(responses)
                board_features.extend([responses_array.mean(), responses_array.max(), responses_array.std()])
            board_features.extend([boards[board_index, 0].sum(), boards[board_index, 3].sum(), boards[board_index].max()])
            features.append(board_features)
        return np.array(features, dtype=np.float64)


@dataclass
class FeedForwardPlacementNet:
    input_dim: int
    hidden_dim: int = 32
    output_dim: int = 8
    lr: float = 0.04
    weight_decay: float = 5e-4
    seed: int = 19

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.w1 = rng.normal(0, 0.16, size=(self.input_dim, self.hidden_dim))
        self.b1 = np.zeros(self.hidden_dim)
        self.w2 = rng.normal(0, 0.16, size=(self.hidden_dim, self.output_dim))
        self.b2 = np.zeros(self.output_dim)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        h = relu(x @ self.w1 + self.b1)
        return softmax(h @ self.w2 + self.b2)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 120, batch_size: int = 64) -> float:
        rng = np.random.default_rng(self.seed + 1)
        n = len(x)
        last_loss = 0.0
        y_onehot = np.eye(self.output_dim)[y]

        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                xb = x[idx]
                yb = y_onehot[idx]
                m = len(xb)

                z1 = xb @ self.w1 + self.b1
                h1 = relu(z1)
                logits = h1 @ self.w2 + self.b2
                probs = softmax(logits)
                last_loss = float(-np.mean(np.sum(yb * np.log(probs + 1e-8), axis=1)))

                grad_logits = (probs - yb) / m
                gw2 = h1.T @ grad_logits + self.weight_decay * self.w2
                gb2 = grad_logits.sum(axis=0)
                gh1 = grad_logits @ self.w2.T
                gz1 = gh1 * (z1 > 0)
                gw1 = xb.T @ gz1 + self.weight_decay * self.w1
                gb1 = gz1.sum(axis=0)

                self.w2 -= self.lr * gw2
                self.b2 -= self.lr * gb2
                self.w1 -= self.lr * gw1
                self.b1 -= self.lr * gb1
        return last_loss


def train_test_split(x: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, seed: int = 23) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    test_size = int(len(x) * test_ratio)
    test_idx = order[:test_size]
    train_idx = order[test_size:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]

