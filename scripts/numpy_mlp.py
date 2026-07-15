"""两层 NumPy 神经网络：在 XOR 数据上演示前向传播和反向传播。"""

from __future__ import annotations

import numpy as np


class BinaryMLP:
    def __init__(self, input_size: int = 2, hidden_size: int = 8, learning_rate: float = 0.5, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 0.5, size=(input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = rng.normal(0, 0.5, size=(hidden_size, 1))
        self.b2 = np.zeros((1, 1))
        self.learning_rate = learning_rate
        self.losses: list[float] = []

    @staticmethod
    def sigmoid(values: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(values, -500, 500)))

    def forward(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden = np.tanh(features @ self.w1 + self.b1)
        probabilities = self.sigmoid(hidden @ self.w2 + self.b2)
        return hidden, probabilities

    def fit(self, features: np.ndarray, targets: np.ndarray, epochs: int = 10_000) -> "BinaryMLP":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64).reshape(-1, 1)
        for _ in range(epochs):
            hidden, probabilities = self.forward(x)
            clipped = np.clip(probabilities, 1e-8, 1 - 1e-8)
            self.losses.append(float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))))

            output_delta = (probabilities - y) / len(x)
            w2_gradient = hidden.T @ output_delta
            b2_gradient = output_delta.sum(axis=0, keepdims=True)
            hidden_delta = (output_delta @ self.w2.T) * (1 - hidden**2)
            w1_gradient = x.T @ hidden_delta
            b1_gradient = hidden_delta.sum(axis=0, keepdims=True)

            self.w2 -= self.learning_rate * w2_gradient
            self.b2 -= self.learning_rate * b2_gradient
            self.w1 -= self.learning_rate * w1_gradient
            self.b1 -= self.learning_rate * b1_gradient
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.forward(np.asarray(features, dtype=np.float64))[1].ravel()

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.predict_proba(features) >= 0.5).astype(int)


def main() -> None:
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([0, 1, 1, 0])
    model = BinaryMLP().fit(x, y)
    print("概率:", np.round(model.predict_proba(x), 3))
    print("类别:", model.predict(x))
    print(f"损失: {model.losses[0]:.4f} -> {model.losses[-1]:.4f}")


if __name__ == "__main__":
    main()
