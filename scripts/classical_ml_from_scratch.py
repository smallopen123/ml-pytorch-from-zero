"""使用 NumPy 从零实现线性回归、逻辑回归、KNN 和 K-Means。"""

from __future__ import annotations

import numpy as np


def add_bias(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("features 必须是二维数组 [样本数, 特征数]")
    return np.c_[np.ones(features.shape[0]), features]


class LinearRegressionGD:
    def __init__(self, learning_rate: float = 0.05, epochs: int = 2_000) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: np.ndarray | None = None
        self.losses: list[float] = []

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "LinearRegressionGD":
        x = add_bias(features)
        y = np.asarray(targets, dtype=np.float64)
        self.weights = np.zeros(x.shape[1])
        for _ in range(self.epochs):
            errors = x @ self.weights - y
            self.losses.append(float(np.mean(errors**2)))
            gradient = 2 * x.T @ errors / len(x)
            self.weights -= self.learning_rate * gradient
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("请先调用 fit")
        return add_bias(features) @ self.weights


class LogisticRegressionGD:
    def __init__(self, learning_rate: float = 0.1, epochs: int = 3_000) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: np.ndarray | None = None

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        values = np.clip(values, -500, 500)
        return 1 / (1 + np.exp(-values))

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "LogisticRegressionGD":
        x = add_bias(features)
        y = np.asarray(targets, dtype=np.float64)
        self.weights = np.zeros(x.shape[1])
        for _ in range(self.epochs):
            probabilities = self._sigmoid(x @ self.weights)
            gradient = x.T @ (probabilities - y) / len(x)
            self.weights -= self.learning_rate * gradient
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("请先调用 fit")
        return self._sigmoid(add_bias(features) @ self.weights)

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(features) >= threshold).astype(int)


class KNNClassifier:
    def __init__(self, neighbors: int = 3) -> None:
        if neighbors <= 0:
            raise ValueError("neighbors 必须大于 0")
        self.neighbors = neighbors
        self.features: np.ndarray | None = None
        self.targets: np.ndarray | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "KNNClassifier":
        self.features = np.asarray(features, dtype=np.float64)
        self.targets = np.asarray(targets)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.features is None or self.targets is None:
            raise RuntimeError("请先调用 fit")
        predictions = []
        for sample in np.asarray(features, dtype=np.float64):
            distances = np.linalg.norm(self.features - sample, axis=1)
            indices = np.argsort(distances)[: self.neighbors]
            labels, counts = np.unique(self.targets[indices], return_counts=True)
            predictions.append(labels[np.argmax(counts)])
        return np.asarray(predictions)


class KMeans:
    def __init__(self, clusters: int = 2, max_iterations: int = 100, seed: int = 42) -> None:
        self.clusters = clusters
        self.max_iterations = max_iterations
        self.seed = seed
        self.centers: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "KMeans":
        x = np.asarray(features, dtype=np.float64)
        if self.clusters > len(x):
            raise ValueError("聚类数不能超过样本数")
        rng = np.random.default_rng(self.seed)
        self.centers = x[rng.choice(len(x), self.clusters, replace=False)].copy()
        for _ in range(self.max_iterations):
            labels = self.predict(x)
            new_centers = np.array([x[labels == index].mean(axis=0) for index in range(self.clusters)])
            if np.allclose(new_centers, self.centers):
                break
            self.centers = new_centers
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.centers is None:
            raise RuntimeError("请先调用 fit")
        distances = np.linalg.norm(np.asarray(features)[:, None, :] - self.centers[None, :, :], axis=2)
        return np.argmin(distances, axis=1)


def main() -> None:
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    regression = LinearRegressionGD().fit(x, np.array([1.0, 3.0, 5.0, 7.0]))
    print("线性回归预测 x=4:", regression.predict(np.array([[4.0]]))[0])

    classes_x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    classifier = LogisticRegressionGD().fit(classes_x, np.array([0, 0, 1, 1]))
    print("逻辑回归概率:", classifier.predict_proba(np.array([[-0.5], [0.5]])))


if __name__ == "__main__":
    main()
