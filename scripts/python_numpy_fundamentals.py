"""Python 与 NumPy 基础：同一数据处理分别用循环和向量化实现。"""

from __future__ import annotations

import numpy as np


def standardize_python(values: list[float]) -> list[float]:
    """用纯 Python 计算 z-score；适合理解公式，不适合大数组。"""
    if not values:
        raise ValueError("values 不能为空")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = variance**0.5
    if std == 0:
        return [0.0 for _ in values]
    return [(value - mean) / std for value in values]


def standardize_numpy(values: np.ndarray) -> np.ndarray:
    """向量化实现：运算直接作用于整个数组。"""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("values 不能为空")
    std = values.std()
    return np.zeros_like(values) if std == 0 else (values - values.mean()) / std


def main() -> None:
    features = np.array([[170.0, 65.0], [180.0, 80.0], [160.0, 50.0]])
    weights = np.array([0.3, 0.7])
    scores = features @ weights

    print("shape:", features.shape)
    print("每个样本的线性得分:", scores)
    print("第一列标准化:", standardize_numpy(features[:, 0]))
    print("纯 Python 对照:", standardize_python(features[:, 0].tolist()))


if __name__ == "__main__":
    main()
