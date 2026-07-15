"""用有限差分验证均方误差的解析梯度。"""

from __future__ import annotations

import numpy as np


def mse_loss(weight: float, xs: np.ndarray, ys: np.ndarray) -> float:
    predictions = weight * xs
    return float(np.mean((predictions - ys) ** 2))


def analytical_gradient(weight: float, xs: np.ndarray, ys: np.ndarray) -> float:
    return float(2 * np.mean((weight * xs - ys) * xs))


def numerical_gradient(weight: float, xs: np.ndarray, ys: np.ndarray, epsilon: float = 1e-5) -> float:
    """中心差分：(f(w+e)-f(w-e))/(2e)。"""
    return (mse_loss(weight + epsilon, xs, ys) - mse_loss(weight - epsilon, xs, ys)) / (2 * epsilon)


def main() -> None:
    xs = np.array([1.0, 2.0, 3.0])
    ys = np.array([2.0, 4.0, 6.0])
    weight = 0.5
    exact = analytical_gradient(weight, xs, ys)
    approximate = numerical_gradient(weight, xs, ys)
    print(f"解析梯度: {exact:.8f}")
    print(f"数值梯度: {approximate:.8f}")
    print(f"绝对误差: {abs(exact - approximate):.2e}")


if __name__ == "__main__":
    main()
