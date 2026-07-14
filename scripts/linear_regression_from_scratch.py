"""只用 Python 列表实现一元线性回归梯度下降。"""

from __future__ import annotations


def fit_line(
    xs: list[float],
    ys: list[float],
    learning_rate: float = 0.01,
    epochs: int = 2_000,
) -> tuple[float, float, list[float]]:
    if len(xs) != len(ys) or not xs:
        raise ValueError("xs 和 ys 必须非空且长度相同")

    weight = 0.0
    bias = 0.0
    losses: list[float] = []
    sample_count = len(xs)

    for _ in range(epochs):
        predictions = [weight * x + bias for x in xs]
        errors = [prediction - target for prediction, target in zip(predictions, ys)]
        loss = sum(error**2 for error in errors) / sample_count
        losses.append(loss)

        weight_gradient = 2 * sum(error * x for error, x in zip(errors, xs)) / sample_count
        bias_gradient = 2 * sum(errors) / sample_count
        weight -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient

    return weight, bias, losses


def main() -> None:
    hours = [1.0, 2.0, 3.0, 4.0, 5.0]
    scores = [52.0, 57.0, 63.0, 68.0, 73.0]
    weight, bias, losses = fit_line(hours, scores)
    prediction = weight * 6 + bias

    print(f"模型: score = {weight:.2f} * hours + {bias:.2f}")
    print(f"初始损失: {losses[0]:.4f}")
    print(f"最终损失: {losses[-1]:.4f}")
    print(f"学习 6 小时的预测分数: {prediction:.1f}")


if __name__ == "__main__":
    main()
