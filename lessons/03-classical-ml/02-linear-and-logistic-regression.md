# 3.2｜线性回归与逻辑回归：最重要的两个基线

## 1. 线性回归

多特征线性模型：

```text
ŷ = Xw + b
MSE = (1/n) Σ(ŷ-y)²
∇w = (2/n) Xᵀ(ŷ-y)
```

权重表示在其他特征不变时，该特征增加一个单位对预测的线性影响。特征强相关时，单个权重可能不稳定，不能随意解释为因果关系。

L2 正则化在损失中加入 `λ||w||²`，抑制过大的权重；L1 使用 `λ||w||₁`，可能将部分权重压到零。

## 2. 逻辑回归

逻辑回归先得到线性分数 `z=Xw+b`，再通过 sigmoid 转成概率：

```text
p = 1/(1+e^-z)
BCE = -[y log p + (1-y) log(1-p)]
```

二元交叉熵在正确类别概率很低时惩罚很大。训练后用阈值把概率变为类别；阈值属于决策策略，应在验证集上选择。

## 3. 从零实现

[打开 NumPy 实现](../../scripts/classical_ml_from_scratch.py)。脚本中的两个模型都遵循：初始化权重 → 前向预测 → 计算梯度 → 更新权重。

```powershell
python scripts/classical_ml_from_scratch.py
```

## 4. scikit-learn 对照

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_valid)[:, 1]
```

生产学习中优先使用成熟库；从零实现的目的，是理解库在做什么以及如何检查结果。

## 5. 常见错误

- 忘记标准化，导致梯度下降很慢。
- 将类别标签 0/1 当作回归问题直接最小化 MSE。
- 只看训练损失，不看验证指标。
- 将概率 0.51 当成“非常确定”的正类。

## 6. 练习

1. 在线性回归实现中加入 L2 正则化，注意不要惩罚偏置。
2. 画出 sigmoid 在 `[-10,10]` 的曲线。
3. 分别用 0.3、0.5、0.7 阈值计算混淆矩阵。
4. 比较从零逻辑回归与 scikit-learn 的预测。
