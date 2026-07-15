# 1.2｜NumPy：形状、广播与向量化

## 1. 为什么机器学习需要数组

一个数据集通常表示为矩阵 `X`：每一行是样本，每一列是特征；标签记为向量 `y`。如果 `X.shape == (100, 4)`，表示 100 个样本、每个样本 4 个特征。

```python
import numpy as np

X = np.array([[170, 65], [180, 80], [160, 50]], dtype=float)
y = np.array([0, 1, 0])
print(X.shape)  # (3, 2)
```

## 2. 索引必须同时考虑行和列

```python
first_sample = X[0]      # 第 1 行，shape=(2,)
height = X[:, 0]         # 第 1 列，shape=(3,)
subset = X[:2, :]        # 前 2 个样本
```

`X[0]` 与 `X[0:1]` 数值相同但形状不同：前者是一维 `(2,)`，后者是二维 `(1, 2)`。很多矩阵乘法错误来自无意间丢失维度。

## 3. 广播

```python
means = X.mean(axis=0)   # 每一列的均值，shape=(2,)
centered = X - means     # (3,2) - (2,)；means 自动作用到每一行
```

广播从末尾维度比较：维度相等或其中一个是 1 才兼容。广播很方便，但也可能让错误形状“悄悄运行”，因此训练前要打印关键张量的 shape。

## 4. 矩阵乘法

```python
weights = np.array([0.3, 0.7])
scores = X @ weights     # (3,2) @ (2,) -> (3,)
```

线性模型 `ŷ = Xw + b` 就是一批样本同时做加权求和。`*` 是逐元素乘法，`@` 才是矩阵乘法。

## 5. 标准化与数据泄漏

`z = (x - mean) / std` 让不同量纲的特征处于相近尺度。均值和标准差只能用训练集计算，再用于验证集和测试集；如果用全数据统计量，就提前看到了测试数据。

```python
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)
X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean) / train_std
```

## 6. 练习

1. 创建形状 `(4, 3)` 的数组，分别计算按行和按列均值。
2. 用 `@` 同时计算 4 个样本的线性得分。
3. 制造一次不兼容的广播并解释报错中的两个 shape。
4. 说明为什么不能分别用测试集自己的均值标准化测试集。

## 运行配套代码

```powershell
python scripts/python_numpy_fundamentals.py
```
