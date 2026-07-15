# 5.1｜Tensor 与 autograd

## 1. Tensor 是什么

Tensor 是支持 CPU/GPU、自动微分和神经网络算子的多维数组。与 NumPy 类似，最重要的属性是 shape、dtype 和 device。

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x.shape, x.dtype, x.device)
x = x.to("cuda" if torch.cuda.is_available() else "cpu")
```

模型参数、输入和标签必须位于同一 device。分类标签的数据类型也很重要：多分类交叉熵通常需要 `torch.long` 类别索引。

## 2. 广播与矩阵乘法

```python
weights = torch.randn(2, 1)
bias = torch.zeros(1)
logits = x @ weights + bias
```

规则与 NumPy 接近。遇到错误时打印 `x.shape`、`weights.shape`、`targets.shape`，不要靠反复增删 `squeeze()` 猜。

## 3. 自动微分

```python
w = torch.tensor(2.0, requires_grad=True)
loss = (w * 3 - 10) ** 2
loss.backward()
print(w.grad)
```

PyTorch 在前向运算时构建计算图，`backward()` 从标量损失反向计算梯度。梯度会累加，因此训练循环每一步都要清零。

## 4. 禁用梯度

验证和推理不更新参数：

```python
model.eval()
with torch.inference_mode():
    predictions = model(x_valid)
```

`model.eval()` 切换 Dropout/BatchNorm 行为；`inference_mode()` 关闭梯度记录。两者职责不同，通常一起使用。

## 5. 常见陷阱

- 对带梯度 Tensor 直接 `.numpy()`；应先 `.detach().cpu().numpy()`。
- 原地修改计算图需要的值。
- 忘记 `optimizer.zero_grad()`。
- 输入在 GPU、模型仍在 CPU。
- 用概率喂给要求 logits 的损失函数。

## 6. 练习

1. 手算上例的 `d(loss)/dw`，与 `w.grad` 对比。
2. 构造 `(32,10) @ (10,3)` 并说明输出 shape。
3. 连续调用两次 `backward()` 前后观察梯度累加。
4. 解释 `train()`/`eval()` 与是否计算梯度的区别。
