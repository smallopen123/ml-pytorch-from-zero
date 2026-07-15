# 5.2｜标准训练循环：每一行为什么存在

## 1. 五个组件

训练最小系统包括：数据、模型、损失函数、优化器、评价指标。二分类常用一个输出 logit 与 `BCEWithLogitsLoss`。

```python
model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

## 2. 一次训练步骤

```python
model.train()
for features, targets in train_loader:
    logits = model(features)          # 前向传播
    loss = loss_fn(logits, targets)   # 标量目标
    optimizer.zero_grad()             # 清除上一批梯度
    loss.backward()                   # 计算本批梯度
    optimizer.step()                  # 更新参数
```

顺序不能随意打乱。`zero_grad()` 放在前向前或 backward 前都可以，但必须在新梯度累计前执行。

## 3. 验证循环

```python
model.eval()
total_correct = 0
with torch.inference_mode():
    for features, targets in valid_loader:
        logits = model(features)
        predictions = (torch.sigmoid(logits) >= 0.5).float()
        total_correct += (predictions == targets).sum().item()
```

验证不能调用 `backward()` 和 `step()`。损失应按样本数加权汇总，避免最后一个较小批次权重过大。

## 4. 可复现与保存

固定 Python、NumPy、PyTorch 随机种子，记录包版本、数据划分和超参数。保存 `state_dict` 而非整个模型对象：

```python
torch.save({
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "epoch": epoch,
}, "checkpoint.pt")
```

加载时要先用相同结构创建模型，再调用 `load_state_dict`。

## 5. 完整脚本

[打开可运行训练模板](../../scripts/pytorch_training_template.py)

```powershell
python scripts/pytorch_training_template.py
```

## 6. 调试顺序

先尝试让模型过拟合 20 个样本。如果做不到，优先检查标签、shape、损失/输出组合、梯度是否存在、学习率和数据预处理；不要先换更大的模型。

## 7. 练习

1. 把优化器换成 SGD，比较收敛速度。
2. 删除 `zero_grad()`，观察训练。
3. 添加验证集损失，并保存验证损失最低的模型。
4. 打印每层参数梯度范数，检查是否为零或爆炸。
