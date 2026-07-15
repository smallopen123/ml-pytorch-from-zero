# 5.3｜Dataset、DataLoader 与 CNN

## 1. Dataset 与 DataLoader 的分工

`Dataset` 定义“第 i 个样本如何读取”，`DataLoader` 负责批处理、打乱和并行加载。

```python
from torch.utils.data import Dataset, DataLoader

class TableDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]

loader = DataLoader(TableDataset(X, y), batch_size=32, shuffle=True)
```

只打乱训练集；验证和测试不需要打乱。复杂增广只应用于训练集。

## 2. CNN 为什么适合图像

全连接层忽略空间结构且参数很多。卷积核在图像上滑动，同一组权重检测不同位置的局部模式，具备局部连接和权重共享。

输入通常是 `[N,C,H,W]`：批大小、通道、高、宽。二维卷积输出尺寸：

```text
H_out = floor((H + 2P - K)/S) + 1
```

其中 K 是核大小、S 是步幅、P 是填充。

## 3. 小型 CNN

```python
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, 10)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))
```

`AdaptiveAvgPool2d(1)` 将空间维度变为 1×1，减少对固定输入尺寸的依赖。

## 4. 迁移学习

先使用预训练骨干，将分类头替换为任务类别数。小数据先冻结骨干只训练新头，再用更小学习率微调部分或全部骨干。训练和验证预处理必须与预训练权重要求一致。

## 5. 练习

1. 计算 28×28 图像经过 3×3、步幅 1、填充 1 卷积后的尺寸。
2. 打印每个 CNN 层的输出 shape。
3. 比较冻结骨干与全部微调的训练时间和验证结果。
4. 解释为什么测试增广通常不应包含随机翻转。
