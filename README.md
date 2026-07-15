# 从零学习机器学习、深度学习与 PyTorch

一条面向零基础学习者的中文实践路线：先补 Python、NumPy 和必要数学，再学习传统机器学习，随后进入神经网络与 PyTorch，最后通过开源项目完成从“会调用”到“能解释、能修改、能复现”的过渡。

> 建议投入：24 周，每周 6–8 小时。不要同时刷完所有高星教程；主线每阶段只选一个，其他资源用于查漏补缺。

## 🚀 立即开始：按这个顺序点击

1. [第 0 步：环境准备与学习方法](https://github.com/smallopen123/ml-pytorch-from-zero/blob/main/lessons/00-start-here/README.md)
2. [第 1 课：机器学习到底在做什么](https://github.com/smallopen123/ml-pytorch-from-zero/blob/main/lessons/00-start-here/01-what-is-ml.md)
3. [运行第一个从零实现：线性回归](https://github.com/smallopen123/ml-pytorch-from-zero/blob/main/scripts/linear_regression_from_scratch.py)
4. [查看完整 24 周学习路线](https://github.com/smallopen123/ml-pytorch-from-zero/blob/main/ROADMAP.md)
5. [复制每周学习记录模板](https://github.com/smallopen123/ml-pytorch-from-zero/blob/main/PROGRESS.md)

如果你今天第一次开始，只需要完成前两项；不要先安装 GPU/CUDA，也不要同时打开十套教程。

## 📚 分阶段直达

| 阶段 | 学习内容 | 直达入口 |
|---|---|---|
| 0 | 准备、基本概念与第一课 | [立即学习](https://github.com/smallopen123/ml-pytorch-from-zero/tree/main/lessons/00-start-here) |
| 1 | Python、NumPy 与数据工具 | [进入阶段 1](https://github.com/smallopen123/ml-pytorch-from-zero/blob/main/lessons/01-python-numpy/README.md) |
| 2 | 机器学习所需数学 | [进入阶段 2](https://github.com/smallopen123/ml-pytorch-from-zero/blob/main/lessons/02-math-foundations/README.md) |
| 3 | 传统机器学习算法 | [进入阶段 3](https://github.com/smallopen123/ml-pytorch-from-zero/blob/main/lessons/03-classical-ml/README.md) |
| 4 | 深度学习原理 | [进入阶段 4](https://github.com/smallopen123/ml-pytorch-from-zero/blob/main/lessons/04-deep-learning/README.md) |
| 5 | PyTorch 实战 | [进入阶段 5](https://github.com/smallopen123/ml-pytorch-from-zero/blob/main/lessons/05-pytorch/README.md) |
| 项目 | 7 级递进项目 | [查看项目阶梯](https://github.com/smallopen123/ml-pytorch-from-zero/blob/main/projects/README.md) |
| 资源 | 高星教程与项目精选 | [查看资源索引](https://github.com/smallopen123/ml-pytorch-from-zero/blob/main/resources/README.md) |

## 🌐 外部主线教材直达

- 传统机器学习：[Microsoft ML for Beginners 中文目录](https://github.com/microsoft/ML-For-Beginners/tree/main/translations/zh-CN)
- 深度学习中文主线：[《动手学深度学习》在线阅读](https://zh.d2l.ai/)
- PyTorch 零基础实战：[Learn PyTorch 在线教程](https://www.learnpytorch.io/)
- PyTorch 权威用法：[PyTorch 官方教程](https://docs.pytorch.org/tutorials/)

## 你将学会什么

- 用 Python、NumPy、Pandas 和 Matplotlib 完成数据处理与可视化
- 理解线性代数、概率、微积分中与机器学习直接相关的概念
- 从零实现线性回归、逻辑回归、KNN、决策树、聚类和基础神经网络
- 正确使用训练集、验证集、测试集，以及常见评价指标
- 使用 PyTorch 的张量、自动微分、`nn.Module`、`Dataset` 和 `DataLoader`
- 完成图像分类、表格分类和文本分类项目，并阅读优秀开源代码

## 从这里开始

1. 阅读 [学习方法与环境准备](lessons/00-start-here/README.md)。
2. 按 [24 周路线图](ROADMAP.md) 学习，不跳过传统机器学习与数据评估。
3. 运行环境检查：`python scripts/check_environment.py`。
4. 完成 [第一课：机器学习到底在做什么](lessons/00-start-here/01-what-is-ml.md)。
5. 每周在 [学习记录模板](PROGRESS.md) 中记录实验、错误和复盘。

## 仓库导航

- [开始学习](https://github.com/smallopen123/ml-pytorch-from-zero/tree/main/lessons/00-start-here)：概念、工具和学习方法
- [Python 与 NumPy](https://github.com/smallopen123/ml-pytorch-from-zero/tree/main/lessons/01-python-numpy)：Python 与科学计算
- [数学基础](https://github.com/smallopen123/ml-pytorch-from-zero/tree/main/lessons/02-math-foundations)：只学真正用得到的数学
- [传统机器学习](https://github.com/smallopen123/ml-pytorch-from-zero/tree/main/lessons/03-classical-ml)：传统机器学习算法与评估
- [深度学习](https://github.com/smallopen123/ml-pytorch-from-zero/tree/main/lessons/04-deep-learning)：神经网络、反向传播和优化
- [PyTorch](https://github.com/smallopen123/ml-pytorch-from-zero/tree/main/lessons/05-pytorch)：PyTorch 工程化训练流程
- [项目阶梯](https://github.com/smallopen123/ml-pytorch-from-zero/tree/main/projects)：递进式项目和验收标准
- [资源索引](https://github.com/smallopen123/ml-pytorch-from-zero/tree/main/resources)：按星标、适用阶段和维护状态整理的外部资源

## 学习原则

每个主题都走四遍：直觉解释 → 手算/画图 → NumPy 从零实现 → 框架实现。代码跑通不等于学会；你还需要回答“输入输出是什么、损失为何这样定义、数据泄漏在哪里、指标为什么合适”。

## 当前进度

核心课程已经包含 Python/NumPy、必要数学、传统机器学习、神经网络反向传播、PyTorch 训练流程和端到端表格分类项目。配套提供 7 个可运行脚本与自动测试；更高级的 CNN、NLP 和开源复现项目将继续按路线图扩展。

## 许可证

本仓库原创内容采用 [MIT License](LICENSE)。外部教程和项目只提供链接、元数据与学习建议，其版权和许可证归各自作者所有。
