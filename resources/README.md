# 高星教程与开源项目索引

数据快照：**2026-07-14**。星标会变化；下表按本次 GitHub API 查询结果排序。这里的“排名”是在经过“相关、可公开访问、对学习有价值”筛选后的候选集中排名，不宣称是 GitHub 全站的绝对榜单。

## 零基础与传统机器学习教程

| 排名 | 仓库 | Stars | 最适合 | 建议 |
|---:|---|---:|---|---|
| 1 | [microsoft/ML-For-Beginners](https://github.com/microsoft/ML-For-Beginners) | 88,120 | 零基础传统 ML 主线 | 12 周课程结构完整，优先学习 |
| 2 | [eriklindernoren/ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch) | 32,259 | 算法从零实现 | 学完概念后对照代码，不作为第一本教材 |
| 3 | [trekhleb/homemade-machine-learning](https://github.com/trekhleb/homemade-machine-learning) | 24,686 | 直观算法实现 | 用于补充可视化和实现细节 |
| 4 | [ageron/handson-ml3](https://github.com/ageron/handson-ml3) | 13,714 | 端到端实践 | 配合对应图书使用，注意内容许可 |

## 深度学习与 PyTorch 教程

| 排名 | 仓库 | Stars | 最适合 | 建议 |
|---:|---|---:|---|---|
| 1 | [d2l-ai/d2l-zh](https://github.com/d2l-ai/d2l-zh) | 78,883 | 中文深度学习主线 | 理论与代码兼顾，阶段 4 开始使用 |
| 2 | [yunjey/pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial) | 32,409 | 简洁 PyTorch 示例 | 部分依赖/写法较旧，重理解、不盲目照抄环境 |
| 3 | [fastai/fastbook](https://github.com/fastai/fastbook) | 25,118 | 自顶向下应用 | 上手快；其文字内容有额外版权限制，只链接不搬运 |
| 4 | [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) | 23,489 | 神经网络底层原理 | 适合反向传播与语言模型直觉 |
| 5 | [mrdbourke/pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning) | 18,369 | PyTorch 零到实战 | 阶段 5 的优质辅助主线 |
| 6 | [karpathy/micrograd](https://github.com/karpathy/micrograd) | 16,713 | 自动微分 | 代码小，适合读完反向传播后逐行理解 |
| 7 | [pytorch/tutorials](https://github.com/pytorch/tutorials) | 9,236 | 官方权威用法 | 遇到 API 用法优先查官方教程 |

## 值得阅读的高星开源项目

| 排名 | 仓库 | Stars | 学什么 | 何时读 |
|---:|---|---:|---|---|
| 1 | [huggingface/transformers](https://github.com/huggingface/transformers) | 162,598 | 模型生态、配置、训练接口 | 完成基础 NLP 项目后 |
| 2 | [pytorch/pytorch](https://github.com/pytorch/pytorch) | 101,814 | 框架架构、算子和测试 | 先从文档和小模块读起 |
| 3 | [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) | 66,679 | 统一 API、估计器设计和测试 | 完成传统 ML 阶段后 |
| 4 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | 59,478 | 视觉训练流水线与工程组织 | 完成 CNN 项目后 |
| 5 | [pytorch/examples](https://github.com/pytorch/examples) | 23,952 | 官方参考实现 | 学到对应模型时按需阅读 |

## 如何使用这些资源

- 主线只选一个：传统 ML 首选 Microsoft，深度学习首选 D2L 中文版，PyTorch 实战可选 mrdbourke。
- 每学完一个主题，再看一个“from scratch”实现；先自己写，再对照。
- 大型项目不要从第一行开始读。先找 README、最小示例、数据入口、模型定义、训练循环和测试。
- 记录查看日期、commit/tag 和环境；项目更新后结果可能不同。
- 尊重各仓库许可证。链接不代表可以复制、改写或重新分发全部内容。
