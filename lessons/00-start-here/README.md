# 00｜开始之前

## 需要准备

- 一台能运行 Python 3.10+ 的电脑；前 17 周不需要独立显卡
- Git、VS Code 或其他编辑器
- 每周稳定的 6–8 小时
- 一个“错误日志”：记录报错、原因、修复方法和仍未理解的点

## 环境安装

推荐先用 CPU 环境，减少 CUDA 带来的额外问题。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_environment.py
```

Linux/macOS 激活命令为 `source .venv/bin/activate`。需要 GPU 时，再根据 PyTorch 官方安装选择器安装与你的驱动匹配的版本，不要随意复制旧教程中的 CUDA 命令。

## 学习时如何提问

一次高质量提问至少包含：目标、最小代码、完整报错、Python/PyTorch 版本、预期结果、实际结果，以及你已经尝试过什么。不要只贴一句“为什么跑不了”。

## 完成标准

完成本阶段后，你应能：

- 从终端运行 Python 文件
- 创建和激活虚拟环境
- 使用 Git 提交学习记录
- 解释分类、回归、聚类的区别
- 解释为什么测试集不能参与调参
