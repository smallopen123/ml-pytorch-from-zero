# 项目 1｜端到端表格分类

这个项目使用 scikit-learn 自带的乳腺癌诊断数据，只用于学习机器学习流程，不用于真实医疗决策。

## 目标

根据数值特征预测良性/恶性类别，建立一个可复现的逻辑回归基线。重点不是追逐最高分，而是正确完成拆分、预处理、交叉验证、最终测试和错误解释。

## 1. 明确数据边界

每行是一份样本，每列是数值特征，标签是二分类。真实项目必须确认是否存在同一患者的重复样本、采集时间和标签来源；这些信息决定拆分方式。

## 2. 分层拆分

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

`stratify=y` 使两份数据的类别比例接近。测试集从此封存，模型和阈值选择只在训练数据内部完成。

## 3. 用 Pipeline 防止泄漏

```python
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)
```

交叉验证时，每一折只用该折训练部分拟合标准化参数。若先对全体训练数据执行 `fit_transform` 再交叉验证，验证折信息会进入均值和标准差。

## 4. 指标

脚本同时报告 ROC-AUC、混淆矩阵、精确率、召回率和 F1。医疗筛查常关心漏诊，但这个公开数据示例没有真实成本设定，因此不要擅自声称某个阈值适合临床。

## 5. 运行

```powershell
pip install scikit-learn
python scripts/project_tabular_classification.py
```

[打开完整项目代码](../scripts/project_tabular_classification.py)

## 6. 必做改进

1. 加入 `DummyClassifier` 多数类基线。
2. 比较逻辑回归、KNN 和随机森林的交叉验证结果。
3. 在验证流程中选择阈值，并解释精确率/召回率变化。
4. 列出错误样本的特征，寻找系统性错误。
5. 写下数据、指标和结论的局限。

## 验收清单

- 测试集未参与模型选择。
- 预处理位于 Pipeline 内。
- 报告交叉验证均值与波动。
- 报告最终测试结果和混淆矩阵。
- 不把相关性或模型权重解释成因果关系。
