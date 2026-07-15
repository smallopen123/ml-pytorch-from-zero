# 1.1｜Python：先学会表达数据与步骤

## 学习目标

学完后你应能读懂变量、容器、循环、函数、模块和异常，并能把重复步骤封装为函数。机器学习代码本质上仍是普通 Python：读取数据、变换数据、训练模型、评价结果。

## 1. 变量与类型

变量是对象的名字，不是固定类型的盒子：

```python
age = 18                 # int
learning_rate = 0.01     # float
name = "模型 A"          # str
is_trained = False       # bool
```

`type(value)` 查看类型。涉及金额、类别或缺失值时要特别检查类型；字符串 `"10"` 与数字 `10` 不能直接相加。

## 2. 列表、字典和元组

```python
scores = [0.81, 0.85, 0.83]
experiment = {"model": "logistic", "accuracy": 0.85}
shape = (32, 10)  # 32 个样本、10 个特征

average = sum(scores) / len(scores)
best = max(scores)
```

列表保存有顺序的同类数据；字典保存“字段 → 值”；元组常用于不会改变的结构，例如数组形状。

## 3. 条件与循环

```python
for score in scores:
    if score >= 0.85:
        print("达到目标")
    else:
        print("继续实验")
```

循环适合理解算法，但大规模数值运算应优先用 NumPy 向量化。循环不是错误，只是数组计算时通常更慢、更容易写错索引。

## 4. 函数：明确输入与输出

```python
def accuracy(predictions: list[int], targets: list[int]) -> float:
    if len(predictions) != len(targets):
        raise ValueError("长度必须相同")
    correct = sum(p == y for p, y in zip(predictions, targets))
    return correct / len(targets)
```

好的函数有清楚的名字、参数、返回值和边界检查。训练代码要避免依赖到处变化的全局变量。

## 5. 模块与入口

```python
def main() -> None:
    print(accuracy([1, 0, 1], [1, 1, 1]))

if __name__ == "__main__":
    main()
```

这样文件既能直接运行，也能被测试文件安全导入。

## 6. 练习

1. 写 `mean(values)`，空列表时抛出 `ValueError`。
2. 写 `confusion_counts(predictions, targets)`，返回 TP、TN、FP、FN。
3. 把训练配置保存为字典，并用循环打印每个键值。
4. 解释为什么函数返回结果通常比直接 `print` 更容易测试。

## 运行配套代码

[打开 Python/NumPy 配套代码](../../scripts/python_numpy_fundamentals.py)
