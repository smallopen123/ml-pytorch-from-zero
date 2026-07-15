"""端到端表格分类：数据拆分、Pipeline、交叉验证和测试评估。"""

from __future__ import annotations

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    dataset = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        stratify=dataset.target,
        random_state=42,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2_000, random_state=42),
    )

    cv_auc = cross_val_score(model, x_train, y_train, cv=5, scoring="roc_auc")
    print(f"5 折验证 ROC-AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    print(f"测试 ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")
    print("混淆矩阵:\n", confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions, target_names=dataset.target_names))


if __name__ == "__main__":
    main()
