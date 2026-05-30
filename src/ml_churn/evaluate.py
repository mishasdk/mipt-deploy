import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score


def compute_metrics(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: pd.DataFrame | None = None,
    y_train: pd.Series | None = None,
    cv: int = 5,
) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
    }

    if X_train is not None and y_train is not None:
        cv_roc = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="roc_auc"
        ).mean()
        metrics["CV ROC-AUC (5-fold)"] = round(cv_roc, 4)
    return metrics


def summary_table(results: dict) -> pd.io.formats.style.Styler:
    df = pd.DataFrame(results).T
    df.index.name = "Model"
    return df.style.highlight_max(
        axis=0, props="font-weight: bold; color: green"
    ).format("{:.4f}")
