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
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
    }

    if X_train is not None and y_train is not None:
        cv_roc = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="roc_auc"
        ).mean()
        metrics["cv_roc_auc"] = round(cv_roc, 4)

    return metrics
