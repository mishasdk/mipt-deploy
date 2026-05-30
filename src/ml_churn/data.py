from pathlib import Path
from typing import NamedTuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class DataSplit(NamedTuple):
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_data(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df = df.drop(columns=["customerID"])
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])
    return df


def make_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DataSplit:
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return DataSplit(X_train, X_test, y_train, y_test)
