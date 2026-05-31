import pandas as pd
from sqlalchemy.engine import Engine

from ml_churn import feature_store


class RequestFeatureProvider:
    def __init__(self, feature_order: list[str] | None = None) -> None:
        self._feature_order = feature_order

    def to_frame(self, instances: list[dict[str, float]]) -> pd.DataFrame:
        df = pd.DataFrame(instances)
        if self._feature_order:
            missing = [c for c in self._feature_order if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required features: {missing}")
            df = df[self._feature_order]
        return df


class FeatureStoreProvider:
    def __init__(self, feature_order: list[str], engine: Engine) -> None:
        self._feature_order = feature_order
        self._engine = engine

    def to_frame(self, customer_ids: list[str]) -> pd.DataFrame:
        return feature_store.read_online(
            customer_ids, self._feature_order, engine=self._engine
        )
