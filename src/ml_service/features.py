import pandas as pd


class RequestFeatureProvider:
    """Builds the model input frame from features supplied in the request."""

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
