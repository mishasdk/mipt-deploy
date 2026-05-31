import os

import pandas as pd
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import Engine

ENTITY_COL = "customerID"
LABEL_COL = "Churn"
DEFAULT_TABLE = "customer_features"


def get_engine(url: str | None = None) -> Engine:
    url = url or os.environ["FEATURE_STORE_URI"]
    return create_engine(url, future=True)


def materialize(
    df: pd.DataFrame,
    engine: Engine | None = None,
    table: str = DEFAULT_TABLE,
) -> int:
    """Full-refresh the feature table from df"""
    if ENTITY_COL not in df.columns:
        raise ValueError(f"Feature frame must contain entity column '{ENTITY_COL}'")

    engine = engine or get_engine()
    df.to_sql(table, engine, if_exists="replace", index=False)
    return len(df)


def read_training_frame(
    engine: Engine | None = None,
    table: str = DEFAULT_TABLE,
) -> pd.DataFrame:
    """Read the full feature frame (features + label) for offline training."""
    engine = engine or get_engine()
    return pd.read_sql_table(table, engine)


def read_online(
    customer_ids: list[str],
    feature_order: list[str],
    engine: Engine | None = None,
    table: str = DEFAULT_TABLE,
) -> pd.DataFrame:
    """Look up features for the given entity keys, ordered by ``feature_order``.
    """
    if not customer_ids:
        raise ValueError("`customer_ids` must be non-empty")

    engine = engine or get_engine()
    cols = ", ".join(f'"{c}"' for c in [ENTITY_COL, *feature_order])
    stmt = text(
        f'SELECT {cols} FROM {table} WHERE "{ENTITY_COL}" IN :ids'
    ).bindparams(bindparam("ids", expanding=True))
    with engine.connect() as conn:
        rows = pd.read_sql_query(stmt, conn, params={"ids": list(customer_ids)})

    found = set(rows[ENTITY_COL])
    missing = [cid for cid in customer_ids if cid not in found]
    if missing:
        raise ValueError(f"Unknown customer_ids (not in feature store): {missing}")

    # Preserve request order and the model's expected feature column order.
    rows = rows.set_index(ENTITY_COL).loc[customer_ids]
    return rows[feature_order].reset_index(drop=True)
