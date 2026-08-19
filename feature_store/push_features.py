"""
Push feature-engineered data to Neon PostgreSQL for the Feast offline store.

Reads datasets/processed/feature_engineered_dataset.csv, adds:
    - match_id       : "{HomeTeam}_vs_{AwayTeam}_{YYYY-MM-DD}"  (entity key)
    - event_timestamp: the match Date column (UTC timezone-aware)

Then drops and recreates the `feast_match_features` table in Neon with only
the columns referenced by Feast feature views (no raw columns with SQL
reserved-word names like AS, Month, Year, Day).

Usage:
    python feature_store/push_features.py
"""

import sys
import os
from pathlib import Path

# Allow running from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import pandas as pd
import sqlalchemy
from sqlalchemy import text
from src.logger import logging
from src.exception import CustomException
from src.components.features import FEATURE_COLS as FEATURE_COLUMNS

TABLE_NAME = "feast_match_features"
CSV_PATH = PROJECT_ROOT / "datasets" / "processed" / "feature_engineered_dataset.csv"

# FEATURE_COLUMNS is imported from features.py (the single source of truth also
# used by model_training.py / model_inference.py) rather than hand-duplicated
# here, so the Feast store can never silently drift from what the models
# actually train on. It also avoids SQL reserved-word column names (AS, Month,
# Year, Day) since features.py only lists engineered columns.


def build_connection_string() -> str:
    """Build SQLAlchemy connection string from .env variables."""
    user     = os.getenv("NEON_DB_ROLE")
    password = os.getenv("NEON_DB_PASSWORD")
    host     = os.getenv("NEON_DB_HOST")
    dbname   = os.getenv("NEON_DB_NAME")

    if not all([user, password, host, dbname]):
        raise EnvironmentError(
            "Missing one or more env vars: NEON_DB_ROLE, NEON_DB_PASSWORD, "
            "NEON_DB_HOST, NEON_DB_NAME"
        )
    return f"postgresql+psycopg2://{user}:{password}@{host}:5432/{dbname}?sslmode=require"


def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    """Load the feature CSV and add Feast-required columns."""
    logging.info(f"Loading feature dataset from {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    # ── Entity key ──────────────────────────────────────────────────────────
    df["match_id"] = (
        df["HomeTeam"].str.replace(" ", "_")
        + "_vs_"
        + df["AwayTeam"].str.replace(" ", "_")
        + "_"
        + pd.to_datetime(df["Date"], format="mixed").dt.strftime("%Y-%m-%d")
    )

    # ── Feast requires a timezone-aware event_timestamp ──────────────────────
    df["event_timestamp"] = pd.to_datetime(df["Date"], format="mixed").dt.tz_localize("UTC")

    # Keep only Feast feature columns + entity key + timestamp
    cols_to_keep = ["match_id", "event_timestamp"] + FEATURE_COLUMNS
    df = df[cols_to_keep]

    logging.info(f"Prepared {len(df)} rows | {df.shape[1]} columns")
    logging.info(f"Sample match_ids: {df['match_id'].head(3).tolist()}")
    return df


def push_to_neon(df: pd.DataFrame, conn_str: str) -> None:
    """Drop-and-replace the feature table in Neon."""
    engine = sqlalchemy.create_engine(conn_str)

    logging.info(f"Pushing {len(df)} rows to Neon table: {TABLE_NAME}")
    df.to_sql(
        name=TABLE_NAME,
        con=engine,
        if_exists="replace",   # drop + recreate on each full push
        index=False,
        method="multi",
        chunksize=500,
    )
    logging.info(f"Successfully pushed to {TABLE_NAME}")

    # Quick row count sanity check
    with engine.connect() as conn:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar()
    logging.info(f"Row count in Neon {TABLE_NAME}: {count}")
    print(f"Pushed {count} rows to {TABLE_NAME} in Neon.")


if __name__ == "__main__":
    try:
        conn_str = build_connection_string()
        df = load_and_prepare(CSV_PATH)
        push_to_neon(df, conn_str)
    except Exception as e:
        raise CustomException(e, sys) from e
