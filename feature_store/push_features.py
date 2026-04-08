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

TABLE_NAME = "feast_match_features"
CSV_PATH = PROJECT_ROOT / "datasets" / "processed" / "feature_engineered_dataset.csv"

# Only these columns are pushed to Neon — avoids SQL reserved-word column names
# (AS, Month, Year, Day) and keeps the table lean.
FEATURE_COLUMNS = [
    # categorical
    "home_team_encoded", "away_team_encoded", "referee_encoded",
    "day_encoded", "htr_encoded", "month_sin", "month_cos",
    # team form
    "home_points_last5", "home_goals_scored_avg5", "home_goals_conceded_avg5",
    "home_sot_avg5", "home_clean_sheets_last5",
    "home_points_last10", "home_goals_scored_avg10", "home_goals_conceded_avg10",
    "home_win_streak",
    "away_points_last5", "away_goals_scored_avg5", "away_goals_conceded_avg5",
    "away_sot_avg5", "away_clean_sheets_last5",
    "away_points_last10", "away_goals_scored_avg10", "away_goals_conceded_avg10",
    "away_win_streak",
    # home/away split
    "home_win_rate_last10", "home_goals_avg_last10",
    "away_win_rate_last10", "away_goals_avg_last10",
    # h2h
    "h2h_meetings", "h2h_home_win_rate", "h2h_avg_total_goals", "h2h_btts_rate",
    # shot quality
    "home_conversion_rate_avg5", "home_sot_ratio_avg5",
    "away_conversion_rate_avg5", "away_sot_ratio_avg5",
    # halftime
    "home_2nd_half_goals_avg5", "home_lead_hold_rate", "home_comeback_rate",
    "away_2nd_half_goals_avg5", "away_lead_hold_rate", "away_comeback_rate",
    # referee
    "ref_avg_yellows", "ref_avg_fouls", "ref_home_win_rate",
    # temporal
    "match_week", "season_phase", "home_days_rest", "away_days_rest",
]


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
