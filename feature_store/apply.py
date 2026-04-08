"""
Register all Soca Scores feature definitions with the Feast feature store.

Run this whenever you add or update a feature view:
    python feature_store/apply.py

Registry is stored locally at feature_store/data/registry.db
Offline/online store connections go to Neon PostgreSQL (direct endpoint).
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Set env vars (strip any surrounding quotes from .env values)
for key in ["NEON_DB_ROLE", "NEON_DB_PASSWORD", "NEON_DB_HOST_DIRECT", "NEON_DB_NAME"]:
    os.environ[key] = os.getenv(key, "").strip('"').strip("'")

os.chdir(Path(__file__).parent)

from feast import FeatureStore
from src.logger import logging
from src.exception import CustomException

from . import entities
from . import feature_views


def run_apply() -> None:
    try:
        fs = FeatureStore(repo_path=".")
        logging.info(f"Feast project: {fs.project}")

        objects = [
            entities.match,
            feature_views.categorical_fv,
            feature_views.team_form_fv,
            feature_views.home_away_split_fv,
            feature_views.h2h_fv,
            feature_views.shot_quality_fv,
            feature_views.halftime_fv,
            feature_views.referee_fv,
            feature_views.temporal_fv,
        ]

        fs.apply(objects)
        logging.info("feast apply completed")

        print("\nRegistered feature views:")
        for fv in fs.list_feature_views():
            print(f"  {fv.name}: {len(fv.schema)} features")

        print(f"\nRegistry: {Path('.') / 'data' / 'registry.db'}")
        print("Run push_features.py first if the Neon table needs refreshing.")

    except Exception as e:
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    run_apply()
