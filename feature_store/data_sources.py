"""
Data sources for Soca Scores feature store.

The table `feast_match_features` is loaded into Neon by running:
    python feature_store/push_features.py
"""

from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)

match_features_source = PostgreSQLSource(
    name="match_features_source",
    query="SELECT * FROM feast_match_features",
    timestamp_field="event_timestamp",
)
