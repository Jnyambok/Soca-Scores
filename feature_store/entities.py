from feast import Entity, ValueType

# A match is uniquely identified by HomeTeam + AwayTeam + Date,
# encoded as a single string key: "{HomeTeam}_vs_{AwayTeam}_{YYYY-MM-DD}"
match = Entity(
    name="match_id",
    value_type=ValueType.STRING,
    description="Unique identifier for a football match: HomeTeam_vs_AwayTeam_Date",
)
