"""
Feast Feature Views for Soca Scores EPL match prediction.

Feature groups:
    1. categorical_fv       — encoded team IDs, referee, day, month cyclical
    2. team_form_fv         — rolling home/away form (last 5 & 10 matches)
    3. home_away_split_fv   — home-specific and away-specific rolling stats
    4. h2h_fv               — head-to-head historical stats
    5. shot_quality_fv      — conversion rate and SOT ratio rolling averages
    6. halftime_fv          — 2nd-half goals, lead-hold/comeback rates
    7. referee_fv           — per-referee cards/fouls averages
    8. temporal_fv          — match_week, season_phase, days_rest
"""

from datetime import timedelta

from feast import FeatureView, Field
from feast.types import Float64, Int64

from .data_sources import match_features_source
from .entities import match

# ── 1. Categorical / Encoding ────────────────────────────────────────────────
categorical_fv = FeatureView(
    name="categorical_features",
    entities=[match],
    ttl=timedelta(days=365 * 5),
    schema=[
        Field(name="home_team_encoded", dtype=Int64),
        Field(name="away_team_encoded", dtype=Int64),
        Field(name="referee_encoded", dtype=Int64),
        Field(name="day_encoded", dtype=Int64),
        Field(name="htr_encoded", dtype=Int64),
        Field(name="month_sin", dtype=Float64),
        Field(name="month_cos", dtype=Float64),
    ],
    source=match_features_source,
    description="Label-encoded and cyclically encoded categorical identifiers",
)

# ── 2. Team Form (rolling last 5 & 10) ──────────────────────────────────────
team_form_fv = FeatureView(
    name="team_form_features",
    entities=[match],
    ttl=timedelta(days=365 * 5),
    schema=[
        Field(name="home_points_last5", dtype=Float64),
        Field(name="home_goals_scored_avg5", dtype=Float64),
        Field(name="home_goals_conceded_avg5", dtype=Float64),
        Field(name="home_sot_avg5", dtype=Float64),
        Field(name="home_clean_sheets_last5", dtype=Float64),
        Field(name="home_points_last10", dtype=Float64),
        Field(name="home_goals_scored_avg10", dtype=Float64),
        Field(name="home_goals_conceded_avg10", dtype=Float64),
        Field(name="home_win_streak", dtype=Float64),
        Field(name="away_points_last5", dtype=Float64),
        Field(name="away_goals_scored_avg5", dtype=Float64),
        Field(name="away_goals_conceded_avg5", dtype=Float64),
        Field(name="away_sot_avg5", dtype=Float64),
        Field(name="away_clean_sheets_last5", dtype=Float64),
        Field(name="away_points_last10", dtype=Float64),
        Field(name="away_goals_scored_avg10", dtype=Float64),
        Field(name="away_goals_conceded_avg10", dtype=Float64),
        Field(name="away_win_streak", dtype=Float64),
    ],
    source=match_features_source,
    description="Rolling form statistics for home and away teams (last 5 and 10 matches)",
)

# ── 3. Home / Away Split Form ────────────────────────────────────────────────
home_away_split_fv = FeatureView(
    name="home_away_split_features",
    entities=[match],
    ttl=timedelta(days=365 * 5),
    schema=[
        Field(name="home_win_rate_last10", dtype=Float64),
        Field(name="home_goals_avg_last10", dtype=Float64),
        Field(name="away_win_rate_last10", dtype=Float64),
        Field(name="away_goals_avg_last10", dtype=Float64),
    ],
    source=match_features_source,
    description="Home-only and away-only rolling win rate and goal average (last 10)",
)

# ── 4. Head-to-Head ──────────────────────────────────────────────────────────
h2h_fv = FeatureView(
    name="h2h_features",
    entities=[match],
    ttl=timedelta(days=365 * 5),
    schema=[
        Field(name="h2h_meetings", dtype=Float64),
        Field(name="h2h_home_win_rate", dtype=Float64),
        Field(name="h2h_avg_total_goals", dtype=Float64),
        Field(name="h2h_btts_rate", dtype=Float64),
    ],
    source=match_features_source,
    description="Historical head-to-head statistics between the two teams",
)

# ── 5. Shot Quality ──────────────────────────────────────────────────────────
shot_quality_fv = FeatureView(
    name="shot_quality_features",
    entities=[match],
    ttl=timedelta(days=365 * 5),
    schema=[
        Field(name="home_conversion_rate_avg5", dtype=Float64),
        Field(name="home_sot_ratio_avg5", dtype=Float64),
        Field(name="away_conversion_rate_avg5", dtype=Float64),
        Field(name="away_sot_ratio_avg5", dtype=Float64),
    ],
    source=match_features_source,
    description="Shot conversion rate and shots-on-target ratio (rolling last 5)",
)

# ── 6. Half-Time Patterns ────────────────────────────────────────────────────
halftime_fv = FeatureView(
    name="halftime_features",
    entities=[match],
    ttl=timedelta(days=365 * 5),
    schema=[
        Field(name="home_2nd_half_goals_avg5", dtype=Float64),
        Field(name="home_lead_hold_rate", dtype=Float64),
        Field(name="home_comeback_rate", dtype=Float64),
        Field(name="away_2nd_half_goals_avg5", dtype=Float64),
        Field(name="away_lead_hold_rate", dtype=Float64),
        Field(name="away_comeback_rate", dtype=Float64),
    ],
    source=match_features_source,
    description="2nd-half goal patterns and lead-hold / comeback rates",
)

# ── 7. Referee ───────────────────────────────────────────────────────────────
referee_fv = FeatureView(
    name="referee_features",
    entities=[match],
    ttl=timedelta(days=365 * 5),
    schema=[
        Field(name="ref_avg_yellows", dtype=Float64),
        Field(name="ref_avg_fouls", dtype=Float64),
        Field(name="ref_home_win_rate", dtype=Float64),
    ],
    source=match_features_source,
    description="Per-referee historical averages for yellow cards, fouls, and home win rate",
)

# ── 8. Temporal / Fixture ────────────────────────────────────────────────────
temporal_fv = FeatureView(
    name="temporal_features",
    entities=[match],
    ttl=timedelta(days=365 * 5),
    schema=[
        Field(name="match_week", dtype=Int64),
        Field(name="season_phase", dtype=Int64),
        Field(name="home_days_rest", dtype=Float64),
        Field(name="away_days_rest", dtype=Float64),
    ],
    source=match_features_source,
    description="Fixture context: match week, season phase, and days since last match per team",
)
