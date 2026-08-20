"""
Run with: python -m pytest (from the project root, so `src` resolves on sys.path).
"""

import pickle
from pathlib import Path

import pandas as pd
import pytest

from src.components.team_aliases import canonicalize_team_name

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_PATH = PROJECT_ROOT / "datasets" / "common_data" / "PL_Fixtures_2026-27.xlsx"
ENCODER_PATH = PROJECT_ROOT / "models" / "team_encoder.pkl"

# The one team with genuinely zero Premier League history in the dataset --
# expected to fail canonicalization and hit ModelInference's fallback instead.
# See team_aliases.py's module docstring for why.
EXPECTED_UNRESOLVED = {"Coventry City"}


@pytest.fixture(scope="module")
def fixture_teams() -> set:
    df = pd.read_excel(FIXTURES_PATH)
    return set(df["Home Team"]) | set(df["Away Team"])


@pytest.fixture(scope="module")
def known_encoder_teams() -> set:
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    return set(encoder.classes_.tolist())


def test_all_fixture_teams_resolve_except_documented_exceptions(fixture_teams, known_encoder_teams):
    unresolved = {t for t in fixture_teams if canonicalize_team_name(t) not in known_encoder_teams}
    assert unresolved == EXPECTED_UNRESOLVED, (
        f"Unexpected unresolved teams: {unresolved - EXPECTED_UNRESOLVED} -- "
        f"add them to TEAM_ALIASES, or to EXPECTED_UNRESOLVED if they're "
        f"genuinely new to the dataset like Coventry City."
    )


def test_alias_map_targets_are_all_real_encoder_names(known_encoder_teams):
    from src.components.team_aliases import TEAM_ALIASES

    bad_targets = {v for v in TEAM_ALIASES.values() if v not in known_encoder_teams}
    assert not bad_targets, f"TEAM_ALIASES points to names the encoder doesn't know: {bad_targets}"
