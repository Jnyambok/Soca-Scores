"""
Maps fixture-list team names to the short names used throughout the historical
dataset and the trained encoders (footballdata.uk convention, e.g. "Man City").

Fixture lists (e.g. the official 2026/27 PL fixture list) use the FA's
long-form club names (e.g. "Manchester City"), which don't match the
encoder's classes_ as exact strings. Without this mapping, 10 of the 20
2026/27 clubs would silently fall through to ModelInference's "unknown team"
fallback (median encoding + league-average features) despite having 20
years of real history in the dataset.

Coventry City is deliberately NOT in this map -- it has zero Premier League
matches in the historical dataset, so there is no encoder-side name for it
to alias to. It's expected to hit the same "unknown team" fallback that
handles any genuinely new team (e.g. a future promotion), which is the
correct behaviour for it, not a bug to fix here.
"""

TEAM_ALIASES = {
    "AFC Bournemouth":         "Bournemouth",
    "Brighton & Hove Albion":  "Brighton",
    "Hull City":               "Hull",
    "Ipswich Town":            "Ipswich",
    "Leeds United":            "Leeds",
    "Manchester City":         "Man City",
    "Manchester United":       "Man United",
    "Newcastle United":        "Newcastle",
    "Nottingham Forest":       "Nott'm Forest",
    "Tottenham Hotspur":       "Tottenham",
}


# Case/whitespace-insensitive lookup so API callers that don't send exact
# FA capitalization (e.g. "manchester city", or a trailing space from
# copy-pasted fixture text) still resolve instead of silently degrading
# to the unknown-team fallback.
_ALIASES_BY_NORMALIZED_KEY = {
    alias.strip().casefold(): canonical for alias, canonical in TEAM_ALIASES.items()
}


def canonicalize_team_name(name: str) -> str:
    """Map a fixture-list team name to its encoder-known name, case/whitespace
    insensitively. Names not in TEAM_ALIASES (already-correct names, and
    genuinely new teams like Coventry City) pass through unchanged (stripped)."""
    if not name:
        return name
    return _ALIASES_BY_NORMALIZED_KEY.get(name.strip().casefold(), name.strip())
