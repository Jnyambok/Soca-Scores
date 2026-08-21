"""
Runs every fixture in a given game week through ModelInference and prints
the results as JSON -- used to generate the data baked into the public
Game Week Predictions page.

Usage:
    python -m experiments.scripts.predict_gameweek --gameweek 1
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.components.model_inference import ModelInference

FIXTURES_PATH = PROJECT_ROOT / "datasets" / "common_data" / "PL_Fixtures_2026-27.xlsx"
MATCHES_PER_GAMEWEEK = 10


def load_gameweek(gameweek: int) -> pd.DataFrame:
    df = pd.read_excel(FIXTURES_PATH)
    start = (gameweek - 1) * MATCHES_PER_GAMEWEEK
    end = start + MATCHES_PER_GAMEWEEK
    return df.iloc[start:end].reset_index(drop=True)


def run(gameweek: int) -> list:
    fixtures = load_gameweek(gameweek)
    inferencer = ModelInference()

    results = []
    for _, row in fixtures.iterrows():
        date_obj = pd.to_datetime(row["Match Date"])
        pred = inferencer.predict(
            home_team=row["Home Team"],
            away_team=row["Away Team"],
            date=date_obj.strftime("%Y-%m-%d"),
            referee=None,
            match_week=gameweek,
        )
        pred["kickoff_date"] = date_obj.strftime("%a %d %b")
        results.append(pred)
        print(f"{row['Home Team']:>26} vs {row['Away Team']:<26} -> "
              f"{pred['predictions']['result']['prediction']} "
              f"({pred['predictions']['result']['home_win']:.0%} / "
              f"{pred['predictions']['result']['draw']:.0%} / "
              f"{pred['predictions']['result']['away_win']:.0%})")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameweek", type=int, default=1)
    parser.add_argument("--out", type=str, default=None, help="Optional path to write JSON output")
    args = parser.parse_args()

    predictions = run(args.gameweek)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"\nWrote {len(predictions)} predictions to {args.out}")
