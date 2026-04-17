"""
Module for running inference using trained XGBoost models.

Loads the 5 registered models from MLflow and the saved encoders,
constructs a feature vector for a given match, and returns predictions
for all 5 targets.

Used directly by the FastAPI app (api/main.py).
"""

import sys
import pickle
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException

from src.components.model_training import FEATURE_COLS


RESULT_MAP = {0: "Away Win", 1: "Draw", 2: "Home Win"}

DAY_ORDER = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
}
MONTH_ORDER = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}


@dataclass
class ModelInferenceConfig:
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    models_dir: Path = None
    mlflow_uri: str = None
    registry_uri: str = None
    data_path: Path = None

    def __post_init__(self) -> None:
        self.models_dir = self.project_root / "models"
        self.mlflow_uri   = f"sqlite:///{self.project_root / 'mlflow.db'}"
        self.registry_uri = f"sqlite:///{self.project_root / 'mlflow.db'}"
        self.data_path  = self.project_root / "datasets" / "processed" / "feature_engineered_dataset.csv"


class ModelInference:
    def __init__(self) -> None:
        self.config = ModelInferenceConfig()
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_registry_uri(self.config.registry_uri)
        self._load_models()
        self._load_encoders()
        self._load_history()

    def _load_models(self) -> None:
        try:
            logging.info("Loading models from MLflow registry")
            self.model_result  = mlflow.xgboost.load_model("models:/soca_result/latest")
            self.model_btts    = mlflow.xgboost.load_model("models:/soca_btts/latest")
            self.model_over25  = mlflow.xgboost.load_model("models:/soca_over25/latest")
            self.model_over15  = mlflow.xgboost.load_model("models:/soca_over15/latest")
            self.model_goals   = mlflow.xgboost.load_model("models:/soca_goals/latest")
            logging.info("All 5 models loaded")
        except Exception as e:
            raise CustomException(e, sys) from e

    def _load_encoders(self) -> None:
        try:
            with open(self.config.models_dir / "team_encoder.pkl", "rb") as f:
                self.team_encoder = pickle.load(f)
            with open(self.config.models_dir / "referee_encoder.pkl", "rb") as f:
                self.referee_encoder = pickle.load(f)
            logging.info("Encoders loaded")
        except Exception as e:
            raise CustomException(e, sys) from e

    def _load_history(self) -> None:
        try:
            self.df = pd.read_csv(self.config.data_path)
            self.df["Date"] = pd.to_datetime(self.df["Date"])
            self.df = self.df.sort_values("Date").reset_index(drop=True)
            logging.info(f"Historical data loaded: {len(self.df)} rows")
        except Exception as e:
            raise CustomException(e, sys) from e

    def _get_team_features(self, team: str, side: str) -> dict:
        col  = "HomeTeam" if side == "home" else "AwayTeam"
        rows = self.df[self.df[col] == team]
        if rows.empty:
            raise ValueError(f"Team not found in history: {team}")
        latest = rows.iloc[-1]
        return {c: latest[c] for c in FEATURE_COLS if c.startswith(side + "_")}

    def _get_h2h_features(self, home_team: str, away_team: str) -> dict:
        h2h = self.df[
            ((self.df["HomeTeam"] == home_team) & (self.df["AwayTeam"] == away_team)) |
            ((self.df["HomeTeam"] == away_team) & (self.df["AwayTeam"] == home_team))
        ]
        if h2h.empty:
            return {
                "h2h_meetings": 0,
                "h2h_home_win_rate":   float(self.df["h2h_home_win_rate"].mean()),
                "h2h_avg_total_goals": float(self.df["h2h_avg_total_goals"].mean()),
                "h2h_btts_rate":       float(self.df["h2h_btts_rate"].mean()),
            }
        latest = h2h.iloc[-1]
        return {c: latest[c] for c in ["h2h_meetings", "h2h_home_win_rate",
                                        "h2h_avg_total_goals", "h2h_btts_rate"]}

    def _get_referee_features(self, referee: str) -> dict:
        rows = self.df[self.df["Referee"] == referee]
        if rows.empty:
            return {
                "ref_avg_yellows":   float(self.df["ref_avg_yellows"].mean()),
                "ref_avg_fouls":     float(self.df["ref_avg_fouls"].mean()),
                "ref_home_win_rate": float(self.df["ref_home_win_rate"].mean()),
            }
        return {c: rows.iloc[-1][c] for c in ["ref_avg_yellows", "ref_avg_fouls", "ref_home_win_rate"]}

    def build_features(self, home_team: str, away_team: str, date: str,
                       referee: str, match_week: int = 20) -> pd.DataFrame:
        try:
            date_obj  = pd.to_datetime(date)
            month_num = date_obj.month
            day_name  = date_obj.day_name()

            features = {}
            features["home_team_encoded"] = int(self.team_encoder.transform([home_team])[0])
            features["away_team_encoded"] = int(self.team_encoder.transform([away_team])[0])
            features["referee_encoded"]   = int(self.referee_encoder.transform([referee])[0])
            features["day_encoded"]        = DAY_ORDER.get(day_name, 5)
            features["month_sin"]          = float(np.sin(2 * np.pi * month_num / 12))
            features["month_cos"]          = float(np.cos(2 * np.pi * month_num / 12))

            features.update(self._get_team_features(home_team, "home"))
            features.update(self._get_team_features(away_team, "away"))
            features.update(self._get_h2h_features(home_team, away_team))
            features.update(self._get_referee_features(referee))

            features["match_week"]     = match_week
            features["season_phase"]   = 0 if match_week <= 10 else (1 if match_week <= 28 else 2)
            features["home_days_rest"] = 7.0
            features["away_days_rest"] = 7.0

            return pd.DataFrame([features])[FEATURE_COLS]
        except Exception as e:
            raise CustomException(e, sys) from e

    def predict(self, home_team: str, away_team: str, date: str,
                referee: str, match_week: int = 20) -> dict:
        try:
            X = self.build_features(home_team, away_team, date, referee, match_week)

            result_proba = self.model_result.predict_proba(X)[0]
            result_pred  = int(self.model_result.predict(X)[0])
            btts_proba   = float(self.model_btts.predict_proba(X)[0][1])
            over25_proba = float(self.model_over25.predict_proba(X)[0][1])
            over15_proba = float(self.model_over15.predict_proba(X)[0][1])
            goals_pred   = float(self.model_goals.predict(X)[0])

            return {
                "match": f"{home_team} vs {away_team}",
                "date": date,
                "predictions": {
                    "result": {
                        "prediction":  RESULT_MAP[result_pred],
                        "home_win":    round(float(result_proba[2]), 3),
                        "draw":        round(float(result_proba[1]), 3),
                        "away_win":    round(float(result_proba[0]), 3),
                    },
                    "btts": {
                        "prediction": "Yes" if btts_proba > 0.5 else "No",
                        "probability": round(btts_proba, 3),
                    },
                    "over_2_5": {
                        "prediction": "Yes" if over25_proba > 0.5 else "No",
                        "probability": round(over25_proba, 3),
                    },
                    "over_1_5": {
                        "prediction": "Yes" if over15_proba > 0.5 else "No",
                        "probability": round(over15_proba, 3),
                    },
                    "total_goals": {
                        "predicted": round(goals_pred, 2),
                    },
                },
            }
        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    import json

    inferencer = ModelInference()

    matches = [
        {"home_team": "Tottenham",   "away_team": "Brighton",  "date": "2026-04-18", "referee": "S Attwell", "match_week": 33},
        {"home_team": "Chelsea",   "away_team": "Man United",    "date": "2026-04-18", "referee": "M Oliver",  "match_week": 33},
        {"home_team": "Everton", "away_team": "Liverpool",   "date": "2026-04-19", "referee": "C Kavanagh",  "match_week": 33},
        {"home_team": "Man City", "away_team": "Arsenal",   "date": "2026-04-19", "referee": "A Taylor",  "match_week": 33},
    ]

    for match in matches:
        result = inferencer.predict(**match)
        print(json.dumps(result, indent=2))
