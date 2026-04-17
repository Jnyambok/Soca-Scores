"""
Module for training XGBoost models for EPL match prediction.

Trains 5 models (one per target) and logs all runs to MLflow:
    1. result_encoded  — multi-class  (Home Win / Draw / Away Win)
    2. btts            — binary       (Both Teams to Score)
    3. over_2_5        — binary       (Over 2.5 Goals)
    4. over_1_5        — binary       (Over 1.5 Goals)
    5. total_goals     — regression   (Expected Goals)

Encoders (team + referee LabelEncoders) are saved to models/ for inference.
"""

import sys
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

from src.logger import logging
from src.exception import CustomException


FEATURE_COLS = [
    # categorical (htr_encoded excluded: it is the current match HT result = leakage)
    "home_team_encoded", "away_team_encoded", "referee_encoded",
    "day_encoded", "month_sin", "month_cos",
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
    # head-to-head
    "h2h_meetings", "h2h_home_win_rate", "h2h_avg_total_goals", "h2h_btts_rate",
    # shot quality
    "home_conversion_rate_avg5", "home_sot_ratio_avg5",
    "away_conversion_rate_avg5", "away_sot_ratio_avg5",
    # half-time patterns (rolling from previous matches, not the current one)
    "home_2nd_half_goals_avg5", "home_lead_hold_rate", "home_comeback_rate",
    "away_2nd_half_goals_avg5", "away_lead_hold_rate", "away_comeback_rate",
    # referee
    "ref_avg_yellows", "ref_avg_fouls", "ref_home_win_rate",
    # temporal
    "match_week", "season_phase", "home_days_rest", "away_days_rest",
]


@dataclass
class ModelTrainingConfig:
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    data_path: Path = None
    models_dir: Path = None
    mlflow_uri: str = None
    registry_uri: str = None
    experiment_name: str = "soca_scores_match_prediction"
    split_date: str = "2023-08-01"

    classifier_params: dict = field(default_factory=lambda: dict(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42,
    ))
    regressor_params: dict = field(default_factory=lambda: dict(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", eval_metric="mae", random_state=42,
    ))

    def __post_init__(self) -> None:
        self.data_path    = self.project_root / "datasets" / "processed" / "feature_engineered_dataset.csv"
        self.models_dir   = self.project_root / "models"
        self.mlflow_uri   = f"sqlite:///{self.project_root / 'mlflow.db'}"
        self.registry_uri = f"sqlite:///{self.project_root / 'mlflow.db'}"
        self.models_dir.mkdir(parents=True, exist_ok=True)


class ModelTraining:
    def __init__(self) -> None:
        self.config = ModelTrainingConfig()

    def load_and_split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            logging.info("Loading feature dataset")
            df = pd.read_csv(self.config.data_path)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)

            train = df[df["Date"] < self.config.split_date].copy()
            test  = df[df["Date"] >= self.config.split_date].copy()

            logging.info(f"Train: {len(train)} | Test: {len(test)} | Features: {len(FEATURE_COLS)}")
            print(f"Train: {len(train)} matches | Test: {len(test)} matches")
            return train, test, df
        except Exception as e:
            raise CustomException(e, sys) from e

    def save_encoders(self, df: pd.DataFrame) -> None:
        try:
            logging.info("Fitting and saving encoders")
            team_enc = LabelEncoder()
            team_enc.fit(pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique())

            ref_enc = LabelEncoder()
            ref_enc.fit(df["Referee"].unique())

            with open(self.config.models_dir / "team_encoder.pkl", "wb") as f:
                pickle.dump(team_enc, f)
            with open(self.config.models_dir / "referee_encoder.pkl", "wb") as f:
                pickle.dump(ref_enc, f)

            logging.info(f"Encoders saved — teams: {len(team_enc.classes_)} | referees: {len(ref_enc.classes_)}")
            print(f"Encoders saved to {self.config.models_dir}")
        except Exception as e:
            raise CustomException(e, sys) from e

    def _train_classifier(self, run_name, model_name, X_train, X_test, y_train, y_test,
                          objective, num_class=None):
        params = {**self.config.classifier_params, "objective": objective}
        if num_class:
            params["num_class"]   = num_class
            params["eval_metric"] = "mlogloss"
        else:
            params["eval_metric"] = "logloss"

        with mlflow.start_run(run_name=run_name):
            model = XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

            preds = model.predict(X_test)
            acc   = accuracy_score(y_test, preds)
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", acc)

            if num_class:
                f1 = f1_score(y_test, preds, average="weighted")
                mlflow.log_metric("f1_weighted", f1)
                print(f"{run_name:20s} Accuracy: {acc:.3f} | F1: {f1:.3f}")
            else:
                proba = model.predict_proba(X_test)[:, 1]
                auc   = roc_auc_score(y_test, proba)
                mlflow.log_metric("roc_auc", auc)
                print(f"{run_name:20s} Accuracy: {acc:.3f} | AUC: {auc:.3f}")

            mlflow.xgboost.log_model(model, name="model", registered_model_name=model_name)

    def _train_regressor(self, run_name, model_name, X_train, X_test, y_train, y_test):
        with mlflow.start_run(run_name=run_name):
            model = XGBRegressor(**self.config.regressor_params)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

            preds = model.predict(X_test)
            mae   = mean_absolute_error(y_test, preds)
            mlflow.log_params(self.config.regressor_params)
            mlflow.log_metric("mae", mae)
            mlflow.xgboost.log_model(model, name="model", registered_model_name=model_name)
            print(f"{run_name:20s} MAE: {mae:.3f}")

    def train_all(self) -> None:
        try:
            train, test, df = self.load_and_split()
            self.save_encoders(df)

            mlflow.set_tracking_uri(self.config.mlflow_uri)
            mlflow.set_registry_uri(self.config.registry_uri)
            mlflow.set_experiment(self.config.experiment_name)

            X_train, X_test = train[FEATURE_COLS], test[FEATURE_COLS]

            logging.info("Training all 5 models")

            self._train_classifier(
                "result_prediction", "soca_result",
                X_train, X_test, train["result_encoded"], test["result_encoded"],
                objective="multi:softprob", num_class=3,
            )
            self._train_classifier(
                "btts_prediction", "soca_btts",
                X_train, X_test, train["btts"], test["btts"],
                objective="binary:logistic",
            )
            self._train_classifier(
                "over25_prediction", "soca_over25",
                X_train, X_test, train["over_2_5"], test["over_2_5"],
                objective="binary:logistic",
            )
            self._train_classifier(
                "over15_prediction", "soca_over15",
                X_train, X_test, train["over_1_5"], test["over_1_5"],
                objective="binary:logistic",
            )
            self._train_regressor(
                "goals_prediction", "soca_goals",
                X_train, X_test, train["total_goals"], test["total_goals"],
            )

            logging.info("All models trained and registered in MLflow")
            print(f"\nMLflow UI: mlflow ui --backend-store-uri {self.config.registry_uri}")
        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    ModelTraining().train_all()
