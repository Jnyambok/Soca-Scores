"""
Module for engineering features from cleaned EPL data.

Steps:
    1. encode_targets        — match result, BTTS, over/under labels
    2. encode_categoricals   — teams, referee, day (label), month (cyclical)
    3. compute_team_form     — rolling 5 & 10-match form + win streak
    4. compute_home_away_split — home-only and away-only rolling stats
    5. compute_h2h           — head-to-head historical stats
    6. compute_shot_quality  — shot conversion & SOT ratio rolling averages
    7. compute_halftime      — 2nd-half goals, lead-hold, comeback rates
    8. compute_referee       — per-referee yellow cards, fouls, home win rate
    9. compute_temporal      — match week, season phase, days rest
    10. fill_nulls_and_save  — impute NaNs and write to processed/
"""

import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.logger import logging
from src.exception import CustomException


@dataclass
class FeatureEngineeringConfig:
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    cleaned_csv: Path = None
    output_dir: Path = None
    output_csv: Path = None

    def __post_init__(self) -> None:
        self.cleaned_csv = self.project_root / "datasets" / "cleaned_ingested_data" / "cleaned_ingested_data.csv"
        self.output_dir  = self.project_root / "datasets" / "processed"
        self.output_csv  = self.output_dir / "feature_engineered_dataset.csv"
        self.output_dir.mkdir(parents=True, exist_ok=True)


class FeatureEngineering:
    def __init__(self) -> None:
        self.config = FeatureEngineeringConfig()

    # ── Step 1 ────────────────────────────────────────────────────────────────
    def encode_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            print("Feature Engineering Step 1: Encoding target labels...")
            logging.info("Step 1: Encoding target labels")

            df["result_encoded"] = df["FTR"].map({"H": 2, "D": 1, "A": 0})
            df["total_goals"]    = df["FTHG"] + df["FTAG"]
            df["btts"]           = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
            df["over_2_5"]       = (df["total_goals"] > 2).astype(int)
            df["over_1_5"]       = (df["total_goals"] > 1).astype(int)

            logging.info("Step 1 complete")
            print("Step 1: Target labels encoded...DONE")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── Step 2 ────────────────────────────────────────────────────────────────
    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            print("Feature Engineering Step 2: Encoding categorical features...")
            logging.info("Step 2: Encoding categorical features")

            df["htr_encoded"] = df["HTR"].map({"H": 2, "D": 1, "A": 0})

            team_encoder = LabelEncoder()
            team_encoder.fit(pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique())
            df["home_team_encoded"] = team_encoder.transform(df["HomeTeam"])
            df["away_team_encoded"] = team_encoder.transform(df["AwayTeam"])

            referee_encoder = LabelEncoder()
            df["referee_encoded"] = referee_encoder.fit_transform(df["Referee"])

            day_order = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                         "Friday": 4, "Saturday": 5, "Sunday": 6}
            df["day_encoded"] = df["Day"].map(day_order)

            month_order = {"January": 1, "February": 2, "March": 3, "April": 4,
                           "May": 5, "June": 6, "July": 7, "August": 8,
                           "September": 9, "October": 10, "November": 11, "December": 12}
            month_num = df["Month"].map(month_order)
            df["month_sin"] = np.sin(2 * np.pi * month_num / 12)
            df["month_cos"] = np.cos(2 * np.pi * month_num / 12)

            logging.info("Step 2 complete")
            print("Step 2: Categorical features encoded...DONE")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── Step 3 ────────────────────────────────────────────────────────────────
    def compute_team_form(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            print("Feature Engineering Step 3: Computing team form features...")
            logging.info("Step 3: Computing team rolling form")

            roll_cols = ["points", "goals_scored", "goals_conceded", "sot", "clean_sheet", "won"]

            home_view = df[["Date", "HomeTeam", "FTHG", "FTAG", "FTR", "HST"]].copy()
            home_view.columns = ["date", "team", "goals_scored", "goals_conceded", "ftr", "sot"]
            home_view["points"]      = home_view["ftr"].map({"H": 3, "D": 1, "A": 0})
            home_view["won"]         = (home_view["ftr"] == "H").astype(int)
            home_view["clean_sheet"] = (home_view["goals_conceded"] == 0).astype(int)

            away_view = df[["Date", "AwayTeam", "FTAG", "FTHG", "FTR", "AST"]].copy()
            away_view.columns = ["date", "team", "goals_scored", "goals_conceded", "ftr", "sot"]
            away_view["points"]      = away_view["ftr"].map({"A": 3, "D": 1, "H": 0})
            away_view["won"]         = (away_view["ftr"] == "A").astype(int)
            away_view["clean_sheet"] = (away_view["goals_conceded"] == 0).astype(int)

            team_matches = (
                pd.concat([home_view, away_view])
                .sort_values(["team", "date"])
                .reset_index(drop=True)
            )

            def rolling_agg(group, window):
                shifted = group[roll_cols].shift(1)
                return shifted.rolling(window, min_periods=1).mean()

            form5  = team_matches.groupby("team", group_keys=False).apply(lambda g: rolling_agg(g, 5),  include_groups=False)
            form10 = team_matches.groupby("team", group_keys=False).apply(lambda g: rolling_agg(g, 10), include_groups=False)

            team_matches["points_last5"]         = form5["points"]
            team_matches["goals_scored_avg5"]    = form5["goals_scored"]
            team_matches["goals_conceded_avg5"]  = form5["goals_conceded"]
            team_matches["sot_avg5"]             = form5["sot"]
            team_matches["clean_sheets_last5"]   = form5["clean_sheet"]
            team_matches["points_last10"]        = form10["points"]
            team_matches["goals_scored_avg10"]   = form10["goals_scored"]
            team_matches["goals_conceded_avg10"] = form10["goals_conceded"]

            def win_streak(group):
                streaks, count = [], 0
                for won in group["won"].shift(1).fillna(0):
                    count = count + 1 if won == 1 else 0
                    streaks.append(count)
                return pd.Series(streaks, index=group.index)

            team_matches["win_streak"] = team_matches.groupby("team", group_keys=False).apply(win_streak, include_groups=False)

            home_form_cols = ["Date", "HomeTeam", "home_points_last5", "home_goals_scored_avg5",
                              "home_goals_conceded_avg5", "home_sot_avg5", "home_clean_sheets_last5",
                              "home_points_last10", "home_goals_scored_avg10", "home_goals_conceded_avg10",
                              "home_win_streak"]
            away_form_cols = ["Date", "AwayTeam", "away_points_last5", "away_goals_scored_avg5",
                              "away_goals_conceded_avg5", "away_sot_avg5", "away_clean_sheets_last5",
                              "away_points_last10", "away_goals_scored_avg10", "away_goals_conceded_avg10",
                              "away_win_streak"]

            form_home = team_matches.rename(columns={
                "team": "HomeTeam", "date": "Date",
                "points_last5": "home_points_last5", "goals_scored_avg5": "home_goals_scored_avg5",
                "goals_conceded_avg5": "home_goals_conceded_avg5", "sot_avg5": "home_sot_avg5",
                "clean_sheets_last5": "home_clean_sheets_last5", "points_last10": "home_points_last10",
                "goals_scored_avg10": "home_goals_scored_avg10", "goals_conceded_avg10": "home_goals_conceded_avg10",
                "win_streak": "home_win_streak",
            })
            form_away = team_matches.rename(columns={
                "team": "AwayTeam", "date": "Date",
                "points_last5": "away_points_last5", "goals_scored_avg5": "away_goals_scored_avg5",
                "goals_conceded_avg5": "away_goals_conceded_avg5", "sot_avg5": "away_sot_avg5",
                "clean_sheets_last5": "away_clean_sheets_last5", "points_last10": "away_points_last10",
                "goals_scored_avg10": "away_goals_scored_avg10", "goals_conceded_avg10": "away_goals_conceded_avg10",
                "win_streak": "away_win_streak",
            })

            df = df.merge(form_home[home_form_cols], on=["Date", "HomeTeam"], how="left")
            df = df.merge(form_away[away_form_cols], on=["Date", "AwayTeam"], how="left")

            logging.info(f"Step 3 complete. Shape: {df.shape}")
            print("Step 3: Team form features computed...DONE")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── Step 4 ────────────────────────────────────────────────────────────────
    def compute_home_away_split(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            print("Feature Engineering Step 4: Computing home/away split features...")
            logging.info("Step 4: Home/away split rolling stats")

            home_only = df[["Date", "HomeTeam", "FTHG", "FTR"]].copy()
            home_only.columns = ["date", "team", "goals_scored", "ftr"]
            home_only["won"] = (home_only["ftr"] == "H").astype(int)
            home_only = home_only.sort_values(["team", "date"])

            away_only = df[["Date", "AwayTeam", "FTAG", "FTR"]].copy()
            away_only.columns = ["date", "team", "goals_scored", "ftr"]
            away_only["won"] = (away_only["ftr"] == "A").astype(int)
            away_only = away_only.sort_values(["team", "date"])

            def rolling_split(group, window=10):
                shifted = group[["won", "goals_scored"]].shift(1)
                return shifted.rolling(window, min_periods=1).mean()

            hs = home_only.groupby("team", group_keys=False).apply(rolling_split, include_groups=False)
            home_only["home_win_rate_last10"]  = hs["won"]
            home_only["home_goals_avg_last10"] = hs["goals_scored"]

            aws = away_only.groupby("team", group_keys=False).apply(rolling_split, include_groups=False)
            away_only["away_win_rate_last10"]  = aws["won"]
            away_only["away_goals_avg_last10"] = aws["goals_scored"]

            df = df.merge(
                home_only[["date", "team", "home_win_rate_last10", "home_goals_avg_last10"]]
                .rename(columns={"team": "HomeTeam", "date": "Date"}),
                on=["Date", "HomeTeam"], how="left"
            )
            df = df.merge(
                away_only[["date", "team", "away_win_rate_last10", "away_goals_avg_last10"]]
                .rename(columns={"team": "AwayTeam", "date": "Date"}),
                on=["Date", "AwayTeam"], how="left"
            )

            logging.info(f"Step 4 complete. Shape: {df.shape}")
            print("Step 4: Home/away split features computed...DONE")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── Step 5 ────────────────────────────────────────────────────────────────
    def compute_h2h(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            print("Feature Engineering Step 5: Computing H2H features (slow — row-by-row)...")
            logging.info("Step 5: Head-to-head features")

            h2h_src = df[["Date", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"]].copy()
            h2h_src["team_a"]      = h2h_src[["HomeTeam", "AwayTeam"]].min(axis=1)
            h2h_src["team_b"]      = h2h_src[["HomeTeam", "AwayTeam"]].max(axis=1)
            h2h_src["total_goals"] = h2h_src["FTHG"] + h2h_src["FTAG"]
            h2h_src["btts_match"]  = ((h2h_src["FTHG"] > 0) & (h2h_src["FTAG"] > 0)).astype(int)
            h2h_src["home_win"]    = (h2h_src["FTR"] == "H").astype(int)
            h2h_src = h2h_src.sort_values("Date").reset_index(drop=True)

            h2h_features = []
            for _, row in h2h_src.iterrows():
                prior = h2h_src[
                    (h2h_src["team_a"] == row["team_a"]) &
                    (h2h_src["team_b"] == row["team_b"]) &
                    (h2h_src["Date"] < row["Date"])
                ]
                if len(prior) == 0:
                    h2h_features.append({"h2h_meetings": 0, "h2h_home_win_rate": np.nan,
                                         "h2h_avg_total_goals": np.nan, "h2h_btts_rate": np.nan})
                else:
                    h2h_features.append({
                        "h2h_meetings":        len(prior),
                        "h2h_home_win_rate":   prior["home_win"].mean(),
                        "h2h_avg_total_goals": prior["total_goals"].mean(),
                        "h2h_btts_rate":       prior["btts_match"].mean(),
                    })

            df = pd.concat([df, pd.DataFrame(h2h_features, index=h2h_src.index)], axis=1)

            logging.info(f"Step 5 complete. Shape: {df.shape}")
            print("Step 5: H2H features computed...DONE")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── Step 6 ────────────────────────────────────────────────────────────────
    def compute_shot_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            print("Feature Engineering Step 6: Computing shot quality features...")
            logging.info("Step 6: Shot quality rolling features")

            shot_home = df[["Date", "HomeTeam", "FTHG", "HST", "HS"]].copy()
            shot_home.columns = ["date", "team", "goals", "sot", "shots"]
            shot_home["conversion"] = shot_home["goals"] / shot_home["sot"].replace(0, np.nan)
            shot_home["sot_ratio"]  = shot_home["sot"]   / shot_home["shots"].replace(0, np.nan)
            shot_home = shot_home.sort_values(["team", "date"])

            shot_away = df[["Date", "AwayTeam", "FTAG", "AST", "AS"]].copy()
            shot_away.columns = ["date", "team", "goals", "sot", "shots"]
            shot_away["conversion"] = shot_away["goals"] / shot_away["sot"].replace(0, np.nan)
            shot_away["sot_ratio"]  = shot_away["sot"]   / shot_away["shots"].replace(0, np.nan)
            shot_away = shot_away.sort_values(["team", "date"])

            def rolling_shot(group):
                shifted = group[["conversion", "sot_ratio"]].shift(1)
                return shifted.rolling(5, min_periods=1).mean()

            sh = shot_home.groupby("team", group_keys=False).apply(rolling_shot, include_groups=False)
            shot_home["home_conversion_rate_avg5"] = sh["conversion"]
            shot_home["home_sot_ratio_avg5"]       = sh["sot_ratio"]

            sa = shot_away.groupby("team", group_keys=False).apply(rolling_shot, include_groups=False)
            shot_away["away_conversion_rate_avg5"] = sa["conversion"]
            shot_away["away_sot_ratio_avg5"]       = sa["sot_ratio"]

            df = df.merge(
                shot_home[["date", "team", "home_conversion_rate_avg5", "home_sot_ratio_avg5"]]
                .rename(columns={"team": "HomeTeam", "date": "Date"}),
                on=["Date", "HomeTeam"], how="left"
            )
            df = df.merge(
                shot_away[["date", "team", "away_conversion_rate_avg5", "away_sot_ratio_avg5"]]
                .rename(columns={"team": "AwayTeam", "date": "Date"}),
                on=["Date", "AwayTeam"], how="left"
            )

            logging.info(f"Step 6 complete. Shape: {df.shape}")
            print("Step 6: Shot quality features computed...DONE")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── Step 7 ────────────────────────────────────────────────────────────────
    def compute_halftime(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            print("Feature Engineering Step 7: Computing half-time pattern features...")
            logging.info("Step 7: Half-time pattern features")

            ht = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "HTAG", "HTR", "FTR"]].copy()

            ht_home = ht[["Date", "HomeTeam", "FTHG", "HTHG", "HTR", "FTR"]].copy()
            ht_home.columns = ["date", "team", "fthg", "hthg", "htr", "ftr"]
            ht_home["sh_goals"]  = ht_home["fthg"] - ht_home["hthg"]
            ht_home["lead_hold"] = ((ht_home["htr"] == "H") & (ht_home["ftr"] == "H")).astype(int)
            ht_home["comeback"]  = ((ht_home["htr"] == "A") & (ht_home["ftr"].isin(["H", "D"]))).astype(int)
            ht_home = ht_home.sort_values(["team", "date"])

            ht_away = ht[["Date", "AwayTeam", "FTAG", "HTAG", "HTR", "FTR"]].copy()
            ht_away.columns = ["date", "team", "ftag", "htag", "htr", "ftr"]
            ht_away["sh_goals"]  = ht_away["ftag"] - ht_away["htag"]
            ht_away["lead_hold"] = ((ht_away["htr"] == "A") & (ht_away["ftr"] == "A")).astype(int)
            ht_away["comeback"]  = ((ht_away["htr"] == "H") & (ht_away["ftr"].isin(["A", "D"]))).astype(int)
            ht_away = ht_away.sort_values(["team", "date"])

            def rolling_ht(group):
                shifted = group[["sh_goals", "lead_hold", "comeback"]].shift(1)
                return shifted.rolling(5, min_periods=1).mean()

            rh = ht_home.groupby("team", group_keys=False).apply(rolling_ht, include_groups=False)
            ht_home["home_2nd_half_goals_avg5"] = rh["sh_goals"]
            ht_home["home_lead_hold_rate"]      = rh["lead_hold"]
            ht_home["home_comeback_rate"]       = rh["comeback"]

            ra = ht_away.groupby("team", group_keys=False).apply(rolling_ht, include_groups=False)
            ht_away["away_2nd_half_goals_avg5"] = ra["sh_goals"]
            ht_away["away_lead_hold_rate"]      = ra["lead_hold"]
            ht_away["away_comeback_rate"]       = ra["comeback"]

            df = df.merge(
                ht_home[["date", "team", "home_2nd_half_goals_avg5", "home_lead_hold_rate", "home_comeback_rate"]]
                .rename(columns={"team": "HomeTeam", "date": "Date"}),
                on=["Date", "HomeTeam"], how="left"
            )
            df = df.merge(
                ht_away[["date", "team", "away_2nd_half_goals_avg5", "away_lead_hold_rate", "away_comeback_rate"]]
                .rename(columns={"team": "AwayTeam", "date": "Date"}),
                on=["Date", "AwayTeam"], how="left"
            )

            logging.info(f"Step 7 complete. Shape: {df.shape}")
            print("Step 7: Half-time features computed...DONE")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── Step 8 ────────────────────────────────────────────────────────────────
    def compute_referee(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            print("Feature Engineering Step 8: Computing referee features (slow — row-by-row)...")
            logging.info("Step 8: Referee historical features")

            ref_src = df[["Date", "Referee", "HY", "AY", "HF", "AF", "FTR"]].copy()
            ref_src["total_yellows"] = ref_src["HY"] + ref_src["AY"]
            ref_src["total_fouls"]   = ref_src["HF"] + ref_src["AF"]
            ref_src["home_win"]      = (ref_src["FTR"] == "H").astype(int)
            ref_src = ref_src.sort_values("Date").reset_index(drop=True)

            ref_features = []
            for _, row in ref_src.iterrows():
                prior = ref_src[(ref_src["Referee"] == row["Referee"]) & (ref_src["Date"] < row["Date"])]
                if len(prior) == 0:
                    ref_features.append({"ref_avg_yellows": np.nan, "ref_avg_fouls": np.nan,
                                         "ref_home_win_rate": np.nan})
                else:
                    ref_features.append({
                        "ref_avg_yellows":   prior["total_yellows"].mean(),
                        "ref_avg_fouls":     prior["total_fouls"].mean(),
                        "ref_home_win_rate": prior["home_win"].mean(),
                    })

            df = pd.concat([df, pd.DataFrame(ref_features, index=ref_src.index)], axis=1)

            logging.info(f"Step 8 complete. Shape: {df.shape}")
            print("Step 8: Referee features computed...DONE")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── Step 9 ────────────────────────────────────────────────────────────────
    def compute_temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            print("Feature Engineering Step 9: Computing temporal features...")
            logging.info("Step 9: Temporal and fixture features")

            df["match_week"]  = df.groupby("season_id")["Date"].rank(method="dense").astype(int)
            df["season_phase"] = df["match_week"].apply(
                lambda w: 0 if w <= 10 else (1 if w <= 28 else 2)
            )

            all_matches = pd.concat([
                df[["Date", "HomeTeam"]].rename(columns={"HomeTeam": "team"}),
                df[["Date", "AwayTeam"]].rename(columns={"AwayTeam": "team"}),
            ]).sort_values(["team", "Date"])

            all_matches["days_rest"] = all_matches.groupby("team")["Date"].diff().dt.days

            home_rest = all_matches.rename(columns={"team": "HomeTeam", "days_rest": "home_days_rest"})
            away_rest = all_matches.rename(columns={"team": "AwayTeam", "days_rest": "away_days_rest"})

            df = df.merge(home_rest[["Date", "HomeTeam", "home_days_rest"]], on=["Date", "HomeTeam"], how="left")
            df = df.merge(away_rest[["Date", "AwayTeam", "away_days_rest"]], on=["Date", "AwayTeam"], how="left")

            median_rest = df["home_days_rest"].median()
            df["home_days_rest"] = df["home_days_rest"].fillna(median_rest)
            df["away_days_rest"] = df["away_days_rest"].fillna(median_rest)

            logging.info(f"Step 9 complete. Shape: {df.shape}")
            print("Step 9: Temporal features computed...DONE")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── Step 10 ───────────────────────────────────────────────────────────────
    def fill_nulls_and_save(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            print("Feature Engineering Step 10: Filling nulls and saving...")
            logging.info("Step 10: NaN imputation and save")

            h2h_cols = ["h2h_home_win_rate", "h2h_avg_total_goals", "h2h_btts_rate"]
            ref_cols = ["ref_avg_yellows", "ref_avg_fouls", "ref_home_win_rate"]
            for col in h2h_cols + ref_cols:
                df[col] = df[col].fillna(df[col].mean())

            rolling_like = [c for c in df.columns if any(x in c for x in [
                "points_last", "goals_scored", "goals_conceded", "sot_avg", "clean_sheets",
                "win_rate", "goals_avg", "conversion_rate", "sot_ratio",
                "2nd_half_goals", "lead_hold", "comeback",
            ])]
            df[rolling_like] = df[rolling_like].fillna(0)

            df.to_csv(self.config.output_csv, index=False)

            logging.info(f"Saved to {self.config.output_csv} | shape: {df.shape}")
            print(f"Step 10: Saved to {self.config.output_csv}")
            print(f"Final shape: {df.shape} | Remaining NaNs: {df.isnull().sum().sum()}")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    fe = FeatureEngineering()

    df = pd.read_csv(fe.config.cleaned_csv, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"Loaded: {df.shape}")

    df = fe.encode_targets(df)
    df = fe.encode_categoricals(df)
    df = fe.compute_team_form(df)
    df = fe.compute_home_away_split(df)
    df = fe.compute_h2h(df)
    df = fe.compute_shot_quality(df)
    df = fe.compute_halftime(df)
    df = fe.compute_referee(df)
    df = fe.compute_temporal(df)
    df = fe.fill_nulls_and_save(df)
