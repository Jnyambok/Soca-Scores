import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from src.components.model_inference import ModelInference

st.set_page_config(
    page_title="Soca Scores — EPL Predictor",
    page_icon="⚽",
    layout="centered",
)


@st.cache_resource
def load_inferencer():
    return ModelInference()


inferencer = load_inferencer()

teams    = sorted(inferencer.team_encoder.classes_.tolist())
referees = sorted(inferencer.referee_encoder.classes_.tolist())

st.title("Soca Scores")
st.subheader("EPL Match Prediction")
st.caption("Predicts match result, BTTS, over/under goals, and expected goals.")

st.divider()

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home Team", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
with col2:
    away_team = st.selectbox("Away Team", teams, index=teams.index("Chelsea") if "Chelsea" in teams else 1)

col3, col4 = st.columns(2)
with col3:
    referee = st.selectbox("Referee", referees)
with col4:
    match_week = st.number_input("Match Week", min_value=1, max_value=38, value=33)

match_date = st.date_input("Match Date")

st.divider()

if st.button("Predict", use_container_width=True, type="primary"):
    if home_team == away_team:
        st.error("Home and away teams cannot be the same.")
    else:
        try:
            with st.spinner("Running predictions..."):
                result = inferencer.predict(
                    home_team=home_team,
                    away_team=away_team,
                    date=str(match_date),
                    referee=referee,
                    match_week=int(match_week),
                )

            preds = result["predictions"]

            st.success(f"**{home_team} vs {away_team}** — {match_date}")

            st.subheader("Match Result")
            res = preds["result"]
            col_hw, col_d, col_aw = st.columns(3)
            col_hw.metric("Home Win", f"{res['home_win']*100:.1f}%")
            col_d.metric("Draw", f"{res['draw']*100:.1f}%")
            col_aw.metric("Away Win", f"{res['away_win']*100:.1f}%")
            st.info(f"Predicted: **{res['prediction']}**")

            st.divider()

            st.subheader("Goals")
            col_g, col_o25, col_o15 = st.columns(3)
            col_g.metric("Expected Goals", preds["total_goals"]["predicted"])
            col_o25.metric("Over 2.5", f"{preds['over_2_5']['probability']*100:.1f}%",
                           delta=preds["over_2_5"]["prediction"])
            col_o15.metric("Over 1.5", f"{preds['over_1_5']['probability']*100:.1f}%",
                           delta=preds["over_1_5"]["prediction"])

            st.divider()

            st.subheader("Both Teams to Score")
            btts = preds["btts"]
            st.metric("BTTS Probability", f"{btts['probability']*100:.1f}%",
                      delta=btts["prediction"])

        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Prediction failed: {e}")
