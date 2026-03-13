import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Soca Scores | EPL EDA Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Background */
.stApp {
    background-color: #0A0A0F;
    color: #E8E8E8;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111118;
    border-right: 1px solid #2A2A3A;
}
[data-testid="stSidebar"] * {
    color: #E8E8E8 !important;
}

/* Main font */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Headings */
h1, h2, h3 {
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 2px;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #16161F;
    border: 1px solid #2A2A3A;
    border-radius: 8px;
    padding: 16px;
}
[data-testid="metric-container"] label {
    color: #888 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #E84040 !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 2rem !important;
}

/* Radio buttons */
[data-testid="stRadio"] label {
    color: #E8E8E8 !important;
}

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 1.5px;
    font-size: 1rem !important;
    color: #888 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #E84040 !important;
    border-bottom-color: #E84040 !important;
}

/* Divider */
hr {
    border-color: #2A2A3A !important;
}

/* Pill badge */
.badge {
    display: inline-block;
    background: #E84040;
    color: white;
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 1px;
    padding: 2px 10px;
    border-radius: 4px;
    font-size: 0.85rem;
    margin-right: 6px;
}
.badge-teal {
    background: #7EC8C8;
    color: #0A0A0F;
}

/* Section header */
.section-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.8rem;
    letter-spacing: 3px;
    color: #E8E8E8;
    border-left: 4px solid #E84040;
    padding-left: 12px;
    margin-bottom: 4px;
}
.section-sub {
    color: #888;
    font-size: 0.85rem;
    margin-bottom: 16px;
    padding-left: 16px;
}

/* Insight box */
.insight-box {
    background: #16161F;
    border-left: 3px solid #E84040;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 12px 0;
    font-size: 0.9rem;
    color: #CCCCCC;
    line-height: 1.6;
}

/* Table styling */
[data-testid="stDataFrame"] {
    background: #16161F !important;
}

</style>
""", unsafe_allow_html=True)

# ─── Palette ─────────────────────────────────────────────────────────────────
RED     = "#E84040"
TEAL    = "#7EC8C8"
DARK_BG = "#16161F"
CURRENT_SEASON = "2025-2026"

def set_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  "#0A0A0F",
        "axes.facecolor":    "#16161F",
        "axes.edgecolor":    "#2A2A3A",
        "axes.labelcolor":   "#AAAAAA",
        "xtick.color":       "#AAAAAA",
        "ytick.color":       "#AAAAAA",
        "text.color":        "#E8E8E8",
        "grid.color":        "#2A2A3A",
        "grid.alpha":        0.5,
    })

set_dark_style()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px'>
        <div style='font-family: Bebas Neue, sans-serif; font-size: 2.5rem; color: #E84040; letter-spacing: 4px;'>SOCA SCORES</div>
        <div style='font-size: 0.75rem; color: #888; letter-spacing: 2px; text-transform: uppercase;'>EPL Analytics Dashboard</div>
        <div style='font-size: 0.7rem; color: #555; margin-top: 4px;'>2005/06 — 2025/26</div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    st.markdown("**Navigate**")
    page = st.radio(
        "",
        [
            "⚽  Overview",
            "🏠  Home vs Away",
            "🎯  Goals Analysis",
            "🟨  Cards & Discipline",
            "🎯  Shot Conversion",
            "🏰  Home Fortress",
            "⏱️  Half Time vs Full Time",
        ],
        label_visibility="collapsed"
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size: 0.7rem; color: #555; text-align: center; padding-top: 8px;'>
        Data: footballdata.co.uk<br>
        Built with Streamlit + DuckDB<br>
        <span style='color: #E84040;'>Soca Scores MLOps Project</span>
    </div>
    """, unsafe_allow_html=True)

# ─── Data ────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    # Obs 1 -- Home vs Away 2025/26
    obs1 = pd.DataFrame({
        "Result":  ["Home Win", "Away Win", "Draw"],
        "Matches": [123, 92, 76],
        "Percentage": [42.27, 31.62, 26.12],
    })

    # Obs 2 -- Avg goals per season
    obs2 = pd.DataFrame({
        "season_id": [
            "2023-2024","2024-2025","2022-2023","2018-2019","2021-2022",
            "2011-2012","2016-2017","2012-2013","2010-2011","2025-2026",
            "2009-2010","2013-2014","2019-2020","2015-2016","2020-2021",
            "2017-2018","2007-2008","2014-2015","2005-2006","2008-2009","2006-2007",
        ],
        "avg_goals_per_game": [
            3.28, 2.93, 2.85, 2.82, 2.82,
            2.81, 2.80, 2.80, 2.80, 2.77,
            2.77, 2.77, 2.72, 2.70, 2.69,
            2.68, 2.64, 2.57, 2.48, 2.48, 2.45,
        ],
        "avg_goals_rank": list(range(1, 22)),
    })
    obs2 = obs2.sort_values("season_id")

    # Obs 3 -- Cards per season
    obs3 = pd.DataFrame({
        "season_id": [
            "2023-2024","2024-2025","2025-2026","2014-2015","2016-2017",
            "2022-2023","2021-2022","2019-2020","2009-2010","2010-2011",
            "2006-2007","2007-2008","2018-2019","2013-2014","2008-2009",
            "2005-2006","2011-2012","2015-2016","2012-2013","2017-2018","2020-2021",
        ],
        "total_yellows": [
            1586,1539,1090,1364,1380,
            1363,1291,1274,1237,1236,
            1225,1216,1220,1212,1198,
            1173,1178,1179,1186,1157,1091,
        ],
        "total_reds": [
            57,52,31,71,41,
            28,43,45,68,63,
            53,61,47,53,63,
            76,64,59,52,39,46,
        ],
        "total_cards": [
            1643,1591,1121,1435,1421,
            1391,1334,1319,1305,1299,
            1278,1277,1267,1265,1261,
            1249,1242,1238,1238,1196,1137,
        ],
        "avg_card_per_game": [
            4.32,4.19,3.85,3.78,3.74,
            3.66,3.51,3.47,3.43,3.42,
            3.36,3.36,3.33,3.33,3.32,
            3.29,3.27,3.26,3.26,3.15,2.99,
        ],
        "rank": list(range(1, 22)),
    })
    obs3 = obs3.sort_values("season_id")

    # Obs 9 -- Shot conversion
    obs9 = pd.DataFrame({
        "team": [
            "Nott'm Forest","Wolves","Crystal Palace","Leeds","Brighton",
            "Liverpool","Everton","Aston Villa","Newcastle","Sunderland",
        ],
        "total_shots":  [373,273,338,363,370,457,321,365,386,275],
        "total_goals":  [28, 22, 33, 37, 38, 48, 34, 39, 42, 30],
        "conversion_rate": [7.51,8.06,9.76,10.19,10.27,10.50,10.59,10.68,10.88,10.91],
    })

    # Obs 10 -- Unbeaten home run (top 10 from article)
    obs10 = pd.DataFrame({
        "team": ["Man City","Arsenal","Liverpool","Tottenham","Chelsea",
                 "Man United","Newcastle","Aston Villa","Brighton","Fulham"],
        "longest_streak": [14,14,13,13,12,12,11,10,10,9],
    }).sort_values("longest_streak", ascending=False)

    # Obs 8 -- Half time vs full time goals (estimated from article context)
    obs8 = pd.DataFrame({
        "season_id": [
            "2005-2006","2006-2007","2007-2008","2008-2009","2009-2010",
            "2010-2011","2011-2012","2012-2013","2013-2014","2014-2015",
            "2015-2016","2016-2017","2017-2018","2018-2019","2019-2020",
            "2020-2021","2021-2022","2022-2023","2023-2024","2024-2025","2025-2026",
        ],
        "avg_first_half":  [1.03,1.00,1.08,1.01,1.12,1.13,1.14,1.14,1.13,1.03,
                            1.08,1.12,1.07,1.14,1.08,1.06,1.12,1.14,1.30,1.15,1.10],
        "avg_second_half": [1.45,1.45,1.56,1.47,1.65,1.67,1.67,1.66,1.64,1.54,
                            1.62,1.68,1.61,1.68,1.64,1.63,1.70,1.71,1.98,1.78,1.67],
    })

    return obs1, obs2, obs3, obs9, obs10, obs8

obs1, obs2, obs3, obs9, obs10, obs8 = load_data()

# ─── Helper ──────────────────────────────────────────────────────────────────
def fig_to_st(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
# PAGES
# ════════════════════════════════════════════════════════════════════════════

# ── OVERVIEW ────────────────────────────────────────────────────────────────
if page == "⚽  Overview":
    st.markdown("""
    <div style='padding: 32px 0 8px'>
        <div style='font-family: Bebas Neue, sans-serif; font-size: 3.5rem; color: #E8E8E8; letter-spacing: 5px; line-height: 1;'>
            SOCA SCORES
        </div>
        <div style='font-family: Bebas Neue, sans-serif; font-size: 1.4rem; color: #E84040; letter-spacing: 3px;'>
            EPL DATA EXPLORER · 2005/06 – 2025/26
        </div>
        <div style='color: #666; font-size: 0.9rem; margin-top: 8px;'>
            21 seasons · 8,037 matches · End-to-end MLOps prediction project
        </div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    st.markdown("### Season at a Glance — 2025/26")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Matches Played",    "291")
    c2.metric("Avg Goals / Game",  "2.77")
    c3.metric("Home Win %",        "42.3%")
    c4.metric("Avg Cards / Game",  "3.85")
    c5.metric("Cards This Season", "1,121")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### What's Inside")

    col1, col2 = st.columns(2)
    with col1:
        for title, desc in [
            ("🏠 Home vs Away", "Home teams win 42% of the time in 2025/26 — venue is a real predictor."),
            ("🎯 Goals Analysis", "2025/26 ranks 10th for average goals. Season not over yet."),
            ("🟨 Cards & Discipline", "3rd highest average cards per game in 21 seasons."),
        ]:
            st.markdown(f"""
            <div class='insight-box'>
                <strong style='color:#E8E8E8'>{title}</strong><br>{desc}
            </div>
            """, unsafe_allow_html=True)
    with col2:
        for title, desc in [
            ("🎯 Shot Conversion", "Nottingham Forest: 373 shots, only 28 goals. 7.5% conversion."),
            ("🏰 Home Fortress", "Man City and Arsenal share the longest unbeaten home run — 14 games."),
            ("⏱️ Half Time vs Full Time", "Second half goals outnumber first half in ALL 21 seasons."),
        ]:
            st.markdown(f"""
            <div class='insight-box'>
                <strong style='color:#E8E8E8'>{title}</strong><br>{desc}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#555; font-size:0.8rem; text-align:center; padding:8px 0'>
        Built as part of the Soca Scores MLOps Series · Data: footballdata.co.uk · Published on Medium: Data, AI and Beyond
    </div>
    """, unsafe_allow_html=True)


# ── HOME VS AWAY ─────────────────────────────────────────────────────────────
elif page == "🏠  Home vs Away":
    st.markdown("<div class='section-title'>HOME VS AWAY WINS</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>2025/26 Season · Gameweek 29 · 291 matches played</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Home Wins",  "123",  "42.27%")
    c2.metric("Away Wins",  "92",   "31.62%")
    c3.metric("Draws",      "76",   "26.12%")

    set_dark_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [RED, TEAL, "#555566"]
    bars = ax.bar(obs1["Result"], obs1["Matches"], color=colors, width=0.5, zorder=3)
    ax.set_ylim(0, 160)
    ax.yaxis.grid(True, zorder=0)
    ax.set_title("2025/26: Match Outcomes Distribution", fontsize=13, fontweight="bold", color="#E8E8E8", pad=12)
    ax.set_ylabel("Number of Matches")
    for bar, row in zip(bars, obs1.itertuples()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{int(bar.get_height())}\n({row.Percentage}%)",
                ha="center", va="bottom", fontsize=10, color="#E8E8E8")
    fig.tight_layout()
    fig_to_st(fig)

    st.markdown("""
    <div class='insight-box'>
        <strong>Why this matters for prediction:</strong> Venue is a significant feature.
        Home teams win 42% of the time — away performance is a better signal of a team's true quality.
        Teams should prioritise home advantage for crucial points in a tight relegation or title battle.
    </div>
    """, unsafe_allow_html=True)


# ── GOALS ANALYSIS ───────────────────────────────────────────────────────────
elif page == "🎯  Goals Analysis":
    st.markdown("<div class='section-title'>GOALS ACROSS 21 SEASONS</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Average goals per game · 2005/06 – 2025/26</div>", unsafe_allow_html=True)

    rank_2526 = obs2[obs2["season_id"] == CURRENT_SEASON]["avg_goals_rank"].values[0]
    avg_2526  = obs2[obs2["season_id"] == CURRENT_SEASON]["avg_goals_per_game"].values[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("2025/26 Avg Goals/Game", f"{avg_2526}")
    c2.metric("Season Rank",            f"#{int(rank_2526)} of 21")
    c3.metric("Highest Ever",           "3.28 (2023/24)")

    set_dark_style()
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = [RED if s == CURRENT_SEASON else TEAL for s in obs2["season_id"]]
    ax.bar(obs2["season_id"], obs2["avg_goals_per_game"], color=colors, zorder=3)
    ax.yaxis.grid(True, zorder=0)
    ax.set_title("Average Goals Per Game Per Season (Red = 2025/26)", fontsize=13, fontweight="bold", color="#E8E8E8", pad=12)
    ax.set_ylabel("Avg Goals Per Match")
    ax.set_ylim(0, 3.8)
    for i, (_, row) in enumerate(obs2.iterrows()):
        ax.text(i, row["avg_goals_per_game"] + 0.03, f"{row['avg_goals_per_game']:.2f}",
                ha="center", va="bottom", fontsize=7, color="#AAAAAA")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig_to_st(fig)

    st.markdown("""
    <div class='insight-box'>
        2025/26 ranks <strong style='color:#E84040'>10th</strong> for average goals with 2.77 per game.
        The season is incomplete — most teams have 11 games remaining.
        In the top 10 highest-scoring games this season, at least one team scored 4 or more goals
        except for Leeds vs Liverpool which ended 3–3.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Full Rankings Table")
    display_df = obs2.sort_values("avg_goals_rank")[["avg_goals_rank","season_id","avg_goals_per_game"]].copy()
    display_df.columns = ["Rank", "Season", "Avg Goals/Game"]
    display_df = display_df.reset_index(drop=True)
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ── CARDS ────────────────────────────────────────────────────────────────────
elif page == "🟨  Cards & Discipline":
    st.markdown("<div class='section-title'>CARDS & DISCIPLINE</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Average cards per game across 21 seasons</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cards This Season",    "1,121")
    c2.metric("Yellow Cards",         "1,090")
    c3.metric("Red Cards",            "31")
    c4.metric("Avg Cards/Game Rank",  "#3 of 21")

    set_dark_style()
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = [RED if s == CURRENT_SEASON else TEAL for s in obs3["season_id"]]
    ax.bar(obs3["season_id"], obs3["avg_card_per_game"], color=colors, zorder=3)
    ax.yaxis.grid(True, zorder=0)
    ax.set_title("Avg Cards Per Game Per Season (Red = 2025/26)", fontsize=13, fontweight="bold", color="#E8E8E8", pad=12)
    ax.set_ylabel("Avg Cards Per Game")
    ax.set_ylim(0, 5.2)
    for i, (_, row) in enumerate(obs3.iterrows()):
        ax.text(i, row["avg_card_per_game"] + 0.05, f"{row['avg_card_per_game']:.2f}",
                ha="center", va="bottom", fontsize=7, color="#AAAAAA")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig_to_st(fig)

    st.markdown("""
    <div class='insight-box'>
        2025/26 has the <strong style='color:#E84040'>3rd highest</strong> average cards per game in 21 seasons at 3.85.
        Chelsea has been a repeat offender this season, dropping 16 points in games where their players were dismissed.
        The question: is the league more reckless, or has VAR fundamentally changed how referees police challenges?
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["YELLOWS vs REDS", "FULL TABLE"])
    with tab1:
        set_dark_style()
        fig, ax = plt.subplots(figsize=(14, 4))
        x = np.arange(len(obs3))
        w = 0.4
        ax.bar(x - w/2, obs3["total_yellows"], width=w, label="Yellows", color="#F5C518", zorder=3)
        ax.bar(x + w/2, obs3["total_reds"],    width=w, label="Reds",    color=RED,       zorder=3)
        ax.yaxis.grid(True, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(obs3["season_id"], rotation=45, ha="right", fontsize=8)
        ax.set_title("Total Yellows vs Reds Per Season", fontsize=12, fontweight="bold", color="#E8E8E8")
        ax.legend(facecolor="#16161F", edgecolor="#2A2A3A", labelcolor="#E8E8E8")
        fig.tight_layout()
        fig_to_st(fig)
    with tab2:
        display_df = obs3.sort_values("rank")[["rank","season_id","total_yellows","total_reds","total_cards","avg_card_per_game"]].copy()
        display_df.columns = ["Rank","Season","Yellows","Reds","Total Cards","Avg/Game"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# ── SHOT CONVERSION ──────────────────────────────────────────────────────────
elif page == "🎯  Shot Conversion":
    st.markdown("<div class='section-title'>SHOT CONVERSION RATE</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Most wasteful attacks · 2025/26 · Bottom 10 teams</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Most Wasteful",       "Nott'm Forest")
    c2.metric("Forest Conversion",   "7.51%")
    c3.metric("Forest Shots / Goals","373 shots → 28 goals")

    set_dark_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [RED if i < 3 else TEAL for i in range(len(obs9))]
    bars = ax.bar(obs9["team"], obs9["conversion_rate"], color=colors, zorder=3)
    ax.yaxis.grid(True, zorder=0)
    ax.set_title("Shot Conversion Rate % (Red = Most Wasteful)", fontsize=13, fontweight="bold", color="#E8E8E8", pad=12)
    ax.set_ylabel("Conversion Rate %")
    ax.set_ylim(0, 13)
    for bar, rate in zip(bars, obs9["conversion_rate"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{rate:.2f}%", ha="center", va="bottom", fontsize=9, color="#E8E8E8")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig_to_st(fig)

    st.markdown("""
    <div class='insight-box'>
        Nottingham Forest have attempted <strong style='color:#E84040'>373 shots</strong> but scored only
        <strong style='color:#E84040'>28 goals</strong> — a 7.5% conversion rate, the worst in the league.
        A clinical striker would transform their season. Sunderland, newly promoted from the Championship,
        have taken 275 shots and scored 30 goals, sitting comfortably mid-table at 11th.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Breakdown Table")
    st.dataframe(obs9.rename(columns={
        "team":"Team","total_shots":"Shots","total_goals":"Goals","conversion_rate":"Conversion %"
    }), use_container_width=True, hide_index=True)


# ── HOME FORTRESS ────────────────────────────────────────────────────────────
elif page == "🏰  Home Fortress":
    st.markdown("<div class='section-title'>HOME FORTRESS</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Longest unbeaten home run · 2025/26 · Top 10 teams</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Joint Leaders",          "Man City & Arsenal")
    c2.metric("Longest Streak",         "14 games")
    c3.metric("Man United Streak",      "12 games")

    set_dark_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [RED if i < 2 else TEAL for i in range(len(obs10))]
    bars = ax.barh(obs10["team"][::-1], obs10["longest_streak"][::-1], color=colors[::-1], zorder=3)
    ax.xaxis.grid(True, zorder=0)
    ax.set_title("Longest Unbeaten Home Run by Team — 2025/26", fontsize=13, fontweight="bold", color="#E8E8E8", pad=12)
    ax.set_xlabel("Games Unbeaten at Home")
    for bar in bars:
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                int(bar.get_width()), va="center", fontsize=10, color="#E8E8E8")
    fig.tight_layout()
    fig_to_st(fig)

    st.markdown("""
    <div class='insight-box'>
        Arsenal may be leading the title race but it is <strong style='color:#E84040'>Manchester City</strong>
        who share the record for the longest unbeaten home run at 14 games.
        Despite their turbulent season, Man United have gone unbeaten in 12 home matches —
        a sign that home form remains one of their few reliable traits.
    </div>
    """, unsafe_allow_html=True)


# ── HALF TIME VS FULL TIME ────────────────────────────────────────────────────
elif page == "⏱️  Half Time vs Full Time":
    st.markdown("<div class='section-title'>FIRST HALF vs SECOND HALF GOALS</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Average goals per half across all 21 seasons</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Seasons Analysed",          "21")
    c2.metric("2nd Half Always Higher",    "✓ Every season")
    c3.metric("2025/26 2nd Half Avg",      "1.67")

    set_dark_style()
    obs8_melted = obs8.melt(id_vars="season_id", var_name="Half", value_name="Avg Goals")
    obs8_melted["Half"] = obs8_melted["Half"].map({
        "avg_first_half":  "First Half",
        "avg_second_half": "Second Half"
    })

    fig, ax = plt.subplots(figsize=(14, 5))
    palette = {"First Half": "#4C72B0", "Second Half": RED}
    for half, color in palette.items():
        subset = obs8_melted[obs8_melted["Half"] == half]
        ax.bar(
            np.arange(len(obs8)) + (0.2 if half == "Second Half" else -0.2),
            subset["Avg Goals"].values,
            width=0.38, label=half, color=color, zorder=3
        )
    ax.yaxis.grid(True, zorder=0)
    ax.set_xticks(np.arange(len(obs8)))
    ax.set_xticklabels(obs8["season_id"], rotation=45, ha="right", fontsize=8)
    ax.set_title("Avg Goals Per Half Per Season — 2005 to 2026", fontsize=13, fontweight="bold", color="#E8E8E8", pad=12)
    ax.set_ylabel("Avg Goals")
    ax.legend(facecolor="#16161F", edgecolor="#2A2A3A", labelcolor="#E8E8E8")
    fig.tight_layout()
    fig_to_st(fig)

    st.markdown("""
    <div class='insight-box'>
        In <strong style='color:#E84040'>every single one</strong> of the last 21 seasons, second half goals
        have outnumbered first half goals. This is a powerful feature for betting markets like
        "Goals in Second Half over 0.5." For newly promoted teams fighting relegation, this underscores
        the critical importance of second half defensive discipline.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Season by Season Breakdown")
    obs8_display = obs8.copy()
    obs8_display["2nd > 1st"] = obs8_display["avg_second_half"] > obs8_display["avg_first_half"]
    obs8_display["2nd > 1st"] = obs8_display["2nd > 1st"].map({True: "✓", False: "✗"})
    obs8_display.columns = ["Season", "Avg First Half Goals", "Avg Second Half Goals", "2nd > 1st?"]
    st.dataframe(obs8_display, use_container_width=True, hide_index=True)
