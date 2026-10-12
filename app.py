"""PL Predictions as a web app.   Run: streamlit run app.py

Three things: this matchday's predictions next to the bookmakers, any fixture you pick
(home team v away team, predicted with the current ratings and form), and the record so
far this season plus the seven-season backtest. Uses the results-only model, the one that
can run live (the lineup model needs confirmed XIs, which are not available ahead of time).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from build_dataset import RAW, parse_openfootball, season_of  # noqa: E402
from evaluate import CLASSES, RESULTS_FEATS, encode, make_model  # noqa: E402
from features import build_features  # noqa: E402

st.set_page_config(page_title="PL Predictions", page_icon="⚽", layout="wide")
LABEL = {"H": "Home win", "D": "Draw", "A": "Away win"}
REPO = "https://github.com/jcosta92/PL_Predictions"


@st.cache_data(show_spinner=False)
def load():
    played = pd.read_csv(ROOT / "data/processed/matches.csv", parse_dates=["Date"])
    fixtures = parse_openfootball(RAW / "openfootball_pl_2026-27.txt", 2026, played=False)
    fixtures["Season"] = fixtures.Date.apply(season_of)
    elo = json.load(open(ROOT / "reports/metrics.json"))["elo"]
    return played, fixtures, elo


@st.cache_resource(show_spinner=False)
def model_and_ratings():
    played, fixtures, elo = load()
    df = build_features(pd.concat([played, fixtures], ignore_index=True), **elo)
    train = df.FTHG.notna() & (df.Season >= "2001/02")
    m = make_model("logreg").fit(df.loc[train, RESULTS_FEATS], encode(df.loc[train, "Result"]))
    return m, df, elo


def predict_pair(home: str, away: str) -> np.ndarray:
    """Probabilities for home v away played now: append a one-off fixture and rebuild features."""
    played, _, elo = load()
    m, _, _ = model_and_ratings()
    fx = pd.DataFrame([{"Date": played.Date.max() + pd.Timedelta(days=1), "HomeTeam": home, "AwayTeam": away,
                        "Season": played.Season.max()}])
    df = build_features(pd.concat([played, fx], ignore_index=True), **elo)
    return m.predict_proba(df[df.FTHG.isna()].tail(1)[RESULTS_FEATS])[0]


def market(df):
    inv = 1 / df[["OddH", "OddD", "OddA"]].to_numpy(dtype=float)
    return inv / inv.sum(1, keepdims=True)


def prob_row(p, pick=None):
    cols = st.columns(3)
    for col, k, v in zip(cols, "HDA", p):
        col.metric(LABEL[k], f"{100 * v:.0f}%", help="model's pick" if k == pick else None)


played, fixtures, elo = load()
m, df_all, _ = model_and_ratings()
current = played[played.Season == played.Season.max()]
teams = sorted(set(current.HomeTeam) | set(current.AwayTeam) | set(fixtures.HomeTeam))

st.title("PL Predictions")
st.caption(f"Premier League {played.Season.max()} · results-only model · results up to {played.Date.max():%d %b %Y} · "
           f"[code and write-up]({REPO})")

tab_next, tab_pick, tab_season, tab_back = st.tabs(["This matchday", "Pick a fixture", "Season so far", "Backtest"])

with tab_next:
    nxt = pd.read_csv(ROOT / "reports/next_matchday_predictions.csv", parse_dates=["Date"])
    st.subheader(f"Matchday {nxt.Date.min():%d}–{nxt.Date.max():%d %b %Y}")
    show = nxt.assign(Fixture=nxt.HomeTeam + " v " + nxt.AwayTeam, Pick=nxt.Pick.map(LABEL),
                      **{"Home %": (100 * nxt.pH).round(0), "Draw %": (100 * nxt.pD).round(0), "Away %": (100 * nxt.pA).round(0),
                         "Elo home": nxt.HomeElo.round(0), "Elo away": nxt.AwayElo.round(0)})
    st.dataframe(show[["Date", "Fixture", "Home %", "Draw %", "Away %", "Pick", "Elo home", "Elo away"]]
                 .assign(Date=show.Date.dt.strftime("%a %d %b")).set_index("Date"), width="stretch")
    st.caption("Bookmakers' odds for upcoming fixtures are added when data/raw/football_data_full/fixtures.csv is present.")

with tab_pick:
    c1, c2 = st.columns(2)
    home = c1.selectbox("Home team", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
    away = c2.selectbox("Away team", [t for t in teams if t != home], index=0)
    p = predict_pair(home, away)
    st.subheader(f"{home} v {away}, if played now")
    prob_row(p, "HDA"[p.argmax()])
    ratings = df_all[df_all.FTHG.isna()].drop_duplicates(["HomeTeam"]).set_index("HomeTeam")["HomeElo"].to_dict() | \
        df_all[df_all.FTHG.isna()].drop_duplicates(["AwayTeam"]).set_index("AwayTeam")["AwayElo"].to_dict()
    st.caption(f"Current Elo: {home} {ratings.get(home, float('nan')):.0f}, {away} {ratings.get(away, float('nan')):.0f}. "
               "Home advantage is worth about 50 Elo points. The model also uses each side's last 5 and 10 results and rest days.")

with tab_season:
    rec = pd.read_csv(ROOT / "reports/test_2026_27_predictions.csv", parse_dates=["Date"])
    rec = rec.merge(current[["Date", "HomeTeam", "AwayTeam", "OddH", "OddD", "OddA"]], on=["Date", "HomeTeam", "AwayTeam"], how="left")
    ok = rec.OddH.notna()
    rec["Bookmakers"] = None
    if ok.any():
        rec.loc[ok, "Bookmakers"] = np.array(list("HDA"))[market(rec[ok]).argmax(1)]
    k = st.columns(4)
    k[0].metric("Model accuracy", f"{100 * rec.Correct.mean():.0f}%", f"{len(rec)} matches", delta_color="off")
    if ok.any():
        k[1].metric("Bookmakers", f"{100 * (rec.loc[ok, 'Bookmakers'] == rec.loc[ok, 'Result']).mean():.0f}%", f"{ok.sum()} with odds", delta_color="off")
        k[2].metric("Model on those", f"{100 * rec.loc[ok, 'Correct'].mean():.0f}%", f"{ok.sum()} matches", delta_color="off")
    k[3].metric("Home wins", f"{100 * (rec.Result == 'H').mean():.0f}%", "long run 46%", delta_color="off")
    dis = rec[ok & (rec.Bookmakers != rec.Pick)]
    if len(dis):
        st.markdown("**Where the model and the bookmakers disagreed**")
        st.dataframe(dis.assign(Fixture=dis.HomeTeam + " v " + dis.AwayTeam, Score=dis.FTHG.astype(int).astype(str) + "–" + dis.FTAG.astype(int).astype(str),
                                Model=dis.Pick.map(LABEL), Bookmakers=dis.Bookmakers.map(LABEL), Result=dis.Result.map(LABEL))
                     [["Date", "Fixture", "Score", "Result", "Model", "Bookmakers"]].set_index("Date"), width="stretch")
    st.markdown("**All matches**")
    st.dataframe(rec.sort_values("Date", ascending=False).assign(
        Fixture=rec.HomeTeam + " v " + rec.AwayTeam, Score=rec.FTHG.astype(int).astype(str) + "–" + rec.FTAG.astype(int).astype(str),
        Model=rec.Pick.map(LABEL), Bookmakers=rec.Bookmakers.map(LABEL), Result=rec.Result.map(LABEL),
        **{"Home %": (100 * rec.pH).round(0)})[["Date", "Fixture", "Score", "Result", "Model", "Correct", "Bookmakers"]].set_index("Date"),
        width="stretch")

with tab_back:
    bt = pd.read_csv(ROOT / "reports/final_backtest_by_season.csv")
    n = bt.matches.sum()
    k = st.columns(3)
    k[0].metric("Final model", f"{100 * (bt.ensemble * bt.matches).sum() / n:.1f}%", f"{n:,} matches, 2019/20–2025/26", delta_color="off")
    k[1].metric("Bookmakers' favourite", f"{100 * (bt.bookmakers * bt.matches).sum() / n:.1f}%", "same matches", delta_color="off")
    k[2].metric("Always pick home", "43.4%", "same matches", delta_color="off")
    chart = bt.set_index("season")[["ensemble", "bookmakers"]].rename(columns={"ensemble": "model", "bookmakers": "bookmakers"}) * 100
    st.bar_chart(chart, height=300)
    st.markdown("Each season is predicted by a model trained only on earlier seasons. The final model adds confirmed lineups and an "
                "ensemble of five variants; it cannot run before lineups are known, so this app uses the results-only version "
                "(52.7% on the same backtest). Head-to-head history, weather, fixture congestion, manager changes, expected goals, "
                "player ratings and suspensions were all tested and add nothing. At 65%+ confidence the model is right about 73% of the time.")
    st.caption("For learning purposes. Not betting advice.")
