"""Assemble every pre-match feature built so far into one table, plus a few new ones.

Everything here is known before kickoff and uses only earlier matches:
- v1 features (Elo, 5/10-match form, rest days)            features.py
- v2 features (ClubElo, pi-ratings, 6/19-match and EWM form,
  shots-on-target form)                                     features_v2.py
- time-weighted Poisson expected goals                      poisson.py
- confirmed lineups and squad value                         lineups.py
- new in v3:
    AbsEloDiff, EloDiff2   non-linear Elo terms, so the draw probability can peak
                           when the teams are evenly matched
    LeagueHA               league-wide home advantage: mean home goal margin over the
                           previous 200 league matches (drops to ~0 behind closed doors)
    NoCrowd                1 for matches played without spectators (June 2020 - May 2021)
    LogMktHA, LogMktD      the bookmakers' pre-match odds as log-ratios. Only used in the
                           "market-aware" variants, which answer a different question:
                           can the football model add anything to the market price?
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import build_features
from test_lineups import LINEUP, SQUAD

ROOT = Path(__file__).resolve().parents[1]
KEY = ["Date", "HomeTeam", "AwayTeam"]

V2 = ["ClubEloDiff", "HomeClubElo", "AwayClubElo", "PiH", "PiA", "PiExp", "PiDiff",
      "PiSTH", "PiSTA", "PiSTExp", "PiSTDiff",
      "D_GD_l6", "D_GD_l19", "D_GD_ewm", "D_Pts_l6", "D_Pts_l19", "D_Pts_ewm",
      "D_STD_l6", "D_STD_l19", "D_STD_ewm"]
POIS = ["PoisXGH", "PoisXGA", "PoisXGDiff", "PoisH", "PoisD", "PoisA"]
NEW = ["AbsEloDiff", "EloDiff2", "LeagueHA", "NoCrowd"]
MARKET = ["LogMktHA", "LogMktD"]


def league_home_advantage(df: pd.DataFrame, window: int = 200) -> pd.Series:
    """Mean home goal margin over the previous `window` league matches, same for every
    match on a given date (matches on the same day are not used for each other)."""
    d = df.sort_values("Date")
    margin = (d.FTHG - d.FTAG)
    roll = margin.shift(1).rolling(window, min_periods=50).mean()
    first_of_day = roll.groupby(d.Date).transform("first")
    return first_of_day.reindex(df.index)


def assemble() -> pd.DataFrame:
    m = pd.read_csv(ROOT / "data" / "processed" / "matches.csv", parse_dates=["Date"])
    elo = json.load(open(ROOT / "reports" / "metrics.json"))["elo"]
    df = build_features(m, **elo)

    v2 = pd.read_pickle(ROOT / "data" / "processed" / "features_v2.pkl")
    v2 = v2[v2.League == "E0"][KEY + V2]
    pois = pd.read_pickle(ROOT / "data" / "processed" / "features.pkl")[KEY + POIS]
    lu = pd.read_csv(ROOT / "data" / "processed" / "lineup_features.csv", parse_dates=["Date"])
    df = (df.merge(v2, on=KEY, how="left").merge(pois, on=KEY, how="left")
            .merge(lu[KEY + SQUAD + LINEUP], on=KEY, how="left"))

    df["AbsEloDiff"] = df.EloDiff.abs()
    df["EloDiff2"] = df.EloDiff ** 2 / 1e4
    df["LeagueHA"] = league_home_advantage(df)
    df["NoCrowd"] = df.Date.between("2020-06-17", "2021-05-16").astype(int)
    df["LogMktHA"] = np.log(df.MktH / df.MktA)
    df["LogMktD"] = np.log(df.MktD / (1 - df.MktD))
    return df


if __name__ == "__main__":
    d = assemble()
    print(d.shape)
    print(d[NEW + MARKET].describe().round(3))
    print(d.groupby("Season")["LeagueHA"].mean().round(3).tail(10))
