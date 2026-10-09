"""Pre-match features. Every value for a match uses only matches played BEFORE it.

This is the core fix versus the 2023 Mid_project_PL notebook, where the rolling
averages included the match being predicted (no shift), so the model could see
the final score through features like "average goals, last 5 matches".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POINTS = {"W": 3, "D": 1, "L": 0}


# ---------------------------------------------------------------- Elo ratings
def add_elo(df: pd.DataFrame, k: float = 20.0, home_adv: float = 60.0,
            carry: float = 0.8, promoted_offset: float = -60.0) -> pd.DataFrame:
    """Walk through matches in date order and store each team's rating BEFORE kickoff.

    - k: update speed; goal difference scales the update (bigger wins move more).
    - home_adv: Elo points added to the home side when computing expectations.
    - carry: between seasons, ratings shrink toward the league mean (1500) by (1 - carry).
    - promoted_offset: a team new to the league starts this far below 1500.
    """
    df = df.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)
    ratings: dict[str, float] = {}
    last_season: dict[str, str] = {}
    season_prev, prev_season_name = None, None
    home_elo, away_elo = np.empty(len(df)), np.empty(len(df))

    for i, r in enumerate(df.itertuples(index=False)):
        if r.Season != season_prev:  # new season: regress everyone toward the mean
            ratings = {t: 1500 + carry * (v - 1500) for t, v in ratings.items()}
            prev_season_name, season_prev = season_prev, r.Season
        for team in (r.HomeTeam, r.AwayTeam):
            # first match ever, or back after relegation: start as a promoted side
            if team not in last_season or last_season[team] not in (r.Season, prev_season_name):
                ratings[team] = 1500 + promoted_offset
            last_season[team] = r.Season
        h, a = ratings[r.HomeTeam], ratings[r.AwayTeam]
        home_elo[i], away_elo[i] = h, a

        if pd.isna(r.FTHG):  # fixture not played yet: record pre-match ratings only
            continue
        exp_h = 1 / (1 + 10 ** ((a - (h + home_adv)) / 400))
        score_h = 1.0 if r.FTHG > r.FTAG else 0.5 if r.FTHG == r.FTAG else 0.0
        gd = abs(r.FTHG - r.FTAG)
        mult = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
        delta = k * mult * (score_h - exp_h)
        ratings[r.HomeTeam] = h + delta
        ratings[r.AwayTeam] = a - delta

    df["HomeElo"], df["AwayElo"] = home_elo, away_elo
    df["EloDiff"] = df["HomeElo"] + home_adv - df["AwayElo"]
    df["EloExpHome"] = 1 / (1 + 10 ** (-df["EloDiff"] / 400))
    return df


# ------------------------------------------------------- rolling team form
def _team_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match, from that team's point of view."""
    base = ["match_id", "Date", "Season"]
    home = df[base + ["HomeTeam", "FTHG", "FTAG", "HS", "AS", "HST", "AST"]].rename(columns={
        "HomeTeam": "Team", "FTHG": "GF", "FTAG": "GA", "HS": "SF", "AS": "SA", "HST": "STF", "AST": "STA"})
    home["is_home"] = 1
    away = df[base + ["AwayTeam", "FTAG", "FTHG", "AS", "HS", "AST", "HST"]].rename(columns={
        "AwayTeam": "Team", "FTAG": "GF", "FTHG": "GA", "AS": "SF", "HS": "SA", "AST": "STF", "HST": "STA"})
    away["is_home"] = 0
    long = pd.concat([home, away], ignore_index=True).sort_values(["Team", "Date", "match_id"])
    long["Pts"] = np.select([long.GF > long.GA, long.GF == long.GA], [3, 1], 0)
    # A "spell" is an unbroken run in the Premier League. A gap of more than 150 days
    # means the team was relegated and came back: its old form must not carry over.
    gap = long.groupby("Team")["Date"].diff().dt.days
    long["spell"] = long["Team"] + "_" + (gap.isna() | (gap > 150)).groupby(long["Team"]).cumsum().astype(str)
    return long


def add_form(df: pd.DataFrame, windows=(5, 10)) -> pd.DataFrame:
    df = df.copy()
    df["match_id"] = np.arange(len(df))
    long = _team_long(df)
    g = long.groupby("spell", group_keys=False)

    stats = ["GF", "GA", "Pts", "SF", "SA", "STF", "STA"]
    feats = []
    for w in windows:
        for s in stats:
            col = f"{s}_l{w}"
            # shift(1): the current match is NEVER inside its own average
            long[col] = g[s].apply(lambda x: x.shift(1).rolling(w, min_periods=3).mean())
            feats.append(col)
    long["rest_days"] = g["Date"].apply(lambda x: x.diff().dt.days).clip(upper=14)
    long["n_prev"] = g.cumcount()  # matches of history available
    feats += ["rest_days", "n_prev"]

    h = long[long.is_home == 1].set_index("match_id")[feats].add_prefix("H_")
    a = long[long.is_home == 0].set_index("match_id")[feats].add_prefix("A_")
    df = df.join(h, on="match_id").join(a, on="match_id")
    for w in windows:
        for s in stats:
            df[f"D_{s}_l{w}"] = df[f"H_{s}_l{w}"] - df[f"A_{s}_l{w}"]
    return df.drop(columns="match_id")


# -------------------------------------------------------- bookmaker benchmark
def add_market_probs(df: pd.DataFrame) -> pd.DataFrame:
    """Implied probabilities from closing-style 1X2 odds, margin removed."""
    inv = 1 / df[["OddH", "OddD", "OddA"]]
    total = inv.sum(axis=1)
    df["MktH"], df["MktD"], df["MktA"] = (inv["OddH"] / total, inv["OddD"] / total, inv["OddA"] / total)
    return df


def build_features(matches: pd.DataFrame, **elo_kw) -> pd.DataFrame:
    df = add_elo(matches, **elo_kw)
    df = add_form(df)
    return add_market_probs(df)
