"""Version 2 features, computed per league. All values use only earlier matches.

New versus features.py:
- ClubElo pre-match ratings (cross-league, so promoted teams are rated properly)
- pi-ratings (Constantinou & Fenton, 2013): separate home and away strength per
  team, updated from goal difference with diminishing returns for big margins
- shot-on-target "pi-rating": the same method fed with shots-on-target
  difference, a rough stand-in for expected goals
- longer and exponentially weighted form windows
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import add_elo


# ----------------------------------------------------------------- pi-ratings
def _psi(r: float) -> float:
    return np.sign(r) * (10 ** (abs(r) / 3) - 1)


def add_pi(df: pd.DataFrame, value_home: str, value_away: str, prefix: str,
           lam: float = 0.035, gamma: float = 0.7) -> pd.DataFrame:
    """Pi-ratings on any home-minus-away quantity (goals by default)."""
    rh: dict[str, float] = {}   # rating when playing at home
    ra: dict[str, float] = {}   # rating when playing away
    out = np.full((len(df), 3), np.nan)
    for i, (h, a, vh, va) in enumerate(zip(df.HomeTeam, df.AwayTeam, df[value_home], df[value_away])):
        hh, ha_ = rh.get(h, 0.0), ra.get(h, 0.0)
        ah, aa = rh.get(a, 0.0), ra.get(a, 0.0)
        expected = _psi(hh) - _psi(aa)
        out[i] = (hh, aa, expected)
        if pd.isna(vh) or pd.isna(va):
            continue
        err = (vh - va) - expected
        step = np.sign(err) * 3 * np.log10(1 + abs(err))
        rh[h] = hh + step * lam
        ra[h] = ha_ + step * lam * gamma
        ra[a] = aa - step * lam
        rh[a] = ah - step * lam * gamma
    df[[f"{prefix}H", f"{prefix}A", f"{prefix}Exp"]] = out
    return df


# ------------------------------------------------------------- form windows
def add_form_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mid"] = np.arange(len(df))
    base = ["mid", "Date"]
    home = df[base + ["HomeTeam", "FTHG", "FTAG", "HST", "AST", "HS", "AS"]].set_axis(
        base + ["Team", "GF", "GA", "STF", "STA", "SF", "SA"], axis=1).assign(is_home=1)
    away = df[base + ["AwayTeam", "FTAG", "FTHG", "AST", "HST", "AS", "HS"]].set_axis(
        base + ["Team", "GF", "GA", "STF", "STA", "SF", "SA"], axis=1).assign(is_home=0)
    long = pd.concat([home, away], ignore_index=True).sort_values(["Team", "Date", "mid"])
    long["Pts"] = np.select([long.GF > long.GA, long.GF == long.GA], [3, 1], 0)
    long["GD"] = long.GF - long.GA
    long["STD"] = long.STF - long.STA
    gap = long.groupby("Team")["Date"].diff().dt.days
    long["spell"] = long.Team + "_" + (gap.isna() | (gap > 150)).groupby(long.Team).cumsum().astype(str)
    g = long.groupby("spell", group_keys=False)
    feats = []
    for s in ["GD", "Pts", "STD"]:
        for w in (6, 19):
            c = f"{s}_l{w}"
            long[c] = g[s].apply(lambda x: x.shift(1).rolling(w, min_periods=3).mean())
            feats.append(c)
        c = f"{s}_ewm"
        long[c] = g[s].apply(lambda x: x.shift(1).ewm(halflife=8, min_periods=3).mean())
        feats.append(c)
    long["n_prev"] = g.cumcount()
    long["rest"] = g["Date"].apply(lambda x: x.diff().dt.days).clip(upper=14)
    feats += ["n_prev", "rest"]
    h = long[long.is_home == 1].set_index("mid")[feats].add_prefix("H_")
    a = long[long.is_home == 0].set_index("mid")[feats].add_prefix("A_")
    df = df.join(h, on="mid").join(a, on="mid")
    for c in feats[:-2]:
        df[f"D_{c}"] = df[f"H_{c}"] - df[f"A_{c}"]
    return df.drop(columns="mid")


def build_features_v2(matches: pd.DataFrame, elo_cfg: dict) -> pd.DataFrame:
    parts = []
    for league, d in matches.groupby("League"):
        d = d.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)
        d = add_elo(d, **elo_cfg)
        d = add_pi(d, "FTHG", "FTAG", "Pi")
        d = add_pi(d, "HST", "AST", "PiST", lam=0.02)
        d = add_form_v2(d)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True).sort_values(["Date", "League", "HomeTeam"]).reset_index(drop=True)
    df["ClubEloDiff"] = df.HomeClubElo - df.AwayClubElo
    df["PiDiff"] = df.PiH - df.PiA
    df["PiSTDiff"] = df.PiSTH - df.PiSTA
    inv = 1 / df[["OddH", "OddD", "OddA"]]
    tot = inv.sum(axis=1)
    df["MktH"], df["MktD"], df["MktA"] = inv.OddH / tot, inv.OddD / tot, inv.OddA / tot
    return df
