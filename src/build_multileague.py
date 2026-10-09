"""Build a multi-league match table (extra training data).

Same source as build_dataset.py (Football-Data.co.uk via xgabora's dataset), but
keeps the other big leagues too. Adds ClubElo pre-match ratings, which are
calibrated ACROSS leagues and divisions through European and cup matches, so a
newly promoted Premier League team arrives with a meaningful rating instead of
a guess.

For the 2026/27 Premier League matches that only openfootball has (after
31 Aug 2026), ClubElo is carried forward from each team's last known rating
using the ClubElo update rule.

Output: data/processed/matches_multi.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from build_dataset import RAW, parse_openfootball, season_of

ROOT = Path(__file__).resolve().parents[1]
LEAGUES = ["E0", "E1", "SP1", "I1", "D1", "F1"]  # PL, Championship, LaLiga, Serie A, Bundesliga, Ligue 1


def load(source_csv: Path) -> pd.DataFrame:
    m = pd.read_csv(source_csv, low_memory=False)
    m = m[m.Division.isin(LEAGUES)].rename(columns={
        "MatchDate": "Date", "FTHome": "FTHG", "FTAway": "FTAG",
        "HomeShots": "HS", "AwayShots": "AS", "HomeTarget": "HST", "AwayTarget": "AST",
        "OddHome": "OddH", "OddDraw": "OddD", "OddAway": "OddA",
        "HomeElo": "HomeClubElo", "AwayElo": "AwayClubElo", "Division": "League"})
    m["Date"] = pd.to_datetime(m["Date"])
    keep = ["League", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HS", "AS", "HST", "AST",
            "HomeCorners", "AwayCorners", "OddH", "OddD", "OddA", "HomeClubElo", "AwayClubElo"]
    return m[keep]


def carry_clubelo(df: pd.DataFrame, k: float = 20.0, hfa: float = 60.0) -> pd.DataFrame:
    """Fill missing ClubElo for late 2026/27 rows by continuing each team's rating."""
    df = df.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)
    last: dict[str, float] = {}
    for i, r in df.iterrows():
        h = r.HomeClubElo if pd.notna(r.HomeClubElo) else last.get(r.HomeTeam)
        a = r.AwayClubElo if pd.notna(r.AwayClubElo) else last.get(r.AwayTeam)
        df.at[i, "HomeClubElo"], df.at[i, "AwayClubElo"] = h, a
        if h is None or a is None or pd.isna(r.FTHG):
            continue
        exp_h = 1 / (1 + 10 ** ((a - h - hfa) / 400))
        res = 1.0 if r.FTHG > r.FTAG else 0.5 if r.FTHG == r.FTAG else 0.0
        gd = abs(r.FTHG - r.FTAG)
        delta = k * (1 + gd) ** 0.5 * (res - exp_h) if gd else k * (res - exp_h)
        last[r.HomeTeam], last[r.AwayTeam] = h + delta, a - delta
    return df


def main() -> None:
    m = load(RAW / "football_data_top_leagues.csv")
    of = parse_openfootball(RAW / "openfootball_pl_2026-27.txt", 2026)
    have = set(map(tuple, m.loc[(m.League == "E0") & (m.Date >= "2026-07-01"), ["HomeTeam", "AwayTeam"]].values))
    extra = of[[tuple(r) not in have for r in of[["HomeTeam", "AwayTeam"]].values]].assign(League="E0")
    df = pd.concat([m, extra], ignore_index=True)

    e0 = carry_clubelo(df[df.League == "E0"].copy())
    df = pd.concat([df[df.League != "E0"], e0], ignore_index=True)
    df = df.sort_values(["Date", "League", "HomeTeam"]).reset_index(drop=True)
    df["Season"] = df["Date"].apply(season_of)
    df["Result"] = np.select([df.FTHG > df.FTAG, df.FTHG < df.FTAG], ["H", "A"], "D")
    df.to_csv(ROOT / "data" / "processed" / "matches_multi.csv", index=False)
    print(df.groupby("League").size().to_string())
    print("E0 rows missing ClubElo:", df[(df.League == "E0")][["HomeClubElo", "AwayClubElo"]].isna().any(axis=1).sum())


if __name__ == "__main__":
    main()
