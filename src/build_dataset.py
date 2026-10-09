"""Build one clean match table for the Premier League.

Sources
-------
- data/raw/football_data_E0_2000_2026.csv
    Premier League rows from github.com/xgabora/Club-Football-Match-Data-2000-2025
    (MIT), which repackages Football-Data.co.uk. Covers 2000/01 to 31 Aug 2026,
    with shots and pre-match bookmaker odds.
- data/raw/openfootball_pl_2026-27.txt
    github.com/openfootball/england (CC0). Results for the rest of 2026/27,
    which the first source does not have yet. Scores only.

Output
------
data/processed/matches.csv : one row per played match, sorted by date.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

# openfootball name -> Football-Data name
NAME_MAP = {
    "AFC Bournemouth": "Bournemouth",
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Burnley FC": "Burnley",
    "Chelsea FC": "Chelsea",
    "Coventry City FC": "Coventry",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Hull City AFC": "Hull",
    "Ipswich Town FC": "Ipswich",
    "Leeds United FC": "Leeds",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nott'm Forest",
    "Sunderland AFC": "Sunderland",
    "Tottenham Hotspur FC": "Tottenham",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
}

MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
DATE_RE = re.compile(r"^\s+(?:[A-Z][a-z]{2})\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
MATCH_RE = re.compile(
    r"^\s+(?:\d{2}:\d{2}\s+)?(.+?)\s+v\s+(.+?)(?:\s+(\d+)-(\d+)(?:\s+\(\d+-\d+\))?)?\s*$")


def parse_openfootball(path: Path, season_start_year: int, played: bool = True) -> pd.DataFrame:
    """Parse an openfootball .txt season file into played matches (or upcoming fixtures)."""
    rows, current = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(("=", "#", "▪")) or not line.strip():
            continue
        d = DATE_RE.match(line)
        if d:
            mon, day, year = d.group(1), int(d.group(2)), d.group(3)
            month = MONTHS[mon]
            year = int(year) if year else (season_start_year if month >= 7 else season_start_year + 1)
            current = pd.Timestamp(year=year, month=month, day=day)
            continue
        m = MATCH_RE.match(line)
        if m and (m.group(3) is not None) == played:
            home, away = m.group(1).strip(), m.group(2).strip()
            score = (int(m.group(3)), int(m.group(4))) if played else (None, None)
            rows.append({"Date": current, "HomeTeam": NAME_MAP[home], "AwayTeam": NAME_MAP[away],
                         "FTHG": score[0], "FTAG": score[1]})
    return pd.DataFrame(rows)


def season_of(date: pd.Timestamp) -> str:
    start = date.year if date.month >= 7 else date.year - 1
    return f"{start}/{str(start + 1)[-2:]}"


def load_football_data() -> pd.DataFrame:
    fd = pd.read_csv(RAW / "football_data_E0_2000_2026.csv", low_memory=False)
    fd = fd.rename(columns={
        "MatchDate": "Date", "FTHome": "FTHG", "FTAway": "FTAG",
        "HomeShots": "HS", "AwayShots": "AS", "HomeTarget": "HST", "AwayTarget": "AST",
        "OddHome": "OddH", "OddDraw": "OddD", "OddAway": "OddA",
    })
    fd["Date"] = pd.to_datetime(fd["Date"])
    keep = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HS", "AS", "HST", "AST",
            "OddH", "OddD", "OddA"]
    return fd[keep]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fd = load_football_data()
    of = parse_openfootball(RAW / "openfootball_pl_2026-27.txt", 2026)

    # Consistency check on the matches both sources contain.
    key = ["HomeTeam", "AwayTeam"]
    overlap = of.merge(fd[fd["Date"] >= "2026-07-01"], on=key, suffixes=("_of", "_fd"))
    bad = overlap[(overlap.FTHG_of != overlap.FTHG_fd) | (overlap.FTAG_of != overlap.FTAG_fd)]
    print(f"Overlap 2026/27: {len(overlap)} matches, score mismatches: {len(bad)}")
    assert bad.empty, bad

    # Add only the 2026/27 matches Football-Data does not have yet.
    have = set(map(tuple, fd.loc[fd["Date"] >= "2026-07-01", key].values))
    extra = of[[tuple(r) not in have for r in of[key].values]]
    df = pd.concat([fd, extra], ignore_index=True).sort_values(["Date", "HomeTeam"]).reset_index(drop=True)

    df["Season"] = df["Date"].apply(season_of)
    df["Result"] = (df.FTHG > df.FTAG).map({True: "H"}).fillna(
        (df.FTHG < df.FTAG).map({True: "A"})).fillna("D")
    df.to_csv(OUT / "matches.csv", index=False)
    print(f"Added {len(extra)} matches from openfootball. Total {len(df)} matches, "
          f"{df.Season.min()} to {df.Season.max()}.")
    print(df.groupby("Season").size().tail(4).to_string())


if __name__ == "__main__":
    main()
