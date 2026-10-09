"""Two extra signals, both leakage-free, and a fair test of whether they help.

1. Head-to-head: previous meetings between the same two clubs (last 6 years,
   up to the last 6 meetings, any venue; and the same fixture at the same ground).
2. Weather on match day at the home team's city (Visual Crossing daily data
   collected for the 2023 project; covers 11 Aug 2017 to 17 Apr 2023).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# Football-Data club name -> city in the weather files
CITY = {
    "Arsenal": "London", "Chelsea": "London", "Tottenham": "London", "West Ham": "London",
    "Crystal Palace": "London", "Fulham": "London", "Brentford": "Brentford",
    "Aston Villa": "Birmingham", "Bournemouth": "Bournemouth", "Brighton": "Brighton and Hove",
    "Burnley": "Burnley", "Cardiff": "Cardiff", "Everton": "Liverpool", "Liverpool": "Liverpool",
    "Huddersfield": "Huddersfield", "Leeds": "Leeds", "Leicester": "Leicester",
    "Man City": "Manchester", "Man United": "Manchester", "Newcastle": "Newcastle upon Tyne",
    "Norwich": "Norwich", "Nott'm Forest": "Nottingham", "Sheffield United": "Sheffield",
    "Southampton": "Southampton", "Stoke": "Stoke", "Swansea": "Swansea", "Watford": "Watford",
    "West Brom": "West Bromwich", "Wolves": "Wolverhampton",
}
WEATHER_COLS = ["temp", "precip", "windspeed", "humidity", "cloudcover"]


# ------------------------------------------------------------ head-to-head
def add_h2h(df: pd.DataFrame, years: int = 6, last_n: int = 6) -> pd.DataFrame:
    """Uses only meetings BEFORE the match date. Everything from the home side's view."""
    df = df.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)
    past: dict[frozenset, list] = defaultdict(list)   # pair -> [(date, home, away, hg, ag)]
    out = np.full((len(df), 5), np.nan)
    window = pd.Timedelta(days=365 * years)
    for i, r in enumerate(df.itertuples(index=False)):
        key = frozenset((r.HomeTeam, r.AwayTeam))
        meetings = [m for m in past[key] if r.Date - m[0] <= window and m[0] < r.Date][-last_n:]
        if meetings:
            gd = [(hg - ag) if h == r.HomeTeam else (ag - hg) for _, h, _, hg, ag in meetings]
            pts = [3 if g > 0 else 1 if g == 0 else 0 for g in gd]
            same = [hg - ag for d, h, _, hg, ag in meetings if h == r.HomeTeam]
            out[i] = (len(meetings), np.mean(gd), np.mean(pts),
                      np.mean(same) if same else np.nan, np.mean([g == 0 for g in gd]))
        else:
            out[i, 0] = 0
        if not pd.isna(r.FTHG):
            past[key].append((r.Date, r.HomeTeam, r.AwayTeam, r.FTHG, r.FTAG))
    df[["H2H_n", "H2H_gd", "H2H_pts", "H2H_venue_gd", "H2H_draw_rate"]] = out
    return df


# ----------------------------------------------------------------- weather
def load_weather() -> pd.DataFrame:
    files = sorted(RAW.glob("weather_*.csv"))
    w = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    w["Date"] = pd.to_datetime(w["datetime"])
    return w.rename(columns={"name": "City"}).drop_duplicates(["City", "Date"])[["City", "Date"] + WEATHER_COLS]


def add_weather(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["City"] = df.HomeTeam.map(CITY)
    df = df.merge(load_weather(), on=["City", "Date"], how="left")
    df["rainy"] = (df.precip >= 2).astype(float).where(df.precip.notna())
    df["windy"] = (df.windspeed >= 30).astype(float).where(df.windspeed.notna())
    return df
