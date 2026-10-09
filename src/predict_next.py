"""Predict the next matchday of 2026/27 with the model chosen in evaluate.py.

Trains on every played match, then scores the upcoming fixtures. Output:
reports/next_matchday_predictions.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_dataset import RAW, parse_openfootball, season_of
from evaluate import CLASSES, RESULTS_FEATS, encode, make_model
from features import build_features

ROOT = Path(__file__).resolve().parents[1]


def main(n_days: int = 4) -> None:
    played = pd.read_csv(ROOT / "data" / "processed" / "matches.csv", parse_dates=["Date"])
    fixtures = parse_openfootball(RAW / "openfootball_pl_2026-27.txt", 2026, played=False)
    first = fixtures.Date.min()
    fixtures = fixtures[fixtures.Date <= first + pd.Timedelta(days=n_days)].copy()
    fixtures["Season"] = fixtures.Date.apply(season_of)

    elo_cfg = json.load(open(ROOT / "reports" / "metrics.json"))["elo"]
    df = build_features(pd.concat([played, fixtures], ignore_index=True), **elo_cfg)
    train = df.FTHG.notna() & (df.Season >= "2001/02")
    upcoming = df.FTHG.isna()

    model = make_model("logreg").fit(df.loc[train, RESULTS_FEATS], encode(df.loc[train, "Result"]))
    p = model.predict_proba(df.loc[upcoming, RESULTS_FEATS])
    out = df.loc[upcoming, ["Date", "HomeTeam", "AwayTeam", "HomeElo", "AwayElo"]].copy()
    out[["pH", "pD", "pA"]] = p.round(3)
    out["Pick"] = np.array(CLASSES)[p.argmax(1)]
    out[["HomeElo", "AwayElo"]] = out[["HomeElo", "AwayElo"]].round(0)
    out.to_csv(ROOT / "reports" / "next_matchday_predictions.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
