"""Opening vs closing odds from the full Football-Data.co.uk season files (2019/20 onwards).

Football-Data records average market odds twice: AvgH/AvgD/AvgA (collected a few days
before the match) and AvgCH/AvgCD/AvgCA (at kickoff). How the odds move between the two
reflects late information: team news, injuries, and informed money.

move_X = log(closing probability of X / opening probability of X), margin removed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from build_dataset import CANONICAL


def _probs(df, cols):
    inv = 1 / df[cols].astype(float)
    return inv.div(inv.sum(axis=1), axis=0).to_numpy()


def load_odds(files: list[Path]) -> pd.DataFrame:
    parts = []
    for f in files:
        d = pd.read_csv(f, encoding="latin1")
        d = d.dropna(subset=["HomeTeam", "AwayTeam"])
        d["Date"] = pd.to_datetime(d["Date"], dayfirst=True)
        d[["HomeTeam", "AwayTeam"]] = d[["HomeTeam", "AwayTeam"]].replace(CANONICAL)
        o = _probs(d, ["AvgH", "AvgD", "AvgA"])
        c = _probs(d, ["AvgCH", "AvgCD", "AvgCA"])
        out = d[["Date", "HomeTeam", "AwayTeam"]].copy()
        out[["oH", "oD", "oA"]] = o
        out[["cH", "cD", "cA"]] = c
        out[["move_H", "move_D", "move_A"]] = np.log(c / o)
        parts.append(out)
    return pd.concat(parts, ignore_index=True)
