"""Reproduce the two mistakes of the 2023 notebook and show their effect.

Mistake 1: rolling averages include the match being predicted (no shift).
Mistake 2: accuracy measured on all matches, including the training rows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from features import _team_long

ROOT = Path(__file__).resolve().parents[1]


def form_features(df: pd.DataFrame, leak: bool) -> pd.DataFrame:
    df = df.copy()
    df["match_id"] = np.arange(len(df))
    long = _team_long(df)
    g = long.groupby("Team", group_keys=False)
    cols = []
    for s in ["GF", "GA", "Pts", "SF", "SA", "STF", "STA"]:
        shift = 0 if leak else 1
        long[f"{s}_l5"] = g[s].apply(lambda x: x.shift(shift).rolling(5, min_periods=5).mean())
        cols.append(f"{s}_l5")
    h = long[long.is_home == 1].set_index("match_id")[cols].add_prefix("H_")
    a = long[long.is_home == 0].set_index("match_id")[cols].add_prefix("A_")
    return df.join(h, on="match_id").join(a, on="match_id").drop(columns="match_id")


def main():
    m = pd.read_csv(ROOT / "data" / "processed" / "matches.csv", parse_dates=["Date"])
    m = m[(m.Season >= "2017/18") & (m.Season <= "2022/23")]  # same era as the 2023 project
    rows = []
    for leak in (True, False):
        df = form_features(m, leak).dropna(subset=["H_GF_l5", "A_GF_l5"])
        feats = [c for c in df.columns if c.endswith("_l5")]
        X, y = df[feats], df["Result"]
        hold = df.Season == "2022/23"
        Xtr, Xte, ytr, yte = train_test_split(X[~hold], y[~hold], test_size=0.2, random_state=42)
        rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(Xtr, ytr)
        rows.append({
            "rolling averages": "include current match (2023 bug)" if leak else "previous matches only (fixed)",
            "scored on all rows incl. training": (rf.predict(X[~hold]) == y[~hold]).mean(),
            "random 20% split": (rf.predict(Xte) == yte).mean(),
            "unseen next season 2022/23": (rf.predict(X[hold]) == y[hold]).mean(),
        })
    out = pd.DataFrame(rows)
    out.round(3).to_csv(ROOT / "reports" / "leakage_demo.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
