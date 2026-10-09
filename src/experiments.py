"""Compare feature sets, training pools and models on the Premier League backtest.

Every configuration is scored the same way: for each season 2019/20-2025/26,
train on all earlier seasons (from the chosen pool of leagues), predict that
Premier League season. 2026/27 is not touched here.

Output: reports/experiments.csv (one row per configuration)
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["H", "D", "A"]
BACKTEST = ["2019/20", "2020/21", "2021/22", "2022/23", "2023/24", "2024/25", "2025/26"]

BASE = ["EloDiff", "HomeElo", "AwayElo", "D_GD_l6", "D_Pts_l6", "D_GD_l19", "D_Pts_l19",
        "D_GD_ewm", "D_Pts_ewm", "H_n_prev", "A_n_prev", "H_rest", "A_rest"]
CLUBELO = ["ClubEloDiff", "HomeClubElo", "AwayClubElo"]
PI = ["PiH", "PiA", "PiExp", "PiDiff"]
SHOTS = ["PiSTH", "PiSTA", "PiSTExp", "PiSTDiff", "D_STD_l6", "D_STD_l19", "D_STD_ewm"]
FEATURE_SETS = {
    "base": BASE,
    "base+clubelo": BASE + CLUBELO,
    "base+clubelo+pi": BASE + CLUBELO + PI,
    "base+clubelo+pi+shots": BASE + CLUBELO + PI + SHOTS,
}
POOLS = {
    "PL only": ["E0"],
    "PL+Championship": ["E0", "E1"],
    "top-5 leagues": ["E0", "SP1", "I1", "D1", "F1"],
    "top-5 + Championship": ["E0", "E1", "SP1", "I1", "D1", "F1"],
}


def encode(s: pd.Series) -> np.ndarray:
    return s.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()


def rps(p: np.ndarray, y: np.ndarray) -> float:
    o = np.eye(3)[y]
    return float(np.mean(((np.cumsum(p, 1)[:, :2] - np.cumsum(o, 1)[:, :2]) ** 2).sum(1) / 2))


def model(kind: str):
    if kind == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.05, max_iter=3000))
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=250,
                                          min_samples_leaf=150, l2_regularization=2.0,
                                          random_state=0)


def league_dummies(df: pd.DataFrame, leagues: list[str]) -> list[str]:
    cols = []
    for lg in leagues[1:]:  # E0 is the reference level
        df[f"lg_{lg}"] = (df.League == lg).astype(float)
        cols.append(f"lg_{lg}")
    return cols


def backtest_probs(df, feats, leagues, kind):
    feats = feats + league_dummies(df, leagues)
    usable = df.League.isin(leagues) & (df.Season >= "2001/02")
    probs, ys = [], []
    for s in BACKTEST:
        tr = usable & (df.Season < s)
        te = (df.League == "E0") & (df.Season == s)
        if kind == "ensemble":
            p = np.mean([model(k).fit(df.loc[tr, feats], encode(df.loc[tr, "Result"]))
                         .predict_proba(df.loc[te, feats]) for k in ("logreg", "gbm")], axis=0)
        else:
            p = model(kind).fit(df.loc[tr, feats], encode(df.loc[tr, "Result"])).predict_proba(df.loc[te, feats])
        probs.append(p)
        ys.append(encode(df.loc[te, "Result"]))
    return np.vstack(probs), np.concatenate(ys)


def main():
    df = pd.read_pickle(ROOT / "data" / "processed" / "features_v2.pkl")
    rows = []
    for (fs, feats), (pool, leagues), kind in itertools.product(
            FEATURE_SETS.items(), POOLS.items(), ["logreg", "gbm", "ensemble"]):
        p, y = backtest_probs(df.copy(), feats, leagues, kind)
        rows.append({"features": fs, "pool": pool, "model": kind,
                     "accuracy": (p.argmax(1) == y).mean(), "rps": rps(p, y)})
        print(rows[-1], flush=True)
    # bookmaker reference on the same matches
    e = df[(df.League == "E0") & df.Season.isin(BACKTEST)]
    ok = e[["MktH", "MktD", "MktA"]].notna().all(axis=1)
    pm, ym = e.loc[ok, ["MktH", "MktD", "MktA"]].to_numpy(), encode(e.loc[ok, "Result"])
    rows.append({"features": "bookmakers", "pool": "-", "model": "-",
                 "accuracy": (pm.argmax(1) == ym).mean(), "rps": rps(pm, ym)})
    out = pd.DataFrame(rows).sort_values("rps")
    out.round(4).to_csv(ROOT / "reports" / "experiments.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
