"""Final v2 model: evaluate on the backtest and 2026/27, and predict the next matchday.

Chosen in experiments.py by lowest RPS on the 2019/20-2025/26 backtest:
  features : base + ClubElo + pi-ratings + shot-on-target ratings
  training : Premier League + Championship
  model    : average of logistic regression and gradient boosting
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_dataset import RAW, parse_openfootball, season_of
from build_multileague import carry_clubelo
from experiments import BACKTEST, CLASSES, FEATURE_SETS, encode, league_dummies, model, rps
from features_v2 import build_features_v2

ROOT = Path(__file__).resolve().parents[1]
FEATS = FEATURE_SETS["base+clubelo+pi+shots"]
POOL = ["E0", "E1"]
TEST = "2026/27"


def fit_predict(df, train, test, feats):
    X, y = df.loc[train, feats], encode(df.loc[train, "Result"])
    return np.mean([model(k).fit(X, y).predict_proba(df.loc[test, feats]) for k in ("logreg", "gbm")], axis=0)


def summary(p, y):
    return {"n": int(len(y)), "accuracy": float((p.argmax(1) == y).mean()), "rps": rps(p, y)}


def main():
    matches = pd.read_csv(ROOT / "data" / "processed" / "matches_multi.csv", parse_dates=["Date"])
    elo_cfg = json.load(open(ROOT / "reports" / "metrics.json"))["elo"]

    # add upcoming PL fixtures (no result) so they get pre-match features too
    fx = parse_openfootball(RAW / "openfootball_pl_2026-27.txt", 2026, played=False)
    fx = fx[fx.Date <= fx.Date.min() + pd.Timedelta(days=4)].assign(League="E0")
    fx["Season"] = fx.Date.apply(season_of)
    allm = pd.concat([matches, fx], ignore_index=True)
    e0 = carry_clubelo(allm[allm.League == "E0"].copy())
    allm = pd.concat([allm[allm.League != "E0"], e0], ignore_index=True)
    df = build_features_v2(allm, elo_cfg)
    feats = FEATS + league_dummies(df, POOL)
    played = df.FTHG.notna()
    usable = played & df.League.isin(POOL) & (df.Season >= "2001/02")

    # backtest, season by season
    rows, all_p, all_y = [], [], []
    for s in BACKTEST:
        te = played & (df.League == "E0") & (df.Season == s)
        p, y = fit_predict(df, usable & (df.Season < s), te, feats), encode(df.loc[te, "Result"])
        all_p.append(p); all_y.append(y)
        mk = df.loc[te, ["MktH", "MktD", "MktA"]].to_numpy()
        ok = ~np.isnan(mk).any(1)
        rows.append({"season": s, "always_home": float((y == 0).mean()),
                     "model_v2": float((p.argmax(1) == y).mean()),
                     "bookmakers": float((mk[ok].argmax(1) == y[ok]).mean())})
    bt = pd.DataFrame(rows)
    v1 = pd.read_csv(ROOT / "reports" / "backtest_by_season.csv")[["season", "results_logreg"]]
    bt = bt.merge(v1.rename(columns={"results_logreg": "model_v1"}), on="season")
    P, Y = np.vstack(all_p), np.concatenate(all_y)

    # accuracy by confidence
    conf = P.max(1)
    bands = [(0, .45), (.45, .55), (.55, .65), (.65, 1.01)]
    conf_tbl = pd.DataFrame([{"confidence": f"{lo:.0%}-{min(hi, 1):.0%}", "matches": int(((conf >= lo) & (conf < hi)).sum()),
                              "share": float(((conf >= lo) & (conf < hi)).mean()),
                              "accuracy": float((P[(conf >= lo) & (conf < hi)].argmax(1) == Y[(conf >= lo) & (conf < hi)]).mean())}
                             for lo, hi in bands])

    # test season 2026/27
    te = played & (df.League == "E0") & (df.Season == TEST)
    pt = fit_predict(df, usable & (df.Season < TEST), te, feats)
    yt = encode(df.loc[te, "Result"])
    test_preds = df.loc[te, ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Result"]].copy()
    test_preds[["pH", "pD", "pA"]] = pt.round(3)
    test_preds["Pick"] = np.array(CLASSES)[pt.argmax(1)]
    test_preds["Correct"] = test_preds.Pick == test_preds.Result

    # next matchday: train on every played match
    nxt = df.FTHG.isna() & (df.League == "E0")
    pn = fit_predict(df, usable, nxt, feats)
    next_preds = df.loc[nxt, ["Date", "HomeTeam", "AwayTeam"]].copy()
    next_preds[["pH", "pD", "pA"]] = pn.round(3)
    next_preds["Pick"] = np.array(CLASSES)[pn.argmax(1)]
    next_preds["Confidence"] = pn.max(1).round(3)

    out = ROOT / "reports"
    bt.round(3).to_csv(out / "v2_backtest_by_season.csv", index=False)
    conf_tbl.round(3).to_csv(out / "v2_accuracy_by_confidence.csv", index=False)
    test_preds.to_csv(out / "v2_test_2026_27_predictions.csv", index=False)
    next_preds.to_csv(out / "v2_next_matchday_predictions.csv", index=False)
    metrics = {"backtest": summary(P, Y), "test_2026_27": summary(pt, yt),
               "backtest_mean_by_season": bt.drop(columns="season").mean().round(4).to_dict()}
    json.dump(metrics, open(out / "v2_metrics.json", "w"), indent=2)

    print(bt.round(3).to_string(index=False))
    print("\nmeans:", metrics["backtest_mean_by_season"])
    print("\n", conf_tbl.round(3).to_string(index=False))
    print("\nTEST 2026/27:", metrics["test_2026_27"])
    print("\n", next_preds.to_string(index=False))


if __name__ == "__main__":
    main()
