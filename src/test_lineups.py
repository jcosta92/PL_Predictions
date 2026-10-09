"""Do confirmed lineups (team news) improve the model?

Lineup data exists from 2013/14, so all variants below are trained on the same
matches (2013/14 onwards, earlier seasons only) and scored on the same seasons,
2019/20-2025/26. The current v1 model, trained from 2001, is included as a reference.

Variants
  A  v1 features                                        (same training rows as B-D)
  B  v1 + squad value (known days before, no lineup needed)
  C  v1 + squad value + lineup features (XI value, XI vs usual, regulars out)
  D  C with gradient boosting instead of logistic regression
  ref  current v1 model trained from 2001/02
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import BACKTEST, RESULTS_FEATS, encode, make_model
from features import build_features

ROOT = Path(__file__).resolve().parents[1]
SQUAD = ["H_squad_logval", "A_squad_logval", "D_squad_logval"]
LINEUP = ["H_xi_logval", "A_xi_logval", "D_xi_logval", "H_xi_vs_usual", "A_xi_vs_usual",
          "D_xi_vs_usual", "H_regulars_out", "A_regulars_out", "D_regulars_out"]
VARIANTS = {
    "A: v1 features": (RESULTS_FEATS, "logreg", "2013/14"),
    "B: + squad value (no lineup)": (RESULTS_FEATS + SQUAD, "logreg", "2013/14"),
    "C: + squad value + confirmed lineup": (RESULTS_FEATS + SQUAD + LINEUP, "logreg", "2013/14"),
    "D: C with gradient boosting": (RESULTS_FEATS + SQUAD + LINEUP, "gbm", "2013/14"),
    "ref: current v1 (trained from 2001)": (RESULTS_FEATS, "logreg", "2001/02"),
}


def rps_each(p, y):
    o = np.eye(3)[y]
    return ((np.cumsum(p, 1)[:, :2] - np.cumsum(o, 1)[:, :2]) ** 2).sum(1) / 2


def main():
    m = pd.read_csv(ROOT / "data" / "processed" / "matches.csv", parse_dates=["Date"])
    elo = json.load(open(ROOT / "reports" / "metrics.json"))["elo"]
    df = build_features(m, **elo)
    lu = pd.read_csv(ROOT / "data" / "processed" / "lineup_features.csv", parse_dates=["Date"])
    df = df.merge(lu[["Date", "HomeTeam", "AwayTeam"] + SQUAD + LINEUP], on=["Date", "HomeTeam", "AwayTeam"], how="left")

    res = {k: {"correct": [], "rps": [], "p": []} for k in VARIANTS}
    per_season, ys, mkt = [], [], []
    for s in BACKTEST:
        te = df.Season == s
        y = encode(df.loc[te, "Result"])
        ys.append(y)
        mk = df.loc[te, ["OddH", "OddD", "OddA"]].to_numpy()
        inv = 1 / mk
        mkt.append(inv / inv.sum(1, keepdims=True))
        row = {"season": s}
        for name, (feats, kind, start) in VARIANTS.items():
            tr = (df.Season >= start) & (df.Season < s)
            p = make_model(kind).fit(df.loc[tr, feats], encode(df.loc[tr, "Result"])).predict_proba(df.loc[te, feats])
            res[name]["correct"].append(p.argmax(1) == y)
            res[name]["rps"].append(rps_each(p, y))
            res[name]["p"].append(p)
            row[name] = (p.argmax(1) == y).mean()
        row["bookmakers"] = (mkt[-1].argmax(1) == y).mean()
        per_season.append(row)

    Y, M = np.concatenate(ys), np.vstack(mkt)
    ok = ~np.isnan(M).any(1)
    summary = []
    base_c = np.concatenate(res["A: v1 features"]["correct"])
    base_r = np.concatenate(res["A: v1 features"]["rps"])
    for name in VARIANTS:
        c, r = np.concatenate(res[name]["correct"]), np.concatenate(res[name]["rps"])
        d_c, d_r = c.astype(float) - base_c, r - base_r
        summary.append({"model": name, "accuracy": c.mean(), "rps": r.mean(),
                        "acc_vs_A_pts": 100 * d_c.mean(), "acc_vs_A_se": 100 * d_c.std(ddof=1) / np.sqrt(len(c)),
                        "rps_vs_A": d_r.mean(), "rps_vs_A_se": d_r.std(ddof=1) / np.sqrt(len(r))})
    summary.append({"model": "bookmakers", "accuracy": (M[ok].argmax(1) == Y[ok]).mean(),
                    "rps": rps_each(M[ok], Y[ok]).mean()})
    summary = pd.DataFrame(summary)

    # does the lineup model agree more with the bookmakers?
    pc = np.vstack(res["C: + squad value + confirmed lineup"]["p"])
    pa = np.vstack(res["A: v1 features"]["p"])
    agree = {"A": float((pa[ok].argmax(1) == M[ok].argmax(1)).mean()),
             "C": float((pc[ok].argmax(1) == M[ok].argmax(1)).mean())}

    # weakened teams: matches where a side's XI is >=15% below its usual value
    te_all = df.Season.isin(BACKTEST)
    weak = (df.loc[te_all, ["H_xi_vs_usual", "A_xi_vs_usual"]].min(axis=1) <= np.log(0.85)).to_numpy()
    sub = {"matches": int(weak.sum()), "share": float(weak.mean()),
           "A_acc": float(base_c[weak].mean()),
           "C_acc": float(np.concatenate(res["C: + squad value + confirmed lineup"]["correct"])[weak].mean()),
           "bookmakers_acc": float((M[weak & ok].argmax(1) == Y[weak & ok]).mean())}

    out = ROOT / "reports"
    pd.DataFrame(per_season).round(4).to_csv(out / "lineup_backtest_by_season.csv", index=False)
    summary.round(4).to_csv(out / "lineup_summary.csv", index=False)
    json.dump({"agreement_with_bookmakers": agree, "weakened_lineup_matches": sub},
              open(out / "lineup_extra.json", "w"), indent=2)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(per_season).round(3).to_string(index=False))
    print(summary.round(4).to_string(index=False))
    print("agreement with bookmakers:", agree)
    print("weakened-lineup matches:", sub)


if __name__ == "__main__":
    main()
