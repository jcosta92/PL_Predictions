"""Second round of attempts to beat the lineup model (same protocol as improve.py).

Variants
- detailed lineup features from lineups_v2.py (who is missing, by line, cohesion, bench)
- recalibration: refit only the intercepts and a temperature on the lineup model's
  out-of-sample probabilities, using the previous backtest seasons (tests whether the
  model's home-win bias in recent seasons can be corrected without new features)
- stacked ensemble: weights for several models fitted on the development half,
  then applied unchanged to the confirmation half

Needs reports/improve_probs.npz from improve.py.
Output: reports/improve2_results.csv, reports/improve2_probs.npz
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from evaluate import BACKTEST, encode
from features_v3 import KEY, assemble
from improve import BASE, CONFIRM, DEV, OrdinalLogit, backtest, compare, logreg, rps_each
from lineups_v2 import COLS, LINES

ROOT = Path(__file__).resolve().parents[1]

LINESET = [f"{s}_line_vs_usual_{ln}" for s in "HA" for ln in LINES] + [f"D_line_logval_{ln}" for ln in LINES]
ABSENT = [f"{s}_{c}" for s in "HA" for c in ("star_out", "missing_share", "gk_change", "captain_out")]
SHAPE = [f"{s}_{c}" for s in "HA" for c in ("cohesion", "bench_logval", "xi_age")] + ["D_cohesion", "D_bench_logval"]
V2_ALL = LINESET + ABSENT + SHAPE


def recalibrate(df, base_p, y, seasons):
    """For each backtest season after the first, fit a 3-class logistic regression on
    [log pH/pA, log pD/(1-pD)] using the earlier backtest seasons, and apply it."""
    lr = np.column_stack([np.log(base_p[:, 0] / base_p[:, 2]), np.log(base_p[:, 1] / (1 - base_p[:, 1]))])
    out = base_p.copy()
    for s in BACKTEST[1:]:
        tr, te = seasons.isin(BACKTEST[:BACKTEST.index(s)]).to_numpy(), (seasons == s).to_numpy()
        clf = LogisticRegression(C=10.0, max_iter=2000).fit(lr[tr], y[tr])
        out[te] = clf.predict_proba(lr[te])
    return out


def stack(parts, y, fit_sel):
    """Non-negative weights summing to 1 that minimise RPS on fit_sel."""
    k = len(parts)

    def loss(w):
        w = np.abs(w) / np.abs(w).sum()
        p = sum(wi * pi[fit_sel] for wi, pi in zip(w, parts))
        return rps_each(p, y[fit_sel]).mean()

    res = minimize(loss, np.ones(k) / k, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 2000})
    w = np.abs(res.x) / np.abs(res.x).sum()
    return w, sum(wi * pi for wi, pi in zip(w, parts))


def main():
    df = assemble()
    v2 = pd.read_csv(ROOT / "data" / "processed" / "lineup_features_v2.csv", parse_dates=["Date"])
    df = df.merge(v2[KEY + [f"{s}_{c}" for s in "HAD" for c in COLS]], on=KEY, how="left")
    te_all = df.Season.isin(BACKTEST)
    y = encode(df.loc[te_all, "Result"])
    seasons = df.loc[te_all, "Season"].reset_index(drop=True)

    prev = np.load(ROOT / "reports" / "improve_probs.npz", allow_pickle=True)
    assert (prev["y"] == y).all()
    probs = {k: prev[k] for k in prev.files if k not in ("y", "seasons")}
    base = probs["lineup model"]
    rows = []

    def report(name, p, group="C. second round"):
        probs[name] = p
        rows.extend(compare(name, base, p, y, seasons, group))
        r = [x for x in rows if x["variant"] == name]
        print(f"{name:58s} dev {r[0]['rps_change_in_se']:+5.2f}  confirm {r[1]['rps_change_in_se']:+5.2f}  "
              f"all rps {r[2]['rps']:.4f} acc {r[2]['accuracy']:.4f}", flush=True)

    report("lineup model", base)
    # --- detailed lineup features
    report("+ XI value by line (GK/DEF/MID/ATT)", backtest(df, BASE + LINESET, logreg))
    report("+ who is missing (star, value share, GK, captain)", backtest(df, BASE + ABSENT, logreg))
    report("+ cohesion, bench value, age", backtest(df, BASE + SHAPE, logreg))
    report("+ all detailed lineup features", backtest(df, BASE + V2_ALL, logreg))
    report("+ all detailed lineup features, C=0.03", backtest(df, BASE + V2_ALL, lambda: logreg(0.03)))
    report("ordinal logit + all detailed lineup features", backtest(df, BASE + V2_ALL, lambda: OrdinalLogit(0.1)))

    # --- recalibration of the lineup model (first backtest season unchanged)
    report("recalibrated lineup model (intercepts + temperature)", recalibrate(df, base, y, seasons))

    # --- stacked ensembles: weights fitted on the development half only
    cands = {"lineup model": base,
             "ordinal logit, lineup features": probs["ordinal logit, lineup features"],
             "+ long-window form (19 games, EWM, shots on target)": probs["+ long-window form (19 games, EWM, shots on target)"],
             "gradient boosting (shallow, regularised), all features": probs["gradient boosting (shallow, regularised), all features"],
             "+ Poisson expected goals": probs["+ Poisson expected goals"]}
    w, p = stack(list(cands.values()), y, seasons.isin(DEV).to_numpy())
    print("stack weights (fitted on dev):", {k: round(float(v), 3) for k, v in zip(cands, w)})
    report("stacked ensemble (weights fitted on dev half)", p)
    p_eq = np.mean(list(cands.values()), axis=0)
    report("equal-weight ensemble of the same five models", p_eq)

    # per-season table for the final ensemble
    mkt = df.loc[te_all, ["MktH", "MktD", "MktA"]].to_numpy()
    per = []
    for s in BACKTEST:
        sel = (seasons == s).to_numpy()
        per.append({"season": s, "matches": int(sel.sum()),
                    "lineup_model": (base[sel].argmax(1) == y[sel]).mean(),
                    "ensemble": (p_eq[sel].argmax(1) == y[sel]).mean(),
                    "bookmakers": (mkt[sel].argmax(1) == y[sel]).mean(),
                    "ensemble_rps": rps_each(p_eq[sel], y[sel]).mean(),
                    "bookmakers_rps": rps_each(mkt[sel], y[sel]).mean()})
    pd.DataFrame(per).round(4).to_csv(ROOT / "reports" / "final_backtest_by_season.csv", index=False)
    conf = p_eq.max(1)
    for t in (0.5, 0.6, 0.65, 0.7):
        sel = conf >= t
        print(f"ensemble confidence >= {t:.2f}: {sel.mean():.0%} of matches, accuracy {(p_eq[sel].argmax(1) == y[sel]).mean():.1%}")
    d = rps_each(p_eq, y) - rps_each(mkt, y)
    print(f"ensemble vs bookmakers: RPS {d.mean():+.5f} ({d.mean() / (d.std(ddof=1) / np.sqrt(len(d))):+.2f} SE)")

    out = pd.DataFrame(rows)
    out.round(5).to_csv(ROOT / "reports" / "improve2_results.csv", index=False)
    np.savez_compressed(ROOT / "reports" / "improve2_probs.npz", y=y, seasons=seasons.to_numpy(),
                        **{k: v for k, v in probs.items()})
    json.dump({"stack_weights": {k: float(v) for k, v in zip(cands, w)}},
              open(ROOT / "reports" / "improve2_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
