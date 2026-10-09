"""Train and evaluate home-win / draw / away-win models with strict time ordering.

Protocol
--------
1. Elo settings and model choice are tuned ONLY on seasons up to 2024/25,
   scored on 2025/26 (validation). 2026/27 is never looked at while tuning.
2. Backtest: for each season 2019/20-2025/26, train on all earlier seasons,
   predict that season. Gives a stable accuracy over ~2,660 matches.
3. Final test: train on everything up to 2025/26, predict 2026/27 to date.

Metrics: accuracy, log loss, and RPS (ranked probability score, the standard
football forecasting metric: it penalises predicting an away win when the
result was a home win more than predicting a draw). Lower is better for both.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build_features
from poisson import add_poisson

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["H", "D", "A"]  # ordered for RPS
VALID, TEST = "2025/26", "2026/27"
BACKTEST = ["2019/20", "2020/21", "2021/22", "2022/23", "2023/24", "2024/25", "2025/26"]

ELO_ONLY = ["EloDiff"]
RESULTS_FEATS = ["EloDiff", "HomeElo", "AwayElo",
                 "D_GF_l5", "D_GA_l5", "D_Pts_l5", "D_GF_l10", "D_GA_l10", "D_Pts_l10",
                 "H_n_prev", "A_n_prev", "H_rest_days", "A_rest_days"]
POIS_ONLY = ["PoisXGH", "PoisXGA", "PoisXGDiff"]
RESULTS_POIS = RESULTS_FEATS + POIS_ONLY
SHOT_FEATS = RESULTS_FEATS + ["D_SF_l10", "D_SA_l10", "D_STF_l10", "D_STA_l10",
                              "D_SF_l5", "D_SA_l5", "D_STF_l5", "D_STA_l5"]


def rps(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(3)[y]
    cum_p, cum_o = np.cumsum(probs, 1)[:, :2], np.cumsum(onehot, 1)[:, :2]
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, 1) / 2))


def score(probs: np.ndarray, y: np.ndarray) -> dict:
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, probs.argmax(1))),
            "log_loss": float(log_loss(y, probs, labels=[0, 1, 2])), "rps": rps(probs, y)}


def encode(results: pd.Series) -> np.ndarray:
    return results.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()


def make_model(kind: str):
    if kind == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.1, max_iter=2000))
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=300,
                                              min_samples_leaf=80, l2_regularization=1.0,
                                              random_state=0)
    raise ValueError(kind)


def fit_predict(df, train_mask, test_mask, feats, kind):
    model = make_model(kind)
    model.fit(df.loc[train_mask, feats], encode(df.loc[train_mask, "Result"]))
    return model.predict_proba(df.loc[test_mask, feats])


def baselines(df, mask):
    y = encode(df.loc[mask, "Result"])
    n = mask.sum()
    home = np.tile([1.0, 0.0, 0.0], (n, 1))  # "always home": only accuracy is meaningful
    out = {"always_home_win": {"n": int(n), "accuracy": float((y == 0).mean())}}
    mkt = df.loc[mask, ["MktH", "MktD", "MktA"]]
    ok = mkt.notna().all(axis=1).to_numpy()
    if ok.any():
        out["bookmakers"] = score(mkt.to_numpy()[ok], y[ok])
    return out


def tune_elo(matches):
    """Pick Elo settings by validation log loss of an Elo-only model (2025/26 held out)."""
    best = None
    for k, ha, carry in itertools.product([15, 20, 25, 30], [50, 65, 80], [0.7, 0.8, 0.9]):
        df = build_features(matches, k=k, home_adv=ha, carry=carry)
        tr, va = df.Season < VALID, df.Season == VALID
        p = fit_predict(df, tr & (df.Season >= "2001/02"), va, ELO_ONLY, "logreg")
        ll = log_loss(encode(df.loc[va, "Result"]), p, labels=[0, 1, 2])
        if best is None or ll < best[0]:
            best = (ll, {"k": k, "home_adv": ha, "carry": carry})
    return best[1]


def main():
    matches = pd.read_csv(ROOT / "data" / "processed" / "matches.csv", parse_dates=["Date"])
    elo_cfg = tune_elo(matches)
    print("Elo settings:", elo_cfg)
    cache = ROOT / "data" / "processed" / "features.pkl"
    if cache.exists():
        df = pd.read_pickle(cache)
    else:
        df = add_poisson(build_features(matches, **elo_cfg))
        df.to_pickle(cache)
    usable = df.Season >= "2001/02"  # first season only warms up Elo/form

    candidates = {"elo_logreg": (ELO_ONLY, "logreg"),
                  "results_logreg": (RESULTS_FEATS, "logreg"),
                  "shots_logreg": (SHOT_FEATS, "logreg"),
                  "poisson_logreg": (POIS_ONLY, "logreg"),
                  "results_poisson_logreg": (RESULTS_POIS, "logreg"),
                  "results_poisson_gbm": (RESULTS_POIS, "gbm"),
                  "shots_gbm": (SHOT_FEATS, "gbm")}

    # 1) validation on 2025/26 to choose the model
    val_mask = df.Season == VALID
    val = {name: score(fit_predict(df, usable & (df.Season < VALID), val_mask, f, k),
                       encode(df.loc[val_mask, "Result"])) for name, (f, k) in candidates.items()}
    val.update(baselines(df, val_mask))

    # 2) season-by-season backtest
    bt_rows = []
    for season in BACKTEST:
        m = df.Season == season
        y = encode(df.loc[m, "Result"])
        row = {"season": season, "always_home_win": float((y == 0).mean())}
        for name, (f, k) in candidates.items():
            row[name] = score(fit_predict(df, usable & (df.Season < season), m, f, k), y)["accuracy"]
        b = baselines(df, m)
        row["bookmakers"] = b.get("bookmakers", {}).get("accuracy", np.nan)
        bt_rows.append(row)
    backtest = pd.DataFrame(bt_rows)

    # 3) final test on 2026/27 with the model picked on validation (lowest RPS)
    chosen = min((n for n in candidates), key=lambda n: val[n]["rps"])
    f, k = candidates[chosen]
    test_mask = df.Season == TEST
    p_test = fit_predict(df, usable & (df.Season < TEST), test_mask, f, k)
    y_test = encode(df.loc[test_mask, "Result"])
    test = {chosen: score(p_test, y_test), **baselines(df, test_mask)}
    test["elo_logreg"] = score(fit_predict(df, usable & (df.Season < TEST), test_mask, ELO_ONLY, "logreg"), y_test)

    preds = df.loc[test_mask, ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Result"]].copy()
    preds[["pH", "pD", "pA"]] = p_test.round(3)
    preds["Pick"] = np.array(CLASSES)[p_test.argmax(1)]
    preds["Correct"] = preds.Pick == preds.Result

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    backtest.round(3).to_csv(out / "backtest_by_season.csv", index=False)
    preds.to_csv(out / "test_2026_27_predictions.csv", index=False)
    json.dump({"elo": elo_cfg, "chosen_model": chosen, "validation_2025_26": val,
               "test_2026_27": test}, open(out / "metrics.json", "w"), indent=2)

    pd.set_option("display.width", 160)
    print("\nVALIDATION 2025/26\n", pd.DataFrame(val).T.round(3))
    print("\nBACKTEST accuracy by season\n", backtest.round(3).to_string(index=False))
    print("\nBacktest mean:\n", backtest.drop(columns="season").mean().round(3).to_string())
    print(f"\nTEST 2026/27 (chosen: {chosen})\n", pd.DataFrame(test).T.round(3))


if __name__ == "__main__":
    main()
