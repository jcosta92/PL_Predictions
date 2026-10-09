"""Try to beat the lineup model: more features, tuning, other model families, ensembles.

Protocol (stricter than before, because many variants are tried):
- Every variant is backtested on 2019/20-2025/26, each season trained only on
  2013/14 up to the previous season (lineup data starts in 2013/14).
- The seven seasons are split in two. The DEVELOPMENT half (2019/20-2022/23,
  1,520 matches) is used to choose a variant. The CONFIRMATION half (2023/24-2025/26,
  1,140 matches) is only looked at afterwards, to see whether the choice holds up on
  seasons that played no part in the choice.
- Each variant is compared with the lineup model on the same matches (paired RPS
  difference with its standard error).

Two questions are answered separately:
  A. Football-only: how good can the predictions get from football data alone?
  B. Market-aware: if the bookmakers' pre-match odds are also an input, does the
     football model add anything to them?

Output: reports/improve_results.csv, reports/improve_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import BACKTEST, RESULTS_FEATS, encode
from features_v3 import MARKET, assemble
from test_lineups import LINEUP, SQUAD

ROOT = Path(__file__).resolve().parents[1]
DEV, CONFIRM = BACKTEST[:4], BACKTEST[4:]
TRAIN_FROM = "2013/14"

BASE = RESULTS_FEATS + SQUAD + LINEUP
LONG = ["D_GD_l19", "D_Pts_l19", "D_GD_ewm", "D_Pts_ewm", "D_STD_l6", "D_STD_l19", "D_STD_ewm"]
RATINGS = ["ClubEloDiff", "HomeClubElo", "AwayClubElo", "PiDiff", "PiExp", "PiSTDiff", "PiSTExp"]
POIS = ["PoisXGH", "PoisXGA", "PoisXGDiff"]
NONLIN = ["AbsEloDiff", "EloDiff2"]
HOMEADV = ["LeagueHA", "NoCrowd"]
ALL_FOOTBALL = BASE + LONG + RATINGS + POIS + NONLIN + HOMEADV


# ------------------------------------------------------------------ models
def logreg(C=0.1):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, max_iter=3000))


def gbm(depth=2, lr=0.03, iters=200, leaf=150, l2=3.0):
    return HistGradientBoostingClassifier(max_depth=depth, learning_rate=lr, max_iter=iters,
                                          min_samples_leaf=leaf, l2_regularization=l2, random_state=0)


class OrdinalLogit:
    """Proportional-odds model: one latent strength difference z = Xw and two cut
    points. P(away) = s(t1 - z), P(away or draw) = s(t2 - z). Fewer parameters than the
    multinomial model, and the draw sits naturally between the two wins."""

    def __init__(self, C=0.1):
        self.C = C
        self.prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    def fit(self, X, y, sample_weight=None):
        X = self.prep.fit_transform(X)
        n, k = X.shape
        # ordinal level: 0 = away, 1 = draw, 2 = home  (y: 0=H, 1=D, 2=A)
        lvl = 2 - y
        w0 = np.zeros(k + 2)
        w0[k], w0[k + 1] = -0.5, 0.5
        sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight)
        lam = 1 / (2 * self.C * n)

        def nll(p):
            w, t1, t2 = p[:k], p[k], p[k + 1]
            z = X @ w
            c1, c2 = expit(t1 - z), expit(t2 - z)
            pa, pd_, ph = c1, c2 - c1, 1 - c2
            probs = np.clip(np.select([lvl == 0, lvl == 1], [pa, pd_], ph), 1e-9, 1)
            return -(sw * np.log(probs)).sum() / sw.sum() + lam * (w @ w)

        res = minimize(nll, w0, method="L-BFGS-B")
        self.w_, self.t1_, self.t2_ = res.x[:k], res.x[k], res.x[k + 1]
        return self

    def predict_proba(self, X):
        z = self.prep.transform(X) @ self.w_
        c1, c2 = expit(self.t1_ - z), expit(self.t2_ - z)
        return np.column_stack([1 - c2, c2 - c1, c1])  # H, D, A


# ------------------------------------------------------------------ scoring
def rps_each(p, y):
    o = np.eye(3)[y]
    return ((np.cumsum(p, 1)[:, :2] - np.cumsum(o, 1)[:, :2]) ** 2).sum(1) / 2


def backtest(df, feats, make, halflife=None):
    """Out-of-sample probabilities for every BACKTEST season.
    halflife: optional time decay of training weights, in seasons."""
    seasons = sorted(df.Season.unique())
    ps = []
    for s in BACKTEST:
        tr = (df.Season >= TRAIN_FROM) & (df.Season < s)
        te = df.Season == s
        sw = None
        if halflife:
            age = seasons.index(s) - df.loc[tr, "Season"].map(seasons.index).to_numpy()
            sw = 0.5 ** (age / halflife)
        m = make()
        X, y = df.loc[tr, feats], encode(df.loc[tr, "Result"])
        if sw is None:
            m.fit(X, y)
        elif isinstance(m, OrdinalLogit) or isinstance(m, HistGradientBoostingClassifier):
            m.fit(X, y, sample_weight=sw)
        else:
            m.fit(X, y, logisticregression__sample_weight=sw)
        ps.append(m.predict_proba(df.loc[te, feats]))
    return np.vstack(ps)


def compare(name, base_p, p, y, seasons, group):
    rows = []
    for label, sel in (("dev", seasons.isin(DEV)), ("confirm", seasons.isin(CONFIRM)), ("all", np.ones(len(y), bool))):
        b, n, yy = base_p[sel], p[sel], y[sel]
        dr = rps_each(n, yy) - rps_each(b, yy)
        dc = (n.argmax(1) == yy).astype(float) - (b.argmax(1) == yy)
        k = len(yy)
        rows.append({"group": group, "variant": name, "split": label, "matches": k,
                     "accuracy": (n.argmax(1) == yy).mean(), "rps": rps_each(n, yy).mean(),
                     "acc_change_pts": 100 * dc.mean(), "rps_change": dr.mean(),
                     "rps_se": dr.std(ddof=1) / np.sqrt(k),
                     "rps_change_in_se": dr.mean() / (dr.std(ddof=1) / np.sqrt(k)) if dr.std() > 0 else 0.0})
    return rows


def main():
    df = assemble()
    te_all = df.Season.isin(BACKTEST)
    y = encode(df.loc[te_all, "Result"])
    seasons = df.loc[te_all, "Season"].reset_index(drop=True)
    mkt = df.loc[te_all, ["MktH", "MktD", "MktA"]].to_numpy()

    probs = {}
    rows = []

    def run(name, feats, make, group="A. football only", halflife=None, base="lineup model"):
        p = backtest(df, feats, make, halflife)
        probs[name] = p
        rows.extend(compare(name, probs[base], p, y, seasons, group))
        r = [x for x in rows if x["variant"] == name]
        print(f"{name:55s} dev {r[0]['rps_change_in_se']:+5.2f}  confirm {r[1]['rps_change_in_se']:+5.2f}  "
              f"all rps {r[2]['rps']:.4f} acc {r[2]['accuracy']:.4f}", flush=True)
        return p

    run("lineup model", BASE, logreg)

    # --- features
    run("+ long-window form (19 games, EWM, shots on target)", BASE + LONG, logreg)
    run("+ ClubElo and pi-ratings", BASE + RATINGS, logreg)
    run("+ Poisson expected goals", BASE + POIS, logreg)
    run("+ non-linear Elo (abs, squared)", BASE + NONLIN, logreg)
    run("+ league home advantage, no-crowd flag", BASE + HOMEADV, logreg)
    run("+ all of the above", ALL_FOOTBALL, logreg)

    # --- regularisation
    for C in (0.02, 0.05, 0.3, 1.0):
        run(f"lineup model, C={C}", BASE, lambda C=C: logreg(C))
        run(f"all features, C={C}", ALL_FOOTBALL, lambda C=C: logreg(C))

    # --- recency weighting
    for hl in (2, 4, 8):
        run(f"lineup model, half-life {hl} seasons", BASE, logreg, halflife=hl)
        run(f"all features, half-life {hl} seasons", ALL_FOOTBALL, logreg, halflife=hl)

    # --- other model families
    run("ordinal logit, lineup features", BASE, lambda: OrdinalLogit(0.1))
    run("ordinal logit, all features", ALL_FOOTBALL, lambda: OrdinalLogit(0.1))
    run("gradient boosting (shallow, regularised), lineup features", BASE, gbm)
    run("gradient boosting (shallow, regularised), all features", ALL_FOOTBALL, gbm)
    run("gradient boosting, depth 3", ALL_FOOTBALL, lambda: gbm(depth=3, iters=300, leaf=100))

    # --- ensembles (simple averages of probabilities, nothing fitted)
    def avg(name, parts, group="A. football only"):
        p = np.mean([probs[k] for k in parts], axis=0)
        probs[name] = p
        rows.extend(compare(name, probs["lineup model"], p, y, seasons, group))
        r = [x for x in rows if x["variant"] == name]
        print(f"{name:55s} dev {r[0]['rps_change_in_se']:+5.2f}  confirm {r[1]['rps_change_in_se']:+5.2f}  "
              f"all rps {r[2]['rps']:.4f} acc {r[2]['accuracy']:.4f}", flush=True)

    avg("ensemble: logreg + ordinal (lineup features)", ["lineup model", "ordinal logit, lineup features"])
    avg("ensemble: logreg + gbm (lineup features)", ["lineup model", "gradient boosting (shallow, regularised), lineup features"])
    avg("ensemble: logreg + ordinal + gbm (lineup features)",
        ["lineup model", "ordinal logit, lineup features", "gradient boosting (shallow, regularised), lineup features"])
    avg("ensemble: logreg + ordinal + gbm (all features)",
        ["+ all of the above", "ordinal logit, all features", "gradient boosting (shallow, regularised), all features"])
    avg("ensemble: lineup logreg + all-features gbm", ["lineup model", "gradient boosting (shallow, regularised), all features"])

    # --- B. market-aware
    probs["bookmakers"] = mkt
    rows.extend(compare("bookmakers (reference)", probs["lineup model"], mkt, y, seasons, "B. market-aware"))
    run("market odds only (logreg)", MARKET, logreg, group="B. market-aware")
    run("market + lineup features", BASE + MARKET, logreg, group="B. market-aware")
    run("market + all football features", ALL_FOOTBALL + MARKET, logreg, group="B. market-aware")
    run("market + all football features, C=0.03", ALL_FOOTBALL + MARKET, lambda: logreg(0.03), group="B. market-aware")
    run("market + all football features, half-life 4", ALL_FOOTBALL + MARKET, logreg, group="B. market-aware", halflife=4)
    avg("ensemble: market + lineup / bookmakers", ["market + lineup features", "bookmakers"], group="B. market-aware")

    out = pd.DataFrame(rows)
    out.round(5).to_csv(ROOT / "reports" / "improve_results.csv", index=False)
    np.savez_compressed(ROOT / "reports" / "improve_probs.npz", y=y, seasons=seasons.to_numpy(), **{k: v for k, v in probs.items()})

    # choice on the development half, then the confirmation half
    foot = out[(out.group == "A. football only") & (out.split == "dev")].sort_values("rps")
    best = foot.iloc[0]["variant"]
    conf = out[(out.variant == best) & (out.split == "confirm")].iloc[0]
    summary = {"chosen_on_dev": best,
               "dev": foot.iloc[0][["accuracy", "rps", "rps_change", "rps_change_in_se"]].to_dict(),
               "confirm": conf[["accuracy", "rps", "rps_change", "rps_change_in_se"]].to_dict()}
    json.dump(summary, open(ROOT / "reports" / "improve_summary.json", "w"), indent=2, default=float)
    print("\nchosen on development half:", best)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
