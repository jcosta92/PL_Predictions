"""Test each extra signal against the lineup model (best model so far).

Signals covering 2013/14-2025/26 are added directly to the lineup model and
backtested on 2019/20-2025/26 (2,660 matches). Each is compared with the lineup
model on the SAME matches (paired difference + standard error).

xG and odds movement only exist from 2019/20, so they are tested as a second stage:
does adding them to the lineup model's out-of-sample probabilities help, using only
earlier seasons of that era to fit?

With 8 separate tests, one may cross 2 standard errors by luck, so a signal is only
called useful if it improves RPS by clearly more than 2 standard errors.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from evaluate import BACKTEST, RESULTS_FEATS, encode, make_model
from features import build_features
from test_lineups import LINEUP, SQUAD

ROOT = Path(__file__).resolve().parents[1]
BASE = RESULTS_FEATS + SQUAD + LINEUP

GROUPS = {
    "1. Fixture congestion (all competitions, Europe)": [
        "H_rest_all", "A_rest_all", "H_games_14d", "A_games_14d",
        "H_euro_before", "A_euro_before", "H_euro_after", "A_euro_after"],
    "2. What's at stake (table, motivation)": [
        "D_gap_top4", "D_gap_safety", "H_nothing_to_play", "A_nothing_to_play",
        "H_relegation_fight", "A_relegation_fight"],
    "3. Manager change": ["H_new_mgr", "A_new_mgr", "H_mgr_games", "A_mgr_games"],
    "5. Player ratings (plus-minus)": ["H_xi_rating", "A_xi_rating", "D_xi_rating",
                                       "H_xi_rating_vs_usual", "A_xi_rating_vs_usual"],
    "6. Unavailable players (suspended / not in squad / benched)": [
        "H_regulars_suspended", "A_regulars_suspended", "H_regulars_not_in_squad",
        "A_regulars_not_in_squad", "H_regulars_benched", "A_regulars_benched"],
}
XG = ["D_xGD_l6", "D_xGD_l19", "H_npxG_l19", "H_npxGA_l19", "A_npxG_l19", "A_npxGA_l19"]


def rps_each(p, y):
    o = np.eye(3)[y]
    return ((np.cumsum(p, 1)[:, :2] - np.cumsum(o, 1)[:, :2]) ** 2).sum(1) / 2


def paired(name, base_p, new_p, y):
    bc, nc = base_p.argmax(1) == y, new_p.argmax(1) == y
    br, nr = rps_each(base_p, y), rps_each(new_p, y)
    dc, dr = nc.astype(float) - bc, nr - br
    n = len(y)
    return {"signal": name, "matches": n, "accuracy": nc.mean(), "base_accuracy": bc.mean(),
            "acc_change_pts": 100 * dc.mean(), "acc_se_pts": 100 * dc.std(ddof=1) / np.sqrt(n),
            "rps": nr.mean(), "rps_change": dr.mean(), "rps_se": dr.std(ddof=1) / np.sqrt(n),
            "rps_change_in_se": dr.mean() / (dr.std(ddof=1) / np.sqrt(n))}


def backtest(df, feats):
    ps, ys = [], []
    for s in BACKTEST:
        tr = (df.Season >= "2013/14") & (df.Season < s)
        te = df.Season == s
        m = make_model("logreg").fit(df.loc[tr, feats], encode(df.loc[tr, "Result"]))
        ps.append(m.predict_proba(df.loc[te, feats]))
        ys.append(encode(df.loc[te, "Result"]))
    return np.vstack(ps), np.concatenate(ys)


def second_stage(df, base_p, extra, seasons):
    """Fit [base logits] vs [base logits + extra] on earlier seasons of the era; compare."""
    d = df[df.Season.isin(BACKTEST)].copy().reset_index(drop=True)
    d[["pH", "pD", "pA"]] = base_p
    d["lr_HA"], d["lr_D"] = np.log(d.pH / d.pA), np.log(d.pD / (1 - d.pD))
    d = d.dropna(subset=extra).reset_index(drop=True)
    out_b, out_n, ys = [], [], []
    for s in seasons[1:]:
        tr, te = d.Season.isin(seasons[:seasons.index(s)]), d.Season == s
        if te.sum() == 0:
            continue
        y_tr = encode(d.loc[tr, "Result"])
        for cols, store in ((["lr_HA", "lr_D"], out_b), (["lr_HA", "lr_D"] + extra, out_n)):
            mu, sd = d.loc[tr, cols].mean(), d.loc[tr, cols].std()
            clf = LogisticRegression(C=0.5, max_iter=3000).fit((d.loc[tr, cols] - mu) / sd, y_tr)
            store.append(clf.predict_proba((d.loc[te, cols] - mu) / sd))
        ys.append(encode(d.loc[te, "Result"]))
    return np.vstack(out_b), np.vstack(out_n), np.concatenate(ys)


def main():
    m = pd.read_csv(ROOT / "data" / "processed" / "matches.csv", parse_dates=["Date"])
    elo = json.load(open(ROOT / "reports" / "metrics.json"))["elo"]
    df = build_features(m, **elo)
    key = ["Date", "HomeTeam", "AwayTeam"]
    lu = pd.read_csv(ROOT / "data" / "processed" / "lineup_features.csv", parse_dates=["Date"])
    ms = pd.read_csv(ROOT / "data" / "processed" / "more_signals.csv", parse_dates=["Date"])
    df = df.merge(lu[key + SQUAD + LINEUP], on=key, how="left").merge(ms.drop(columns="game_id"), on=key, how="left")

    base_p, y = backtest(df, BASE)
    rows = []
    all_new = []
    for name, cols in GROUPS.items():
        p, _ = backtest(df, BASE + cols)
        rows.append(paired(name, base_p, p, y))
        all_new += cols
        print(rows[-1]["signal"], round(rows[-1]["rps_change"], 5), round(rows[-1]["rps_change_in_se"], 2), flush=True)
    p_all, _ = backtest(df, BASE + all_new)
    rows.append(paired("All five combined", base_p, p_all, y))

    # xG: second stage in the Understat era
    xg_seasons = ["2019/20", "2020/21", "2021/22", "2022/23", "2023/24", "2024/25"]
    b, n, yx = second_stage(df, base_p, XG, xg_seasons)
    rows.append(paired("4a. Expected goals (xG), six features, 2020/21-2024/25", b, n, yx))
    b, n, yx = second_stage(df, base_p, ["D_xGD_l19"], xg_seasons)
    rows.append(paired("4b. xG: single 19-match xG difference (second attempt)", b, n, yx))

    # odds movement, if the full Football-Data files are present
    fd_dir = ROOT / "data" / "raw" / "football_data_full"
    files = sorted(fd_dir.glob("E0_*.csv")) if fd_dir.exists() else []
    if files:
        from odds_movement import load_odds
        od = load_odds(files)
        df = df.merge(od, on=key, how="left")
        om = ["move_H", "move_A", "move_D"]
        seasons = [s for s in BACKTEST if df.loc[df.Season == s, "move_H"].notna().any()]
        b, n, yo = second_stage(df, base_p, om, seasons)
        rows.append(paired("7. Odds movement (opening -> closing)", b, n, yo))
        # reference: the bookmakers' own opening and closing prices on the same seasons
        ref = df[df.Season.isin(seasons[1:])].dropna(subset=["oH", "cH"])
        yr = encode(ref.Result)
        for label, cols in (("Bookmakers, opening odds", ["oH", "oD", "oA"]), ("Bookmakers, closing odds", ["cH", "cD", "cA"])):
            pm = ref[cols].to_numpy()
            rows.append({"signal": label, "matches": len(yr), "accuracy": (pm.argmax(1) == yr).mean(),
                         "rps": rps_each(pm, yr).mean()})
    out = pd.DataFrame(rows)
    out.round(5).to_csv(ROOT / "reports" / "more_signals_test.csv", index=False)
    pd.set_option("display.width", 220)
    print(out[["signal", "matches", "accuracy", "acc_change_pts", "acc_se_pts", "rps", "rps_change", "rps_se",
               "rps_change_in_se"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
