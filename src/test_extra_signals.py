"""Does head-to-head history or weather improve the v1 model?

Head-to-head: add the H2H features to the v1 model and run the same 7-season
backtest. Compare season by season and report the paired difference.

Weather: only available for Aug 2017 to Apr 2023, so it cannot go into the
main model. Instead, for each season 2019/20-2022/23:
  1. take the v1 model's out-of-sample probabilities,
  2. fit a small second-stage model on earlier weather-era seasons, once with
     only the v1 probabilities and once with probabilities + weather,
  3. compare both on the season.
If weather carries information the v1 model lacks, version (b) should win.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from evaluate import BACKTEST, RESULTS_FEATS, encode, fit_predict, rps
from extra_signals import WEATHER_COLS, add_h2h, add_weather
from features import build_features

ROOT = Path(__file__).resolve().parents[1]
H2H = ["H2H_n", "H2H_gd", "H2H_pts", "H2H_venue_gd", "H2H_draw_rate"]


def paired(a_correct, b_correct, a_rps, b_rps):
    d_acc, d_rps = b_correct.astype(float) - a_correct, b_rps - a_rps
    n = len(d_acc)
    return {"acc_change_pts": 100 * d_acc.mean(), "acc_change_se_pts": 100 * d_acc.std(ddof=1) / np.sqrt(n),
            "rps_change": d_rps.mean(), "rps_change_se": d_rps.std(ddof=1) / np.sqrt(n)}


def rps_each(p, y):
    o = np.eye(3)[y]
    return ((np.cumsum(p, 1)[:, :2] - np.cumsum(o, 1)[:, :2]) ** 2).sum(1) / 2


def main():
    m = pd.read_csv(ROOT / "data" / "processed" / "matches.csv", parse_dates=["Date"])
    elo = json.load(open(ROOT / "reports" / "metrics.json"))["elo"]
    df = add_h2h(build_features(m, **elo))
    usable = df.Season >= "2001/02"

    # ---------------- head-to-head
    rows, base_p, base_y, seasons = [], [], [], []
    A_c, B_c, A_r, B_r = [], [], [], []
    for s in BACKTEST:
        te = df.Season == s
        y = encode(df.loc[te, "Result"])
        pa = fit_predict(df, usable & (df.Season < s), te, RESULTS_FEATS, "logreg")
        pb = fit_predict(df, usable & (df.Season < s), te, RESULTS_FEATS + H2H, "logreg")
        rows.append({"season": s, "v1": (pa.argmax(1) == y).mean(), "v1 + head-to-head": (pb.argmax(1) == y).mean()})
        A_c.append(pa.argmax(1) == y); B_c.append(pb.argmax(1) == y)
        A_r.append(rps_each(pa, y)); B_r.append(rps_each(pb, y))
        base_p.append(pa); base_y.append(y); seasons += [s] * len(y)
    h2h_tbl = pd.DataFrame(rows)
    h2h_stats = paired(np.concatenate(A_c), np.concatenate(B_c), np.concatenate(A_r), np.concatenate(B_r))

    # ---------------- weather (second stage on v1 out-of-sample probabilities)
    bt = df[df.Season.isin(BACKTEST)].copy()
    bt[["p_H", "p_D", "p_A"]] = np.vstack(base_p)
    # v1 probabilities for 2017/18 and 2018/19 too, so stage 2 has training data
    early = []
    for s in ["2017/18", "2018/19"]:
        te = df.Season == s
        e = df[te].copy()
        e[["p_H", "p_D", "p_A"]] = fit_predict(df, usable & (df.Season < s), te, RESULTS_FEATS, "logreg")
        early.append(e)
    w = add_weather(pd.concat(early + [bt], ignore_index=True))
    w = w[w.temp.notna()].reset_index(drop=True)
    w["lr_HA"] = np.log(w.p_H / w.p_A)
    w["lr_D"] = np.log(w.p_D / (1 - w.p_D))
    base_cols = ["lr_HA", "lr_D"]
    wx_cols = base_cols + WEATHER_COLS + ["rainy", "windy"]

    w_rows, Ac, Bc, Ar, Br = [], [], [], [], []
    for s in ["2019/20", "2020/21", "2021/22", "2022/23"]:
        tr, te = w.Season < s, w.Season == s
        y_tr, y_te = encode(w.loc[tr, "Result"]), encode(w.loc[te, "Result"])
        res = {}
        for name, cols in (("v1 probabilities only", base_cols), ("v1 probabilities + weather", wx_cols)):
            mu, sd = w.loc[tr, cols].mean(), w.loc[tr, cols].std()
            clf = LogisticRegression(C=0.5, max_iter=2000).fit((w.loc[tr, cols] - mu) / sd, y_tr)
            p = clf.predict_proba((w.loc[te, cols] - mu) / sd)
            res[name] = (p.argmax(1) == y_te, rps_each(p, y_te))
        Ac.append(res["v1 probabilities only"][0]); Bc.append(res["v1 probabilities + weather"][0])
        Ar.append(res["v1 probabilities only"][1]); Br.append(res["v1 probabilities + weather"][1])
        w_rows.append({"season": s, "matches": int(te.sum()),
                       "v1 probabilities only": res["v1 probabilities only"][0].mean(),
                       "v1 probabilities + weather": res["v1 probabilities + weather"][0].mean()})
    wx_tbl = pd.DataFrame(w_rows)
    wx_stats = paired(np.concatenate(Ac), np.concatenate(Bc), np.concatenate(Ar), np.concatenate(Br))

    # descriptive: does weather change how matches end?
    w["goals"] = w.FTHG + w.FTAG
    w["draw"] = (w.Result == "D").astype(float)
    w["home_win"] = (w.Result == "H").astype(float)
    desc = []
    for label, mask in [("all weather-era matches", w.precip.notna()),
                        ("dry (<2 mm rain)", w.precip < 2), ("rainy (>=2 mm)", w.precip >= 2),
                        ("calm (<30 km/h wind)", w.windspeed < 30), ("windy (>=30 km/h)", w.windspeed >= 30),
                        ("cold (<8 C)", w.temp < 8), ("warm (>=15 C)", w.temp >= 15)]:
        x = w[mask]
        desc.append({"conditions": label, "matches": len(x), "home win %": 100 * x.home_win.mean(),
                     "draw %": 100 * x.draw.mean(), "goals per match": x.goals.mean()})
    desc = pd.DataFrame(desc)

    out = ROOT / "reports"
    h2h_tbl.round(4).to_csv(out / "h2h_backtest.csv", index=False)
    wx_tbl.round(4).to_csv(out / "weather_backtest.csv", index=False)
    desc.round(2).to_csv(out / "weather_descriptive.csv", index=False)
    json.dump({"head_to_head": h2h_stats, "weather": wx_stats}, open(out / "extra_signals.json", "w"), indent=2)

    print(h2h_tbl.round(3).to_string(index=False))
    print("H2H paired:", {k: round(v, 4) for k, v in h2h_stats.items()})
    print(wx_tbl.round(3).to_string(index=False))
    print("Weather paired:", {k: round(v, 4) for k, v in wx_stats.items()})
    print(desc.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
