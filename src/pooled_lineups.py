"""Does training the lineup model on five leagues help the Premier League predictions?

The lineup features have only ~2,300 training matches when the backtest starts
(2013/14-2018/19). Transfermarkt has lineups for La Liga, Serie A, Bundesliga and
Ligue 1 as well, so the same features can be built there and the model trained on
all five leagues (with a league indicator). It is still scored on Premier League
matches only, and compared with the same feature set trained on the Premier League
alone, and with the lineup model.

Transfermarkt club names are matched to Football-Data names by date: for each
Transfermarkt match, the Football-Data match in the same league on the same day with
the most similar team names.

Output: data/processed/lineup_features_pooled.csv, reports/pooled_lineups.csv
"""
from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

import lineups
from evaluate import BACKTEST, encode
from improve import CONFIRM, DEV, compare, logreg, rps_each
from test_lineups import LINEUP, SQUAD

ROOT = Path(__file__).resolve().parents[1]
TM = lineups.TM
COMPS = {"GB1": "E0", "ES1": "SP1", "IT1": "I1", "L1": "D1", "FR1": "F1"}
FEATS = ["EloDiff", "HomeElo", "AwayElo", "D_GD_l6", "D_Pts_l6", "D_GD_l19", "D_Pts_l19",
         "D_GD_ewm", "D_Pts_ewm", "H_n_prev", "A_n_prev", "H_rest", "A_rest"] + SQUAD + LINEUP


def load_starters_all():
    games = pd.read_csv(TM / "games.csv.gz", parse_dates=["date"])
    games = games[games.competition_id.isin(COMPS)][["game_id", "competition_id", "date", "home_club_id",
                                                     "away_club_id", "home_club_name", "away_club_name"]]
    lu = pd.read_csv(TM / "game_lineups.csv.gz", usecols=["game_id", "player_id", "club_id", "type"])
    lu = lu[(lu.type == "starting_lineup") & lu.game_id.isin(games.game_id)]
    lu = lu.merge(games[["game_id", "date"]], on="game_id")
    val = pd.read_csv(TM / "player_valuations.csv.gz", usecols=["player_id", "date", "market_value_in_eur"],
                      parse_dates=["date"]).sort_values("date")
    lu = lu.sort_values("date")
    lu["lookup"] = lu["date"] - pd.Timedelta(days=1)
    lu = pd.merge_asof(lu, val.rename(columns={"date": "val_date"}), left_on="lookup", right_on="val_date",
                       by="player_id", direction="backward")
    lu["value"] = lu.market_value_in_eur.fillna(lu.market_value_in_eur.quantile(0.05))
    return games, lu


def _norm(s: str) -> str:
    s = s.lower()
    for w in ("fc", "cf", "ac", "as", "us", "ss", "sc", "ssc", "afc", "sv", "vfb", "vfl", "tsg", "fsv", "cd", "ud", "rcd",
              "1.", "1899", "04", "05", "96", "1860", "club", "de", "calcio", "olympique", "stade", "racing", "real"):
        s = f" {s} ".replace(f" {w} ", " ")
    return " ".join(s.replace("-", " ").split())


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def match_names(games: pd.DataFrame, fd: pd.DataFrame) -> dict[tuple[str, str], str]:
    """TM (league, club name) -> Football-Data name, by majority over date-matched games."""
    votes: dict[tuple[str, str], dict[str, int]] = {}
    fd_by = {k: g for k, g in fd.groupby(["League", "Date"])}
    for r in games.itertuples(index=False):
        lg = COMPS[r.competition_id]
        cands = pd.concat([fd_by.get((lg, r.date + pd.Timedelta(days=d)), pd.DataFrame()) for d in (-1, 0, 1)])
        if cands.empty:
            continue
        score = [sim(r.home_club_name, h) + sim(r.away_club_name, a) for h, a in zip(cands.HomeTeam, cands.AwayTeam)]
        best = cands.iloc[int(np.argmax(score))]
        if max(score) < 1.0:
            continue
        for tm, fdn in ((r.home_club_name, best.HomeTeam), (r.away_club_name, best.AwayTeam)):
            votes.setdefault((lg, tm), {}).setdefault(fdn, 0)
            votes[(lg, tm)][fdn] += 1
    return {k: max(v, key=v.get) for k, v in votes.items()}


def build() -> pd.DataFrame:
    games, lu = load_starters_all()
    feats = lineups.team_rows(games, lu)
    wide = feats.pivot(index="game_id", columns="side")
    wide.columns = [f"{'H' if s == 'home' else 'A'}_{c}" for c, s in wide.columns]
    wide = wide.reset_index().merge(games, on="game_id")
    for c in ["xi_logval", "xi_vs_usual", "regulars_out", "squad_logval"]:
        wide[f"D_{c}"] = wide[f"H_{c}"] - wide[f"A_{c}"]
    wide["League"] = wide.competition_id.map(COMPS)

    fd = pd.read_pickle(ROOT / "data" / "processed" / "features_v2.pkl")[["League", "Date", "HomeTeam", "AwayTeam"]]
    names = match_names(games, fd)
    wide["HomeTeam"] = [names.get((lg, n)) for lg, n in zip(wide.League, wide.home_club_name)]
    wide["AwayTeam"] = [names.get((lg, n)) for lg, n in zip(wide.League, wide.away_club_name)]
    # the Premier League map is hand-checked in lineups.py; use it there
    pl = wide.League == "E0"
    wide.loc[pl, "HomeTeam"] = wide.loc[pl, "home_club_name"].map(lineups.TM_TO_FD)
    wide.loc[pl, "AwayTeam"] = wide.loc[pl, "away_club_name"].map(lineups.TM_TO_FD)
    print("unmapped club names:", wide[wide.HomeTeam.isna() | wide.AwayTeam.isna()].groupby("League").size().to_dict())
    wide = wide.rename(columns={"date": "Date"}).drop(columns=["home_club_id", "away_club_id", "home_club_name",
                                                               "away_club_name", "competition_id"])
    wide.to_csv(ROOT / "data" / "processed" / "lineup_features_pooled.csv", index=False)
    return wide


def main():
    pooled = build()
    v2 = pd.read_pickle(ROOT / "data" / "processed" / "features_v2.pkl")
    key = ["League", "Date", "HomeTeam", "AwayTeam"]
    df = v2.merge(pooled[key + SQUAD + LINEUP], on=key, how="left")
    df = df[df.League.isin(COMPS.values())].reset_index(drop=True)
    print("matches with lineup features, by league:",
          df[df.Season >= "2013/14"].groupby("League").H_xi_logval.apply(lambda s: f"{s.notna().mean():.0%}").to_dict())
    for lg in COMPS.values():
        df[f"lg_{lg}"] = (df.League == lg).astype(int)
    lg_cols = [f"lg_{lg}" for lg in COMPS.values()]

    def run(pool, feats):
        ps = []
        for s in BACKTEST:
            tr = df.League.isin(pool) & (df.Season >= "2013/14") & (df.Season < s)
            te = (df.League == "E0") & (df.Season == s)
            m = logreg().fit(df.loc[tr, feats], encode(df.loc[tr, "Result"]))
            ps.append(m.predict_proba(df.loc[te, feats]))
        return np.vstack(ps)

    te = (df.League == "E0") & df.Season.isin(BACKTEST)
    y = encode(df.loc[te, "Result"])
    seasons = df.loc[te, "Season"].reset_index(drop=True)
    prev = np.load(ROOT / "reports" / "improve_probs.npz", allow_pickle=True)
    assert (prev["y"] == y).all(), "row order differs from improve.py"
    base = prev["lineup model"]

    rows = compare("lineup model", base, base, y, seasons, "D. pooled leagues")
    variants = {
        "same features, v2 form, PL only": (["E0"], FEATS),
        "pooled: 5 leagues, league indicator": (list(COMPS.values()), FEATS + lg_cols),
        "pooled: 5 leagues, no indicator": (list(COMPS.values()), FEATS),
    }
    for name, (pool, feats) in variants.items():
        p = run(pool, feats)
        rows.extend(compare(name, base, p, y, seasons, "D. pooled leagues"))
        r = rows[-3:]
        print(f"{name:45s} dev {r[0]['rps_change_in_se']:+5.2f}  confirm {r[1]['rps_change_in_se']:+5.2f}  "
              f"all rps {r[2]['rps']:.4f} acc {r[2]['accuracy']:.4f}", flush=True)
    pd.DataFrame(rows).round(5).to_csv(ROOT / "reports" / "pooled_lineups.csv", index=False)


if __name__ == "__main__":
    main()
