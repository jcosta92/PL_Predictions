"""More detailed team-news features from the confirmed lineups (Transfermarkt).

lineups.py values the starting XI as one number. This file asks WHO is missing and
how the XI is put together. All features use only matches before the one predicted.

Per team and match:
- line_logval_{GK,DEF,MID,ATT} : log value of the starters in each line
- line_vs_usual_{...}           : same, relative to the team's previous 10 league matches
- star_out        : 1 if the most valuable regular (last 10 matches) is not starting
- missing_share   : value of the regulars who are not starting, as a share of the
                    usual XI value (0 = full-strength side)
- gk_change       : 1 if today's goalkeeper is not the team's usual goalkeeper
- captain_out     : 1 if the player who captained most of the last 10 matches is absent
- cohesion        : mean number of the previous 10 league matches that today's starters
                    started together (11 = the same XI as all of the last 10 games)
- bench_logval    : log value of the named substitutes (squad depth)
- xi_age          : mean age of the starting XI in years

Output: data/processed/lineup_features_v2.csv
"""
from __future__ import annotations

from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd

from lineups import TM, TM_TO_FD

ROOT = Path(__file__).resolve().parents[1]
LINE = {"Goalkeeper": "GK",
        "Centre-Back": "DEF", "Right-Back": "DEF", "Left-Back": "DEF", "Defender": "DEF", "Sweeper": "DEF",
        "Central Midfield": "MID", "Defensive Midfield": "MID", "Attacking Midfield": "MID",
        "Right Midfield": "MID", "Left Midfield": "MID", "midfield": "MID", "Midfield": "MID",
        "Centre-Forward": "ATT", "Left Winger": "ATT", "Right Winger": "ATT", "Second Striker": "ATT", "Attack": "ATT"}
LINES = ["GK", "DEF", "MID", "ATT"]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.read_csv(TM / "games.csv.gz", parse_dates=["date"])
    games = games[games.competition_id == "GB1"][["game_id", "date", "home_club_id", "away_club_id",
                                                   "home_club_name", "away_club_name"]]
    lu = pd.read_csv(TM / "game_lineups.csv.gz",
                     usecols=["game_id", "player_id", "club_id", "type", "position", "team_captain"])
    lu = lu[lu.game_id.isin(games.game_id)].merge(games[["game_id", "date"]], on="game_id")
    lu["line"] = lu.position.map(LINE).fillna("MID")

    val = pd.read_csv(TM / "player_valuations.csv.gz", usecols=["player_id", "date", "market_value_in_eur"],
                      parse_dates=["date"]).sort_values("date")
    lu = lu.sort_values("date")
    lu["lookup"] = lu["date"] - pd.Timedelta(days=1)
    lu = pd.merge_asof(lu, val.rename(columns={"date": "val_date"}), left_on="lookup", right_on="val_date",
                       by="player_id", direction="backward")
    lu["value"] = lu.market_value_in_eur.fillna(lu.market_value_in_eur.quantile(0.05))

    dob = pd.read_csv(TM / "players.csv.gz", usecols=["player_id", "date_of_birth"], parse_dates=["date_of_birth"])
    lu = lu.merge(dob, on="player_id", how="left")
    lu["age"] = (lu.date - lu.date_of_birth).dt.days / 365.25
    return games, lu


def team_rows(games: pd.DataFrame, lu: pd.DataFrame) -> pd.DataFrame:
    starters = lu[lu.type == "starting_lineup"]
    bench = lu[lu.type == "substitutes"].groupby(["game_id", "club_id"])["value"].sum().rename("bench_value")
    per = starters.groupby(["game_id", "club_id"]).agg(
        players=("player_id", list), values=("value", list), lines=("line", list),
        captains=("team_captain", list), xi_age=("age", "mean")).reset_index()
    per = per.join(bench, on=["game_id", "club_id"])

    rows = []
    for side in ("home", "away"):
        t = games[["game_id", "date", f"{side}_club_id"]].rename(columns={f"{side}_club_id": "club_id"})
        rows.append(t.merge(per, on=["game_id", "club_id"], how="left").assign(side=side))
    t = pd.concat(rows).sort_values(["club_id", "date"]).reset_index(drop=True)

    out = []
    for club, d in t.groupby("club_id", sort=False):
        hist = deque(maxlen=10)          # (players, line value dict, gk, captain)
        last_val: dict[int, float] = {}
        for r in d.itertuples(index=False):
            players = r.players if isinstance(r.players, list) else []
            feat = {"game_id": r.game_id, "side": r.side,
                    "bench_logval": np.log(r.bench_value) if r.bench_value and r.bench_value > 0 else np.nan,
                    "xi_age": r.xi_age}
            line_val = {ln: 0.0 for ln in LINES}
            gk, cap = None, None
            if players:
                for p, v, ln, c in zip(players, r.values, r.lines, r.captains):
                    line_val[ln] += v
                    if ln == "GK" and gk is None:
                        gk = p
                    if c == 1:
                        cap = p
            for ln in LINES:
                feat[f"line_logval_{ln}"] = np.log(line_val[ln]) if line_val[ln] > 0 else np.nan
                usual = np.mean([h[1][ln] for h in hist]) if hist else np.nan
                feat[f"line_vs_usual_{ln}"] = (np.log(line_val[ln] / usual)
                                               if hist and line_val[ln] > 0 and usual > 0 else np.nan)
            if len(hist) >= 5 and players:
                counts = Counter(p for h in hist for p in h[0])
                regulars = [p for p, _ in counts.most_common(11)]
                usual_xi_val = np.mean([sum(h[1].values()) for h in hist])
                missing = [p for p in regulars if p not in players]
                star = max(regulars, key=lambda p: last_val.get(p, 0.0))
                feat["star_out"] = int(star not in players)
                feat["missing_share"] = sum(last_val.get(p, 0.0) for p in missing) / usual_xi_val if usual_xi_val > 0 else np.nan
                usual_gk = Counter(h[2] for h in hist if h[2] is not None).most_common(1)
                feat["gk_change"] = int(bool(usual_gk) and gk != usual_gk[0][0])
                usual_cap = Counter(h[3] for h in hist if h[3] is not None).most_common(1)
                feat["captain_out"] = int(bool(usual_cap) and usual_cap[0][0] not in players)
                feat["cohesion"] = np.mean([counts.get(p, 0) for p in players])
            else:
                for c in ("star_out", "missing_share", "gk_change", "captain_out", "cohesion"):
                    feat[c] = np.nan
            out.append(feat)
            if players:  # update history after computing today's features
                hist.append((players, line_val, gk, cap))
                for p, v in zip(players, r.values):
                    last_val[p] = v
    return pd.DataFrame(out)


COLS = ["bench_logval", "xi_age", "star_out", "missing_share", "gk_change", "captain_out", "cohesion"] + \
       [f"line_logval_{ln}" for ln in LINES] + [f"line_vs_usual_{ln}" for ln in LINES]


def main() -> None:
    games, lu = load()
    feats = team_rows(games, lu)
    wide = feats.pivot(index="game_id", columns="side")
    wide.columns = [f"{'H' if s == 'home' else 'A'}_{c}" for c, s in wide.columns]
    wide = wide.reset_index().merge(games, on="game_id")
    wide["HomeTeam"] = wide.home_club_name.map(TM_TO_FD)
    wide["AwayTeam"] = wide.away_club_name.map(TM_TO_FD)
    for c in COLS:
        wide[f"D_{c}"] = wide[f"H_{c}"] - wide[f"A_{c}"]
    wide = wide.rename(columns={"date": "Date"}).drop(columns=["home_club_id", "away_club_id",
                                                               "home_club_name", "away_club_name"])
    wide.to_csv(ROOT / "data" / "processed" / "lineup_features_v2.csv", index=False)
    print(len(wide), "games")
    print(wide[[f"H_{c}" for c in COLS]].describe().T.round(3))


if __name__ == "__main__":
    main()
