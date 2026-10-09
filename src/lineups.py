"""Team-news features from confirmed starting lineups (Transfermarkt, 2012/13-2025/26).

Injuries, suspensions and rotation all show up the same way: a player who
normally starts isn't in today's XI. Lineups are published ~1 hour before
kickoff, so these are legitimate pre-match features.

For every Premier League match and team:
- xi_logval      : log of the starting XI's total market value. Each player's value
                   is his latest Transfermarkt valuation dated BEFORE the match.
- xi_vs_usual    : today's XI value relative to the team's average XI value over its
                   previous 10 league matches (log difference). Negative = weakened.
- regulars_out   : how many of the team's 11 most frequent starters over the previous
                   10 league matches are not starting today.
- squad_logval   : log value of the 11 most valuable players who started at least once
                   in the previous 20 league matches. Known days before kickoff (no
                   lineup needed), so it shows how much of the gain needs the lineup.

Source: github.com/dcaribou/transfermarkt-datasets (CC0), files in data/raw/transfermarkt/.
Output: data/processed/lineup_features.csv
"""
from __future__ import annotations

from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TM = ROOT / "data" / "raw" / "transfermarkt"

TM_TO_FD = {
    "Arsenal FC": "Arsenal", "West Ham United": "West Ham", "Manchester City": "Man City",
    "Everton FC": "Everton", "Chelsea FC": "Chelsea", "Manchester United": "Man United",
    "Tottenham Hotspur": "Tottenham", "Liverpool FC": "Liverpool", "Newcastle United": "Newcastle",
    "Crystal Palace": "Crystal Palace", "Southampton FC": "Southampton", "Aston Villa": "Aston Villa",
    "Leicester City": "Leicester", "Burnley FC": "Burnley", "AFC Bournemouth": "Bournemouth",
    "Brighton & Hove Albion": "Brighton", "Fulham FC": "Fulham", "Wolverhampton Wanderers": "Wolves",
    "West Bromwich Albion": "West Brom", "Swansea City": "Swansea", "Stoke City": "Stoke",
    "Sunderland AFC": "Sunderland", "Watford FC": "Watford", "Norwich City": "Norwich",
    "Brentford FC": "Brentford", "Leeds United": "Leeds", "Nottingham Forest": "Nott'm Forest",
    "Hull City": "Hull", "Sheffield United": "Sheffield United", "Queens Park Rangers": "QPR",
    "Cardiff City": "Cardiff", "Huddersfield Town": "Huddersfield", "Reading FC": "Reading",
    "Wigan Athletic": "Wigan", "Middlesbrough FC": "Middlesbrough", "Luton Town": "Luton",
    "Ipswich Town": "Ipswich",
}


def load_starters() -> pd.DataFrame:
    games = pd.read_csv(TM / "games.csv.gz", parse_dates=["date"])
    games = games[games.competition_id == "GB1"][["game_id", "date", "home_club_id", "away_club_id",
                                                   "home_club_name", "away_club_name"]]
    lu = pd.read_csv(TM / "game_lineups.csv.gz", usecols=["game_id", "player_id", "club_id", "type"])
    lu = lu[(lu.type == "starting_lineup") & lu.game_id.isin(games.game_id)]
    lu = lu.merge(games[["game_id", "date"]], on="game_id")

    # each player's value: latest valuation strictly before the match date
    val = pd.read_csv(TM / "player_valuations.csv.gz", usecols=["player_id", "date", "market_value_in_eur"],
                      parse_dates=["date"]).sort_values("date")
    lu = lu.sort_values("date")
    lu["lookup"] = lu["date"] - pd.Timedelta(days=1)
    lu = pd.merge_asof(lu, val.rename(columns={"date": "val_date"}), left_on="lookup", right_on="val_date",
                       by="player_id", direction="backward")
    # players with no earlier valuation (young debutants): smallest typical value
    lu["value"] = lu.market_value_in_eur.fillna(lu.market_value_in_eur.quantile(0.05))
    return games, lu


def team_rows(games: pd.DataFrame, lu: pd.DataFrame) -> pd.DataFrame:
    xi = lu.groupby(["game_id", "club_id"]).agg(xi_value=("value", "sum"), n_starters=("player_id", "size"),
                                                players=("player_id", list)).reset_index()
    rows = []
    for side in ("home", "away"):
        t = games[["game_id", "date", f"{side}_club_id"]].rename(columns={f"{side}_club_id": "club_id"})
        rows.append(t.merge(xi, on=["game_id", "club_id"], how="left").assign(side=side))
    t = pd.concat(rows).sort_values(["club_id", "date"]).reset_index(drop=True)

    value_now = lu.set_index(["game_id", "player_id"])["value"]
    out = []
    for club, d in t.groupby("club_id", sort=False):
        hist_xi, hist_players = deque(maxlen=10), deque(maxlen=20)
        last_vals: dict[int, float] = {}
        for r in d.itertuples(index=False):
            players = r.players if isinstance(r.players, list) else []
            usual = np.mean(hist_xi) if hist_xi else np.nan
            counts = Counter(p for lst in list(hist_players)[-10:] for p in lst)
            regulars = [p for p, _ in counts.most_common(11)]
            recent = {p for lst in hist_players for p in lst}
            squad = sorted((last_vals[p] for p in recent if p in last_vals), reverse=True)[:11]
            out.append({
                "game_id": r.game_id, "side": r.side,
                "xi_logval": np.log(r.xi_value) if r.xi_value and r.xi_value > 0 else np.nan,
                "xi_vs_usual": np.log(r.xi_value / usual) if hist_xi and r.xi_value else np.nan,
                "regulars_out": sum(p not in players for p in regulars) if len(hist_xi) >= 5 else np.nan,
                "squad_logval": np.log(sum(squad)) if len(squad) >= 11 else np.nan,
                "n_starters": r.n_starters,
            })
            # update history AFTER computing today's features (no leakage)
            if players:
                hist_xi.append(r.xi_value)
                hist_players.append(players)
                for p in players:
                    last_vals[p] = value_now.get((r.game_id, p), last_vals.get(p, np.nan))
    return pd.DataFrame(out)


def main() -> None:
    games, lu = load_starters()
    feats = team_rows(games, lu)
    wide = feats.pivot(index="game_id", columns="side")
    wide.columns = [f"{'H' if s == 'home' else 'A'}_{c}" for c, s in wide.columns]
    wide = wide.reset_index().merge(games, on="game_id")
    wide["HomeTeam"] = wide.home_club_name.map(TM_TO_FD)
    wide["AwayTeam"] = wide.away_club_name.map(TM_TO_FD)
    for c in ["xi_logval", "xi_vs_usual", "regulars_out", "squad_logval"]:
        wide[f"D_{c}"] = wide[f"H_{c}"] - wide[f"A_{c}"]
    wide = wide.rename(columns={"date": "Date"}).drop(columns=["home_club_id", "away_club_id",
                                                               "home_club_name", "away_club_name"])
    out = ROOT / "data" / "processed" / "lineup_features.csv"
    wide.to_csv(out, index=False)
    print(f"{len(wide)} games; unmapped names: {wide.HomeTeam.isna().sum() + wide.AwayTeam.isna().sum()}")
    print("starters per team:", wide[["H_n_starters", "A_n_starters"]].stack().value_counts().to_dict())
    print(wide[["D_xi_logval", "D_xi_vs_usual", "D_regulars_out", "D_squad_logval"]].describe().round(2))


if __name__ == "__main__":
    main()
