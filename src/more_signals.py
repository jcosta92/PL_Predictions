"""Six more pre-match signals, each computed only from information available before kickoff.

1. Fixture congestion  : rest days and games in the last 14 days across ALL competitions,
                         a European match in the 4 days before, and one in the 4 days after
                         (fixtures are known in advance, and managers rotate for them).
2. What's at stake     : league table on the day (from earlier results only): points gap to
                         4th place and to the relegation zone, games left, and a "nothing to
                         play for" flag for the last 10 rounds.
3. Manager change      : league games since the current manager took over; new-manager flag.
4. Expected goals (xG) : rolling xG for/against from Understat (2019/20-2024/25 only).
5. Player ratings      : on-pitch goal difference per 90 minutes for every player (plus-minus,
                         shrunk toward zero), from goals and substitution minutes. The XI's
                         average rating and its gap to the team's usual XI.
6. Unavailable players : regulars suspended (red cards, 5/10-yellow bans, known days ahead),
                         and regulars missing from the whole matchday squad (not even on the
                         bench = injured or suspended) vs. merely benched (rotation).

Output: data/processed/more_signals.csv (one row per Premier League match, 2012/13-2025/26)
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from lineups import TM_TO_FD

ROOT = Path(__file__).resolve().parents[1]
TM = ROOT / "data" / "raw" / "transfermarkt"
EURO = {"CL", "EL", "UCOL", "CLQ", "ELQ", "ECLQ"}

UNDERSTAT_TO_FD = {
    "Arsenal": "Arsenal", "Aston_Villa": "Aston Villa", "Bournemouth": "Bournemouth", "Brentford": "Brentford",
    "Brighton": "Brighton", "Burnley": "Burnley", "Chelsea": "Chelsea", "Crystal_Palace": "Crystal Palace",
    "Everton": "Everton", "Fulham": "Fulham", "Ipswich": "Ipswich", "Leeds": "Leeds", "Leicester": "Leicester",
    "Liverpool": "Liverpool", "Luton": "Luton", "Manchester_City": "Man City", "Manchester_United": "Man United",
    "Newcastle_United": "Newcastle", "Norwich": "Norwich", "Nottingham_Forest": "Nott'm Forest",
    "Sheffield_United": "Sheffield United", "Southampton": "Southampton", "Tottenham": "Tottenham",
    "Watford": "Watford", "West_Bromwich_Albion": "West Brom", "West_Ham": "West Ham",
    "Wolverhampton_Wanderers": "Wolves",
}


def load():
    games = pd.read_csv(TM / "games.csv.gz", parse_dates=["date"])
    pl = games[games.competition_id == "GB1"].copy()
    pl["HomeTeam"] = pl.home_club_name.map(TM_TO_FD)
    pl["AwayTeam"] = pl.away_club_name.map(TM_TO_FD)
    return games, pl.sort_values(["date", "game_id"]).reset_index(drop=True)


# ------------------------------------------------------------- 1. congestion
def congestion(games: pd.DataFrame, pl: pd.DataFrame) -> pd.DataFrame:
    clubs = set(pl.home_club_id) | set(pl.away_club_id)
    rows = []
    for side in ("home", "away"):
        t = games[[f"{side}_club_id", "date", "competition_id", "game_id"]].rename(columns={f"{side}_club_id": "club"})
        rows.append(t[t.club.isin(clubs)])
    allg = pd.concat(rows).drop_duplicates(["club", "game_id"]).sort_values(["club", "date"])
    by_club = {c: (d.date.to_numpy(), d.competition_id.to_numpy()) for c, d in allg.groupby("club")}
    out = []
    for r in pl.itertuples(index=False):
        rec = {"game_id": r.game_id}
        for side, club in (("H", r.home_club_id), ("A", r.away_club_id)):
            dates, comps = by_club[club]
            d = np.datetime64(r.date)
            before = dates < d
            prev = dates[before]
            rec[f"{side}_rest_all"] = min((d - prev[-1]).astype("timedelta64[D]").astype(int), 14) if len(prev) else 14
            rec[f"{side}_games_14d"] = int(((prev >= d - np.timedelta64(14, "D"))).sum())
            win_b = before & (dates >= d - np.timedelta64(4, "D"))
            win_a = (dates > d) & (dates <= d + np.timedelta64(4, "D"))
            rec[f"{side}_euro_before"] = int(any(c in EURO for c in comps[win_b]))
            rec[f"{side}_euro_after"] = int(any(c in EURO for c in comps[win_a]))
        out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------- 2. stakes (table)
def stakes(pl: pd.DataFrame) -> pd.DataFrame:
    out = []
    for season, d in pl.groupby("season"):
        pts, gd, played = defaultdict(int), defaultdict(int), defaultdict(int)
        teams = set(d.HomeTeam) | set(d.AwayTeam)
        for date, day in d.groupby("date", sort=True):
            # table as it stood before this date
            tbl = sorted(teams, key=lambda t: (pts[t], gd[t]), reverse=True)
            pos = {t: i + 1 for i, t in enumerate(tbl)}
            p4, p17 = pts[tbl[3]], pts[tbl[16]]
            for r in day.itertuples(index=False):
                rec = {"game_id": r.game_id}
                for side, t in (("H", r.HomeTeam), ("A", r.AwayTeam)):
                    left = 38 - played[t]
                    gap_top4 = pts[t] - p4          # >0: inside the top 4 by that many points
                    gap_safety = pts[t] - p17       # vs 17th (last safe place)
                    reach = 3 * left
                    late = left <= 10
                    nothing = late and gap_safety > reach * 0.5 and gap_top4 < -reach * 0.5 and pos[t] > 6
                    rec.update({f"{side}_pos": pos[t] if played[t] else np.nan,
                                f"{side}_gap_top4": gap_top4, f"{side}_gap_safety": gap_safety,
                                f"{side}_games_left": left, f"{side}_nothing_to_play": int(nothing),
                                f"{side}_relegation_fight": int(late and gap_safety <= reach * 0.5)})
                out.append(rec)
            for r in day.itertuples(index=False):   # update AFTER the day's features
                hg, ag = r.home_club_goals, r.away_club_goals
                for t, gf, ga in ((r.HomeTeam, hg, ag), (r.AwayTeam, ag, hg)):
                    played[t] += 1
                    gd[t] += gf - ga
                    pts[t] += 3 if gf > ga else 1 if gf == ga else 0
    return pd.DataFrame(out)


# --------------------------------------------------------- 3. manager change
def managers(pl: pd.DataFrame) -> pd.DataFrame:
    since: dict[str, int] = {}
    current: dict[str, str] = {}
    out = []
    for r in pl.itertuples(index=False):
        rec = {"game_id": r.game_id}
        for side, t, mgr in (("H", r.HomeTeam, r.home_club_manager_name), ("A", r.AwayTeam, r.away_club_manager_name)):
            if current.get(t) != mgr:
                since[t] = 0 if t in current else 50   # unknown history at the start: treat as established
                current[t] = mgr
            rec[f"{side}_mgr_games"] = min(since[t], 50)
            rec[f"{side}_new_mgr"] = int(since[t] < 6)
        out.append(rec)
        since[r.HomeTeam] += 1
        since[r.AwayTeam] += 1
    return pd.DataFrame(out)


# -------------------------------------------------------------------- 4. xG
def xg(pl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in sorted((ROOT / "data" / "raw" / "understat").glob("*/understat_*.csv")):
        team = UNDERSTAT_TO_FD[f.stem.replace("understat_", "")]
        d = pd.read_csv(f, usecols=["date", "h_a", "xG", "xGA", "npxG", "npxGA"])
        d["Team"] = team
        rows.append(d)
    u = pd.concat(rows)
    u["date"] = pd.to_datetime(u.date).dt.normalize()
    u = u.drop_duplicates(["Team", "date"]).sort_values(["Team", "date"])
    g = u.groupby("Team", group_keys=False)
    for s in ["xG", "xGA", "npxG", "npxGA"]:
        u[f"{s}_l6"] = g[s].apply(lambda x: x.shift(1).rolling(6, min_periods=3).mean())
        u[f"{s}_l19"] = g[s].apply(lambda x: x.shift(1).rolling(19, min_periods=5).mean())
    feats = [c for c in u.columns if c.endswith(("_l6", "_l19"))]
    out = pl[["game_id", "date", "HomeTeam", "AwayTeam"]].copy()
    out["date"] = out.date.dt.normalize()
    for side, col in (("H", "HomeTeam"), ("A", "AwayTeam")):
        x = u[["Team", "date"] + feats].rename(columns={"Team": col}).add_prefix(f"{side}_")
        x = x.rename(columns={f"{side}_{col}": col, f"{side}_date": "date"})
        out = out.merge(x, on=[col, "date"], how="left")
    for w in ("l6", "l19"):
        out[f"D_xGD_{w}"] = (out[f"H_npxG_{w}"] - out[f"H_npxGA_{w}"]) - (out[f"A_npxG_{w}"] - out[f"A_npxGA_{w}"])
    return out.drop(columns=["date", "HomeTeam", "AwayTeam"])


# ------------------------------------------- 5 & 6. player ratings, absences
def players(pl: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    lu = pd.read_csv(TM / "game_lineups.csv.gz", usecols=["game_id", "player_id", "club_id", "type"])
    lu = lu[lu.game_id.isin(pl.game_id)]
    ev = pd.read_csv(TM / "game_events.csv.gz", usecols=["game_id", "minute", "type", "club_id", "player_id",
                                                         "player_in_id", "description"])
    ev = ev[ev.game_id.isin(pl.game_id)]
    goals = ev[ev.type == "Goals"].groupby("game_id")
    subs = ev[ev.type == "Substitutions"].groupby("game_id")
    cards = ev[ev.type == "Cards"].copy()
    cards["kind"] = np.select([cards.description.str.contains("Second yellow"), cards.description.str.contains("Red card")],
                              ["second_yellow", "red"], "yellow")
    cards = cards.groupby("game_id")
    starters = lu[lu.type == "starting_lineup"].groupby(["game_id", "club_id"]).player_id.apply(list).to_dict()
    squad = lu.groupby(["game_id", "club_id"]).player_id.apply(set).to_dict()

    # rating state
    pm_gd, pm_min = defaultdict(lambda: deque(maxlen=38)), defaultdict(lambda: deque(maxlen=38))
    team_hist = defaultdict(lambda: deque(maxlen=10))          # recent XIs per club
    team_xi_rating = defaultdict(lambda: deque(maxlen=10))
    yellows, banned = defaultdict(int), defaultdict(int)       # (season, player) -> yellows; player -> games banned
    player_club: dict[int, int] = {}                           # last club each player appeared for
    league_games = defaultdict(int)                            # (season, club) -> league games played
    # bans are also served in domestic cup games (FA Cup, League Cup)
    cups = games[games.competition_id.isin(["FAC", "CGB"])]
    cup_dates = defaultdict(list)
    for side in ("home", "away"):
        for c, d in zip(cups[f"{side}_club_id"], cups.date):
            cup_dates[c].append(np.datetime64(d))
    cup_dates = {c: np.sort(np.array(v)) for c, v in cup_dates.items()}
    last_pl = {}

    def serve(club, n):
        if n <= 0:
            return
        for p in [p for p, b in banned.items() if b > 0 and player_club.get(p) == club]:
            banned[p] = max(0, banned[p] - n)
    season_of_game = dict(zip(pl.game_id, pl.season))
    K = 8.0                                                     # shrinkage, in 90-minute units

    def rating(p):
        m = sum(pm_min[p]) / 90
        return sum(pm_gd[p]) / (m + K)

    out = []
    for r in pl.itertuples(index=False):
        rec = {"game_id": r.game_id}
        d_now = np.datetime64(r.date)
        for club in (r.home_club_id, r.away_club_id):   # cup games since the last league game serve bans
            if club in last_pl and club in cup_dates:
                cd = cup_dates[club]
                serve(club, int(((cd > last_pl[club]) & (cd < d_now)).sum()))
        for side, club in (("H", r.home_club_id), ("A", r.away_club_id)):
            xi = starters.get((r.game_id, club), [])
            sq = squad.get((r.game_id, club), set())
            counts = Counter(p for lst in team_hist[club] for p in lst)
            regulars = [p for p, _ in counts.most_common(11)] if len(team_hist[club]) >= 5 else []
            xi_r = np.mean([rating(p) for p in xi]) if xi else np.nan
            usual = np.mean(team_xi_rating[club]) if team_xi_rating[club] else np.nan
            rec.update({
                f"{side}_xi_rating": xi_r,
                f"{side}_xi_rating_vs_usual": xi_r - usual if xi and team_xi_rating[club] else np.nan,
                f"{side}_regulars_suspended": sum(banned[p] > 0 for p in regulars) if regulars else np.nan,
                f"{side}_regulars_not_in_squad": sum(p not in sq for p in regulars) if regulars and sq else np.nan,
                f"{side}_regulars_benched": sum(p in sq and p not in xi for p in regulars) if regulars and sq else np.nan,
            })
        out.append(rec)

        # ---- update state with this match (after features were taken)
        season = season_of_game[r.game_id]
        for club in (r.home_club_id, r.away_club_id):
            serve(club, 1)                                      # this league game served one game of any ban
            last_pl[club] = d_now
            league_games[(season, club)] += 1
            for p in squad.get((r.game_id, club), set()):
                player_club[p] = club
            xi = starters.get((r.game_id, club), [])
            if xi:
                team_hist[club].append(xi)
                team_xi_rating[club].append(np.mean([rating(p) for p in xi]))
        if r.game_id in cards.groups:
            for c in cards.get_group(r.game_id).itertuples(index=False):
                p = c.player_id
                rnd = league_games[(season, c.club_id)]          # league round just played
                if c.kind == "yellow":
                    yellows[(season, p)] += 1
                    n = yellows[(season, p)]
                    if n == 5 and rnd <= 19:
                        banned[p] += 1
                    elif n == 10 and rnd <= 32:
                        banned[p] += 2
                    elif n == 15:
                        banned[p] += 3
                else:
                    banned[p] += 1                              # red or second yellow: at least one game
        # on-pitch intervals for plus-minus
        on = {}
        for club in (r.home_club_id, r.away_club_id):
            for p in starters.get((r.game_id, club), []):
                on[p] = [0, 90, club]
        if r.game_id in subs.groups:
            for s in subs.get_group(r.game_id).itertuples(index=False):
                m = min(int(s.minute), 90) if pd.notna(s.minute) else 90
                if s.player_id in on:
                    on[s.player_id][1] = m
                if pd.notna(s.player_in_id):
                    on[int(s.player_in_id)] = [m, 90, s.club_id]
        glist = list(goals.get_group(r.game_id).itertuples(index=False)) if r.game_id in goals.groups else []
        for p, (start, end, club) in on.items():
            gd = sum((1 if g.club_id == club else -1) for g in glist
                     if pd.notna(g.minute) and start < min(g.minute, 90) <= end)
            if end > start:
                pm_gd[p].append(gd)
                pm_min[p].append(end - start)
    return pd.DataFrame(out)


def main() -> None:
    games, pl = load()
    parts = [congestion(games, pl), stakes(pl), managers(pl), xg(pl), players(pl, games)]
    df = pl[["game_id", "date", "HomeTeam", "AwayTeam"]].rename(columns={"date": "Date"})
    for p in parts:
        df = df.merge(p, on="game_id", how="left")
    for c in ["rest_all", "games_14d", "euro_before", "euro_after", "gap_top4", "gap_safety", "mgr_games",
              "xi_rating", "xi_rating_vs_usual", "regulars_suspended", "regulars_not_in_squad", "regulars_benched"]:
        df[f"D_{c}"] = df[f"H_{c}"] - df[f"A_{c}"]
    df.to_csv(ROOT / "data" / "processed" / "more_signals.csv", index=False)
    print(df.shape)
    show = ["H_rest_all", "H_games_14d", "H_euro_before", "H_euro_after", "H_nothing_to_play", "H_relegation_fight",
            "H_new_mgr", "H_npxG_l6", "H_xi_rating", "H_regulars_suspended", "H_regulars_not_in_squad", "H_regulars_benched"]
    print(df[show].describe().T[["count", "mean", "min", "max"]].round(2))


if __name__ == "__main__":
    main()
