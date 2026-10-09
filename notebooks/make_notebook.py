"""Generates PL_Predictions_walkthrough.ipynb (run once; the notebook is the deliverable)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Premier League predictions: the full process

This notebook walks through how the model is built, step by step, with the reasoning behind each choice. Every number in it is calculated here from the data in this repo.

**The question:** before kickoff, can we predict whether a Premier League match ends in a home win (H), a draw (D) or an away win (A)?

**The plan**
1. Load the data and see what "normal" looks like
2. Set the benchmarks a model has to beat
3. The 2023 mistake: data leakage, and how to avoid it
4. Build features using only information available before kickoff
5. Train a model and test it on a season it has never seen
6. Repeat that test over 7 seasons to get a reliable number
7. Test your ideas: head-to-head, weather, team news (lineups) and seven more signals
8. What else was tried
9. When to trust the model: accuracy by confidence
10. This season and this weekend
11. Conclusions
""")

code("""
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# chart style: one colour per role, quiet axes
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.figsize": (9, 4), "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.8, "axes.axisbelow": True, "lines.linewidth": 2, "font.size": 10})
pd.set_option("display.width", 140)
""")

md("""
## 1. The data

`data/processed/matches.csv` combines two public sources:
- **Football-Data.co.uk** (via a GitHub mirror): every Premier League match since 2000/01, with score, shots and bookmaker odds.
- **openfootball**: results for the 2026/27 matches played after 31 August, which the first source doesn't have yet.

Built by `src/build_dataset.py`. Where the two sources overlap, the scores agree exactly.
""")

code("""
matches = pd.read_csv(ROOT / "data/processed/matches.csv", parse_dates=["Date"])
print(f"{len(matches):,} matches from {matches.Date.min():%d %b %Y} to {matches.Date.max():%d %b %Y}")
matches[["Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Result", "HS", "AS", "OddH", "OddD", "OddA"]].tail()
""")

md("""
**How often does each result happen?** This is the first thing to know, because a model has to do better than simply guessing the most common result.
""")

code("""
print((matches.Result.value_counts(normalize=True) * 100).round(1).rename("% of matches"))
""")

md("""
The home team wins about 45% of the time. So a "model" that always says *home win* is already right 45% of the time. That's the first benchmark.

The home advantage isn't constant, though. In 2020/21, when games were played without crowds because of COVID, it nearly disappeared:
""")

code("""
by_season = matches[matches.Season < "2026/27"].groupby("Season").Result.apply(lambda r: (r == "H").mean() * 100)
fig, ax = plt.subplots()
ax.plot(range(len(by_season)), by_season.values, color=BLUE, marker="o", markersize=4)
ax.set_xticks(range(0, len(by_season), 3), by_season.index[::3])
i = list(by_season.index).index("2020/21")
ax.annotate("2020/21: empty stadiums", (i, by_season.iloc[i]), xytext=(i - 7.5, by_season.iloc[i] + 0.3),
            color=INK, arrowprops=dict(arrowstyle="-", color=MUTED))
ax.set_ylabel("Home wins (% of matches)")
ax.set_title("Home win rate by season", loc="left", color=INK)
plt.show()
""")

md("""
## 2. The benchmarks

Any model must be compared to:
1. **Always pick the home team**: about 43–45%.
2. **The bookmakers.** Their odds turn into probabilities: odds of 2.00 mean roughly a 50% chance. Picking the bookmakers' favourite is the strongest public benchmark there is, because bookmakers price in team news, injuries and huge amounts of data.

`src/features.py` removes the bookmaker margin, so the three probabilities add up to 100%.
""")

code("""
from features import add_market_probs
mk = add_market_probs(matches.copy()).dropna(subset=["MktH"])
recent = mk[(mk.Season >= "2019/20") & (mk.Season <= "2025/26")]
fav = np.array(["H", "D", "A"])[recent[["MktH", "MktD", "MktA"]].to_numpy().argmax(1)]
print(f"Always home win, 2019/20-2025/26 : {(recent.Result == 'H').mean():.1%}")
print(f"Bookmakers' favourite, same seasons: {(fav == recent.Result).mean():.1%}")
print(f"How often the bookmakers' favourite is a draw: {(fav == 'D').mean():.1%}")
""")

md("""
Two lessons from this:
- Even the bookmakers are right only about **54%** of the time. Football is random: one deflection changes the result.
- **The favourite is never a draw.** Draws make up about 25% of matches, but in the 2,660 matches from 2019/20 to 2025/26 the bookmakers never once made a draw the single *most likely* outcome. So any model scored on "pick the most likely result" will rarely predict a draw. That isn't a flaw; it's how the probabilities work.
""")

md("""
## 3. The 2023 mistake: data leakage

**Leakage** means giving the model information it wouldn't have before kickoff. The 2023 notebook had two leaks:

1. **"Average goals in the last 5 matches" included the match being predicted.** If a team won 4–0, its "recent average" already contained those 4 goals, so the answer was hidden in the question.
2. **Accuracy was measured on matches the model had trained on.** A random forest memorises its training data, so it scores itself very well on it.

Here's the first leak on a real example. The fix is a single `.shift(1)`, which moves the window back one match so that it ends *before* the current game:
""")

code("""
ars = matches[(matches.HomeTeam == "Arsenal") | (matches.AwayTeam == "Arsenal")].copy()
ars["Arsenal_goals"] = np.where(ars.HomeTeam == "Arsenal", ars.FTHG, ars.FTAG)
ars = ars[ars.Season == "2025/26"].head(8)[["Date", "HomeTeam", "AwayTeam", "Arsenal_goals"]]
ars["WRONG: avg incl. this match"] = ars.Arsenal_goals.rolling(5, min_periods=1).mean().round(2)
ars["RIGHT: avg of previous matches"] = ars.Arsenal_goals.shift(1).rolling(5, min_periods=1).mean().round(2)
ars
""")

md("""
`src/leakage_demo.py` rebuilds the 2023 setup on the same seasons and shows what each leak was worth:
""")

code("""
pd.read_csv(ROOT / "reports/leakage_demo.csv")
""")

md("""
How to read this table:
- **With the leak**, the model looks brilliant: 92% on its own training data, 65% on the next season.
- **Fixed**, it still scores 89% on its training data. That's memorisation, and it's why you must never score a model on data it has seen.
- The honest number is the last column: about **50%** on a season it has never seen.

The rule used from here on: **every feature for a match is calculated only from matches played before it, and every accuracy figure is measured on matches the model has never seen.**
""")

md("""
## 4. Features: what the model gets to see

All built in `src/features.py`, using only past matches:

| Feature | What it means |
|---|---|
| **Elo rating** | A running strength score for each team. Win and you take points from your opponent; beat a strong team and you take more; win by a lot and you take more still. Ratings drift back toward the average between seasons, and promoted teams start below average. |
| **EloDiff** | Home Elo + home advantage − away Elo. The single most useful number. |
| **Form (last 5 and 10 matches)** | Goals scored, goals conceded and points per game, calculated *before* the match. |
| **Rest days** | Days since each team's previous match. |
| **Matches of history** | How much data exists on the team, so the model knows when a rating is still uncertain. |

**Each team's last results are already in here**, twice over: in the form features directly, and indirectly in Elo, which is a weighted summary of every previous result.
""")

code("""
from features import build_features
elo_cfg = json.load(open(ROOT / "reports/metrics.json"))["elo"]
print("Elo settings, tuned on seasons before 2025/26:", elo_cfg)
df = build_features(matches, **elo_cfg)

fig, ax = plt.subplots()
for team, col in [("Man City", BLUE), ("Liverpool", ORANGE), ("Arsenal", AQUA)]:
    h = df[df.HomeTeam == team][["Date", "HomeElo"]].rename(columns={"HomeElo": "Elo"})
    a = df[df.AwayTeam == team][["Date", "AwayElo"]].rename(columns={"AwayElo": "Elo"})
    s = pd.concat([h, a]).sort_values("Date")
    s = s[s.Date >= "2012-07-01"]
    ax.plot(s.Date, s.Elo, color=col, label=team, linewidth=1.6)
    ax.annotate(team, (s.Date.iloc[-1], s.Elo.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                color=INK, va="center")
ax.axhline(1500, color=MUTED, linewidth=1, linestyle="--")
ax.legend(loc="upper left", frameon=False)
ax.set_ylabel("Elo rating (1500 = league average)")
ax.set_title("Elo ratings, 2012 to today", loc="left", color=INK)
plt.show()
""")

md("""
Here's everything the model sees for one match, all of it known before kickoff:
""")

code("""
from evaluate import RESULTS_FEATS
example = df[(df.HomeTeam == "Liverpool") & (df.AwayTeam == "Man City")].iloc[-1]
print(example[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].to_string(), "\\n")
example[RESULTS_FEATS].round(2).to_frame("value")
""")

md("""
## 5. Train on the past, test on an unseen season

**Time-based split:** train on every season up to 2024/25, then predict 2025/26. A random split would let the model learn from the future.

**The model** is a multinomial logistic regression. It turns the features into three probabilities that add up to 100%. It was chosen because it was as accurate as the more complex models (section 8) while being simple and hard to overfit. `C=0.1` adds regularisation, which stops the model leaning too hard on any one feature.
""")

code("""
from evaluate import make_model, encode, CLASSES
usable = df.Season >= "2001/02"            # first season only warms up the ratings
train = usable & (df.Season < "2025/26")
test = df.Season == "2025/26"

model = make_model("logreg").fit(df.loc[train, RESULTS_FEATS], encode(df.loc[train, "Result"]))
proba = model.predict_proba(df.loc[test, RESULTS_FEATS])
pick = np.array(CLASSES)[proba.argmax(1)]
actual = df.loc[test, "Result"].to_numpy()

print(f"Trained on {train.sum():,} matches, tested on {test.sum()} matches of 2025/26")
print(f"Model accuracy       : {(pick == actual).mean():.1%}")
print(f"Always home win      : {(actual == 'H').mean():.1%}")
mk25 = df.loc[test, ["MktH", "MktD", "MktA"]].to_numpy()
print(f"Bookmakers' favourite: {(np.array(CLASSES)[mk25.argmax(1)] == actual).mean():.1%}")
""")

md("""
**Where it goes right and wrong.** Rows are what happened; columns are what the model picked.
""")

code("""
pd.crosstab(pd.Series(actual, name="Actual"), pd.Series(pick, name="Predicted")).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
""")

md("""
As expected, the model almost never picks a draw. It gets most home wins right, about half the away wins, and every draw wrong. That's the same pattern as the bookmakers.
""")

md("""
## 6. One season can be lucky or unlucky, so test seven

Accuracy swings by 5–8 points from season to season. So the same test is repeated for every season from 2019/20 to 2025/26, each time training only on earlier seasons. That's 2,660 matches in total.
""")

code("""
bt = pd.read_csv(ROOT / "reports/backtest_by_season.csv")
fig, ax = plt.subplots()
x = range(len(bt))
for colname, label, col in [("bookmakers", "Bookmakers", ORANGE), ("results_logreg", "Model", BLUE),
                            ("always_home_win", "Always home win", AQUA)]:
    ax.plot(x, bt[colname] * 100, color=col, marker="o", markersize=5, label=label)
ax.set_xticks(list(x), bt.season)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(35, 62)
ax.legend(frameon=False, loc="lower left", ncol=3)
ax.set_title("Accuracy by season, each predicted only from earlier seasons", loc="left", color=INK)
plt.show()
print(bt[["season", "always_home_win", "results_logreg", "bookmakers"]].mean(numeric_only=True).mul(100).round(1).rename("average %"))
""")

md("""
Over 2,660 matches the model averages about **53%**, against 43% for always picking home and 54% for the bookmakers. One standard error on 2,660 matches is about 1 point, so the model is genuinely close to the bookmakers but just below them.
""")

md("""
## 7. Your ideas: head-to-head history and weather

### 7a. Head-to-head
For each match, `src/extra_signals.py` looks up the previous meetings between the same two clubs: the last 6 meetings within 6 years. It summarises them from the home team's point of view: average goal difference, points per game, the result at this same ground, and the draw rate.

Example: Liverpool vs Man City.
""")

code("""
from extra_signals import add_h2h
h2h = add_h2h(df)
pair = h2h[((h2h.HomeTeam == "Liverpool") & (h2h.AwayTeam == "Man City")) | ((h2h.HomeTeam == "Man City") & (h2h.AwayTeam == "Liverpool"))]
pair[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "H2H_n", "H2H_gd", "H2H_pts", "H2H_venue_gd"]].tail(6).round(2)
""")

md("""
**The test:** the v1 model with and without these features, on the same 7 seasons. Same matches, same model, and the only difference is the head-to-head columns. `src/test_extra_signals.py` produces these tables.
""")

code("""
extra = json.load(open(ROOT / "reports/extra_signals.json"))
print(pd.read_csv(ROOT / "reports/h2h_backtest.csv").round(3).to_string(index=False))
h = extra["head_to_head"]
print(f"\\nAccuracy change: {h['acc_change_pts']:+.2f} points (standard error {h['acc_change_se_pts']:.2f})")
print(f"RPS change     : {h['rps_change']:+.4f} (lower is better; standard error {h['rps_change_se']:.4f})")
""")

md("""
**Result: head-to-head adds nothing.** Accuracy is unchanged over 2,660 matches, and the probabilities got very slightly worse.

Why it doesn't help, even though it sounds like it should:
- **Elo and form already contain it.** If Man City usually beat Burnley, that's mostly because Man City are much stronger, and Elo already says so. Head-to-head adds no new information.
- **Tiny samples.** Six meetings over six years cover different squads and managers. That's mostly noise, and a model that leans on noise gets worse.
- The "bogey team" effect people talk about is mostly memory bias: the surprising results are the ones we remember.

### 7b. Weather
The 2023 project collected daily weather for each Premier League city from August 2017 to April 2023. I matched each match to the weather in the home team's city that day.

First, does weather change how matches end at all?
""")

code("""
pd.read_csv(ROOT / "reports/weather_descriptive.csv")
""")

md("""
Weather has small effects, but they're about *how many goals*, not *who wins*:
- Rain nudges the draw rate up by about 1 point.
- Wind lowers goals slightly.
- The home-win rate barely moves.

Weather affects both teams, so it rarely changes which one is better.

**The test:** take the model's probabilities, then let a second small model adjust them with and without weather. Both are trained only on earlier seasons and tested on 2019/20–2022/23.
""")

code("""
print(pd.read_csv(ROOT / "reports/weather_backtest.csv").round(3).to_string(index=False))
w = extra["weather"]
print(f"\\nAccuracy change: {w['acc_change_pts']:+.2f} points (standard error {w['acc_change_se_pts']:.2f})")
print(f"RPS change     : {w['rps_change']:+.4f} (standard error {w['rps_change_se']:.4f})")
""")

md("""
**Result: weather makes predictions worse.** Accuracy dips slightly, and the probability quality is clearly worse (the RPS change is about 4 standard errors). The model finds patterns in the weather data that don't repeat the next season. That's overfitting.

Weather would matter more if you were predicting *the number of goals* rather than win/draw/loss. It's also limited by data: only 6 seasons are available, and live use would need a weather forecast feed.
""")

md("""
### 7c. Team news: confirmed starting lineups

Injuries, suspensions and rotation all show up the same way: a player who normally starts isn't in today's XI. Lineups are announced about an hour before kickoff, so they're legitimate pre-match information.

The data is the Transfermarkt dataset (`data/raw/transfermarkt/`): the starting XI for every Premier League match from 2013/14 to 2025/26, plus each player's market value over time. `src/lineups.py` builds four features per team, using only information from before the match:

| Feature | Meaning |
|---|---|
| **XI value** | Total market value of today's starting XI. Each player is valued at his latest Transfermarkt valuation *before* the match. |
| **XI vs usual** | Today's XI value compared with the team's average over its previous 10 league matches. Negative means a weakened team. |
| **Regulars out** | How many of the team's 11 most frequent recent starters aren't starting today. |
| **Squad value** | Value of the 11 most valuable players used in the previous 20 matches. Known days in advance, with no lineup needed. |

An example: the matches where a team fielded its most weakened XI compared with usual.
""")

code("""
lu = pd.read_csv(ROOT / "data/processed/lineup_features.csv", parse_dates=["Date"])
lu = lu.merge(matches[["Date", "HomeTeam", "AwayTeam", "Result", "FTHG", "FTAG"]], on=["Date", "HomeTeam", "AwayTeam"])
weakest = lu.assign(home_xi_vs_usual_pct=lambda d: (np.exp(d.H_xi_vs_usual) - 1) * 100).nsmallest(5, "home_xi_vs_usual_pct")
weakest[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "home_xi_vs_usual_pct", "H_regulars_out"]].round(1)
""")

md("""
**The test** (`src/test_lineups.py`): lineups only exist from 2013/14, so every variant is trained on the same matches (2013/14 onwards, earlier seasons only) and scored on the same 2,660 matches from 2019/20 to 2025/26.
""")

code("""
lsum = pd.read_csv(ROOT / "reports/lineup_summary.csv")
lsum.assign(accuracy=lambda d: (d.accuracy * 100).round(1))
""")

md("""
**Result: this is the first new input that measurably helps, but only a little.**
- **Probability quality improves clearly.** RPS drops by about 0.0017, which is roughly 2.4 standard errors, so it's not noise. The model now knows when a team is weakened.
- **Accuracy rises about 0.2 points**, which is within noise. Lineup news usually changes *how likely* the favourite is to win, not *which team* is the favourite.
- **Squad value on its own adds nothing.** Elo already knows how strong a club is. The gain comes from the lineup itself: who is actually playing today.
- **Gradient boosting overfits** these features. Logistic regression stays the better model.
- **The bookmakers are still ahead** (RPS 0.198 against 0.202). They also know the lineups, and they see injury news days earlier, plus fitness, tactics and betting flows.

**Using it live** would mean collecting the confirmed lineups about an hour before each kickoff, for example with an Apify scraper. The historical dataset stops in May 2026, so it can't produce lineup features for the current season.
""")

md("""
### 7d. Seven more signals

Every idea from the brainstorm was tested against the lineup model, the best model so far. All of them use only information available before kickoff (`src/more_signals.py`, `src/odds_movement.py`; the test is `src/test_more_signals.py`):

| # | Signal | How it's measured |
|---|---|---|
| 1 | Fixture congestion | Rest days and games in the last 14 days across *all* competitions; a European game in the 4 days before or after |
| 2 | What's at stake | League table on the day: points gap to 4th place and to safety; "nothing to play for" and "relegation fight" flags for the last 10 rounds |
| 3 | Manager change | League games since the current manager took over |
| 4 | Expected goals (xG) | Rolling Understat xG for and against (2019/20 to April 2025 only) |
| 5 | Player ratings | Each player's on-pitch goal difference per 90 minutes (plus-minus), averaged over the starting XI |
| 6 | Unavailable players | Regulars suspended (red cards, yellow-card bans), missing from the whole matchday squad, or only on the bench |
| 7 | Odds movement | How the bookmakers' odds moved between a few days before and kickoff (2019/20 onwards) |

**How to read the table:**
- `rps_change` below 0 means better probabilities; above 0 means worse.
- `rps_change_in_se` is that change measured in standard errors.
- With this many tests, one can look good by luck, so only a clear improvement beyond −2 counts.
""")

code("""
mt = pd.read_csv(ROOT / "reports/more_signals_test.csv")
cols = ["signal", "matches", "accuracy", "acc_change_pts", "rps", "rps_change", "rps_change_in_se"]
mt[[c for c in cols if c in mt.columns]].assign(accuracy=lambda d: (d.accuracy * 100).round(1)).round(4)
""")

md("""
**What the results say**
- **Congestion, stakes and unavailable players** made the probabilities slightly *worse*. The model already sees most of this: the lineup features show who is rested or missing, and Elo and form show how strong the team is. The extra columns mostly add noise.
- **Manager change and player ratings** made no difference. Plus-minus ratings mostly measure the team a player is in, which Elo already knows.
- **All five combined** is clearly worse, by about 2 standard errors. That's overfitting: more columns, same information.
- **xG:** six xG features didn't help. A single season-long xG difference *did* (about 2.5 standard errors). That fits published research: xG measures team strength better than goals do. But it was a second attempt, so treat it as promising rather than proven. It needs a longer xG history to confirm.
- **Odds movement** captures late news and informed money. If its rows are in the table above, compare them with the bookmakers' closing odds. If they aren't, the Football-Data files haven't been added yet.

The pattern is the same as everything before: **the remaining edge is information, not more columns.** Lineups helped because they are new information. Most of the other signals are already reflected in Elo, form and the lineups.
""")

md("""
### 7e. Trying to beat the lineup model: the final model

`src/improve.py`, `src/improve2.py` and `src/pooled_lineups.py` test everything left that could plausibly help: longer form windows, ClubElo and pi-ratings, Poisson expected goals, non-linear Elo terms, league home advantage, regularisation, recency weighting, an ordinal model, gradient boosting, more detailed lineup features (who is missing, by line; goalkeeper change; captain out; cohesion; bench value; age), recalibration, training on five leagues, and ensembles.

Because so many variants are tried, the protocol is stricter: the seven backtest seasons are split into a **development half** (2019/20–2022/23) used to choose, and a **confirmation half** (2023/24–2025/26) only looked at afterwards. A change that only looks good on the development half is luck.
""")

code("""
r1 = pd.read_csv(ROOT / "reports/improve_results.csv")
r2 = pd.read_csv(ROOT / "reports/improve2_results.csv")
r3 = pd.read_csv(ROOT / "reports/pooled_lineups.csv")
res = pd.concat([r1[r1.group == "A. football only"], r2, r3]).drop_duplicates(["variant", "split"])
wide = res.pivot(index="variant", columns="split", values="rps_change_in_se")[["dev", "confirm"]].round(2)
wide["rps_all"] = res[res.split == "all"].set_index("variant").rps.round(4)
wide["accuracy_all"] = (100 * res[res.split == "all"].set_index("variant").accuracy).round(1)
wide = wide.loc[res.variant.drop_duplicates()]   # keep the order they were run in
wide.columns = ["dev (SE)", "confirm (SE)", "RPS, all", "accuracy %, all"]
wide
""")

md("""
**What held up.** Only the ensembles improve on the confirmation half (about −3.3 standard errors). The final model is the **equal-weight average of five models**: the lineup model, an ordinal logit on the same features, the lineup model with long-window form, the lineup model with Poisson expected goals, and a shallow gradient boosting on all features. Each one alone is within noise of the lineup model; averaging helps because their errors differ. Weights fitted on the development half came out close to equal, so nothing is fitted.

Everything else failed the same test, including the richer lineup detail, recalibration and five-league training. The market-aware check (`improve.py`, group B) shows the model adds nothing to the bookmakers' odds.
""")

code("""
fb = pd.read_csv(ROOT / "reports/final_backtest_by_season.csv")
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(fb))
for i, (c, lab, col) in enumerate([("lineup_model", "lineup model", "#2a78d6"), ("ensemble", "final ensemble", "#eb6834"), ("bookmakers", "bookmakers", "#1baf7a")]):
    ax.bar(x + (i - 1) * 0.27, 100 * fb[c], 0.27, label=lab, color=col)
ax.set_xticks(x, fb.season); ax.set_ylim(40, 65); ax.set_ylabel("accuracy, %"); ax.legend(frameon=False)
ax.set_title("Final ensemble vs lineup model and bookmakers, by season")
plt.show()
fb.assign(**{c: (100 * fb[c]).round(1) for c in ["lineup_model", "ensemble", "bookmakers"]})
""")

md("""
## 8. Everything else that was tried

`src/experiments.py` tested 48 combinations on the same 7-season backtest:
- **Extra training data:** the Championship and the other top-5 leagues, up to 59,800 matches.
- **ClubElo ratings:** these compare clubs across leagues through European games, which helps rate promoted teams.
- **Pi-ratings:** separate home and away strength for each team.
- **Shots-on-target ratings:** a rough substitute for expected goals.
- **Models:** logistic regression, gradient boosting, and the average of the two.

**RPS** (ranked probability score) measures the quality of the *probabilities*, not just the pick. It treats predicting an away win when the home team won as worse than predicting a draw. Lower is better.
""")

code("""
exp = pd.read_csv(ROOT / "reports/experiments.csv")
print("Best 8 configurations by RPS, plus the bookmaker reference:")
exp.head(9).assign(accuracy=lambda d: (d.accuracy * 100).round(1))
""")

code("""
models = exp[exp.features != "bookmakers"]
print(f"Accuracy range across all {len(models)} configurations: {models.accuracy.min():.1%} to {models.accuracy.max():.1%}")
""")

md("""
Every configuration lands within about one standard error (±1 point) of the others. More data and fancier ratings make the probabilities *slightly* sharper, but they don't change which result gets picked, so accuracy doesn't move. The data simply has no more signal in it.
""")

md("""
## 9. When to trust the model

Accuracy over all matches hides something useful. The model knows when it's unsure. Grouping the 2,660 backtest predictions by how confident the model was (`src/evaluate_v2.py`):
""")

code("""
conf = pd.read_csv(ROOT / "reports/v2_accuracy_by_confidence.csv")
fig, ax = plt.subplots(figsize=(7, 3.5))
bars = ax.bar(conf.confidence, conf.accuracy * 100, color=BLUE, width=0.55)
for b, (acc, share) in zip(bars, zip(conf.accuracy, conf.share)):
    ax.annotate(f"{acc:.0%}\\n({share:.0%} of matches)", (b.get_x() + b.get_width() / 2, b.get_height()),
                xytext=(0, 4), textcoords="offset points", ha="center", color=INK, fontsize=9)
ax.set_ylim(0, 90)
ax.set_xlabel("Model's probability for its pick")
ax.set_ylabel("Accuracy (%)")
ax.set_title("The more confident the model, the more often it's right", loc="left", color=INK)
plt.show()
conf
""")

md("""
When the model gives its pick at least a 65% chance (about 1 match in 5), it's right about **73%** of the time. When it's below 45%, it's barely better than guessing. For a demo, *"here's the pick, and here's how much to trust it"* is far more useful than one accuracy number.
""")

md("""
## 10. This season and this weekend

The 2026/27 season so far: 50 matches, never seen during training or model selection.
""")

code("""
t = pd.read_csv(ROOT / "reports/test_2026_27_predictions.csv")
print(f"Correct: {t.Correct.sum()} of {len(t)} ({t.Correct.mean():.0%}).  Home wins so far this season: {(t.Result == 'H').mean():.0%} (usual: ~45%)")
t[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "pH", "pD", "pA", "Pick", "Correct"]].tail(10)
""")

md("""
50 matches is too few to judge a model; accuracy over 50 matches easily swings ±7 points. This season has also started with unusually few home wins, which hurts every model that knows about home advantage. The 7-season backtest is the number to trust.

**Next matchday** (`src/predict_next.py`, trained on every match played so far):
""")

code("""
pd.read_csv(ROOT / "reports/next_matchday_predictions.csv")
""")

md("""
## 11. Conclusions

1. **Honest accuracy is about 53%**: 52.7% from results only, 53.4% for the final model (results + lineups, ensemble of five), against 43% for always picking home and 54% for the bookmakers. The 82% in 2023 came from leakage.
2. **Each team's recent results are the core of the model**, through Elo and form.
3. **Head-to-head history and weather were tested properly and don't help.** Elo already covers head-to-head, and weather affects goals, not winners.
4. **More data and more complex models didn't help either.** Longer form, other ratings, tuning, recency weighting, other model families, recalibration and five-league training all failed a development/confirmation split. Only averaging five model variants held up, and it is a small gain (RPS 0.2022 → 0.2013). The limit is the information, not the algorithm.
5. **Of everything tested, only team news (lineups) clearly helped, with xG as a promising second.** Congestion, stakes, manager changes, player ratings and suspensions added nothing beyond what the model already knew.
6. **Team news, in more detail:** Confirmed lineups (who is missing compared with usual) made the probabilities clearly better, though accuracy barely moved. Using them live needs the lineups collected about an hour before kickoff, for example with an Apify scraper. The bookmakers are still slightly ahead because they know the same news earlier, and more of it.
7. **Report confidence, not just a pick.** At 65%+ confidence (22% of matches) the final model is right 73% of the time; at 70%+ (15% of matches), 75%.
""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "PL_Predictions_walkthrough.ipynb")
print("written")
