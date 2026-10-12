# Premier League Predictions

Predicts home win / draw / away win for Premier League matches, using only information available before kickoff, and measures itself honestly against simple baselines and the bookmakers.

This is a rebuild of my 2023 bootcamp project [Mid_project_PL](https://github.com/jcosta92/Mid_project_PL). That project reported 82% accuracy. That figure was wrong: the model was effectively being shown the answer. This repo finds the bugs, fixes them, and reports the real number.

**Start with [`notebooks/PL_Predictions_walkthrough.ipynb`](notebooks/PL_Predictions_walkthrough.ipynb)**, which walks through the whole process step by step. `site/index.html` is a self-contained page with this matchday's predictions next to the bookmakers and the season's running record; regenerate it weekly with the commands under Run.

## Results

| | Accuracy | Notes |
|---|---|---|
| Always pick the home team | 43.4% | 2019/20–2025/26 backtest, 2,660 matches |
| This model, results only (v1) | 52.7% | same backtest, each season predicted using only earlier seasons |
| **Final model: results + confirmed lineups, ensemble of five** | **53.4%** | same backtest; probability score (RPS) 0.2013 vs 0.2039 for v1 |
| Bookmakers' favourite | 54.4% | same backtest; RPS 0.1981 |
| **This model, 2026/27 so far** | **46% (23/50)** | unseen test season; only 36% of these matches were home wins |
| Bookmakers, 2026/27 so far | 55% | odds only available for the first 20 matches |

Accuracy per season varies by about 5–8 points, so 50 matches on their own say little. The seven-season backtest is the more reliable figure.

The results-only model gets within about 1.7 points of the bookmakers. Confirmed lineups and an ensemble of five model variants close about a quarter of the remaining gap in probability quality (RPS 0.2039 → 0.2013; the bookmakers are at 0.1981). Everything else that was tried, listed below, made no measurable difference. Adding bookmaker odds as a feature did not improve on the odds alone.

The 2026/27 test figure is for the results-only model: the lineup data ends in May 2026, so the final model cannot be scored on the current season yet.

## What was wrong in 2023, and what changed

`src/leakage_demo.py` reproduces the old setup on the same seasons (2017/18–2022/23):

| Rolling averages | Scored on training rows | Random 20% split | Unseen next season |
|---|---|---|---|
| Include the current match (2023 bug) | 92.5% | 62.3% | 65.2% |
| Previous matches only (fixed) | 89.0% | 45.0% | 49.6% |

1. **The form averages included the match being predicted.** For example, "average goals in the last 5 matches" counted that match's own goals, so the final score leaked into the inputs. Fixed with `shift(1)` in `src/features.py`, and checked against a hand calculation.
2. **Accuracy was measured on the training data.** A random forest memorises its training rows. 89% on training rows against 50% on new matches is memorisation, not skill. Now every number comes from matches the model has never seen.
3. **Random splits replaced by time-based splits.** The model always trains on the past and predicts the future, as it would in real use.
4. **The "85% of bets won" figure counted home-win-or-draw as one prediction**, which covers two of the three outcomes. It has been removed.
5. **Matchday-only inputs dropped.** Attendance and starting captain are only known on the day.

## How it works

- `src/build_dataset.py`: merges Football-Data.co.uk results, shots and odds (2000/01 to Aug 2026) with openfootball results for the rest of 2026/27. The 20 matches both sources share have identical scores.
- `src/features.py`: pre-match features only.
  - Elo ratings, updated after every match, weighted by goal difference, with ratings shrinking toward the mean between seasons and promoted teams starting below average.
  - Rolling 5- and 10-match goals, points and shots for and against.
  - Rest days.
  - Form resets when a team returns after relegation.
  - One spelling per club (Football-Data writes Nottingham Forest two ways), and the COVID-delayed 2019/20 season is labelled correctly (it ran into July 2020).
- `src/poisson.py`: a time-weighted Poisson team-strength model, refitted every week. It was tested, but adding it didn't improve accuracy, so it isn't in the final model.
- `src/evaluate.py`: tunes Elo settings and picks the model on 2025/26 only, then runs the seven-season backtest and the 2026/27 test. It reports accuracy, log loss and RPS (ranked probability score, the standard football forecasting metric).
- `src/predict_next.py`: retrains on every played match and predicts the next matchday.

The final model is a regularised multinomial logistic regression. Gradient boosting and shot-based features were tried and were no better.

## Version 2 experiments: can more data or better models beat 53%?

`src/experiments.py` tests 48 combinations on the same seven-season backtest (2,660 matches). Every season is predicted using only earlier seasons.

**More data**
- **ClubElo ratings.** These are calibrated across leagues through cup and European matches, so newly promoted teams arrive with a proper rating.
- **Extra training matches** from the Championship and the other top-5 leagues: up to 59,800 matches, against 9,800 for the Premier League alone.

**Stronger ratings**
- **Pi-ratings** (Constantinou & Fenton, 2013): separate home and away strength for each team.
- **A shots-on-target pi-rating**, as a rough stand-in for expected goals.
- **Exponentially weighted form.**

**Models**
- Logistic regression, gradient boosting, and an average of the two.
- A rule for predicting draws.

| | Accuracy | RPS (lower is better) |
|---|---|---|
| v1 (this repo's main model) | 52.7% | |
| Best v2 (PL + Championship, all v2 features, LR + GBM average) | 52.9% | 0.2024 |
| Range across all 48 v2 configurations | 51.9% – 52.9% | 0.2024 – 0.2078 |
| Bookmakers | 54.4% | 0.1981 |
| v2 on 2026/27 so far (50 matches) | 42% | 0.2050 |

**Conclusion.** None of it moved accuracy. The extra data and ratings made the probabilities slightly sharper on the backtest (better RPS), but every accuracy difference is within noise. For 2,660 matches, one standard error is about 1 point. The draw rule changed accuracy by 0.03 points. A model built on results, shots and ratings tops out around 53%. The v1 model stays the main one: it's simpler and just as accurate.

The remaining gap to the bookmakers is information this data doesn't contain: team news, injuries, rotation and lineups.

**Where the model is useful:** accuracy depends heavily on how confident the model is (`reports/v2_accuracy_by_confidence.csv`):

| Model confidence | Share of matches | Accuracy |
|---|---|---|
| under 45% | 33% | 39% |
| 45–55% | 26% | 50% |
| 55–65% | 20% | 58% |
| 65% or more | 21% | **73%** |

Run: `python build_multileague.py && python experiments.py && python evaluate_v2.py`

## Head-to-head history and weather

`src/extra_signals.py` builds both signals and `src/test_extra_signals.py` tests them. Both are leakage-free.

- **Head-to-head:** the last 6 meetings between the two clubs within 6 years (goal difference, points, result at the same ground, draw rate). Added to the v1 model on the 7-season backtest, it changed accuracy by **0.0 points** (standard error 0.2), and the probabilities got slightly worse. Elo and form already capture "this team usually beats that one", and six meetings across different squads are mostly noise.
- **Weather:** daily weather in the home team's city, Aug 2017 to Apr 2023, collected for the 2023 project. Rain raises the draw rate by about 1 point and wind lowers goals slightly. Neither changes who wins, because weather affects both teams. Used as an adjustment to the v1 probabilities, it changed accuracy by **−0.3 points** and made the probabilities clearly worse, about 4 standard errors, because the model was fitting noise.

## Team news: confirmed lineups

`src/lineups.py` turns Transfermarkt starting lineups and player market values into team-news features for every match from 2013/14 to 2025/26. Each player is valued only at valuations dated before the match.
- **XI value:** total value of today's starting XI.
- **XI vs usual:** today's XI value compared with the team's last 10 matches.
- **Regulars out:** how many usual starters are missing.
- **Squad value:** known days in advance.

Lineups are announced about an hour before kickoff.

`src/test_lineups.py` trains every variant on the same matches and scores them on the 2019/20–2025/26 backtest (2,660 matches):

| | Accuracy | RPS | RPS change vs v1 features |
|---|---|---|---|
| v1 features (trained 2013/14+) | 52.8% | 0.2039 | |
| + squad value (no lineup) | 52.8% | 0.2042 | +0.0002 (se 0.0004) |
| **+ squad value + confirmed lineup** | **53.0%** | **0.2022** | **−0.0017 (se 0.0007)** |
| same, gradient boosting | 52.6% | 0.2085 | +0.0046 |
| Bookmakers | 54.4% | 0.1981 | |

**Conclusion.** Lineups are the only input in this project that measurably improves the predictions. Probability quality improves by about 2.4 standard errors. Accuracy barely moves, because team news mostly changes *how likely* the favourite is to win rather than *which team* is the favourite. The bookmakers keep an edge because they get injury news earlier.

Using this live needs the confirmed XI collected before each kickoff, for example with an Apify scraper. The historical dataset ends in May 2026.

The raw Transfermarkt files are too large for GitHub (`game_lineups` is 126 MB). Download them from [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) (CC0) into `data/raw/transfermarkt/`. The processed `data/processed/lineup_features.csv` is included.

## Seven more signals

`src/more_signals.py` builds seven further candidate signals and `src/test_more_signals.py` tests each one against the lineup model above. Every signal uses only information available before kickoff.

1. **Fixture congestion:** rest days counting all competitions, games in the last 14 days, and European matches just before or after.
2. **What's at stake:** points gap to the top 4 and to safety, "nothing to play for", and relegation fights.
3. **Manager change:** new manager, and games since the manager arrived.
4. **Expected goals (xG):** rolling xG, xGA and non-penalty xG over 6 and 19 matches, from Understat. The data runs from 2019/20 to early April 2025 only.
5. **Player ratings:** plus-minus ratings for each player, shrunk towards zero, summed over today's XI.
6. **Unavailable players:** regulars suspended (derived from the card rules), regulars not in the squad, and regulars benched.
7. **Odds movement:** change between the opening and closing market prices.

Signals 1, 2, 3, 5 and 6 cover the full 2019/20–2025/26 backtest (2,660 matches). xG and odds movement only exist from 2019/20. For those, a second-stage model is fitted on earlier seasons of that era, and it is compared with the lineup model's own probabilities on the same matches.

The table compares each signal with the lineup model on the same matches. A negative RPS change is better. "SE" is the change divided by its standard error.

| Signal | Matches | Accuracy | RPS | RPS change | SE |
|---|---|---|---|---|---|
| Lineup model (base) | 2,660 | 53.0% | 0.2022 | | |
| 1. Fixture congestion | 2,660 | 53.1% | 0.2028 | +0.0005 | +1.7 |
| 2. What's at stake | 2,660 | 53.0% | 0.2027 | +0.0005 | +1.7 |
| 3. Manager change | 2,660 | 53.4% | 0.2022 | −0.0001 | −0.2 |
| 5. Player ratings | 2,660 | 53.0% | 0.2023 | +0.0001 | +0.3 |
| 6. Unavailable players | 2,660 | 52.9% | 0.2028 | +0.0006 | +1.6 |
| All five combined | 2,660 | 52.9% | 0.2037 | +0.0015 | +2.1 |
| 4a. xG, six features (2020/21 to Apr 2025) | 1,795 | 53.6% (base 54.1%) | 0.2038 | +0.0008 | +0.7 |
| 4b. xG, single 19-match xG difference | 1,795 | 54.3% (base 54.1%) | 0.2012 | −0.0018 | −2.5 |
| Bookmakers, same 1,795 matches | 1,795 | 55.5% | 0.1973 | | |
| 7. Odds movement | | | | | not run |

**Conclusions**

- **Signals 1, 2, 3, 5 and 6 add nothing.** Each one is within noise, and most make the probabilities slightly worse. Combining all five makes them clearly worse (+2.1 SE), which is overfitting. The manager-change accuracy gain (+0.4 points, se 0.28) is not significant.
- **Most of these are already priced in.** Elo and form already capture the effect of a new manager or a run of fixtures. The lineup already shows which players are missing, so the reason they are missing adds nothing.
- **xG is the only candidate.** A single 19-match xG difference improves RPS by 2.5 standard errors and closes about a third of the gap to the bookmakers. It was the *second* xG specification tried, after the six-feature version failed, and 8 signals were tested. So it is promising but not proven. It needs a season the model has not seen yet. Understat data for 2025/26 onwards is required to check it.
- **The suspension feature is approximate.** Card counts give the right ban in most cases, but 14% of the players flagged as suspended still made the squad (appeals and rescinded cards).
- **Odds movement needs the full Football-Data season files** (`E0.csv` per season, with the `AvgH`/`AvgCH` columns). The site is not reachable from the environment this was built in. Save them as `data/raw/football_data_full/E0_1920.csv` … `E0_2526.csv` and rerun `python test_more_signals.py`; the test, and the bookmakers' opening and closing reference rows, run automatically. Movement is a market signal: it can only be used close to kickoff, and it measures how much late information the bookmakers had rather than adding football knowledge.

## Trying to beat the lineup model

`src/improve.py`, `src/improve2.py` and `src/pooled_lineups.py` try everything left that could plausibly improve on the lineup model. Because many variants are tried, the protocol is stricter than before: the seven backtest seasons are split into a **development half** (2019/20–2022/23, 1,520 matches), used to choose, and a **confirmation half** (2023/24–2025/26, 1,140 matches), looked at only afterwards. A change that only looks good on the development half is luck.

Every variant is compared with the lineup model on the same matches. A negative RPS change is better; "SE" is the change in standard errors.

| Variant | Dev SE | Confirm SE | RPS, all | Accuracy, all |
|---|---|---|---|---|
| Lineup model (base) | | | 0.2022 | 53.0% |
| + long-window form (19 games, EWM, shots on target) | −0.2 | −1.9 | 0.2015 | 53.0% |
| + ClubElo and pi-ratings | −0.3 | +0.4 | 0.2022 | 52.9% |
| + Poisson expected goals | −0.3 | −0.6 | 0.2021 | 53.2% |
| + non-linear Elo terms | +1.5 | −1.2 | 0.2023 | 52.7% |
| + league home advantage, no-crowd flag | +1.4 | +2.0 | 0.2030 | 53.2% |
| + all of the above | +0.8 | −0.3 | 0.2025 | 52.7% |
| Regularisation C = 0.02 … 1.0 (8 variants) | −0.8 … +1.2 | −1.0 … +1.6 | 0.2019–0.2030 | |
| Recency weighting, half-life 2/4/8 seasons (6 variants) | +1.0 … +3.0 | −1.1 … +0.0 | 0.2022–0.2038 | |
| Ordinal (proportional-odds) logit | −0.8 | −1.4 | 0.2019 | 53.2% |
| Gradient boosting, shallow and regularised | +2.3 | +1.7 | 0.2045 | 53.0% |
| Detailed lineup features: XI value by line, star player out, GK change, captain out, cohesion, bench value, age (6 variants) | −0.9 … +1.2 | −1.2 … +3.4 | 0.2015–0.2034 | |
| Recalibrated lineup model (intercepts and temperature) | +2.0 | +0.1 | 0.2036 | 52.6% |
| Lineup model trained on five leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1) | −1.8 | −0.3 | 0.2015 | 53.2% |
| Stacked ensemble, weights fitted on the development half | −1.7 | **−3.4** | 0.2011 | 53.4% |
| **Equal-weight ensemble of five models** | −1.7 | **−3.3** | **0.2013** | **53.4%** |

The five models in the ensemble: the lineup model, the ordinal logit on the same features, the lineup model with long-window form, the lineup model with Poisson expected goals, and a shallow gradient boosting on all features. Each on its own is within noise of the lineup model; averaging them is what helps, because their errors differ. The weights fitted on the development half were close to equal, so the final model uses equal weights (nothing fitted).

**Market-aware check.** With the bookmakers' odds as an input, the model does not beat the odds alone (+0.3 SE on all matches), and the final ensemble is still behind the bookmakers by 3.3 SE. The model has no information the market lacks.

**Conclusions**

- The final model is the equal-weight ensemble: 53.4% and RPS 0.2013 over 2,660 matches, against 52.7% and 0.2039 for results only, and 54.4% and 0.1981 for the bookmakers.
- Nothing else held up on the confirmation half. More features, other model families, tuning, recalibration, richer lineup detail and five-league training all failed the same test. This is a data limit, not a modelling one.
- At 65% or higher confidence (22% of matches) the ensemble is right 73% of the time; at 70% or higher (15% of matches), 75%.

## Web app

`streamlit run app.py` opens the model as a web app: this matchday's predictions, any fixture you pick (predicted with the current ratings and form), the season record against the bookmakers, and the backtest. To host it, sign in at share.streamlit.io with GitHub, *Create app*, repo `jcosta92/PL_Predictions`, branch `main`, main file `app.py`. Every push to `main` redeploys.

## Run

```bash
pip install -r requirements.txt
cd src
python build_dataset.py
python evaluate.py      # writes reports/metrics.json, backtest and test predictions
python leakage_demo.py
python predict_next.py  # writes reports/next_matchday_predictions.csv
python test_extra_signals.py
python build_multileague.py && python evaluate_v2.py && python experiments.py   # v2 experiments
python lineups.py && python test_lineups.py   # needs the Transfermarkt files
python more_signals.py && python test_more_signals.py   # needs the Transfermarkt files
python improve.py && python lineups_v2.py && python improve2.py && python pooled_lineups.py   # final model
python build_page.py   # writes site/index.html: this matchday vs the bookmakers, season record
```

The notebook (`notebooks/`) needs `jupyter` and `matplotlib` as well.

## Data

- [Club Football Match Data](https://github.com/xgabora/Club-Football-Match-Data-2000-2025) (MIT), sourced from [Football-Data.co.uk](https://www.football-data.co.uk/)
- [openfootball/england](https://github.com/openfootball/england) (CC0)
- [ClubElo](http://clubelo.com/) ratings, included in the Club Football Match Data set
- [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) (CC0): lineups, player market values, appearances, cards and managers
- [Understat](https://understat.com/) team xG, 2019/20 to April 2025, via the [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) repository

For match analysis and learning purposes. Not betting advice.
