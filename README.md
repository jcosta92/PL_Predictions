# Premier League Predictions

Predicts home win / draw / away win for Premier League matches, using only information available before kickoff, and measures itself honestly against simple baselines and the bookmakers.

This is a rebuild of my 2023 bootcamp project [Mid_project_PL](https://github.com/jcosta92/Mid_project_PL). That project reported 82% accuracy. That figure was wrong: the model was effectively being shown the answer. This repo finds the bugs, fixes them, and reports the real number.

## Results

| | Accuracy | Notes |
|---|---|---|
| Always pick the home team | 43.5% | 2019/20–2025/26 backtest, 2,660 matches |
| **This model** | **52.9%** | same backtest, each season predicted using only earlier seasons |
| Bookmakers' favourite | 54.4% | same backtest |
| **This model, 2026/27 so far** | **48% (24/50)** | unseen test season; only 36% of these matches were home wins |
| Bookmakers, 2026/27 so far | 55% | odds only available for the first 20 matches |

Accuracy per season varies by about 5–8 points, so 50 matches on their own say little. The seven-season backtest is the more reliable figure.

The model gets within about 1.5 points of the bookmakers using only past results. Closing that gap needs information the betting market has and this data doesn't: team news, injuries, lineups and expected goals. Adding bookmaker odds as a feature did not improve on the odds alone (53.9% vs 54.4%).

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
- `src/poisson.py`: a time-weighted Poisson team-strength model, refitted every week. It was tested, but adding it didn't improve accuracy, so it isn't in the final model.
- `src/evaluate.py`: tunes Elo settings and picks the model on 2025/26 only, then runs the seven-season backtest and the 2026/27 test. It reports accuracy, log loss and RPS (ranked probability score, the standard football forecasting metric).
- `src/predict_next.py`: retrains on every played match and predicts the next matchday.

The final model is a regularised multinomial logistic regression. Gradient boosting and shot-based features were tried and were no better.

## Run

```bash
pip install -r requirements.txt
cd src
python build_dataset.py
python evaluate.py      # writes reports/metrics.json, backtest and test predictions
python leakage_demo.py
python predict_next.py  # writes reports/next_matchday_predictions.csv
```

## Data

- [Club Football Match Data](https://github.com/xgabora/Club-Football-Match-Data-2000-2025) (MIT), sourced from [Football-Data.co.uk](https://www.football-data.co.uk/)
- [openfootball/england](https://github.com/openfootball/england) (CC0)

For match analysis and learning purposes. Not betting advice.
