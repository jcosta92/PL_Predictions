# Premier League Predictions

Predicts home win / draw / away win for Premier League matches, using only information available before kickoff, and measures itself honestly against simple baselines and the bookmakers.

This is a rebuild of my 2023 bootcamp project [Mid_project_PL](https://github.com/jcosta92/Mid_project_PL). That project reported 82% accuracy. That figure was wrong: the model was effectively being shown the answer. This repo finds the bugs, fixes them, and reports the real number.

**Start with [`notebooks/PL_Predictions_walkthrough.ipynb`](notebooks/PL_Predictions_walkthrough.ipynb)**, which walks through the whole process step by step.

## Results

| | Accuracy | Notes |
|---|---|---|
| Always pick the home team | 43.4% | 2019/20–2025/26 backtest, 2,660 matches |
| **This model** | **52.7%** | same backtest, each season predicted using only earlier seasons |
| Bookmakers' favourite | 54.4% | same backtest |
| **This model, 2026/27 so far** | **46% (23/50)** | unseen test season; only 36% of these matches were home wins |
| Bookmakers, 2026/27 so far | 55% | odds only available for the first 20 matches |

Accuracy per season varies by about 5–8 points, so 50 matches on their own say little. The seven-season backtest is the more reliable figure.

The model gets within about 1.7 points of the bookmakers using only past results. Closing that gap needs information the betting market has and this data doesn't: team news, injuries, lineups and expected goals. Adding bookmaker odds as a feature did not improve on the odds alone.

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

## Run

```bash
pip install -r requirements.txt
cd src
python build_dataset.py
python evaluate.py      # writes reports/metrics.json, backtest and test predictions
python leakage_demo.py
python predict_next.py  # writes reports/next_matchday_predictions.csv
python test_extra_signals.py
```

The notebook (`notebooks/`) needs `jupyter` and `matplotlib` as well.

## Data

- [Club Football Match Data](https://github.com/xgabora/Club-Football-Match-Data-2000-2025) (MIT), sourced from [Football-Data.co.uk](https://www.football-data.co.uk/)
- [openfootball/england](https://github.com/openfootball/england) (CC0)
- [ClubElo](http://clubelo.com/) ratings, included in the Club Football Match Data set

For match analysis and learning purposes. Not betting advice.
