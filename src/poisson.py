"""Time-weighted Poisson team-strength model (a light Dixon-Coles style model).

For every match date D, fit on matches strictly before D (last 2 years, recent
games weighted more):
    log(goals) = intercept + home_advantage + attack[team] - defence[opponent]
Then turn the two expected goal counts into P(home win), P(draw), P(away win)
with a Poisson score grid. Fitting only on past dates keeps it leakage-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

WINDOW_DAYS = 730
HALF_LIFE_DAYS = 180
MAX_GOALS = 10


def _design(home, away, teams):
    """Rows alternate: home team's goals, then away team's goals, for each match."""
    idx = {t: i for i, t in enumerate(teams)}
    hi = np.array([idx[t] for t in home]); ai = np.array([idx[t] for t in away])
    n, k = len(hi), len(teams)
    X = np.zeros((2 * n, 1 + 2 * k))
    r = np.arange(n)
    X[2 * r, 0] = 1                    # home advantage
    X[2 * r, 1 + hi] = 1               # home attack
    X[2 * r, 1 + k + ai] = 1           # away defence
    X[2 * r + 1, 1 + ai] = 1           # away attack
    X[2 * r + 1, 1 + k + hi] = 1       # home defence
    return X


def _outcome_probs(lh, la):
    g = np.arange(MAX_GOALS + 1)
    grid = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    return np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()


def add_poisson(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    out = np.full((len(df), 5), np.nan)
    # Refit once per calendar week, using only matches before that week starts.
    week = df["Date"].dt.to_period("W-MON").dt.start_time
    for date in np.sort(week.unique()):
        hist = df[(df.Date < date) & (df.Date >= date - pd.Timedelta(days=WINDOW_DAYS))]
        today = np.flatnonzero((week == date).to_numpy())
        if len(hist) < 200:
            continue
        teams = sorted(set(hist.HomeTeam) | set(hist.AwayTeam)
                       | set(df.HomeTeam.iloc[today]) | set(df.AwayTeam.iloc[today]))
        X = _design(hist.HomeTeam, hist.AwayTeam, teams)
        y = np.column_stack([hist.FTHG, hist.FTAG]).ravel()
        age = (date - hist.Date).dt.days.to_numpy()
        w = np.repeat(0.5 ** (age / HALF_LIFE_DAYS), 2)
        model = PoissonRegressor(alpha=1e-3, max_iter=300).fit(X, y, sample_weight=w)
        Xt = _design(df.HomeTeam.iloc[today], df.AwayTeam.iloc[today], teams)
        mu = model.predict(Xt).reshape(-1, 2)
        for row, (lh, la) in zip(today, mu):
            out[row] = (lh, la, *_outcome_probs(lh, la))
    df[["PoisXGH", "PoisXGA", "PoisH", "PoisD", "PoisA"]] = out
    df["PoisXGDiff"] = df.PoisXGH - df.PoisXGA
    return df
