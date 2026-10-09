"""Build the live page: this matchday's predictions next to the bookmakers, and the
running record for the season. Output: site/index.html (self-contained, no server).

Inputs (all produced by the other scripts):
  reports/next_matchday_predictions.csv     predict_next.py
  reports/test_2026_27_predictions.csv      evaluate.py
  data/processed/matches.csv                build_dataset.py (odds for played matches)
  reports/final_backtest_by_season.csv      improve2.py
Optional: data/raw/football_data_full/fixtures.csv (Football-Data, odds for upcoming
fixtures) adds the bookmakers' column for the next matchday.

Weekly refresh:
  copy the latest openfootball 2026-27 file to data/raw/openfootball_pl_2026-27.txt
  python build_dataset.py && python evaluate.py && python predict_next.py && python build_page.py
"""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/jcosta92/PL_Predictions"
LABEL = {"H": "Home win", "D": "Draw", "A": "Away win"}


def market(df: pd.DataFrame) -> np.ndarray:
    inv = 1 / df[["OddH", "OddD", "OddA"]].to_numpy(dtype=float)
    return inv / inv.sum(1, keepdims=True)


def fixture_odds() -> pd.DataFrame | None:
    f = ROOT / "data" / "raw" / "football_data_full" / "fixtures.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, encoding="latin1")
    d = d[d.Div == "E0"].dropna(subset=["HomeTeam", "AwayTeam"])
    d["Date"] = pd.to_datetime(d.Date, dayfirst=True)
    d = d.replace({"HomeTeam": {"Nottm Forest": "Nott'm Forest"}, "AwayTeam": {"Nottm Forest": "Nott'm Forest"}})
    cols = [c for c in ("AvgH", "AvgD", "AvgA") if c in d.columns] or ["B365H", "B365D", "B365A"]
    d = d.rename(columns=dict(zip(cols, ["OddH", "OddD", "OddA"])))
    return d[["Date", "HomeTeam", "AwayTeam", "OddH", "OddD", "OddA"]]


def prob_bar(p: np.ndarray, pick: str) -> str:
    segs = []
    for k, v in zip("HDA", p):
        cls = f"seg seg-{k}" + (" seg-pick" if k == pick else "")
        segs.append(f'<span class="{cls}" style="width:{100 * v:.1f}%" title="{LABEL[k]} {100 * v:.0f}%">'
                    f'{f"{100 * v:.0f}" if v >= 0.12 else ""}</span>')
    return f'<div class="bar">{"".join(segs)}</div>'


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def e(s) -> str:
    return html.escape(str(s))


def main() -> None:
    nxt = pd.read_csv(ROOT / "reports" / "next_matchday_predictions.csv", parse_dates=["Date"])
    played = pd.read_csv(ROOT / "reports" / "test_2026_27_predictions.csv", parse_dates=["Date"])
    m = pd.read_csv(ROOT / "data" / "processed" / "matches.csv", parse_dates=["Date"])
    season = m[m.Season == "2026/27"][["Date", "HomeTeam", "AwayTeam", "OddH", "OddD", "OddA"]]
    played = played.merge(season, on=["Date", "HomeTeam", "AwayTeam"], how="left")
    fo = fixture_odds()
    if fo is not None:
        nxt = nxt.merge(fo, on=["Date", "HomeTeam", "AwayTeam"], how="left")
    bt = pd.read_csv(ROOT / "reports" / "final_backtest_by_season.csv")
    last_played = played.Date.max()

    # ---- this matchday
    rows = []
    for r in nxt.sort_values(["Date", "HomeTeam"]).itertuples(index=False):
        p = np.array([r.pH, r.pD, r.pA])
        bk = "<span class='muted'>no odds yet</span>"
        if "OddH" in nxt.columns and pd.notna(getattr(r, "OddH", np.nan)):
            q = market(pd.DataFrame([r._asdict()]))[0]
            bk = f'{LABEL["HDA"[q.argmax()]]} <span class="muted">{pct(q.max())}</span>'
        rows.append(f"""<tr>
  <td class="date">{r.Date.strftime('%a %d %b')}</td>
  <td class="fix"><span class="team">{e(r.HomeTeam)}</span> <span class="v">v</span> <span class="team">{e(r.AwayTeam)}</span>
      <span class="elo">Elo {r.HomeElo:.0f} · {r.AwayElo:.0f}</span></td>
  <td class="probs">{prob_bar(p, r.Pick)}</td>
  <td class="pick"><span class="pill pill-{r.Pick}">{LABEL[r.Pick]}</span> <span class="muted">{pct(p.max())}</span></td>
  <td class="bk">{bk}</td>
</tr>""")
    md_dates = f"{nxt.Date.min().strftime('%d')}–{nxt.Date.max().strftime('%d %b %Y')}"

    # ---- season record
    played = played.sort_values("Date")
    played["bk_pick"] = None
    ok = played.OddH.notna()
    if ok.any():
        q = market(played[ok])
        played.loc[ok, "bk_pick"] = np.array(list("HDA"))[q.argmax(1)]
        played.loc[ok, "bk_conf"] = q.max(1)
    n = len(played)
    acc_model = played.Correct.mean()
    with_odds = played[ok]
    acc_bk = (with_odds.bk_pick == with_odds.Result).mean() if len(with_odds) else np.nan
    acc_model_same = with_odds.Correct.mean() if len(with_odds) else np.nan
    home_share = (played.Result == "H").mean()
    dis = with_odds[with_odds.bk_pick != with_odds.Pick]
    dis_rows = "".join(f"""<tr><td class="date">{r.Date.strftime('%d %b')}</td>
  <td class="fix"><span class="team">{e(r.HomeTeam)}</span> <span class="v">v</span> <span class="team">{e(r.AwayTeam)}</span></td>
  <td>{int(r.FTHG)}–{int(r.FTAG)}</td>
  <td><span class="pill pill-{r.Pick}">{LABEL[r.Pick]}</span> {'✓' if r.Correct else '✗'}</td>
  <td><span class="pill pill-{r.bk_pick}">{LABEL[r.bk_pick]}</span> {'✓' if r.bk_pick == r.Result else '✗'}</td></tr>"""
                       for r in dis.itertuples(index=False))
    rec_rows = "".join(f"""<tr><td class="date">{r.Date.strftime('%d %b')}</td>
  <td class="fix"><span class="team">{e(r.HomeTeam)}</span> <span class="v">v</span> <span class="team">{e(r.AwayTeam)}</span></td>
  <td>{int(r.FTHG)}–{int(r.FTAG)}</td>
  <td class="probs">{prob_bar(np.array([r.pH, r.pD, r.pA]), r.Pick)}</td>
  <td class="{'hit' if r.Correct else 'miss'}">{LABEL[r.Pick]} {'✓' if r.Correct else '✗'}</td>
  <td>{LABEL[r.bk_pick] + (' ✓' if r.bk_pick == r.Result else ' ✗') if r.bk_pick else '<span class="muted">—</span>'}</td></tr>"""
                       for r in played.sort_values("Date", ascending=False).itertuples(index=False))

    # ---- backtest
    bt_rows = "".join(f"<tr><td>{r.season}</td><td>{pct(r.ensemble)}</td><td>{pct(r.bookmakers)}</td></tr>"
                      for r in bt.itertuples(index=False))
    bt_all = {"ens": (bt.ensemble * bt.matches).sum() / bt.matches.sum(),
              "bk": (bt.bookmakers * bt.matches).sum() / bt.matches.sum(), "n": int(bt.matches.sum())}

    page = f"""<title>PL Predictions</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Source+Sans+3:wght@400;600&display=swap">
<style>
/* layout: one column, scoreboard tables; the model is green, the market is amber */
:root {{
  --bg: #f6f7f4; --surface: #ffffff; --fg: #1b2420; --muted: #66706b; --line: #d9ded9;
  --model: #1f7a4d; --model-soft: #d6eee0; --market: #b4731a; --market-soft: #f6e6cc;
  --draw: #8a93a0; --away: #3f5fa8; --away-soft: #dde5f6; --hit: #1f7a4d; --miss: #b23a3a;
  --display: "Barlow Condensed", "Arial Narrow", Impact, sans-serif;
  --body: "Source Sans 3", "Segoe UI", Helvetica, Arial, sans-serif;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #141a17; --surface: #1d2622; --fg: #e8ece9; --muted: #9aa49e; --line: #2f3a35;
  --model: #4cc48a; --model-soft: #1f3d2f; --market: #e0a04a; --market-soft: #40301a;
  --draw: #6f7a86; --away: #7f9ee0; --away-soft: #253352; --hit: #4cc48a; --miss: #e06a6a; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #141a17; --surface: #1d2622; --fg: #e8ece9; --muted: #9aa49e; --line: #2f3a35;
  --model: #4cc48a; --model-soft: #1f3d2f; --market: #e0a04a; --market-soft: #40301a;
  --draw: #6f7a86; --away: #7f9ee0; --away-soft: #253352; --hit: #4cc48a; --miss: #e06a6a; color-scheme: dark }}
body {{ background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 16px; line-height: 1.45; }}
.wrap {{ max-width: 960px; margin: 0 auto; padding-block: 28px 48px; padding-inline: 16px; }}
h1, h2 {{ font-family: var(--display); font-weight: 700; letter-spacing: .01em; text-wrap: balance; margin: 0; }}
h1 {{ font-size: clamp(34px, 6vw, 52px); line-height: 1; text-transform: uppercase; }}
h2 {{ font-size: 26px; margin-top: 40px; padding-bottom: 6px; border-bottom: 3px solid var(--fg); }}
.sub {{ color: var(--muted); margin: 6px 0 0; max-width: 62ch; }}
.eyebrow {{ font-family: var(--display); font-weight: 600; text-transform: uppercase; letter-spacing: .08em; font-size: 14px; color: var(--muted); }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-top: 18px; }}
.tile {{ background: var(--surface); border: 1px solid var(--line); padding: 12px 14px; }}
.tile .n {{ font-family: var(--display); font-size: 40px; font-weight: 700; line-height: 1; font-variant-numeric: tabular-nums; }}
.tile .l {{ color: var(--muted); font-size: 14px; margin-top: 4px; }}
.tile.model .n {{ color: var(--model); }} .tile.market .n {{ color: var(--market); }}
.tablewrap {{ overflow-x: auto; margin-top: 16px; min-width: 0; max-width: 100%; }}
.two > * {{ min-width: 0; }}
code {{ overflow-wrap: anywhere; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
th {{ text-align: left; font-family: var(--display); font-weight: 600; text-transform: uppercase; letter-spacing: .06em; font-size: 13px; color: var(--muted); padding: 6px 8px; border-bottom: 1px solid var(--line); white-space: nowrap; }}
td {{ padding: 9px 8px; border-bottom: 1px solid var(--line); vertical-align: middle; }}
td.date {{ white-space: nowrap; color: var(--muted); font-size: 14px; }}
.team {{ font-weight: 600; }} .v {{ color: var(--muted); padding: 0 3px; }}
.elo {{ display: block; color: var(--muted); font-size: 13px; }}
.probs {{ min-width: 180px; }}
.bar {{ display: flex; height: 22px; width: 100%; max-width: 260px; border: 1px solid var(--line); background: var(--surface); }}
.seg {{ display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600; color: #fff; overflow: hidden; opacity: .55; }}
.seg-pick {{ opacity: 1; }}
.seg-H {{ background: var(--model); }} .seg-D {{ background: var(--draw); }} .seg-A {{ background: var(--away); }}
.pill {{ display: inline-block; padding: 2px 8px; font-size: 13px; font-weight: 600; white-space: nowrap; }}
.pill-H {{ background: var(--model-soft); color: var(--model); }} .pill-D {{ background: var(--line); color: var(--fg); }} .pill-A {{ background: var(--away-soft); color: var(--away); }}
.muted {{ color: var(--muted); }} .hit {{ color: var(--hit); font-weight: 600; }} .miss {{ color: var(--miss); font-weight: 600; }}
.note {{ background: var(--surface); border-left: 4px solid var(--market); padding: 10px 14px; margin-top: 16px; max-width: 70ch; }}
.legend {{ display: flex; gap: 16px; flex-wrap: wrap; font-size: 13px; color: var(--muted); margin-top: 8px; }}
.legend i {{ display: inline-block; width: 12px; height: 12px; vertical-align: -1px; margin-right: 4px; }}
.two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }} @media (max-width: 640px) {{ .two {{ grid-template-columns: 1fr; }} }}
p {{ max-width: 68ch; }}
a {{ color: var(--model); }}
footer {{ margin-top: 48px; color: var(--muted); font-size: 14px; }}
</style>
<div class="wrap">
  <div class="eyebrow">Premier League 2026/27 · results-only model · results up to {last_played.strftime('%d %b %Y')}</div>
  <h1>PL Predictions</h1>
  <p class="sub">Each match, the model's win / draw / away probabilities next to the bookmakers' favourite, and a running count of who was right. Built from results alone, with no leakage: every number for a match uses only matches played before it.</p>

  <h2>This matchday · {md_dates}</h2>
  <div class="legend"><span><i style="background:var(--model)"></i>home win</span><span><i style="background:var(--draw)"></i>draw</span><span><i style="background:var(--away)"></i>away win</span><span>bright segment = the model's pick</span></div>
  <div class="tablewrap"><table>
    <thead><tr><th>Kick-off</th><th>Fixture</th><th>Model probabilities</th><th>Model pick</th><th>Bookmakers</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table></div>
  {"" if fo is not None else '<div class="note">Bookmakers\' odds for upcoming fixtures are added when <code>data/raw/football_data_full/fixtures.csv</code> from Football-Data.co.uk is present. The odds for played matches below come from the dataset.</div>'}

  <h2>Season so far</h2>
  <div class="tiles">
    <div class="tile model"><div class="n">{pct(acc_model)}</div><div class="l">model, {n} matches</div></div>
    <div class="tile market"><div class="n">{pct(acc_bk) if len(with_odds) else '—'}</div><div class="l">bookmakers, {len(with_odds)} matches with odds</div></div>
    <div class="tile"><div class="n">{pct(acc_model_same) if len(with_odds) else '—'}</div><div class="l">model on those same {len(with_odds)}</div></div>
    <div class="tile"><div class="n">{pct(home_share)}</div><div class="l">home wins this season (long-run 46%)</div></div>
  </div>
  <p class="sub">{n} matches is a small sample: accuracy swings by 5–8 points from season to season, and this season has had unusually few home wins so far. The seven-season backtest below is the reliable number.</p>

  <div class="eyebrow" style="margin-top:28px">Where the model and the bookmakers disagreed</div>
  {"<div class='tablewrap'><table><thead><tr><th>Date</th><th>Fixture</th><th>Score</th><th>Model</th><th>Bookmakers</th></tr></thead><tbody>" + dis_rows + "</tbody></table></div>" if len(dis) else "<p class='muted'>No disagreements yet on the matches with odds.</p>"}

  <div class="eyebrow" style="margin-top:28px">All matches</div>
  <div class="tablewrap"><table>
    <thead><tr><th>Date</th><th>Fixture</th><th>Score</th><th>Model probabilities</th><th>Model pick</th><th>Bookmakers</th></tr></thead>
    <tbody>{rec_rows}</tbody>
  </table></div>

  <h2>Backtest, 2019/20 to 2025/26</h2>
  <div class="two">
    <div>
      <div class="tiles" style="margin-top:0">
        <div class="tile model"><div class="n">{pct(bt_all['ens'])}</div><div class="l">final model, {bt_all['n']:,} matches</div></div>
        <div class="tile market"><div class="n">{pct(bt_all['bk'])}</div><div class="l">bookmakers' favourite</div></div>
        <div class="tile"><div class="n">43%</div><div class="l">always pick the home side</div></div>
      </div>
      <p class="sub">Each season predicted with a model trained only on earlier seasons. The final model uses confirmed lineups, which are not available live, so this page runs the results-only version (52.7% on the same backtest). At 65% or higher confidence, about one match in five, the model is right 73% of the time.</p>
    </div>
    <div class="tablewrap" style="margin-top:0"><table>
      <thead><tr><th>Season</th><th>Model</th><th>Bookmakers</th></tr></thead><tbody>{bt_rows}</tbody>
    </table></div>
  </div>

  <h2>How it works</h2>
  <p>Elo ratings updated after every match (bigger wins move the rating more, promoted sides start below the league mean), recent form over the last 5 and 10 matches, rest days, and a multinomial logistic regression on top. Everything was tested against the bookmakers with paired statistics, including head-to-head history, weather, fixture congestion, manager changes, expected goals, player ratings, suspensions and confirmed lineups. Only lineups moved the number. Code, data and the full write-up: <a href="{REPO}">github.com/jcosta92/PL_Predictions</a>.</p>
  <footer>Match analysis for learning purposes. Not betting advice. Data: Football-Data.co.uk via Club-Football-Match-Data (MIT), openfootball (CC0).</footer>
</div>
"""
    out = ROOT / "site" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB): {len(nxt)} fixtures, {n} played, {len(dis)} disagreements")


if __name__ == "__main__":
    main()
