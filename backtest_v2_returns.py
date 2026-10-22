"""Backtest v2, part 2: v2's monthly rankings against the next month's returns, with error bars.

WHY THIS EXISTS - ``plan/backtest-v2.md`` step 4: "a regression harness, not just a report ...
with confidence intervals, not a point estimate". ``backtest_v2.py`` builds what the screener
would have said each month from data known then; this module asks how those rankings did over
the following month, and says how sure the answer is.

* **Returns:** total return (adjusted close) from the last trading day on or before each
  month-end to the last on or before the next, for every name scored that month.
* **Per month:** the rank information coefficient (Spearman, composite vs next-month return),
  the same for each category score, and the decile spread (top tenth minus bottom tenth, and top
  tenth minus the scored universe's mean).
* **Across months:** the mean, its standard error from the month-to-month spread (1-month windows
  do not overlap, so each month is one observation - the independence point of
  ``research/2026-08-10-ic-evidence-independence.md``), a 95% interval, the t-statistic, and the
  share of months positive.

**What this cannot say, printed with every result** (``CAVEATS``): the universe is missing every
name that has left the index since (``cache/v2_census.json`` counts them - survivorship, step 2);
the ranking is the filings-and-prices screener, not the published one (no Revisions, no forward
EPS growth); and under CLAUDE.md rule 5 no number from it decides a weight or a methodology change
before 2027-02-11. It is colour for noticing what to research, labelled as such.

Usage: ``python backtest_v2_returns.py [--lookahead]`` (reads ``cache/v2_scores.parquet`` - or the
diagnostic ``v2_scores_lookahead.parquet`` - and ``cache/v2_daily.parquet``; writes
``cache/v2_returns.json``). The first run's record: ``research/measurements/2026-10-15-backtest-v2-returns*.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import universe_history as uh

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "cache" / "v2_returns.json"       # the latest run; dated copies go in research/measurements

CAVEATS = (
    "Survivorship: names that have left the S&P 500 since are not scored (no free price history); "
    "cache/v2_census.json counts them per month.",
    "The filings-and-prices screener: Revisions and forward EPS growth carry no weight (no free "
    "history), so this is not the published ranking.",
    "Not decision-grade (CLAUDE.md rule 5): no weight or methodology change rests on these figures "
    "before 2027-02-11.",
)


def daily_wide(daily: pd.DataFrame) -> pd.DataFrame:
    """Adjusted close per trading day: dates x tickers, tickers named as the universe names them.

    The one place the daily panel is reshaped. ``month_end_prices`` takes its month-end rows from
    this frame and a within-month price path (``v2_experiments.stop_loss_inputs``) takes its daily
    rows from it, so a stop measured against a month-end price is measured against the same series
    the forward return divides - keep it that way."""
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["ticker"] = d["ticker"].map(lambda s: s if s.startswith("^") else uh.normalize_ticker(s))
    return d.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()


def month_end_prices(daily: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    """Adjusted close on the last trading day on or before each month-end: months x tickers."""
    wide = daily_wide(daily)
    rows = {m: wide[wide.index <= pd.Timestamp(m)].ffill().iloc[-1] if (wide.index <= pd.Timestamp(m)).any()
            else pd.Series(dtype=float) for m in months}
    return pd.DataFrame(rows).T


def forward_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Return from each month-end to the next; the last month has none."""
    return prices.shift(-1) / prices - 1


def summarise(series: pd.Series) -> dict:
    """Mean, standard error, 95% interval, t and share positive for one monthly series."""
    x = pd.Series(series).dropna().astype(float)
    n = len(x)
    if n < 2:
        return {"n_months": n}
    m, sd = float(x.mean()), float(x.std(ddof=1))
    se = sd / np.sqrt(n)
    return {"n_months": n, "mean": round(m, 5), "se": round(se, 5),
            "ci95": [round(m - 1.96 * se, 5), round(m + 1.96 * se, 5)],
            "t": round(m / se, 2) if se > 0 else None, "share_positive": round(float((x > 0).mean()), 3)}


def evaluate(scores: pd.DataFrame, fwd: pd.DataFrame, n_deciles: int = 10) -> dict:
    """Per-month IC and decile spreads, and their summaries."""
    cats = [c for c in scores.columns if c.endswith("_score")]
    rows = []
    for m, g in scores.groupby("month"):
        if m not in fwd.index:
            continue
        r = fwd.loc[m].reindex(g["Ticker"].values).values
        g = g.assign(ret=r).dropna(subset=["ret", "Composite"])
        if len(g) < 50:
            continue
        row = {"month": m, "n": int(len(g)),
               "ic": float(g["Composite"].rank().corr(g["ret"].rank()))}
        for c in cats:
            ok = g[c].notna()
            if ok.sum() >= 50:
                row[f"ic_{c}"] = float(g.loc[ok, c].rank().corr(g.loc[ok, "ret"].rank()))
        dec = np.ceil(g["Composite"].rank(pct=True, method="first") * n_deciles).clip(1, n_deciles)
        top, bot = g.loc[dec == n_deciles, "ret"].mean(), g.loc[dec == 1, "ret"].mean()
        row.update(top=float(top), bottom=float(bot), universe=float(g["ret"].mean()),
                   spread=float(top - bot), top_minus_universe=float(top - g["ret"].mean()))
        rows.append(row)
    per = pd.DataFrame(rows)
    if per.empty:
        return {"caveats": list(CAVEATS), "months": 0, "per_month": []}
    out = {"caveats": list(CAVEATS), "months": int(len(per)),
           "ic": summarise(per["ic"]), "spread_top_minus_bottom": summarise(per["spread"]),
           "top_minus_universe": summarise(per["top_minus_universe"]),
           "category_ic": {c.replace("ic_", "").replace("_score", ""): summarise(per[c])
                           for c in per.columns if c.startswith("ic_")},
           "per_month": per.round(5).to_dict("records")}
    return out


def main():
    import sys
    lookahead = "--lookahead" in sys.argv          # the diagnostic arm (v1's look-ahead, on purpose)
    name = "v2_scores_lookahead.parquet" if lookahead else "v2_scores.parquet"
    scores = pd.read_parquet(ROOT / "cache" / name)
    daily = pd.read_parquet(ROOT / "cache" / "v2_daily.parquet")
    months = sorted(scores["month"].unique())
    nxt = uh.month_ends(months[-1], (pd.Timestamp(months[-1]) + pd.DateOffset(months=1)).strftime("%Y-%m-%d"))
    prices = month_end_prices(daily, months + [m for m in nxt if m not in months])
    out = evaluate(scores, forward_returns(prices))
    out["arm"] = ("DIAGNOSTIC - every month scored on today's filings (v1's look-ahead), to measure "
                  "what it does to the result; not a v2 result") if lookahead else "v2"
    path = OUT.with_name("v2_returns_lookahead.json") if lookahead else OUT
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_month"}, indent=1))


if __name__ == "__main__":
    main()
