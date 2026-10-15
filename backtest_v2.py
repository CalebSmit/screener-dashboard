"""Backtest v2, part 1: what the screener would have said each month, from what was known then.

WHY THIS EXISTS - ``plan/backtest-v2.md`` step 3. ``backtest.py`` v1 scores 2020-2026 on today's
fundamentals and today's universe; measured, that moves 75.8% of its name-months to a different
decile from what the screener would have said at the time
(``research/2026-10-15-pit-record-validation.md``). This module builds the honest version of the
*rankings* - not yet the returns:

* **Universe:** the S&P 500 as it stood at each month-end (``universe_history.members_on``,
  offline cache). Members with no current raw record (they left the index since) or no price are
  **counted and named in the census, never silently dropped** - that gap is survivorship, step 2.
* **Fundamentals:** ``pit_record.record_as_of`` - the engine's raw record rebuilt from SEC filings
  filed by the month-end, valued at that day's close (split-adjusted, not dividend-adjusted, and
  put on the filings' share basis by ``split_after``).
* **Momentum and risk inputs:** ``factor_engine.price_history_fields`` - the live fetch's own code
  - on the 13 months of daily history to the month-end (reproduces the live fetch's fields exactly:
  ``tests/test_backtest_v2.py``); beta and alpha against ``^SP500TR`` over the year to the
  month-end, the risk-free rate from that day's ``^IRX``, as the live run reads them.
* **The momentum rule:** ``market_panic_state`` on the index's history *to that month-end* -
  reading today's regime would be a third look-ahead, inside the weights.
* **What has no free history is removed from the weights, openly.** Analyst estimates and short
  interest cannot be known retrospectively for free, so v2 scores the *filings-and-prices*
  screener: Revisions (all of it) and ``forward_eps_growth`` carry weight 0 and the engine
  renormalises the rest (``v2_config``). That is a different screener from the live one by those
  15.85 points, and every output says so.
* **Scoring:** the engine's own pipeline, in the live run's order - ``compute_metrics``, the
  coverage filter, the universe filter, sector percentiles, the transform, category scores, the
  month's regime, the composite, the rank.

**No returns are computed here.** Forward returns, deciles against returns and the A/B harness
with confidence intervals are step 4. Under CLAUDE.md rule 5 nothing this produces decides
anything before 2027-02-11, and production code must not import it (``tests/test_backtest_v2.py``).

Usage::

    python backtest_v2.py --start 2020-01-31 --end 2026-09-30 [--shard k/K] [--merge K]

Writes ``cache/v2_scores.parquet`` (month, ticker, composite, rank, category scores) and
``cache/v2_census.json`` (per month: members, scored, and every member not scored, by reason).
Inputs (gitignored ``cache/``): ``v2_daily.parquet`` (daily close, adjusted close, volume per
ticker), ``v2_index.parquet`` (^SP500TR, ^GSPC, ^IRX since 2015), ``splits.parquet``, and
``data/sec/pit/facts.parquet``.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import factor_engine as fe
import pit_record as pr
import universe_history as uh

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
SCORES_PATH = CACHE / "v2_scores.parquet"
CENSUS_PATH = CACHE / "v2_census.json"

#: Weighted metrics whose inputs have no free retrospective source (pit_record's
#: ``no_free_history`` class). v2 gives them weight 0 rather than pretend.
UNMEASURABLE_METRICS = ("forward_eps_growth", "fy1_revision_3m", "analyst_surprise",
                        "price_target_upside", "consecutive_beat_streak",
                        "short_interest_ratio", "earnings_acceleration", "peg_ratio",
                        "analyst_rating", "short_pct_float", "insider_ownership")
INDEX = "^SP500TR"
WINDOW_MONTHS = 13            # the fetch's history(period="13mo")


def v2_config(cfg: dict) -> dict:
    """``cfg`` with every unmeasurable metric at weight 0, and a category left with no weight
    at 0 itself. The engine renormalises what remains, as it does for any missing metric."""
    out = copy.deepcopy(cfg)
    for table in ("metric_weights", "bank_metric_weights"):
        for cat, metrics in (out.get(table) or {}).items():
            for m in metrics:
                if m in UNMEASURABLE_METRICS:
                    metrics[m] = 0
    for cat in list(out["factor_weights"]):
        if not any(float(w) > 0 for w in (out["metric_weights"].get(cat) or {}).values()):
            out["factor_weights"][cat] = 0
    return out


class Inputs:
    """Everything v2 reads, loaded once, offline."""

    def __init__(self, daily: pd.DataFrame, index: pd.DataFrame, splits: pd.DataFrame,
                 facts: pd.DataFrame, base: dict):
        daily = daily.copy()
        daily["date"] = pd.to_datetime(daily["date"])
        # one ticker form everywhere - the universe's (``BRK-B``, not ``BRK.B``); a cache written
        # in the other form would otherwise leave those names unpriced, silently
        daily["ticker"] = daily["ticker"].map(lambda s: s if s.startswith("^") else uh.normalize_ticker(s))
        splits = splits.copy()
        splits["ticker"] = splits["ticker"].map(uh.normalize_ticker)
        self.daily = {t: g.set_index("date").sort_index() for t, g in daily.groupby("ticker")}
        index = index.copy()
        index["date"] = pd.to_datetime(index["date"])
        self.index = {t: g.set_index("date")["close"].sort_index() for t, g in index.groupby("ticker")}
        self.splits = splits
        self.facts = {t: g for t, g in facts.groupby("ticker")}
        self.empty_facts = facts.iloc[:0]
        self.base = base

    @classmethod
    def load(cls, root: Path = ROOT) -> "Inputs":
        runs = sorted((root / "runs").glob("*/00_raw_fetch.parquet"), key=lambda p: p.stat().st_mtime)
        raw = pd.read_parquet(runs[-1])
        static = pr.FIELD_CLASSES["static"]
        base = {r["Ticker"]: {k: r.get(k) for k in static} for r in raw.to_dict("records")}
        return cls(pd.read_parquet(root / "cache" / "v2_daily.parquet"),
                   pd.read_parquet(root / "cache" / "v2_index.parquet"),
                   pd.read_parquet(root / "cache" / "splits.parquet"),
                   pd.read_parquet(root / "data" / "sec" / "pit" / "facts.parquet"), base)


def history_window(g: pd.DataFrame, as_of: pd.Timestamp, splits: pd.DataFrame | None = None) -> pd.DataFrame:
    """The fetch's 13-month daily frame, ending at ``as_of``: Close adjusted for splits and
    dividends (as ``history(auto_adjust=True)``), Volume, and the declared splits as
    ``Stock Splits`` - without them ``check_price_series_integrity`` never arms, and a series that
    mixes two price scales would be scored (the MNST defect, 2026-08-26)."""
    w = g[(g.index > as_of - pd.DateOffset(months=WINDOW_MONTHS)) & (g.index <= as_of)]
    out = pd.DataFrame({"Close": w["adj_close"], "Volume": w["volume"]}).dropna(subset=["Close"])
    out["Stock Splits"] = 0.0
    if splits is not None and not splits.empty:
        for d, r in zip(pd.to_datetime(splits["date"]), splits["ratio"]):
            if d in out.index:
                out.loc[d, "Stock Splits"] = float(r)
    return out


def market_inputs(inputs: Inputs, as_of: pd.Timestamp, rule: dict | None):
    """(market log returns over the year to ``as_of``, risk-free rate, momentum-rule reading),
    each as the live run reads it on that day."""
    idx = inputs.index[INDEX]
    yr = idx[(idx.index > as_of - pd.DateOffset(years=1)) & (idx.index <= as_of)]
    mr = np.log(yr / yr.shift(1)).dropna()
    irx = inputs.index["^IRX"]
    irx = irx[irx.index <= as_of].dropna()
    rf = float(irx.iloc[-1]) / 100.0 if len(irx) else 0.045
    state = {"state": "UNKNOWN"}
    if rule and rule.get("enabled"):
        closes = idx[(idx.index > as_of - pd.DateOffset(years=3)) & (idx.index <= as_of)]
        state = fe.market_panic_state(closes, rule)
    return mr, rf, state


def record_at(ticker: str, inputs: Inputs, as_of: pd.Timestamp, filings_as_of: pd.Timestamp | None = None):
    """(record, None) or (None, reason) for one member at one month-end.

    ``filings_as_of`` is a **diagnostic only**: read the filings as of another date (today) to
    rebuild v1's look-ahead on purpose and measure what it does to the result. Never set it to
    produce a v2 result."""
    base = inputs.base.get(ticker)
    if base is None:
        return None, "no_current_record"
    g = inputs.daily.get(ticker)
    if g is None:
        return None, "no_price"
    hist = history_window(g, as_of, inputs.splits[inputs.splits["ticker"] == ticker])
    if len(hist) < 10:
        return None, "no_price"
    close = g.loc[g.index <= as_of, "close"].dropna()
    if close.empty:
        return None, "no_price"
    price = float(close.iloc[-1])
    facts = inputs.facts.get(ticker, inputs.empty_facts)
    f_date = filings_as_of if filings_as_of is not None else as_of
    rec = pr.record_as_of(dict(base), facts, f_date, price,
                          split=pr.split_after(inputs.splits, ticker, f_date))
    if not np.isfinite(rec.get("netIncome", np.nan)) and not np.isfinite(rec.get("totalAssets", np.nan)):
        return None, "no_filings"
    for k in pr.FIELD_CLASSES["no_free_history"]:
        rec[k] = np.nan
    rec.update(fe.price_history_fields(hist))
    yr = close[close.index > as_of - pd.DateOffset(years=1)]
    rec["fiftyTwoWeekHigh"] = float(yr.max()) if len(yr) else np.nan   # closes, not intraday highs
    return rec, None


def score_month(records: list[dict], cfg: dict, market_returns: pd.Series, rf: float,
                state: dict) -> pd.DataFrame:
    """The live run's scoring order, on one month's records."""
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        df = fe.compute_metrics(records, market_returns, cfg, risk_free_rate=rf)
        df = df[~df.get("_skipped", pd.Series(False, index=df.index)).fillna(False)].copy()
        mc, applicable = fe.applicable_coverage(df, cfg)
        need = (applicable * cfg["data_quality"]["min_data_coverage_pct"] / 100).apply(lambda x: max(1, int(x)))
        df = df[mc >= need].copy()
        df = fe.apply_universe_filters(df, cfg)
        df = fe.compute_sector_percentiles(df)
        df = fe.apply_percentile_transform(df, cfg)
        df = fe.compute_category_scores(df, cfg)
        month_cfg = copy.deepcopy(cfg)
        if state.get("state") == "PANIC":
            month_cfg["factor_weights"] = fe.apply_momentum_regime(cfg["factor_weights"], "PANIC")
        df = fe.compute_composite(df, month_cfg)
        df = fe.rank_stocks(df)
    return df


def build_month(as_of: str, inputs: Inputs, cfg: dict, filings_as_of: pd.Timestamp | None = None):
    """(scores frame, census dict) for one month-end."""
    t = pd.Timestamp(as_of)
    snap = uh.members_on(as_of, allow_network=False)
    records, missing = [], {}
    for tk in snap.tickers:
        rec, why = record_at(tk, inputs, t, filings_as_of)
        if rec is None:
            missing.setdefault(why, []).append(tk)
        else:
            records.append(rec)
    mr, rf, state = market_inputs(inputs, t, cfg.get("momentum_regime"))
    df = score_month(records, cfg, mr, rf, state)
    cats = [c for c in cfg["factor_weights"] if f"{c}_score" in df]
    out = df[["Ticker", "Composite", "Rank"] + [f"{c}_score" for c in cats]].copy()
    out.insert(0, "month", as_of)
    census = {"month": as_of, "members": snap.n_members, "staleness_days": snap.staleness_days,
              "built": len(records), "scored": int(len(out)),
              "dropped_by_engine_filters": len(records) - int(len(out)),
              "not_scoreable": {k: sorted(v) for k, v in missing.items()},
              "regime": state.get("state"), "risk_free": round(rf, 4)}
    return out, census


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-31")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--shard", default="")
    ap.add_argument("--merge", type=int, default=0)
    ap.add_argument("--diagnostic-filings-as-of", default="",
                    help="DIAGNOSTIC: read every month's filings as of this date (v1's look-ahead, on "
                         "purpose); writes v2_scores_lookahead*.parquet, never the v2 result")
    args = ap.parse_args()
    global SCORES_PATH, CENSUS_PATH
    tag = ""
    f_as_of = None
    if args.diagnostic_filings_as_of:
        f_as_of = pd.Timestamp(args.diagnostic_filings_as_of)
        tag = "_lookahead"
        SCORES_PATH = CACHE / "v2_scores_lookahead.parquet"
        CENSUS_PATH = CACHE / "v2_census_lookahead.json"
    if args.merge:
        parts = [CACHE / f"v2_scores{tag}.shard{k}.parquet" for k in range(args.merge)]
        cens = [CACHE / f"v2_census{tag}.shard{k}.json" for k in range(args.merge)]
        pd.concat([pd.read_parquet(p) for p in parts]).sort_values(["month", "Rank"]).to_parquet(SCORES_PATH)
        allc = sorted(sum((json.loads(c.read_text(encoding="utf-8")) for c in cens), []),
                      key=lambda r: r["month"])
        CENSUS_PATH.write_text(json.dumps(allc, indent=1), encoding="utf-8")
        for p in parts + cens:
            p.unlink()
        print(f"merged {len(allc)} months -> {SCORES_PATH.name}, {CENSUS_PATH.name}")
        return
    cfg = v2_config(yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")))
    inputs = Inputs.load()
    months = uh.month_ends(args.start, args.end)
    scores_path, census_path = SCORES_PATH, CENSUS_PATH
    if args.shard:
        k, K = (int(x) for x in args.shard.split("/"))
        months = months[k::K]
        scores_path = CACHE / f"v2_scores{tag}.shard{k}.parquet"
        census_path = CACHE / f"v2_census{tag}.shard{k}.json"
    frames, census = [], []
    for m in months:
        try:
            out, c = build_month(m, inputs, cfg, f_as_of)
        except uh.UniverseHistoryError as e:
            # no cached membership for the month: say so, never guess with today's list
            census.append({"month": m, "skipped": f"no cached universe: {e}"})
            print(f"{m}: skipped - no cached universe", flush=True)
            continue
        frames.append(out)
        census.append(c)
        miss = {k: len(v) for k, v in c["not_scoreable"].items()}
        print(f"{m}: {c['members']} members, {c['scored']} scored, not scoreable {miss}, "
              f"regime {c['regime']}", flush=True)
    (pd.concat(frames) if frames else pd.DataFrame(columns=["month", "Ticker", "Composite", "Rank"])
     ).to_parquet(scores_path)
    census_path.write_text(json.dumps(census, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
