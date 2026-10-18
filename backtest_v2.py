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
import subprocess
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
                 facts: pd.DataFrame, base: dict, ciks: dict | None = None,
                 month_ciks: dict | None = None, base_run: str = "unrecorded"):
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
        # ticker -> SEC CIK for every symbol the membership history has used; the CIK survives a
        # rename, so a member listed under an old symbol is found under its current one
        self.ciks = ciks or {}
        # month-end -> {symbol: CIK} as that month's revision listed them (from mid-2014; the older
        # revisions carry no CIK column): who a symbol belonged to *then*
        self.month_ciks = month_ciks or {}
        # which live run's raw fetch the static fields (sector, industry, share class) came from.
        # ``load`` takes the newest, so it changes as the data loop runs; a month built in August
        # and a month built today can disagree on a company's sector, and the sector is the peer
        # group the engine ranks within. Recorded so a ledger row can name it (item 8b).
        self.base_run = base_run
        self._by_cik: dict = {}
        for tk, cik in self.ciks.items():
            if tk in base:
                self._by_cik.setdefault(cik, []).append(tk)

    def current_symbol(self, ticker: str, members: set, taken: set, cik: int | None = None) -> str | None:
        """The symbol a member listed as ``ticker`` trades and files under today, when it was
        renamed (FB -> META, ANTM -> ELV): the one current symbol with the same SEC registrant
        number that is not itself a member that month and not already claimed by another member.
        ``None`` when there is none - a company that has since left the index stays missing, which
        is the survivorship this cannot fix."""
        cik = cik if cik is not None else self.ciks.get(ticker)
        if cik is None:
            return None
        cands = sorted(t for t in self._by_cik.get(cik, []) if t not in members and t not in taken)
        return cands[0] if cands else None

    @classmethod
    def load(cls, root: Path = ROOT) -> "Inputs":
        runs = sorted((root / "runs").glob("*/00_raw_fetch.parquet"), key=lambda p: p.stat().st_mtime)
        raw = pd.read_parquet(runs[-1])
        static = pr.FIELD_CLASSES["static"]
        base = {r["Ticker"]: {k: r.get(k) for k in static} for r in raw.to_dict("records")}
        return cls(pd.read_parquet(root / "cache" / "v2_daily.parquet"),
                   pd.read_parquet(root / "cache" / "v2_index.parquet"),
                   pd.read_parquet(root / "cache" / "splits.parquet"),
                   pd.read_parquet(root / "data" / "sec" / "pit" / "facts.parquet"), base,
                   _load_ciks(root), _load_month_ciks(root), base_run=runs[-1].parent.name)


def _load_month_ciks(root: Path) -> dict:
    """``data/universe_history/ciks_by_month.json`` (2026-10-15): each month's own symbol -> CIK."""
    import json as _json
    p = root / "data" / "universe_history" / "ciks_by_month.json"
    try:
        raw = _json.loads(p.read_text(encoding="utf-8"))
        return {m: {k: int(v) for k, v in d["ciks"].items()} for m, d in raw.items() if d.get("ciks")}
    except (OSError, KeyError, ValueError, TypeError, AttributeError):
        return {}


def _load_ciks(root: Path) -> dict:
    """The membership history's ticker -> CIK map (``data/universe_history/ticker_ciks.json``,
    built from the same Wikipedia revisions as the membership), or ``{}`` without it."""
    import json as _json
    p = root / "data" / "universe_history" / "ticker_ciks.json"
    try:
        return {k: int(v) for k, v in _json.loads(p.read_text(encoding="utf-8"))["ciks"].items()}
    except (OSError, KeyError, ValueError, TypeError):
        return {}


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


def metrics_month(records: list[dict], cfg: dict, market_returns: pd.Series, rf: float) -> pd.DataFrame:
    """The slow, weight-independent half of a month's scoring: every stock's metrics."""
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        df = fe.compute_metrics(records, market_returns, cfg, risk_free_rate=rf)
        return df[~df.get("_skipped", pd.Series(False, index=df.index)).fillna(False)].copy()


def rank_month(metrics: pd.DataFrame, cfg: dict, state: dict) -> pd.DataFrame:
    """The fast half: a month's metrics to its ranking under ``cfg``'s weights, in the live
    run's order (coverage filter, universe filter, sector percentiles, transform, categories, the
    month's regime, composite, rank). Split out 2026-10-15 so an experiment that only changes
    weights reruns this and not the fetch-shaped half (``v2_experiments.py``)."""
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        df = metrics.copy()
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
        # a weighting that scores only some categories (an experiment's narrow screen) can leave a
        # stock with no composite at all; it is left out of that month's ranking, not ranked as a
        # number it does not have. The published weighting never produces one.
        df = df[df["Composite"].notna()].copy()
        df = fe.rank_stocks(df)
    return df


def score_month(records: list[dict], cfg: dict, market_returns: pd.Series, rf: float,
                state: dict) -> pd.DataFrame:
    """The live run's scoring order, on one month's records."""
    return rank_month(metrics_month(records, cfg, market_returns, rf), cfg, state)


METRICS_DIR = CACHE / "v2_metrics"
BUILD_RECORD = METRICS_DIR / "_build.json"


def record_build(months: list[str], base_run: str, path: Path | None = None) -> Path:
    """What the cached months were built from, beside the months themselves (item 8b).

    The monthly metrics are a cache that experiments re-rank without recomputing, and the research
    engine's ledger has to be able to name what went into them. Two things are not recoverable from
    the parquet files: which live run supplied the static fields, and which commit built them. This
    merges the record rather than replacing it, because a build is often sharded or run a period at
    a time, so each month carries its own."""
    p = path or BUILD_RECORD
    try:
        old = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    except (OSError, ValueError):
        old = {}
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, cwd=ROOT).stdout.strip() or "?"
    except OSError:
        commit = "?"
    by_month = dict(old.get("months") or {})
    for m in months:
        by_month[m] = {"base_run": base_run, "commit": commit}
    runs = sorted({v["base_run"] for v in by_month.values()})
    rec = {"base_run": runs[0] if len(runs) == 1 else "+".join(runs),
           "commits": sorted({v["commit"] for v in by_month.values()}),
           "months": dict(sorted(by_month.items()))}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return p


def merge_build_records(k: int, path: Path | None = None) -> Path:
    """Fold the shards' build records into one. Shards run as separate processes, so each writes
    its own file - a shared read-modify-write would lose whichever shard finished first."""
    p = path or BUILD_RECORD
    by_month = {}
    try:
        by_month.update((json.loads(p.read_text(encoding="utf-8")).get("months") or {}) if p.is_file() else {})
    except (OSError, ValueError):
        pass
    for i in range(k):
        q = p.with_name(f"{p.stem}.shard{i}.json")
        if q.is_file():
            try:
                by_month.update(json.loads(q.read_text(encoding="utf-8")).get("months") or {})
            except (OSError, ValueError):
                continue
            q.unlink()
    runs = sorted({v["base_run"] for v in by_month.values()}) or ["unrecorded"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"base_run": runs[0] if len(runs) == 1 else "+".join(runs),
                             "commits": sorted({v["commit"] for v in by_month.values()}),
                             "months": dict(sorted(by_month.items()))}, indent=1), encoding="utf-8")
    return p


def save_metrics(as_of: str, metrics: pd.DataFrame, state: dict, path: Path | None = None) -> Path:
    """One month's metrics, for experiments to re-rank without recomputing them. Columns holding
    lists or dicts (per-stock detail the ranking never reads) are kept as JSON text."""
    import json as _json
    out = metrics.copy()
    for c in out.columns:
        if out[c].dtype == object and out[c].map(lambda v: isinstance(v, (list, dict, tuple))).any():
            out[c] = out[c].map(lambda v: _json.dumps(v, default=str) if isinstance(v, (list, dict, tuple)) else v)
    out["_month_state"] = state.get("state")
    p = path or (METRICS_DIR / f"{as_of}.parquet")
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(p)
    return p


def build_month(as_of: str, inputs: Inputs, cfg: dict, filings_as_of: pd.Timestamp | None = None):
    """(scores frame, census dict) for one month-end."""
    t = pd.Timestamp(as_of)
    snap = uh.members_on(as_of, allow_network=False)
    records, missing, renamed = [], {}, {}
    members = set(snap.tickers)
    then = inputs.month_ciks.get(as_of, {})
    then_symbol = {c: s for s, c in then.items()}
    for tk in snap.tickers:
        sym = tk
        cik_then, cik_now = then.get(tk), inputs.ciks.get(tk)
        if tk in inputs.base and cik_then is not None and cik_now is not None and cik_then != cik_now:
            # The month's revision names a different registrant for the symbol than today's.
            # Three cases, told apart by where the two registrants were that month:
            # 1. the then-company trades today under another symbol (IR in 2020 is TT today):
            #    score it there;
            # 2. today's registrant was itself a member that month under another symbol (JCI's
            #    filings today are Tyco's, and TYC was a member until 2016): today's data is that
            #    other member's, so this one is counted, never scored with it;
            # 3. neither - a holding-company reorganisation (Exxon, Disney, Cigna) or a stale CIK
            #    in the revision (Fox, Xerox in early 2020): today's symbol stands, and the filings
            #    as of the month decide (a new registrant has none before it existed).
            cur = inputs.current_symbol(tk, members, set(renamed.values()), cik=cik_then)
            if cur is not None:
                sym = renamed[tk] = cur
            elif then_symbol.get(cik_now) not in (None, tk):
                missing.setdefault("symbol_reused", []).append(tk)
                continue
        elif tk not in inputs.base:
            cur = inputs.current_symbol(tk, members, set(renamed.values()), cik=cik_then)
            if cur is not None:
                sym = renamed[tk] = cur
        rec, why = record_at(sym, inputs, t, filings_as_of)
        if rec is None:
            missing.setdefault(why, []).append(tk)
        else:
            records.append(rec)
    mr, rf, state = market_inputs(inputs, t, cfg.get("momentum_regime"))
    metrics = metrics_month(records, cfg, mr, rf)
    if filings_as_of is None:
        try:
            save_metrics(as_of, metrics, state)
        except Exception as exc:  # noqa: BLE001 - a cache for experiments must never stop a build
            print(f"{as_of}: metrics not cached ({type(exc).__name__}: {exc})", flush=True)
    df = rank_month(metrics, cfg, state)
    cats = [c for c in cfg["factor_weights"] if f"{c}_score" in df]
    out = df[["Ticker", "Composite", "Rank"] + [f"{c}_score" for c in cats]].copy()
    out.insert(0, "month", as_of)
    census = {"month": as_of, "members": snap.n_members, "staleness_days": snap.staleness_days,
              "built": len(records), "scored": int(len(out)),
              "dropped_by_engine_filters": len(records) - int(len(out)),
              "not_scoreable": {k: sorted(v) for k, v in missing.items()},
              "renamed": dict(sorted(renamed.items())),
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
    ap.add_argument("--tag", default="", help="a separate output for another period, e.g. _presample "
                    "(2012-2019, registered 2026-10-15): v2_scores<tag>.parquet / v2_census<tag>.json, so "
                    "it never replaces the 2020-2026 result. Monthly metrics share cache/v2_metrics/")
    args = ap.parse_args()
    global SCORES_PATH, CENSUS_PATH
    tag = ""
    f_as_of = None
    if args.diagnostic_filings_as_of:
        f_as_of = pd.Timestamp(args.diagnostic_filings_as_of)
        tag = "_lookahead"
        SCORES_PATH = CACHE / "v2_scores_lookahead.parquet"
        CENSUS_PATH = CACHE / "v2_census_lookahead.json"
    elif args.tag:
        tag = args.tag
        SCORES_PATH = CACHE / f"v2_scores{tag}.parquet"
        CENSUS_PATH = CACHE / f"v2_census{tag}.json"
    if args.merge:
        parts = [CACHE / f"v2_scores{tag}.shard{k}.parquet" for k in range(args.merge)]
        cens = [CACHE / f"v2_census{tag}.shard{k}.json" for k in range(args.merge)]
        pd.concat([pd.read_parquet(p) for p in parts]).sort_values(["month", "Rank"]).to_parquet(SCORES_PATH)
        allc = sorted(sum((json.loads(c.read_text(encoding="utf-8")) for c in cens), []),
                      key=lambda r: r["month"])
        CENSUS_PATH.write_text(json.dumps(allc, indent=1), encoding="utf-8")
        for p in parts + cens:
            p.unlink()
        merge_build_records(args.merge)
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
    frames, census, built = [], [], []
    for m in months:
        try:
            out, c = build_month(m, inputs, cfg, f_as_of)
            if f_as_of is None:
                built.append(m)          # the diagnostic build caches no metrics, so it records none
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
    if built:
        k = args.shard.split("/")[0] if args.shard else None
        record_build(built, inputs.base_run,
                     BUILD_RECORD.with_name(f"{BUILD_RECORD.stem}.shard{k}.json") if k else None)


if __name__ == "__main__":
    main()
