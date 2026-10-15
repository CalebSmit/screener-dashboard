#!/usr/bin/env python3
"""Refuse to publish a degraded screener run.

Every failure this guards against actually happened, in the same week:

* **2026-08-06** - no network, so the screener substituted synthetic values for
  all 503 tickers and emitted a normal-looking 2.6 MB payload. Only a failed
  push kept fabricated scores off the public site.
* **2026-08-07 / 2026-08-10** - runs warm-started from cache and skipped the
  fetch stage entirely. 0 of 503 stocks had a price or an analyst target, every
  category's dispersion collapsed 25-36%, and the published Top 5 was wrong for
  three days.

* **2026-10-14** - the source stopped serving one `.info` field and
  `dividend_yield` fell from 402 of 501 stocks to 6. This script said PASS, the
  claims gate said PASS, and the run published; the code loop's test suite found
  it four hours later. Nothing here had ever compared one run's per-metric
  availability with the last one's. `check_metric_availability` does.

Both runs reported "0 issues logged, 0 fetch failures". The screener does not
consider "I fetched nothing" to be a problem, and file size does not reveal it -
a fully synthetic run produces a perfectly normal-sized payload.

The dispersion history had already *recorded* the 08-07 collapse. Nothing was
watching it. This script is the thing that watches.

Exit codes:
    0  healthy - safe to publish
    1  degraded - discard the run
    2  could not evaluate (missing inputs)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
DISPERSION_HISTORY = ROOT / "improvement" / "dispersion_history.csv"

CATEGORIES = ["valuation", "quality", "growth", "momentum", "risk"]

# A real fetch writes these. A cached warm-start does not - which is exactly
# how the 08-07 and 08-10 runs published price-less data.
FETCH_EVIDENCE = ["00_raw_fetch.parquet", "01_raw_metrics.parquet"]

# Thresholds. Deliberately loose: this catches collapse, not normal drift.
MIN_PRICE_COVERAGE = 0.90     # >=90% of stocks must have a price
MIN_TARGET_COVERAGE = 0.50    # analyst coverage is genuinely patchy; 0% is not
MAX_DISPERSION_DROP = 0.20    # >20% below trailing median = collapse
MIN_HISTORY_FOR_CHECK = 3     # need a few prior runs before comparing

# Momentum and risk are 23% of composite weight and every metric in both is
# derived from one `Ticker.history()` call per stock. Since 2026-08-26 a
# series that mixes two price scales is rejected outright and those metrics
# are withheld (factor_engine.check_price_series_integrity), which is right
# for the one-off case - MNST, 1 of 502 - but would be a silent catastrophe
# if a Yahoo-side change ever rejected the universe. Dispersion would not
# catch it: with most stocks NaN it is computed over whatever survives.
MIN_CATEGORY_COVERAGE = 0.90


class Result:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    @property
    def healthy(self) -> bool:
        return not self.failures


def latest_run_dir() -> Path | None:
    if not RUNS_DIR.exists():
        return None
    candidates = [
        d for d in RUNS_DIR.iterdir()
        if d.is_dir() and (d / "meta.json").exists() and not d.name.startswith("test_")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def check_fetch_happened(run_dir: Path, res: Result) -> None:
    """A cached run that never fetched cannot have prices or analyst targets."""
    missing = [f for f in FETCH_EVIDENCE if not (run_dir / f).exists()]
    if missing:
        res.fail(
            f"no evidence of a live fetch - missing {', '.join(missing)} in "
            f"{run_dir.name}. The run warm-started from cache and skipped "
            f"fetching, so prices and analyst targets will be absent."
        )
    else:
        res.note(f"fetch evidence present ({', '.join(FETCH_EVIDENCE)})")


def load_payload(run_dir: Path) -> dict | None:
    path = run_dir / "dashboard_data.js"
    if not path.exists():
        path = ROOT / "dashboard_data.js"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text[text.index("{"):].rstrip().rstrip(";"))
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"dashboard_data.js did not parse: {exc}") from exc


def check_coverage(payload: dict, res: Result) -> None:
    detail = payload.get("stock_detail") or {}
    if not detail:
        res.fail("stock_detail is empty")
        return

    n = len(detail)
    priced = sum(1 for v in detail.values() if v.get("price"))
    targeted = sum(1 for v in detail.values() if v.get("pt_mean"))

    price_cov = priced / n
    target_cov = targeted / n

    if price_cov < MIN_PRICE_COVERAGE:
        res.fail(
            f"only {priced}/{n} stocks have a price ({price_cov:.0%}, "
            f"need >={MIN_PRICE_COVERAGE:.0%})"
        )
    else:
        res.note(f"price coverage {priced}/{n} ({price_cov:.0%})")

    if target_cov < MIN_TARGET_COVERAGE:
        res.fail(
            f"only {targeted}/{n} stocks have an analyst target "
            f"({target_cov:.0%}, need >={MIN_TARGET_COVERAGE:.0%})"
        )
    else:
        res.note(f"analyst target coverage {targeted}/{n} ({target_cov:.0%})")

    check_price_derived_coverage(payload, res)


def check_price_derived_coverage(payload: dict, res: Result) -> None:
    """Momentum and risk go missing together when a price series is rejected.

    One or two names is the mechanism working. Most of the universe means
    the upstream feed changed shape, and publishing 23% of the composite as
    "withheld" across the board is not a screener.
    """
    rows = payload.get("table_data") or []
    if not rows:
        return

    n = len(rows)
    for cat in ("momentum", "risk"):
        key = f"{cat}_score"
        present = sum(1 for r in rows if r.get(key) is not None)
        cov = present / n
        if cov < MIN_CATEGORY_COVERAGE:
            res.fail(
                f"only {present}/{n} stocks have a {cat} score "
                f"({cov:.0%}, need >={MIN_CATEGORY_COVERAGE:.0%}) - "
                f"price histories are being rejected en masse"
            )
        else:
            res.note(f"{cat} coverage {present}/{n} ({cov:.0%})")


def check_metric_availability(res: Result) -> None:
    """A metric that goes missing for most of the universe is a source event.

    **2026-10-14**: the source stopped serving ``.info["dividendRate"]`` and
    ``dividend_yield`` fell from 402 of 501 stocks to **6**. Every gate in the
    publishing loop passed - HEALTH: PASS, 454 claims-gate tests green - because
    nothing compared this run's per-metric availability with the last run's. The
    *code* loop's test suite caught it four hours later, through the input-churn
    tripwire, by which time the run was published and the live page read
    "Dividend Yield: None" for Coca-Cola, J&J and Procter & Gamble. This is the
    counterpart that belongs in the loop that publishes.

    The comparison is the snapshot directory's, read through ``history`` so that
    "most of the universe" has **one** definition shared with the input-churn
    record (``history.universe_wide_changes``: more than half the shared
    tickers). The snapshot for this run is written at step 13 of
    ``run_screener.py``, before this script runs, and ``data-run.ps1`` cleans it
    away again when a run is discarded.

    A metric that **carries weight** is a failure: every stock's composite would
    be computed on a different metric set from yesterday's, which is a
    measurement change wearing a ranking's clothes, and dispersion does not
    notice because renormalisation keeps the spread. A metric that carries
    **none** is a note: it is display-only, and jamming the evidence loop over a
    row in the Company Snapshot is the wrong trade - the same reasoning as the
    node and pytest fallbacks in both publish paths.
    """
    sys.path.insert(0, str(ROOT))
    try:
        import history as H
    except ImportError as exc:
        res.note(f"metric availability not checked - history unavailable ({exc})")
        return

    index = H.snapshot_index()
    dates = sorted(index)
    if len(dates) < 2:
        res.note("metric availability not checked - fewer than two snapshots")
        return
    try:
        current = H._to_run_snapshot(dates[-1], pd.read_parquet(index[dates[-1]]))
        baseline = H._to_run_snapshot(dates[-2], pd.read_parquet(index[dates[-2]]))
    except (OSError, ValueError, KeyError) as exc:
        res.note(f"metric availability not checked - snapshots unreadable ({exc})")
        return

    wide = H.universe_wide_changes(current, baseline)
    if not wide:
        res.note(f"no metric changed availability across the universe "
                 f"vs {dates[-2]}")
        return

    weighted = _weighted_metric_names()
    for col in sorted(wide):
        metric = col[:-4] if col.endswith("_pct") else col
        n_cur = sum(1 for t in current.ranks
                    if col not in current.missing_metrics.get(t, frozenset()))
        n_base = sum(1 for t in baseline.ranks
                     if col not in baseline.missing_metrics.get(t, frozenset()))
        where = (f"{metric}: available for {n_cur}/{len(current.ranks)} stocks, "
                 f"was {n_base}/{len(baseline.ranks)} on {dates[-2]}")
        # Only a *loss* can refuse a publish. The day after an outage the
        # universe regains the metric, and that run is better than the one
        # before it, not worse - a gate that discarded the recovery would turn a
        # one-day source outage into a permanent one.
        if n_cur >= n_base:
            res.note(f"{where} - regained, not lost")
        elif weighted is None:
            res.note(f"{where} - cannot tell whether it carries weight")
        elif metric in weighted:
            res.fail(f"{where}. It carries weight, so every composite is "
                     f"computed on a different metric set from the last run - "
                     f"a measurement change, not a ranking change.")
        else:
            res.note(f"{where} - carries no weight, so no score moved")


# The fields compared with the filings, and the line above which a count of disagreements is
# called out. Written 2026-10-14 from that day's measurement, before any run had been checked:
# over 10% apart, total assets 0 stocks, operating cash flow 1, equity 2, net income 7, revenue
# 13 (research/2026-10-14-yahoo-vs-sec.md). Capex and gross profit are left out: capex differs by
# definition for 46 stocks, and gross profit is compared for too few to say much.
CROSSCHECK_FIELDS = ("total_assets", "operating_cash_flow", "equity", "net_income", "revenue")
CROSSCHECK_ATTENTION = 25          # 5% of the universe, about twice the noisiest field's count
# Market value against the filed share count has a higher resting level - splits after the
# filed quarter and partnership-unit structures put 22 stocks over 10% on 2026-10-14 - so its
# own line, about three times that. A market value served at the wrong scale for a whole sector
# or source would cross it the same morning.
MARKET_VALUE_ATTENTION = 60
# The SEC cache refreshes weekly (sec_fundamentals.FACTS_MAX_AGE_DAYS = 7). Twice that means the
# refresh has been failing, and every comparison above is being made against older and older
# filings without saying so.
CACHE_STALE_DAYS = 14


def check_top_of_ranking(payload: dict, res: Result) -> None:
    """A NOTE naming the top-25 stocks whose rank rests on something worth a second look
    (``ranking_audit.top_flags``, 2026-10-15): earnings lifted by non-operating income, equity
    thinned by buybacks, an input off the company's filing, statements a period behind. Never a
    refusal - each is the data as defined, said out loud."""
    try:
        import ranking_audit as RA
        flags = RA.top_flags(payload)
    except Exception as exc:  # noqa: BLE001
        res.note(f"top of the ranking not audited - {type(exc).__name__}: {exc}")
        return
    if not flags:
        res.note(RA.TOP_LINE + "nothing")
        return
    res.note(RA.TOP_LINE + "; ".join(f"{f['ticker']} (#{f['rank']}: {', '.join(f['why'])})" for f in flags))


def check_inputs_against_filings(payload: dict, res: Result, facts=None) -> None:
    """The run's inputs against the companies' own SEC figures - a NOTE, never a failure.

    **2026-10-14.** The dividend outage was a field vanishing, which check_metric_availability
    now catches. A field that changes scale or meaning while still being served passes every
    check in this file: coverage is unchanged, and renormalisation keeps dispersion. Rebuilt
    from the filings for the same quarter, total assets agreed with Yahoo within 0.5% for 99.8%
    of companies that day, so a field disagreeing for dozens of companies at once is a source
    event (sec_crosscheck.py; research/2026-10-14-yahoo-vs-sec.md).

    **A note, not a gate**, until the counts have a few weeks of history: a gate on a
    measurement nobody has watched yet jams the loop on its first odd morning, the same reason
    weight-0 metrics are notes above. It reads the weekly SEC cache and nothing else - no
    network - and anything going wrong here is reported and skipped, never raised.
    """
    sys.path.insert(0, str(ROOT))
    try:
        import sec_crosscheck as X
        if facts is None:
            from sec_fundamentals import FACTS_PATH
            if not FACTS_PATH.exists():
                res.note("inputs not checked against SEC filings - no companyfacts cache yet")
                return
            facts = pd.read_parquet(FACTS_PATH)
            import time as _time
            cache_days = (_time.time() - FACTS_PATH.stat().st_mtime) / 86400
        else:
            cache_days = None
        fields = [f for f in X.FIELDS if f[0] in CROSSCHECK_FIELDS]
        stats = X.summary(X.compare(payload, facts, fields))
    except Exception as exc:  # noqa: BLE001 - a note must never stop a publish
        res.note(f"inputs not checked against SEC filings - {type(exc).__name__}: {exc}")
        return
    parts = [f"{name.replace('_', ' ')} {s['within_half_pct']}/{s['compared']} within 0.5% "
             f"({s['over_10pct']} over 10%)" for name, s in stats.items()]
    try:
        mv = X.market_value_check(payload, facts)
        far = int(((mv["ratio"] - 1).abs() > X.FAR).sum())
        parts.append(f"market value / price vs filed shares {int(((mv['ratio'] - 1).abs() <= 0.05).sum())}/{len(mv)} "
                     f"within 5% ({far} over 10%)")
    except Exception as exc:  # noqa: BLE001
        far = 0
        parts.append(f"market value not checked ({type(exc).__name__})")
    try:
        # scored a period behind a filing the company made more than 30 days before the run - the
        # source's lag, not the company's (2026-10-15: 16 stocks incl. AMZN, BAC, BA)
        import pandas as _pd
        _run = _pd.Timestamp(str((payload.get("kpis") or {}).get("run_timestamp") or _pd.Timestamp.today())[:10])
        _late = sorted(t for t, nf in X.newer_filings(payload, facts).items()
                       if (_run - _pd.Timestamp(nf["filed"])).days > 30)
        parts.append(f"{len(_late)} stocks scored a period behind a filing over 30 days old"
                     + (f" ({', '.join(_late[:8])}{'...' if len(_late) > 8 else ''})" if _late else ""))
    except Exception as exc:  # noqa: BLE001
        parts.append(f"filing lag not checked ({type(exc).__name__})")
    if cache_days is not None:
        parts.append(f"SEC cache {cache_days:.0f} days old")
    res.note("inputs against SEC filings: " + "; ".join(parts))
    if cache_days is not None and cache_days > CACHE_STALE_DAYS:
        res.note(f"ATTENTION - the SEC companyfacts cache is {cache_days:.0f} days old (it refreshes weekly; the "
                 f"line is {CACHE_STALE_DAYS}). Its refresh has probably been failing - see "
                 f"sec_fundamentals.refresh_companyfacts in the run log. The comparisons above are against "
                 f"older filings than they should be.")
    if far > MARKET_VALUE_ATTENTION:
        res.note(f"ATTENTION - market value implies a share count more than 10% from the filed one for {far} "
                 f"stocks (the line is {MARKET_VALUE_ATTENTION}; 2026-10-14 had 22, mostly splits and "
                 f"partnership units). Every valuation ratio rests on market value. Not a refusal - see "
                 f"research/2026-10-14-yahoo-vs-sec.md.")
    for name, s in stats.items():
        if s["over_10pct"] > CROSSCHECK_ATTENTION:
            res.note(f"ATTENTION - {name.replace('_', ' ')} differs from the companies' filings by more "
                     f"than 10% for {s['over_10pct']} stocks (the line is {CROSSCHECK_ATTENTION}; 2026-10-14 "
                     f"had 0-13 per field). A source changing a field's scale or meaning looks like this. "
                     f"Not a refusal - see research/2026-10-14-yahoo-vs-sec.md.")


def _weighted_metric_names() -> set[str] | None:
    """Every metric with weight in any profile, or None if that cannot be read.

    ``factor_engine.weighted_metric_sets`` is the one place weights are resolved
    (CLAUDE.md row 0.8); None rather than an empty set, so "cannot tell" cannot
    read as "nothing carries weight".
    """
    try:
        import factor_engine as F
        gen, bank = F.weighted_metric_sets(F.load_config())
        return set(gen) | set(bank)
    except Exception as exc:  # noqa: BLE001 - a gate reports, it does not raise
        print(f"  (could not read the weights: {type(exc).__name__}: {exc})")
        return None


def check_dispersion(payload: dict, res: Result) -> None:
    """Compare this run's category dispersion to the trailing median.

    This is the check that would have caught 2026-08-07 automatically.
    """
    rows = payload.get("table_data") or []
    if not rows:
        res.fail("table_data is empty")
        return

    current: dict[str, float] = {}
    for cat in CATEGORIES:
        vals = [r[f"{cat}_score"] for r in rows if r.get(f"{cat}_score") is not None]
        if len(vals) < 2:
            res.fail(f"{cat}: fewer than 2 scored stocks")
            return
        current[cat] = statistics.pstdev(vals)

    if not DISPERSION_HISTORY.exists():
        res.note("no dispersion history yet - skipping regression check")
        return

    hist = pd.read_csv(DISPERSION_HISTORY, header=None)
    # Columns: run_date, then one dispersion value per category in CATEGORIES order.
    if hist.shape[1] < len(CATEGORIES) + 1 or len(hist) < MIN_HISTORY_FOR_CHECK:
        res.note(
            f"dispersion history too short ({len(hist)} rows) - "
            f"skipping regression check"
        )
        return

    # Exclude any row written by this run (same date, appended before this check).
    baseline = hist.iloc[:-1] if len(hist) > MIN_HISTORY_FOR_CHECK else hist

    for i, cat in enumerate(CATEGORIES, start=1):
        try:
            prior = pd.to_numeric(baseline[i], errors="coerce").dropna().tolist()
        except KeyError:
            continue
        if len(prior) < MIN_HISTORY_FOR_CHECK:
            continue
        median = statistics.median(prior[-10:])
        if median <= 0:
            continue
        drop = (median - current[cat]) / median
        if drop > MAX_DISPERSION_DROP:
            res.fail(
                f"{cat} dispersion collapsed: {current[cat]:.1f} vs trailing "
                f"median {median:.1f} ({drop:.0%} drop, limit "
                f"{MAX_DISPERSION_DROP:.0%}). Data is far less differentiated "
                f"than usual - the ranking cannot be trusted."
            )
        else:
            res.note(f"{cat} dispersion {current[cat]:.1f} vs median {median:.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir or latest_run_dir()
    if run_dir is None or not run_dir.exists():
        print("HEALTH: cannot evaluate - no run directory found")
        return 2

    res = Result()
    check_fetch_happened(run_dir, res)

    try:
        payload = load_payload(run_dir)
    except ValueError as exc:
        print(f"HEALTH: cannot evaluate - {exc}")
        return 2

    if payload is None:
        print("HEALTH: cannot evaluate - no dashboard_data.js found")
        return 2

    check_coverage(payload, res)
    check_metric_availability(res)
    check_dispersion(payload, res)
    check_inputs_against_filings(payload, res)
    check_top_of_ranking(payload, res)

    if not args.quiet:
        print(f"HEALTH CHECK for run {run_dir.name}")
        for n in res.notes:
            print(f"  ok   {n}")
        for f in res.failures:
            print(f"  FAIL {f}")

    if res.healthy:
        print("HEALTH: PASS - safe to publish")
        return 0

    print(f"HEALTH: FAIL - {len(res.failures)} problem(s). Do not publish.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
