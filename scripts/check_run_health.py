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

* **2026-10-22** - a trading bot now acts on what this loop publishes, so three checks compare
  the run with the ranking the site last published (rank continuity), count the stocks that went
  unscored for missing data, and date every price against the NYSE session it should come from.
  See the block above ``main``.

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))   # run_context, history, factor_engine live at the root
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
        # An artifact the run itself should have written is absent or belongs to another run:
        # exit 2, "could not evaluate", which data-run.ps1 treats as a discard. Kept apart from
        # `failures` (exit 1, the run is degraded) so the log says which it was.
        self.unevaluable: list[str] = []

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    def cannot_evaluate(self, msg: str) -> None:
        self.unevaluable.append(msg)

    @property
    def healthy(self) -> bool:
        return not self.failures


def latest_run_dir() -> Path | None:
    """The run this gate judges: ``run_context.latest_run_dir``, keyed on ``meta.json``
    ``start_time``.

    Until 2026-10-16 this picked the newest folder by **filesystem mtime** while
    ``generate_dashboard._find_latest_run`` - the generator ``data-run.ps1`` runs straight after
    this gate - picked by ``start_time``. OneDrive rewrites folder mtimes when it syncs, so the
    two could name different runs: the gate approving one and the generator publishing another.
    One function now answers for both.
    """
    from run_context import latest_run_dir as _shared
    return _shared(RUNS_DIR)


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
    """The payload **this run** wrote, from its own folder - or None.

    **No fallback to the project root** (removed 2026-10-16). The root ``dashboard_data.js`` is
    whatever was last published. The run folder lacks a payload in exactly one case - step 12 of
    ``run_screener.py`` (dashboard generation) failed, which the screener reports only as a
    WARNING - and in that case the fallback had this gate judging *yesterday's* published page
    and approving it, after which ``data-run.ps1`` went on to regenerate and publish a page the
    gate had never seen. A missing run payload is a missing artifact of the run itself: the
    caller exits 2, which ``data-run.ps1`` treats as a discard.
    """
    path = run_dir / "dashboard_data.js"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text[text.index("{"):].rstrip().rstrip(";"))
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"dashboard_data.js did not parse: {exc}") from exc


def _run_meta(run_dir: Path) -> dict:
    try:
        return json.loads((run_dir / "meta.json").read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return {}


def payload_run_mismatch(payload: dict, run_dir: Path) -> str | None:
    """Why the payload is not this run's, or None if it is.

    ``kpis.run_timestamp`` is copied verbatim from ``meta.json`` ``start_time`` by the
    generator, so a payload from any other run - a stale copy, a file left by an earlier run
    under the same folder name - cannot carry this run's stamp.
    """
    started = _run_meta(run_dir).get("start_time")
    if not started:
        return f"run {run_dir.name} has no meta.json start_time to match the payload against"
    stamp = (payload.get("kpis") or {}).get("run_timestamp")
    if stamp != started:
        return (f"the payload in {run_dir.name} is not this run's: kpis.run_timestamp is "
                f"{stamp!r}, the run's meta.json start_time is {started!r}")
    return None


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


def _snapshot_expected(run_dir: Path) -> str | None:
    """None if this run should have recorded a snapshot, else the reason it deliberately did not.

    Mirrors ``run_screener.py`` step 13: no snapshot when the run's own config snapshot sets
    ``improvement.enabled: false``, or for a ``--tickers`` subset run (``meta.json``
    ``cli_flags.tickers``) - recording a subset would replace the universe's observation.
    """
    if ((_run_meta(run_dir).get("cli_flags") or {}).get("tickers")):
        return "a --tickers subset run records no snapshot"
    try:
        import yaml
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
        if (cfg.get("improvement") or {}).get("enabled", True) is False:
            return "the run's config sets improvement.enabled: false, so it records no snapshot"
    except (OSError, ValueError, ImportError, AttributeError):
        pass
    return None


def _newest_snapshot_is_this_runs(run_dir: Path, index: dict, res: Result) -> bool:
    """True if the newest snapshot was written by this run; otherwise records why and is False.

    **2026-10-16.** The comparison below takes the two newest snapshots and calls the newer one
    "this run". ``record_run_snapshot`` failing is only a WARNING in ``run_screener.py`` (step
    13), and then the two newest are *yesterday and the day before*: the gate compared two old
    runs, said PASS about this one, and ``check_dispersion`` - which drops the history's last row
    as this run's - quietly measured against a baseline missing its newest day. A snapshot
    this run should have written and did not is a missing artifact of the run itself, so it is
    ``cannot_evaluate`` (exit 2, a discard), the same as a missing payload.

    The file is ``improvement/snapshots/{run_date}_{run_id}.parquet``; both halves must match.
    The run date is the date of the run's start or end (step 13 stamps ``datetime.now()``).
    """
    why_not = _snapshot_expected(run_dir)
    if why_not:
        res.note(f"metric availability not checked - {why_not}")
        return False
    meta = _run_meta(run_dir)
    run_id = str(meta.get("run_id") or run_dir.name)
    run_dates = {str(meta.get(k) or "")[:10] for k in ("start_time", "end_time")} - {""}
    if not index:
        res.cannot_evaluate(f"no improvement snapshot at all - run {run_id} should have "
                            f"written one at step 13 (record_run_snapshot)")
        return False
    newest = max(index)
    path = Path(index[newest])
    if newest not in run_dates or path.stem != f"{newest}_{run_id}":
        res.cannot_evaluate(
            f"the newest improvement snapshot is {path.name}, not this run's "
            f"({'/'.join(sorted(run_dates)) or '?'}_{run_id}.parquet) - record_run_snapshot "
            f"failed or wrote elsewhere, so this run's metric availability cannot be compared")
        return False
    return True


def check_metric_availability(res: Result, run_dir: Path | None = None) -> None:
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

    Given ``run_dir`` (``main`` always passes it), the newest snapshot must be **this run's**
    (``_newest_snapshot_is_this_runs``) or the run cannot be evaluated.
    """
    sys.path.insert(0, str(ROOT))
    try:
        import history as H
    except ImportError as exc:
        res.note(f"metric availability not checked - history unavailable ({exc})")
        return

    index = H.snapshot_index()
    dates = sorted(index)
    if run_dir is not None and not _newest_snapshot_is_this_runs(run_dir, index, res):
        return
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


# =========================================================================== 2026-10-22
# The trading bot's inputs (OWNER_FOCUS 2026-10-21, items 1-3; screener-bot ROADMAP 1.3-1.5).
#
# Every check above asks whether *this run* is internally sound. None asked whether it agrees
# with the ranking the site already published - and since 2026-10-09 a bot trades whatever this
# loop publishes. Three ways a run can be internally sound and still wrong to trade:
#
#   1. it is scrambled - 2026-07-28 published a ranking with Spearman 0.016 against the run
#      before it, 24 of 25 top names new, and every check of its day passed;
#   2. it scored a shrunken universe - 2026-10-22's meta.json shows 2 stocks unscored while the
#      fetch log says 0 failures, because a record that arrives *empty* is dropped by the
#      coverage filter, which nothing counted;
#   3. its prices are old - nothing recorded which session a price belongs to, so yesterday's
#      prices re-served under today's date would have passed the price-coverage check.
#
# Thresholds measured against every published version of dashboard_data.js since 2026-08-21:
# research/measurements/2026-10-22-input-gates-replay.py and its JSON. Read them before moving
# a number - each was set so that no ordinary day in that history would have failed.

RANK_MIN_SPEARMAN = 0.90      # ordinary days measured >= 0.981; the scrambled 07-28 run 0.016
RANK_TOP_N = 25
RANK_MAX_NEW_IN_TOP = 8       # ordinary days measured <= 4 new names; 07-28 had 24
RANK_FALL_LINE = 40
RANK_MAX_FALLEN = 3           # top-25 names now ranked below #40 (or no longer ranked at all)

MIN_SCORED = 490              # 498-503 on every published run; 500 companies after share classes
MAX_EXCLUDED_SHARE = 0.02     # of the universe, unscored for missing data (11+ of 503 fails)

MIN_PRICE_ON_SESSION = 0.95   # of stocks whose price is dated on the expected NYSE session
MAX_UNCHANGED_ACROSS_SESSION = 0.50

CHANGELOG = ROOT / "METHODOLOGY_CHANGELOG.md"
ET_ZONE = "America/New_York"


def _parse_payload_text(text: str) -> dict:
    return json.loads(text[text.index("{"):].rstrip().rstrip(";"))


def published_payload(ref: str = "HEAD") -> dict | None:
    """The ranking the site last published: ``dashboard_data.js`` as committed at ``ref``.

    **Not the working-tree file.** ``run_screener.py`` step 12 copies the run's own payload over
    the root ``dashboard_data.js`` before this gate runs, so the file on disk is the run being
    judged. The committed one is what GitHub Pages serves and what the bot last read; in
    ``data-run.ps1`` HEAD is ``main`` as just pulled (and since 2026-10-22 a failed pull stops the
    run, so HEAD cannot be a stale ``main``). None when git cannot produce it.
    """
    import subprocess
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "show", f"{ref}:dashboard_data.js"],
                           capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0 or not r.stdout:
        return None
    try:
        return _parse_payload_text(r.stdout.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


def _ranks(payload: dict) -> dict[str, float]:
    out = {}
    for r in payload.get("table_data") or []:
        t, k = r.get("Ticker"), r.get("Rank")
        if t is not None and isinstance(k, (int, float)):
            out[str(t)] = float(k)
    return out


def _spearman(a: list[float], b: list[float]) -> float:
    ra = pd.Series(a).rank(method="average")
    rb = pd.Series(b).rank(method="average")
    return float(ra.corr(rb))


def rank_continuity(current: dict, baseline: dict) -> dict:
    """How far ``current``'s ranking is from ``baseline``'s - the three numbers the gate reads.

    ``spearman`` is over the names ranked in both. ``new_in_top`` are names in the current top
    ``RANK_TOP_N`` that were not in the baseline's. ``fallen`` are baseline top-``RANK_TOP_N``
    names now ranked below ``RANK_FALL_LINE`` **or not ranked at all** - a top name that
    vanished is the same event for anything holding it.
    """
    cur, base = _ranks(current), _ranks(baseline)
    common = sorted(set(cur) & set(base))
    rho = _spearman([cur[t] for t in common], [base[t] for t in common]) if len(common) >= 3 else None
    top_cur = {t for t, k in cur.items() if k <= RANK_TOP_N}
    top_base = {t for t, k in base.items() if k <= RANK_TOP_N}
    fallen = sorted(t for t in top_base if cur.get(t, float("inf")) > RANK_FALL_LINE)
    return {"spearman": rho, "n_common": len(common), "new_in_top": sorted(top_cur - top_base),
            "fallen": fallen}


def methodology_version(payload: dict) -> str | None:
    """``kpis.methodology_version``: ``RunContext.config_hash`` of the config the run scored with.

    Chosen 2026-10-22 because it already is this project's definition of "a different scoring":
    it is the key the scored-table cache is stored under, it covers every weight, the
    data-quality and universe rules, sector neutrality, the trap filters, the momentum regime
    rule and the share-class declaration, and its ``scoring_schema`` counter is bumped by every
    scoring-code change that a cache must not hide (15 bumps to 2026-10-21, five of them on the
    2026-10-09 overhaul). It is conservative - a non-scoring column change bumps it too - which
    can only make the override below *available*; it never fails a run.
    """
    v = (payload.get("kpis") or {}).get("methodology_version")
    return str(v) if v else None


def changelog_dates(text: str) -> set[str]:
    """The dates of ``METHODOLOGY_CHANGELOG.md``'s entries - its ``## YYYY-MM-DD`` headings."""
    import re
    return set(re.findall(r"^## (\d{4}-\d{2}-\d{2})\b", text or "", flags=re.M))


def _read_changelog() -> str:
    try:
        return CHANGELOG.read_text(encoding="utf-8")
    except OSError:
        return ""


def continuity_override(payload: dict, baseline: dict, changelog_text: str) -> str | None:
    """Why a broken continuity is a documented methodology change, or None if it is not.

    Both must hold: the methodology version differs from the published run's, **and**
    ``METHODOLOGY_CHANGELOG.md`` has an entry dated between the published run's date and this
    run's date, inclusive. The owner's wording was "an entry dated today"; the window starts at
    the published run instead because a session that changes the methodology at 06:00 and does
    not republish leaves the *next day's* 02:00 run as the first one scored under it - with
    "today" only, that run fails, the published ranking never moves, and every later run is
    compared with the same stale page and fails too: a gate that can only fail closed, which
    jams the loop (CLAUDE.md row 0.5/0.7). The window admits exactly the entries written since
    the ranking being replaced was published. Measured, not hypothetical: 6 of the 46 data-loop
    runs from 2026-08-21 to 2026-10-22 were the first scored under a change whose entry is dated
    the day before (09-02, 09-03, 09-11, 10-13, 10-20, 10-22); none needed the waiver, but under
    "today" only not one of them could have had it.
    """
    cur_v, base_v = methodology_version(payload), methodology_version(baseline)
    if not cur_v or not base_v or cur_v == base_v:
        return None
    run_day = str((payload.get("kpis") or {}).get("run_timestamp") or "")[:10]
    base_day = str((baseline.get("kpis") or {}).get("run_timestamp") or "")[:10]
    if not run_day or not base_day:
        return None
    dated = sorted(d for d in changelog_dates(changelog_text) if base_day <= d <= run_day)
    if not dated:
        return None
    return (f"the methodology version changed ({base_v} -> {cur_v}) and METHODOLOGY_CHANGELOG.md "
            f"has an entry dated {', '.join(dated)}")


def check_rank_continuity(payload: dict, baseline: dict | None, res: Result,
                          changelog_text: str | None = None, run_dir: Path | None = None) -> None:
    """FAIL when this run's ranking breaks from the last published one (bot ROADMAP 1.3).

    Below Spearman ``RANK_MIN_SPEARMAN`` over the names ranked in both, more than
    ``RANK_MAX_NEW_IN_TOP`` new names in the top ``RANK_TOP_N``, or more than ``RANK_MAX_FALLEN``
    of the published top ``RANK_TOP_N`` now below #``RANK_FALL_LINE``. Waived only by
    ``continuity_override``. No published ranking to compare with is a NOTE: it is the
    environment failing, not the run, and a gate that cannot read its baseline must not jam the
    loop (the node/pytest fallback precedent).

    **Replayed 2026-10-22** over the 62 versions the site published from 2026-08-21 to 2026-10-22
    (46 of them data-loop runs): it fails none of them. Ordinary data-loop days measured Spearman
    >= 0.981, <= 4 new top-25 names, <= 1 fallen; the 2026-10-09 overhaul (owner-run, published
    by merge) 0.935, 6 new, 1 fallen - inside the limits outright, and the override was available
    too (its version changed; the changelog has 2026-10-09 entries). It fails the known-bad pages:
    2026-07-28 (0.856, 10 new, 7 fallen), 2026-08-06's synthetic run and 2026-08-07's price-less
    one (-0.017, 22 new, 23 fallen).

    **Two limits, both measured.** (1) A refused run leaves the published ranking standing, and
    every later run is compared with it: ordinary drift stays inside every limit for 10 runs (two
    weeks; worst case exactly 8 new and 3 fallen) and crosses the new-names limit at 20, so a
    refusal that nobody resolves turns into a stop within about a month. (2) The baseline is
    whatever ``main`` serves, and a code session's full run publishes without this gate - so a bad
    page published that way would make every good run after it fail (the replay shows the
    2026-08-10 manual repair failing against the broken 08-07 page). OWNER_FOCUS item 5 (only the
    data loop's run is tradable, ``signals.json``) is the place to move the baseline to the last
    *data-loop* run. Until then both cases are a discarded run with this check's reason in the
    log and the morning brief - visible, never silent; the site keeps the page it last published.
    """
    if run_dir is not None and (_run_meta(run_dir).get("cli_flags") or {}).get("tickers"):
        res.note("rank continuity not checked - a --tickers subset run is never published")
        return
    if baseline is None:
        res.note("rank continuity NOT checked - the last published dashboard_data.js could not be "
                 "read from git")
        return
    c = rank_continuity(payload, baseline)
    if c["spearman"] is None:
        res.note(f"rank continuity NOT checked - only {c['n_common']} names ranked in both runs")
        return
    base_when = str((baseline.get("kpis") or {}).get("run_timestamp") or "?")[:16]
    summary = (f"rank continuity vs the published run of {base_when}: Spearman {c['spearman']:.3f} "
               f"over {c['n_common']} names, {len(c['new_in_top'])} new in the top {RANK_TOP_N}, "
               f"{len(c['fallen'])} of the published top {RANK_TOP_N} below #{RANK_FALL_LINE}")
    broken = []
    if c["spearman"] < RANK_MIN_SPEARMAN:
        broken.append(f"Spearman {c['spearman']:.3f} < {RANK_MIN_SPEARMAN}")
    if len(c["new_in_top"]) > RANK_MAX_NEW_IN_TOP:
        broken.append(f"{len(c['new_in_top'])} new top-{RANK_TOP_N} names > {RANK_MAX_NEW_IN_TOP} "
                      f"({', '.join(c['new_in_top'][:12])})")
    if len(c["fallen"]) > RANK_MAX_FALLEN:
        broken.append(f"{len(c['fallen'])} published top-{RANK_TOP_N} names below "
                      f"#{RANK_FALL_LINE} > {RANK_MAX_FALLEN} ({', '.join(c['fallen'][:12])})")
    if not broken:
        res.note(summary)
        return
    why = continuity_override(payload, baseline,
                              _read_changelog() if changelog_text is None else changelog_text)
    if why:
        res.note(f"{summary} - outside the continuity limits ({'; '.join(broken)}), allowed "
                 f"because {why}")
        return
    res.fail(f"{summary}. The ranking broke from the published one ({'; '.join(broken)}) with no "
             f"documented methodology change - a scrambled or mis-joined run looks like this, and "
             f"the trading bot would act on it.")


def check_scored_universe(payload: dict, run_dir: Path, res: Result) -> None:
    """FAIL below ``MIN_SCORED`` ranked, or more than ``MAX_EXCLUDED_SHARE`` of the universe
    unscored for missing data (bot ROADMAP 1.4).

    The unscored count is ``universe.json``'s ``failed``: fetch errors, records that came back
    empty, and stocks the coverage filter dropped. The last two are logged as ``missing_metric``,
    not ``fetch_failure``, so until 2026-10-22 the data loop's 40% gate never counted them -
    2026-10-22 had 2 unscored and a fetch log reading 0 failures. A ``--tickers`` subset run is
    exempt by design.
    """
    meta = _run_meta(run_dir)
    if (meta.get("cli_flags") or {}).get("tickers"):
        res.note("scored-universe floor not checked - a --tickers subset run")
        return
    scored = len(payload.get("table_data") or [])
    if scored < MIN_SCORED:
        res.fail(f"only {scored} stocks are ranked (need >= {MIN_SCORED}) - the universe shrank, "
                 f"and the bot would be choosing from a partial list")
    try:
        u = json.loads((run_dir / "universe.json").read_text(encoding="utf-8"))
        failed = list(u.get("failed") or [])
        universe = int(u.get("scored_count", len(u.get("scored") or []))) + len(failed)
    except (OSError, ValueError, TypeError):
        res.cannot_evaluate(f"run {run_dir.name} has no readable universe.json, so how many stocks "
                            f"went unscored cannot be counted")
        return
    share = len(failed) / universe if universe else 1.0
    line = (f"{scored} ranked; {len(failed)} of {universe} unscored for missing data "
            f"({share:.1%}, limit {MAX_EXCLUDED_SHARE:.0%})"
            + (f": {', '.join(failed[:12])}{'...' if len(failed) > 12 else ''}" if failed else ""))
    if share > MAX_EXCLUDED_SHARE:
        res.fail(line + " - records that arrive empty or thin are dropped silently; this many is "
                        "a source event")
    elif scored >= MIN_SCORED:
        res.note(line)


# --------------------------------------------------------------------------- NYSE sessions
def _easter(year: int):
    """Gregorian Easter Sunday (Anonymous Gregorian / Meeus-Jones-Butcher algorithm)."""
    from datetime import date
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def nyse_holidays(year: int) -> set:
    """The NYSE's full-day holidays by its published rules (Rule 7.2 and the holiday schedule).

    A Saturday holiday is observed the Friday before, a Sunday one the Monday after - except New
    Year's Day on a Saturday, which the exchange does not observe on December 31. Unscheduled
    closures (a national day of mourning, 2025-01-09) are not rules and are not here: on such a
    day this gate expects a session that did not happen and discards that one run, and the next
    run expects a real session again. A one-day discard, never a jam.
    """
    from datetime import date, timedelta

    def nth(month, weekday, n):          # n-th weekday (Mon=0) of a month
        d = date(year, month, 1)
        d += timedelta(days=(weekday - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)

    def last(month, weekday):
        d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
        return d - timedelta(days=(d.weekday() - weekday) % 7)

    def observed(d):
        return d - timedelta(days=1) if d.weekday() == 5 else d + timedelta(days=1) if d.weekday() == 6 else d

    out = {
        nth(1, 0, 3),                                  # Martin Luther King Jr. Day
        nth(2, 0, 3),                                  # Washington's Birthday
        _easter(year) - timedelta(days=2),             # Good Friday
        last(5, 0),                                    # Memorial Day
        observed(date(year, 7, 4)),                    # Independence Day
        nth(9, 0, 1),                                  # Labor Day
        nth(11, 3, 4),                                 # Thanksgiving
        observed(date(year, 12, 25)),                  # Christmas
    }
    ny = date(year, 1, 1)
    if ny.weekday() != 5:
        out.add(observed(ny))
    if year >= 2022:
        out.add(observed(date(year, 6, 19)))           # Juneteenth
    return out


def is_nyse_session(d) -> bool:
    return d.weekday() < 5 and d not in nyse_holidays(d.year)


def _to_eastern(at):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    if isinstance(at, str):
        at = datetime.fromisoformat(at)
    # A naive stamp is the run machine's local time (run_context writes datetime.now()), which
    # astimezone() reads as local.
    return at.astimezone(ZoneInfo(ET_ZONE))


def expected_session(at):
    """The NYSE session a price fetched at ``at`` should come from: the latest session that had
    opened (09:30 Eastern) by then. At 02:00 Central that is the previous trading day; a
    catch-up run after the open is today, intraday."""
    from datetime import time, timedelta
    et = _to_eastern(at)
    d = et.date()
    if not (is_nyse_session(d) and et.time() >= time(9, 30)):
        d -= timedelta(days=1)
        while not is_nyse_session(d):
            d -= timedelta(days=1)
    return d


def check_price_freshness(payload: dict, baseline: dict | None, res: Result,
                          run_dir: Path | None = None) -> None:
    """FAIL unless >= ``MIN_PRICE_ON_SESSION`` of prices are dated on the expected NYSE session,
    or if more than ``MAX_UNCHANGED_ACROSS_SESSION`` of prices are unchanged from the published
    run although a session has closed in between (bot ROADMAP 1.5).

    ``stock_detail[t].price_date`` is the session of the price the page publishes (the quote's
    ``regularMarketTime``, or the last daily bar when the quote is missing;
    ``factor_engine.published_price_date``). The expected session is taken at the run's start
    *and* its end, so a fetch that straddles an open is not refused for it. A stock with no
    price or no date counts against the share - it cannot be shown to be fresh.
    """
    detail = payload.get("stock_detail") or {}
    if not detail:
        return
    meta = _run_meta(run_dir) if run_dir is not None else {}
    stamps = [s for s in (meta.get("start_time"), meta.get("end_time"),
                          (payload.get("kpis") or {}).get("run_timestamp")) if s]
    if not stamps:
        res.note("price freshness NOT checked - the run has no timestamp")
        return
    sessions = {expected_session(s) for s in stamps[:2]}
    want = {d.isoformat() for d in sessions}
    from collections import Counter
    dates = Counter(str(v.get("price_date")) if v.get("price") and v.get("price_date") else "none"
                    for v in detail.values())
    on = sum(n for d, n in dates.items() if d in want)
    share = on / len(detail)
    other = ", ".join(f"{d} x{n}" for d, n in dates.most_common() if d not in want)
    line = (f"price dates: {on}/{len(detail)} on the expected session {'/'.join(sorted(want))} "
            f"({share:.1%}, need >= {MIN_PRICE_ON_SESSION:.0%})" + (f"; others: {other}" if other else ""))
    if share < MIN_PRICE_ON_SESSION:
        res.fail(line + " - the prices are not the latest session's, and the bot sizes orders "
                        "and checks fills against them")
    else:
        res.note(line)

    if baseline is None:
        return
    base_ts = (baseline.get("kpis") or {}).get("run_timestamp")
    if not base_ts:
        return
    base_session = expected_session(base_ts)
    if base_session >= min(sessions):
        res.note(f"prices not compared with the published run - no session has closed since it "
                 f"({base_session.isoformat()})")
        return
    bdet = baseline.get("stock_detail") or {}
    pairs = [(v.get("price"), (bdet.get(t) or {}).get("price")) for t, v in detail.items()]
    pairs = [(a, b) for a, b in pairs if isinstance(a, (int, float)) and isinstance(b, (int, float))]
    if not pairs:
        return
    same = sum(1 for a, b in pairs if abs(a - b) < 1e-9)
    s_share = same / len(pairs)
    line = (f"{same}/{len(pairs)} prices unchanged from the published run although the "
            f"{base_session.isoformat()} session has been followed by "
            f"{'/'.join(sorted(want))} ({s_share:.1%}, limit {MAX_UNCHANGED_ACROSS_SESSION:.0%})")
    if s_share > MAX_UNCHANGED_ACROSS_SESSION:
        res.fail(line + " - yesterday's prices served again")
    else:
        res.note(line)


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
        print(f"HEALTH: cannot evaluate - run {run_dir.name} has no dashboard_data.js of its own "
              f"(run_screener.py step 12, dashboard generation, failed or was skipped). The root "
              f"payload is the last published run, not this one, so it is not judged instead.")
        return 2

    mismatch = payload_run_mismatch(payload, run_dir)
    if mismatch:
        print(f"HEALTH: cannot evaluate - {mismatch}")
        return 2

    check_coverage(payload, res)
    check_metric_availability(res, run_dir)
    check_dispersion(payload, res)
    # The trading bot's inputs (2026-10-22): against the ranking the site last published.
    published = published_payload()
    check_rank_continuity(payload, published, res, run_dir=run_dir)
    check_scored_universe(payload, run_dir, res)
    check_price_freshness(payload, published, res, run_dir)
    check_inputs_against_filings(payload, res)
    check_top_of_ranking(payload, res)

    if not args.quiet:
        print(f"HEALTH CHECK for run {run_dir.name}")
        for n in res.notes:
            print(f"  ok   {n}")
        for f in res.failures:
            print(f"  FAIL {f}")
        for u in res.unevaluable:
            print(f"  MISSING {u}")

    if res.failures:
        print(f"HEALTH: FAIL - {len(res.failures)} problem(s). Do not publish.")
        return 1

    if res.unevaluable:
        print(f"HEALTH: cannot evaluate - {len(res.unevaluable)} artifact(s) of this run "
              f"missing: {'; '.join(res.unevaluable)}")
        return 2

    print("HEALTH: PASS - safe to publish")
    return 0


if __name__ == "__main__":
    sys.exit(main())
