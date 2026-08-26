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
    check_dispersion(payload, res)

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
