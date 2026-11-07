"""Analyst-target coverage inside size bands, on the published S&P 500 run.

WHY. `scripts/check_run_health.check_target_coverage_by_band` applies
`MIN_TARGET_COVERAGE` (0.50) inside each size band instead of to the universe mean
(universe-expansion phase 1, `plan/universe-expansion.md`). Two things need numbers before
that gate is trusted, and neither is in the plan:

1. **It cannot fire spuriously on the universe we publish.** Phase 4 bands the S&P 500
   itself by NYSE breakpoints, so the 500 will one day be split across bands. If the
   smallest band of today's index sat near 0.50, the gate would start failing good runs the
   day the split arrives.
2. **It is not vacuous.** With one band the check is the universe-wide check, and 497/499
   passes either way. So the band path has to be exercised against a split where a band
   genuinely fails while the universe passes - the same "the zero means something" standard
   `tests/test_size_bands.py` holds the percentile identity to.

Inputs: the newest run's own `00_raw_fetch.parquet` (marketCap) and its `dashboard_data.js`
(`stock_detail[t].pt_mean`). No network, no clock, nothing published is touched.

Run: `python research/measurements/2026-11-07-target-coverage-by-band.py`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import check_run_health as h  # noqa: E402
from run_context import latest_run_dir  # noqa: E402


def main() -> int:
    run_dir = latest_run_dir(ROOT / "runs")
    if run_dir is None:
        print("MISSING DATA: no run directory")
        return 3
    payload = h.load_payload(run_dir)
    if payload is None:
        print(f"MISSING DATA: {run_dir.name} has no payload of its own")
        return 3
    detail = payload["stock_detail"]

    raw = pd.read_parquet(run_dir / "00_raw_fetch.parquet", columns=["Ticker", "marketCap"])
    mc = {str(t): v for t, v in zip(raw["Ticker"], raw["marketCap"]) if pd.notna(v)}
    scored = [t for t in detail if t in mc]
    scored.sort(key=lambda t: mc[t])

    print(f"run {run_dir.name}: {len(detail)} scored, {len(scored)} with a market cap")
    n = len(scored)
    targeted = sum(1 for t in detail if detail[t].get("pt_mean"))
    print(f"universe-wide analyst-target coverage: {targeted}/{len(detail)} "
          f"({targeted / len(detail):.1%}); floor {h.MIN_TARGET_COVERAGE:.0%}")

    # --- 1. deciles of the index's own market cap -----------------------------------------
    print("\n(1) coverage by market-cap decile of today's S&P 500 (smallest first):")
    worst = 1.0
    for d in range(10):
        lo, hi = n * d // 10, n * (d + 1) // 10
        names = scored[lo:hi]
        got = sum(1 for t in names if detail[t].get("pt_mean"))
        cov = got / len(names)
        worst = min(worst, cov)
        print(f"  decile {d + 1:>2}  n={len(names):>3}  cap "
              f"${mc[names[0]] / 1e9:>8.1f}B-${mc[names[-1]] / 1e9:>8.1f}B  "
              f"targets {got:>3}/{len(names):<3} {cov:>6.1%}")
    print(f"  -> thinnest decile {worst:.1%}, i.e. {worst / h.MIN_TARGET_COVERAGE:.2f}x the "
          f"{h.MIN_TARGET_COVERAGE:.0%} floor. The gate cannot fire on this universe however "
          f"it is banded.")

    # --- 2. the gate itself, on a three-way split and on a split built to fail ------------
    def verdict(bands: dict[str, str]) -> tuple[list[str], list[str]]:
        res = h.Result()
        h.check_target_coverage_by_band(payload, bands, res)
        return res.failures, res.notes

    thirds = {t: ("small", "mid", "large")[min(2, i * 3 // n)]
              for i, t in enumerate(scored)}
    fails, notes = verdict(thirds)
    print(f"\n(2) the real gate on a three-way cap split: {len(fails)} failure(s)")
    for line in notes + fails:
        print("   ", line)

    # Built to fail: the 40 smallest stocks become their own band and lose their targets.
    # The universe keeps >= 50%, so only the per-band check can see it. This is the
    # non-vacuousness check - nothing about today's data, everything about the rule.
    thin = dict(thirds)
    stripped = scored[:40]
    for t in stripped:
        thin[t] = "thin"
    probe = json.loads(json.dumps(payload))
    for t in stripped:
        probe["stock_detail"][t]["pt_mean"] = None
    res = h.Result()
    h.check_coverage(probe, res, bands=thin)
    uni = [f for f in res.failures if "band" not in f]
    band = [f for f in res.failures if "band" in f]
    print(f"\n(3) 40 smallest stocks stripped of targets and made their own band:")
    print(f"    universe-wide check: {len(uni)} failure(s) "
          f"({(targeted - 40) / len(detail):.1%} still clears {h.MIN_TARGET_COVERAGE:.0%})")
    print(f"    per-band check:      {len(band)} failure(s)")
    for f in band:
        print("     ", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
