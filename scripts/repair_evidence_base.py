#!/usr/bin/env python3
"""One-time repair of the improvement-engine evidence base (2026-08-24).

`performance_history.csv` accumulated duplicate (run_date, ticker) rows and
weekend run dates, and `live_ic_history.csv` was computed from them. Both
defects inflate the observation count the improvement engine gates on. The
engine now normalizes on every read and write, so this script only has to fix
what is already on disk; after it runs, the ordinary data loop keeps both files
correct on its own.

Idempotent - running it twice changes nothing the second time.

Usage:  python scripts/repair_evidence_base.py [--dry-run]
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import improvement_engine as ie  # noqa: E402


def main(dry_run: bool = False) -> int:
    if not ie.PERFORMANCE_HISTORY_PATH.exists():
        print("No performance_history.csv - nothing to repair.")
        return 0

    raw = pd.read_csv(ie.PERFORMANCE_HISTORY_PATH)
    norm = ie._normalize_performance_history(raw)

    dropped_dates = sorted(set(raw["run_date"]) - set(norm["run_date"]))
    print(f"performance_history.csv: {len(raw):,} rows -> {len(norm):,} rows")
    print(f"  run dates: {raw['run_date'].nunique()} -> {norm['run_date'].nunique()}")
    if dropped_dates:
        print(f"  weekend run dates removed: {', '.join(dropped_dates)}")

    if dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    norm.to_csv(ie.PERFORMANCE_HISTORY_PATH, index=False)

    # live_ic_history.csv is derived, so regenerating it from the repaired
    # performance history is the whole fix.
    print("\nRegenerating live_ic_history.csv:")
    for horizon in ie.HORIZON_DAYS:
        result = ie.compute_live_ic(horizon=horizon)
        n = 0 if result is None else len(result)
        dates = [] if result is None else sorted(result["run_date"])
        n_eff = ie._effective_observations(dates, horizon)
        print(f"  {horizon}: {n} observation(s), {n_eff} effective (non-overlapping)")

    if ie.LIVE_IC_HISTORY_PATH.exists():
        ic = pd.read_csv(ie.LIVE_IC_HISTORY_PATH)
        print(f"\nlive_ic_history.csv: {len(ic)} rows, "
              f"newest {max(ic['run_date'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(dry_run="--dry-run" in sys.argv))
