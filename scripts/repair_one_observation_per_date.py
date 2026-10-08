"""Collapse the evidence base to one observation per run date.

CLAUDE.md priority 0.6. `record_dispersion` appended unconditionally and
`improvement/snapshots/` kept every file, so days with two runs are counted more than
once. `compute_forward_returns` already reads only the **last** snapshot per date, so
that is the rule applied here too: keep the last row / last file for each date.

Measured before running, 2026-10-08:
  dispersion_history.csv  52 rows, 45 distinct dates (2026-04-14 x4, 2026-07-28 x3,
                          2026-03-16 x2, 2026-08-26 x2)
  improvement/snapshots   79 files, 52 distinct dates

Nothing that is read changes value: the rows and files removed are the ones every
consumer already discarded. Everything removed is recoverable from git history.

    python scripts/repair_one_observation_per_date.py --dry-run
    python scripts/repair_one_observation_per_date.py --apply
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DISP = ROOT / "improvement" / "dispersion_history.csv"
SNAPS = ROOT / "improvement" / "snapshots"
VOL = ROOT / "factor_vol_history.csv"


def plan_dispersion():
    if not DISP.exists():
        return None, []
    df = pd.read_csv(DISP)
    before = len(df)
    kept = (df.assign(_d=df["date"].astype(str))
              .drop_duplicates(subset="_d", keep="last")
              .sort_values("_d", kind="stable")
              .drop(columns="_d")
              .reset_index(drop=True))
    dropped = df["date"].astype(str).value_counts()
    dropped = {k: int(v) for k, v in dropped[dropped > 1].items()}
    return (before, len(kept), kept), dropped


def plan_vol():
    """`factor_vol_history.csv` feeds `adjust_momentum_weight`: the current run's momentum
    dispersion is ranked against every row here, and that percentile sets the momentum
    weight. A date repeated nine times is nine votes for one day's data."""
    if not VOL.exists():
        return None, {}
    df = pd.read_csv(VOL)
    before = len(df)
    kept = (df.assign(_d=df["date"].astype(str))
              .drop_duplicates(subset="_d", keep="last")
              .sort_values("_d", kind="stable")
              .drop(columns="_d")
              .reset_index(drop=True))
    dup = df["date"].astype(str).value_counts()
    return (before, len(kept), kept), {k: int(v) for k, v in dup[dup > 1].items()}


def plan_snapshots():
    if not SNAPS.exists():
        return [], []
    by_date = {}
    for p in sorted(SNAPS.glob("*.parquet")):
        by_date.setdefault(p.stem.split("_")[0], []).append(p)
    stale = [p for ps in by_date.values() for p in ps[:-1]]
    return sorted(SNAPS.glob("*.parquet")), stale


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    disp, dup_dates = plan_dispersion()
    vol, dup_vol = plan_vol()
    all_snaps, stale = plan_snapshots()

    if disp:
        before, after, kept = disp
        print(f"dispersion_history.csv: {before} rows -> {after} "
              f"({before - after} duplicate rows on {len(dup_dates)} dates: {dup_dates})")
    if vol:
        before, after, _ = vol
        print(f"factor_vol_history.csv: {before} rows -> {after} "
              f"({before - after} duplicate rows on {len(dup_vol)} dates: {dup_vol})")
    print(f"improvement/snapshots:  {len(all_snaps)} files -> {len(all_snaps) - len(stale)} "
          f"({len(stale)} superseded)")

    if args.dry_run:
        for p in stale:
            print(f"  would remove {p.name}")
        return 0

    if disp:
        _, _, kept = disp
        kept.to_csv(DISP, index=False)
        print(f"wrote {DISP.relative_to(ROOT)} with {len(kept)} rows")
    if vol:
        _, _, kept_vol = vol
        kept_vol.to_csv(VOL, index=False)
        print(f"wrote {VOL.relative_to(ROOT)} with {len(kept_vol)} rows")
    for p in stale:
        p.unlink()
    if stale:
        print(f"removed {len(stale)} superseded snapshot file(s)")

    # Verify, the same way the test does.
    for path, label in ((DISP, "dispersion"), (VOL, "momentum vol")):
        if path.exists():
            d = pd.read_csv(path)
            assert d["date"].astype(str).is_unique, f"{label} still repeats a date"
    seen = {}
    for p in SNAPS.glob("*.parquet"):
        seen.setdefault(p.stem.split("_")[0], []).append(p.name)
    assert all(len(v) == 1 for v in seen.values()), "snapshots still repeat a date"
    print(f"verified: one dispersion row and one snapshot per date "
          f"({len(seen)} dates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
