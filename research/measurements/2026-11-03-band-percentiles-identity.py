"""Pre-registration 3's exit test, actually run, plus the min_market_cap interaction.

`research/2026-11-02-universe-expansion-phase0.md` section 3 asserts:

    "`compute_sector_percentiles` groups by `(Sector, size_band)`. The S&P 500 band is today's
     universe, so its percentiles are unchanged by construction. That is the phase-1 exit test,
     met for free."

"By construction" is an argument, not a measurement, and it has a moving part the note states
in two different ways. Section 7 says `universe.min_market_cap: 2e9` must become per-universe --
"Keep 2e9 for the S&P 500 (byte-identical) and set 0 for S&P index bands" -- but the S&P 500 *is*
an S&P index band, so the instruction contradicts itself for exactly the band whose
byte-identity is the exit test. Which branch is right depends on a number neither the note nor
the plan measures: whether any S&P 500 member sits below $2B.

This script:
  1. Runs `compute_sector_percentiles` as it stands on the published run.
  2. Runs a band-grouped variant -- group by (Sector, size_band), fall back to **band**-wide --
     with a single constant band, and compares every `*_pct` column exactly.
  3. Does the same with a deliberately *wrong* grouping, to confirm the comparison can fail
     (otherwise step 2 passes vacuously).
  4. Measures the S&P 500's market-cap floor against `universe.min_market_cap`.

Run:  python research/measurements/2026-11-03-band-percentiles-identity.py
Reads the published run only. Writes nothing.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import factor_engine as fe  # noqa: E402
import run_context  # noqa: E402


def band_percentiles(df: pd.DataFrame, band_col: str) -> pd.DataFrame:
    """compute_sector_percentiles, grouped by (Sector, band) with a band-wide fallback.

    A transcription of factor_engine.compute_sector_percentiles with exactly the two changes
    pre-registration 3 asks for, so the comparison below is of the proposal and nothing else.
    """
    df = df.copy()
    pct = {c: f"{c}_pct" for c in fe.METRIC_COLS}
    for c in pct.values():
        df[c] = np.nan

    # Fallback is now band-wide, not universe-wide: one rank table per band.
    band_ranks: dict[tuple, dict[str, pd.Series]] = {}
    for band, bgrp in df.groupby(band_col):
        band_ranks[band] = {
            col: fe._directed_pct(bgrp[col], fe.METRIC_DIR.get(col, True))
            for col in fe.METRIC_COLS
            if col in df.columns
        }

    for (sector, band), grp in df.groupby(["Sector", band_col]):
        for col in fe.METRIC_COLS:
            pc = pct[col]
            if col not in df.columns:
                df.loc[grp.index, pc] = 50.0
                continue
            valid = grp[col].dropna()
            if len(valid) < fe.SECTOR_MIN_PEERS:
                ranks = band_ranks[band].get(col)
                if ranks is not None:
                    df.loc[grp.index, pc] = ranks.loc[grp.index]
                else:
                    df.loc[grp.index, pc] = 50.0
                continue
            df.loc[grp.index, pc] = fe._directed_pct(grp[col], fe.METRIC_DIR.get(col, True))
    return df


def compare(a: pd.DataFrame, b: pd.DataFrame, label: str) -> int:
    cols = [f"{c}_pct" for c in fe.METRIC_COLS if f"{c}_pct" in a.columns]
    diffs = 0
    worst = (None, 0.0)
    for c in cols:
        x, y = a[c], b[c]
        both_nan = x.isna() & y.isna()
        unequal = ~both_nan & ~(x.fillna(-1e18) == y.fillna(-1e18))
        n = int(unequal.sum())
        if n:
            diffs += n
            m = float((x[unequal] - y[unequal]).abs().max())
            if m > worst[1]:
                worst = (c, m)
    print(f"  {label}: {len(cols)} percentile columns x {len(a)} rows = "
          f"{len(cols) * len(a):,} values; {diffs} differ", end="")
    print(f" (largest |delta| {worst[1]:.6g} in {worst[0]})" if diffs else "")
    return diffs


def main() -> int:
    run_dir = pathlib.Path(run_context.latest_run_dir())
    src = pd.read_parquet(run_dir / "02_outliers_flagged.parquet")
    print(f"run {run_dir.name}: {len(src)} rows")

    # --- 1/2. the exit test -------------------------------------------------------
    print("\n=== Pre-registration 3's exit test ===")
    today = fe.compute_sector_percentiles(src.copy())

    one_band = src.copy()
    one_band["size_band"] = "S&P 500"
    proposed = band_percentiles(one_band, "size_band")
    n = compare(today, proposed, "today vs (Sector, size_band) with one band")
    print(f"  VERDICT: {'byte-identical -- the note is right' if n == 0 else 'DIFFERS'}")

    # --- 3. can the comparison fail? ----------------------------------------------
    print("\n=== Floor check: does the comparison detect a real regrouping? ===")
    print("  (if a deliberately wrong grouping also compares equal, step 2 proved nothing)")
    split = src.copy()
    # Split the universe in half by size, the way a real band split would. size_log_mcap is
    # the Size metric's own column on this frame, monotone in market cap.
    cap = pd.to_numeric(split["size_log_mcap"], errors="coerce")
    split["size_band"] = np.where(cap >= cap.median(), "big", "small")
    two = band_percentiles(split, "size_band")
    n2 = compare(today, two, "today vs two cap-split bands")
    print(f"  VERDICT: {'the comparison has teeth' if n2 > 0 else 'VACUOUS -- step 2 is meaningless'}")

    # --- 4. the min_market_cap interaction ----------------------------------------
    print("\n=== The note's own contradiction: universe.min_market_cap ===")
    cfg = fe.load_config()
    # factor_engine.apply_universe_filters coerces with float(), so a YAML string is fine.
    raw_floor = cfg.get("universe", {}).get("min_market_cap")
    floor = float(raw_floor) if raw_floor is not None else None
    print(f"  config universe.min_market_cap = {raw_floor!r} -> {floor:,.0f}"
          if floor is not None else "  config universe.min_market_cap is absent")
    # The filter runs on the raw fetch's marketCap, before 02 is written.
    raw = pd.read_parquet(run_dir / "00_raw_fetch.parquet")
    caps = pd.to_numeric(raw["marketCap"], errors="coerce").dropna()
    print(f"  raw fetch: {len(raw)} rows, {len(caps)} with a market cap; "
          f"min ${caps.min() / 1e9:,.3f}B, p1 ${caps.quantile(0.01) / 1e9:,.3f}B, "
          f"p5 ${caps.quantile(0.05) / 1e9:,.3f}B")
    if floor:
        below = int((caps < floor).sum())
        print(f"  members below the ${floor / 1e9:,.1f}B floor: {below}")
        smallest = caps.nsmallest(5)
        print("  smallest five: " + ", ".join(
            f"{raw.loc[i, 'Ticker']} ${v / 1e9:.2f}B" for i, v in smallest.items()))
        if below == 0:
            print("  => setting min_market_cap to 0 for EVERY S&P index band, including the")
            print("     500, is byte-identical today. The note's special case for the 500 is")
            print("     not needed, and one rule is easier to explain than two.")
        else:
            print("  => the floor is ALREADY removing index members from the published run;")
            print("     that is a present-day finding, not a phase-1 parameter question.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
