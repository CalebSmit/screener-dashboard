"""Does the market backdrop's Sahm sentence match the rule's own record?

The published panel says, of the Sahm indicator:

    "at +0.50pp or more it has marked the start of every US recession since 1970 (Sahm 2019)"

Two testable assertions sit in that sentence:

  (a) *sensitivity* - the rule reached +0.50pp around every US recession since 1970;
  (b) *"the start"* - the crossing dates the beginning of the recession.

And one it is silent about:

  (c) *specificity* - how often it reached +0.50pp with no recession following.
      The panel's own yield-curve sentence says "at least one false alarm"; this one
      says nothing.

Measured from two free FRED series, exactly the ones the module already uses:
  SAHMREALTIME - the Sahm rule computed on unemployment as first published
  USREC        - the NBER recession indicator (1 in a recession month)

Run offline-friendly: both series are cached under data/market/ by the usual fetch,
but this script will download them if the cache has no full history (the module's
cache keeps 15 years; the rule's record needs all of it).

Usage:  python -I -X utf8 research/measurements/2026-10-17-sahm-rule-record.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "measurements"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
THRESHOLD = 0.50


def series(sid: str) -> pd.Series:
    """Full history of one FRED series, cached under data/measurements/."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"fred-{sid}-full.csv"
    if not path.exists():
        import requests

        r = requests.get(FRED.format(sid=sid), timeout=60)
        r.raise_for_status()
        if not r.text.startswith("observation_date"):
            raise SystemExit(f"unexpected FRED response for {sid}: {r.text[:120]!r}")
        path.write_text(r.text, encoding="utf-8")
    df = pd.read_csv(io.StringIO(path.read_text(encoding="utf-8")))
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    return s.dropna()


def episodes(flag: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Contiguous runs of a 0/1 monthly flag, as (first month, last month)."""
    out, start = [], None
    for d, v in flag.items():
        if v == 1 and start is None:
            start = d
        elif v == 0 and start is not None:
            out.append((start, prev))
            start = None
        prev = d
    if start is not None:
        out.append((start, flag.index[-1]))
    return out


def main() -> int:
    sahm = series("SAHMREALTIME")
    rec = series("USREC")
    rec = rec.loc[rec.index >= sahm.index[0]]

    print(f"SAHMREALTIME: {len(sahm)} months, {sahm.index[0].date()} .. {sahm.index[-1].date()}")
    print(f"USREC:        {len(rec)} months, {rec.index[0].date()} .. {rec.index[-1].date()}")
    print(f"latest Sahm reading: {sahm.iloc[-1]:+.2f}pp on {sahm.index[-1].date()}\n")

    recs = [r for r in episodes(rec) if r[0].year >= 1970]
    trig = episodes((sahm >= THRESHOLD).astype(int))

    print(f"-- NBER recessions since 1970 ({len(recs)}), and the first Sahm crossing of "
          f"+{THRESHOLD:.2f}pp in each window --")
    print(f"{'recession':<22} {'first crossing':<16} {'lead/lag vs start':>18}")
    misses = []
    for start, end in recs:
        # the crossing that belongs to this recession: the first one from 12 months before
        # the NBER start to the NBER end
        window = sahm.loc[(sahm.index >= start - pd.DateOffset(months=12)) & (sahm.index <= end)]
        over = window.loc[window >= THRESHOLD]
        if len(over):
            first = over.index[0]
            months = (first.year - start.year) * 12 + (first.month - start.month)
            rel = f"{months:+d} months" if months else "same month"
        else:
            first, rel = None, "NEVER"
            misses.append(start)
        print(f"{start.date()}..{end.date()}  {str(first.date()) if first is not None else '-':<16} {rel:>18}")

    print(f"\nrecessions with no crossing in the window: {len(misses)} {[str(m.date()) for m in misses]}")

    print(f"\n-- every episode at or above +{THRESHOLD:.2f}pp, and whether a recession "
          f"began within 12 months of its start --")
    false_alarms = []
    for start, end in trig:
        ahead = rec.loc[(rec.index >= start - pd.DateOffset(months=1)) & (rec.index <= start + pd.DateOffset(months=12))]
        in_rec = bool((ahead == 1).any())
        # was the economy already in recession when it crossed?
        already = bool(rec.get(start, 0) == 1)
        peak = sahm.loc[start:end].max()
        tag = "in/near recession" if in_rec else "NO RECESSION within 12 months"
        if not in_rec:
            false_alarms.append((start, end, peak))
        print(f"{start.date()}..{end.date()}  peak {peak:+.2f}pp  already in recession at crossing: "
              f"{'yes' if already else 'no':<3}  {tag}")

    print(f"\nepisodes with no recession within 12 months: {len(false_alarms)}")
    for start, end, peak in false_alarms:
        print(f"  {start.date()}..{end.date()}, peak {peak:+.2f}pp")

    # (b) the word "start": how many crossings came at or before the NBER start month.
    # USREC flags the month *after* the NBER peak as the first recession month, so quote both
    # conventions - Sahm's own chapter counts from the first recession month (p. 78: "two months
    # after the 2001 recession began"), and a reader saying "the start" means the peak.
    print("\n-- the word \"start\": lead/lag of the first crossing, both NBER conventions --")
    lags_first, lags_peak = [], []
    for start, end in recs:
        peak = start - pd.DateOffset(months=1)          # the NBER peak month itself
        window = sahm.loc[(sahm.index >= start - pd.DateOffset(months=12)) & (sahm.index <= end)]
        over = window.loc[window >= THRESHOLD]
        if not len(over):
            continue
        first = over.index[0]
        lf = (first.year - start.year) * 12 + (first.month - start.month)
        lp = (first.year - peak.year) * 12 + (first.month - peak.month)
        lags_first.append(lf)
        lags_peak.append(lp)
        print(f"  {start.date()}  crossing {first.date()}  {lf:+d} vs first recession month, "
              f"{lp:+d} vs NBER peak ({peak.date()})")
    after_first = sum(1 for v in lags_first if v > 0)
    after_peak = sum(1 for v in lags_peak if v > 0)
    print(f"\n  crossed AFTER the first recession month: {after_first} of {len(lags_first)} "
          f"(mean {sum(lags_first) / len(lags_first):+.1f} months)")
    print(f"  crossed AFTER the NBER peak month:       {after_peak} of {len(lags_peak)} "
          f"(mean {sum(lags_peak) / len(lags_peak):+.1f} months)")

    # (c) the two off-cycle episodes, month by month, so the note can quote them
    print("\n-- the off-cycle episodes, month by month --")
    for label, lo, hi in (("1976", "1976-06-01", "1977-06-01"), ("2024", "2024-01-01", "2025-06-01")):
        w = sahm.loc[lo:hi]
        print(f"  {label}: " + ", ".join(f"{d:%Y-%m} {v:+.2f}" for d, v in w.items()))
        nxt = rec.loc[(rec.index > pd.Timestamp(lo)) & (rec == 1)]
        print(f"        next recession month on record after {lo[:7]}: "
              f"{nxt.index[0].date() if len(nxt) else 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
