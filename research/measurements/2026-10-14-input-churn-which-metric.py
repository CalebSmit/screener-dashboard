"""Which metric percentile went missing, for how many stocks, and on which run.

`tests/test_input_churn.py::test_build_history_on_real_snapshots_reports_some_churn`
went red at the 2026-10-14 06:00 baseline having been green the day before: 852 of
1002 comparison entries carry input churn, against a guard of "fewer than half".
The guard is a tripwire on the real snapshot directory, so the question is which
metric and why - not how to quiet the test.

Usage: python research/measurements/2026-10-14-input-churn-which-metric.py
Reads improvement/snapshots/ only. No network, writes nothing.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import history as H  # noqa: E402


def main() -> None:
    idx = H.snapshot_index()
    dates = sorted(idx)
    print(f"{len(dates)} snapshot dates, {dates[0]} .. {dates[-1]}\n")

    # Per-date: how many tickers lack each metric percentile.
    print("Metric percentiles NaN per date (newest 8 dates, metrics absent for >50 tickers):")
    recent = dates[-8:]
    frames = {}
    for d in recent:
        df = pd.read_parquet(idx[d])
        frames[d] = df
    cols = sorted(set().union(*[set(H.metric_pct_columns(f)) for f in frames.values()]))
    rows = []
    for c in cols:
        counts = []
        for d in recent:
            f = frames[d]
            counts.append(int(f[c].isna().sum()) if c in f.columns else None)
        if any(v is not None and v > 50 for v in counts):
            rows.append((c, counts))
    hdr = "  " + "metric".ljust(34) + "".join(d[5:].rjust(8) for d in recent)
    print(hdr)
    for c, counts in rows:
        print("  " + c.ljust(34) + "".join(("-" if v is None else str(v)).rjust(8) for v in counts))

    # The churn the history actually reports, and which metric drives it.
    out = H.build_history()
    print(f"\nbuild_history available={out.get('available')} "
          f"baselines={out.get('baselines')}")
    entries = [(t, k, e) for t, d in out["delta"].items()
               for k in ("m1", "prev") if (e := d.get(k))]
    flagged = [(t, k, e) for t, k, e in entries if e.get("ch")]
    print(f"entries={len(entries)} flagged={len(flagged)} "
          f"({100 * len(flagged) / max(1, len(entries)):.1f}%)")
    print("  ch shapes: " + str(Counter(tuple(e['ch']) for _, _, e in flagged).most_common()))
    print("  by baseline: " + str(Counter(k for _, k, _ in flagged).most_common()))

    # Name the metric: rebuild the two snapshots the history compared.
    cur_date = dates[-1]
    cur = pd.read_parquet(idx[cur_date])
    for key in ("prev", "m1"):
        b = (out.get("baselines") or {}).get(key)
        if not b:
            continue
        bdate = b if isinstance(b, str) else b.get("date")
        if bdate not in idx:
            print(f"\n  baseline {key}={bdate} not in the index")
            continue
        base = pd.read_parquet(idx[bdate])
        shared = set(H.metric_pct_columns(cur)) & set(H.metric_pct_columns(base))
        lost = Counter()
        gained = Counter()
        bi = base.set_index("Ticker")
        ci = cur.set_index("Ticker")
        both = [t for t in ci.index if t in bi.index]
        for c in sorted(shared):
            now_na = ci.loc[both, c].isna()
            was_na = bi.loc[both, c].isna()
            lost[c] = int((now_na & ~was_na).sum())
            gained[c] = int((~now_na & was_na).sum())
        print(f"\n  {cur_date} vs {key}={bdate}: {len(both)} shared tickers")
        for c, n in lost.most_common(6):
            if n:
                print(f"    lost   {c:<34} {n}")
        for c, n in gained.most_common(6):
            if n:
                print(f"    gained {c:<34} {n}")


if __name__ == "__main__":
    main()
