"""The filing cross-check's range table, re-run on the pre-registered date.

`research/2026-10-14-yahoo-vs-sec.md` section 5 (2026-11-02) committed to re-running its table on
the first data run on or after 2026-11-16 and to one rule: total assets, equity and operating
cash flow become a gate at the existing bar of 25 **only if each stayed at or below 5 on every
run**. This prints the table from every `logs/datarun-*.log` carrying the check's line, and the
per-run series for the three candidate fields so a move can be dated.

    python research/measurements/2026-11-17-filing-crosscheck-range.py
"""
from __future__ import annotations

import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LINE = re.compile(r"inputs against SEC filings: (.*)")
FIELD = re.compile(r"([a-z /]+?) (\d+)/(\d+) within [\d.]+% \((\d+) over 10%\)")
CANDIDATES = ("total assets", "equity", "operating cash flow")
BAR = 5


def read() -> list[tuple[str, dict[str, tuple[int, int, int]]]]:
    out = []
    for p in sorted((ROOT / "logs").glob("datarun-*.log")):
        m = None
        for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = LINE.search(ln) or m
        if not m:
            continue
        fields = {name.strip(): (int(a), int(n), int(over))
                  for name, a, n, over in FIELD.findall(m.group(1))}
        if fields:
            out.append((p.stem.replace("datarun-", ""), fields))
    return out


if __name__ == "__main__":
    runs = read()
    names: list[str] = []
    for _, f in runs:
        for k in f:
            if k not in names:
                names.append(k)
    print(f"{len(runs)} runs, {runs[0][0]} .. {runs[-1][0]}\n")
    print(f"{'field':<40}{'runs':>5}{'min':>5}{'median':>8}{'max':>5}")
    for k in names:
        v = [f[k][2] for _, f in runs if k in f]
        print(f"{k:<40}{len(v):>5}{min(v):>5}{statistics.median(v):>8g}{max(v):>5}")
    print("\nper run, stocks over 10% (compared):")
    print(f"{'run':<20}" + "".join(f"{k:>24}" for k in CANDIDATES))
    for stamp, f in runs:
        print(f"{stamp:<20}" + "".join(
            f"{(str(f[k][2]) + ' (' + str(f[k][1]) + ')') if k in f else '-':>24}" for k in CANDIDATES))
    print()
    for k in CANDIDATES:
        v = [(s, f[k][2]) for s, f in runs if k in f]
        worst = max(x for _, x in v)
        first = next((s for s, x in v if x > BAR), None)
        print(f"{k}: max {worst} -> {'MET (<= 5 on every run)' if worst <= BAR else 'NOT MET, first over 5 on ' + first}")

    # which stocks, and what each side is reading - from the newest run's own payload
    import sys
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    import pandas as pd
    import check_run_health as H
    import run_context
    import sec_crosscheck as X
    run_dir = run_context.latest_run_dir()
    payload = H.load_payload(Path(run_dir))
    from sec_fundamentals import FACTS_PATH
    facts = pd.read_parquet(FACTS_PATH)
    df = X.compare(payload, facts)
    c = df[(df["field"] == "operating_cash_flow") & (df["status"] == "compared")].copy()
    far = c[c["rel_diff"].abs() > X.FAR].sort_values("rel_diff")
    print(f"\noperating cash flow over 10% in {Path(run_dir).name}: {len(far)} of {len(c)}")
    for _, r in far.iterrows():
        det = payload["stock_detail"][r["ticker"]]
        print(f"  {r['ticker']:<6} run {r['yahoo'] / 1e9:>9.3f}bn  filing {r['sec'] / 1e9:>9.3f}bn  "
              f"{r['rel_diff']:+.1%}  asof cf {r['asof']}  ({r['how']})  "
              f"inp keys with 'roll'/'ltm': {[k for k in (det.get('inp') or {}) if 'roll' in k.lower() or 'ltm' in k.lower()]}")
