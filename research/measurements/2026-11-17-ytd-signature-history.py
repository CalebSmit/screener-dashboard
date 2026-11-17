"""Does the filings-side signature of a year-to-date column fire on anything it should not?

`sec_crosscheck.ytd_columns` flags a stock when the run's operating cash flow exceeds the filed
twelve months by the filed year-to-date figure one quarter before one of the last four quarter
ends (within 0.5% of the run's figure). `2026-11-17-ytd-in-a-quarterly-column.py` found, from
the source's own columns, that exactly one stock has such a column inside its twelve months
today (DexCom). This runs the signature over every published page git still holds and prints
every stock it names on any of them, beside how many stocks on that page were over 2% and over
10% from the filing for any reason - so a signature that fired on ordinary disagreements would
show as a crowd.

    python research/measurements/2026-11-17-ytd-signature-history.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import sec_crosscheck as X  # noqa: E402
from sec_fundamentals import FACTS_PATH  # noqa: E402

if __name__ == "__main__":
    facts = pd.read_parquet(FACTS_PATH)
    ocf = [f for f in X.FIELDS if f[0] == "operating_cash_flow"]
    log = subprocess.run(["git", "log", "--format=%H %ad %s", "--date=short", "--", "dashboard_data.js"],
                         capture_output=True, text=True, cwd=ROOT, check=True).stdout.splitlines()
    seen, named, pages = set(), defaultdict(list), 0
    stock_days = over2 = over10 = 0
    for line in reversed(log):
        sha, day, subject = line.split(" ", 2)
        if not subject.startswith("data:") or day in seen:
            continue
        js = subprocess.run(["git", "show", f"{sha}:dashboard_data.js"], capture_output=True, cwd=ROOT).stdout.decode("utf-8", "replace")
        try:
            payload = json.loads(js[js.index("{"): js.rindex("}") + 1])
        except ValueError:
            continue
        df = X.compare(payload, facts, ocf)
        c = df[df["status"] == "compared"] if "status" in df else df
        if c.empty:
            continue                      # a page from before `inp` was published
        seen.add(day)
        pages += 1
        a = c["rel_diff"].abs()
        got = X.ytd_columns(payload, facts)
        stock_days += len(c); over2 += int((a > 0.02).sum()); over10 += int((a > X.FAR).sum())
        print(f"{day}: compared {len(c)}, over 2% {int((a > 0.02).sum())}, over 10% {int((a > X.FAR).sum())}, "
              f"signature fires on {sorted(got)}")
        for t, e in got.items():
            named[t].append((day, e))
    print(f"\n{pages} pages, {stock_days} stock-days compared; {over2} over 2% from the filing, {over10} over 10%; "
          f"the signature fired on {sum(len(v) for v in named.values())} stock-days, {len(named)} stocks")
    for t, hits in named.items():
        d0, e = hits[0]
        print(f"  {t}: {len(hits)} pages, {d0} to {hits[-1][0]}; run {e['run'] / 1e9:.3f}bn, filed {e['filed'] / 1e9:.3f}bn, "
              f"column {e['column_end']} carries {e['column_ytd'] / 1e9:.3f}bn = {e['months']} months, quarter {e['column_quarter'] / 1e9:.3f}bn")
        print("   ", X.ytd_column_note(e))
