"""Force the weekly `companyfacts` refresh so the cache carries the lease tags.

`sec_fundamentals.FACT_CONCEPTS` gained the two captive-finance lease tags on 2026-10-14; the
cache on disk predates them, and it refreshes only once a week. This runs the same refresh the
data loop runs, with force=True, so the tags are present for today's work and for tonight's
publish. Writes only the gitignored cache.

    python research/measurements/2026-10-15-refresh-facts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import sec_fundamentals as S  # noqa: E402

text = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
data = json.loads(text[text.index("{"): text.rindex("}") + 1])
tickers = sorted(data["stock_detail"])
print(f"refreshing companyfacts for {len(tickers)} tickers (force)")
df = S.refresh_companyfacts(tickers, force=True)
print("rows:", 0 if df is None else len(df))
if df is not None:
    lease = df[df["concept"].str.contains("Lease", case=False)]
    print("lease-tag rows:", len(lease), "companies:", lease["ticker"].nunique())
