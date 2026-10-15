"""Which stocks does the lease-capex note select, and which does it leave out?

For `research/2026-10-14-capex-definition.md` section 5.6. The note on the FCF-yield row fires on
a filed tag rather than on the named list of section 4, so the thing to measure is what the tag
selects across the whole universe: every company whose own filings report purchases of assets to
lease out, the share of the run's capex that is, and where the 10% threshold falls.

Offline: the weekly `companyfacts` cache and the published payload.

    python research/measurements/2026-10-15-lease-capex-scope.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import sec_crosscheck as X  # noqa: E402
from sec_fundamentals import FACTS_PATH  # noqa: E402

text = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
payload = json.loads(text[text.index("{"): text.rindex("}") + 1])
facts = pd.read_parquet(FACTS_PATH)
print(f"cache: {len(facts):,} facts, {facts['ticker'].nunique()} companies")

# Every company with the tag at all, before the threshold.
loose = dict(X.lease_book(payload, facts))
with_tag = {}
old_min = X.LEASE_SHARE_MIN
X.LEASE_SHARE_MIN = 0.0
with_tag = X.lease_book(payload, facts)
X.LEASE_SHARE_MIN = old_min

print(f"\ncompanies filing a purchases-for-lease tag with a comparable capex: {len(with_tag)}")
rows = sorted(with_tag.items(), key=lambda kv: -kv[1]["share"])
for t, e in rows:
    det = payload["stock_detail"][t]
    fy = (det.get("raw") or {}).get("fcf_yield")
    proceeds = "n/a" if not e["proceeds"] else f"{e['proceeds']/1e9:,.2f}B"
    yld = "n/a" if fy is None else f"{fy:+.2%}"
    print(f"  {t:6} {str(det.get('sector'))[:22]:22} capex {e['capex']/1e9:>7,.2f}B  for-lease "
          f"{e['purchases']/1e9:>7,.2f}B = {e['share']*100:>5.1f}%  proceeds {proceeds:>9}"
          f"  FCF yield {yld:>8}  {'NOTE' if t in loose else '-'}")
print(f"\nat the {old_min:.0%} threshold the note fires for {len(loose)}: {', '.join(sorted(loose))}")
print("proceeds leg filed for:", ", ".join(sorted(t for t, e in with_tag.items() if e["proceeds"])) or "none")
