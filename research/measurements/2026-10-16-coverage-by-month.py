"""Each month's coverage in backtest v2: the share of that month's S&P 500 members it could build.

Written 2026-10-16 for the audit of ``research.html``. The page said Size's 2012-2019 lead "is
largest in the months with the most members missing and disappears where coverage is fullest".
Coverage rises almost one for one with the calendar (57% of members in 2012, 97% in 2026), so
coverage and era cannot be told apart in this data, and the page's sentence has to be one the
numbers can carry. ``research_page.size_coverage_relation`` reads this file together with the
ledger's own monthly ICs and the page's sentence is generated from what it finds.

Reads ``cache/v2_census.json`` and ``cache/v2_census_presample.json`` (``backtest_v2.py``'s per-month
census: ``members`` that month, ``built`` = members with a point-in-time record). Writes
``research/measurements/2026-10-16-coverage-by-month.json``, committed, so the page can be rebuilt
without the cache. Not decision-grade (rule 5).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix(".json")


def main() -> None:
    cov = {}
    for name in ("v2_census_presample.json", "v2_census.json"):
        for c in json.loads((ROOT / "cache" / name).read_text(encoding="utf-8")):
            if "members" in c and c.get("members"):
                cov[c["month"]] = {"members": int(c["members"]), "built": int(c["built"]),
                                   "coverage": round(c["built"] / c["members"], 4)}
    OUT.write_text(json.dumps({"what": "built / members per month, backtest v2 census",
                               "months": dict(sorted(cov.items()))}, indent=1) + "\n", encoding="utf-8")
    print(f"{len(cov)} months -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
