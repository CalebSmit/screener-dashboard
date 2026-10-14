"""What the live site shows for Dividend Yield, and where the number comes from.

Two separate derivations of one quantity exist, both off `.info["dividendRate"]`,
which the source stopped serving (see 2026-10-14-dividend-field-probe.py):

  * `factor_engine` -> `dividend_yield` (weight 0, listed in the Valuation panel)
  * `generate_dashboard` -> `financials.dividend_yield` (the Company Snapshot row)

CLAUDE.md row 0.8 says the page shows the engine's numbers and never its own
re-derivation of them. This counts both, on the committed payload, so the
user-facing size of the outage is a measured number rather than an inference.

Usage: python research/measurements/2026-10-14-published-dividend-yield.py
Reads dashboard_data.js only. No network, writes nothing.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

PAYLOAD = Path("dashboard_data.js")


def load() -> dict:
    text = PAYLOAD.read_text(encoding="utf-8")
    m = re.match(r"\s*window\.SCREENER_DATA\s*=\s*", text)
    if not m:
        raise SystemExit("payload does not open with the expected assignment")
    body = text[m.end():].rstrip().rstrip(";")
    return json.loads(body)


def main() -> None:
    data = load()
    detail = data["stock_detail"]
    n = len(detail)
    raw_have = [t for t, d in detail.items()
                if (d.get("raw") or {}).get("dividend_yield") is not None]
    fin_have = [t for t, d in detail.items()
                if (d.get("financials") or {}).get("dividend_yield") is not None]
    print(f"run {data['kpis'].get('run_timestamp')}  stocks {n}\n")
    print(f"  engine raw.dividend_yield present        {len(raw_have)}/{n}")
    print(f"  page   financials.dividend_yield present {len(fin_have)}/{n}")
    print(f"  the two agree where both exist:          "
          f"{sum(1 for t in raw_have if t in fin_have)} overlap")

    # Do they agree in value? A second derivation that drifts is the row-0.8 risk.
    disagree = []
    for t in raw_have:
        a = detail[t]["raw"]["dividend_yield"]
        b = (detail[t].get("financials") or {}).get("dividend_yield")
        if b is None or abs(a - b) > 1e-4:
            disagree.append((t, a, b))
    print(f"  of those, differ by more than 1e-4:      {len(disagree)}")
    for t, a, b in disagree[:10]:
        print(f"      {t:<6} engine {a} page {b}")

    have = sorted(raw_have)
    print(f"\n  stocks that still have a yield: {have}")

    # What a reader sees in the Valuation panel for a stock without one.
    pct_have = sum(1 for d in detail.values()
                   if (d.get("pct") or {}).get("dividend_yield") is not None)
    print(f"  pct.dividend_yield present               {pct_have}/{n}")


if __name__ == "__main__":
    main()
