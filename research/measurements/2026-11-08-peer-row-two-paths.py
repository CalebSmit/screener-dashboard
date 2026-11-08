"""The peer table builds one row two ways. Do the two ways agree?

`renderPeerComparison` prints a nine-column table: the stock whose sheet is open, then its
sector peers. The open stock's row comes from `stock_detail[t].self_metrics`, computed in
Python at build time. Every peer's row comes from `buildPeerRow(t)`, computed in JS from
`stock_detail[t].raw` and `stock_detail[t].financials`.

Two code paths for one table's rows, and every stock appears on both sides of them: a stock
is the subject of its own sheet and a peer in its sector-mates' sheets. So for each of the
nine columns this asks whether the same company prints the same figure either way.

CLAUDE.md row 0.8 is the standard being tested: "the page shows the engine's numbers and
never its own re-derivation of them".

Run:  python research/measurements/2026-11-08-peer-row-two-paths.py

Reads the committed `dashboard_data.js` (read-only; writes nothing).
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(path: Path) -> dict:
    txt = path.read_text(encoding="utf-8")
    m = re.match(r"^\s*window\.SCREENER_DATA\s*=\s*", txt)
    if not m:
        raise SystemExit(f"{path.name} does not open with the expected assignment")
    return json.loads(txt[m.end():].rstrip().rstrip(";"))


def js_round(x: float) -> float:
    """JavaScript's Math.round: half away from zero for positives, half *up* for negatives.

    Python's round() is half-to-even, so the two disagree at an exact .5 and the sign of
    that disagreement is what this script is looking for."""
    return math.floor(x + 0.5)


def build_peer_row(d: dict) -> dict:
    """`buildPeerRow` in generate_dashboard.py, transcribed field for field."""
    raw = d.get("raw") or {}
    fin = d.get("financials") or {}
    ey = raw.get("earnings_yield")
    pe = js_round(10.0 / ey) / 10 if (ey is not None and ey > 0) else None
    return {
        "rev_growth": raw.get("revenue_growth"),
        "pe_ratio": pe,
        "net_margin": fin.get("net_margin"),
        "roic": raw.get("roic"),
        "roe": raw.get("roe"),
        "debt_equity": raw.get("debt_equity"),
        "div_yield": fin.get("dividend_yield"),
        "fcf_yield": raw.get("fcf_yield"),
        "mcap": fin.get("market_cap"),
    }


FIELDS = ("rev_growth", "pe_ratio", "net_margin", "roic", "roe",
          "debt_equity", "div_yield", "fcf_yield", "mcap")


def main() -> int:
    path = ROOT / "dashboard_data.js"
    if not path.exists():
        print(f"MISSING DATA: {path} is not present; nothing to measure")
        return 3
    data = load(path)
    detail = data.get("stock_detail") or {}
    print(f"{len(detail)} stocks in stock_detail, from run {data.get('kpis', {}).get('run_timestamp')}")

    counts = {f: 0 for f in FIELDS}
    examples: dict[str, list] = {f: [] for f in FIELDS}
    no_self = []
    compared = 0

    for t in sorted(detail):
        d = detail[t]
        self_row = d.get("self_metrics")
        if not self_row:
            no_self.append(t)
            continue
        compared += 1
        peer_row = build_peer_row(d)
        for f in FIELDS:
            a, b = self_row.get(f), peer_row.get(f)
            if a is None and b is None:
                continue
            if a is None or b is None or a != b:
                counts[f] += 1
                if len(examples[f]) < 6:
                    examples[f].append((t, a, b))

    print(f"compared {compared} stocks; {len(no_self)} have no self_metrics {no_self[:8]}")
    print()
    print(f"{'column':<14} {'disagreements':>13}   examples (ticker: self_metrics vs buildPeerRow)")
    for f in FIELDS:
        ex = "  ".join(f"{t}: {a} vs {b}" for t, a, b in examples[f])
        print(f"{f:<14} {counts[f]:>13}   {ex}")

    total = sum(counts.values())
    print()
    print(f"TOTAL disagreeing cells: {total} of {compared * len(FIELDS)}")

    # The P/E column has two independent reasons to diverge; separate them, because only one
    # of the two is visible from the payload alone.
    ey_round = 0
    for t in sorted(detail):
        d = detail[t]
        self_row = d.get("self_metrics") or {}
        raw = d.get("raw") or {}
        ey = raw.get("earnings_yield")
        if ey is None or ey <= 0 or self_row.get("pe_ratio") is None:
            continue
        # Python's own rounding on the *published* (4dp) earnings yield
        if round(1.0 / ey, 1) != self_row["pe_ratio"]:
            ey_round += 1
    print(f"P/E: {ey_round} stocks where Python's own round() on the *published* 4dp "
          f"earnings yield does not reproduce the published self_metrics P/E "
          f"(so the published figure was computed from an unrounded input).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
