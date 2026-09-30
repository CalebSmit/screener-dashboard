"""Why did each S&P 500 exit leave, 2020-01..2026-08?

This is the companion to `2026-09-30-delisted-price-requirement.py`, and it
exists to stop one claim in the accompanying note being an inference.

The price census establishes *whether* a free source carries an exited name.
It cannot say *why* the name left, and that matters for the delisting-return
question. Shumway (1997, `JF` 52(1)) shows CRSP's missing delisting returns are
large and concentrated in **performance-related** delistings — the convention
is to substitute -30% (NYSE/AMEX) or -55% (Nasdaq, Shumway & Warther 1999).
If this universe's exits were mostly performance delistings, a vendor selling
last-traded prices would leave a large, signed hole. If they are mostly
acquisitions and market-cap demotions, it would not: an acquired stock trades
at or near the announced consideration until the deal closes, and a demoted
company simply keeps trading.

Source: the "Historical components of the S&P 500" Wikipedia article, which
carries an Effective Date / Added / Removed / **Reason** / References table with
inline citations to S&P DJI press releases. Wikipedia is the contemporaneous
record, not the index — the same caveat `universe_history.py` carries — so the
classification below is reported with its unmatched residual rather than as a
census.

Run:  python research/measurements/2026-09-30-exit-reasons.py
"""

from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import universe_history as uh  # noqa: E402

ARTICLE = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
CENSUS = ROOT / "research" / "measurements" / "2026-09-30-delisted-price-census.json"

# Ordered: the first pattern that matches wins, so the more specific
# terminal-outcome reasons are tested before the generic ones.
RULES = [
    ("bankruptcy/receivership",
     r"receivership|bankrupt|chapter 11|liquidat|wound? up|winding up"),
    ("acquired / merged / taken private",
     r"acquir|merg|taken private|purchase[d]? by|buyout|tender offer|"
     r"combin\w* with|bought by"),
    ("spin-off / restructuring",
     r"spin[- ]?off|split into|separat\w+ into|reorganiz|restructur|"
     r"redomicil|reincorporat"),
    ("market-cap / representation",
     r"market cap|representation|no longer|more representative|size|"
     r"liquidity|ceased to be representative"),
]


def classify(reason: str) -> str:
    text = (reason or "").lower()
    if not text.strip() or text.strip() == "nan":
        return "no reason given"
    for label, pattern in RULES:
        if re.search(pattern, text):
            return label
    return "other / unparsed"


def _flatten(cols):
    """Wikipedia's table header is two rows; pandas gives a MultiIndex."""
    out = []
    for c in cols:
        if isinstance(c, tuple):
            parts = [str(p) for p in c if not str(p).startswith("Unnamed")]
            out.append(" ".join(dict.fromkeys(parts)).strip())
        else:
            out.append(str(c))
    return out


def load_changes() -> pd.DataFrame:
    session = uh._session()
    html = session.get(ARTICLE, timeout=60).text
    frames = pd.read_html(io.StringIO(html))
    best = None
    for f in frames:
        cols = _flatten(f.columns)
        joined = " | ".join(cols).lower()
        if "reason" in joined and "removed" in joined:
            f = f.copy()
            f.columns = cols
            if best is None or len(f) > len(best):
                best = f
    if best is None:
        raise SystemExit("No additions/removals table with a Reason column "
                         "found on the article. It may have been restructured; "
                         "check the page before trusting any prior run.")
    return best


def main() -> int:
    if not CENSUS.exists():
        print(f"Run the price census first — {CENSUS.name} is missing.")
        return 1
    census = json.loads(CENSUS.read_text())
    exits = set(census["exited"]) | set(census["unresolved"])

    changes = load_changes()
    cols = list(changes.columns)
    print(f"Table: {len(changes)} rows, columns {cols}")

    date_col = next(c for c in cols if "date" in c.lower())
    reason_col = next(c for c in cols if c.lower().strip() == "reason"
                      or c.lower().endswith("reason"))
    # "Removed Ticker" under a two-row header; fall back to any Removed column.
    rem_cols = [c for c in cols if "removed" in c.lower()]
    tick_col = next((c for c in rem_cols if "ticker" in c.lower()
                     or "symbol" in c.lower()), rem_cols[0])

    print(f"Using: date={date_col!r} removed={tick_col!r} reason={reason_col!r}")

    reasons = {}
    for _, row in changes.iterrows():
        t = uh.normalize_ticker(row[tick_col])
        if not t or t == "NAN":
            continue
        # Keep the *latest* removal for a ticker that left more than once.
        reasons.setdefault(t, []).append((str(row[date_col]),
                                          str(row[reason_col])))

    matched, unmatched = {}, []
    for t in sorted(exits):
        if t in reasons:
            matched[t] = sorted(reasons[t])[-1]
        else:
            unmatched.append(t)

    buckets = {}
    for t, (date, reason) in matched.items():
        buckets.setdefault(classify(reason), []).append((t, date, reason))

    print()
    print("=" * 78)
    print(f"EXIT REASONS — {len(exits)} genuine exits, "
          f"{len(matched)} matched to the changes table, "
          f"{len(unmatched)} unmatched")
    print("=" * 78)
    for label in sorted(buckets, key=lambda k: -len(buckets[k])):
        rows = buckets[label]
        print(f"{label:<36}{len(rows):>4}  "
              f"({100 * len(rows) / max(len(matched), 1):.0f}% of matched)")
    print()
    for label in sorted(buckets, key=lambda k: -len(buckets[k])):
        print("-" * 78)
        print(label.upper())
        for t, date, reason in sorted(buckets[label]):
            print(f"  {t:<6} {date:<12} {reason[:88]}")
    if unmatched:
        print()
        print("-" * 78)
        print("Unmatched (no removal row on the article — the table starts "
              "in Dec 2011 but is 'selected' changes, not a census):")
        print("  " + ", ".join(unmatched))

    # Cross-tabulate against the price census. The question this answers is
    # whether the free-data gap is concentrated in the reasons that are hard to
    # price. If the missing name-months were mostly bankruptcies, a vendor
    # selling last-traded prices would not help; if they are mostly
    # acquisitions, it would, because the index itself exits at the last
    # traded close (S&P DJI, "Deletions").
    by_ticker = {r["ticker"]: r for r in census["rows"]}
    print()
    print("=" * 78)
    print("COVERAGE BY REMOVAL REASON — is the free gap in the hard cases?")
    print("=" * 78)
    print(f"{'reason':<36}{'names':>6}{'no series':>11}"
          f"{'months req':>12}{'months missing':>16}")
    print("-" * 78)
    order = sorted(buckets, key=lambda k: -len(buckets[k]))
    for label in order:
        tickers = [t for t, _, _ in buckets[label]]
        rows = [by_ticker[t] for t in tickers if t in by_ticker]
        req = sum(r["n_months"] for r in rows)
        miss = sum(len(r["missing"]) for r in rows)
        none = sum(1 for r in rows if r["rows"] == 0)
        pct = f"{100 * miss / req:.0f}%" if req else "-"
        print(f"{label:<36}{len(rows):>6}{none:>11}{req:>12}"
              f"{miss:>11} ({pct})")
    unm = [by_ticker[t] for t in unmatched if t in by_ticker]
    if unm:
        req = sum(r["n_months"] for r in unm)
        miss = sum(len(r["missing"]) for r in unm)
        none = sum(1 for r in unm if r["rows"] == 0)
        pct = f"{100 * miss / req:.0f}%" if req else "-"
        print(f"{'(unmatched)':<36}{len(unm):>6}{none:>11}{req:>12}"
              f"{miss:>11} ({pct})")

    out = ROOT / "research" / "measurements" / "2026-09-30-exit-reasons.json"
    out.write_text(json.dumps({
        "n_exits": len(exits), "n_matched": len(matched),
        "unmatched": unmatched,
        "buckets": {k: [list(r) for r in v] for k, v in buckets.items()},
    }, indent=1))
    print()
    print(f"Wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
