"""What the one-off-charge note selects on the published payload - 2026-10-16

``sec_crosscheck.SPECIAL_ITEM_SHARE_MIN`` was written at 0.10 before this was run (the same number
and reasoning as ``LEASE_SHARE_MIN``: a tenth of the metric's numerator). This records what it
picks, so the selection is on the record rather than fitted to it, and so a later session can see
whether the threshold has drifted away from a sentence worth printing.

Offline; reads the committed payload.

Usage: python research/measurements/2026-10-16-special-item-selection.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).with_suffix(".json")


def load_payload():
    txt = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    return json.loads(re.sub(r"^\s*window\.SCREENER_DATA\s*=\s*", "", txt).rstrip().rstrip(";"))


def main():
    import pandas as pd  # noqa: F401  - sec_crosscheck needs it
    import sec_crosscheck as X
    payload = load_payload()
    detail = payload["stock_detail"]
    facts = pd.read_parquet(ROOT / "data" / "sec" / "pit" / "facts.parquet")
    facts = facts[facts["ticker"].isin(list(detail))]
    facts["end"] = pd.to_datetime(facts["end"])
    facts["start"] = pd.to_datetime(facts["start"])
    facts["filed"] = pd.to_datetime(facts["filed"])
    found = X.special_items({"stock_detail": detail}, facts)
    rank = {r["Ticker"]: r.get("Rank") for r in payload.get("table_data") or []}
    rows = []
    for t, e in sorted(found.items(), key=lambda kv: -kv[1]["share"]):
        rows.append({"ticker": t, "sector": detail[t].get("sector"), "rank": rank.get(t),
                     "charge": e["charge"], "net_income": e["net_income"],
                     "share": e["share"], "concept": e["concept"], "asof": e["asof"],
                     "note": X.special_item_note(e)})
    OUT.write_text(json.dumps({"threshold": X.SPECIAL_ITEM_SHARE_MIN,
                               "stocks_in_payload": len(detail),
                               "selected": len(rows), "rows": rows}, indent=1), encoding="utf-8")
    print(f"threshold {X.SPECIAL_ITEM_SHARE_MIN:.0%} of net income; "
          f"{len(rows)} of {len(detail)} stocks selected\n")
    for r in rows[:18]:
        print(f"  {r['ticker']:6s} {r['share'] * 100:>7.0f}%  ${r['charge'] / 1e6:>9,.0f}M charge on "
              f"${r['net_income'] / 1e6:>9,.0f}M net  {r['concept']}")
    if rows:
        print("\nthe longest note, as it would read on the row:\n")
        print("  " + max(rows, key=lambda r: len(r["note"]))["note"])
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
