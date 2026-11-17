"""What the year-to-date note changes in the payload the next run publishes: nothing but itself.

`scripts/rehearse_publish.py` leaves the page it regenerated in logs/rehearse-publish/site/.
This compares that `dashboard_data.js` with the committed one, key by key, and prints every
difference - so "no number, rank or sentence moved" is a diff, not a claim.

    python scripts/rehearse_publish.py && python research/measurements/2026-11-17-ytd-note-payload-diff.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(p: Path) -> dict:
    t = p.read_text(encoding="utf-8")
    return json.loads(t[t.index("{"): t.rindex("}") + 1])


if __name__ == "__main__":
    old = load(ROOT / "dashboard_data.js")
    new = load(ROOT / "logs" / "rehearse-publish" / "site" / "dashboard_data.js")
    print("top-level keys added:", sorted(set(new) - set(old)), " removed:", sorted(set(old) - set(new)))
    for k in sorted(set(old) & set(new)):
        if k != "stock_detail" and old[k] != new[k]:
            print(f"top-level key differs: {k}")
    print("src_notes:", new.get("src_notes"))
    od, nd = old["stock_detail"], new["stock_detail"]
    print("stocks:", len(od), "->", len(nd), " same set:", set(od) == set(nd))
    changed = {}
    for t in sorted(set(od) & set(nd)):
        keys = [k for k in sorted(set(od[t]) | set(nd[t])) if od[t].get(k) != nd[t].get(k)]
        if keys:
            changed[t] = keys
    print(f"stocks with any difference: {len(changed)}")
    by_key: dict[str, list[str]] = {}
    for t, keys in changed.items():
        for k in keys:
            by_key.setdefault(k, []).append(t)
    for k, ts in sorted(by_key.items()):
        print(f"  {k}: {len(ts)} stocks {ts[:12]}{' ...' if len(ts) > 12 else ''}")
    # `flags` differs on every stock for a reason that is not this change: the 06:00 session's
    # `annual_instead` key, which the committed payload (built at 02:00) predates. Name the subkeys.
    sub = set()
    for t in by_key.get("flags", []):
        a, b = od[t].get("flags") or {}, nd[t].get("flags") or {}
        sub |= {k for k in set(a) | set(b) if a.get(k) != b.get(k)} | {f"{k} (key added)" for k in set(b) - set(a)} \
            | {f"{k} (key removed)" for k in set(a) - set(b)}
    print("  flags subkeys that differ:", sorted(sub))
    for t, s in nd.items():
        if s.get("srcn"):
            print(f"\n{t} srcn keys {sorted(s['srcn'])}, xwhy {s.get('xwhy')}\n  {s['srcn']['fcf_yield']}")
