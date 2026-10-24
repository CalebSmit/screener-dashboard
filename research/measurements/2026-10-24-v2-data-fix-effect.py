"""What the two free data fixes changed in backtest v2's record (2026-10-24).

`research/2026-10-20-free-survivorship-sources.md` section 6; the fixes are in
`universe_history.PRE_REVISION_CIKS` (pre-2014 renames) and `backtest_v2._load_facts`
(each member's predecessor registrant's filings).

Run with the pre-fix census and scores copied aside, then the rebuilt ones:

    python research/measurements/2026-10-24-v2-data-fix-effect.py --before <dir> [--after cache]

Reports, for each of the two builds (`_presample` 2012-2019 and 2020-2026):

* member-months scored before and after, and the change by census reason;
* which symbols stopped being unscoreable, and for what reason they were;
* **whether any month-stock that was already scored changed its composite** - the fixes are meant
  only to *add* names, and a moved composite elsewhere would mean a predecessor's figure reached a
  record that already had one.

Writes its summary beside this script as JSON.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = Path(__file__).with_suffix(".json")
BUILDS = {"presample_2012_2019": ("v2_census_presample.json", "v2_scores_presample.parquet"),
          "main_2020_2026": ("v2_census.json", "v2_scores.parquet")}


def census_totals(rows: list[dict]) -> dict:
    tot = collections.Counter()
    per = collections.defaultdict(collections.Counter)
    for r in rows:
        for reason, tickers in (r.get("not_scoreable") or {}).items():
            tot[reason] += len(tickers)
            for t in tickers:
                per[reason][t] += 1
    return {"members": sum(r.get("members", 0) for r in rows),
            "built": sum(r.get("built", 0) for r in rows),
            "scored": sum(r.get("scored", 0) for r in rows),
            "not_scoreable": dict(sorted(tot.items())),
            "per_symbol": {k: dict(v) for k, v in per.items()}}


def compare_scores(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """Did any (month, ticker) that was already scored change its composite?"""
    b = before.set_index(["month", "Ticker"])["Composite"]
    a = after.set_index(["month", "Ticker"])["Composite"]
    shared = b.index.intersection(a.index)
    diff = (a.loc[shared] - b.loc[shared]).abs()
    moved = diff[diff > 1e-9].sort_values(ascending=False)
    added = a.index.difference(b.index)
    removed = b.index.difference(a.index)
    by_ticker = collections.Counter(t for _, t in moved.index)
    return {"rows_before": int(len(b)), "rows_after": int(len(a)),
            "shared": int(len(shared)),
            "shared_composites_moved": int(len(moved)),
            "largest_move": float(moved.iloc[0]) if len(moved) else 0.0,
            "moved_by_ticker": dict(by_ticker.most_common(25)),
            "added_name_months": int(len(added)),
            "added_by_ticker": dict(collections.Counter(t for _, t in added).most_common(40)),
            "removed_name_months": int(len(removed)),
            "removed_by_ticker": dict(collections.Counter(t for _, t in removed).most_common(40))}


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True, help="directory holding the pre-fix census/scores")
    ap.add_argument("--after", default="cache")
    args = ap.parse_args()
    bdir, adir = ROOT / args.before, ROOT / args.after
    out = {"before_dir": args.before, "after_dir": args.after, "builds": {}}
    for name, (cfile, sfile) in BUILDS.items():
        if not (bdir / cfile).is_file() or not (adir / cfile).is_file():
            out["builds"][name] = {"skipped": "census missing on one side"}
            continue
        b = census_totals(json.loads((bdir / cfile).read_text(encoding="utf-8")))
        a = census_totals(json.loads((adir / cfile).read_text(encoding="utf-8")))
        reasons = sorted(set(b["not_scoreable"]) | set(a["not_scoreable"]))
        entry = {"scored_before": b["scored"], "scored_after": a["scored"],
                 "scored_change": a["scored"] - b["scored"],
                 "built_before": b["built"], "built_after": a["built"],
                 "members": b["members"],
                 "by_reason": {r: {"before": b["not_scoreable"].get(r, 0),
                                   "after": a["not_scoreable"].get(r, 0),
                                   "change": a["not_scoreable"].get(r, 0) - b["not_scoreable"].get(r, 0)}
                               for r in reasons},
                 "symbols_rescued": {}}
        for r in reasons:
            bp, apr = b["per_symbol"].get(r, {}), a["per_symbol"].get(r, {})
            rescued = {t: bp[t] - apr.get(t, 0) for t in bp if bp[t] - apr.get(t, 0) != 0}
            if rescued:
                entry["symbols_rescued"][r] = dict(sorted(rescued.items(), key=lambda kv: -kv[1]))
        if (bdir / sfile).is_file() and (adir / sfile).is_file():
            entry["scores"] = compare_scores(pd.read_parquet(bdir / sfile),
                                             pd.read_parquet(adir / sfile))
        out["builds"][name] = entry
    SUMMARY.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "symbols_rescued"}
                      for k, v in out["builds"].items()}, indent=1))
    for name, v in out["builds"].items():
        print(f"\n== {name} rescued ==")
        print(json.dumps(v.get("symbols_rescued", {}), indent=1))
    return out


if __name__ == "__main__":
    main()
