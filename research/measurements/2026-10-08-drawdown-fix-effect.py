"""What the `max_drawdown_1y` price-path fix did to the published numbers.

Compares the rebuilt payload against the one the 02:00 run published (read from git), so
the expected effect in METHODOLOGY_CHANGELOG.md 2026-10-08 is checked rather than asserted.

    python research/measurements/2026-10-08-drawdown-fix-effect.py
"""
import json
import re
import subprocess
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _parse(txt):
    return json.loads(re.sub(r"^[^=]*=\s*", "", txt.strip().rstrip(";"), count=1))


def load_new():
    with open("dashboard_data.js", encoding="utf-8") as fh:
        return _parse(fh.read())


def load_old(ref="HEAD"):
    out = subprocess.run(["git", "show", f"{ref}:dashboard_data.js"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    out.check_returncode()
    return _parse(out.stdout)


def series(payload, field, src="raw"):
    d = {}
    for t, s in payload["stock_detail"].items():
        v = (s.get(src) or {}).get(field)
        if v is not None:
            d[t] = v
    return pd.Series(d, dtype=float)


def main():
    new, old = load_new(), load_old()
    print(f"stocks: old {len(old['stock_detail'])}, new {len(new['stock_detail'])}")

    # --- the metric itself
    o, n = series(old, "max_drawdown_1y"), series(new, "max_drawdown_1y")
    both = o.index.intersection(n.index)
    diff = (n[both] - o[both]) * 100
    print(f"\nmax_drawdown_1y, {len(both)} stocks in both runs")
    print(f"  new is a SMALLER fall for {(diff > 0).sum()} of {len(both)}, "
          f"larger for {(diff < 0).sum()}, unchanged for {(diff == 0).sum()}")
    print(f"  change in pp: median {diff.median():+.3f}  mean {diff.mean():+.3f}  "
          f"min {diff.min():+.3f}  max {diff.max():+.3f}")
    worst = diff.sort_values(ascending=False).head(5)
    for t, v in worst.items():
        print(f"    {t:6s} {o[t]*100:7.2f}% -> {n[t]*100:7.2f}%  ({v:+.2f}pp)")

    # --- the percentile the screener actually scores
    op, np_ = series(old, "max_drawdown_1y", "pct"), series(new, "max_drawdown_1y", "pct")
    b2 = op.index.intersection(np_.index)
    if len(b2):
        pd_ = (np_[b2] - op[b2])
        print(f"\nits sector percentile: Spearman {op[b2].corr(np_[b2], method='spearman'):.6f}, "
              f"{(pd_.abs() > 0.5).sum()} of {len(b2)} move by more than half a point, "
              f"max move {pd_.abs().max():.1f}")

    # --- category and composite
    for field, label in [("risk", "Risk category score"), ("Composite", "composite")]:
        oo = series(old, field) if field == "Composite" else None
        if field == "Composite":
            oo = pd.Series({t: s["raw"].get("Composite", s.get("composite"))
                            for t, s in old["stock_detail"].items()}, dtype=float)
            nn = pd.Series({t: s["raw"].get("Composite", s.get("composite"))
                            for t, s in new["stock_detail"].items()}, dtype=float)
        else:
            oo = pd.Series({t: (s.get("cat") or {}).get(field)
                            for t, s in old["stock_detail"].items()}, dtype=float)
            nn = pd.Series({t: (s.get("cat") or {}).get(field)
                            for t, s in new["stock_detail"].items()}, dtype=float)
        oo, nn = oo.dropna(), nn.dropna()
        b3 = oo.index.intersection(nn.index)
        if not len(b3):
            print(f"\n{label}: not comparable from this payload shape")
            continue
        d3 = (nn[b3] - oo[b3]).abs()
        print(f"\n{label}: {len(b3)} stocks, {(d3 > 0.05).sum()} move by more than 0.05, "
              f"median |move| {d3.median():.3f}, max {d3.max():.3f}")

    # --- the ranking, which is what a reader sees
    tbl_o = {r["Ticker"]: r["Rank"] for r in old["table_data"] if "Rank" in r}
    tbl_n = {r["Ticker"]: r["Rank"] for r in new["table_data"] if "Rank" in r}
    common = set(tbl_o) & set(tbl_n)
    moves = pd.Series({t: tbl_n[t] - tbl_o[t] for t in common}, dtype=float)
    print(f"\nrank: {int((moves != 0).sum())} of {len(common)} stocks move, "
          f"median |move| {moves.abs().median():.1f}, max {int(moves.abs().max())} places")
    print(f"  Spearman of the two rankings "
          f"{pd.Series(tbl_o).reindex(sorted(common)).corr(pd.Series(tbl_n).reindex(sorted(common)), method='spearman'):.6f}")
    top_o = [r["Ticker"] for r in sorted(old["table_data"], key=lambda r: r["Rank"])[:10]]
    top_n = [r["Ticker"] for r in sorted(new["table_data"], key=lambda r: r["Rank"])[:10]]
    print(f"  top 10 before: {top_o}")
    print(f"  top 10 after : {top_n}")

    # --- the new inputs
    have = sum(1 for s in new["stock_detail"].values()
               if (s.get("inp") or {}).get("mdd_peak") is not None)
    print(f"\nnew inputs: mdd_peak published for {have} of {len(new['stock_detail'])} stocks")
    bad = 0
    for t, s in new["stock_detail"].items():
        inp = s.get("inp") or {}
        pk, tr, pub = inp.get("mdd_peak"), inp.get("mdd_trough"), s["raw"].get("max_drawdown_1y")
        if pk is None or tr is None or pub is None:
            continue
        if abs((tr - pk) / pk - pub) > 1e-4:
            bad += 1
    print(f"  pairs that do NOT rebuild the published value: {bad}")


if __name__ == "__main__":
    main()
