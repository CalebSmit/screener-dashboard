"""What a scoring change did to the ranking's factor exposures: one command, before and after.

`research/2026-11-02-growth-exposure-drift.md` section 6.4 (2026-11-03) asked that every scoring
change record "the composite's correlation with each category score before and after", because the
2026-10-09 changes moved the ranking's lean on growth in a step nobody saw for three weeks. It stayed
prose, and the 2026-11-24 Momentum change did not record it - the 2026-11-25 synthesis session found
that change had moved the lean +0.026 -> +0.006 (note section 7). This makes the habit one command.

    python scripts/exposure_shift.py <before> <after>

Each side is a git revision (its committed `dashboard_data.js`) or a path to a payload file. For a
change made today, compare the morning's published page (`HEAD` before your commit, or `main`) with a
full run under the change (the working-tree `dashboard_data.js` after `run_screener.py`). Both pages
should read the same closing session, or the difference mixes the change with a day of markets.

Prints the composite's Spearman correlation with each of the eight category scores (the first row is
the brief's "lean on growth"), then the category pairs that moved most. Rank correlations of
published scores only - no return, IC or backtest figure (rules 4 and 5). It reports; it decides
nothing.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CATS = ("valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment")


def load_table(ref: str) -> pd.DataFrame:
    """`table_data` from a payload file path, or from `dashboard_data.js` at a git revision."""
    path = Path(ref)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
    else:
        text = subprocess.run(["git", "show", f"{ref}:dashboard_data.js"], cwd=ROOT,
                              capture_output=True, check=True).stdout.decode("utf-8")
    data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    return pd.DataFrame(data["table_data"])


def _rho(t: pd.DataFrame, a: str, b: str) -> float:
    s = t[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
    return float(s[a].corr(s[b], method="spearman"))


def exposures(t: pd.DataFrame) -> dict[str, float]:
    """The composite's rank correlation with each category score."""
    return {c: _rho(t, "Composite", f"{c}_score") for c in CATS}


def pairs(t: pd.DataFrame) -> dict[str, float]:
    """All 28 category-score pairs."""
    return {f"{a}~{b}": _rho(t, f"{a}_score", f"{b}_score") for a, b in itertools.combinations(CATS, 2)}


def shift(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    eb, ea, pb, pa = exposures(before), exposures(after), pairs(before), pairs(after)
    moved = sorted(((k, pb[k], pa[k]) for k in pb), key=lambda r: -abs(r[2] - r[1]))
    return {"exposure": {c: (eb[c], ea[c]) for c in CATS},
            "pairs_moved_most": moved[:8]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("before")
    ap.add_argument("after")
    args = ap.parse_args(argv)
    out = shift(load_table(args.before), load_table(args.after))
    print("Composite against each category (Spearman): before -> after (change)")
    for c, (b, a) in out["exposure"].items():
        print(f"  {c:<11} {b:+.3f} -> {a:+.3f} ({a - b:+.3f})")
    print("Category pairs that moved most:")
    for k, b, a in out["pairs_moved_most"]:
        print(f"  {k:<22} {b:+.3f} -> {a:+.3f} ({a - b:+.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
