"""How residual momentum fits the rest of the screener, on the first data-loop run that carries it.

Synthesis measurement for NIGHTLY_LOG 2026-11-25. Rank (Spearman) correlations only, read from the
published payloads as committed - no forward return, IC or backtest figure (rules 4 and 5).

Before: the 02:00 data-loop page of 2026-11-24 (0679feb), the last scored with jensens_alpha.
Isolated: 87b7be9, the 11-24 06:00 full run - same closing session as Before, residual momentum.
After:  the 02:00 data-loop page of 2026-11-25 (c1dd9e1), the first data-loop run with residual momentum.

Checks the two predictions `research/2026-11-20-jensens-alpha-in-momentum.md` made (sections 2 and 5):
  (a) the new metric is ~0.77 like return_12_1 where jensens_alpha was ~0.97;
  (b) it removes a second low-beta reward inside Momentum (Momentum's score loaded on beta through the
      alpha's `- beta x (m12 - rf)` term).
and measures what the note did not: the 28 category pairs before and after, and the budgeted
revisions~momentum pair (tests/test_revisions_momentum_budget.py, bar 0.35).

Run: python research/measurements/2026-11-25-momentum-fit.py  (writes the .json beside it)
"""
from __future__ import annotations

import itertools
import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BEFORE, AFTER = "0679feb", "c1dd9e1"
# Same closing session as BEFORE, rescored with residual momentum by the 11-24 06:00 session: isolates
# the metric change from a day of market moves.
ISOLATED = "87b7be9"
CATS = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]
MOM_METRICS = ["return_12_1", "return_6m", "jensens_alpha", "residual_momentum"]


def payload(sha: str) -> dict:
    text = subprocess.run(["git", "show", f"{sha}:dashboard_data.js"], cwd=ROOT, capture_output=True,
                          check=True).stdout.decode("utf-8")
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


def frame(d: dict) -> pd.DataFrame:
    t = pd.DataFrame(d["table_data"]).set_index("Ticker")
    for c in CATS:
        t[c] = pd.to_numeric(t[f"{c}_score"], errors="coerce")
    t["composite"] = pd.to_numeric(t["Composite"], errors="coerce")
    raw = pd.DataFrame({k: v.get("raw", {}) for k, v in d["stock_detail"].items()}).T
    for m in MOM_METRICS + ["beta", "volatility", "max_drawdown_1y"]:
        t[m] = pd.to_numeric(raw[m], errors="coerce") if m in raw else float("nan")
    return t


def rho(t: pd.DataFrame, a: str, b: str) -> float | None:
    s = t[[a, b]].dropna()
    return None if len(s) < 30 else round(float(s[a].corr(s[b], method="spearman")), 3)


def main() -> None:
    out: dict = {"shas": {"before": BEFORE, "isolated": ISOLATED, "after": AFTER}}
    frames = {"before": frame(payload(BEFORE)), "isolated": frame(payload(ISOLATED)),
              "after": frame(payload(AFTER))}
    for side, t in frames.items():
        rec = {"n": int(len(t)),
               "coverage": {m: int(t[m].notna().sum()) for m in MOM_METRICS},
               "momentum_metric_vs_return_12_1": {m: rho(t, m, "return_12_1") for m in MOM_METRICS[1:]},
               "momentum_metric_vs_return_6m": {m: rho(t, m, "return_6m") for m in MOM_METRICS[2:]},
               "vs_beta": {m: rho(t, m, "beta") for m in MOM_METRICS + ["momentum", "risk"]},
               "vs_volatility": {m: rho(t, m, "volatility") for m in MOM_METRICS + ["momentum"]},
               # The ranking's exposure to each category; composite~growth is the brief's "lean on growth".
               "composite_vs_category": {c: rho(t, "composite", c) for c in CATS},
               "category_pairs": {f"{a}~{b}": rho(t, a, b) for a, b in itertools.combinations(CATS, 2)}}
        out[side] = rec
    for side in ("isolated", "after"):
        moved = {k: (out["before"]["category_pairs"][k], out[side]["category_pairs"][k])
                 for k in out[side]["category_pairs"]}
        out[f"pairs_moved_most_{side}"] = sorted(
            ({"pair": k, "before": b, side: a, "change": round(a - b, 3)} for k, (b, a) in moved.items()
             if a is not None and b is not None), key=lambda r: -abs(r["change"]))[:8]
    (Path(__file__).with_suffix(".json")).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
