"""The effect of residual momentum replacing Jensen's alpha, on one day's prices (2026-11-24).

Compares the page the 02:00 data run published (git HEAD's dashboard_data.js, run of 2026-11-24
02:00) with the full run made by the 06:00 session under the new metric. Both runs read the same
closing session (2026-11-23), so the difference is the methodology change plus a few hours of
analyst-estimate and option-quote drift - not a market move. Descriptive only: no forward return,
no IC (rules 4 and 5).

Usage: python research/measurements/2026-11-24-residual-momentum-effect.py [old_ref]
Writes the .json beside this file.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def load(text: str) -> dict:
    s = text[text.index("{"): text.rstrip().rstrip(";").rindex("}") + 1]
    return json.loads(s)


def frame(d: dict) -> pd.DataFrame:
    t = pd.DataFrame(d["table_data"]).set_index("Ticker")
    return t


def main():
    ref = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    old = load(subprocess.run(["git", "show", f"{ref}:dashboard_data.js"], cwd=ROOT,
                              capture_output=True, text=True, encoding="utf-8").stdout)
    new = load((ROOT / "dashboard_data.js").read_text(encoding="utf-8"))
    a, b = frame(old), frame(new)
    common = a.index.intersection(b.index)
    ra, rb = a.loc[common, "Rank"].astype(float), b.loc[common, "Rank"].astype(float)
    moves = (rb - ra)
    top_a = set(a.index[a["Rank"] <= 25]); top_b = set(b.index[b["Rank"] <= 25])
    top40_a = set(a.index[a["Rank"] <= 40]); top40_b = set(b.index[b["Rank"] <= 40])
    mom_a = a.loc[common, "momentum_score"].astype(float) if "momentum_score" in a else None
    mom_b = b.loc[common, "momentum_score"].astype(float) if "momentum_score" in b else None
    det = new["stock_detail"]
    have = [t for t, s in det.items() if (s.get("raw") or {}).get("residual_momentum") is not None]
    missing = sorted(t for t in det if t not in have)
    out = {
        "old": {"ref": ref, "run_timestamp": old["kpis"].get("run_timestamp")},
        "new": {"run_timestamp": new["kpis"].get("run_timestamp")},
        "n_common": int(len(common)),
        "spearman_rank": round(float(ra.corr(rb, method="spearman")), 4),
        "spearman_momentum_score": (round(float(mom_a.corr(mom_b, method="spearman")), 4)
                                    if mom_a is not None and mom_b is not None else None),
        "median_abs_rank_move": float(moves.abs().median()),
        "p90_abs_rank_move": float(moves.abs().quantile(0.9)),
        "max_abs_rank_move": int(moves.abs().max()),
        "largest_moves": {t: {"from": int(ra[t]), "to": int(rb[t])}
                          for t in moves.abs().sort_values(ascending=False).index[:10]},
        "top25_in": sorted(top_b - top_a), "top25_out": sorted(top_a - top_b),
        "top40_in": sorted(top40_b - top40_a), "top40_out": sorted(top40_a - top40_b),
        "top5_old": list(a.sort_values("Rank").index[:5]), "top5_new": list(b.sort_values("Rank").index[:5]),
        "residual_momentum_coverage": f"{len(have)}/{len(det)}",
        "residual_momentum_missing": missing,
        "missing_ranks": {t: {"from": int(ra[t]) if t in ra else None, "to": int(rb[t]) if t in rb else None}
                          for t in missing},
    }
    raw = pd.Series({t: det[t]["raw"]["residual_momentum"] for t in have}, dtype=float)
    r121 = pd.Series({t: (det[t].get("raw") or {}).get("return_12_1") for t in have}, dtype=float)
    out["spearman_resid_vs_return_12_1_raw"] = round(float(raw.corr(r121, method="spearman")), 4)
    out["residual_momentum_quantiles"] = {q: round(float(raw.quantile(q)), 3) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
    p = Path(__file__).with_suffix(".json")
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
