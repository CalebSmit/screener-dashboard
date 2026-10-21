"""Is what Size does to the composite a one-day reading? (factor-lab question 4)

research/2026-10-20-size-weight.md measures, on the 2026-10-20 run, that the other seven
categories leave the ranking almost neutral to size and that Size's 5 points add a small-company
tilt of about +0.13 (Spearman of the composite with the Size score). The 2026-08-31 note's second
"what would change my mind" item was exactly this: a reading on one day's payload. This repeats it
on every published payload in git history from the first data run of each week, using the
published category scores and weights of that payload.

The composite is approximated as the weighted mean of the categories a stock has (renormalised) -
no coverage discount - and the script reports how closely that reproduces each payload's own
Composite before reading anything off it. Structure only: no return is read (CLAUDE.md rule 4).

Run from the repo root:  python research/measurements/2026-10-20-size-exposure-over-time.py
Writes research/measurements/2026-10-20-size-exposure-over-time.json.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix(".json")
CATS = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]


def commits() -> list[tuple[str, str]]:
    log = subprocess.run(["git", "log", "--format=%h %ad", "--date=short", "--", "dashboard_data.js"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    seen, out = set(), []
    for line in reversed([x for x in log if x.strip()]):          # oldest first
        sha, day = line.split()
        week = pd.Timestamp(day).strftime("%G-%V")
        if week not in seen and day >= "2026-08-01":
            seen.add(week)
            out.append((sha, day))
    return out


def blend(scores: pd.DataFrame, w: dict) -> pd.Series:
    ws = pd.Series({c: float(w.get(c, 0)) for c in CATS})
    have = scores[CATS].notna()
    return (scores[CATS].fillna(0) * ws).sum(axis=1) / (have * ws).sum(axis=1)


def measure(sha: str) -> dict | None:
    text = subprocess.run(["git", "show", f"{sha}:dashboard_data.js"], cwd=ROOT, capture_output=True,
                          check=True).stdout.decode("utf-8")
    data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    t = pd.DataFrame(data["table_data"])
    if not all(c + "_score" in t for c in CATS):
        return None
    sc = t[[c + "_score" for c in CATS]].apply(pd.to_numeric, errors="coerce")
    sc.columns = CATS
    w = (data.get("weights") or {}).get("factor_weights") or {}
    if not w:
        return None
    comp = pd.to_numeric(t["Composite"], errors="coerce")
    approx = blend(sc, w)
    others = {**w, "size": 0}
    rest = [c for c in CATS if c != "size"]
    return {"run_timestamp": data["kpis"].get("run_timestamp"), "stocks": int(len(t)),
            "size_weight": w.get("size"),
            "approx_vs_published_spearman": round(float(approx.corr(comp, method="spearman")), 4),
            "composite_vs_size": round(float(comp.corr(sc["size"], method="spearman")), 3),
            "without_size_vs_size": round(float(blend(sc, others).corr(sc["size"], method="spearman")), 3),
            "size_vs_valuation": round(float(sc["size"].corr(sc["valuation"], method="spearman")), 3),
            "size_vs_investment": round(float(sc["size"].corr(sc["investment"], method="spearman")), 3),
            "size_vs_growth": round(float(sc["size"].corr(sc["growth"], method="spearman")), 3),
            "size_vs_rest_mean": round(float(sc["size"].corr(sc[rest].mean(axis=1), method="spearman")), 3),
            "top25_changed_without_size": 25 - len(set(approx.nlargest(25).index) & set(blend(sc, others).nlargest(25).index))}


def main() -> None:
    rows = []
    for sha, day in commits():
        r = measure(sha)
        if r:
            rows.append({"commit": sha, "date": day, **r})
            print(rows[-1])
    df = pd.DataFrame(rows)
    summary = {k: {"min": float(df[k].min()), "median": float(df[k].median()), "max": float(df[k].max())}
               for k in ("composite_vs_size", "without_size_vs_size", "size_vs_valuation", "size_vs_investment",
                         "size_vs_growth", "size_vs_rest_mean", "top25_changed_without_size",
                         "approx_vs_published_spearman")}
    OUT.write_text(json.dumps({"payloads": rows, "summary": summary}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
