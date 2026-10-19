"""Is the pair's position on the revenue-minus-asset-growth spread stable, or one run's artifact?

Written for research/2026-10-19-revenue-growth-vs-asset-growth.md section 8. Its central claim -
that `revenue_growth` and `asset_growth` together are a position on the SPREAD between them
rather than on either level - is measured on a single run (2026-10-19 02:00). The Growth weight
review left exactly this open about its own headline number
(research/2026-10-12-growth-category.md section 11: "What the ranking's growth exposure does over
time. -0.01 is one run.").

So: replay the same three correlations over every committed `dashboard_data.js` in the chosen
window, reading each payload out of git rather than from disk, and report the series. A claim that
holds on one run and wanders on the others is a claim about that run.

Three numbers per run, all Spearman, all on the stocks carrying both metrics:

  pair_vs_revenue  the pair's composite points against raw revenue growth   (near 0 is the claim)
  pair_vs_asset    the pair's composite points against raw asset growth     (strongly negative)
  pair_vs_spread   the pair's composite points against the within-sector
                   rank spread, revenue rank minus asset rank              (near +0.9 is the claim)
  comp_vs_spread   the finished composite against the same spread          (the inherited position)

The weights are read from each commit's own `config.yaml`, not from today's, so a run from before
a reweight is scored at the weights it was actually published under.

Structure only - no forward return, no IC, no backtest number (CLAUDE.md rules 4 and 5). Reads
git history; writes only the JSON beside it.

Run from the repo root:
    python research/measurements/2026-10-19-spread-exposure-over-time.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix(".json")
RG, AG = "revenue_growth", "asset_growth"


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", check=True).stdout


def commits(limit: int) -> list[tuple[str, str]]:
    out = git("log", "--format=%h %ad", "--date=short", "--", "dashboard_data.js")
    rows = [l.split() for l in out.splitlines() if l.strip()]
    return [(h, d) for h, d in rows][:limit]


def payload_at(sha: str) -> dict | None:
    try:
        text = git("show", f"{sha}:dashboard_data.js")
    except subprocess.CalledProcessError:
        return None
    try:
        return json.loads(text[text.index("{"): text.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return None


def weights_at(sha: str) -> tuple[float, float] | None:
    """(points on revenue_growth, points on asset_growth) as that commit configured them."""
    try:
        cfg = yaml.safe_load(git("show", f"{sha}:config.yaml"))
    except (subprocess.CalledProcessError, yaml.YAMLError):
        return None
    try:
        fw, mw = cfg["factor_weights"], cfg["metric_weights"]
        return (fw["growth"] * mw["growth"][RG] / 100.0,
                fw["investment"] * mw["investment"][AG] / 100.0)
    except (KeyError, TypeError):
        return None


def one_run(sha: str) -> dict | None:
    data, pts = payload_at(sha), weights_at(sha)
    if data is None or pts is None:
        return None
    pts_rg, pts_ag = pts
    rows = []
    for ticker, s in (data.get("stock_detail") or {}).items():
        raw, pct = s.get("raw") or {}, s.get("pct") or {}
        rows.append({"sector": s.get("sector"), "composite": s.get("composite"),
                     "rg_raw": raw.get(RG), "ag_raw": raw.get(AG),
                     "rg_pct": pct.get(RG), "ag_pct": pct.get(AG)})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c != "sector":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["rg_raw", "ag_raw", "rg_pct", "ag_pct"])
    if len(df) < 100:
        return None
    df["pts_pair"] = (df["rg_pct"] * pts_rg + df["ag_pct"] * pts_ag) / 100.0
    df["spread_rank"] = (df.groupby("sector")["rg_raw"].rank(pct=True)
                         - df.groupby("sector")["ag_raw"].rank(pct=True))
    sp = lambda a, b: round(float(df[a].corr(df[b], method="spearman")), 4)
    comp = df.dropna(subset=["composite"])
    return {
        "n": int(len(df)),
        "points_revenue_growth": round(pts_rg, 3),
        "points_asset_growth": round(pts_ag, 3),
        "pair_vs_revenue": sp("pts_pair", "rg_raw"),
        "pair_vs_asset": sp("pts_pair", "ag_raw"),
        "pair_vs_spread": sp("pts_pair", "spread_rank"),
        "comp_vs_spread": (round(float(comp["composite"].corr(comp["spread_rank"],
                                                              method="spearman")), 4)
                           if len(comp) >= 100 else None),
        "raw_overlap": sp("rg_raw", "ag_raw"),
    }


def main() -> None:
    picked = commits(limit=40)
    print(f"replaying {len(picked)} committed payloads\n")
    hdr = (f"{'date':>11} {'sha':>8} {'n':>4} {'rg pts':>7} {'ag pts':>7} "
           f"{'vs rev':>7} {'vs asset':>9} {'vs SPREAD':>10} {'comp':>7} {'overlap':>8}")
    print(hdr); print("-" * len(hdr))
    series, skipped = [], []
    for sha, date in picked:
        r = one_run(sha)
        if r is None:
            skipped.append((sha, date))
            continue
        r["sha"], r["date"] = sha, date
        series.append(r)
        print(f"{date:>11} {sha:>8} {r['n']:>4} {r['points_revenue_growth']:>7.2f} "
              f"{r['points_asset_growth']:>7.2f} {r['pair_vs_revenue']:>+7.3f} "
              f"{r['pair_vs_asset']:>+9.3f} {r['pair_vs_spread']:>+10.3f} "
              f"{(r['comp_vs_spread'] if r['comp_vs_spread'] is not None else float('nan')):>+7.3f} "
              f"{r['raw_overlap']:>+8.3f}")
    if skipped:
        print(f"\nskipped {len(skipped)} commit(s) whose payload or config could not be read: "
              f"{', '.join(f'{d} {s}' for s, d in skipped)}")

    s = pd.DataFrame(series)
    res = {
        "runs": len(series),
        "span": [series[-1]["date"], series[0]["date"]] if series else None,
        "skipped": [{"sha": h, "date": d} for h, d in skipped],
        "series": series,
    }
    print("\n--- is the claim stable? ---")
    for col, claim in (("pair_vs_spread", "the pair is a position on the spread (near +0.9)"),
                       ("pair_vs_revenue", "and not on revenue growth (near 0)"),
                       ("pair_vs_asset", "asset-growth exposure survives (strongly negative)"),
                       ("comp_vs_spread", "the composite inherits it")):
        v = s[col].dropna()
        res[col] = {"min": round(float(v.min()), 4), "max": round(float(v.max()), 4),
                    "median": round(float(v.median()), 4), "n": int(len(v))}
        print(f"  {claim:<52s} min {v.min():+.3f}  median {v.median():+.3f}  max {v.max():+.3f}")

    OUT.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
