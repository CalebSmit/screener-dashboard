"""Deciding the OPEN proposal in research/2026-10-26-price-target-upside.md §6.2.

The proposal: take `price_target_upside`'s 13.5 points off the scored *level* of implied
upside and put them on a 90-day *change* in the consensus target, scaled by the target's own
level (MSCI `PTGF_C`'s construction) rather than by price.

The note left four things for the deciding session, in its own priority order:

  2. **"Check it is not `fy1_revision_3m` again. ... This is the first thing to measure, and
     it is cheap: both series are in the committed payload history."** If analysts move
     targets and FY1 EPS estimates together, the candidate is a duplicate and the honest
     outcome is no change at all.
  1. Re-run the §8.4 spanning regression - the Revisions category has a written rule in
     config.yaml requiring it before any further momentum-adjacent metric enters.
  3. Decide the fallback-cohort question (a 90-day lookback that does not exist for every
     stock on every run).

This script measures all three, plus the one the note did not ask for and which turns out to
matter most: **is the candidate itself a price signal?** The level is 71.4% price at 30 days
(note §1.2); a *change* in the target has no current price in it by construction, but if
analysts chase prices then Δtarget is still partly Δprice, which would put a momentum-adjacent
metric into Revisions - the exact thing config.yaml:128 forbids without §8.4.

Everything is read from the committed history of dashboard_data.js (one version per published
run date; each carries `pt_mean`, `price`, `sector`, `cat_scores` and `raw.fy1_revision_3m`
per stock) plus the newest run's parquets. Nothing here reads a forward return, an IC or a
backtest figure, so CLAUDE.md rules 4 and 5 do not bind. Imports no scoring module.

Run from the repo root:
    python research/measurements/2026-10-26-target-change-candidate.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix(".json")

CATS = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]
# Windows to build the candidate over. 90 is the proposal's own; 30 and 63 are the shorter
# variants S&P Global MI and MSCI document, and they have far more end dates available.
WINDOWS = (30, 63, 90)
# Tolerance when matching a run date to "W days before this one". A quarter of the window,
# floored at 3 days, is the same rule the note's term-structure table used.
def tol(w: int) -> int:
    return max(3, int(w * 0.25))


# --------------------------------------------------------------------- payload history
def payload_versions() -> list[tuple[str, str]]:
    """(date, sha) for the newest committed dashboard_data.js on each date, oldest first."""
    out = subprocess.run(
        ["git", "log", "--format=%H %ad", "--date=short", "--", "dashboard_data.js"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout
    seen, rows = set(), []
    for line in out.splitlines():
        sha, date = line.split()
        if date in seen:
            continue
        seen.add(date)
        rows.append((date, sha))
    return sorted(rows)


def read_version(sha: str) -> pd.DataFrame | None:
    """Per-stock frame from one committed payload: price, pt_mean, sector, fy1, cat scores."""
    blob = subprocess.run(["git", "show", f"{sha}:dashboard_data.js"],
                          cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8", "replace")
    try:
        data = json.loads(blob[blob.index("{"): blob.rindex("}") + 1])
    except ValueError:
        return None
    detail = data.get("stock_detail") or {}
    rows = {}
    for t, sd in detail.items():
        if not isinstance(sd, dict):
            continue
        price, pt = sd.get("price"), sd.get("pt_mean")
        raw = sd.get("raw") or {}
        cs = sd.get("cat_scores") or {}
        rec = {
            "price": float(price) if isinstance(price, (int, float)) and price > 0 else np.nan,
            "pt_mean": float(pt) if isinstance(pt, (int, float)) and pt > 0 else np.nan,
            "sector": sd.get("sector"),
            "n_analysts": sd.get("num_analysts"),
            "fy1_revision_3m": raw.get("fy1_revision_3m", np.nan),
            "price_target_upside": raw.get("price_target_upside", np.nan),
            "return_12_1": raw.get("return_12_1", np.nan),
            "return_6m": raw.get("return_6m", np.nan),
            "jensens_alpha": raw.get("jensens_alpha", np.nan),
            "volatility": raw.get("volatility", np.nan),
        }
        for c in CATS:
            rec[c + "_score"] = cs.get(c, np.nan)
        rows[t] = rec
    if len(rows) < 100:
        return None
    df = pd.DataFrame.from_dict(rows, orient="index")
    for c in df.columns:
        if c != "sector":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# --------------------------------------------------------------------- candidate construction
def candidate(now: pd.Series, then: pd.Series, scale: str) -> pd.Series:
    """The 90-day change in the consensus target, on one of three documented scalings.

    msci   - MSCI `PTGF_C`: the change over the mean absolute level of the two targets. A
             sign-robust symmetric percentage change; the proposal prefers this one.
    ratio  - S&P Global MI's documented signal, TP_t / TP_{t-W} - 1.
    price  - the note's own first draft, (TP_t - TP_{t-W}) / price_t, which §6.2 argues
             against because it puts the price back in the denominator. Measured so the
             argument can be checked rather than asserted.
    """
    if scale == "msci":
        denom = (now.abs() + then.abs()) / 2.0
    elif scale == "ratio":
        denom = then.abs()
    else:
        raise ValueError(scale)
    out = (now - then) / denom.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def spearman(a: pd.Series, b: pd.Series, min_n: int = 30) -> tuple[float, int]:
    ok = a.notna() & b.notna()
    if ok.sum() < min_n:
        return float("nan"), int(ok.sum())
    return float(a[ok].corr(b[ok], method="spearman")), int(ok.sum())


def within_sector(df: pd.DataFrame, x: str, y: str) -> tuple[float, int]:
    """Pool sector-demeaned percentile ranks - what a sector-relative screener scores."""
    parts = []
    for _, g in df.groupby("sector"):
        ok = g[x].notna() & g[y].notna()
        if ok.sum() < 5:
            continue
        sub = g[ok]
        rx, ry = sub[x].rank(pct=True), sub[y].rank(pct=True)
        parts.append(pd.DataFrame({"x": rx - rx.mean(), "y": ry - ry.mean()}))
    if not parts:
        return float("nan"), 0
    p = pd.concat(parts)
    return float(p["x"].corr(p["y"])), len(p)


def ols_r2(y: pd.Series, X: pd.DataFrame) -> tuple[float, int]:
    """R2 of y's rank on the ranks of X's columns. The §8.4 spanning construction."""
    ok = y.notna() & X.notna().all(axis=1)
    if ok.sum() < 50:
        return float("nan"), int(ok.sum())
    yy = y[ok].rank().values.astype(float)
    xx = X[ok].rank().values.astype(float)
    xx = np.column_stack([np.ones(len(xx)), xx])
    beta, *_ = np.linalg.lstsq(xx, yy, rcond=None)
    resid = yy - xx @ beta
    ss_tot = float(((yy - yy.mean()) ** 2).sum())
    return (1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")), int(ok.sum())


# --------------------------------------------------------------------- main
def main() -> None:
    versions = payload_versions()
    frames: dict[str, pd.DataFrame] = {}
    for date, sha in versions:
        f = read_version(sha)
        if f is not None:
            frames[date] = f
    dates = sorted(frames)
    print(f"read {len(dates)} committed payload versions: {dates[0]} .. {dates[-1]}\n")

    result = {"dates_read": dates, "n_dates": len(dates), "windows": {}}

    for w in WINDOWS:
        # Every end date for which a start date sits within tolerance of W days before it.
        pairs = []
        for i, b in enumerate(dates):
            want = pd.Timestamp(b) - pd.Timedelta(days=w)
            cand = [d for d in dates[:i] if abs((pd.Timestamp(d) - want).days) <= tol(w)]
            if not cand:
                continue
            a = min(cand, key=lambda d: abs((pd.Timestamp(d) - want).days))
            pairs.append((a, b))
        if not pairs:
            result["windows"][str(w)] = {"n_end_dates": 0}
            continue

        per_date = []
        for a, b in pairs:
            fa, fb = frames[a], frames[b]
            common = fa.index.intersection(fb.index)
            if len(common) < 100:
                continue
            d = fb.loc[common].copy()
            tnow, tthen = fb.loc[common, "pt_mean"], fa.loc[common, "pt_mean"]
            d["cand_msci"] = candidate(tnow, tthen, "msci")
            d["cand_ratio"] = candidate(tnow, tthen, "ratio")
            d["cand_price"] = ((tnow - tthen) / fb.loc[common, "price"]).replace(
                [np.inf, -np.inf], np.nan)
            d["dlog_price"] = np.log(fb.loc[common, "price"] / fa.loc[common, "price"])
            d["dlog_target"] = np.log(tnow / tthen)

            row: dict = {"from": a, "to": b,
                         "gap_days": (pd.Timestamp(b) - pd.Timestamp(a)).days,
                         "n_common": int(len(common))}
            # --- the decisive duplication check, matched horizon
            for sc in ("msci", "ratio", "price"):
                r, n = spearman(d["cand_" + sc], d["fy1_revision_3m"])
                rw, nw = within_sector(d, "cand_" + sc, "fy1_revision_3m")
                row[f"dup_{sc}_pooled"] = None if r != r else round(r, 4)
                row[f"dup_{sc}_within"] = None if rw != rw else round(rw, 4)
                row[f"dup_n"] = n
            # --- is the candidate itself a price signal?
            for other in ("dlog_price", "return_12_1", "return_6m", "jensens_alpha",
                          "volatility", "price_target_upside", "momentum_score",
                          "risk_score", "revisions_score"):
                if other not in d.columns:
                    continue
                r, _ = spearman(d["cand_msci"], d[other])
                rw, _ = within_sector(d, "cand_msci", other)
                row[f"vs_{other}_pooled"] = None if r != r else round(r, 4)
                row[f"vs_{other}_within"] = None if rw != rw else round(rw, 4)
            # --- coverage: how many stocks can the candidate be computed for?
            row["cand_coverage_n"] = int(d["cand_msci"].notna().sum())
            row["cand_coverage_pct"] = round(100.0 * d["cand_msci"].notna().mean(), 1)
            row["ptu_coverage_pct"] = round(100.0 * d["price_target_upside"].notna().mean(), 1)
            row["pct_target_unchanged_over_window"] = round(
                100.0 * float((tnow == tthen).mean()), 1)
            # --- the §8.4 spanning regression, on this end date
            X8 = d[[c + "_score" for c in CATS]]
            r2_8, n8 = ols_r2(d["cand_msci"], X8)
            r2_mom, _ = ols_r2(d["cand_msci"], d[["momentum_score"]])
            row["spanning_r2_all8"] = None if r2_8 != r2_8 else round(r2_8, 4)
            row["spanning_unspanned_pct"] = None if r2_8 != r2_8 else round(100 * (1 - r2_8), 1)
            row["spanning_r2_momentum_only"] = None if r2_mom != r2_mom else round(r2_mom, 4)
            row["spanning_n"] = n8
            # the same two numbers for the incumbent, for a like-for-like comparison
            r2i, _ = ols_r2(d["price_target_upside"], X8)
            r2im, _ = ols_r2(d["price_target_upside"], d[["momentum_score"]])
            row["incumbent_spanning_r2_all8"] = None if r2i != r2i else round(r2i, 4)
            row["incumbent_spanning_r2_momentum_only"] = None if r2im != r2im else round(r2im, 4)
            per_date.append(row)

        def agg(key: str) -> dict | None:
            vals = [r[key] for r in per_date if r.get(key) is not None]
            if not vals:
                return None
            return {"median": round(float(np.median(vals)), 4),
                    "mean": round(float(np.mean(vals)), 4),
                    "min": round(float(np.min(vals)), 4),
                    "max": round(float(np.max(vals)), 4),
                    "n_dates": len(vals)}

        keys = [k for k in (per_date[0].keys() if per_date else [])
                if k.startswith(("dup_", "vs_", "spanning_", "incumbent_", "cand_cov",
                                 "ptu_cov", "pct_target"))
                and not k.endswith("_n")]
        result["windows"][str(w)] = {
            "n_end_dates": len(per_date),
            "tolerance_days": tol(w),
            "per_date": per_date,
            "summary": {k: agg(k) for k in keys},
        }

    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")

    # ------------------------------------------------------------------ report
    for w in WINDOWS:
        block = result["windows"][str(w)]
        print(f"=== WINDOW {w} DAYS  ({block.get('n_end_dates', 0)} end dates, "
              f"tolerance +-{block.get('tolerance_days', '-')}d) ===")
        if not block.get("n_end_dates"):
            print("   no end date has a start date in range\n")
            continue
        s = block["summary"]

        def line(label: str, key: str) -> None:
            v = s.get(key)
            if not v:
                print(f"   {label:46s} -")
                return
            print(f"   {label:46s} median {v['median']:+.4f}   "
                  f"[{v['min']:+.4f}, {v['max']:+.4f}]  over {v['n_dates']} dates")

        print("  1. DUPLICATION with fy1_revision_3m (the decisive check)")
        for sc in ("msci", "ratio", "price"):
            line(f"{sc} scaling, pooled", f"dup_{sc}_pooled")
            line(f"{sc} scaling, WITHIN SECTOR", f"dup_{sc}_within")
        print("  2. IS THE CANDIDATE ITSELF A PRICE SIGNAL?")
        for other, lab in (("dlog_price", "d log price over the same window"),
                           ("return_12_1", "return_12_1"),
                           ("return_6m", "return_6m"),
                           ("jensens_alpha", "jensens_alpha"),
                           ("volatility", "volatility"),
                           ("momentum_score", "momentum category score"),
                           ("risk_score", "risk category score"),
                           ("revisions_score", "revisions category score"),
                           ("price_target_upside", "the incumbent level")):
            line(lab + ", within sector", f"vs_{other}_within")
        print("  3. SPANNING (SS8.4 construction: rank on the 8 category ranks)")
        line("candidate unspanned %", "spanning_unspanned_pct")
        line("candidate R2 on all 8", "spanning_r2_all8")
        line("candidate R2 on momentum alone", "spanning_r2_momentum_only")
        line("INCUMBENT R2 on all 8", "incumbent_spanning_r2_all8")
        line("INCUMBENT R2 on momentum alone", "incumbent_spanning_r2_momentum_only")
        print("  4. COVERAGE")
        line("candidate coverage %", "cand_coverage_pct")
        line("incumbent coverage %", "ptu_coverage_pct")
        line("target exactly unchanged over window %", "pct_target_unchanged_over_window")
        print()

    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
