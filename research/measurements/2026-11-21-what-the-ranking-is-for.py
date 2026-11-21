"""CLAUDE.md 0.15: what the ranking is for - section 4 of research/2026-11-19-what-the-ranking-is-for.md.

Implements the criteria pre-registered in that note's section 3 (fixed 2026-11-19) and the
operational choices written into its section 4a, which were committed BEFORE this script was first
run (the git history is the proof). Nothing here may be changed after the first run except to fix a
bug, and a bug fix is recorded in the note.

  A  (screen)        top 40 by composite in next month's bottom decile: per-month share, one-sided
                     95% upper bound below 10% in BOTH eras.
  B  (slow forecast) rank IC at 6 and 12 months on non-overlapping windows: pooled 95% interval
                     above zero AND the same sign in both eras; power stated first, and a failure
                     the design could not have detected (0.05 at 80% power) is "undecidable".
  C  (fewer factors) only if B holds: a category's own IC above the composite's, paired by window,
                     interval at 0.05/8 excluding zero in both eras.
  D  the answer if A-C all fail or are undecidable.
  Positive control   the engine's registered look-ahead control, run through the engine's own
                     pairing in this same process; a result counts only if it is seen.

Reuses v2_experiments / backtest_v2_returns throughout - never a second implementation of a
ranking, a return or a control. Writes nothing to the ledger (this is a measurement, not an
experiment row), so no research-engine row goes stale. Backtest observation, not decision-grade
before 2027-02-11 (CLAUDE.md rule 5); no weight is fitted to it (note section 3).

Run: python research/measurements/2026-11-21-what-the-ranking-is-for.py   (~3-5 minutes)
Writes the .json beside this file.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import backtest_v2_returns as br  # noqa: E402
import v2_experiments as ve  # noqa: E402

OUT = Path(__file__).with_suffix(".json")

ERAS = {"2012-2019": ("2012-01-31", "2019-12-31"),      # ve.PRESAMPLE
        "2020-2026": ("2020-01-31", "2026-08-31")}      # ve.SAMPLE
TOP_N = 40
HORIZONS = (6, 12)
MDE_TARGET = 0.05            # the IC the note asks the design to be able to see (Grinold & Kahn's "good")
C_ALPHA = 0.05 / 8           # Bonferroni across the eight categories (note section 3, C)


def nw_se(x: np.ndarray, lags: int = ve.NW_LAGS) -> float:
    """Newey-West (Bartlett) standard error of the mean - the engine's own _dependence formula."""
    n = len(x)
    d = x - x.mean()
    g0 = float(d @ d) / n
    lrv = g0
    for k in range(1, min(lags, n - 1) + 1):
        lrv += 2 * (1 - k / (lags + 1)) * float(d[k:] @ d[:-k]) / n
    return math.sqrt(max(lrv, 1e-18) / n)


def stats(x, alpha_two_sided: float = 0.05) -> dict:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    if n < 3:
        return {"n": n}
    m, sd = float(x.mean()), float(x.std(ddof=1))
    se = sd / math.sqrt(n)
    q = ve.t_critical(alpha_two_sided, n - 1)
    out = {"n": n, "mean": round(m, 5), "sd": round(sd, 5), "se": round(se, 5),
           "t": round(m / se, 2) if se > 0 else None,
           "ci": [round(m - q * se, 5), round(m + q * se, 5)], "ci_alpha": alpha_two_sided}
    if n >= 10:
        out["se_nw"] = round(nw_se(x), 5)
    return out


def in_era(month: str, era: str) -> bool:
    a, b = ERAS[era]
    return a <= month <= b


def main() -> None:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "research/measurements/"
                            "2026-11-21-what-the-ranking-is-for.py", *ve.RESULT_PATHS], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()

    months = ve.load_months()
    ms = sorted(months)
    cfg = ve.base_config()
    print(f"ranking {len(ms)} months with the baseline configuration ...", flush=True)
    scores = ve.rank_all(months, cfg)
    cats = [c for c in scores.columns if c.endswith("_score")]

    daily = pd.read_parquet(ROOT / "cache" / "v2_daily.parquet")
    last_px = pd.Timestamp(pd.to_datetime(daily["date"]).max())
    after = []
    m = pd.Timestamp(ms[-1])
    while True:
        m = m + pd.offsets.MonthEnd(1)
        if m > last_px:
            break
        after.append(m.strftime("%Y-%m-%d"))
    grid = ms + after                                     # month-ends with a price on or before them
    prices = br.month_end_prices(daily, grid)
    pos = {g: i for i, g in enumerate(grid)}

    def fwd(month: str, h: int) -> pd.Series | None:
        i = pos[month]
        if i + h >= len(grid):
            return None
        return prices.iloc[i + h] / prices.iloc[i] - 1

    by_month = dict(tuple(scores.groupby("month")))

    # ---------------------------------------------------------------- positive control (first)
    ctrl = ve.load_controls()[0]
    c_scores, c_prov = ve.control_scores(ctrl)
    pm_c = ve.per_month(c_scores, ve.forward_returns(ms), baseline_scores=scores)
    c_res = {"summary": ve.summarise(pm_c)}
    n_tests = len(ve.load_registry())
    c_v = ve.control_verdict(ctrl, c_res, n_tests)
    control = {"id": ctrl["id"], "n_tests": n_tests, "provenance": c_prov,
               "sees": c_v.get("sees"), "mean": c_v.get("mean"), "t": c_v.get("t"),
               "n": c_v.get("n"), "in_band": c_v.get("in_band"), "words": c_v.get("words")}
    print(f"positive control: sees={control['sees']} mean={control['mean']} t={control['t']}", flush=True)

    # ---------------------------------------------------------------- A: the screen
    a_rows = []
    for m in ms:
        r = fwd(m, 1)
        if r is None:
            continue
        g = by_month[m].assign(ret=r.reindex(by_month[m]["Ticker"].values).values)
        g = g.dropna(subset=["ret", "Composite"])
        if len(g) < ve.MIN_STOCKS_A_MONTH:
            continue
        dec = np.ceil(g["ret"].rank(pct=True, method="first") * 10).clip(1, 10)
        # "top 40 by composite", literally: highest Composite first, ties by ticker
        top = g.sort_values(["Composite", "Ticker"], ascending=[False, True]).head(TOP_N)
        in_bottom = (dec.loc[top.index] == 1)
        a_rows.append({"month": m, "n_ranked": int(len(g)), "n_top": int(len(top)),
                       "share_bottom": float(in_bottom.mean()),
                       "share_top_decile": float((dec.loc[top.index] == 10).mean())})
    a = pd.DataFrame(a_rows)
    A = {"per_era": {}, "per_month": a.round(5).to_dict("records")}
    holds_all = True
    for era in ERAS:
        x = a.loc[a["month"].map(lambda s: in_era(s, era)), "share_bottom"].to_numpy(float)
        st = stats(x)
        se = max(st["se"], st.get("se_nw", 0.0))
        upper = st["mean"] + ve.t_critical(0.10, st["n"] - 1) * se     # one-sided 95%
        st.update(se_used=round(se, 5), upper_one_sided_95=round(upper, 5),
                  holds=bool(upper < 0.10),
                  top_decile_share=round(float(a.loc[a["month"].map(lambda s: in_era(s, era)),
                                                     "share_top_decile"].mean()), 5))
        A["per_era"][era] = st
        holds_all &= st["holds"]
    pooled = stats(a["share_bottom"])
    A["pooled"] = pooled
    A["holds"] = bool(holds_all)

    # ---------------------------------------------------------------- B: the slow forecast
    B = {}
    win_ic: dict[int, pd.DataFrame] = {}
    for h in HORIZONS:
        rows = []
        for era in ERAS:
            em = [m for m in ms if in_era(m, era)]
            for m in em[::h]:
                r = fwd(m, h)
                if r is None:
                    continue
                g = by_month[m].assign(ret=r.reindex(by_month[m]["Ticker"].values).values)
                g = g.dropna(subset=["ret", "Composite"])
                if len(g) < ve.MIN_STOCKS_A_MONTH:
                    continue
                row = {"era": era, "start": m, "n": int(len(g)),
                       "ic": float(g["Composite"].rank().corr(g["ret"].rank()))}
                for c in cats:
                    ok = g[c].notna()
                    if ok.sum() >= ve.MIN_STOCKS_A_MONTH:
                        row[c] = float(g.loc[ok, c].rank().corr(g.loc[ok, "ret"].rank()))
                rows.append(row)
        w = pd.DataFrame(rows)
        win_ic[h] = w
        pooled = stats(w["ic"])
        # power, stated before the verdict: the smallest IC this many windows detects 80% of the
        # time at two-sided 5%, from the windows' own dispersion (the mean plays no part)
        df = pooled["n"] - 1
        mde = (ve.t_critical(0.05, df) + ve.t_critical(0.40, df)) * pooled["se"]
        eras = {era: stats(w.loc[w["era"] == era, "ic"]) for era in ERAS}
        means = [eras[e].get("mean") for e in ERAS]
        same_sign = all(v is not None and v != 0 for v in means) and len({np.sign(v) for v in means}) == 1
        holds = bool(pooled["ci"][0] > 0 and same_sign)
        verdict = "holds" if holds else ("undecidable" if mde > MDE_TARGET else "fails")
        B[f"{h}m"] = {"windows": int(len(w)), "detectable_80pct": round(mde, 4),
                      "can_detect_0.05": bool(mde <= MDE_TARGET), "pooled": pooled, "eras": eras,
                      "same_sign": bool(same_sign), "verdict": verdict,
                      "per_window": w.round(5).to_dict("records")}
    B["holds"] = any(B[f"{h}m"]["verdict"] == "holds" for h in HORIZONS)

    # ---------------------------------------------------------------- C: fewer factors
    C = {"ran": False, "why": "B did not hold - C is tested only if B holds (note section 3)"}
    if B["holds"]:
        C = {"ran": True, "horizons": {}}
        for h in HORIZONS:
            if B[f"{h}m"]["verdict"] != "holds":
                continue
            w = win_ic[h]
            res = {}
            for c in cats:
                if c not in w:
                    continue
                per = {}
                for era in ERAS:
                    sl = w[w["era"] == era].dropna(subset=[c])
                    per[era] = stats(sl[c] - sl["ic"], alpha_two_sided=C_ALPHA)
                res[c] = {"eras": per, "holds": all(p.get("n", 0) >= 3 and p["ci"][0] > 0
                                                    for p in per.values())}
            C["horizons"][f"{h}m"] = res

    D = not (A["holds"] or B["holds"] or (C.get("ran") and any(
        v["holds"] for hz in C.get("horizons", {}).values() for v in hz.values())))

    out = {"commit": commit, "inputs_dirty": bool(dirty), "months_ranked": len(ms),
           "last_price_date": str(last_px.date()), "price_grid_end": grid[-1],
           "control": control, "counts": bool(control["sees"]),
           "A": A, "B": B, "C": C, "D_is_the_answer": bool(D),
           "caveats": list(br.CAVEATS)}
    OUT.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")

    print(json.dumps({"control": control, "counts": out["counts"],
                      "A": {k: v for k, v in A.items() if k != "per_month"},
                      "B": {k: ({kk: vv for kk, vv in v.items() if kk != "per_window"}
                                if isinstance(v, dict) else v) for k, v in B.items()},
                      "C": C, "D_is_the_answer": out["D_is_the_answer"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
