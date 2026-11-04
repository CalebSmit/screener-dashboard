"""Review of research/2026-11-02-investment-inside-quality.md section 4, independently.

The note measured the weight-preserving fold of Investment into Quality with a scratch
script it did not commit, on the 2026-10-30 payload. This rebuilds the measurement from
scratch - different code, written from `factor_engine` rather than from the note - and
runs it on whatever payload is committed today, plus every earlier payload in
`git log` if `--history` is passed. A structural claim that moves from run to run is
reported as a range, not a number.

No return, IC or backtest figure is used (CLAUDE.md rules 4 and 5): this measures the
arithmetic of a relabelling on one cross-section.

Usage:
    python research/measurements/2026-11-04-investment-fold-review.py
    python research/measurements/2026-11-04-investment-fold-review.py --history 6
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import factor_engine as fe  # noqa: E402
from factor_engine import load_config  # noqa: E402

QUALITY_POINTS_TODAY = 22.0
INVESTMENT_POINTS_TODAY = 5.0


def load_payload(text: str) -> dict:
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


def profile_for(sd: dict, row: dict, pio_threshold: float, pio_enabled: bool) -> str:
    """Which Quality weight table the engine scored this stock with.

    The engine's rule (`compute_category_scores`): bank first, then the Piotroski
    low-valuation table for a non-bank whose valuation score is below the threshold.
    """
    if sd.get("flags", {}).get("is_bank"):
        return "bank"
    val = row.get("valuation_score")
    if pio_enabled and val is not None and val < pio_threshold:
        return "pio_lowval"
    return "generic"


def weighted_avg(pct: dict, table: dict) -> float | None:
    """The engine's per-row renormalised weighted average over the metrics present."""
    num = den = 0.0
    for metric, weight in table.items():
        if weight == 0:
            continue
        value = pct.get(metric)
        if value is None:
            continue
        num += value * weight
        den += weight
    return num / den if den > 0 else None


def composite_from(cat_scores: dict, fw: dict, discount: float) -> float | None:
    num = den = 0.0
    for cat, weight in fw.items():
        if weight == 0:
            continue
        score = cat_scores.get(cat)
        if score is None or (isinstance(score, float) and np.isnan(score)):
            continue
        num += score * weight
        den += weight
    if den == 0:
        return None
    return (num / den) * (1 - discount)


def trap_flags(df: pd.DataFrame, cfg: dict) -> tuple[set, set]:
    """Reproduce the two flags from category scores alone, as the engine computes them."""
    vtf = cfg.get("value_trap_filters", {})
    gtf = cfg.get("growth_trap_filters", {})
    value = set()
    if vtf.get("enabled", True):
        cheap = df["valuation"].ge(df["valuation"].quantile(
            vtf.get("valuation_percentile", 70) / 100.0)).fillna(False)
        layers = 0
        for key, floor_key, default in (("quality", "quality_floor_percentile", 30),
                                        ("momentum", "momentum_floor_percentile", 30),
                                        ("revisions", "revisions_floor_percentile", 30)):
            thr = df[key].quantile(vtf.get(floor_key, default) / 100.0)
            layers = layers + df[key].le(thr).fillna(False).astype(int)
        value = set(df.index[cheap & (layers >= 2)])
    growth = set()
    if gtf.get("enabled", False):
        g1 = df["growth"].ge(df["growth"].quantile(
            gtf.get("growth_ceiling_percentile", 70) / 100.0)).fillna(False)
        g2 = df["quality"].le(df["quality"].quantile(
            gtf.get("quality_floor_percentile", 35) / 100.0)).fillna(False)
        g3 = df["revisions"].le(df["revisions"].quantile(
            gtf.get("revisions_floor_percentile", 35) / 100.0)).fillna(False)
        growth = set(df.index[g1 & (g2 | g3)])
    return value, growth


def ranks_from(composites: pd.Series) -> pd.Series:
    """The engine's order: round to two decimals, then rank with ties on the lower rank."""
    return composites.round(2).rank(ascending=False, method="min").astype(int)


def measure(payload: dict, cfg: dict, label: str, verbose: bool = True) -> dict:
    fw = {k: float(v) for k, v in payload["weights"]["factor_weights"].items()}
    profiles = payload["weights"]["profiles"]
    pio_cfg = cfg.get("piotroski_conditional", {})
    pio_enabled = bool(pio_cfg.get("enabled", False))
    pio_threshold = float(pio_cfg.get("valuation_threshold", 50))

    q_metrics = [m for m, w in profiles["quality"]["generic"].items()]
    rows = {r["Ticker"]: r for r in payload["table_data"]}

    # ---- today's scores, reproduced, and the fold ----
    out = {}
    for ticker, row in rows.items():
        sd = payload["stock_detail"].get(ticker)
        if sd is None:
            continue
        pct = {k: v for k, v in sd["pct"].items()}
        pid = profile_for(sd, row, pio_threshold, pio_enabled)
        q_table = {m: w / 100.0 for m, w in profiles["quality"][pid].items()}
        # Banks reach asset growth through the generic Investment table today, so a
        # folded bank Quality table must carry it explicitly or 46 banks lose it.
        i_table = {m: w / 100.0 for m, w in
                   profiles["investment"].get(pid, profiles["investment"]["generic"]).items()}

        q_now = weighted_avg(pct, q_table)
        i_now = weighted_avg(pct, i_table)

        # The weight-preserving fold: every metric keeps its composite points.
        folded_points = QUALITY_POINTS_TODAY + INVESTMENT_POINTS_TODAY
        scale = QUALITY_POINTS_TODAY / folded_points
        fold_table = {m: w * scale for m, w in q_table.items()}
        for m, w in i_table.items():
            fold_table[m] = fold_table.get(m, 0.0) + w * (INVESTMENT_POINTS_TODAY / folded_points)
        q_fold = weighted_avg(pct, fold_table)

        discount = 0.0
        cov = sd.get("cov") or {}
        cov_cfg = cfg.get("data_quality", {}).get("coverage_discount", {})
        if cov_cfg.get("enabled", False) and cov.get("of"):
            coverage = cov["n"] / cov["of"]
            threshold = cov_cfg.get("threshold", 0.80)
            if coverage < threshold:
                discount = (threshold - coverage) * cov_cfg.get("penalty_rate", 0.15)

        published_cats = dict(sd["cat_scores"])
        now_cats = dict(published_cats)
        fold_cats = {k: v for k, v in published_cats.items() if k != "investment"}
        fold_cats["quality"] = q_fold

        fold_fw = {k: v for k, v in fw.items() if k != "investment"}
        fold_fw["quality"] = fw["quality"] + fw["investment"]

        out[ticker] = {
            "profile": pid,
            "q_published": published_cats.get("quality"),
            "q_reproduced": q_now,
            "i_published": published_cats.get("investment"),
            "i_reproduced": i_now,
            "q_folded": q_fold,
            "comp_published": sd["composite"],
            "comp_reproduced": composite_from(now_cats, fw, discount),
            "comp_folded": composite_from(fold_cats, fold_fw, discount),
            "rank_published": sd["rank"],
            "has_asset_growth": pct.get("asset_growth") is not None,
            "missing_quality_metrics": [m for m, w in q_table.items()
                                        if w > 0 and pct.get(m) is None],
            "cats": published_cats,
        }

    df = pd.DataFrame(out).T
    for col in ("q_published", "q_reproduced", "i_published", "i_reproduced", "q_folded",
                "comp_published", "comp_reproduced", "comp_folded"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    res: dict = {"label": label, "n": len(df)}

    # ---- fidelity of the reproduction ----
    res["max_q_err"] = float((df["q_published"] - df["q_reproduced"]).abs().max())
    res["max_i_err"] = float((df["i_published"] - df["i_reproduced"]).abs().max())
    res["max_comp_err"] = float((df["comp_published"] - df["comp_reproduced"]).abs().max())

    # ---- neutrality of the fold ----
    delta = (df["comp_folded"] - df["comp_reproduced"]).abs()
    res["neutral"] = int((delta <= 0.005).sum())
    res["non_neutral"] = int((delta > 0.005).sum())
    res["max_comp_move"] = float(delta.max())

    rank_now = ranks_from(df["comp_reproduced"])
    rank_fold = ranks_from(df["comp_folded"])
    top25_now = set(rank_now[rank_now <= 25].index)
    top25_fold = set(rank_fold[rank_fold <= 25].index)
    res["top25_in"] = sorted(top25_fold - top25_now)
    res["top25_out"] = sorted(top25_now - top25_fold)
    res["top25_reordered"] = int(sum(
        1 for t in top25_now & top25_fold if rank_now[t] != rank_fold[t]))
    res["moved_any"] = int((rank_now != rank_fold).sum())
    res["moved_gt5"] = int(((rank_now - rank_fold).abs() > 5).sum())

    # ---- renormalisation: who the fold is not neutral for, and why ----
    res["missing_asset_growth"] = sorted(df.index[~df["has_asset_growth"].astype(bool)])
    res["missing_a_quality_metric"] = int(sum(
        1 for v in df["missing_quality_metrics"] if v))
    worst = (df["q_folded"] - df["q_published"]).abs().sort_values(ascending=False).head(5)
    res["largest_quality_moves"] = [
        (t, round(float(df.at[t, "q_published"]), 2), round(float(df.at[t, "q_folded"]), 2),
         int(rank_now[t]), int(rank_fold[t])) for t in worst.index
        if pd.notna(df.at[t, "q_folded"])]
    # The note's section 4 item 1 cites the largest *composite* moves, which is where
    # renormalisation shows up - a stock missing a Quality metric, or missing asset growth.
    worst_c = delta.sort_values(ascending=False).head(5)
    res["largest_composite_moves"] = [
        (t, round(float(df.at[t, "comp_reproduced"]), 2),
         round(float(df.at[t, "comp_folded"]), 2), int(rank_now[t]), int(rank_fold[t]),
         "no asset growth" if not df.at[t, "has_asset_growth"]
         else f"missing {df.at[t, 'missing_quality_metrics']}" or "all metrics")
        for t in worst_c.index]
    res["no_quality_score"] = sorted(df.index[df["q_published"].isna()])

    # ---- dispersion, against the health gate's bar ----
    res["q_sd_now"] = float(df["q_published"].std())
    res["q_sd_fold"] = float(df["q_folded"].std())
    res["q_sd_drop"] = 1 - res["q_sd_fold"] / res["q_sd_now"]
    res["q_mean_abs_move"] = float((df["q_folded"] - df["q_published"]).abs().mean())
    res["q_spearman"] = float(df["q_published"].corr(df["q_folded"], method="spearman"))

    # ---- how Investment sits against the other seven ----
    cats = pd.DataFrame(list(df["cats"]), index=df.index).apply(pd.to_numeric, errors="coerce")
    res["investment_vs"] = {
        c: round(float(cats["investment"].corr(cats[c], method="spearman")), 3)
        for c in cats.columns if c != "investment"}

    # ---- trap flags ----
    now_flags = trap_flags(cats.rename(columns=str), cfg)
    folded = cats.copy()
    folded["quality"] = df["q_folded"]
    fold_flags = trap_flags(folded, cfg)
    res["vt_now"], res["gt_now"] = len(now_flags[0]), len(now_flags[1])
    res["vt_fold"], res["gt_fold"] = len(fold_flags[0]), len(fold_flags[1])
    res["vt_flipped"] = len(now_flags[0] ^ fold_flags[0])
    res["gt_flipped"] = len(now_flags[1] ^ fold_flags[1])
    res["flips_in_top25"] = len((now_flags[0] ^ fold_flags[0] | now_flags[1] ^ fold_flags[1])
                                & top25_now)

    if verbose:
        report(res)
    return res


def report(r: dict) -> None:
    print(f"\n=== {r['label']}  ({r['n']} stocks) ===")
    print(f"reproduction: max |published - reproduced|  quality {r['max_q_err']:.2e}  "
          f"investment {r['max_i_err']:.2e}  composite {r['max_comp_err']:.2e}")
    print(f"fold neutral (|dComposite| <= 0.005): {r['neutral']} of {r['n']}; "
          f"non-neutral {r['non_neutral']}; largest move {r['max_comp_move']:.3f} points")
    print(f"top 25: {len(r['top25_in'])} in {r['top25_in']}, "
          f"{len(r['top25_out'])} out {r['top25_out']}, "
          f"{r['top25_reordered']} reordered inside it")
    print(f"ranks: {r['moved_any']} stocks change rank, {r['moved_gt5']} by more than five")
    print(f"missing asset growth: {r['missing_asset_growth']}; "
          f"{r['missing_a_quality_metric']} stocks miss a weighted Quality metric")
    print(f"no Quality score at all: {r['no_quality_score']}")
    print("largest Quality moves (ticker, now, folded, rank now, rank folded):")
    for row in r["largest_quality_moves"]:
        print(f"   {row}")
    print("largest COMPOSITE moves (ticker, now, folded, rank now, rank folded, why):")
    for row in r["largest_composite_moves"]:
        print(f"   {row}")
    print(f"Quality dispersion: {r['q_sd_now']:.2f} -> {r['q_sd_fold']:.2f} "
          f"({r['q_sd_drop'] * 100:.1f}% drop; health gate bar is 20%)")
    print(f"Quality mean |move| {r['q_mean_abs_move']:.2f} points, "
          f"Spearman(now, folded) {r['q_spearman']:.3f}")
    print(f"Investment vs the other seven (Spearman): {r['investment_vs']}")
    print(f"value traps {r['vt_now']} -> {r['vt_fold']} ({r['vt_flipped']} flipped); "
          f"growth traps {r['gt_now']} -> {r['gt_fold']} ({r['gt_flipped']} flipped); "
          f"{r['flips_in_top25']} flips inside today's top 25")


def arithmetic_checks() -> None:
    """The fold's weight table, from config, independent of any run."""
    cfg = load_config()
    fw = cfg["factor_weights"]
    q, i = float(fw["quality"]), float(fw["investment"])
    folded = q + i
    print("\n=== the fold's arithmetic, from config.yaml ===")
    print(f"Quality {q} + Investment {i} = {folded} composite points")
    print(f"asset_growth's share of a folded Quality: {i / folded * 100:.4f}%  "
          f"(note says 18.5185%)")
    print(f"rescale factor for today's Quality metrics: {q / folded:.6f} (= {q:.0f}/{folded:.0f})")
    for cat in ("quality", "investment"):
        for pid, table in fe.metric_weight_profiles(cfg, cat).items():
            pts = {m: round(w * fw[cat], 4) for m, w in table.items() if w > 0}
            print(f"  {cat}/{pid}: composite points {pts}")
    print("Check: every metric's composite points are preserved by construction, since")
    print("  (w * q/folded) * folded = w * q  and  (i/folded) * folded = i.")


def history(n: int, cfg: dict) -> None:
    """The same measurement on the last n committed payloads, so a count has a range."""
    shas = subprocess.run(
        ["git", "log", "--format=%H %ad", "--date=short", "-n", str(n * 3), "--",
         "dashboard_data.js"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    results = []
    seen = set()
    for line in shas:
        if not line.strip():
            continue
        sha, date = line.split()[0], line.split()[1]
        if date in seen:
            continue
        seen.add(date)
        blob = subprocess.run(["git", "show", f"{sha}:dashboard_data.js"], cwd=ROOT,
                              capture_output=True, text=True)
        if blob.returncode != 0:
            continue
        try:
            payload = load_payload(blob.stdout)
            results.append(measure(payload, cfg, f"{date} ({sha[:8]})", verbose=False))
        except Exception as exc:  # noqa: BLE001 - a payload shape that moved is a skip
            print(f"  {date}: skipped ({type(exc).__name__}: {exc})")
        if len(results) >= n:
            break
    print(f"\n=== the same measurement over {len(results)} committed payloads ===")
    print(f"{'run':<22}{'neutral':>8}{'non-neu':>9}{'top25 in/out':>14}"
          f"{'Q sd drop':>11}{'vt flip':>9}{'gt flip':>9}")
    for r in results:
        swap = f"{len(r['top25_in'])}/{len(r['top25_out'])}"
        print(f"{r['label']:<22}{r['neutral']:>8}{r['non_neutral']:>9}{swap:>14}"
              f"{r['q_sd_drop'] * 100:>10.1f}%{r['vt_flipped']:>9}{r['gt_flipped']:>9}")
    if results:
        drops = [r["q_sd_drop"] for r in results]
        print(f"\nQuality dispersion drop across these runs: "
              f"{min(drops) * 100:.1f}% to {max(drops) * 100:.1f}% "
              f"(the health gate refuses a run at 20%)")
        swaps = [len(r["top25_in"]) for r in results]
        print(f"top-25 entries caused by the fold: {min(swaps)} to {max(swaps)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=int, default=0,
                    help="also measure the last N committed payloads")
    args = ap.parse_args()
    cfg = load_config()
    arithmetic_checks()
    payload = load_payload((ROOT / "dashboard_data.js").read_text(encoding="utf-8"))
    measure(payload, cfg, f"committed payload, run {payload['kpis'].get('run_timestamp')}")
    if args.history:
        history(args.history, cfg)


if __name__ == "__main__":
    main()
