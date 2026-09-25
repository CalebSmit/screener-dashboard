"""Measure the dead two-source fallbacks in `compute_metrics`, and the exact
blast radius of repairing them.

Context
-------
Nine numeric inputs to `compute_metrics()` have two possible sources and were
all written as the nested form ``d.get(A, d.get(B, np.nan))``.  That only
reaches B when key A is *absent*.  Every one of these keys is written
unconditionally by `_fetch_single_ticker_inner()` - `_safe()` and `_stmt_val()`
both return NaN rather than omitting the key - so the fallback can only fire
on an exception path, never on the missing-data path it was written for.

Part 1 counts, per fallback pair, how many names have a NaN first source and a
usable second source.  Part 2 recomputes every metric for the whole universe
under the pre-change and post-change `factor_engine` and diffs them, so the
behaviour change is measured rather than argued.

Both parts read the committed raw fetches in `runs/*/00_raw_fetch.parquet`,
which the data loop retains for the newest ~20 runs.

Usage
-----
    python research/measurements/2026-09-25-dead-two-source-fallbacks.py

`--ab` is skipped automatically if the pre-change `factor_engine.py` cannot be
read out of git (e.g. the change has since been squashed away).
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# (preferred source, backup source, what it feeds)
PAIRS = [
    ("totalDebt",     "totalDebt_bs", "enterprise value -> ev_ebitda, ev_sales, fcf_yield"),
    ("totalDebt_bs",  "totalDebt",    "invested capital -> roic, net_debt_to_ebitda, debt_equity"),
    ("totalCash",     "cash_bs",      "enterprise value -> ev_ebitda, ev_sales, fcf_yield"),
    ("cash_bs",       "totalCash",    "invested capital -> roic, net_debt_to_ebitda"),
    ("ebit_annual",   "ebit",         "operating_leverage"),
    ("currentPrice",  "price_latest", "peg_ratio, price_target_upside, proximity_52w_high, pb_ratio, _current_price"),
]

# Metrics that can move. Beta/Jensen/Sharpe depend on the market series, which
# is regenerated here rather than restored, so they are reported separately.
MARKET_DEPENDENT = {"beta", "jensens_alpha", "sharpe_ratio", "sortino_ratio"}


def raw_fetches() -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(str(ROOT / "runs" / "*" / "00_raw_fetch.parquet")))]


def part1_incidence() -> None:
    files = raw_fetches()
    print(f"=== Part 1: incidence across {len(files)} retained raw fetch(es) ===\n")
    if not files:
        print("  no runs/*/00_raw_fetch.parquet found - nothing to measure")
        return

    for f in files:
        d = pd.read_parquet(f)
        print(f"{f.parent.name}  ({len(d)} names)")
        for pref, back, feeds in PAIRS:
            if pref not in d.columns:
                print(f"  {pref:14s} -> column absent")
                continue
            n_nan = int(d[pref].isna().sum())
            if back in d.columns:
                rescuable = d[pref].isna() & d[back].notna()
                names = ", ".join(sorted(d.loc[rescuable, "Ticker"].astype(str)))
                n_res = int(rescuable.sum())
            else:
                names, n_res = "(backup column absent)", 0
            print(f"  {pref:14s} NaN={n_nan:4d}  rescuable by {back:14s} = {n_res:3d}"
                  f"{('  [' + names + ']') if names and n_res else ''}")
            if n_res:
                print(f"       feeds: {feeds}")
        print()


def _load_pre_change_module():
    """Import `factor_engine.py` as of the commit before this change."""
    for rev in ("HEAD", "HEAD~1"):
        try:
            src = subprocess.run(
                ["git", "show", f"{rev}:factor_engine.py"],
                cwd=ROOT, capture_output=True, text=True, check=True,
            ).stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        if "_coalesce" in src:
            continue  # this revision already has the fix
        tmp = Path(tempfile.mkdtemp()) / "factor_engine_pre.py"
        tmp.write_text(src, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("factor_engine_pre", tmp)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["factor_engine_pre"] = mod
        spec.loader.exec_module(mod)
        print(f"  (pre-change factor_engine.py loaded from {rev})")
        return mod
    return None


def _records(df: pd.DataFrame) -> list[dict]:
    return df.to_dict(orient="records")


def part2_ab() -> None:
    print("=== Part 2: A/B over the full universe ===\n")
    files = raw_fetches()
    if not files:
        print("  no raw fetches - skipped")
        return
    pre = _load_pre_change_module()
    if pre is None:
        print("  pre-change factor_engine.py not reachable from git - skipped")
        return

    import factor_engine as post

    f = files[-1]
    raw = pd.read_parquet(f)
    recs = _records(raw)

    # One fixed market series for both sides, so any difference is the change.
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2025-09-01", periods=260)
    mr = pd.Series(rng.normal(0.0004, 0.01, 260), index=idx)

    a = pre.compute_metrics(_records(raw), mr).set_index("Ticker")
    b = post.compute_metrics(recs, mr).set_index("Ticker")

    print(f"  universe: {f.parent.name}, {len(a)} names\n")

    shared = [c for c in a.columns if c in b.columns
              and pd.api.types.is_numeric_dtype(a[c])
              and pd.api.types.is_numeric_dtype(b[c])]

    gained, lost, changed = [], [], []
    for c in sorted(shared):
        if c in MARKET_DEPENDENT:
            continue
        av, bv = a[c], b[c]
        g = av.isna() & bv.notna()
        l = av.notna() & bv.isna()
        ch = av.notna() & bv.notna() & ~np.isclose(
            av.astype(float), bv.astype(float), rtol=1e-9, atol=0, equal_nan=True)
        if g.any():
            gained.append((c, sorted(av.index[g])))
        if l.any():
            lost.append((c, sorted(av.index[l])))
        if ch.any():
            changed.append((c, sorted(av.index[ch])))

    print("  METRICS GAINED (was NaN, now computes):")
    if not gained:
        print("    none")
    for c, names in gained:
        print(f"    {c:22s} +{len(names):3d}  {', '.join(names[:12])}"
              f"{' ...' if len(names) > 12 else ''}")

    print("\n  METRICS LOST (computed before, NaN now) - must be empty:")
    if not lost:
        print("    none")
    for c, names in lost:
        print(f"    {c:22s} -{len(names):3d}  {', '.join(names[:12])}")

    print("\n  VALUES CHANGED (computed both sides, different number):")
    if not changed:
        print("    none")
    for c, names in changed:
        print(f"    {c:22s} ~{len(names):3d}  {', '.join(names[:12])}"
              f"{' ...' if len(names) > 12 else ''}")

    # Category-weight impact of the gains, for the affected names only.
    print("\n  Quality/valuation weight restored per affected name:")
    # config.yaml metric_weights. fcf_yield is the heaviest valuation metric
    # and its denominator is EV, so a missing EV costs it too.
    weights = {"roic": 27, "net_debt_to_ebitda": 18,
               "fcf_yield": 45, "ev_ebitda": 25, "ev_sales": 10}
    QUALITY = {"roic", "net_debt_to_ebitda"}
    VALUATION = {"fcf_yield", "ev_ebitda", "ev_sales"}
    per_name: dict[str, list[str]] = {}
    for c, names in gained:
        if c in weights:
            for n in names:
                per_name.setdefault(n, []).append(c)
    if not per_name:
        print("    none")
    for n, cols in sorted(per_name.items()):
        q = sum(weights[c] for c in cols if c in QUALITY)
        v = sum(weights[c] for c in cols if c in VALUATION)
        bits = []
        if q:
            bits.append(f"{q}/100 quality")
        if v:
            bits.append(f"{v}/100 valuation")
        print(f"    {n:6s} {', '.join(sorted(cols)):45s} {' + '.join(bits)}")


def _score(mod, raw: pd.DataFrame, mr: pd.Series, cfg: dict) -> pd.DataFrame:
    """metrics -> sector percentiles -> category scores -> composite -> rank.

    Mirrors `run_screener.run_pipeline()` from `compute_metrics` to
    `rank_stocks`, minus the artifact writes and the regime momentum
    adjustment (which reads trailing dispersion off disk and is identical on
    both sides).
    """
    df = mod.compute_metrics(_records(raw), mr, cfg)
    df = mod.compute_sector_percentiles(df)
    df = mod.apply_percentile_transform(df, cfg)
    df = mod.compute_category_scores(df, cfg)
    df = mod.compute_composite(df, cfg)
    return mod.rank_stocks(df)


def part3_ranks() -> None:
    """Rank and composite effect, so the changelog's expected effect is a
    number someone can check rather than a claim."""
    print("\n=== Part 3: composite and rank effect ===\n")
    files = raw_fetches()
    pre = _load_pre_change_module()
    if not files or pre is None:
        print("  skipped")
        return

    import factor_engine as post
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

    raw = pd.read_parquet(files[-1])
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2025-09-01", periods=260)
    mr = pd.Series(rng.normal(0.0004, 0.01, 260), index=idx)

    a = _score(pre, raw, mr, cfg).set_index("Ticker")
    b = _score(post, raw, mr, cfg).set_index("Ticker")

    print("  CAVEAT: `_daily_returns` is a dict column and does not survive the")
    print("  parquet round-trip, so beta, sortino_ratio, max_drawdown_1y and")
    print("  jensens_alpha are absent on BOTH sides. The deltas below are valid")
    print("  (identical treatment either way); the absolute ranks are NOT the")
    print("  live ranks and must not be quoted as such.\n")

    both = a.index.intersection(b.index)
    d_comp = (b.loc[both, "Composite"] - a.loc[both, "Composite"]).astype(float)
    d_rank = (b.loc[both, "Rank"] - a.loc[both, "Rank"]).astype(float)

    moved = d_rank[d_rank != 0].abs().sort_values(ascending=False)
    print(f"  names whose rank moves at all: {len(moved)} of {len(both)}")
    print(f"  largest composite change:      {d_comp.abs().max():.4f}")
    print(f"  median |rank change|:          "
          f"{moved.median() if len(moved) else 0:.1f}")
    print("\n  the three repaired names:")
    for t in ("ANET", "ISRG", "FISV"):
        if t in both:
            print(f"    {t:5s} composite {a.loc[t,'Composite']:7.3f} -> "
                  f"{b.loc[t,'Composite']:7.3f}   rank "
                  f"{int(a.loc[t,'Rank']):4d} -> {int(b.loc[t,'Rank']):4d}")
    print("\n  largest movers overall (top 10 by |rank change|):")
    for t in moved.head(10).index:
        print(f"    {t:5s} rank {int(a.loc[t,'Rank']):4d} -> "
              f"{int(b.loc[t,'Rank']):4d}  ({int(d_rank[t]):+d})   "
              f"composite {d_comp[t]:+.3f}")

    top50_before = set(a.nsmallest(50, "Rank").index)
    top50_after = set(b.nsmallest(50, "Rank").index)
    print(f"\n  top-50 membership change: "
          f"{len(top50_after - top50_before)} in, "
          f"{len(top50_before - top50_after)} out")
    if top50_after - top50_before:
        print(f"    in:  {', '.join(sorted(top50_after - top50_before))}")
    if top50_before - top50_after:
        print(f"    out: {', '.join(sorted(top50_before - top50_after))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ab", action="store_true",
                    help="incidence counts only; skip the A/B recompute")
    args = ap.parse_args()
    part1_incidence()
    if not args.skip_ab:
        part2_ab()
        part3_ranks()


if __name__ == "__main__":
    main()
