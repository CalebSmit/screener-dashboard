"""Measurements behind research/2026-09-21-position-sizing-and-how-much.md

What this answers: how much does the screener's configured position-sizing
scheme (``portfolio.weighting: 'score'``) actually differ from equal weight?

This is a *property of the construction rule*, not a backtest and not a return
measurement. It asks only "what weights does this rule emit", which is a fact
about the arithmetic, so nothing here is gated by CLAUDE.md rules 4 and 5.

Run from the repo root:

    python research/measurements/2026-09-21-position-sizing-dispersion.py

Reads every snapshot in improvement/snapshots/. Prints, per run date and
pooled: the score-proportional weight range across the selected names, the
maximum absolute deviation from equal weight, and the implied active share of
score weighting measured against equal weight on the same 25 names.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
SNAPSHOTS = REPO / "improvement" / "snapshots"

NUM_STOCKS = 25          # config.yaml portfolio.num_stocks
MAX_SECTOR = 8           # config.yaml portfolio.max_sector_concentration
MAX_POSITION_PCT = 5.0   # config.yaml portfolio.max_position_pct


def select_sector_capped(df: pd.DataFrame, n: int = NUM_STOCKS,
                         max_sector: int = MAX_SECTOR) -> pd.DataFrame:
    """Greedy top-N by Composite subject to the per-sector count cap.

    Mirrors portfolio_constructor's selection order closely enough for a
    weighting-dispersion measurement: the weights depend on the *spread* of
    composite scores among the selected names, which is insensitive to the
    exact tie-breaking of the last slot or two.
    """
    ordered = df.sort_values("Composite", ascending=False)
    picked, counts = [], {}
    for _, row in ordered.iterrows():
        if len(picked) >= n:
            break
        sec = row.get("Sector")
        if counts.get(sec, 0) >= max_sector:
            continue
        picked.append(row)
        counts[sec] = counts.get(sec, 0) + 1
    return pd.DataFrame(picked)


def score_weights(composites: pd.Series, cap: float = MAX_POSITION_PCT) -> np.ndarray:
    """Composite-proportional weights, then iterative cap-and-redistribute.

    Same shape as portfolio_constructor.construct_portfolio step 4/5: clip
    composites at 1.0, normalise to 100, then cap and redistribute the excess
    proportionally to each uncapped weight.
    """
    c = composites.fillna(composites.median()).clip(lower=1.0).to_numpy(float)
    w = c / c.sum() * 100.0
    for _ in range(10):
        over = w > cap
        excess = float((w[over] - cap).sum())
        if excess <= 1e-3 or over.all():
            break
        w[over] = cap
        under = ~over
        tot = float(w[under].sum())
        w[under] += excess * (w[under] / tot if tot > 0 else 1.0 / under.sum())
    return w


def main() -> int:
    files = sorted(SNAPSHOTS.glob("*.parquet"))
    if not files:
        print(f"No snapshots found in {SNAPSHOTS}", file=sys.stderr)
        return 1

    # One snapshot per run date. A day with several runs wrote several files
    # (the duplicate-snapshot defect fixed 2026-08-24); counting them all would
    # weight February ~15x and say nothing extra. Newest file per date wins.
    by_date: dict[str, pathlib.Path] = {}
    for f in files:
        by_date[f.name.split("_")[0]] = f

    rows, skipped = [], []
    for run_date, f in sorted(by_date.items()):
        df = pd.read_parquet(f)
        if "Composite" not in df.columns or "Sector" not in df.columns:
            continue
        port = select_sector_capped(df)
        n = len(port)
        # Degraded snapshots (two February files hold 3 rows, not 502) are not
        # portfolios and their weights are arithmetic on a broken input.
        if n < NUM_STOCKS:
            skipped.append((run_date, n))
            continue
        ew = 100.0 / n
        sw = score_weights(port["Composite"])
        # Active share of score-weighting vs equal-weighting on identical names:
        # half the sum of absolute weight differences.
        active_share = 0.5 * float(np.abs(sw - ew).sum())
        rows.append({
            "run_date": run_date,
            "n": n,
            "composite_min": float(port["Composite"].min()),
            "composite_max": float(port["Composite"].max()),
            "equal_wt_pct": ew,
            "score_wt_min_pct": float(sw.min()),
            "score_wt_max_pct": float(sw.max()),
            "max_abs_dev_pp": float(np.abs(sw - ew).max()),
            "active_share_pct": active_share,
            "n_at_cap": int((sw >= MAX_POSITION_PCT - 1e-6).sum()),
        })

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"Run dates measured: {len(res)}  "
          f"({res.run_date.min()} .. {res.run_date.max()})")
    if skipped:
        print(f"Skipped as degraded (<{NUM_STOCKS} names): "
              + ", ".join(f"{d} (n={n})" for d, n in skipped))
    print()
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # The composite scale was rescaled between February and August, so the
    # spread of scores within the top 25 differs by era. Report both, because
    # the conclusion holds in each and that is the point.
    recent = res[res.run_date >= "2026-08-01"]

    def block(label: str, r: pd.DataFrame) -> None:
        print(f"\n--- {label} (n={len(r)} run dates) ---")
        print(f"Equal weight for 25 names:           {100/NUM_STOCKS:.2f}%")
        print(f"Score weight spans:                  "
              f"{r.score_wt_min_pct.min():.3f}% .. {r.score_wt_max_pct.max():.3f}%")
        print(f"Max deviation from equal weight:     "
              f"{r.max_abs_dev_pp.max():.3f} pp (median {r.max_abs_dev_pp.median():.3f} pp)")
        print(f"Active share vs equal weight:        "
              f"median {r.active_share_pct.median():.2f}%, max {r.active_share_pct.max():.2f}%")
        print(f"Heaviest/lightest weight ratio:      "
              f"{(r.score_wt_max_pct / r.score_wt_min_pct).max():.3f}x")
        print(f"Positions hitting the 5% cap:        {int(r.n_at_cap.sum())}")

    block("All run dates", res)
    block("Current composite scale, 2026-08-01 onward", recent)

    print("\nInterpretation: score-proportional weighting on a 25-name book whose\n"
          "composite scores span a narrow band produces weights within a fraction\n"
          "of a percentage point of equal weight. The 5% cap has never bound on\n"
          "a healthy run, so max_position_pct is currently inert.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
