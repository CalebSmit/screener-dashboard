#!/usr/bin/env python3
"""
Size the price component of ``backtest.py``'s look-ahead bias.

`plan/backtest-v2.md` step 1 asked for two measurements. Survivorship was done
2026-09-24 and restated 2026-09-30 (**11.4% of the name-month panel**).
Look-ahead had never been measured at all. This is the half that needs no
vendor, no purchase and nobody's permission, which is why the 2026-09-30 session
nominated it as the next step on the plan.

What it measures
----------------
``backtest.simulate_monthly_scores()`` holds **83.1% of composite weight** at a
single Phase-1 snapshot (``lookahead.weight_buckets`` derives that from
``config.yaml``). Of that, **28.0 points depend on nothing but the share price**
— they are ratios of a price-independent fundamental to a market value — and the
harness is already holding the monthly price panel it would need to restate
them. A further **6.1 points** are derived from a price *history* and are also
held constant, though a single month-end price cannot restate them.

So this measurement answers: *if the backtest restated only the metrics it has
the data to restate, how differently would it rank the universe?* Two arms,
identical in every other respect:

  A (v1): every static metric at its snapshot value, momentum/risk recomputed at
          the rebalance month exactly as ``simulate_monthly_scores`` does.
  B:      the same, with the six price-restatable metrics restated at that
          month's price via ``lookahead.restate_at_price``.

The gap between them is a **lower bound** on total look-ahead: the 49.0 points
that need point-in-time filings and analyst estimates are held constant in both
arms, and the 6.1 price-history points are held constant in both arms too.

No returns and no IC are computed anywhere in this script. Every number it
prints is a property of the scoring harness — a rank, a decile, a weight share —
so nothing here is a backtest result under ``CLAUDE.md`` rule 5.

Reproducing
-----------
    python research/measurements/2026-10-01-lookahead-price-component.py

The snapshot comes from the committed ``dashboard_data.js`` (the 2026-10-01 data
run), so the inputs are checkable from the repository. The monthly price panel is
fetched once from yfinance and cached under ``cache/`` (gitignored); pass
``--offline`` to fail rather than fetch. Output is written next to this file as
``2026-10-01-lookahead-price-component.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import lookahead  # noqa: E402
from factor_engine import (  # noqa: E402
    METRIC_COLS,
    _is_bank_like,
    adjust_momentum_weight,
    compute_category_scores,
    compute_composite,
    compute_sector_percentiles,
    load_config,
)
from backtest import _recompute_momentum_risk  # noqa: E402

OUT_JSON = Path(__file__).with_suffix(".json")
PRICE_CACHE = ROOT / "cache" / "lookahead_prices.parquet"
PANEL_START = "2018-12-01"   # 13 months of history before backtest.BACKTEST_START
BENCHMARK = "^GSPC"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_snapshot() -> pd.DataFrame:
    """Build the Phase-1 snapshot frame from the committed dashboard payload."""
    text = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    payload = json.loads(text[text.index("=") + 1:].strip().rstrip(";"))
    detail = payload["stock_detail"]

    rows = []
    for ticker, d in detail.items():
        fin = d.get("financials") or {}
        row = {"Ticker": ticker,
               "Company": d.get("company"),
               "Sector": d.get("sector"),
               "market_cap": fin.get("market_cap"),
               "enterprise_value": fin.get("enterprise_value"),
               "price": d.get("price"),
               "pt_mean": d.get("pt_mean"),
               # The trailing dividend yield lives in ``financials``, not in
               # ``raw`` — the metric carries weight 0 so it is not published
               # as a scored input. Only the drag sensitivity reads it.
               "div_yield_fin": fin.get("dividend_yield"),
               "published_composite": d.get("composite"),
               "published_rank": d.get("rank"),
               "_is_bank_like": _is_bank_like(ticker, d.get("sector") or "",
                                              d.get("industry") or "")}
        raw = d.get("raw") or {}
        for m in METRIC_COLS:
            row[m] = raw.get(m, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Ticker")
    return df


def load_prices(tickers: list, offline: bool = False) -> pd.DataFrame:
    """Monthly adjusted closes for the universe plus the benchmark."""
    if PRICE_CACHE.exists():
        print(f"[CACHE HIT] {PRICE_CACHE.name}")
        return pd.read_parquet(PRICE_CACHE)
    if offline:
        raise SystemExit(f"--offline and no price cache at {PRICE_CACHE}")

    import yfinance as yf
    symbols = list(tickers) + [BENCHMARK]
    print(f"[FETCH] {len(symbols)} monthly series from {PANEL_START}")
    data = yf.download(symbols, start=PANEL_START, auto_adjust=True,
                       interval="1mo", group_by="ticker", threads=True,
                       progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data.xs("Close", axis=1, level=1)
    else:
        prices = data[["Close"]].copy()
        prices.columns = symbols[:1]
    prices = prices.dropna(how="all")
    PRICE_CACHE.parent.mkdir(exist_ok=True)
    prices.to_parquet(PRICE_CACHE)
    return prices


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Run one arm through the real scoring chain.

    ``compute_category_scores`` narrates the weight-0 metrics the payload does
    not carry, identically on every call; 160 repetitions of it would bury the
    result, so its stdout is swallowed. Nothing else here is silenced.
    """
    df = frame.reset_index()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        df = compute_sector_percentiles(df)
        df = compute_category_scores(df, cfg)
        df = compute_composite(df, cfg)
    return df.set_index("Ticker")


def _explain_reconstruction_gap(snap: pd.DataFrame, cfg: dict,
                                pub: pd.Series) -> dict:
    """Account for the gap between ``score()`` and the published composite.

    ``score()`` reproduces ``backtest.simulate_monthly_scores``'s chain:
    percentiles, category scores, composite. The *live* pipeline
    (``run_screener.py``) inserts ``adjust_momentum_weight`` between the last
    two, which rescales the momentum and valuation factor weights to the
    prevailing volatility regime. Adding that one call closes the gap, which is
    how the reconstruction is shown faithful to the thing being measured rather
    than merely close to the site.

    It is also a finding in its own right: the backtest scores a different
    weighting than the site publishes, and the regime is read from the *current*
    run, so a historical rebalance would need the regime of its own month.

    ``adjust_momentum_weight`` **appends a row to
    ``<root>/factor_vol_history.csv``** every time it is called, and that file is
    tracked and feeds the live regime decision. The first version of this script
    passed ``ROOT`` and duplicated the day's row twice — the same
    evidence-inflation shape as ``CLAUDE.md`` priority 0.6, arriving from a
    measurement rather than a run. It is handed a throwaway directory instead,
    seeded with a copy of the real history so the regime it computes is the real
    one. ``tests/test_lookahead.py`` fails if the tracked file moves.
    """
    scratch = Path(tempfile.mkdtemp(prefix="lookahead_vol_"))
    real_hist = ROOT / "factor_vol_history.csv"
    if real_hist.exists():
        shutil.copy2(real_hist, scratch / "factor_vol_history.csv")

    df = snap.reset_index()
    with contextlib.redirect_stdout(io.StringIO()):
        df = compute_sector_percentiles(df)
        df = compute_category_scores(df, cfg)
        cfg_regime = adjust_momentum_weight(df, cfg, str(scratch))
        df = compute_composite(df, cfg_regime)
    shutil.rmtree(scratch, ignore_errors=True)
    df = df.set_index("Ticker")
    both = df["Composite"].dropna().index.intersection(pub.dropna().index)
    diff = (df.loc[both, "Composite"] - pub.loc[both]).abs()
    rr = df["Composite"].rank(ascending=False, method="first")
    pr = pd.to_numeric(snap["published_rank"], errors="coerce")
    return {
        "configured_momentum_weight": float(cfg["factor_weights"]["momentum"]),
        "regime_adjusted_momentum_weight": float(
            cfg_regime["factor_weights"]["momentum"]),
        "regime_adjusted_valuation_weight": float(
            cfg_regime["factor_weights"]["valuation"]),
        "with_regime_step_median_abs_diff": round(float(diff.median()), 4),
        "with_regime_step_max_abs_diff": round(float(diff.max()), 4),
        "with_regime_step_names_over_1pt": int((diff > 1).sum()),
        "with_regime_step_median_abs_rank_diff": round(
            float((rr - pr).abs().median()), 2),
        "with_regime_step_p95_abs_rank_diff": round(
            float((rr - pr).abs().quantile(0.95)), 2),
    }


def deciles(composite: pd.Series, n: int = 10) -> pd.Series:
    """``backtest._assign_deciles``, applied to one arm's composite."""
    return np.ceil(composite.rank(pct=True, method="first") * n).clip(1, n).astype(int)


def dividend_drag(snapshot: pd.DataFrame, months_back: pd.Series) -> pd.Series:
    """Per-ticker factor undoing the dividend part of an adjusted-price ratio.

    An adjusted series reinvests dividends, so ``P_adj(m)/P_adj(T)`` is smaller
    than the raw price ratio by the cumulative payout over the gap. Approximated
    as ``(1 + y) ** (years)`` using the snapshot's own trailing dividend yield —
    good enough to show whether the headline depends on it, which is the only
    claim made for it.
    """
    y = pd.to_numeric(snapshot["div_yield_fin"], errors="coerce").fillna(0.0)
    return (1.0 + y) ** months_back


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="fail rather than fetch the price panel")
    ap.add_argument("--max-months", type=int, default=0,
                    help="limit months scored (smoke test)")
    args = ap.parse_args()

    cfg = load_config()
    buckets = lookahead.weight_buckets(cfg)
    gap = lookahead.restatable_weight_gap(cfg)
    print("\n=== Weight accounting (derived from config.yaml) ===")
    for k in ("recomputed", "price_restatable", "price_derived_held",
              "needs_point_in_time", "held_constant", "unclassified"):
        print(f"  {k:24s} {buckets[k]:7.3f}%")
    print(f"  {'restatable not restated':24s} {gap:7.3f}%")
    assert abs(buckets["recomputed"] + buckets["held_constant"] - 100.0) < 1e-3
    assert buckets["unclassified"] == 0

    snap = load_snapshot()
    print(f"\nSnapshot: {len(snap)} tickers from dashboard_data.js")

    prices = load_prices(snap.index.tolist(), offline=args.offline)
    prices.index = pd.to_datetime(prices.index)
    have = [t for t in snap.index if t in prices.columns]
    snap = snap.loc[have]
    print(f"Priced:   {len(have)} tickers; panel {prices.index.min():%Y-%m} "
          f"to {prices.index.max():%Y-%m} ({len(prices)} months)")

    # The snapshot's own price against the panel's latest close: a check that
    # the ratio denominator is the right number, not an assumption.
    latest = prices.index.max()
    panel_last = prices.loc[latest, have]
    payload_price = pd.to_numeric(snap["price"], errors="coerce")
    rel = (panel_last / payload_price - 1.0).abs().dropna()
    print(f"Panel latest vs payload price: median |diff| "
          f"{rel.median() * 100:.2f}%, p90 {rel.quantile(0.9) * 100:.2f}%")

    # Self-check before measuring anything: scoring the snapshot as-is, with no
    # dynamic override, must reproduce the composite the site published. If it
    # does not, the reconstruction is wrong and every number below is noise.
    # The payload rounds ``raw`` to 4dp and omits the seven weight-0 metrics, so
    # an exact match is not expected — a tight one is.
    recon = score(snap, cfg)["Composite"]
    pub = pd.to_numeric(snap["published_composite"], errors="coerce")
    both = recon.dropna().index.intersection(pub.dropna().index)
    recon_check = {
        "n": int(len(both)),
        "spearman": round(float(recon.loc[both].corr(pub.loc[both], method="spearman")), 6),
        "max_abs_diff": round(float((recon.loc[both] - pub.loc[both]).abs().max()), 4),
        "median_abs_diff": round(float((recon.loc[both] - pub.loc[both]).abs().median()), 4),
    }
    recon_check.update(_explain_reconstruction_gap(snap, cfg, pub))
    print(f"Reconstruction vs published composite: rho={recon_check['spearman']}, "
          f"median |diff|={recon_check['median_abs_diff']}, "
          f"max |diff|={recon_check['max_abs_diff']}")
    print(f"  ...and with the live pipeline's regime step: "
          f"median |diff|={recon_check['with_regime_step_median_abs_diff']}, "
          f"median |rank diff|={recon_check['with_regime_step_median_abs_rank_diff']} "
          f"(momentum {recon_check['configured_momentum_weight']} -> "
          f"{recon_check['regime_adjusted_momentum_weight']})")

    months = [m for m in sorted(prices.index) if m >= prices.index[13]]
    if args.max_months:
        months = months[-args.max_months:]
    print(f"Scoring {len(months)} rebalance months x 2 arms\n")

    per_month = []
    migrate_counts = {"total": 0, "decile_changed": 0, "decile_2plus": 0,
                      "top_decile_a": 0, "top_decile_both": 0,
                      "quintile_changed": 0}
    first_rows = None

    for i, m in enumerate(months):
        # Shared dynamic metrics: identical in both arms by construction.
        dyn = {t: _recompute_momentum_risk(prices, m, t) for t in have}
        dyn_df = pd.DataFrame(dyn).T

        arm_a = snap.copy()
        for c in dyn_df.columns:
            arm_a[c] = dyn_df[c]

        ratio = (prices.loc[m, have] / panel_last).astype(float)
        arm_b = lookahead.restate_at_price(snap, ratio, cfg.get("metric_clamps"))
        for c in dyn_df.columns:
            arm_b[c] = dyn_df[c]

        sa = score(arm_a, cfg)
        sb = score(arm_b, cfg)
        ca, cb = sa["Composite"], sb["Composite"]
        common = ca.dropna().index.intersection(cb.dropna().index)
        ca, cb = ca.loc[common], cb.loc[common]

        da, db = deciles(ca), deciles(cb)
        ra = ca.rank(ascending=False, method="first")
        rb = cb.rank(ascending=False, method="first")

        rho = ca.corr(cb, method="spearman")
        changed = (da != db)
        two_plus = (da - db).abs() >= 2
        quint_changed = (np.ceil(ca.rank(pct=True) * 5) != np.ceil(cb.rank(pct=True) * 5))
        top_a = da == 10
        per_month.append({
            "month": m.strftime("%Y-%m"),
            "n": int(len(common)),
            "spearman": float(rho),
            "decile_changed_pct": float(changed.mean() * 100),
            "decile_2plus_pct": float(two_plus.mean() * 100),
            "median_abs_rank_change": float((ra - rb).abs().median()),
            "p90_abs_rank_change": float((ra - rb).abs().quantile(0.9)),
            "top_decile_retained_pct": float((top_a & (db == 10)).sum()
                                             / max(int(top_a.sum()), 1) * 100),
            "median_abs_composite_change": float((ca - cb).abs().median()),
        })
        migrate_counts["total"] += len(common)
        migrate_counts["decile_changed"] += int(changed.sum())
        migrate_counts["decile_2plus"] += int(two_plus.sum())
        migrate_counts["quintile_changed"] += int(quint_changed.sum())
        migrate_counts["top_decile_a"] += int(top_a.sum())
        migrate_counts["top_decile_both"] += int((top_a & (db == 10)).sum())

        if first_rows is None:
            first_rows = (m, sa, sb)
        if (i + 1) % 12 == 0 or i == len(months) - 1:
            print(f"  {i + 1}/{len(months)} months (latest {m:%Y-%m}, "
                  f"rho={rho:.3f}, decile moved {changed.mean() * 100:.1f}%)")

    pm = pd.DataFrame(per_month)
    panel = {
        "name_months": int(migrate_counts["total"]),
        "decile_changed_pct": round(100 * migrate_counts["decile_changed"]
                                    / migrate_counts["total"], 3),
        "decile_2plus_pct": round(100 * migrate_counts["decile_2plus"]
                                  / migrate_counts["total"], 3),
        "quintile_changed_pct": round(100 * migrate_counts["quintile_changed"]
                                      / migrate_counts["total"], 3),
        "top_decile_retained_pct": round(100 * migrate_counts["top_decile_both"]
                                         / migrate_counts["top_decile_a"], 3),
        "spearman_median": round(float(pm["spearman"].median()), 4),
        "spearman_min": round(float(pm["spearman"].min()), 4),
        "spearman_oldest_month": round(float(pm["spearman"].iloc[0]), 4),
        "spearman_newest_month": round(float(pm["spearman"].iloc[-1]), 4),
        "median_abs_rank_change_median": round(float(pm["median_abs_rank_change"].median()), 2),
    }

    # Does the error grow with distance from the snapshot? The survivorship
    # measurement found a monotone decay toward the present; the same signature
    # should appear here, and if it does not, the measurement is suspect.
    age_months = np.arange(len(pm))[::-1]
    trend_rho = float(pd.Series(age_months).corr(pm["decile_changed_pct"],
                                                 method="spearman"))

    # Sensitivity: does removing the adjusted-series dividend drag move it?
    years_back = pd.Series(
        [(latest - m).days / 365.25 for m in months], index=range(len(months))
    )
    sens = {}
    for label, which in (("oldest", 0), ("midpoint", len(months) // 2)):
        m = months[which]
        drag = dividend_drag(snap, pd.Series(years_back[which], index=snap.index))
        dyn = {t: _recompute_momentum_risk(prices, m, t) for t in have}
        dyn_df = pd.DataFrame(dyn).T
        ratio = (prices.loc[m, have] / panel_last).astype(float) * drag.loc[have]
        arm_b = lookahead.restate_at_price(snap, ratio, cfg.get("metric_clamps"))
        arm_a = snap.copy()
        for c in dyn_df.columns:
            arm_a[c] = dyn_df[c]
            arm_b[c] = dyn_df[c]
        sa, sb = score(arm_a, cfg), score(arm_b, cfg)
        ca, cb = sa["Composite"], sb["Composite"]
        common = ca.dropna().index.intersection(cb.dropna().index)
        da, db = deciles(ca.loc[common]), deciles(cb.loc[common])
        base = pm.loc[pm["month"] == m.strftime("%Y-%m"), "decile_changed_pct"]
        sens[label] = {
            "month": m.strftime("%Y-%m"),
            "years_back": round(float(years_back[which]), 2),
            "decile_changed_pct_adjusted_prices": round(float(base.iloc[0]), 3),
            "decile_changed_pct_dividend_corrected": round(float((da != db).mean() * 100), 3),
        }

    # r == 1 identity: arm B at the snapshot month must reproduce arm A exactly.
    ident_ratio = pd.Series(1.0, index=snap.index)
    ident = lookahead.restate_at_price(snap, ident_ratio, cfg.get("metric_clamps"))
    ident_max = max(
        float(np.nanmax(np.abs(pd.to_numeric(ident[c], errors="coerce")
                               - pd.to_numeric(snap[c], errors="coerce"))))
        for c in ("ev_ebitda", "ev_sales", "fcf_yield", "earnings_yield", "size_log_mcap")
    )

    result = {
        "generated": "2026-10-01",
        "snapshot_source": "dashboard_data.js (2026-10-01 data run)",
        "weight_buckets_pct": buckets,
        "restatable_weight_not_restated_pct": gap,
        "universe": {"snapshot_tickers": int(len(snap)),
                     "priced": int(len(have)),
                     "panel_first_month": months[0].strftime("%Y-%m"),
                     "panel_last_month": months[-1].strftime("%Y-%m"),
                     "rebalance_months": int(len(months))},
        "reconstruction_check": recon_check,
        "panel": panel,
        "age_vs_decile_change_spearman": round(trend_rho, 4),
        "dividend_drag_sensitivity": sens,
        "identity_check_max_abs_diff": ident_max,
        "panel_price_vs_payload_price_median_abs_pct": round(float(rel.median() * 100), 4),
        "per_month": per_month,
    }
    OUT_JSON.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print("\n=== Panel result ===")
    for k, v in panel.items():
        print(f"  {k:34s} {v}")
    print(f"  {'age vs decile-change spearman':34s} {trend_rho:.4f}")
    print(f"  {'r==1 identity max abs diff':34s} {ident_max:.3e}")
    print("\n=== Dividend-drag sensitivity ===")
    for label, s in sens.items():
        print(f"  {label:9s} {s['month']} ({s['years_back']}y back): "
              f"{s['decile_changed_pct_adjusted_prices']}% -> "
              f"{s['decile_changed_pct_dividend_corrected']}%")
    print(f"\nWrote {OUT_JSON.name}")


if __name__ == "__main__":
    main()
