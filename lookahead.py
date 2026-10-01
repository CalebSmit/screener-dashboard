#!/usr/bin/env python3
"""
lookahead.py — size the look-ahead bias in ``backtest.py``.

**This is a diagnostic instrument, not a fix.** Nothing here makes the backtest
honest and nothing here may be wired into it. `plan/backtest-v2.md` is explicit
that a half-fixed backtest invites exactly the false confidence the bench period
(`CLAUDE.md` rule 5, until 2027-02-11) exists to prevent, and restating a
valuation ratio at a historical *price* while leaving its *fundamental* at
today's value is still look-ahead — just less of it. The point of measuring it
is to find out how much, so the plan's next step can be sized.

``tests/test_lookahead.py`` asserts that neither ``backtest.py`` nor
``run_screener.py`` imports this module. If a future session wires it in, that
test fails and the reason is here.

What look-ahead means in this harness
-------------------------------------
``backtest.simulate_monthly_scores()`` takes one Phase-1 snapshot of all 44
metrics, recomputes four of them from trailing prices at each month-end, and
leaves the rest at their snapshot values. The module docstring describes this as
"Only Momentum and Risk metrics are recomputed from trailing prices", which
overstates what happens: its ``dynamic_cols`` list holds four names, and
Momentum and Risk carry six weighted metrics between them.

So every weighted metric falls into exactly one of four buckets, and this module
is where that classification lives:

``RECOMPUTED``
    Recomputed at each rebalance from the trailing price panel. The only part of
    the score that is honestly point-in-time.
``PRICE_RESTATABLE``
    Held constant, but a single month-end price restates it **exactly** — these
    are ratios of a price-independent fundamental to a market value. The
    harness already holds the price panel it would need, so this is look-ahead
    on data that was never missing.
``PRICE_DERIVED_HELD``
    Held constant, derived from nothing but a price *history*. Recomputable in
    principle from the same panel, but not from one month-end price, so this
    module cannot restate them and does not pretend to.
``NEEDS_POINT_IN_TIME``
    Held constant and genuinely unknowable without point-in-time filings or
    analyst estimates. This is the bucket `plan/backtest-v2.md` step 3 is about,
    and the one that needs a data source this project does not have.

``weight_buckets(cfg)`` returns the composite-weight share of each, derived from
``config.yaml`` rather than written down, so a reweight cannot leave a stale
number in a doc (the 2026-09-10 failure mode — see ``NIGHTLY_LOG.md``
2026-09-28).

The restatement algebra
----------------------
Let ``r = P(m) / P(snapshot)`` for one ticker, and hold every price-independent
quantity (net income, FCF, EBITDA, revenue, share count, the analyst target, the
non-equity part of enterprise value) at its snapshot value. Then:

* ``market_cap(m)  = market_cap * r``
* ``ev(m)          = ev + market_cap * (r - 1)``   — the non-equity part of EV
  (debt, cash, minority interest, preferred) does not move with the share price,
  so it is held, and the equity part scales. ``ev - market_cap`` is used rather
  than a reconstructed net debt because that difference is what the vendor's own
  ``enterpriseValue`` implies, and ``factor_engine`` scores on that figure.
* ``earnings_yield(m) = earnings_yield / r``        — net income / market cap
* ``fcf_yield(m)      = fcf_yield * ev / ev(m)``    — FCF / EV
* ``ev_ebitda(m)      = ev_ebitda * ev(m) / ev``
* ``ev_sales(m)       = ev_sales  * ev(m) / ev``
* ``size_log_mcap(m)  = size_log_mcap - log(r)``    — the metric is ``-log(mc)``
* ``price_target_upside(m) = pt_mean / (price * r) - 1``, then clamped

``price_target_upside`` is rebuilt from the snapshot's ``pt_mean`` and ``price``
rather than by inverting the metric, because the metric is already clamped to
``metric_clamps.price_target_upside`` — inverting a clamped value silently
invents a target for every name sitting on the bound.

Every pipeline guard is reproduced rather than approximated: a metric that is
NaN in the snapshot stays NaN (there is nothing to restate), and ``ev(m) <= 0``
withholds the three EV-based metrics exactly as ``factor_engine`` does, which
matters for net-cash names at low prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# No ``factor_engine`` import: the bucket classification is a statement about
# ``config.yaml`` and ``backtest.py``, and keeping this module free of the
# scoring engine is what lets ``tests/test_lookahead.py`` assert the dependency
# runs one way only.

# --- Bucket classification -------------------------------------------------
# Every entry of METRIC_COLS appears in exactly one bucket. A test asserts that,
# so adding a metric to the registry without classifying it fails the suite
# rather than quietly dropping out of the weight accounting.

#: Recomputed per rebalance by ``backtest.simulate_monthly_scores`` — this must
#: stay identical to that function's ``dynamic_cols`` list, and a test pins it.
RECOMPUTED = ("return_12_1", "return_6m", "volatility", "beta")

#: Held constant, exactly restatable from one month-end price.
PRICE_RESTATABLE = (
    "ev_ebitda",
    "fcf_yield",
    "earnings_yield",
    "ev_sales",
    "pb_ratio",
    "dividend_yield",
    "peg_ratio",
    "price_target_upside",
    "size_log_mcap",
)

#: Held constant, derived from a price *history* rather than a point price.
PRICE_DERIVED_HELD = (
    "jensens_alpha",
    "max_drawdown_1y",
    "sharpe_ratio",
    "sortino_ratio",
    "proximity_52w_high",
)

#: Held constant and genuinely unknowable without point-in-time data.
NEEDS_POINT_IN_TIME = (
    "roic",
    "gross_profit_assets",
    "debt_equity",
    "net_debt_to_ebitda",
    "piotroski_f_score",
    "accruals",
    "operating_leverage",
    "beneish_m_score",
    "roe",
    "roa",
    "equity_ratio",
    "operating_margin",
    "current_ratio",
    "insider_ownership",
    "interest_coverage",
    "forward_eps_growth",
    "revenue_growth",
    "revenue_cagr_3yr",
    "sustainable_growth",
    "fy1_revision_3m",
    "analyst_surprise",
    "earnings_acceleration",
    "consecutive_beat_streak",
    "short_interest_ratio",
    "short_pct_float",
    "analyst_rating",
    "asset_growth",
)

BUCKETS = {
    "recomputed": RECOMPUTED,
    "price_restatable": PRICE_RESTATABLE,
    "price_derived_held": PRICE_DERIVED_HELD,
    "needs_point_in_time": NEEDS_POINT_IN_TIME,
}

#: The subset of ``PRICE_RESTATABLE`` this module can actually restate from a
#: snapshot payload. ``pb_ratio`` and ``dividend_yield`` carry weight 0 today and
#: would need book value / dividend per share, which the payload does not carry;
#: ``peg_ratio`` carries weight 0 and would need the growth denominator.
#: ``restate_at_price`` leaves anything outside this set untouched, and
#: ``restatable_weight_gap`` reports the weight that leaves unmeasured so a
#: future reweight cannot make the measurement silently incomplete.
RESTATED_HERE = (
    "ev_ebitda",
    "fcf_yield",
    "earnings_yield",
    "ev_sales",
    "price_target_upside",
    "size_log_mcap",
)


#: Every metric this module has an opinion about. A test asserts it equals
#: ``factor_engine.METRIC_COLS``, so a new metric cannot join the registry
#: without landing in a bucket and entering the weight accounting.
ALL_CLASSIFIED = frozenset(
    RECOMPUTED + PRICE_RESTATABLE + PRICE_DERIVED_HELD + NEEDS_POINT_IN_TIME
)


def composite_metric_weights(cfg: dict) -> dict:
    """Return ``{metric: share of composite weight in percent}``.

    The product of the category's share of ``factor_weights`` and the metric's
    share of its category's ``metric_weights``. Shares sum to 100 whenever every
    weighted metric belongs to a scored category.
    """
    fw = cfg.get("factor_weights", {}) or {}
    mw = cfg.get("metric_weights", {}) or {}
    fw_total = float(sum(float(v) for v in fw.values()))
    if fw_total <= 0:
        return {}

    out = {}
    for cat, cat_weight in fw.items():
        metrics = mw.get(cat, {}) or {}
        cat_total = float(sum(float(v) for v in metrics.values()))
        cat_share = float(cat_weight) / fw_total
        for metric, w in metrics.items():
            if cat_total <= 0:
                out[metric] = 0.0
            else:
                out[metric] = 100.0 * cat_share * (float(w) / cat_total)
    return out


def weight_buckets(cfg: dict) -> dict:
    """Composite-weight share of each look-ahead bucket, in percent.

    Derived from ``config.yaml``, never written down. ``held_constant`` is the
    sum of the three non-recomputed buckets and is what ``backtest.py``'s
    docstring is implicitly claiming is small.
    """
    per_metric = composite_metric_weights(cfg)
    shares = {}
    for name, metrics in BUCKETS.items():
        shares[name] = round(sum(per_metric.get(m, 0.0) for m in metrics), 4)
    shares["held_constant"] = round(
        shares["price_restatable"]
        + shares["price_derived_held"]
        + shares["needs_point_in_time"],
        4,
    )
    shares["unclassified"] = round(
        sum(w for m, w in per_metric.items() if m not in ALL_CLASSIFIED), 4
    )
    return shares


def restatable_weight_gap(cfg: dict) -> float:
    """Weight share that is price-restatable in principle but not restated here.

    Zero today. If a future reweight turns on ``pb_ratio``, ``dividend_yield`` or
    ``peg_ratio``, this goes positive and the measurement understates look-ahead
    by that much — which is a number a reader needs, not a footnote.
    """
    per_metric = composite_metric_weights(cfg)
    gap = sum(
        per_metric.get(m, 0.0) for m in PRICE_RESTATABLE if m not in RESTATED_HERE
    )
    return round(gap, 4)


# --- Restatement -----------------------------------------------------------
#: Columns a snapshot frame must carry for ``restate_at_price`` to work.
REQUIRED_SNAPSHOT_COLS = (
    "market_cap",
    "enterprise_value",
    "price",
)


def restate_at_price(snapshot: pd.DataFrame, price_ratio: pd.Series,
                     clamps: dict | None = None) -> pd.DataFrame:
    """Restate the price-dependent metrics of ``snapshot`` at a historical price.

    Args:
        snapshot: one row per ticker, indexed by ticker. Must carry
            ``market_cap``, ``enterprise_value`` and ``price`` alongside the
            metric columns; ``pt_mean`` is used for
            ``price_target_upside`` when present.
        price_ratio: ``P(historical) / P(snapshot)`` per ticker, same index.
        clamps: ``config['metric_clamps']``; only
            ``price_target_upside`` is consulted.

    Returns:
        A copy of ``snapshot`` with the ``RESTATED_HERE`` columns replaced. Rows
        whose ratio is missing or non-positive are returned **unchanged except
        that the restated metrics are NaN** — a non-positive price is not a
        price, and silently keeping the snapshot value is the exact bias being
        measured.
    """
    out = snapshot.copy()
    missing = [c for c in REQUIRED_SNAPSHOT_COLS if c not in out.columns]
    if missing:
        raise ValueError(
            f"restate_at_price: snapshot is missing required column(s) {missing}. "
            "Restating a valuation ratio needs the market value it was divided by."
        )

    r = pd.to_numeric(price_ratio, errors="coerce").reindex(out.index)
    valid = r.notna() & (r > 0)

    mc0 = pd.to_numeric(out["market_cap"], errors="coerce")
    ev0 = pd.to_numeric(out["enterprise_value"], errors="coerce")
    p0 = pd.to_numeric(out["price"], errors="coerce")

    mc_h = mc0 * r
    # Hold the non-equity component of EV: debt, cash, minority interest and
    # preferred do not move with the share price.
    ev_h = ev0 + mc0 * (r - 1.0)
    # factor_engine withholds every EV-based metric when EV is not positive.
    ev_ok = valid & ev_h.notna() & (ev_h > 0) & ev0.notna() & (ev0 > 0)
    ev_scale = (ev_h / ev0).where(ev_ok)

    def _col(name):
        """A numeric view of ``name``, or an all-NaN series when it is absent.

        A snapshot that does not carry a metric is a snapshot that cannot restate
        it — that is a missing column, not an error, and ``_set`` skips it. Going
        through ``pd.to_numeric(out.get(name))`` directly would raise on the
        ``None`` instead.
        """
        if name not in out.columns:
            return pd.Series(np.nan, index=out.index, dtype="float64")
        return pd.to_numeric(out[name], errors="coerce")

    def _set(col, values, ok):
        if col not in out.columns:
            return
        # A metric absent from the snapshot cannot be restated.
        out[col] = values.where(ok & _col(col).notna(), np.nan)

    _set("earnings_yield", _col("earnings_yield") / r, valid)
    _set("fcf_yield", _col("fcf_yield") / ev_scale, ev_ok)
    _set("ev_ebitda", _col("ev_ebitda") * ev_scale, ev_ok)
    _set("ev_sales", _col("ev_sales") * ev_scale, ev_ok)
    _set("size_log_mcap", _col("size_log_mcap") - np.log(r.where(valid)),
         valid & mc_h.notna() & (mc_h > 0))

    if "price_target_upside" in out.columns:
        pt = _col("pt_mean")
        p_h = p0 * r
        upside = pt / p_h - 1.0
        lo, hi = (clamps or {}).get("price_target_upside", [-0.50, 1.0])
        upside = upside.clip(float(lo), float(hi))
        base = pd.to_numeric(out["price_target_upside"], errors="coerce")
        ok = valid & base.notna() & pt.notna() & p_h.notna() & (p_h > 0)
        out["price_target_upside"] = upside.where(ok, np.nan)

    # market_cap / enterprise_value / price are carried through restated so a
    # caller can chain, and so a reader can check the algebra.
    out["market_cap"] = mc_h.where(valid, np.nan)
    out["enterprise_value"] = ev_h.where(valid, np.nan)
    out["price"] = (p0 * r).where(valid, np.nan)
    return out
