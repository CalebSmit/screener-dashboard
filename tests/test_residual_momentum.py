"""Residual momentum in place of Jensen's alpha (2026-11-24).

research/2026-11-20-jensens-alpha-in-momentum.md, decision section 9; METHODOLOGY_CHANGELOG.md
2026-11-24. Blitz, Huij & Martens (2011), "Residual momentum", Journal of Empirical Finance 18(3),
506-521, with CAPM on the S&P 500 total-return index in place of three Fama-French factors: a fit
over 36 monthly returns (t-36..t-1, the paper's p. 11), the residuals of months t-12..t-2 summed
and divided by their standard deviation, the fitted intercept left out (p. 12).

What these tests hold, beyond the arithmetic:
- the longer history the fetch now requests (37 months, one request) must not change the window of
  any other price input - volatility, beta's daily returns, drawdown, dollar volume and the 12-1 /
  6-1 prices stay on 13 months (the trap the note's section 7 item 2 named);
- the momentum panic rule reads its last 24 months and 126 days only, so the market series' longer
  request (3y -> 4y) cannot move it;
- a listing younger than three years has no value (missing within Momentum, shared by the other
  two metrics - the existing within-category rule), never a value fitted on fewer months;
- the split-integrity check runs over the whole history the metric reads;
- backtest v2 builds the same input through the same function (CLAUDE.md row 0.8's one computation).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import factor_engine as fe  # noqa: E402
import metric_lineage as ml  # noqa: E402


def _daily(months: int, end="2026-11-20", seed=1, drift=0.0004, vol=0.015, start_price=100.0):
    end = pd.Timestamp(end)
    idx = pd.bdate_range(end - pd.DateOffset(months=months), end)
    rng = np.random.default_rng(seed)
    px = start_price * np.exp(np.cumsum(rng.normal(drift, vol, len(idx))))
    return pd.DataFrame({"Close": px, "Volume": 1e6, "Stock Splits": 0.0}, index=idx)


# --------------------------------------------------------------------------- the arithmetic
def test_the_value_is_the_formation_residuals_over_their_sd():
    rng = np.random.default_rng(3)
    m = 100 * np.cumprod(np.r_[1.0, 1 + rng.normal(0.008, 0.04, 36)])
    s = 40 * np.cumprod(np.r_[1.0, 1 + rng.normal(0.01, 0.08, 36)])
    out = fe.residual_momentum(list(s), list(m))
    rs, rm = s[1:] / s[:-1] - 1, m[1:] / m[:-1] - 1
    beta, alpha = np.polyfit(rm, rs, 1)                    # an independent least-squares fit
    e = rs - (alpha + beta * rm)
    f = e[-12:-1]
    assert len(f) == fe.RESIDUAL_MOMENTUM_FORMATION == 11
    assert out["beta"] == pytest.approx(beta)
    assert out["alpha"] == pytest.approx(alpha)
    assert out["sum"] == pytest.approx(f.sum())
    assert out["sd"] == pytest.approx(f.std(ddof=1))
    assert out["value"] == pytest.approx(f.sum() / f.std(ddof=1))
    # the page's equation is the engine's division
    assert ml.RECOMPUTE["residual_momentum"]({"rm_sum": out["sum"], "rm_sd": out["sd"]}) == pytest.approx(out["value"])


def test_a_constant_risk_free_rate_cannot_change_the_residuals():
    """Why the engine leaves the risk-free rate out: subtracting the same rate from both returns
    moves only the intercept."""
    rng = np.random.default_rng(5)
    rm = rng.normal(0.008, 0.04, 36)
    rs = 0.002 + 1.3 * rm + rng.normal(0, 0.05, 36)
    rf = 0.004
    e1 = rs - np.polyval(np.polyfit(rm, rs, 1), rm)
    e2 = (rs - rf) - np.polyval(np.polyfit(rm - rf, rs - rf, 1), rm - rf)
    assert np.allclose(e1, e2)


def test_a_stock_that_is_the_market_scaled_has_no_value():
    """Residuals of exactly zero have no standard deviation: no number, never a division by ~0."""
    rng = np.random.default_rng(9)
    rm = rng.normal(0.008, 0.04, 36)
    m = 100 * np.cumprod(np.r_[1.0, 1 + rm])
    s = 50 * np.cumprod(np.r_[1.0, 1 + 0.001 + 1.5 * rm])
    assert fe.residual_momentum(list(s), list(m)) is None


@pytest.mark.parametrize("bad", ["short", "nan", "zero"])
def test_unusable_inputs_give_no_value(bad):
    m = list(np.linspace(100, 140, 37))
    s = list(np.linspace(50, 80, 37) * (1 + 0.05 * np.sin(np.arange(37))))
    if bad == "short":
        s = s[1:]
    elif bad == "nan":
        s[5] = float("nan")
    else:
        s[5] = 0.0
    assert fe.residual_momentum(s, m) is None


# --------------------------------------------------------------------------- the fetch's input
def test_thirty_seven_monthly_closes_from_three_years_of_history():
    h = _daily(37)
    out = fe.residual_momentum_fields(h)
    closes = out["_rm_closes"]
    assert len(closes) == fe.RESIDUAL_MOMENTUM_MONTHS + 1 == 37
    dates = sorted(closes)
    assert dates[-1] == "2026-11-20"
    assert dates[0] == "2023-11-20"
    # each is the close on or before its anchor
    for d, v in closes.items():
        assert v == pytest.approx(float(h["Close"].loc[:d].iloc[-1]))


def test_a_listing_younger_than_three_years_has_no_input():
    assert fe.residual_momentum_fields(_daily(30)) == {}


def test_a_mixed_price_scale_anywhere_in_the_three_years_withholds_it():
    """The split check runs over the whole window the metric reads, not only the 13 months the
    Risk inputs use: a 2:1 split two years back that the series applied to only some days."""
    h = _daily(37, seed=4)
    mid = h.index[len(h) // 3]
    h.loc[h.index < mid, "Close"] *= 2.0       # a 2:1 split, declared, applied to half the days
    bad = h.copy()
    bad.loc[h.index[len(h) // 6], "Close"] /= 2.0
    bad.loc[mid, "Stock Splits"] = 2.0
    out = fe.residual_momentum_fields(bad)
    assert "_rm_closes" not in out


def test_the_risk_inputs_keep_their_thirteen_months():
    """The fetch requests 37 months now; every input it computed from 13 months is computed from
    the same 13 months, by backtest v2's own window rule."""
    full = _daily(37, seed=11)
    thirteen = full[full.index > full.index[-1] - pd.DateOffset(months=13)]
    a = fe.price_history_fields(fe.risk_window(full))
    b = fe.price_history_fields(thirteen)
    assert fe.RISK_WINDOW_MONTHS == 13
    for k in ("price_1m_ago", "price_6m_ago", "price_12m_ago", "volatility_1y",
              "_vol_daily_sd", "avg_daily_dollar_volume"):
        assert a[k] == pytest.approx(b[k]), k
    assert a["_daily_returns"] == b["_daily_returns"]
    assert len(a["_daily_returns"]) < 300                      # not three years of returns


def test_the_fetch_makes_one_history_request_and_trims_it():
    src = (ROOT / "factor_engine.py").read_text(encoding="utf-8")
    body = src[src.index("def _fetch_single_ticker_inner"):src.index("def fetch_market_returns")]
    assert body.count("t.history(") == 1, "one price-history request per ticker (settled row ctx)"
    assert "period=RESIDUAL_MOMENTUM_HISTORY" in body
    assert "price_history_fields(hist)" in body and "hist = risk_window(hist_full)" in body
    assert fe.RESIDUAL_MOMENTUM_HISTORY == "37mo"


def test_the_longer_market_series_cannot_move_the_panic_rule():
    import yaml
    rule = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["momentum_regime"]
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2022-11-21", "2026-11-20")
    closes = pd.Series(1000 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(idx)))), index=idx)
    three = closes[closes.index > closes.index[-1] - pd.DateOffset(years=3)]
    assert fe.market_panic_state(closes, rule) == fe.market_panic_state(three, rule)
    import inspect
    assert inspect.signature(fe.fetch_market_closes).parameters["period"].default == "4y"


# --------------------------------------------------------------------------- compute_metrics
def _rec_with(h: pd.DataFrame, ticker="AAA"):
    sys.path.insert(0, str(ROOT / "tests"))
    from test_metrics import _make_rec
    r = _make_rec()
    r["Ticker"] = ticker
    r.update(fe.residual_momentum_fields(h))
    return r


def test_compute_metrics_scores_it_and_publishes_its_terms():
    h = _daily(37, seed=21, vol=0.02)
    mkt = _daily(48, seed=22, vol=0.01, start_price=1000.0)["Close"]
    df = fe.compute_metrics([_rec_with(h)], pd.Series(dtype=float), market_closes=mkt)
    row = df.iloc[0]
    dates = sorted(fe.residual_momentum_fields(h)["_rm_closes"])
    exp = fe.residual_momentum(list(h["Close"].loc[[h["Close"].loc[:d].index[-1] for d in dates]]),
                               fe.closes_at_anchors(mkt, [pd.Timestamp(d) for d in dates]))
    assert row["residual_momentum"] == pytest.approx(exp["value"])
    assert row["_rm_sum"] / row["_rm_sd"] == pytest.approx(row["residual_momentum"])
    assert row["_rm_from"] == dates[0]
    assert "jensens_alpha" not in df.columns


def test_without_the_market_series_it_is_missing_not_guessed():
    df = fe.compute_metrics([_rec_with(_daily(37))], pd.Series(dtype=float))
    assert pd.isna(df.iloc[0]["residual_momentum"])


# --------------------------------------------------------------------------- the registry
def test_it_replaces_jensens_alpha_at_the_same_weight():
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    mom = cfg["metric_weights"]["momentum"]
    assert "jensens_alpha" not in mom
    assert float(mom["residual_momentum"]) == 25.0, "one change at a time: the construction, not the weight"
    assert fe.CAT_METRICS["momentum"].count("residual_momentum") == 1
    assert "jensens_alpha" not in fe.METRIC_COLS
    assert fe.METRIC_DIR["residual_momentum"] is True        # higher is better


def test_the_row_shows_the_engines_division():
    exact, templates = ml.EQUATIONS["residual_momentum"]
    assert exact is True
    assert [k for k, _ in ml.template_slots(templates[0])] == ["rm_sum", "rm_sd"]
    for k in ("rm_sum", "rm_sd", "rm_beta", "rm_from"):
        assert k in ml.ENGINE_KEYS
    assert "jensens_alpha" not in ml.LINEAGE and "jensens_alpha" not in ml.EQUATIONS


def test_backtest_v2_builds_the_same_input_through_the_same_function():
    import pit_record as pr
    assert {"_rm_closes", "_rm_rejected"} <= set(pr.FIELD_CLASSES["price_history"])
    src = (ROOT / "backtest_v2.py").read_text(encoding="utf-8")
    assert "fe.residual_momentum_fields(" in src
    assert "market_closes=market_closes" in src
    import backtest_v2 as v2
    assert v2.RM_WINDOW_MONTHS == 37
