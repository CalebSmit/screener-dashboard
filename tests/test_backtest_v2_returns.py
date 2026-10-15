"""v2's rankings against next-month returns (``backtest_v2_returns.py``, plan/backtest-v2.md step 4).

Pinned: the arithmetic (a ranking that orders next month's returns perfectly has IC 1 and the top
decile wins), the error bars are the month-to-month standard error, a month needs the next
month's price, every output carries its caveats (survivorship, the screener it measures, rule 5),
and production never imports it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backtest_v2_returns as br  # noqa: E402


def _case(n=100, months=("2024-01-31", "2024-02-29", "2024-03-31"), seed=0):
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:03d}" for i in range(n)]
    prices = pd.DataFrame(100.0, index=list(months), columns=tickers)
    for i in range(1, len(months)):
        prices.iloc[i] = prices.iloc[i - 1] * (1 + rng.normal(0, 0.05, n))
    fwd = br.forward_returns(prices)
    scores = []
    for m in months[:-1]:
        r = fwd.loc[m]
        scores.append(pd.DataFrame({"month": m, "Ticker": tickers, "Composite": r.rank().values,
                                    "quality_score": (-r).rank().values}))
    return pd.concat(scores), fwd


def test_a_ranking_that_orders_the_returns_perfectly_has_ic_one():
    scores, fwd = _case()
    out = br.evaluate(scores, fwd)
    assert out["months"] == 2
    assert out["ic"]["mean"] == pytest.approx(1.0)
    assert out["category_ic"]["quality"]["mean"] == pytest.approx(-1.0)
    assert out["spread_top_minus_bottom"]["mean"] > 0
    assert all(r["top"] > r["universe"] > r["bottom"] for r in out["per_month"])


def test_the_last_month_has_no_forward_return_and_is_not_scored():
    scores, fwd = _case()
    assert fwd.iloc[-1].isna().all()
    last = scores.assign(month="2024-03-31")
    assert br.evaluate(last, fwd)["months"] == 0


def test_the_error_bar_is_the_month_to_month_standard_error():
    s = pd.Series([0.02, 0.04, 0.00, 0.06])
    out = br.summarise(s)
    se = s.std(ddof=1) / 2
    assert out["se"] == pytest.approx(se, abs=1e-5)
    assert out["ci95"] == pytest.approx([0.03 - 1.96 * se, 0.03 + 1.96 * se], abs=1e-5)
    assert out["share_positive"] == 0.75
    assert br.summarise(pd.Series([0.1]))["n_months"] == 1          # no interval from one month


def test_every_result_carries_its_caveats():
    scores, fwd = _case()
    out = br.evaluate(scores, fwd)
    text = " ".join(out["caveats"])
    assert "Survivorship" in text and "Revisions" in text and "rule 5" in text


def test_month_end_prices_never_read_past_the_month_end():
    days = pd.bdate_range("2024-01-01", "2024-03-15")
    daily = pd.DataFrame({"ticker": "AAA", "date": days, "adj_close": np.arange(len(days), dtype=float)})
    p = br.month_end_prices(daily, ["2024-01-31", "2024-02-29"])
    assert p.loc["2024-01-31", "AAA"] == float(np.where(days <= "2024-01-31")[0][-1])


def test_no_production_module_imports_it():
    pat = re.compile(r"^\s*(import backtest_v2_returns|from backtest_v2_returns)\b", re.M)
    for p in ROOT.glob("*.py"):
        if p.name in ("backtest_v2_returns.py", "conftest.py"):
            continue
        assert not pat.search(p.read_text(encoding="utf-8", errors="replace")), p.name


def test_v1s_disclaimer_quotes_the_measured_lookahead_not_a_guess():
    """backtest.py v1 used to print an unsourced '2-6% annualized' inflation; the measured effect
    of look-ahead alone on this pipeline is about 1.1% a month. Its figures must be the record's."""
    import json
    src = (ROOT / "backtest.py").read_text(encoding="utf-8")
    assert "2-6% annualized" not in src.replace("('2-6% annualized')", "")
    honest = json.loads((ROOT / "research" / "measurements" / "2026-10-15-backtest-v2-returns.json")
                        .read_text(encoding="utf-8"))
    la = json.loads((ROOT / "research" / "measurements" / "2026-10-15-backtest-v2-returns-lookahead.json")
                    .read_text(encoding="utf-8"))
    assert f"{honest['top_minus_universe']['mean'] * 100:.2f}%/month" in src            # -0.26
    assert f"+{la['top_minus_universe']['mean'] * 100:.2f}%/month (t {la['top_minus_universe']['t']:.1f})" in src
    assert f"{honest['ic']['mean']:.3f}" in src and f"+{la['ic']['mean']:.3f}" in src
    assert "research/2026-10-15-pit-record-validation.md" in src
