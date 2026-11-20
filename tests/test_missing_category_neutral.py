"""A category a stock cannot be scored on counts at 50 at its full weight (2026-11-19).

`research/2026-11-19-missing-category-at-neutral.md`. Before, the category's weight was shared
across the stock's other categories - the same as filling it with the stock's own average - and
SanDisk went from 7th to 1st when its price history failed to download.
"""
import math

import numpy as np
import pandas as pd
import pytest

import calc_trace
import factor_engine as fe
import stock_summary

CATS = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]
FW = dict(valuation=22, quality=22, growth=13, momentum=13, risk=10, revisions=10, size=5,
          investment=5)


def _cfg(fw=None):
    return {"factor_weights": dict(fw or FW),
            "data_quality": {"coverage_discount": {"enabled": False}}}


def _frame(rows):
    df = pd.DataFrame(rows)
    for c in CATS:
        if f"{c}_score" not in df.columns:
            df[f"{c}_score"] = np.nan
    return df


def test_the_engine_and_the_independent_checker_agree_on_the_constant():
    assert fe.MISSING_CATEGORY_SCORE == calc_trace.MISSING_CATEGORY_SCORE == 50.0


def test_a_missing_category_counts_at_50_not_at_the_stocks_own_average():
    """SNDK's shape: strong on six categories, no Momentum or Risk."""
    strong = {f"{c}_score": 90.0 for c in CATS}
    strong.update(momentum_score=np.nan, risk_score=np.nan, Ticker="GAP")
    out = fe.compute_composite(_frame([strong]), _cfg())
    expected = (90.0 * 77 + 50.0 * 23) / 100
    assert out["Composite"].iloc[0] == pytest.approx(expected, abs=0.01)
    assert out["Composite"].iloc[0] < 90.0 - 5   # the old rule gave 90.0


def test_the_rule_is_symmetric_a_weak_stock_is_not_dragged_by_a_gap():
    weak = {f"{c}_score": 10.0 for c in CATS}
    weak.update(quality_score=np.nan, Ticker="WEAK")
    out = fe.compute_composite(_frame([weak]), _cfg())
    assert out["Composite"].iloc[0] == pytest.approx((10.0 * 78 + 50.0 * 22) / 100, abs=0.01)


def test_a_fully_scored_stock_is_unchanged():
    full = {f"{c}_score": float(i * 10 + 5) for i, c in enumerate(CATS)}
    full["Ticker"] = "FULL"
    out = fe.compute_composite(_frame([full]), _cfg())
    expected = sum((i * 10 + 5) * FW[c] for i, c in enumerate(CATS)) / 100
    assert out["Composite"].iloc[0] == pytest.approx(expected, abs=0.01)


def test_a_category_switched_off_for_everyone_is_still_redistributed():
    """Auto-disable / a 0 profile weight is a universe-wide decision, not a data gap."""
    fw = dict(FW, investment=0)
    row = {f"{c}_score": 80.0 for c in CATS}
    row.update(investment_score=np.nan, Ticker="X")
    out = fe.compute_composite(_frame([row]), _cfg(fw))
    assert out["Composite"].iloc[0] == pytest.approx(80.0, abs=0.01)


def test_contributions_count_the_filled_category_and_sum_to_the_composite():
    row = {f"{c}_score": 70.0 for c in CATS}
    row.update(momentum_score=np.nan, risk_score=np.nan, Ticker="GAP")
    df = fe.compute_composite(_frame([row]), _cfg())
    df = fe.compute_factor_contributions(df, _cfg())
    assert df["momentum_contrib"].iloc[0] == pytest.approx(50.0 * 13 / 100, abs=0.01)
    assert df["risk_contrib"].iloc[0] == pytest.approx(50.0 * 10 / 100, abs=0.01)
    total = sum(df[f"{c}_contrib"].iloc[0] for c in CATS)
    assert total == pytest.approx(df["Composite"].iloc[0], abs=0.05)


def test_the_checker_reproduces_a_filled_composite_and_rejects_the_old_rule():
    weights = {"factor_weights": dict(FW), "missing_category_score": 50.0}
    scores = {c: 90.0 for c in CATS}
    scores.update(momentum=None, risk=None)
    new = (90.0 * 77 + 50.0 * 23) / 100
    stock = {"cat_scores": scores, "composite": round(new, 2), "cov": {},
             "contrib": {c: round((scores[c] if scores[c] is not None else 50.0) * FW[c] / 100, 2)
                         for c in CATS}}
    t = calc_trace.composite_trace(weights, stock)
    assert t["composite"] == pytest.approx(new, abs=0.011)
    # The shipped-until-2026-11-19 composite (weight shared out: 90.0) must not reproduce.
    old = dict(stock, composite=90.0)
    assert any("composite" in p for p in calc_trace.verify_stock(weights, old))


def test_the_payload_publishes_the_constant_the_page_prints(tmp_path):
    """The run records the rule it scored with; the generator passes it through and never adds it."""
    import json
    import generate_dashboard as g
    from run_context import RunContext
    ctx = RunContext.__new__(RunContext)
    ctx.run_dir = tmp_path
    path = ctx.save_effective_weights({"factor_weights": dict(FW), "metric_weights": {}})
    saved = json.loads(path.read_text())
    assert saved["missing_category_score"] == fe.MISSING_CATEGORY_SCORE
    assert g._with_weight_profiles(saved, {})["missing_category_score"] == fe.MISSING_CATEGORY_SCORE
    # a run from before the rule: nothing claims it
    assert "missing_category_score" not in g._with_weight_profiles({"factor_weights": dict(FW)}, {})


def test_the_rank_sentence_says_a_gap_counted_at_50():
    detail = {"rank": 3, "composite": 72.36,
              "cat_scores": {c: (None if c in ("momentum", "risk") else 80.0) for c in CATS},
              "contrib": {c: (6.5 if c == "momentum" else 5.0 if c == "risk" else 10.0)
                          for c in CATS}}
    s = stock_summary._sentence_rank(detail, 499)
    assert "its 6 category scores, with the 2 it could not be scored on counted at 50" in s
    assert not stock_summary.advice_terms_in(s)


def test_the_rank_sentence_is_unchanged_for_a_fully_scored_stock():
    detail = {"rank": 1, "composite": 73.4, "cat_scores": {c: 60.0 for c in CATS},
              "contrib": {c: 1.0 for c in CATS}}
    s = stock_summary._sentence_rank(detail, 499)
    assert "counted at 50" not in s and "its 8 category scores" in s


def test_the_methodology_page_states_the_rule_and_not_the_old_one():
    import yaml
    from pathlib import Path
    import run_screener
    root = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    text = run_screener.build_screener_overview(cfg)
    assert "counts at 50, the sector median, at its full weight" in text
    assert "its weight is redistributed to available categories" not in text


def test_a_payload_built_before_the_rule_is_checked_by_the_rule_it_was_built_with():
    """No `missing_category_score` in the payload: weight shared out, as published until 11-19."""
    weights = {"factor_weights": dict(FW)}
    scores = {c: 90.0 for c in CATS}
    scores.update(momentum=None, risk=None)
    stock = {"cat_scores": scores, "composite": 90.0, "cov": {},
             "contrib": {c: (0.0 if scores[c] is None else round(90.0 * FW[c] / 77, 2)) for c in CATS}}
    t = calc_trace.composite_trace(weights, stock)
    assert t["composite"] == pytest.approx(90.0, abs=0.011)
    assert t["published_contrib_sum"] == pytest.approx(t["pre_discount"], abs=0.05)


def test_the_checker_refuses_a_payload_that_states_a_different_constant():
    weights = {"factor_weights": dict(FW), "missing_category_score": 40.0}
    with pytest.raises(ValueError):
        calc_trace.composite_trace(weights, {"cat_scores": {c: 50.0 for c in CATS}})
