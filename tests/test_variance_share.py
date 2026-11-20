"""Each category's share of the composite's spread, beside its weight (2026-11-19, display only).

A weight is not an influence: on 2026-11-19 Growth was weighted 13% and drove ~2% of the ranking.
`generate_dashboard.category_variance_shares`; `research/2026-11-19-fresh-eyes-review.md` finding 7.
"""
import numpy as np
import pandas as pd
import pytest

import generate_dashboard as g

FW = dict(valuation=22, quality=22, growth=13, momentum=13, risk=10, revisions=10, size=5,
          investment=5)


def _df(n=200, seed=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"{c}_score": rng.uniform(0, 100, n) for c in FW})


def test_the_shares_sum_to_one_and_follow_the_weights_when_categories_are_independent():
    sh = g.category_variance_shares(_df(5000), FW, 50.0)
    assert sum(sh.values()) == pytest.approx(1.0, abs=1e-3)
    # independent, equal-variance categories: share ~ w^2 / sum w^2
    w2 = {c: (w / 100) ** 2 for c, w in FW.items()}
    tot = sum(w2.values())
    for c in FW:
        assert sh[c] == pytest.approx(w2[c] / tot, abs=0.02)


def test_a_category_that_runs_against_the_rest_drives_less_than_its_weight():
    df = _df(2000)
    rest = df[["valuation_score", "quality_score"]].mean(axis=1)
    df["growth_score"] = 100 - rest + np.random.default_rng(1).normal(0, 5, len(df))
    sh = g.category_variance_shares(df, FW, 50.0)
    assert sh["growth"] < 0.13 / 2


def test_a_missing_category_follows_the_runs_own_rule():
    df = _df(50)
    df.loc[0, "risk_score"] = np.nan
    assert g.category_variance_shares(df, FW, 50.0) is not None          # filled at 50
    assert g.category_variance_shares(df.iloc[:10], FW, None) is None    # too few after dropping


def test_a_zero_weight_category_is_not_a_term():
    sh = g.category_variance_shares(_df(500), dict(FW, investment=0), 50.0)
    assert "investment" not in sh and sum(sh.values()) == pytest.approx(1.0, abs=1e-3)


def test_the_table_prints_it_beside_the_weight():
    src = g.generate_html()
    assert ">Drives</th>" in src and "D.variance_share" in src
