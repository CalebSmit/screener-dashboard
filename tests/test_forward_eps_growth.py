"""forward_eps_growth: MSCI's 12-month-forward construction (2026-10-09).

research/2026-10-09-forward-eps-growth.md. Next 12 months' EPS blends the current- and
next-fiscal-year consensus by the months M left in the current fiscal year; it is compared with the
sum of the last four reported quarters on the same (consensus) basis; $1 floor on the denominator;
clipped to -75% .. +150%. A base that is a loss gives no growth rate.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tests.test_metrics import _compute_one, _make_rec


def _rec(fy1=5.0, fy2=6.0, quarters=(1.0, 1.0, 1.0, 1.0), months_left=6.0, **kw):
    end = datetime.now(timezone.utc) + timedelta(days=months_left * 30.4375)
    q = [[f"2026-0{i + 1}-30", a, a, 0.0] for i, a in enumerate(quarters)]
    return _make_rec(_fy1_eps_current=fy1, _fy2_eps_current=fy2, _next_fy_end=int(end.timestamp()),
                     _eps_quarters=json.dumps(q), **kw)


def test_the_twelve_month_blend():
    r = _compute_one(_rec(fy1=5.0, fy2=6.0, quarters=(1.0, 1.0, 1.0, 1.0), months_left=6.0))
    # F12 = (6 x 5 + 6 x 6) / 12 = 5.5 against B12 = 4.0
    assert r["_feg_basis"] == "msci_12m"
    assert r["_feg_f12"] == pytest.approx(5.5, rel=1e-3)
    assert r["forward_eps_growth"] == pytest.approx((5.5 - 4.0) / 4.0, rel=1e-3)


def test_months_left_weights_the_current_year():
    near_end = _compute_one(_rec(fy1=5.0, fy2=6.0, months_left=1.0))
    assert near_end["_feg_f12"] == pytest.approx((1 * 5 + 11 * 6) / 12, rel=1e-2)


def test_a_small_base_uses_the_one_dollar_floor():
    r = _compute_one(_rec(fy1=1.0, fy2=1.0, quarters=(0.1, 0.1, 0.1, 0.1)))
    assert r["forward_eps_growth"] == pytest.approx((1.0 - 0.4) / 1.0, rel=1e-3)


def test_a_loss_base_has_no_growth_rate():
    # GILD, 2026-10-09: four quarters summing to -$0.39 scored +150% (the cap)
    r = _compute_one(_rec(fy1=7.0, fy2=8.0, quarters=(1.5, -4.0, 1.0, 1.11)))
    assert r["_feg_basis"] == "loss_base"
    assert np.isnan(r["forward_eps_growth"])


def test_growth_is_clipped():
    assert _compute_one(_rec(fy1=40.0, fy2=40.0))["forward_eps_growth"] == pytest.approx(1.5)
    assert _compute_one(_rec(fy1=0.2, fy2=0.2, quarters=(2.0, 2.0, 2.0, 2.0)))["forward_eps_growth"] == pytest.approx(-0.75)


def test_yahoos_zero_placeholder_is_not_an_estimate():
    r = _compute_one(_rec(fy1=0.0, fy2=6.0))
    assert r.get("_feg_basis") != "msci_12m"


def test_without_consensus_inputs_the_old_form_is_the_fallback():
    r = _compute_one(_make_rec(forwardEps=5.5, trailingEps=5.0))
    assert r["_feg_basis"] == "fy2_over_trailing"
    assert r["forward_eps_growth"] == pytest.approx(0.1, rel=1e-3)
