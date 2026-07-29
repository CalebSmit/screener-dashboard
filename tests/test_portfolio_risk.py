#!/usr/bin/env python3
"""Tests for portfolio_risk.py (Phase 13 — F10 covariance risk, F11 turnover)."""

import numpy as np
import pandas as pd
import pytest

import portfolio_risk as pr


class TestTurnover:
    def test_identical_holdings_zero_turnover(self):
        t = ["A", "B", "C"]
        res = pr.compute_turnover(t, {"A": 1, "B": 1, "C": 1}, t, {"A": 1, "B": 1, "C": 1})
        assert res["name_turnover"] == 0.0
        assert res["weight_turnover"] == 0.0
        assert res["entered"] == [] and res["exited"] == []

    def test_full_replacement(self):
        res = pr.compute_turnover(["A", "B"], None, ["C", "D"], None)
        assert res["name_turnover"] == 2.0  # 4 symdiff / 2 current
        assert res["entered"] == ["A", "B"]
        assert res["exited"] == ["C", "D"]

    def test_partial_turnover(self):
        res = pr.compute_turnover(["A", "B", "C"], None, ["B", "C", "D"], None)
        # symdiff = {A, D} = 2; /3 current
        assert abs(res["name_turnover"] - 2 / 3) < 1e-3
        assert res["entered"] == ["A"]
        assert res["exited"] == ["D"]

    def test_weight_turnover_computed(self):
        res = pr.compute_turnover(
            ["A", "B"], {"A": 60, "B": 40}, ["A", "B"], {"A": 40, "B": 60})
        # 0.5 * (|.6-.4| + |.4-.6|) = 0.2
        assert abs(res["weight_turnover"] - 0.2) < 1e-6

    def test_empty(self):
        res = pr.compute_turnover([], None, [], None)
        assert res["name_turnover"] == 0.0

    def test_cost_estimate(self):
        # 50% turnover, 10 bps: 0.5 * 2 * 0.001 * 100 = 0.1%
        assert abs(pr.estimate_turnover_cost(0.5, 10) - 0.1) < 1e-9


class TestCovarianceRisk:
    def _make_returns(self, n_days=250, seed=0):
        rng = np.random.default_rng(seed)
        # Two correlated names + one independent
        base = rng.normal(0, 0.01, n_days)
        df = pd.DataFrame({
            "A": base + rng.normal(0, 0.003, n_days),
            "B": base + rng.normal(0, 0.003, n_days),   # correlated with A
            "C": rng.normal(0, 0.012, n_days),           # independent
        })
        return df

    def test_unavailable_when_no_returns(self):
        res = pr.compute_covariance_risk(["A"], {"A": 1.0}, None)
        assert res["available"] is False

    def test_basic_risk_report(self):
        df = self._make_returns()
        w = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
        res = pr.compute_covariance_risk(["A", "B", "C"], w, df)
        assert res["available"] is True
        assert res["portfolio_vol_annual"] > 0
        # Diversification: portfolio vol < weighted-avg single vol
        assert res["portfolio_vol_annual"] <= res["weighted_avg_single_vol"] + 1e-9
        assert res["diversification_ratio"] >= 1.0 - 1e-6

    def test_top_correlations_surface_correlated_pair(self):
        df = self._make_returns()
        res = pr.compute_covariance_risk(["A", "B", "C"], {"A": .34, "B": .33, "C": .33}, df)
        top = res["top_correlations"][0]
        # A-B should be the most correlated pair
        assert {top["a"], top["b"]} == {"A", "B"}

    def test_insufficient_history(self):
        df = self._make_returns(n_days=10)
        res = pr.compute_covariance_risk(["A", "B", "C"], {"A": 1, "B": 1, "C": 1}, df)
        assert res["available"] is False
