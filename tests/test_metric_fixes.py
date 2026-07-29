#!/usr/bin/env python3
"""Tests for Phase 13 Track B metric-definition correctness fixes.

F5  forward_eps_growth: NaN when GAAP-vs-normalized EPS ratio is extreme
F19 earnings_yield: NaN (no GAAP-EPS fallback) when LTM NI / MC unavailable
F22 operating_leverage: NaN across an EBIT sign flip; annual-vs-annual endpoints
F36 EBITDA reconstruction: use abs(D&A)
"""
import numpy as np
import pandas as pd
import pytest

from factor_engine import compute_metrics

MR = pd.Series(dtype=float)


def _run(one):
    return compute_metrics([one], MR, cfg={})


class TestForwardEpsGrowthBasis:
    def test_extreme_ratio_nans_metric(self):
        # forward >> trailing (ratio 5x) -> contaminated -> NaN
        row = {"Ticker": "X", "sector": "Technology", "marketCap": 1e10,
               "forwardEps": 5.0, "trailingEps": 1.0}
        df = _run(row)
        assert pd.isna(df["forward_eps_growth"].iloc[0])

    def test_normal_ratio_computes(self):
        # forward modestly above trailing (ratio 1.1x) -> computed
        row = {"Ticker": "Y", "sector": "Technology", "marketCap": 1e10,
               "forwardEps": 2.2, "trailingEps": 2.0}
        df = _run(row)
        assert pd.notna(df["forward_eps_growth"].iloc[0])
        assert df["forward_eps_growth"].iloc[0] > 0


class TestEarningsYieldNoFallback:
    def test_nan_when_ni_missing(self):
        # No LTM net income / marketCap present -> NaN (no trailingEps fallback)
        row = {"Ticker": "Z", "sector": "Technology",
               "trailingEps": 5.0, "currentPrice": 100.0}
        df = _run(row)
        assert pd.isna(df["earnings_yield"].iloc[0])


class TestOperatingLeverageSignFlip:
    def _base(self, ebit_annual, ebit_prior, rev_a=1.2e9, rev_ap=1.0e9):
        return {
            "Ticker": "OL", "sector": "Technology", "marketCap": 5e9,
            "ebit_annual": ebit_annual, "ebit_prior": ebit_prior,
            "ebit": ebit_annual,
            "totalRevenue_annual": rev_a, "totalRevenue_annual_prior": rev_ap,
            "totalRevenue": rev_a, "totalRevenue_prior": rev_ap,
        }

    def test_sign_flip_nans(self):
        # EBIT -50 -> +100 : sign flip -> DOL undefined -> NaN
        df = _run(self._base(ebit_annual=100e6, ebit_prior=-50e6))
        assert pd.isna(df["operating_leverage"].iloc[0])

    def test_same_sign_computes(self):
        # EBIT 100 -> 130 with revenue 1.0 -> 1.2 : DOL computed
        df = _run(self._base(ebit_annual=130e6, ebit_prior=100e6))
        assert pd.notna(df["operating_leverage"].iloc[0])


class TestEbitdaAbsDA:
    def test_negative_da_still_reconstructs(self):
        # D&A stored negative; abs() must be used so EBITDA = EBIT + |D&A|.
        row = {"Ticker": "EB", "sector": "Technology",
               "marketCap": 1e10, "enterpriseValue": 1.2e10,
               "ebit": 1e9, "da_cf": -2e8}  # |D&A| = 0.2B -> EBITDA = 1.2B
        df = _run(row)
        ev_ebitda = df["ev_ebitda"].iloc[0]
        # EV 12B / EBITDA 1.2B = 10.0 (would be 12B/1B=12 if D&A dropped)
        assert pd.notna(ev_ebitda)
        assert abs(ev_ebitda - 10.0) < 0.5
