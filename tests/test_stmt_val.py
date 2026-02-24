"""Tests for _stmt_val() — the fuzzy label-matching function that extracts
every fundamental metric from yfinance financial statements.

Verifies exact match, startswith/endswith substring fallback, col parameter,
and default handling.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from factor_engine import _stmt_val, _stmt_val_ltm, _find_stmt_label


def _make_stmt(labels, values_per_label=2):
    """Build a fake yfinance financial statement DataFrame.

    Columns represent periods (most recent first); rows are line items.
    """
    data = {}
    for i, lbl in enumerate(labels):
        data[lbl] = [(i + 1) * 100 + j for j in range(values_per_label)]
    df = pd.DataFrame(data).T
    df.columns = [f"2025-{c+1:02d}-01" for c in range(values_per_label)]
    return df


class TestExactMatch:
    def test_case_insensitive(self):
        stmt = _make_stmt(["Total Revenue", "Cost Of Revenue"])
        assert _stmt_val(stmt, "total revenue") == 100.0
        assert _stmt_val(stmt, "TOTAL REVENUE") == 100.0
        assert _stmt_val(stmt, "Total Revenue") == 100.0

    def test_exact_match_preferred_over_substring(self):
        """If both exact and substring match exist, exact should win."""
        stmt = _make_stmt(["Net Income", "Net Income From Continuing Operations"])
        val = _stmt_val(stmt, "Net Income")
        assert val == 100.0  # First label (index 0) → value 100

    def test_whitespace_stripped(self):
        stmt = _make_stmt(["  Total Revenue  "])
        assert _stmt_val(stmt, "Total Revenue") == 100.0


class TestSubstringFallback:
    def test_startswith(self):
        """Label starts with target → should match."""
        stmt = _make_stmt(["Operating Income Loss"])
        assert _stmt_val(stmt, "Operating Income") == 100.0

    def test_endswith(self):
        """Label ends with target → should match."""
        stmt = _make_stmt(["TTM Gross Profit"])
        assert _stmt_val(stmt, "Gross Profit") == 100.0

    def test_middle_no_match(self):
        """Target appearing in the middle should NOT match."""
        stmt = _make_stmt(["Net Income From Continuing Operations"])
        # "Operating" is not at start or end of the label
        result = _stmt_val(stmt, "Operating", default=-999)
        assert result == -999

    def test_operating_income_does_not_match_net_income(self):
        """'Operating Income' should NOT match 'Net Income From Continuing Operations'."""
        stmt = _make_stmt(["Net Income From Continuing Operations"])
        result = _stmt_val(stmt, "Operating Income", default=-999)
        assert result == -999


class TestColParameter:
    def test_col_0_returns_most_recent(self):
        stmt = _make_stmt(["Total Revenue"], values_per_label=3)
        val = _stmt_val(stmt, "Total Revenue", col=0)
        assert val == 100.0

    def test_col_1_returns_prior_year(self):
        stmt = _make_stmt(["Total Revenue"], values_per_label=3)
        val = _stmt_val(stmt, "Total Revenue", col=1)
        assert val == 101.0

    def test_col_out_of_range_returns_default(self):
        stmt = _make_stmt(["Total Revenue"], values_per_label=1)
        result = _stmt_val(stmt, "Total Revenue", col=5, default=-999)
        assert result == -999


class TestDefaultHandling:
    def test_none_statement(self):
        assert np.isnan(_stmt_val(None, "Total Revenue"))

    def test_empty_statement(self):
        assert np.isnan(_stmt_val(pd.DataFrame(), "Total Revenue"))

    def test_non_matching_label(self):
        stmt = _make_stmt(["Total Revenue"])
        result = _stmt_val(stmt, "Nonexistent Label", default=-999)
        assert result == -999

    def test_default_is_nan(self):
        stmt = _make_stmt(["Total Revenue"])
        result = _stmt_val(stmt, "Nonexistent Label")
        assert np.isnan(result)


class TestNaNHandling:
    def test_nan_values_skipped(self):
        """If the matched row has NaN in the target column, fall through."""
        stmt = _make_stmt(["Total Revenue"], values_per_label=2)
        # Set col 0 to NaN
        stmt.iloc[0, 0] = np.nan
        val = _stmt_val(stmt, "Total Revenue", col=0)
        # Should return the only non-NaN value (col 1's value)
        assert val == 101.0


# ---- _find_stmt_label() tests ----

class TestFindStmtLabel:
    def test_exact_match(self):
        stmt = _make_stmt(["Total Revenue", "Net Income"])
        assert _find_stmt_label(stmt, "Total Revenue") == "Total Revenue"

    def test_case_insensitive(self):
        stmt = _make_stmt(["Total Revenue"])
        assert _find_stmt_label(stmt, "total revenue") == "Total Revenue"

    def test_substring_startswith(self):
        stmt = _make_stmt(["Operating Income Loss"])
        assert _find_stmt_label(stmt, "Operating Income") == "Operating Income Loss"

    def test_substring_endswith(self):
        stmt = _make_stmt(["TTM Gross Profit"])
        assert _find_stmt_label(stmt, "Gross Profit") == "TTM Gross Profit"

    def test_no_match_returns_none(self):
        stmt = _make_stmt(["Total Revenue"])
        assert _find_stmt_label(stmt, "Nonexistent") is None


# ---- _stmt_val_ltm() tests ----

def _make_quarterly_stmt(labels, n_quarters=8, values=None):
    """Build a fake quarterly financial statement DataFrame.

    Columns represent quarters (most recent first); rows are line items.
    If values is provided, it should be a list of numbers (length >= n_quarters).
    """
    data = {}
    for i, lbl in enumerate(labels):
        if values is not None:
            data[lbl] = values[:n_quarters]
        else:
            data[lbl] = [(i + 1) * 100 + j * 10 for j in range(n_quarters)]
    df = pd.DataFrame(data).T
    dates = pd.date_range(end="2026-03-31", periods=n_quarters, freq="QE")[::-1]
    df.columns = dates[:n_quarters]
    return df


class TestStmtValLtm:
    def test_basic_ltm_sum(self):
        """Sum of 4 most recent quarters."""
        stmt = _make_quarterly_stmt(
            ["Total Revenue"], n_quarters=8,
            values=[100, 90, 95, 85, 80, 75, 70, 65])
        result = _stmt_val_ltm(stmt, "Total Revenue")
        assert result == 100 + 90 + 95 + 85  # 370

    def test_prior_year_ltm(self):
        """Sum of quarters 4-7 (prior-year LTM)."""
        stmt = _make_quarterly_stmt(
            ["Total Revenue"], n_quarters=8,
            values=[100, 90, 95, 85, 80, 75, 70, 65])
        result = _stmt_val_ltm(stmt, "Total Revenue", offset=4)
        assert result == 80 + 75 + 70 + 65  # 290

    def test_partial_3q_annualized(self):
        """3 quarters available at offset=0 → annualize as sum * 4/3."""
        stmt = _make_quarterly_stmt(
            ["Total Revenue"], n_quarters=3,
            values=[100, 90, 95])
        result = _stmt_val_ltm(stmt, "Total Revenue")
        expected = (100 + 90 + 95) * 4 / 3
        assert abs(result - expected) < 0.01

    def test_partial_2q_returns_nan(self):
        """Only 2 quarters → NaN (below 3-quarter minimum)."""
        stmt = _make_quarterly_stmt(
            ["Total Revenue"], n_quarters=2,
            values=[100, 90])
        result = _stmt_val_ltm(stmt, "Total Revenue")
        assert np.isnan(result)

    def test_none_statement(self):
        assert np.isnan(_stmt_val_ltm(None, "Total Revenue"))

    def test_empty_statement(self):
        assert np.isnan(_stmt_val_ltm(pd.DataFrame(), "Total Revenue"))

    def test_fuzzy_matching(self):
        """Fuzzy label matching works same as _stmt_val."""
        stmt = _make_quarterly_stmt(
            ["Operating Income Loss"], n_quarters=4,
            values=[100, 90, 95, 85])
        result = _stmt_val_ltm(stmt, "Operating Income")
        assert result == 100 + 90 + 95 + 85

    def test_label_not_found_returns_default(self):
        stmt = _make_quarterly_stmt(["Total Revenue"], n_quarters=4)
        result = _stmt_val_ltm(stmt, "Nonexistent", default=-999)
        assert result == -999

    def test_prior_year_insufficient_quarters(self):
        """Only 6 quarters available; offset=4 needs 8 → NaN."""
        stmt = _make_quarterly_stmt(
            ["Total Revenue"], n_quarters=6,
            values=[100, 90, 95, 85, 80, 75])
        result = _stmt_val_ltm(stmt, "Total Revenue", offset=4)
        assert np.isnan(result)

    def test_exact_4_quarters(self):
        """Exactly 4 quarters available → works for offset=0."""
        stmt = _make_quarterly_stmt(
            ["Net Income"], n_quarters=4,
            values=[50, 40, 45, 35])
        result = _stmt_val_ltm(stmt, "Net Income")
        assert result == 50 + 40 + 45 + 35

    def test_annualize_not_applied_at_offset(self):
        """Annualization only applies at offset=0, not at offset=4."""
        stmt = _make_quarterly_stmt(
            ["Total Revenue"], n_quarters=7,
            values=[100, 90, 95, 85, 80, 75, 70])
        # offset=4 needs 4 quarters (cols 4,5,6,7) but only 3 available
        result = _stmt_val_ltm(stmt, "Total Revenue", offset=4)
        assert np.isnan(result)
