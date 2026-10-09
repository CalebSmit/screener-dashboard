"""Point-in-time fundamentals (``pit_fundamentals.py``; plan/backtest-v2.md step 3).

Synthetic facts: a value filed after a date must be invisible on it, a restatement filed before
it must win, comparatives in later filings must not change what was known, and a tag switch must
not lose the series. And v1 must not import the module.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pit_fundamentals as pf  # noqa: E402


def _facts():
    rows = [
        # FY2024 net income, filed 2025-02-10; restated in a 10-K/A filed 2025-06-01
        ("AAA", "NetIncomeLoss", "2024-01-01", "2024-12-31", "2025-02-10", "10-K", 100.0),
        ("AAA", "NetIncomeLoss", "2024-01-01", "2024-12-31", "2025-06-01", "10-K/A", 90.0),
        # Q1 2025, filed 2025-05-05; the same quarter reappears as a comparative in 2026
        ("AAA", "NetIncomeLoss", "2025-01-01", "2025-03-31", "2025-05-05", "10-Q", 30.0),
        ("AAA", "NetIncomeLoss", "2025-01-01", "2025-03-31", "2026-05-04", "10-Q", 30.0),
        # assets: instants
        ("AAA", "Assets", None, "2024-12-31", "2025-02-10", "10-K", 1000.0),
        ("AAA", "Assets", None, "2025-03-31", "2025-05-05", "10-Q", 1100.0),
        # revenue switches tag in 2025 (ASC 606 style)
        ("AAA", "Revenues", "2024-01-01", "2024-12-31", "2025-02-10", "10-K", 500.0),
        ("AAA", "RevenueFromContractWithCustomerExcludingAssessedTax", "2025-01-01", "2025-12-31", "2026-02-09", "10-K", 600.0),
    ]
    return pd.DataFrame(rows, columns=["ticker", "concept", "start", "end", "filed", "form", "val"])


def test_nothing_filed_after_the_date_is_visible():
    p = pf.PointInTime(_facts())
    assert p.annual("AAA", "net_income", "2025-02-09") is None          # the 10-K came the next day
    assert p.annual("AAA", "net_income", "2025-02-10").value == 100.0


def test_a_restatement_counts_from_the_day_it_was_filed():
    p = pf.PointInTime(_facts())
    assert p.annual("AAA", "net_income", "2025-05-31").value == 100.0
    assert p.annual("AAA", "net_income", "2025-06-01").value == 90.0


def test_quarters_and_instants_are_separate_from_years():
    p = pf.PointInTime(_facts())
    q = p.quarter("AAA", "net_income", "2025-06-30")
    assert q.value == 30.0 and q.period_end == pd.Timestamp("2025-03-31") and q.filed == pd.Timestamp("2025-05-05")
    assert p.instant("AAA", "total_assets", "2025-04-30").value == 1000.0
    assert p.instant("AAA", "total_assets", "2025-05-05").value == 1100.0


def test_a_tag_switch_keeps_the_series():
    p = pf.PointInTime(_facts())
    assert p.annual("AAA", "revenue", "2025-12-31").value == 500.0
    assert p.annual("AAA", "revenue", "2026-03-01").value == 600.0


def test_unknown_ticker_is_none():
    assert pf.PointInTime(_facts()).annual("ZZZ", "revenue", "2026-01-01") is None


def test_v1_and_production_do_not_import_it():
    """A half-fixed backtest is what plan/backtest-v2.md forbids."""
    for name in ("backtest.py", "run_screener.py", "factor_engine.py", "generate_dashboard.py"):
        src = (ROOT / name).read_text(encoding="utf-8")
        assert not re.search(r"^\s*(import|from)\s+pit_fundamentals\b", src, re.M), name


def test_the_input_map_matches_the_census():
    """The census that builds the cache and this module must ask for the same concepts."""
    src = (ROOT / "research" / "measurements" / "2026-10-09-xbrl-point-in-time-census.py").read_text(encoding="utf-8")
    for concepts in pf.INPUTS.values():
        for c in concepts:
            assert f'"{c}"' in src, c
