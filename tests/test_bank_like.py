"""Which financials are scored with the bank metric set (2026-10-09).

research/2026-10-09-bank-like-financials.md: classify on GICS sub-industry - banks, insurers and
broker-dealers (liabilities are an operating input) on the bank set; insurance brokers, asset
managers, exchanges and payment processors (fee businesses) on the generic set. Until 2026-10-09,
26 of 59 bank-set stocks got there only by the default for an unlisted Yahoo industry.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import factor_engine as fe  # noqa: E402


@pytest.fixture
def universe():
    return json.loads((ROOT / "sp500_tickers.json").read_text(encoding="utf-8"))


def test_no_constituent_reaches_the_bank_set_by_default(universe):
    fin = [r for r in universe if r.get("Sector") == "Financials"]
    if not fin or "SubIndustry" not in fin[0]:
        pytest.skip("universe file predates the GICS sub-industry")
    fe.BANK_DEFAULTED.clear()
    for r in fin:
        fe._is_bank_like(r["Ticker"], r["Sector"], "", r["SubIndustry"])
    assert fe.BANK_DEFAULTED == set(), "add these sub-industries to factor_engine's lists"


@pytest.mark.parametrize("ticker,sub,bank", [
    ("JPM", "Diversified Banks", True),
    ("PGR", "Property & Casualty Insurance", True),
    ("EG", "Reinsurance", True),
    ("GS", "Investment Banking & Brokerage", True),
    ("BRK-B", "Multi-Sector Holdings", True),
    ("AXP", "Consumer Finance", True),
    ("AON", "Insurance Brokers", False),
    ("TROW", "Asset Management & Custody Banks", False),
    ("BX", "Asset Management & Custody Banks", False),
    ("CME", "Financial Exchanges & Data", False),
    ("V", "Transaction & Payment Processing Services", False),
    # overrides inside Asset Management & Custody Banks
    ("BNY", "Asset Management & Custody Banks", True),
    ("STT", "Asset Management & Custody Banks", True),
    ("KKR", "Asset Management & Custody Banks", True),
    ("AMP", "Asset Management & Custody Banks", True),
])
def test_the_gics_rule(ticker, sub, bank):
    assert fe._is_bank_like(ticker, "Financials", "", sub) is bank


def test_a_non_financial_is_never_bank_like():
    assert fe._is_bank_like("AAPL", "Information Technology", "Consumer Electronics", "Technology Hardware") is False


def test_every_override_has_a_reason():
    assert all(isinstance(v, str) and len(v) > 10 for v in fe._BANK_OVERRIDE_TICKERS.values())


def test_the_yahoo_fallback_without_a_sub_industry():
    # EG's Yahoo industry missed the old list's spelling
    assert fe._is_bank_like("EG", "Financial Services", "Insurance - Reinsurance") is True
    assert fe._is_bank_like("AON", "Financial Services", "Insurance Brokers") is False
    assert fe._is_bank_like("TROW", "Financial Services", "Asset Management") is False
    assert fe._is_bank_like("PFG", "Financial Services", "Asset Management") is True      # an insurer Yahoo files there
    assert fe._is_bank_like("JPM", "Financial Services", "Banks—Diversified") is True
    assert fe._is_bank_like("V", "Financial Services", "Credit Services") is False


def test_scoring_reads_the_gics_sector_not_yahoos():
    """XYZ is GICS Financials but Yahoo 'Technology': the rule must see the GICS sector."""
    src = (ROOT / "factor_engine.py").read_text(encoding="utf-8")
    assert '_is_bank_like(ticker, d.get("_gics_sector") or _sector, _industry, d.get("_gics_sub"))' in src
    run = (ROOT / "run_screener.py").read_text(encoding="utf-8")
    assert '_r["_gics_sub"] = _m["SubIndustry"]' in run
    assert run.index('_r["_gics_sub"]') < run.index("df = compute_metrics(raw")
