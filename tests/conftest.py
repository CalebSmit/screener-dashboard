"""Shared fixtures for Multi-Factor Screener tests."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


def pytest_addoption(parser):
    parser.addoption("--regen", action="store_true", default=False,
                     help="Regenerate golden file")

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def cfg():
    """Load the production config.yaml."""
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def mock_raw_data():
    """Load 10-ticker mock data fixture."""
    with open(FIXTURES / "mock_ticker_data.json") as f:
        return json.load(f)


@pytest.fixture
def mock_market_returns():
    """Load mock S&P 500 daily returns."""
    with open(FIXTURES / "mock_market_returns.json") as f:
        data = json.load(f)
    return pd.Series(data["returns"], index=pd.to_datetime(data["dates"]))


@pytest.fixture
def sample_universe_df():
    """A small universe DataFrame for testing."""
    return pd.DataFrame([
        {"Ticker": "AAPL", "Company": "Apple Inc.", "Sector": "Information Technology"},
        {"Ticker": "MSFT", "Company": "Microsoft Corp.", "Sector": "Information Technology"},
        {"Ticker": "JPM", "Company": "JPMorgan Chase", "Sector": "Financials"},
        {"Ticker": "JNJ", "Company": "Johnson & Johnson", "Sector": "Health Care"},
        {"Ticker": "XOM", "Company": "Exxon Mobil", "Sector": "Energy"},
        {"Ticker": "PG", "Company": "Procter & Gamble", "Sector": "Consumer Staples"},
        {"Ticker": "HD", "Company": "Home Depot", "Sector": "Consumer Discretionary"},
        {"Ticker": "NEE", "Company": "NextEra Energy", "Sector": "Utilities"},
        {"Ticker": "AMT", "Company": "American Tower", "Sector": "Real Estate"},
        {"Ticker": "LIN", "Company": "Linde PLC", "Sector": "Materials"},
    ])
