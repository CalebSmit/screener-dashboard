"""Golden-file regression test.

Runs the full scoring pipeline on a fixed 10-ticker mock dataset and
verifies the output matches a known-good snapshot. This detects any
unintended behavior changes in the scoring logic.

To regenerate the golden file after an intentional change:
    pytest tests/test_golden.py --regen

"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from factor_engine import (
    compute_metrics,
    winsorize_metrics,
    compute_sector_percentiles,
    compute_category_scores,
    compute_composite,
    apply_value_trap_flags,
    rank_stocks,
    METRIC_COLS,
)

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
GOLDEN_PATH = FIXTURES / "golden_scores.parquet"

# Columns to compare (the scoring-critical columns)
COMPARE_COLS = (
    ["Ticker", "Sector"]
    + METRIC_COLS
    + ["valuation_score", "quality_score", "growth_score",
       "momentum_score", "risk_score", "revisions_score",
       "Composite", "Rank"]
)


def _run_pipeline_on_fixture(cfg):
    """Run the full scoring pipeline on mock ticker data."""
    with open(FIXTURES / "mock_ticker_data.json") as f:
        raw = json.load(f)

    # Replace None with np.nan (JSON null → None)
    for rec in raw:
        for k, v in rec.items():
            if v is None:
                rec[k] = np.nan

    market_returns = pd.Series(dtype=float)  # No market returns → beta=NaN

    df = compute_metrics(raw, market_returns)
    df = winsorize_metrics(df, 0.01, 0.01)
    df = compute_sector_percentiles(df)
    df = compute_category_scores(df, cfg)
    df = compute_composite(df, cfg)
    df = apply_value_trap_flags(df, cfg)
    df = rank_stocks(df)
    return df


@pytest.fixture
def cfg():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def test_golden_file(cfg, request):
    """Compare pipeline output against golden file."""
    df = _run_pipeline_on_fixture(cfg)

    regen = request.config.getoption("--regen", default=False)
    if regen:
        # Regenerate golden file
        cols = [c for c in COMPARE_COLS if c in df.columns]
        df[cols].to_parquet(str(GOLDEN_PATH), index=False)
        pytest.skip(f"Golden file regenerated at {GOLDEN_PATH}")

    if not GOLDEN_PATH.exists():
        # First run: generate golden file
        cols = [c for c in COMPARE_COLS if c in df.columns]
        df[cols].to_parquet(str(GOLDEN_PATH), index=False)
        pytest.skip(f"Golden file created at {GOLDEN_PATH}. Re-run to verify.")

    expected = pd.read_parquet(str(GOLDEN_PATH))
    actual_cols = [c for c in COMPARE_COLS if c in df.columns and c in expected.columns]

    actual = df[actual_cols].reset_index(drop=True)
    expected = expected[actual_cols].reset_index(drop=True)

    # Sort both by Ticker for stable comparison
    actual = actual.sort_values("Ticker").reset_index(drop=True)
    expected = expected.sort_values("Ticker").reset_index(drop=True)

    pd.testing.assert_frame_equal(
        actual, expected,
        check_dtype=False,
        atol=0.01,  # Allow small floating-point differences
    )


def test_fixture_completeness(cfg):
    """Verify the mock data fixture exercises key paths."""
    df = _run_pipeline_on_fixture(cfg)

    # Should have all 10 tickers (or most — some may be filtered for coverage)
    assert len(df) >= 7, f"Expected >= 7 scored tickers, got {len(df)}"

    # Should have multiple sectors
    assert df["Sector"].nunique() >= 3

    # Key metrics should not be all NaN
    for col in ["ev_ebitda", "roic", "return_12_1", "Composite"]:
        if col in df.columns:
            assert df[col].notna().any(), f"{col} is all NaN"

    # Ranks should be assigned
    assert df["Rank"].notna().all()
    assert (df["Rank"] >= 1).all()
