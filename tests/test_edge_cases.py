"""Edge-case tests for scoring pipeline, portfolio construction,
revisions auto-disable math, and empty DataFrame propagation.
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from factor_engine import (
    winsorize_metrics,
    compute_sector_percentiles,
    compute_category_scores,
    compute_composite,
    apply_value_trap_flags,
    rank_stocks,
    apply_universe_filters,
    METRIC_COLS,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def cfg():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


# =====================================================================
# REVISIONS AUTO-DISABLE WEIGHT REDISTRIBUTION
# =====================================================================

class TestRevisionsAutoDisable:
    def test_weight_redistribution_sums_to_100(self, cfg):
        """After auto-disabling revisions, remaining weights must still sum to ~100."""
        cfg = copy.deepcopy(cfg)
        old_w = cfg["factor_weights"]["revisions"]
        cfg["factor_weights"]["revisions"] = 0
        others = [k for k in cfg["factor_weights"] if k != "revisions"]
        s = sum(cfg["factor_weights"][k] for k in others)
        if s > 0:
            for k in others:
                cfg["factor_weights"][k] += old_w * cfg["factor_weights"][k] / s
            for k in cfg["factor_weights"]:
                cfg["factor_weights"][k] = round(cfg["factor_weights"][k], 2)
        total = sum(cfg["factor_weights"].values())
        assert abs(total - 100) < 0.5, f"Weights sum to {total} after redistribution"

    def test_proportional_redistribution(self, cfg):
        """Weight redistribution should be proportional to existing weights."""
        cfg = copy.deepcopy(cfg)
        orig = copy.deepcopy(cfg["factor_weights"])
        old_w = orig["revisions"]
        # After redistribution, ratio between valuation and quality should be preserved
        cfg["factor_weights"]["revisions"] = 0
        others = [k for k in cfg["factor_weights"] if k != "revisions"]
        s = sum(cfg["factor_weights"][k] for k in others)
        for k in others:
            cfg["factor_weights"][k] += old_w * cfg["factor_weights"][k] / s
        # valuation/quality ratio should be the same as original
        orig_ratio = orig["valuation"] / orig["quality"]
        new_ratio = cfg["factor_weights"]["valuation"] / cfg["factor_weights"]["quality"]
        assert abs(orig_ratio - new_ratio) < 0.01

    def test_zero_weight_category_gets_nothing(self, cfg):
        """A category with 0 weight should still be 0 after redistribution."""
        cfg = copy.deepcopy(cfg)
        cfg["factor_weights"]["risk"] = 0
        cfg["factor_weights"]["revisions"] = 10
        # Now auto-disable revisions
        old_w = cfg["factor_weights"]["revisions"]
        cfg["factor_weights"]["revisions"] = 0
        others = [k for k in cfg["factor_weights"] if k != "revisions"]
        s = sum(cfg["factor_weights"][k] for k in others)
        if s > 0:
            for k in others:
                cfg["factor_weights"][k] += old_w * cfg["factor_weights"][k] / s
        assert cfg["factor_weights"]["risk"] == 0


# =====================================================================
# PORTFOLIO EDGE CASES
# =====================================================================

class TestPortfolioEdgeCases:
    def test_fewer_than_20_stocks(self, cfg):
        """Portfolio with fewer stocks than min-20 guardrail.

        The guardrail only backfills from the filtered universe, so
        the portfolio size is bounded by post-filter count, not the
        raw universe size.
        """
        from portfolio_constructor import construct_portfolio
        # Only 15 stocks — all with reasonable composites
        df = pd.DataFrame({
            "Ticker": [f"T{i}" for i in range(15)],
            "Company": [f"Co{i}" for i in range(15)],
            "Sector": ["Tech"] * 5 + ["Health"] * 5 + ["Energy"] * 5,
            "Composite": list(range(85, 100)),
            "Rank": list(range(1, 16)),
            "Value_Trap_Flag": [False] * 15,
            "volatility": [0.25] * 15,
        })
        port = construct_portfolio(df, cfg)
        # Portfolio should be non-empty and not exceed universe size
        assert 0 < len(port) <= 15

    def test_all_one_sector(self, cfg):
        """All stocks in one sector — sector cap should limit portfolio."""
        from portfolio_constructor import construct_portfolio
        max_sec = cfg.get("portfolio", {}).get("max_sector_concentration", 8)
        df = pd.DataFrame({
            "Ticker": [f"T{i}" for i in range(50)],
            "Company": [f"Co{i}" for i in range(50)],
            "Sector": ["Tech"] * 50,
            "Composite": list(range(50, 100)),
            "Rank": list(range(1, 51)),
            "Value_Trap_Flag": [False] * 50,
            "volatility": [0.25] * 50,
        })
        port = construct_portfolio(df, cfg)
        assert len(port) <= max_sec

    def test_risk_parity_with_nan_vol(self, cfg):
        """NaN volatility should fall back to median, not crash."""
        from portfolio_constructor import construct_portfolio
        cfg = copy.deepcopy(cfg)
        cfg["portfolio"]["weighting"] = "risk_parity"
        df = pd.DataFrame({
            "Ticker": [f"T{i}" for i in range(25)],
            "Company": [f"Co{i}" for i in range(25)],
            "Sector": [["Tech", "Health", "Energy", "Finance", "Utility"][i % 5] for i in range(25)],
            "Composite": list(range(76, 101)),
            "Rank": list(range(1, 26)),
            "Value_Trap_Flag": [False] * 25,
            "volatility": [np.nan if i < 5 else 0.25 for i in range(25)],
        })
        port = construct_portfolio(df, cfg)
        assert port["InvVol_Weight_Pct"].notna().all()
        assert abs(port["InvVol_Weight_Pct"].sum() - 100) < 0.5

    def test_inv_vol_with_near_zero_vol(self, cfg):
        """Near-zero volatility should be floored, not cause infinite weights."""
        from portfolio_constructor import construct_portfolio
        cfg = copy.deepcopy(cfg)
        cfg["portfolio"]["weighting"] = "inverse_vol"
        # Use 40 stocks to ensure enough survive the median filter for
        # position caps to be meaningful (need n >= 100/cap = 20).
        df = pd.DataFrame({
            "Ticker": [f"T{i}" for i in range(40)],
            "Company": [f"Co{i}" for i in range(40)],
            "Sector": [["Tech", "Health", "Energy", "Finance", "Utility"][i % 5] for i in range(40)],
            "Composite": list(range(61, 101)),
            "Rank": list(range(1, 41)),
            "Value_Trap_Flag": [False] * 40,
            "volatility": [0.001 if i == 0 else 0.25 for i in range(40)],
        })
        port = construct_portfolio(df, cfg)
        # With enough stocks, position cap should be respected
        max_pos = cfg["portfolio"]["max_position_pct"]
        assert port["InvVol_Weight_Pct"].max() <= max_pos + 0.1


# =====================================================================
# EMPTY DATAFRAME PROPAGATION
# =====================================================================

class TestEmptyPropagation:
    def test_empty_through_full_pipeline(self, cfg):
        """Empty DataFrame should propagate through scoring without errors."""
        df = pd.DataFrame(columns=["Ticker", "Sector"] + list(METRIC_COLS))
        df = winsorize_metrics(df, 0.01, 0.01)
        df = compute_sector_percentiles(df)
        df = compute_category_scores(df, cfg)
        df = compute_composite(df, cfg)
        df = apply_value_trap_flags(df, cfg)
        assert len(df) == 0
        assert "Composite" in df.columns
        assert "Value_Trap_Flag" in df.columns


# =====================================================================
# ALL-NaN SECTOR FOR A SPECIFIC METRIC
# =====================================================================

class TestAllNaNSector:
    def test_all_nan_metric_in_sector_stays_nan(self, cfg):
        """If all values for a metric in a sector are NaN, universe-wide
        ranking also produces NaN (no data to rank). Category scoring
        then skips the metric via per-row weight redistribution."""
        n = 12
        df = pd.DataFrame({
            "Ticker": [f"T{i}" for i in range(n)],
            "Sector": ["Tech"] * n,
            "roic": [np.nan] * n,
        })
        for col in METRIC_COLS:
            if col not in df.columns:
                if col == "return_12_1":
                    df[col] = [0.1 * i for i in range(n)]
                else:
                    df[col] = np.nan
        df = compute_sector_percentiles(df)
        # All-NaN metric → universe-wide ranking also NaN (na_option="keep")
        assert df["roic_pct"].isna().all()


# =====================================================================
# UNIVERSE FILTERS
# =====================================================================

class TestUniverseFilters:
    def test_filters_below_min_market_cap(self):
        cfg = {"universe": {"min_market_cap": 2e9}}
        df = pd.DataFrame({
            "Ticker": ["BIG", "SMALL", "TINY"],
            "marketCap": [10e9, 1e9, 500e6],
        })
        result = apply_universe_filters(df, cfg)
        assert list(result["Ticker"]) == ["BIG"]

    def test_nan_market_cap_passes_through(self):
        """Stocks with NaN marketCap should NOT be filtered out."""
        cfg = {"universe": {"min_market_cap": 2e9}}
        df = pd.DataFrame({
            "Ticker": ["BIG", "UNKNOWN"],
            "marketCap": [10e9, np.nan],
        })
        result = apply_universe_filters(df, cfg)
        assert len(result) == 2

    def test_no_filter_when_zero(self):
        """min_market_cap=0 should not filter anything."""
        cfg = {"universe": {"min_market_cap": 0}}
        df = pd.DataFrame({
            "Ticker": ["A", "B"],
            "marketCap": [100, 200],
        })
        result = apply_universe_filters(df, cfg)
        assert len(result) == 2


# =====================================================================
# CONFIG WITH ZERO TOTAL FACTOR WEIGHTS
# =====================================================================

class TestZeroWeights:
    def test_all_zero_weights_composite_is_nan(self):
        """All factor weights = 0 → composite should be NaN."""
        cfg_zero = {
            "factor_weights": {
                "valuation": 0, "quality": 0, "growth": 0,
                "momentum": 0, "risk": 0, "revisions": 0,
                "size": 0, "investment": 0,
            },
            "metric_weights": {
                "valuation": {}, "quality": {}, "growth": {},
                "momentum": {}, "risk": {}, "revisions": {},
                "size": {}, "investment": {},
            },
        }
        df = pd.DataFrame({
            "Ticker": ["A", "B"],
            "Sector": ["Tech", "Tech"],
            "valuation_score": [70.0, 60.0],
            "quality_score": [65.0, 55.0],
            "growth_score": [80.0, 70.0],
            "momentum_score": [60.0, 50.0],
            "risk_score": [55.0, 45.0],
            "revisions_score": [50.0, 40.0],
        })
        df = compute_composite(df, cfg_zero)
        assert df["Composite"].isna().all()


# =====================================================================
# AUTO-REDUCE HIGH-NaN METRICS
# =====================================================================

class TestAutoReduceHighNaN:
    def test_triggers_for_high_nan_metric(self):
        """Metric with >70% NaN should have its weight zeroed."""
        from run_screener import _auto_reduce_high_nan_metrics
        cfg = {
            "data_quality": {"auto_reduce_nan_threshold_pct": 70},
            "metric_weights": {
                "valuation": {"ev_ebitda": 25, "fcf_yield": 40, "earnings_yield": 20, "ev_sales": 15},
            },
        }
        # 80% NaN for ev_ebitda
        n = 100
        df = pd.DataFrame({
            "Ticker": [f"T{i}" for i in range(n)],
            "ev_ebitda": [np.nan] * 80 + [10.0] * 20,
            "fcf_yield": [0.05] * n,
            "earnings_yield": [0.07] * n,
            "ev_sales": [2.0] * n,
        })
        _auto_reduce_high_nan_metrics(df, cfg)
        assert cfg["metric_weights"]["valuation"]["ev_ebitda"] == 0

    def test_weights_still_sum_after_reduce(self):
        """After auto-reduce, remaining weights in the category should sum to ~100."""
        from run_screener import _auto_reduce_high_nan_metrics
        cfg = {
            "data_quality": {"auto_reduce_nan_threshold_pct": 70},
            "metric_weights": {
                "valuation": {"ev_ebitda": 25, "fcf_yield": 40, "earnings_yield": 20, "ev_sales": 15},
            },
        }
        n = 100
        df = pd.DataFrame({
            "Ticker": [f"T{i}" for i in range(n)],
            "ev_ebitda": [np.nan] * 80 + [10.0] * 20,
            "fcf_yield": [0.05] * n,
            "earnings_yield": [0.07] * n,
            "ev_sales": [2.0] * n,
        })
        _auto_reduce_high_nan_metrics(df, cfg)
        total = sum(cfg["metric_weights"]["valuation"].values())
        assert abs(total - 100) < 0.5

    def test_no_reduce_below_threshold(self):
        """Metric with <70% NaN should keep its weight."""
        from run_screener import _auto_reduce_high_nan_metrics
        cfg = {
            "data_quality": {"auto_reduce_nan_threshold_pct": 70},
            "metric_weights": {
                "valuation": {"ev_ebitda": 25, "fcf_yield": 40, "earnings_yield": 20, "ev_sales": 15},
            },
        }
        n = 100
        df = pd.DataFrame({
            "Ticker": [f"T{i}" for i in range(n)],
            "ev_ebitda": [np.nan] * 60 + [10.0] * 40,  # 60% NaN, below threshold
            "fcf_yield": [0.05] * n,
            "earnings_yield": [0.07] * n,
            "ev_sales": [2.0] * n,
        })
        _auto_reduce_high_nan_metrics(df, cfg)
        assert cfg["metric_weights"]["valuation"]["ev_ebitda"] == 25
