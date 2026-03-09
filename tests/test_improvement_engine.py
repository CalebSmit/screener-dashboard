#!/usr/bin/env python3
"""Tests for the self-improving screener engine."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Patch paths before importing so tests use temp directories
import improvement_engine as ie


@pytest.fixture(autouse=True)
def temp_improvement_dirs(tmp_path, monkeypatch):
    """Redirect all improvement_engine paths to tmp_path."""
    improvement_dir = tmp_path / "improvement"
    snapshots_dir = improvement_dir / "snapshots"
    price_cache_dir = improvement_dir / "price_cache"
    proposals_dir = improvement_dir / "proposals"
    for d in (improvement_dir, snapshots_dir, price_cache_dir, proposals_dir):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(ie, "IMPROVEMENT_DIR", improvement_dir)
    monkeypatch.setattr(ie, "SNAPSHOTS_DIR", snapshots_dir)
    monkeypatch.setattr(ie, "PRICE_CACHE_DIR", price_cache_dir)
    monkeypatch.setattr(ie, "PROPOSALS_DIR", proposals_dir)
    monkeypatch.setattr(ie, "PERFORMANCE_HISTORY_PATH", improvement_dir / "performance_history.csv")
    monkeypatch.setattr(ie, "LIVE_IC_HISTORY_PATH", improvement_dir / "live_ic_history.csv")
    monkeypatch.setattr(ie, "DISPERSION_HISTORY_PATH", improvement_dir / "dispersion_history.csv")
    monkeypatch.setattr(ie, "CHANGE_LOG_PATH", improvement_dir / "change_log.csv")

    # Also monkeypatch ROOT and CONFIG_PATH for apply_changes tests
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(ie, "ROOT", tmp_path)
    monkeypatch.setattr(ie, "CONFIG_PATH", config_path)

    return tmp_path


@pytest.fixture
def sample_scored_df():
    """Create a sample scored DataFrame for testing."""
    np.random.seed(42)
    n = 100
    sectors = ["Technology"] * 30 + ["Financials"] * 20 + ["Health Care"] * 25 + ["Energy"] * 25
    df = pd.DataFrame({
        "Ticker": [f"T{i:03d}" for i in range(n)],
        "Sector": sectors,
        "Composite": np.random.uniform(20, 100, n),
        "Rank": np.arange(1, n + 1),
        "valuation_score": np.random.uniform(10, 90, n),
        "quality_score": np.random.uniform(10, 90, n),
        "growth_score": np.random.uniform(10, 90, n),
        "momentum_score": np.random.uniform(10, 90, n),
        "risk_score": np.random.uniform(10, 90, n),
        "revisions_score": np.random.uniform(10, 90, n),
        "size_score": np.random.uniform(10, 90, n),
        "investment_score": np.random.uniform(10, 90, n),
        "price_latest": np.random.uniform(20, 500, n),
    })
    return df


@pytest.fixture
def sample_portfolio_df():
    """Create a sample portfolio DataFrame."""
    return pd.DataFrame({
        "Ticker": [f"T{i:03d}" for i in range(25)],
        "Score_Weight_Pct": [4.0] * 25,
    })


@pytest.fixture
def sample_config(tmp_path):
    """Create a minimal config.yaml for testing."""
    import yaml
    cfg = {
        "factor_weights": {
            "valuation": 22, "quality": 22, "growth": 13,
            "momentum": 13, "risk": 10, "revisions": 10,
            "size": 5, "investment": 5,
        },
        "metric_weights": {
            "valuation": {"ev_ebitda": 25, "fcf_yield": 45, "earnings_yield": 20, "ev_sales": 10, "pb_ratio": 0},
            "quality": {"roic": 27, "gross_profit_assets": 20, "net_debt_to_ebitda": 18, "piotroski_f_score": 15, "accruals": 5, "operating_leverage": 8, "beneish_m_score": 7, "roe": 0, "roa": 0, "equity_ratio": 0},
        },
        "improvement": {
            "enabled": True,
            "min_observations_for_proposal": 8,
            "ewm_halflife_months": 6,
            "shrinkage": 0.5,
            "max_change_per_cycle": 3.0,
            "regime_scale_factor": 0.10,
            "auto_apply_threshold": 2.0,
        },
        "universe": {"index": "SP500", "min_market_cap": 2e9, "exclude_sectors": [], "exclude_tickers": []},
        "portfolio": {"num_stocks": 25, "weighting": "equal", "max_position_pct": 5.0, "max_sector_concentration": 8, "min_avg_dollar_volume": 10e6},
        "data_quality": {"winsorize_percentiles": [1, 99], "min_data_coverage_pct": 60, "metric_alert_threshold_pct": 50, "auto_reduce_nan_threshold_pct": 70, "stmt_val_strict": False},
        "caching": {"price_data_refresh_days": 1, "fundamental_data_refresh_days": 7, "estimate_data_refresh_days": 7, "cache_format": "parquet"},
        "output": {},
        "backtesting": {},
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
    return config_path


# =========================================================================
# Phase A: Snapshot Recording
# =========================================================================

class TestRecordSnapshot:
    def test_creates_parquet(self, sample_scored_df, sample_portfolio_df, tmp_path):
        path = ie.record_run_snapshot(
            "test123", "2026-03-01", sample_scored_df, sample_portfolio_df, {}
        )
        assert path.exists()
        snap = pd.read_parquet(path)
        assert "Ticker" in snap.columns
        assert "in_portfolio" in snap.columns
        assert "run_date" in snap.columns
        assert len(snap) == 100

    def test_portfolio_membership_marked(self, sample_scored_df, sample_portfolio_df, tmp_path):
        path = ie.record_run_snapshot(
            "test123", "2026-03-01", sample_scored_df, sample_portfolio_df, {}
        )
        snap = pd.read_parquet(path)
        assert snap["in_portfolio"].sum() == 25
        assert snap[~snap["in_portfolio"]].shape[0] == 75

    def test_no_portfolio(self, sample_scored_df, tmp_path):
        path = ie.record_run_snapshot(
            "test123", "2026-03-01", sample_scored_df, None, {}
        )
        snap = pd.read_parquet(path)
        assert snap["in_portfolio"].sum() == 0

    def test_records_dispersion(self, sample_scored_df, tmp_path):
        ie.record_run_snapshot("test123", "2026-03-01", sample_scored_df, None, {})
        assert ie.DISPERSION_HISTORY_PATH.exists()
        disp = pd.read_csv(ie.DISPERSION_HISTORY_PATH)
        assert len(disp) == 1
        assert "valuation_disp" in disp.columns


# =========================================================================
# Phase B: IC Computation
# =========================================================================

class TestLiveIC:
    def _create_perf_history(self, n_dates=10):
        """Create synthetic performance history for IC testing."""
        np.random.seed(42)
        rows = []
        for i in range(n_dates):
            date = f"2026-01-{i + 1:02d}"
            for j in range(100):
                composite = np.random.uniform(20, 100)
                # Forward returns weakly correlated with scores
                fwd = composite * 0.001 + np.random.normal(0, 0.05)
                rows.append({
                    "run_date": date,
                    "ticker": f"T{j:03d}",
                    "composite_score": composite,
                    "rank": j + 1,
                    "in_portfolio": j < 25,
                    "valuation_score": np.random.uniform(10, 90),
                    "quality_score": np.random.uniform(10, 90),
                    "growth_score": np.random.uniform(10, 90),
                    "momentum_score": np.random.uniform(10, 90),
                    "risk_score": np.random.uniform(10, 90),
                    "revisions_score": np.random.uniform(10, 90),
                    "size_score": np.random.uniform(10, 90),
                    "investment_score": np.random.uniform(10, 90),
                    "fwd_return_1w": fwd * 0.25,
                    "fwd_return_1m": fwd,
                    "fwd_return_3m": fwd * 3,
                })
        df = pd.DataFrame(rows)
        df.to_csv(ie.PERFORMANCE_HISTORY_PATH, index=False)
        return df

    def test_compute_live_ic_basic(self):
        self._create_perf_history(10)
        result = ie.compute_live_ic(horizon="1m")
        assert result is not None
        assert "composite_ic" in result.columns
        assert len(result) == 10

    def test_compute_live_ic_saves_file(self):
        self._create_perf_history(10)
        ie.compute_live_ic(horizon="1m")
        assert ie.LIVE_IC_HISTORY_PATH.exists()

    def test_analyze_trends_insufficient_data(self):
        self._create_perf_history(3)
        ie.compute_live_ic(horizon="1m")
        result = ie.analyze_ic_trends()
        assert "_warning" in result

    def test_analyze_trends_with_data(self):
        self._create_perf_history(10)
        ie.compute_live_ic(horizon="1m")
        result = ie.analyze_ic_trends()
        assert "_n_observations" in result
        assert result["_n_observations"] >= 6
        # Should have at least some category entries
        found_category = False
        for cat in ie.CATEGORY_NAMES:
            if cat in result and isinstance(result[cat], dict):
                found_category = True
                assert "ewm_ic" in result[cat]
                assert "ic_trend" in result[cat]
        assert found_category


# =========================================================================
# Phase C: Weight Optimization
# =========================================================================

class TestWeightOptimization:
    def test_ic_weighted_basic(self):
        ics = {cat: 0.05 for cat in ie.CATEGORY_NAMES}
        weights = ie._ic_weighted_allocation(ics)
        total = sum(weights.values())
        assert abs(total - 100) < 0.5

    def test_ic_weighted_all_negative(self):
        ics = {cat: -0.05 for cat in ie.CATEGORY_NAMES}
        weights = ie._ic_weighted_allocation(ics)
        total = sum(weights.values())
        assert abs(total - 100) < 0.5
        # All should get minimum bounds
        for cat in ie.CATEGORY_NAMES:
            assert weights[cat] >= ie.CATEGORY_BOUNDS[cat][0] - 0.1

    def test_ic_weighted_single_positive(self):
        ics = {cat: -0.05 for cat in ie.CATEGORY_NAMES}
        ics["valuation"] = 0.10
        weights = ie._ic_weighted_allocation(ics)
        # Valuation should get the lion's share
        assert weights["valuation"] > weights["growth"]

    def test_constraints_bounds(self):
        current = {cat: 12.5 for cat in ie.CATEGORY_NAMES}
        proposed = {"valuation": 50, "quality": 50, **{cat: 0 for cat in ie.CATEGORY_NAMES if cat not in ("valuation", "quality")}}
        result = ie._apply_constraints(proposed, current)
        for cat in ie.CATEGORY_NAMES:
            lo, hi = ie.CATEGORY_BOUNDS[cat]
            assert result[cat] >= lo - 0.1, f"{cat} below lower bound"
            assert result[cat] <= hi + 0.1, f"{cat} above upper bound"

    def test_constraints_max_change(self):
        current = {cat: 12.5 for cat in ie.CATEGORY_NAMES}
        proposed = {cat: 25.0 for cat in ie.CATEGORY_NAMES}  # +12.5 each
        result = ie._apply_constraints(proposed, current, max_change=3.0)
        for cat in ie.CATEGORY_NAMES:
            # Change should be <= 3.0 (plus rounding tolerance from normalization)
            assert abs(result[cat] - current[cat]) <= 4.0, f"{cat} changed too much"

    def test_constraints_sum_to_100(self):
        current = {cat: 12.5 for cat in ie.CATEGORY_NAMES}
        proposed = {cat: np.random.uniform(5, 30) for cat in ie.CATEGORY_NAMES}
        result = ie._apply_constraints(proposed, current)
        total = sum(result.values())
        assert abs(total - 100) < 0.5, f"Total is {total}, expected 100"

    def test_propose_insufficient_data(self):
        result = ie.propose_weight_changes()
        assert result["status"] == "insufficient_data"

    def test_shrinkage_at_one_returns_current(self, sample_config):
        """Shrinkage=1 should return weights very close to current."""
        # Create enough IC data
        self._inject_ic_data(12)
        result = ie.propose_weight_changes(shrinkage=1.0, min_observations=6)
        if result["status"] == "proposal_ready":
            for cat in ie.CATEGORY_NAMES:
                # With shrinkage=1, proposed ≈ current (within constraint adjustments)
                diff = abs(result["proposed_weights"][cat] - result["current_weights"][cat])
                assert diff <= 1.0, f"{cat} changed by {diff} with shrinkage=1"

    def _inject_ic_data(self, n_obs):
        """Create synthetic live IC data."""
        np.random.seed(42)
        rows = []
        for i in range(n_obs):
            row = {
                "run_date": f"2026-01-{i + 1:02d}",
                "horizon": "1m",
                "n_tickers": 100,
                "composite_ic": np.random.uniform(-0.05, 0.15),
            }
            for cat in ie.CATEGORY_NAMES:
                row[f"{cat}_ic"] = np.random.uniform(-0.10, 0.15)
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(ie.LIVE_IC_HISTORY_PATH, index=False)


# =========================================================================
# Phase D: Regime Detection
# =========================================================================

class TestRegimeDetection:
    def test_compute_dispersion(self, sample_scored_df):
        result = ie.compute_dispersion(sample_scored_df)
        assert "valuation" in result
        assert result["valuation"] > 0

    def test_record_dispersion(self, tmp_path):
        disp = {"valuation": 25.0, "quality": 20.0}
        ie.record_dispersion("2026-03-01", disp)
        assert ie.DISPERSION_HISTORY_PATH.exists()

    def test_detect_regime_insufficient(self):
        result = ie.detect_regime(min_history=12)
        assert result == {}

    def test_detect_regime_with_history(self):
        np.random.seed(42)
        rows = []
        for i in range(20):
            row = {"date": f"2026-01-{i + 1:02d}"}
            for cat in ie.CATEGORY_NAMES:
                row[f"{cat}_disp"] = np.random.uniform(18, 28)
            rows.append(row)
        # Make the last observation extreme for valuation
        rows[-1]["valuation_disp"] = 35.0
        df = pd.DataFrame(rows)
        df.to_csv(ie.DISPERSION_HISTORY_PATH, index=False)

        result = ie.detect_regime(min_history=12)
        assert "valuation" in result
        assert result["valuation"] == "high"

    def test_regime_adjustments(self):
        weights = {cat: 12.5 for cat in ie.CATEGORY_NAMES}
        regimes = {"valuation": "high", "momentum": "low"}
        adjusted = ie.apply_regime_adjustments(weights, regimes, scale=0.10)
        # Valuation should increase, momentum should decrease
        assert adjusted["valuation"] > adjusted["momentum"]
        # Sum should still be ~100
        assert abs(sum(adjusted.values()) - 100) < 0.5


# =========================================================================
# Phase E: Apply Changes
# =========================================================================

class TestApplyChanges:
    def test_dry_run(self, sample_config, tmp_path):
        changes = {"valuation": 1.0, "quality": -1.0}
        result = ie.apply_changes(changes, reason="test", dry_run=True)
        assert result["dry_run"] is True
        assert result["applied"] is False

    def test_creates_backup(self, sample_config, tmp_path):
        changes = {"valuation": 1.0, "quality": -1.0}
        result = ie.apply_changes(changes, reason="test")
        assert result["applied"] is True
        assert "backup_path" in result
        assert Path(result["backup_path"]).exists()

    def test_logs_to_changelog(self, sample_config, tmp_path):
        changes = {"valuation": 1.0, "quality": -1.0}
        ie.apply_changes(changes, reason="test optimization")
        assert ie.CHANGE_LOG_PATH.exists()
        log = pd.read_csv(ie.CHANGE_LOG_PATH)
        assert len(log) == 2  # Two categories changed
        assert "test optimization" in log["reason"].values

    def test_rejects_invalid_sum(self, sample_config, tmp_path):
        # Try to add 10% to valuation without removing from elsewhere
        changes = {"valuation": 10.0}
        result = ie.apply_changes(changes, reason="test")
        # Should fail validation (sum != 100)
        assert result["applied"] is False


# =========================================================================
# Integration
# =========================================================================

class TestReportGeneration:
    def test_report_with_no_data(self):
        report = ie.generate_improvement_report()
        assert "Screener Improvement Report" in report
        assert "Data Availability" in report

    def test_report_saved_to_proposals(self):
        ie.generate_improvement_report()
        proposals = list(ie.PROPOSALS_DIR.glob("improve_*.md"))
        assert len(proposals) >= 1
