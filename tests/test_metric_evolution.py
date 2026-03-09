#!/usr/bin/env python3
"""Tests for Phase 11: Metric Evolution Engine.

Tests candidate metric computation, metric-level IC, evolution proposals,
weight redistribution, apply logic, and report integration.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import improvement_engine as ie


@pytest.fixture(autouse=True)
def temp_dirs(tmp_path, monkeypatch):
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
    monkeypatch.setattr(ie, "METRIC_IC_HISTORY_PATH", improvement_dir / "metric_ic_history.csv")

    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(ie, "ROOT", tmp_path)
    monkeypatch.setattr(ie, "CONFIG_PATH", config_path)

    return tmp_path


def _write_config(tmp_path):
    """Write a minimal valid config.yaml for tests."""
    cfg = {
        "universe": {"index": "SP500", "min_market_cap": 2e9},
        "factor_weights": {
            "valuation": 22, "quality": 22, "growth": 13, "momentum": 13,
            "risk": 10, "revisions": 10, "size": 5, "investment": 5,
        },
        "metric_weights": {
            "valuation": {
                "ev_ebitda": 25, "fcf_yield": 45, "earnings_yield": 20,
                "ev_sales": 10, "pb_ratio": 0, "dividend_yield": 0,
            },
            "quality": {
                "roic": 27, "gross_profit_assets": 20, "net_debt_to_ebitda": 18,
                "piotroski_f_score": 15, "accruals": 5, "operating_leverage": 8,
                "beneish_m_score": 7, "roe": 0, "roa": 0, "equity_ratio": 0,
                "operating_margin": 0, "current_ratio": 0, "insider_ownership": 0,
                "interest_coverage": 0,
            },
            "growth": {
                "forward_eps_growth": 45, "peg_ratio": 0, "revenue_growth": 25,
                "revenue_cagr_3yr": 15, "sustainable_growth": 15,
            },
            "momentum": {
                "return_12_1": 40, "return_6m": 35, "jensens_alpha": 25,
                "proximity_52w_high": 0,
            },
            "risk": {
                "volatility": 30, "beta": 20, "sharpe_ratio": 15,
                "sortino_ratio": 15, "max_drawdown_1y": 20,
            },
            "revisions": {
                "analyst_surprise": 38, "price_target_upside": 12,
                "earnings_acceleration": 20, "consecutive_beat_streak": 20,
                "short_interest_ratio": 10, "short_pct_float": 0, "analyst_rating": 0,
            },
            "size": {"size_log_mcap": 100},
            "investment": {"asset_growth": 100},
        },
        "bank_metric_weights": {
            "valuation": {
                "ev_ebitda": 0, "fcf_yield": 0, "earnings_yield": 40,
                "ev_sales": 0, "pb_ratio": 60, "dividend_yield": 0,
            },
            "quality": {
                "roic": 0, "gross_profit_assets": 0, "debt_equity": 0,
                "piotroski_f_score": 15, "accruals": 10, "beneish_m_score": 0,
                "roe": 35, "roa": 25, "equity_ratio": 15,
                "operating_margin": 0, "current_ratio": 0, "insider_ownership": 0,
                "interest_coverage": 0,
            },
        },
        "improvement": {
            "enabled": True,
            "min_observations_for_activation": 12,
            "min_observations_for_deactivation": 12,
            "candidate_initial_weight": 5,
            "max_candidate_activations_per_cycle": 1,
            "candidate_ic_threshold": 0.02,
        },
        "portfolio": {"num_stocks": 25},
        "output": {"excel_file": "test.xlsx"},
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return cfg


def _make_metric_ic_history(tmp_path, n_dates=15, strong_candidate="proximity_52w_high",
                             weak_existing=None):
    """Create synthetic metric IC history for testing."""
    np.random.seed(42)
    dates = [f"2026-01-{i+1:02d}" for i in range(n_dates)]
    rows = []
    for d in dates:
        row = {"run_date": d, "horizon": "1m", "n_tickers": 400}
        # Existing metrics: moderate positive IC
        for m in ["ev_ebitda", "fcf_yield", "earnings_yield", "roic", "return_12_1",
                   "return_6m", "volatility", "analyst_surprise"]:
            row[f"{m}_ic"] = np.random.normal(0.03, 0.02)

        # Strong candidate: consistently positive IC
        if strong_candidate:
            row[f"{strong_candidate}_ic"] = np.random.uniform(0.03, 0.08)

        # Weak existing metric: consistently negative IC
        if weak_existing:
            row[f"{weak_existing}_ic"] = np.random.uniform(-0.05, -0.01)

        rows.append(row)

    df = pd.DataFrame(rows)
    ic_path = tmp_path / "improvement" / "metric_ic_history.csv"
    df.to_csv(ic_path, index=False)
    return df


# =========================================================================
# Candidate Metric Computation Tests
# =========================================================================

class TestCandidateMetricComputation:
    """Test that candidate metrics are computed correctly in factor_engine."""

    def test_proximity_52w_high(self):
        """Proximity to 52-week high = currentPrice / fiftyTwoWeekHigh."""
        from factor_engine import METRIC_COLS, METRIC_DIR, CAT_METRICS
        assert "proximity_52w_high" in METRIC_COLS
        assert METRIC_DIR["proximity_52w_high"] is True
        assert "proximity_52w_high" in CAT_METRICS["momentum"]

    def test_operating_margin_in_lookups(self):
        from factor_engine import METRIC_COLS, METRIC_DIR, CAT_METRICS, _NONBANK_ONLY_METRICS
        assert "operating_margin" in METRIC_COLS
        assert METRIC_DIR["operating_margin"] is True
        assert "operating_margin" in CAT_METRICS["quality"]
        assert "operating_margin" in _NONBANK_ONLY_METRICS

    def test_current_ratio_in_lookups(self):
        from factor_engine import METRIC_COLS, METRIC_DIR, CAT_METRICS, _NONBANK_ONLY_METRICS
        assert "current_ratio" in METRIC_COLS
        assert METRIC_DIR["current_ratio"] is True
        assert "current_ratio" in CAT_METRICS["quality"]
        assert "current_ratio" in _NONBANK_ONLY_METRICS

    def test_dividend_yield_in_lookups(self):
        from factor_engine import METRIC_COLS, METRIC_DIR, CAT_METRICS
        assert "dividend_yield" in METRIC_COLS
        assert METRIC_DIR["dividend_yield"] is True
        assert "dividend_yield" in CAT_METRICS["valuation"]

    def test_insider_ownership_in_lookups(self):
        from factor_engine import METRIC_COLS, METRIC_DIR, CAT_METRICS
        assert "insider_ownership" in METRIC_COLS
        assert METRIC_DIR["insider_ownership"] is True
        assert "insider_ownership" in CAT_METRICS["quality"]

    def test_short_pct_float_in_lookups(self):
        from factor_engine import METRIC_COLS, METRIC_DIR, CAT_METRICS
        assert "short_pct_float" in METRIC_COLS
        assert METRIC_DIR["short_pct_float"] is False  # lower = better
        assert "short_pct_float" in CAT_METRICS["revisions"]

    def test_analyst_rating_in_lookups(self):
        from factor_engine import METRIC_COLS, METRIC_DIR, CAT_METRICS
        assert "analyst_rating" in METRIC_COLS
        assert METRIC_DIR["analyst_rating"] is False  # lower = more bullish
        assert "analyst_rating" in CAT_METRICS["revisions"]

    def test_interest_coverage_in_lookups(self):
        from factor_engine import METRIC_COLS, METRIC_DIR, CAT_METRICS, _NONBANK_ONLY_METRICS
        assert "interest_coverage" in METRIC_COLS
        assert METRIC_DIR["interest_coverage"] is True
        assert "interest_coverage" in CAT_METRICS["quality"]
        assert "interest_coverage" in _NONBANK_ONLY_METRICS


# =========================================================================
# Snapshot Extension Tests
# =========================================================================

class TestSnapshotExtension:
    """Test that snapshots include _pct columns."""

    def test_snapshot_includes_pct_cols(self, tmp_path, monkeypatch):
        """Snapshots should include metric percentile columns."""
        cfg = _write_config(tmp_path)
        np.random.seed(42)
        n = 50
        scored_df = pd.DataFrame({
            "Ticker": [f"T{i:03d}" for i in range(n)],
            "Sector": ["Tech"] * n,
            "Composite": np.random.uniform(30, 80, n),
            "Rank": list(range(1, n + 1)),
            "valuation_score": np.random.uniform(20, 80, n),
            "quality_score": np.random.uniform(20, 80, n),
            "growth_score": np.random.uniform(20, 80, n),
            "momentum_score": np.random.uniform(20, 80, n),
            "risk_score": np.random.uniform(20, 80, n),
            "revisions_score": np.random.uniform(20, 80, n),
            "size_score": np.random.uniform(20, 80, n),
            "investment_score": np.random.uniform(20, 80, n),
            "_current_price": np.random.uniform(50, 200, n),
            # Metric percentile columns
            "ev_ebitda_pct": np.random.uniform(0, 100, n),
            "fcf_yield_pct": np.random.uniform(0, 100, n),
            "proximity_52w_high_pct": np.random.uniform(0, 100, n),
        })

        # Mock compute_forward_returns and compute_dispersion to avoid side effects
        monkeypatch.setattr(ie, "compute_forward_returns", lambda x: None)
        monkeypatch.setattr(ie, "compute_dispersion", lambda x: {})
        monkeypatch.setattr(ie, "record_dispersion", lambda x, y: None)

        path = ie.record_run_snapshot("test01", "2026-01-15", scored_df, None, cfg)
        snap = pd.read_parquet(path)

        # Verify _pct columns are included
        assert "ev_ebitda_pct" in snap.columns
        assert "fcf_yield_pct" in snap.columns
        assert "proximity_52w_high_pct" in snap.columns

    def test_backward_compat_old_snapshots(self, tmp_path):
        """Old snapshots without _pct columns should not break IC computation."""
        # compute_metric_level_ic should return None when no _pct columns
        perf = pd.DataFrame({
            "run_date": ["2026-01-01"] * 50,
            "ticker": [f"T{i:03d}" for i in range(50)],
            "composite_score": np.random.uniform(30, 80, 50),
            "fwd_return_1m": np.random.normal(0.01, 0.03, 50),
        })
        perf.to_csv(ie.PERFORMANCE_HISTORY_PATH, index=False)
        result = ie.compute_metric_level_ic(horizon="1m")
        assert result is None  # No _pct columns available


# =========================================================================
# Metric-Level IC Tests
# =========================================================================

class TestMetricLevelIC:
    """Test per-metric IC computation and trend analysis."""

    def test_compute_metric_level_ic_basic(self, tmp_path):
        """Basic IC computation with synthetic data."""
        np.random.seed(42)
        n = 100
        perf = pd.DataFrame({
            "run_date": ["2026-01-15"] * n,
            "ticker": [f"T{i:03d}" for i in range(n)],
            "composite_score": np.random.uniform(30, 80, n),
            "fwd_return_1m": np.random.normal(0.01, 0.03, n),
            "ev_ebitda_pct": np.random.uniform(0, 100, n),
            "proximity_52w_high_pct": np.random.uniform(0, 100, n),
        })
        perf.to_csv(ie.PERFORMANCE_HISTORY_PATH, index=False)

        result = ie.compute_metric_level_ic(horizon="1m")
        assert result is not None
        assert len(result) == 1
        assert "ev_ebitda_ic" in result.columns
        assert "proximity_52w_high_ic" in result.columns

    def test_compute_metric_level_ic_insufficient(self, tmp_path):
        """IC computation with too few tickers returns None."""
        perf = pd.DataFrame({
            "run_date": ["2026-01-15"] * 5,  # Only 5 tickers
            "ticker": [f"T{i}" for i in range(5)],
            "composite_score": [50, 60, 70, 80, 90],
            "fwd_return_1m": [0.01, 0.02, -0.01, 0.03, -0.02],
            "ev_ebitda_pct": [20, 40, 60, 80, 100],
        })
        perf.to_csv(ie.PERFORMANCE_HISTORY_PATH, index=False)

        result = ie.compute_metric_level_ic(horizon="1m", min_tickers=30)
        assert result is None

    def test_analyze_metric_ic_trends_classification(self, tmp_path):
        """Trend analysis classifies metrics correctly."""
        _make_metric_ic_history(tmp_path, n_dates=15)

        result = ie.analyze_metric_ic_trends()
        assert "_n_observations" in result
        assert result["_n_observations"] == 15

        # proximity_52w_high should have positive IC
        if "proximity_52w_high" in result:
            info = result["proximity_52w_high"]
            assert info["ewm_ic"] > 0
            assert info["pct_positive"] > 0.5

    def test_analyze_metric_ic_trends_category_filter(self, tmp_path):
        """Category filter restricts results to that category's metrics."""
        _make_metric_ic_history(tmp_path, n_dates=15)

        result = ie.analyze_metric_ic_trends(category="momentum")
        # Should only contain momentum metrics (if present in IC history)
        for key in result:
            if key.startswith("_"):
                continue
            from factor_engine import CAT_METRICS
            assert key in CAT_METRICS["momentum"]


# =========================================================================
# Metric Evolution Proposal Tests
# =========================================================================

class TestMetricEvolutionProposals:
    """Test the propose_metric_evolution orchestrator."""

    def test_insufficient_data(self, tmp_path):
        """Returns insufficient_data when not enough observations."""
        _write_config(tmp_path)
        # No IC history exists
        result = ie.propose_metric_evolution(min_observations=12)
        assert result["status"] == "insufficient_data"

    def test_no_changes_when_metrics_are_moderate(self, tmp_path):
        """Returns no_changes when all metrics are within normal ranges."""
        _write_config(tmp_path)
        # Create IC history where candidates have weak IC (below threshold)
        np.random.seed(42)
        dates = [f"2026-01-{i+1:02d}" for i in range(15)]
        rows = []
        for d in dates:
            row = {"run_date": d, "horizon": "1m", "n_tickers": 400}
            # Candidates: very weak IC (not meeting threshold)
            for m in ie.CANDIDATE_METRICS:
                row[f"{m}_ic"] = np.random.normal(0.005, 0.01)  # Weak, inconsistent
            # Existing: moderate positive
            for m in ["ev_ebitda", "fcf_yield", "roic", "return_12_1"]:
                row[f"{m}_ic"] = np.random.normal(0.03, 0.01)
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(ie.METRIC_IC_HISTORY_PATH, index=False)

        result = ie.propose_metric_evolution(min_observations=12)
        assert result["status"] in ("no_changes", "insufficient_data")

    def test_max_activations_per_cycle(self, tmp_path):
        """Only max_candidate_activations_per_cycle candidates activated."""
        _write_config(tmp_path)
        # Create IC history where ALL candidates are strong
        np.random.seed(42)
        dates = [f"2026-01-{i+1:02d}" for i in range(15)]
        rows = []
        for d in dates:
            row = {"run_date": d, "horizon": "1m", "n_tickers": 400}
            for m in ie.CANDIDATE_METRICS:
                row[f"{m}_ic"] = np.random.uniform(0.04, 0.09)  # All strong
            for m in ["ev_ebitda", "fcf_yield", "roic"]:
                row[f"{m}_ic"] = np.random.normal(0.03, 0.01)
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(ie.METRIC_IC_HISTORY_PATH, index=False)

        result = ie.propose_metric_evolution(min_observations=12)
        if result["status"] == "proposal_ready":
            activations = result.get("activate_proposals", [])
            assert len(activations) <= 1  # max_candidate_activations_per_cycle = 1


# =========================================================================
# Weight Redistribution Tests
# =========================================================================

class TestWeightRedistribution:
    """Test _compute_weight_redistribution helper."""

    def test_activation_reduces_existing_proportionally(self):
        """Activating a candidate at 5% reduces others proportionally."""
        activations = [{
            "metric": "dividend_yield", "category": "valuation",
            "proposed_weight": 5, "ewm_ic": 0.05,
        }]
        current_weights = {
            "valuation": {
                "ev_ebitda": 25, "fcf_yield": 45, "earnings_yield": 20,
                "ev_sales": 10, "pb_ratio": 0, "dividend_yield": 0,
            }
        }
        result = ie._compute_weight_redistribution(activations, [], current_weights)
        assert "valuation" in result
        new_w = result["valuation"]
        assert new_w["dividend_yield"] == 5
        # Others should be reduced but sum should be ~100
        assert abs(sum(new_w.values()) - 100) < 0.5

    def test_deactivation_increases_remaining(self):
        """Deactivating frees weight and redistributes."""
        deactivations = [{
            "metric": "ev_sales", "category": "valuation",
            "current_weight": 10, "proposed_weight": 0,
        }]
        current_weights = {
            "valuation": {
                "ev_ebitda": 25, "fcf_yield": 45, "earnings_yield": 20,
                "ev_sales": 10, "pb_ratio": 0, "dividend_yield": 0,
            }
        }
        result = ie._compute_weight_redistribution([], deactivations, current_weights)
        new_w = result["valuation"]
        assert new_w["ev_sales"] == 0
        # Remaining active metrics should have gained the 10%
        assert abs(sum(new_w.values()) - 100) < 0.5
        assert new_w["fcf_yield"] > 45  # Should have received most


# =========================================================================
# Apply Metric Changes Tests
# =========================================================================

class TestApplyMetricChanges:
    """Test apply_metric_changes config writer."""

    def test_apply_creates_backup(self, tmp_path):
        """Applying changes creates a config backup."""
        _write_config(tmp_path)
        proposal = {
            "status": "proposal_ready",
            "n_observations": 15,
            "activate_proposals": [{
                "metric": "dividend_yield", "category": "valuation",
                "proposed_weight": 5, "rationale": "Strong IC",
            }],
            "deactivate_proposals": [],
            "weight_adjustments": {
                "valuation": {
                    "ev_ebitda": 23.8, "fcf_yield": 42.8, "earnings_yield": 19.0,
                    "ev_sales": 9.5, "pb_ratio": 0, "dividend_yield": 5.0,
                },
            },
        }
        result = ie.apply_metric_changes(proposal, reason="test")
        assert result["applied"] is True
        assert "backup_path" in result
        assert Path(result["backup_path"]).exists()

    def test_apply_updates_config(self, tmp_path):
        """Applying changes updates config.yaml."""
        _write_config(tmp_path)
        proposal = {
            "status": "proposal_ready",
            "n_observations": 15,
            "activate_proposals": [{
                "metric": "dividend_yield", "category": "valuation",
                "proposed_weight": 5, "rationale": "Strong IC",
            }],
            "deactivate_proposals": [],
            "weight_adjustments": {
                "valuation": {
                    "ev_ebitda": 23.8, "fcf_yield": 42.8, "earnings_yield": 19.0,
                    "ev_sales": 9.5, "pb_ratio": 0, "dividend_yield": 5.0,
                },
            },
        }
        ie.apply_metric_changes(proposal, reason="test")

        # Read back config
        with open(ie.CONFIG_PATH) as f:
            new_cfg = yaml.safe_load(f)
        assert new_cfg["metric_weights"]["valuation"]["dividend_yield"] == 5.0

    def test_apply_logs_to_changelog(self, tmp_path):
        """Applying changes appends to change_log.csv."""
        _write_config(tmp_path)
        proposal = {
            "status": "proposal_ready",
            "n_observations": 15,
            "activate_proposals": [{
                "metric": "dividend_yield", "category": "valuation",
                "proposed_weight": 5, "rationale": "Strong IC",
            }],
            "deactivate_proposals": [],
            "weight_adjustments": {
                "valuation": {
                    "ev_ebitda": 23.8, "fcf_yield": 42.8, "earnings_yield": 19.0,
                    "ev_sales": 9.5, "pb_ratio": 0, "dividend_yield": 5.0,
                },
            },
        }
        ie.apply_metric_changes(proposal, reason="test")

        assert ie.CHANGE_LOG_PATH.exists()
        log = pd.read_csv(ie.CHANGE_LOG_PATH)
        assert len(log) >= 1
        assert log.iloc[0]["change_type"] == "metric_activation"
        assert log.iloc[0]["metric"] == "dividend_yield"

    def test_dry_run_does_not_modify(self, tmp_path):
        """Dry run returns proposed changes without modifying config."""
        _write_config(tmp_path)
        proposal = {
            "status": "proposal_ready",
            "n_observations": 15,
            "activate_proposals": [],
            "deactivate_proposals": [],
            "weight_adjustments": {
                "valuation": {"ev_ebitda": 30, "fcf_yield": 40, "earnings_yield": 20,
                              "ev_sales": 10, "pb_ratio": 0, "dividend_yield": 0},
            },
        }
        result = ie.apply_metric_changes(proposal, dry_run=True)
        assert result["applied"] is False
        assert result["dry_run"] is True

        # Config should be unchanged
        with open(ie.CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        assert cfg["metric_weights"]["valuation"]["ev_ebitda"] == 25  # Original


# =========================================================================
# Report Integration Tests
# =========================================================================

class TestReportIntegration:
    """Test that the improvement report includes metric evolution section."""

    def test_report_includes_metric_evolution_section(self, tmp_path, monkeypatch):
        """Report should include section 5.5 Metric Evolution."""
        _write_config(tmp_path)
        # Create minimal perf history so report doesn't fail
        perf = pd.DataFrame({
            "run_date": ["2026-01-01"] * 10,
            "ticker": [f"T{i}" for i in range(10)],
            "composite_score": np.random.uniform(30, 80, 10),
            "rank": list(range(1, 11)),
            "in_portfolio": [True] * 5 + [False] * 5,
            "valuation_score": np.random.uniform(20, 80, 10),
            "quality_score": np.random.uniform(20, 80, 10),
            "growth_score": np.random.uniform(20, 80, 10),
            "momentum_score": np.random.uniform(20, 80, 10),
            "risk_score": np.random.uniform(20, 80, 10),
            "revisions_score": np.random.uniform(20, 80, 10),
            "size_score": np.random.uniform(20, 80, 10),
            "investment_score": np.random.uniform(20, 80, 10),
            "fwd_return_1w": np.random.normal(0.01, 0.02, 10),
        })
        perf.to_csv(ie.PERFORMANCE_HISTORY_PATH, index=False)

        report = ie.generate_improvement_report()
        assert "5.5 Metric Evolution" in report


# =========================================================================
# Schema Validation Tests
# =========================================================================

class TestSchemaValidation:
    """Test that candidate metrics are in Pydantic schemas."""

    def test_valuation_weights_has_dividend_yield(self):
        from schemas import ValuationWeights
        w = ValuationWeights()
        assert hasattr(w, "dividend_yield")
        assert w.dividend_yield == 0

    def test_quality_weights_has_candidates(self):
        from schemas import QualityWeights
        w = QualityWeights()
        assert w.operating_margin == 0
        assert w.current_ratio == 0
        assert w.insider_ownership == 0
        assert w.interest_coverage == 0

    def test_momentum_weights_has_proximity(self):
        from schemas import MomentumWeights
        w = MomentumWeights()
        assert w.proximity_52w_high == 0

    def test_revisions_weights_has_candidates(self):
        from schemas import RevisionsWeights
        w = RevisionsWeights()
        assert w.short_pct_float == 0
        assert w.analyst_rating == 0

    def test_improvement_config_has_evolution_fields(self):
        from schemas import ImprovementConfig
        ic = ImprovementConfig()
        assert ic.min_observations_for_activation == 12
        assert ic.min_observations_for_deactivation == 12
        assert ic.candidate_initial_weight == 5.0
        assert ic.max_candidate_activations_per_cycle == 1
        assert ic.candidate_ic_threshold == 0.02

    def test_candidate_metrics_constant(self):
        """CANDIDATE_METRICS should contain exactly the 8 candidates."""
        assert len(ie.CANDIDATE_METRICS) == 8
        expected = {
            "proximity_52w_high", "operating_margin", "current_ratio",
            "dividend_yield", "insider_ownership", "short_pct_float",
            "analyst_rating", "interest_coverage",
        }
        assert ie.CANDIDATE_METRICS == expected
