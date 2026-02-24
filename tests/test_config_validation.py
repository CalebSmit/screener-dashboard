"""Tests for config validation via Pydantic schemas.

Verifies that:
- Valid production config passes validation
- Negative weights are rejected
- Factor weights not summing to 100 are rejected
- Metric weights not summing to 100 are rejected
- Missing keys fall back to defaults
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas import RunConfig, MetricWeights, ValuationWeights


ROOT = Path(__file__).resolve().parent.parent


class TestRunConfigValidation:
    def test_production_config_passes(self):
        """The actual config.yaml should pass validation."""
        with open(ROOT / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        rc = RunConfig(**cfg)
        assert rc.factor_weights.valuation == 22
        assert rc.factor_weights.quality == 22

    def test_empty_config_uses_defaults(self):
        """An empty config should use all defaults and pass."""
        rc = RunConfig()
        assert rc.factor_weights.valuation == 22
        assert rc.factor_weights.quality == 22
        assert rc.factor_weights.growth == 13
        assert rc.factor_weights.momentum == 13
        assert rc.factor_weights.risk == 10
        assert rc.factor_weights.revisions == 10
        assert rc.factor_weights.size == 5
        assert rc.factor_weights.investment == 5

    def test_negative_factor_weight_rejected(self):
        """Negative factor weights should raise ValidationError."""
        with pytest.raises(Exception, match="Weight must be >= 0"):
            RunConfig(factor_weights={"valuation": -10, "quality": 25,
                                       "growth": 25, "momentum": 25,
                                       "risk": 25, "revisions": 10,
                                       "size": 0, "investment": 0})

    def test_factor_weights_not_summing_to_100(self):
        """Factor weights not summing to 100 should raise ValidationError."""
        with pytest.raises(Exception, match="Factor weights must sum to 100"):
            RunConfig(factor_weights={"valuation": 50, "quality": 50,
                                       "growth": 15, "momentum": 15,
                                       "risk": 10, "revisions": 10,
                                       "size": 5, "investment": 5})

    def test_factor_weights_sum_within_tolerance(self):
        """Weights summing to 100.3 (within 0.5 tolerance) should pass."""
        rc = RunConfig(factor_weights={"valuation": 22.1, "quality": 22.1,
                                        "growth": 13.1, "momentum": 13,
                                        "risk": 10, "revisions": 10,
                                        "size": 5, "investment": 5})
        assert rc is not None

    def test_factor_weights_sum_outside_tolerance(self):
        """Weights summing to 101 (outside 0.5 tolerance) should fail."""
        with pytest.raises(Exception, match="Factor weights must sum to 100"):
            RunConfig(factor_weights={"valuation": 23, "quality": 22,
                                       "growth": 13, "momentum": 13,
                                       "risk": 10, "revisions": 10,
                                       "size": 5, "investment": 5})

    def test_metric_weights_not_summing_to_100(self):
        """Metric weights within a category not summing to 100 should fail."""
        with pytest.raises(Exception, match="Metric weights must sum to 100"):
            RunConfig(metric_weights={
                "valuation": {"ev_ebitda": 50, "fcf_yield": 50,
                              "earnings_yield": 20, "ev_sales": 15},
            })

    def test_metric_weights_valid(self):
        """Valid per-category metric weights should pass."""
        rc = RunConfig(metric_weights={
            "valuation": {"ev_ebitda": 25, "fcf_yield": 40,
                          "earnings_yield": 20, "ev_sales": 15},
        })
        assert rc.metric_weights.valuation.ev_ebitda == 25

    def test_missing_factor_weights_uses_defaults(self):
        """Config without factor_weights should use defaults."""
        rc = RunConfig()
        total = (rc.factor_weights.valuation + rc.factor_weights.quality
                 + rc.factor_weights.growth + rc.factor_weights.momentum
                 + rc.factor_weights.risk + rc.factor_weights.revisions
                 + rc.factor_weights.size + rc.factor_weights.investment)
        assert abs(total - 100) < 0.5

    def test_portfolio_num_stocks_bounds(self):
        """num_stocks outside [5, 100] should fail."""
        with pytest.raises(Exception):
            RunConfig(portfolio={"num_stocks": 3})
        with pytest.raises(Exception):
            RunConfig(portfolio={"num_stocks": 200})

    def test_data_quality_coverage_bounds(self):
        """min_data_coverage_pct outside [0, 100] should fail."""
        with pytest.raises(Exception):
            RunConfig(data_quality={"min_data_coverage_pct": -5})
        with pytest.raises(Exception):
            RunConfig(data_quality={"min_data_coverage_pct": 150})


class TestMetricWeightSubModels:
    def test_valuation_defaults_sum_to_100(self):
        vw = ValuationWeights()
        total = vw.ev_ebitda + vw.fcf_yield + vw.earnings_yield + vw.ev_sales + vw.pb_ratio
        assert abs(total - 100) < 0.5

    def test_valuation_bad_sum_rejected(self):
        with pytest.raises(Exception, match="Metric weights must sum to 100"):
            ValuationWeights(ev_ebitda=50, fcf_yield=50,
                             earnings_yield=20, ev_sales=15)

    def test_metric_weights_defaults_all_valid(self):
        """All default metric weight categories should pass validation."""
        mw = MetricWeights()
        assert mw.valuation is not None
        assert mw.quality is not None
        assert mw.growth is not None
        assert mw.momentum is not None
        assert mw.risk is not None
        assert mw.revisions is not None
