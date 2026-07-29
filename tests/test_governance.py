#!/usr/bin/env python3
"""Tests for Phase 13 self-improving-engine governance guardrails.

Covers findings F2/F3/F6/F7/F29/F30 from the hedge-fund review:
- improvement.enabled kill switch is enforced
- auto-apply requires allow_auto_apply=True AND statistical significance (IC IR)
- weight optimization refuses to silently fall back to a shorter horizon
- candidate activation applies Benjamini-Hochberg FDR control
- change-log carries full provenance
- cumulative anti-drift cap
"""

import yaml
import numpy as np
import pandas as pd
import pytest

import improvement_engine as ie


@pytest.fixture(autouse=True)
def temp_dirs(tmp_path, monkeypatch):
    imp = tmp_path / "improvement"
    imp.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ie, "IMPROVEMENT_DIR", imp)
    monkeypatch.setattr(ie, "LIVE_IC_HISTORY_PATH", imp / "live_ic_history.csv")
    monkeypatch.setattr(ie, "CHANGE_LOG_PATH", imp / "change_log.csv")
    monkeypatch.setattr(ie, "ROOT", tmp_path)
    monkeypatch.setattr(ie, "CONFIG_PATH", tmp_path / "config.yaml")
    return tmp_path


def _write_config(path, **improvement_overrides):
    imp = {
        "enabled": True,
        "allow_auto_apply": False,
        "min_ic_ir_for_auto_apply": 0.5,
        "optimization_horizon": "1m",
        "max_cumulative_change": 6.0,
        "candidate_multiple_comparisons_correction": True,
        "min_observations_for_proposal": 8,
        "auto_apply_threshold": 2.0,
    }
    imp.update(improvement_overrides)
    cfg = {
        "factor_weights": {
            "valuation": 22, "quality": 22, "growth": 13, "momentum": 13,
            "risk": 10, "revisions": 10, "size": 5, "investment": 5,
        },
        "improvement": imp,
    }
    with open(path, "w") as f:
        yaml.dump(cfg, f)


def _write_ic_history(path, horizon="1m", n=30, mean_ic=0.05, std_ic=0.02, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        row = {"run_date": f"2026-01-{(i % 28) + 1:02d}", "horizon": horizon, "n_tickers": 480}
        row["composite_ic"] = float(rng.normal(mean_ic, std_ic))
        for cat in ie.CATEGORY_NAMES:
            row[f"{cat}_ic"] = float(rng.normal(mean_ic, std_ic))
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


# --- F6: horizon must not silently fall back ------------------------------

def test_analyze_refuses_horizon_fallback(temp_dirs):
    _write_ic_history(ie.LIVE_IC_HISTORY_PATH, horizon="1w", n=30)
    res = ie.analyze_ic_trends(horizon="1m", allow_horizon_fallback=False)
    assert "_warning" in res
    assert res.get("_horizon_used") is None


def test_analyze_allows_fallback_when_explicit(temp_dirs):
    _write_ic_history(ie.LIVE_IC_HISTORY_PATH, horizon="1w", n=30)
    res = ie.analyze_ic_trends(horizon="1m", allow_horizon_fallback=True)
    assert res.get("_horizon_used") == "1w"


# --- F2/F3: auto-apply requires switch + significance ---------------------

def test_auto_apply_blocked_by_default(temp_dirs):
    _write_config(ie.CONFIG_PATH, allow_auto_apply=False)
    _write_ic_history(ie.LIVE_IC_HISTORY_PATH, horizon="1m", n=60, mean_ic=0.08, std_ic=0.02)
    prop = ie.propose_weight_changes(min_observations=8)
    assert prop["status"] == "proposal_ready"
    assert prop["can_auto_apply"] is False
    assert "allow_auto_apply" in prop["auto_apply_block_reason"]


def test_auto_apply_blocked_by_low_significance(temp_dirs):
    # allow_auto_apply on, but IC is pure noise -> low IR -> blocked
    _write_config(ie.CONFIG_PATH, allow_auto_apply=True, min_ic_ir_for_auto_apply=0.5)
    _write_ic_history(ie.LIVE_IC_HISTORY_PATH, horizon="1m", n=60, mean_ic=0.001, std_ic=0.05)
    prop = ie.propose_weight_changes(min_observations=8)
    assert prop["status"] == "proposal_ready"
    assert prop["significance_ok"] is False
    assert prop["can_auto_apply"] is False


def test_proposal_refuses_when_horizon_missing(temp_dirs):
    _write_config(ie.CONFIG_PATH, optimization_horizon="1m")
    _write_ic_history(ie.LIVE_IC_HISTORY_PATH, horizon="1w", n=30)
    prop = ie.propose_weight_changes(min_observations=8)
    assert prop["status"] == "insufficient_data"


# --- F29: change-log provenance -------------------------------------------

def test_apply_changes_records_provenance(temp_dirs):
    _write_config(ie.CONFIG_PATH)
    res = ie.apply_changes(
        {"valuation": 1.0, "quality": -1.0}, reason="test",
        auto_applied=False, confidence="medium", n_observations=30,
        ic_ir=0.6, horizon_used="1m",
    )
    assert res.get("applied") is True
    log = pd.read_csv(ie.CHANGE_LOG_PATH)
    for col in ["auto_applied", "confidence", "ic_ir", "horizon_used", "backup_path", "applied_by"]:
        assert col in log.columns
    assert (log["applied_by"] == "human-approved").all()


def test_apply_changes_blocks_auto_when_switch_off(temp_dirs):
    _write_config(ie.CONFIG_PATH, allow_auto_apply=False)
    res = ie.apply_changes({"valuation": 1.0, "quality": -1.0}, auto_applied=True)
    assert res.get("applied") is False
    assert "allow_auto_apply" in res.get("error", "")


def test_apply_changes_blocked_when_engine_disabled(temp_dirs):
    _write_config(ie.CONFIG_PATH, enabled=False)
    res = ie.apply_changes({"valuation": 1.0, "quality": -1.0})
    assert res.get("applied") is False
    assert "disabled" in res.get("error", "")


# --- F30: cumulative anti-drift cap ---------------------------------------

def test_cumulative_drift_cap_blocks(temp_dirs):
    _write_config(ie.CONFIG_PATH, max_cumulative_change=6.0)
    # Seed change log with prior +5% cumulative on valuation
    pd.DataFrame([
        {"date": "2026-01-01", "change_type": "factor_weight", "category": "valuation",
         "old_value": 22.0, "new_value": 27.0},
    ]).to_csv(ie.CHANGE_LOG_PATH, index=False)
    ok, msg = ie._cumulative_change_ok({"valuation": 2.0})
    assert ok is False
    assert "valuation" in msg


def test_cumulative_drift_cap_allows_within_bound(temp_dirs):
    _write_config(ie.CONFIG_PATH, max_cumulative_change=6.0)
    ok, _ = ie._cumulative_change_ok({"valuation": 2.0})
    assert ok is True


# --- F7: Benjamini-Hochberg FDR -------------------------------------------

def test_bh_all_significant():
    passed = ie._benjamini_hochberg_pass({"a": 0.001, "b": 0.002, "c": 0.004}, alpha=0.10)
    assert passed == {"a", "b", "c"}


def test_bh_none_significant():
    passed = ie._benjamini_hochberg_pass({"a": 0.5, "b": 0.6, "c": 0.9}, alpha=0.10)
    assert passed == set()


def test_bh_partial():
    # One clearly significant, rest noise; with m=8 only the strongest survives
    pvals = {f"m{i}": 0.6 for i in range(7)}
    pvals["strong"] = 0.001
    passed = ie._benjamini_hochberg_pass(pvals, alpha=0.10)
    assert "strong" in passed


def test_ir_to_pvalue_monotone():
    # Higher IR -> lower p-value
    p_low = ie._ir_to_one_sided_pvalue(0.1, 30)
    p_high = ie._ir_to_one_sided_pvalue(1.0, 30)
    assert p_high < p_low
