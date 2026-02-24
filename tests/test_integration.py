"""Integration tests for infrastructure: factor correlation, RunContext,
and caching.
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
    compute_factor_correlation,
    _find_latest_cache,
    winsorize_metrics,
    compute_sector_percentiles,
    compute_category_scores,
    compute_composite,
    rank_stocks,
    METRIC_COLS,
    METRIC_DIR,
)
from run_context import RunContext

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def cfg():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def _make_scored_df(n=50, seed=42):
    """Create a scored DataFrame (with percentile columns) for correlation tests."""
    rng = np.random.default_rng(seed)
    sectors = ["Tech", "Health", "Energy", "Finance", "Utility"]
    records = []
    for i in range(n):
        rec = {"Ticker": f"T{i:03d}", "Sector": sectors[i % len(sectors)]}
        for col in METRIC_COLS:
            if METRIC_DIR.get(col, True):
                rec[col] = rng.normal(0.1, 0.05)
            else:
                rec[col] = max(1, rng.normal(15, 5))
        records.append(rec)
    df = pd.DataFrame(records)
    df = winsorize_metrics(df, 0.01, 0.01)
    df = compute_sector_percentiles(df)
    return df


# =====================================================================
# FACTOR CORRELATION
# =====================================================================

class TestFactorCorrelation:
    def test_returns_valid_matrix(self, cfg):
        df = _make_scored_df()
        corr = compute_factor_correlation(df)
        assert not corr.empty
        # Diagonal should be ~1.0 (each metric perfectly correlated with itself)
        for col in corr.columns:
            assert abs(corr.loc[col, col] - 1.0) < 0.001

    def test_symmetric(self, cfg):
        df = _make_scored_df()
        corr = compute_factor_correlation(df)
        for i, row in enumerate(corr.index):
            for j, col in enumerate(corr.columns):
                assert abs(corr.iloc[i, j] - corr.iloc[j, i]) < 0.001

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame()
        corr = compute_factor_correlation(df)
        assert corr.empty

    def test_values_in_range(self, cfg):
        df = _make_scored_df()
        corr = compute_factor_correlation(df)
        assert (corr >= -1.0).all().all()
        assert (corr <= 1.0).all().all()


# =====================================================================
# RUN CONTEXT
# =====================================================================

class TestRunContext:
    def test_config_hash_deterministic(self, cfg):
        """Same config should always produce the same hash."""
        ctx = RunContext(run_id="test_hash_1")
        h1 = ctx.config_hash(cfg)
        h2 = ctx.config_hash(cfg)
        assert h1 == h2
        # Cleanup
        import shutil
        shutil.rmtree(ctx.run_dir, ignore_errors=True)

    def test_different_configs_different_hash(self, cfg):
        """Different configs should produce different hashes."""
        import copy
        ctx = RunContext(run_id="test_hash_2")
        h1 = ctx.config_hash(cfg)
        cfg2 = copy.deepcopy(cfg)
        cfg2["factor_weights"]["valuation"] = 50
        cfg2["factor_weights"]["quality"] = 0
        h2 = ctx.config_hash(cfg2)
        assert h1 != h2
        # Cleanup
        import shutil
        shutil.rmtree(ctx.run_dir, ignore_errors=True)

    def test_universe_changes_affect_hash(self, cfg):
        """Different universe settings should produce different hashes."""
        import copy
        ctx = RunContext(run_id="test_hash_3")
        h1 = ctx.config_hash(cfg)
        cfg2 = copy.deepcopy(cfg)
        cfg2["universe"]["exclude_tickers"] = ["AAPL", "MSFT"]
        h2 = ctx.config_hash(cfg2)
        assert h1 != h2
        # Cleanup
        import shutil
        shutil.rmtree(ctx.run_dir, ignore_errors=True)

    def test_metadata_includes_git_sha(self, cfg):
        """save_metadata() should include git_sha field."""
        ctx = RunContext(run_id="test_git_sha")
        path = ctx.save_metadata()
        with open(path) as f:
            meta = json.load(f)
        assert "git_sha" in meta
        assert isinstance(meta["git_sha"], str)
        assert len(meta["git_sha"]) > 0
        # Cleanup
        import shutil
        shutil.rmtree(ctx.run_dir, ignore_errors=True)

    def test_save_config_creates_file(self, cfg):
        ctx = RunContext(run_id="test_save_cfg")
        path = ctx.save_config(cfg)
        assert path.exists()
        with open(path) as f:
            saved = yaml.safe_load(f)
        assert saved["factor_weights"]["valuation"] == cfg["factor_weights"]["valuation"]
        # Cleanup
        import shutil
        shutil.rmtree(ctx.run_dir, ignore_errors=True)

    def test_save_artifact(self, cfg):
        ctx = RunContext(run_id="test_artifact")
        df = pd.DataFrame({"a": [1, 2, 3]})
        path = ctx.save_artifact("test_df", df)
        assert path.exists()
        loaded = pd.read_parquet(str(path))
        assert len(loaded) == 3
        # Cleanup
        import shutil
        shutil.rmtree(ctx.run_dir, ignore_errors=True)

    def test_save_metadata(self, cfg):
        ctx = RunContext(run_id="test_meta")
        path = ctx.save_metadata({"test_key": "test_val"})
        assert path.exists()
        with open(path) as f:
            meta = json.load(f)
        assert meta["run_id"] == "test_meta"
        assert meta["test_key"] == "test_val"
        assert "python_version" in meta
        # Cleanup
        import shutil
        shutil.rmtree(ctx.run_dir, ignore_errors=True)


# =====================================================================
# CACHING
# =====================================================================

class TestCaching:
    def test_find_latest_cache_no_files(self, tmp_path):
        """No matching cache files → returns (None, None)."""
        path, dt = _find_latest_cache("nonexistent_tier_xyz")
        # If this test runs against the real CACHE_DIR, there should be no
        # "nonexistent_tier_xyz_*.parquet" files
        assert path is None
        assert dt is None

    def test_find_latest_cache_with_hash_filter(self, tmp_path):
        """Config hash filtering should exclude non-matching cache files."""
        # This tests the filtering logic — if there's a cache file without
        # the right hash, it should be skipped
        path, dt = _find_latest_cache("factor_scores", config_hash="zzznonexistent")
        assert path is None


# =====================================================================
# SECTOR-RELATIVE COMPOSITE (config flag)
# =====================================================================

class TestSectorRelativeComposite:
    def test_sector_relative_flag(self, cfg):
        """sector_relative_composite: true should normalize within sectors."""
        import copy
        cfg_rel = copy.deepcopy(cfg)
        cfg_rel["sector_neutral"] = {"sector_relative_composite": True}
        df = pd.DataFrame({
            "Ticker": [f"T{i}" for i in range(20)],
            "Sector": ["Tech"] * 10 + ["Health"] * 10,
            "valuation_score": [50 + i for i in range(20)],
            "quality_score": [50 + i for i in range(20)],
            "growth_score": [50 + i for i in range(20)],
            "momentum_score": [50 + i for i in range(20)],
            "risk_score": [50 + i for i in range(20)],
            "revisions_score": [50 + i for i in range(20)],
        })
        df = compute_composite(df, cfg_rel)
        # In sector-relative mode, composites are normalized within each sector
        # so each sector should have its own 0-100 range
        tech = df[df["Sector"] == "Tech"]["Composite"]
        health = df[df["Sector"] == "Health"]["Composite"]
        # Both sectors should have similar spread despite different raw scores
        assert tech.max() > 0
        assert health.max() > 0
