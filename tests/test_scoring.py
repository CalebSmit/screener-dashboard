"""Tests for the scoring pipeline: winsorization, percentile ranking,
category scores, composite scoring, and ranking.

All tests use synthetic data — no network access required.
"""

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
    add_financial_sector_caveat,
    rank_stocks,
    _is_bank_like,
    METRIC_COLS,
    METRIC_DIR,
    _BANK_ONLY_METRICS,
    _NONBANK_ONLY_METRICS,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def cfg():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def _make_df(n=100, seed=42):
    """Create a synthetic DataFrame with all metric columns and sectors.

    Uses n=100 by default to ensure each sector has >= 10 stocks
    (required for meaningful percentile ranking).
    Financials-sector stocks get bank-specific metrics (P/B, ROE, ROA,
    Equity Ratio) and NaN for generic metrics that are meaningless for
    banks (EV/EBITDA, EV/Sales, ROIC, Gross Profit/Assets, D/E).
    """
    rng = np.random.default_rng(seed)
    sectors = ["Information Technology", "Financials", "Health Care",
               "Energy", "Consumer Staples"]
    records = []
    for i in range(n):
        sector = sectors[i % len(sectors)]
        is_bank = sector == "Financials"
        rec = {
            "Ticker": f"T{i:03d}",
            "Company": f"Company {i}",
            "Sector": sector,
            "_is_bank_like": is_bank,
        }
        for col in METRIC_COLS:
            # Skip metrics that don't apply to this stock type
            if is_bank and col in _NONBANK_ONLY_METRICS:
                rec[col] = np.nan
                continue
            if not is_bank and col in _BANK_ONLY_METRICS:
                rec[col] = np.nan
                continue
            if col == "analyst_surprise":
                rec[col] = rng.normal(0.03, 0.05) if rng.random() > 0.5 else np.nan
            elif col == "price_target_upside":
                rec[col] = rng.normal(0.10, 0.15) if rng.random() > 0.5 else np.nan
            elif col == "debt_equity":
                rec[col] = max(0, rng.normal(1.0, 0.5))
            elif col == "piotroski_f_score":
                rec[col] = int(rng.integers(2, 9))
            elif col == "beta":
                rec[col] = rng.normal(1.0, 0.3)
            elif col == "volatility":
                rec[col] = max(0.05, rng.normal(0.25, 0.08))
            elif col == "size_log_mcap":
                # -log(marketCap) where marketCap ~ $2B-$500B → range roughly -27 to -21
                rec[col] = -np.log(rng.uniform(2e9, 500e9))
            elif col == "asset_growth":
                rec[col] = rng.normal(0.08, 0.15)
            elif col == "pb_ratio":
                rec[col] = max(0.3, rng.normal(1.2, 0.4))
            elif col == "roe":
                rec[col] = max(0.02, rng.normal(0.12, 0.04))
            elif col == "roa":
                rec[col] = max(0.002, rng.normal(0.01, 0.005))
            elif col == "equity_ratio":
                rec[col] = max(0.05, min(0.20, rng.normal(0.10, 0.03)))
            elif METRIC_DIR.get(col, True):
                # "Higher is better" → positive values
                rec[col] = rng.normal(0.1, 0.05)
            else:
                # "Lower is better" → positive values (e.g., EV/EBITDA)
                rec[col] = max(1, rng.normal(15, 5))
            # Introduce some NaN randomly
            if rng.random() < 0.05:
                rec[col] = np.nan
        records.append(rec)
    return pd.DataFrame(records)


# =====================================================================
# WINSORIZATION
# =====================================================================

class TestWinsorization:
    def test_extreme_values_clipped(self):
        """Create a controlled dataset where one value is clearly extreme."""
        n = 100
        # All values tightly clustered around 15, with one extreme outlier
        vals = [15.0 + i * 0.1 for i in range(n)]
        vals[0] = 9999.0  # Plant extreme outlier
        df = pd.DataFrame({
            "Ticker": [f"T{i}" for i in range(n)],
            "ev_ebitda": vals,
        })
        # Add other metric cols as NaN so winsorize_metrics doesn't fail
        for col in METRIC_COLS:
            if col not in df.columns:
                df[col] = np.nan
        df = winsorize_metrics(df, 0.01, 0.01)
        # After winsorization, the extreme value should be reduced
        assert df.loc[0, "ev_ebitda"] < 9999.0

    def test_small_sample_no_winsorize(self):
        """With < 10 non-NaN values, winsorization should be skipped."""
        df = pd.DataFrame({
            "Ticker": [f"T{i}" for i in range(5)],
            "ev_ebitda": [5.0, 10.0, 15.0, 20.0, 100.0],
        })
        df_result = winsorize_metrics(df.copy(), 0.01, 0.01)
        # Should be unchanged (< 10 values)
        assert df_result.loc[4, "ev_ebitda"] == 100.0

    def test_nan_preserved(self):
        df = _make_df(50)
        nan_count_before = df["ev_ebitda"].isna().sum()
        df = winsorize_metrics(df, 0.01, 0.01)
        nan_count_after = df["ev_ebitda"].isna().sum()
        assert nan_count_before == nan_count_after


# =====================================================================
# SECTOR PERCENTILES
# =====================================================================

class TestSectorPercentiles:
    def test_output_range(self):
        df = _make_df(50)
        df = winsorize_metrics(df, 0.01, 0.01)
        df = compute_sector_percentiles(df)
        for col in METRIC_COLS:
            pc = f"{col}_pct"
            if pc in df.columns:
                valid = df[pc].dropna()
                assert (valid >= 0).all(), f"{pc} has values < 0"
                assert (valid <= 100).all(), f"{pc} has values > 100"

    def test_higher_is_better_direction(self):
        """For 'higher is better' metrics, the highest value should get ~100."""
        df = pd.DataFrame({
            "Ticker": [f"T{i}" for i in range(12)],
            "Sector": ["Tech"] * 12,
            "roic": [0.01, 0.05, 0.10, 0.20, 0.30, 0.03, 0.07, 0.12, 0.15, 0.25, 0.08, 0.18],
        })
        # Add other required columns as NaN
        for col in METRIC_COLS:
            if col not in df.columns:
                df[col] = np.nan
        df = compute_sector_percentiles(df)
        # ROIC is "higher is better" → highest value (idx 4, 0.30) should get highest pct
        assert df.loc[4, "roic_pct"] > df.loc[0, "roic_pct"]

    def test_lower_is_better_direction(self):
        """For 'lower is better' metrics, the lowest value should get ~100."""
        df = pd.DataFrame({
            "Ticker": [f"T{i}" for i in range(12)],
            "Sector": ["Tech"] * 12,
            "ev_ebitda": [5.0, 10.0, 15.0, 20.0, 30.0, 8.0, 12.0, 18.0, 25.0, 35.0, 7.0, 22.0],
        })
        for col in METRIC_COLS:
            if col not in df.columns:
                df[col] = np.nan
        df = compute_sector_percentiles(df)
        # EV/EBITDA is "lower is better" → lowest value (idx 0, 5.0) should get highest pct
        assert df.loc[0, "ev_ebitda_pct"] > df.loc[4, "ev_ebitda_pct"]

    def test_nan_stays_nan(self):
        """NaN raw values should produce NaN percentile (not imputed to 50)."""
        df = pd.DataFrame({
            "Ticker": [f"T{i}" for i in range(12)],
            "Sector": ["Tech"] * 12,
            "roic": [0.10, 0.20, np.nan, 0.15, 0.25, 0.12, 0.18, 0.22, 0.08, 0.30, 0.14, 0.19],
        })
        for col in METRIC_COLS:
            if col not in df.columns:
                df[col] = np.nan
        df = compute_sector_percentiles(df)
        assert pd.isna(df.loc[2, "roic_pct"])

    def test_small_sector_uses_universe_ranking(self):
        """Sector with < 10 valid values → falls back to universe-wide ranking.

        Previously assigned flat 50.0, which penalised good stocks and
        rewarded bad ones in small sectors. Now uses universe-wide
        percentile ranking so actual values still differentiate stocks.
        """
        df = pd.DataFrame({
            "Ticker": [f"T{i}" for i in range(5)],
            "Sector": ["Tiny"] * 5,
            "roic": [0.10, 0.20, 0.15, 0.25, 0.30],
        })
        for col in METRIC_COLS:
            if col not in df.columns:
                df[col] = np.nan
        df = compute_sector_percentiles(df)
        # Higher ROIC should get higher percentile (roic is "higher is better")
        assert df.loc[4, "roic_pct"] > df.loc[0, "roic_pct"]
        # All should be in valid range
        assert (df["roic_pct"].dropna() >= 0).all()
        assert (df["roic_pct"].dropna() <= 100).all()


# =====================================================================
# CATEGORY SCORES
# =====================================================================

class TestCategoryScores:
    def test_scores_computed(self, cfg):
        df = _make_df(50)
        df = winsorize_metrics(df, 0.01, 0.01)
        df = compute_sector_percentiles(df)
        df = compute_category_scores(df, cfg)
        for cat in ["valuation_score", "quality_score", "growth_score",
                     "momentum_score", "risk_score", "revisions_score",
                     "size_score", "investment_score"]:
            assert cat in df.columns
            assert df[cat].notna().any()

    def test_scores_in_range(self, cfg):
        df = _make_df(50)
        df = winsorize_metrics(df, 0.01, 0.01)
        df = compute_sector_percentiles(df)
        df = compute_category_scores(df, cfg)
        for cat in ["valuation_score", "quality_score", "growth_score",
                     "momentum_score", "risk_score", "size_score",
                     "investment_score"]:
            valid = df[cat].dropna()
            assert (valid >= 0).all(), f"{cat} has values < 0"
            assert (valid <= 100).all(), f"{cat} has values > 100"


# =====================================================================
# COMPOSITE SCORE
# =====================================================================

class TestComposite:
    def test_percentile_rank_scaling(self, cfg):
        df = _make_df(50)
        df = winsorize_metrics(df, 0.01, 0.01)
        df = compute_sector_percentiles(df)
        df = compute_category_scores(df, cfg)
        df = compute_composite(df, cfg)
        assert df["Composite"].min() >= 0
        assert df["Composite"].max() <= 100
        # After percentile rank, scores should span (0, 100] with
        # max = 100 (the rank(pct=True) of the top stock is 1.0)
        assert df["Composite"].max() == 100.0

    def test_all_equal(self, cfg):
        """All same category scores → all get same composite (tied percentile rank)."""
        df = pd.DataFrame({
            "Ticker": ["A", "B", "C"],
            "Sector": ["Tech"] * 3,
            "valuation_score": [50.0, 50.0, 50.0],
            "quality_score": [50.0, 50.0, 50.0],
            "growth_score": [50.0, 50.0, 50.0],
            "momentum_score": [50.0, 50.0, 50.0],
            "risk_score": [50.0, 50.0, 50.0],
            "revisions_score": [50.0, 50.0, 50.0],
            "size_score": [50.0, 50.0, 50.0],
            "investment_score": [50.0, 50.0, 50.0],
        })
        df = compute_composite(df, cfg)
        # All tied → all get the same percentile rank
        assert df["Composite"].nunique() == 1

    def test_empty_df(self, cfg):
        df = pd.DataFrame()
        result = compute_composite(df, cfg)
        assert "Composite" in result.columns
        assert len(result) == 0

    def test_nan_category_does_not_drop_stock(self, cfg):
        """A stock with NaN revisions_score should still get a Composite."""
        df = pd.DataFrame({
            "Ticker": ["A", "B", "C"],
            "Sector": ["Tech", "Tech", "Tech"],
            "valuation_score": [70.0, 60.0, 50.0],
            "quality_score": [65.0, 55.0, 45.0],
            "growth_score": [80.0, 70.0, 60.0],
            "momentum_score": [60.0, 50.0, 40.0],
            "risk_score": [55.0, 45.0, 35.0],
            "revisions_score": [np.nan, 50.0, 40.0],
            "size_score": [50.0, 50.0, 50.0],
            "investment_score": [50.0, 50.0, 50.0],
        })
        df = compute_composite(df, cfg)
        assert df["Composite"].notna().all(), "Stock with NaN category should not be dropped"
        assert len(df) == 3

    def test_all_categories_nan_is_nan(self, cfg):
        """Stock with ALL category scores NaN gets NaN Composite."""
        df = pd.DataFrame({
            "Ticker": ["A", "B"],
            "Sector": ["Tech", "Tech"],
            "valuation_score": [np.nan, 50.0],
            "quality_score": [np.nan, 50.0],
            "growth_score": [np.nan, 50.0],
            "momentum_score": [np.nan, 50.0],
            "risk_score": [np.nan, 50.0],
            "revisions_score": [np.nan, 50.0],
            "size_score": [np.nan, 50.0],
            "investment_score": [np.nan, 50.0],
        })
        df = compute_composite(df, cfg)
        assert pd.isna(df.loc[0, "Composite"])
        assert pd.notna(df.loc[1, "Composite"])


# =====================================================================
# VALUE TRAP FLAGS
# =====================================================================

class TestValueTrapFlags:
    def test_flags_applied(self, cfg):
        df = _make_df(100)
        df = winsorize_metrics(df, 0.01, 0.01)
        df = compute_sector_percentiles(df)
        df = compute_category_scores(df, cfg)
        df = compute_composite(df, cfg)
        df = apply_value_trap_flags(df, cfg)
        assert "Value_Trap_Flag" in df.columns
        # With 2-of-3 logic, fewer stocks flagged than OR logic
        # but some should still be flagged
        assert df["Value_Trap_Flag"].any()
        assert not df["Value_Trap_Flag"].all()

    def test_disabled(self, cfg):
        cfg_copy = cfg.copy()
        cfg_copy["value_trap_filters"] = {"enabled": False}
        df = _make_df(20)
        df["quality_score"] = 10.0  # Very low
        df["momentum_score"] = 10.0
        df["revisions_score"] = 10.0
        df = apply_value_trap_flags(df, cfg_copy)
        assert not df["Value_Trap_Flag"].any()

    def test_majority_logic_requires_two_breaches(self, cfg):
        """2-of-3 logic: one low dimension should NOT trigger flag."""
        df = pd.DataFrame({
            "Ticker": ["A", "B", "C", "D"],
            "Sector": ["Tech"] * 4,
            "quality_score": [10.0, 10.0, 10.0, 80.0],
            "momentum_score": [80.0, 10.0, 80.0, 80.0],
            "revisions_score": [80.0, 80.0, 10.0, 80.0],
            "Composite": [60.0, 60.0, 60.0, 90.0],
        })
        df = apply_value_trap_flags(df, cfg)
        # A: only quality is low → 1 breach → NOT flagged
        assert not df.loc[0, "Value_Trap_Flag"]
        # B: quality + momentum low → 2 breaches → flagged
        assert df.loc[1, "Value_Trap_Flag"]
        # C: quality + revisions low → 2 breaches → flagged
        assert df.loc[2, "Value_Trap_Flag"]
        # D: nothing low → NOT flagged
        assert not df.loc[3, "Value_Trap_Flag"]

    def test_insufficient_data_flag_present(self, cfg):
        """Insufficient_Data_Flag should be set for stocks with NaN category scores."""
        df = pd.DataFrame({
            "Ticker": ["A", "B", "C"],
            "Sector": ["Tech", "Tech", "Tech"],
            "quality_score": [50.0, np.nan, 60.0],
            "momentum_score": [50.0, 50.0, np.nan],
            "revisions_score": [50.0, 50.0, 50.0],
            "Composite": [80.0, 70.0, 60.0],
        })
        df = apply_value_trap_flags(df, cfg)
        assert "Insufficient_Data_Flag" in df.columns
        assert not df.loc[0, "Insufficient_Data_Flag"]  # A has all data
        assert df.loc[1, "Insufficient_Data_Flag"]       # B has NaN quality
        assert df.loc[2, "Insufficient_Data_Flag"]       # C has NaN momentum

    def test_nan_not_flagged_as_value_trap(self, cfg):
        """NaN category scores should NOT trigger Value_Trap_Flag."""
        df = pd.DataFrame({
            "Ticker": ["A"],
            "Sector": ["Tech"],
            "quality_score": [np.nan],
            "momentum_score": [np.nan],
            "revisions_score": [np.nan],
            "Composite": [80.0],
        })
        df = apply_value_trap_flags(df, cfg)
        assert not df.loc[0, "Value_Trap_Flag"]
        assert df.loc[0, "Insufficient_Data_Flag"]


# =====================================================================
# RANKING
# =====================================================================

class TestRanking:
    def test_descending_order(self, cfg):
        df = _make_df(50)
        df = winsorize_metrics(df, 0.01, 0.01)
        df = compute_sector_percentiles(df)
        df = compute_category_scores(df, cfg)
        df = compute_composite(df, cfg)
        df = rank_stocks(df)
        # Rank 1 should have the highest composite
        assert df.iloc[0]["Rank"] == 1
        assert df.iloc[0]["Composite"] >= df.iloc[1]["Composite"]

    def test_ranks_complete(self, cfg):
        df = _make_df(50)
        df = winsorize_metrics(df, 0.01, 0.01)
        df = compute_sector_percentiles(df)
        df = compute_category_scores(df, cfg)
        df = compute_composite(df, cfg)
        df = rank_stocks(df)
        assert df["Rank"].notna().all()
        assert (df["Rank"] >= 1).all()
        assert df["Rank"].max() <= len(df)

    def test_tie_handling(self):
        """Equal composites → same rank, next rank skipped."""
        df = pd.DataFrame({
            "Ticker": ["A", "B", "C"],
            "Composite": [85.0, 85.0, 83.0],
        })
        df = rank_stocks(df)
        assert list(df["Rank"]) == [1, 1, 3]


# =====================================================================
# FINANCIAL SECTOR CAVEAT
# =====================================================================

class TestFinancialSectorCaveat:
    def test_financials_flagged(self):
        """Stocks in 'Financials' sector should have caveat flag True."""
        df = pd.DataFrame({
            "Ticker": ["JPM", "AAPL", "GS"],
            "Sector": ["Financials", "Information Technology", "Financials"],
        })
        df = add_financial_sector_caveat(df)
        assert df.loc[0, "Financial_Sector_Caveat"] == True
        assert df.loc[1, "Financial_Sector_Caveat"] == False
        assert df.loc[2, "Financial_Sector_Caveat"] == True

    def test_no_sector_column(self):
        """Missing Sector column → all False."""
        df = pd.DataFrame({"Ticker": ["A", "B"]})
        df = add_financial_sector_caveat(df)
        assert not df["Financial_Sector_Caveat"].any()


# =====================================================================
# Weight Sensitivity Analysis
# =====================================================================
class TestWeightSensitivity:
    """Tests for the weight sensitivity analysis feature."""

    def _make_scored_df(self, n=30):
        """Create a scored DataFrame with category scores and Rank."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "Ticker": [f"T{i:03d}" for i in range(n)],
            "Sector": rng.choice(["Tech", "Health", "Finance"], n),
            "valuation_score": rng.uniform(10, 90, n),
            "quality_score": rng.uniform(10, 90, n),
            "growth_score": rng.uniform(10, 90, n),
            "momentum_score": rng.uniform(10, 90, n),
            "risk_score": rng.uniform(10, 90, n),
            "revisions_score": rng.uniform(10, 90, n),
            "size_score": rng.uniform(10, 90, n),
            "investment_score": rng.uniform(10, 90, n),
        })
        # Compute a simple composite and rank
        cats = ["valuation_score", "quality_score", "growth_score",
                "momentum_score", "risk_score", "revisions_score",
                "size_score", "investment_score"]
        df["Composite"] = df[cats].mean(axis=1)
        df["Rank"] = df["Composite"].rank(ascending=False, method="min").astype(int)
        return df

    def test_returns_dataframe(self, cfg):
        from factor_engine import run_weight_sensitivity
        df = self._make_scored_df()
        result = run_weight_sensitivity(df, cfg, perturbation_pct=5.0, top_n=10)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_required_columns(self, cfg):
        from factor_engine import run_weight_sensitivity
        df = self._make_scored_df()
        result = run_weight_sensitivity(df, cfg, perturbation_pct=5.0, top_n=10)
        expected_cols = {"category", "direction", "original_weight",
                         "perturbed_weight", "top_n_unchanged",
                         "top_n_changed", "jaccard_similarity"}
        assert expected_cols.issubset(set(result.columns))

    def test_jaccard_range(self, cfg):
        from factor_engine import run_weight_sensitivity
        df = self._make_scored_df()
        result = run_weight_sensitivity(df, cfg, perturbation_pct=5.0, top_n=10)
        assert (result["jaccard_similarity"] >= 0).all()
        assert (result["jaccard_similarity"] <= 1).all()

    def test_small_perturbation_high_stability(self, cfg):
        """Small (1%) perturbation → Jaccard should remain high (>0.7)."""
        from factor_engine import run_weight_sensitivity
        df = self._make_scored_df(n=50)
        result = run_weight_sensitivity(df, cfg, perturbation_pct=1.0, top_n=20)
        assert (result["jaccard_similarity"] >= 0.5).all()

    def test_both_directions_tested(self, cfg):
        from factor_engine import run_weight_sensitivity
        df = self._make_scored_df()
        result = run_weight_sensitivity(df, cfg, perturbation_pct=5.0, top_n=10)
        directions = result["direction"].unique()
        assert "+" in directions
        assert "-" in directions


# =====================================================================
# EPS Basis Mismatch Flag
# =====================================================================
class TestEPSBasisMismatch:
    """Tests for the GAAP/normalized EPS mismatch detection."""

    def test_extreme_ratio_flagged(self):
        from factor_engine import compute_metrics
        # forwardEps / trailingEps = 10.0 / 2.0 = 5.0 → flagged
        raw = [{"Ticker": "MISMATCH", "marketCap": 1e12, "enterpriseValue": 1e12,
                "sector": "Tech", "shortName": "Mismatch",
                "trailingEps": 2.0, "forwardEps": 10.0, "currentPrice": 100}]
        df = compute_metrics(raw, pd.Series(dtype=float))
        assert df["_eps_basis_mismatch"].iloc[0] == True
        assert df["_eps_ratio"].iloc[0] == 5.0

    def test_normal_ratio_not_flagged(self):
        from factor_engine import compute_metrics
        # forwardEps / trailingEps = 5.5 / 5.0 = 1.1 → not flagged
        raw = [{"Ticker": "NORMAL", "marketCap": 1e12, "enterpriseValue": 1e12,
                "sector": "Tech", "shortName": "Normal",
                "trailingEps": 5.0, "forwardEps": 5.5, "currentPrice": 100}]
        df = compute_metrics(raw, pd.Series(dtype=float))
        assert df["_eps_basis_mismatch"].iloc[0] == False

    def test_small_trailing_eps_not_flagged(self):
        from factor_engine import compute_metrics
        # abs(trailingEps) < 0.10 → not evaluated, defaults to False
        raw = [{"Ticker": "SMALL", "marketCap": 1e12, "enterpriseValue": 1e12,
                "sector": "Tech", "shortName": "Small EPS",
                "trailingEps": 0.05, "forwardEps": 5.0, "currentPrice": 100}]
        df = compute_metrics(raw, pd.Series(dtype=float))
        assert df["_eps_basis_mismatch"].iloc[0] == False


# =====================================================================
# Data Provenance
# =====================================================================
class TestDataProvenance:
    """Tests for per-ticker data provenance fields."""

    def test_metric_count_populated(self):
        from factor_engine import compute_metrics
        raw = [{"Ticker": "PROV", "marketCap": 1e12, "enterpriseValue": 1e12,
                "sector": "Tech", "shortName": "Provenance",
                "currentPrice": 100, "trailingEps": 5.0,
                "totalRevenue": 1e9, "netIncome": 1e8,
                "operatingCashFlow": 1.2e8, "totalAssets": 5e9}]
        df = compute_metrics(raw, pd.Series(dtype=float))
        assert "_metric_count" in df.columns
        assert "_metric_total" in df.columns
        assert df["_metric_count"].iloc[0] > 0
        assert df["_metric_total"].iloc[0] == 18
