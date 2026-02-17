# IMPLEMENTATION PLAN — Phased Roadmap to Professional-Grade Screener
**Date:** 2026-02-17
**Prerequisite:** Review and approval of FORENSIC_AUDIT_REPORT.md

---

## Phase 0: Instrumentation & Tests (Zero Behavior Change)

**Goal:** Add observability, testing, and type safety without changing any scoring output. A run before and after Phase 0 must produce identical rankings.

### Tasks

| # | Task | Files to Touch | Acceptance Test |
|---|------|---------------|-----------------|
| 0.1 | **Add structured logging** — Replace all `print()` with Python `logging` module. JSON-formatted log entries with level, timestamp, module, message, and optional fields (ticker, metric, value). | `run_screener.py`, `factor_engine.py`, `portfolio_constructor.py` | Log output is valid JSON; pipeline produces identical Excel + Parquet output |
| 0.2 | **Add run_id and config snapshot** — Generate `uuid4` run_id at start. Save a copy of config.yaml + run metadata (start_time, python version, pkg versions) to `runs/{run_id}/`. | `run_screener.py` (new `runs/` dir) | `runs/{run_id}/config.yaml` and `runs/{run_id}/meta.json` exist after each run |
| 0.3 | **Add per-ticker timing** — Log elapsed time for each `fetch_single_ticker()` call. Emit summary stats (min/mean/max/p95 fetch time). | `factor_engine.py` | Log contains `fetch_time_ms` for each ticker |
| 0.4 | **Add typed schemas** — Create Pydantic models for: `RawTickerData`, `FactorScores`, `RunConfig`. Validate data at boundaries. | New `schemas.py` | Invalid data raises `ValidationError`; valid data passes through unchanged |
| 0.5 | **Add data lineage artifacts** — After each pipeline step, save intermediate DataFrames as Parquet: `raw_data.parquet`, `winsorized.parquet`, `percentiles.parquet`, `category_scores.parquet`. | `run_screener.py` | Each run produces 4 intermediate Parquet files in `runs/{run_id}/` |
| 0.6 | **Create golden-file test fixture** — Run pipeline with mocked API data for 10 tickers. Save input fixture + expected output. | New `tests/test_golden.py`, `tests/fixtures/` | `pytest tests/test_golden.py` passes; output matches fixture exactly |
| 0.7 | **Unit tests for all 17 metrics** — Test each metric computation with known inputs/expected outputs, including edge cases (negative earnings, zero denominators, missing data). | New `tests/test_metrics.py` | `pytest tests/test_metrics.py` passes (50+ test cases) |
| 0.8 | **Unit tests for scoring pipeline** — Test winsorization, percentile ranking, category scoring, composite scoring, ranking. | New `tests/test_scoring.py` | `pytest tests/test_scoring.py` passes |
| 0.9 | **Pin dependencies** — Change `requirements.txt` from `>=` to `==` with exact versions. | `requirements.txt` | `pip install -r requirements.txt` installs exact versions |
| 0.10 | **Initialize git repo** — `git init`, create `.gitignore` (exclude `cache/`, `runs/`, `__pycache__/`, `.pytest_cache/`), initial commit. | New `.gitignore` | `git log` shows initial commit |

### Phase 0 Acceptance Criteria
- All existing tests pass (`pytest test_screener.py`)
- New tests pass (`pytest tests/`)
- Pipeline output (factor_output.xlsx) is **byte-for-byte identical** to pre-Phase-0 output (verified with a golden-file test)
- Structured logs are written to `runs/{run_id}/run.log`

---

## Phase 1: Make Outputs Reproducible

**Goal:** Any run can be fully reproduced given its run_id. Two runs with the same config and data produce identical output.

### Tasks

| # | Task | Files to Touch | Acceptance Test |
|---|------|---------------|-----------------|
| 1.1 | **Config-aware caching** — Include a hash of config.yaml factor/metric weights in the cache filename. Cache miss if config changes. | `factor_engine.py` (`_find_latest_cache`, `save_cache`) | Change a weight → cache miss → fresh computation |
| 1.2 | **Save raw fetched data** — Serialize raw yfinance response dicts as `runs/{run_id}/raw_data.parquet` before any computation. | `run_screener.py` | Raw data file exists and can be loaded to reproduce metric computation |
| 1.3 | **Deterministic sample data** — Verify `_generate_sample_data(seed=42)` is truly deterministic. Add test. | `factor_engine.py`, `tests/test_determinism.py` | Two calls with same seed produce identical DataFrame |
| 1.4 | **Record effective weights** — After revisions auto-disable, save the *actual* weights used (not just config weights) to `runs/{run_id}/effective_weights.json`. | `run_screener.py` | Effective weights file matches runtime computation |
| 1.5 | **Snapshot universe** — Save the actual ticker list used (after exclusions, after failures) to `runs/{run_id}/universe.json`. | `run_screener.py` | Universe file lists exactly the tickers that were scored |
| 1.6 | **Add reproducibility test** — Run pipeline twice with mocked data; assert outputs are identical. | `tests/test_reproducibility.py` | Test passes: two runs produce identical rankings |

### Phase 1 Acceptance Criteria
- Given a `run_id`, all inputs (config, universe, raw data, effective weights) and all outputs (scores, rankings, quality log) can be recovered
- Two runs with identical inputs produce bit-identical Parquet output

---

## Phase 2: Make Data "As-Of" Safe (Where Possible)

**Goal:** Minimize look-ahead bias and stale-data risk within the constraints of yfinance as a data source.

### Tasks

| # | Task | Files to Touch | Acceptance Test |
|---|------|---------------|-----------------|
| 2.1 | **Add data freshness checks** — For each ticker, log the date of the most recent financial statement and flag if > 6 months old. | `factor_engine.py` (in `_fetch_single_ticker_inner`) | Data quality log shows `stale_fundamental` entries for old filings |
| 2.2 | **Fix 6M momentum consistency** — Change 6-month return to 6-1 month (exclude most recent month) to match 12-1 convention. | `factor_engine.py:642-645` | 6M return uses `price_1m_ago` instead of `price_latest`; test updated |
| 2.3 | **Fix Piotroski normalization** — Use raw integer score (0-9 based on testable signals) instead of proportional normalization. Require >= 5 testable signals. | `factor_engine.py:556-591` | Piotroski returns integer 0-9; NaN if < 5 testable |
| 2.4 | **Handle negative earnings yield** — Compute negative earnings yield for loss-making companies (don't set to NaN). | `factor_engine.py:516-518` | Loss-making companies get negative earnings yield → low rank |
| 2.5 | **Add calendar-based price lookback** — Use `pd.DateOffset` instead of fixed integer index for momentum price dates. | `factor_engine.py:291-293` | Price dates are accurate to within 3 business days of intended date |
| 2.6 | **Add backtest disclaimers** — Inject prominent disclaimers into backtest output files (CSV header, Excel sheet). | `backtest.py` | Backtest outputs contain disclaimer text |
| 2.7 | **Restructure revisions category** — If only `analyst_surprise` is available, give it 100% of revisions weight (not 25%). Remove NaN placeholder metrics from scoring. | `factor_engine.py`, `config.yaml` | Revisions category uses only non-NaN metrics with adjusted weights |
| 2.8 | **Add fiscal year end detection** — Log fiscal year-end dates for each company; flag when annual data is > 10 months old. | `factor_engine.py` | Fiscal year staleness visible in logs |

### Phase 2 Acceptance Criteria
- All known metric bugs (6M momentum, Piotroski, negative earnings yield) are fixed
- Backtest outputs carry prominent disclaimers
- Data freshness is logged for every ticker
- All Phase 0 golden-file tests are updated for the new expected behavior

---

## Phase 3: Institutional Data Architecture (Optional Plug-In)

**Goal:** Design an abstraction layer so institutional data providers can replace yfinance without changing the scoring logic.

### Tasks

| # | Task | Files to Touch | Acceptance Test |
|---|------|---------------|-----------------|
| 3.1 | **Define DataProvider interface** — ABC with `fetch_ticker(ticker, as_of_date) -> RawTickerData` and `fetch_universe(index, as_of_date) -> list[str]`. | New `data_providers/base.py` | Interface is importable; yfinance adapter passes type checks |
| 3.2 | **Wrap yfinance in adapter** — Move all yfinance code into `data_providers/yfinance_provider.py` implementing the interface. | `factor_engine.py` → `data_providers/yfinance_provider.py` | All existing tests pass with yfinance adapter |
| 3.3 | **Add mock provider** — Create `data_providers/mock_provider.py` that returns deterministic fixture data. | New file | Tests can run offline in < 1 second |
| 3.4 | **Add provider selection to config** — `data_source: yfinance` or `data_source: mock` in config.yaml. | `config.yaml`, `run_screener.py` | Switching providers via config works |
| 3.5 | **Document institutional provider spec** — Write a guide for implementing a Bloomberg/FactSet/Compustat adapter. | New `data_providers/PROVIDER_GUIDE.md` | Guide covers all 26 required fields, as-of-date semantics, and point-in-time requirements |
| 3.6 | **Add point-in-time backtest (with institutional data)** — If provider supports `as_of_date`, use historical data for each backtest month. | `backtest.py` | With mock provider, backtest uses correct historical data per period |

### Phase 3 Acceptance Criteria
- yfinance is fully encapsulated behind an interface
- A new data provider can be added by implementing one class
- All existing tests pass with both yfinance and mock providers
- Backtest can optionally use point-in-time data when provider supports it

---

## Priority Summary

```
Phase 0 (IMMEDIATE):  Zero-risk instrumentation. Do this first.
Phase 1 (WEEK 1-2):   Reproducibility. Essential for any professional use.
Phase 2 (WEEK 2-4):   Data quality fixes. Addresses known metric bugs.
Phase 3 (OPTIONAL):   Architecture for institutional data. Only if budget allows.
```

---

## TEST DESIGN (Step 6)

### Test Directory Structure

```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures
├── fixtures/
│   ├── mock_ticker_data.json      # 10-ticker raw yfinance response fixture
│   ├── expected_scores.parquet    # Expected output for golden-file test
│   ├── edge_cases.json            # Edge case inputs (negative EPS, zero assets, etc.)
│   └── sector_data.json           # Sector-realistic data for N tickers
├── test_metrics.py                # Unit tests for each of 17 metrics
├── test_scoring.py                # Tests for scoring pipeline
├── test_golden.py                 # Golden-file regression test
├── test_integration.py            # End-to-end with mocked API
├── test_portfolio.py              # Portfolio construction tests
├── test_reproducibility.py        # Determinism tests
└── test_properties.py             # Property-based / invariant tests
```

### Test Categories

#### 1. Unit Tests: Individual Metrics (`test_metrics.py`)

Each metric gets a test class with at minimum these cases:

```python
class TestEVEBITDA:
    def test_normal_case(self):
        # EV=100B, EBITDA=10B → EV/EBITDA = 10.0
    def test_negative_ebitda(self):
        # EBITDA = -5B → NaN
    def test_zero_ebitda(self):
        # EBITDA = 0 → NaN
    def test_missing_ev(self):
        # EV = NaN → NaN
    def test_negative_ev(self):
        # EV = -10B → NaN
    def test_ev_fallback(self):
        # enterpriseValue = NaN, but MC/Debt/Cash available → computed EV

class TestROIC:
    def test_normal_case(self):
        # Known inputs → known ROIC
    def test_negative_pretax_income(self):
        # pretaxIncome < 0 → default 21% tax rate
    def test_negative_ic(self):
        # IC < 0 → NaN
    def test_tax_rate_clamping(self):
        # Tax rate > 50% → clamped to 50%
    def test_missing_ebit(self):
        # EBIT = NaN → ROIC = NaN

class TestPiotroskiFScore:
    def test_perfect_score(self):
        # All 9 signals testable, all pass → 9
    def test_all_fail(self):
        # All 9 signals testable, all fail → 0
    def test_partial_data(self):
        # Only 5 signals testable, 3 pass → 3 (raw, not normalized)
    def test_insufficient_data(self):
        # Only 3 signals testable → NaN
    def test_zero_denominators(self):
        # ta=0, cl=0 → those signals untestable

class TestFCFYield:
    def test_normal_negative_capex(self):
        # OCF=10B, CapEx=-3B → FCF=7B
    def test_positive_capex(self):
        # OCF=10B, CapEx=3B → FCF=7B
    def test_missing_capex(self):
        # OCF=10B, CapEx=NaN → FCF=OCF=10B (documented limitation)
    def test_zero_ev(self):
        # EV=0 → NaN

class TestBeta:
    def test_normal_case(self):
        # Known stock/market returns → known beta
    def test_insufficient_overlap(self):
        # < 200 common dates → NaN
    def test_zero_market_variance(self):
        # All market returns identical → NaN
    def test_date_alignment(self):
        # Stock and market have different trading days → correct overlap

class TestEarningsYield:
    def test_positive_eps(self):
        # EPS=5, Price=100 → 0.05
    def test_negative_eps(self):
        # EPS=-3, Price=100 → currently NaN (Phase 2: -0.03)
    def test_zero_price(self):
        # Price=0 → NaN

class TestMomentum12_1:
    def test_normal_case(self):
        # P12=80, P1m=100 → 0.25
    def test_price_decline(self):
        # P12=100, P1m=80 → -0.20
    def test_missing_12m_price(self):
        # P12=NaN → NaN

class TestSustainableGrowth:
    def test_with_dividends(self):
        # NI=100M, Equity=500M, Dividends=30M → ROE=0.2, Ret=0.7, SG=0.14
    def test_no_dividends(self):
        # Dividends=NaN, dividendRate=NaN → full retention → ROE * 1.0
    def test_negative_ni(self):
        # NI < 0 → NaN
```

#### 2. Scoring Pipeline Tests (`test_scoring.py`)

```python
class TestWinsorization:
    def test_values_clipped(self):
        # Extreme values are brought to 1st/99th pctile
    def test_small_sample_no_winsorize(self):
        # < 10 values → no winsorization
    def test_nan_preserved(self):
        # NaN values remain NaN after winsorization

class TestSectorPercentiles:
    def test_higher_is_better(self):
        # Highest ROIC in sector → ~100 percentile
    def test_lower_is_better(self):
        # Lowest EV/EBITDA in sector → ~100 percentile (inverted)
    def test_nan_gets_50(self):
        # NaN metric → 50.0 percentile
    def test_small_sector(self):
        # < 3 valid values → all get 50.0
    def test_all_same_value(self):
        # All stocks same metric value → all get same rank

class TestCategoryScores:
    def test_weights_sum_to_100(self):
        # Verify config weights sum correctly
    def test_partial_metrics(self):
        # Some metrics NaN (→50) → score still computed
    def test_weight_normalization(self):
        # If weights don't sum to 100%, result is normalized

class TestComposite:
    def test_min_max_scaling(self):
        # Lowest composite → 0, highest → 100
    def test_all_equal(self):
        # All same composite → all get 50.0
    def test_sector_relative(self):
        # With sector_relative=True, scaling is per-sector

class TestRanking:
    def test_descending_order(self):
        # Highest composite → Rank 1
    def test_tie_handling(self):
        # Equal composites → same rank, next rank skipped
```

#### 3. Golden-File Regression Test (`test_golden.py`)

```python
def test_golden_file():
    """Fixed input → exact expected output. Detects any unintended behavior change."""
    raw = load_fixture("mock_ticker_data.json")  # 10 tickers
    market_returns = load_fixture("mock_market_returns.json")
    cfg = load_fixture("test_config.yaml")

    df = compute_metrics(raw, market_returns)
    df = winsorize_metrics(df, 0.01, 0.01)
    df = compute_sector_percentiles(df)
    df = compute_category_scores(df, cfg)
    df = compute_composite(df, cfg)
    df = rank_stocks(df)

    expected = pd.read_parquet("tests/fixtures/expected_scores.parquet")
    pd.testing.assert_frame_equal(df[COMPARE_COLS], expected[COMPARE_COLS])
```

#### 4. Integration Test (`test_integration.py`)

```python
def test_end_to_end_mocked(monkeypatch):
    """Full pipeline with 10 mocked tickers, no network access."""
    # Monkeypatch yfinance to return fixture data
    # Run full pipeline via run_screener.main()
    # Assert: Excel file created with 3 sheets
    # Assert: Parquet cache created
    # Assert: Data quality log created
    # Assert: Ranking is deterministic
```

#### 5. Property-Based / Invariant Tests (`test_properties.py`)

```python
class TestRankingInvariants:
    def test_ranks_are_complete(self, scored_df):
        """Every scored stock has a rank."""
        assert scored_df["Rank"].notna().all()

    def test_ranks_are_positive(self, scored_df):
        """All ranks are >= 1."""
        assert (scored_df["Rank"] >= 1).all()

    def test_rank_count_matches(self, scored_df):
        """Number of unique ranks matches number of stocks (allowing ties)."""
        assert scored_df["Rank"].max() <= len(scored_df)

    def test_composite_in_range(self, scored_df):
        """All composite scores are in [0, 100]."""
        assert (scored_df["Composite"] >= 0).all()
        assert (scored_df["Composite"] <= 100).all()

    def test_higher_composite_lower_rank(self, scored_df):
        """Monotonicity: if A.Composite > B.Composite, then A.Rank < B.Rank."""
        for i in range(len(scored_df) - 1):
            if scored_df.iloc[i]["Composite"] > scored_df.iloc[i+1]["Composite"]:
                assert scored_df.iloc[i]["Rank"] <= scored_df.iloc[i+1]["Rank"]

    def test_category_scores_in_range(self, scored_df):
        """All category scores are in [0, 100]."""
        for cat in ["valuation_score", "quality_score", "growth_score",
                     "momentum_score", "risk_score", "revisions_score"]:
            if cat in scored_df.columns:
                valid = scored_df[cat].dropna()
                assert (valid >= 0).all() and (valid <= 100).all()

    def test_percentile_scores_in_range(self, scored_df):
        """All percentile columns are in [0, 100]."""
        pct_cols = [c for c in scored_df.columns if c.endswith("_pct")]
        for col in pct_cols:
            valid = scored_df[col].dropna()
            assert (valid >= 0).all() and (valid <= 100).all()

    def test_weights_sum_to_100(self, portfolio_df):
        """Portfolio weights sum to ~100%."""
        for wt_col in ["Equal_Weight_Pct", "RiskParity_Weight_Pct"]:
            if wt_col in portfolio_df.columns:
                total = portfolio_df[wt_col].sum()
                assert abs(total - 100.0) < 0.1

    def test_sector_cap_respected(self, portfolio_df, cfg):
        """No sector exceeds max_sector_concentration."""
        max_sec = cfg["portfolio"]["max_sector_concentration"]
        for sec, count in portfolio_df["Sector"].value_counts().items():
            assert count <= max_sec
```

#### 6. Reproducibility Tests (`test_reproducibility.py`)

```python
def test_sample_data_deterministic():
    """Two calls to _generate_sample_data with same seed → identical output."""
    df1 = _generate_sample_data(universe_df, seed=42)
    df2 = _generate_sample_data(universe_df, seed=42)
    pd.testing.assert_frame_equal(df1, df2)

def test_pipeline_deterministic():
    """Two full pipeline runs with same mocked input → identical rankings."""
    result1 = run_pipeline_with_mock_data()
    result2 = run_pipeline_with_mock_data()
    pd.testing.assert_frame_equal(result1, result2)
```

### Test Fixtures Required

1. **`mock_ticker_data.json`** — Raw yfinance-like data for 10 tickers spanning at least 3 sectors. Include edge cases: one ticker with negative EPS, one with missing EBITDA, one with very high D/E.
2. **`mock_market_returns.json`** — 252 daily log returns for ^GSPC.
3. **`test_config.yaml`** — Identical to production config.yaml but with known weights.
4. **`expected_scores.parquet`** — Hand-verified expected output for the 10 mock tickers.
5. **`edge_cases.json`** — Individual ticker records designed to test each edge case path.
