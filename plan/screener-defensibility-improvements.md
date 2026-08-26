# Implementation Plan: Screener Defensibility Improvements
## Scope: All CFA/Hedge Fund review suggestions (excluding yfinance replacement)

---

## Overview

Five independent improvement tracks derived from the professional defensibility review.
All changes are surgical — no architectural rewrites. yfinance stays as-is.

---

## Track 1: Data Staleness Threshold (200 → 120 days)

**Problem:** `stale_days = 200` in `factor_engine.py:1933` is too generous. A filing
200 days old (>6 months) is from before the most recent full quarter, meaning key
metrics (ROIC, EV/EBITDA, FCF) can be stale by a full reporting cycle.

**Fix:** Lower threshold to 120 days (4 months — between Q and Q+1 reporting windows).
Add a per-run summary of how many tickers are stale at each threshold.

### Steps

1. **`factor_engine.py:1933`** — Change `stale_days = 200` → `stale_days = 120`.

2. **`factor_engine.py` near line 1933** — Add a config-driven override so operators
   can tune without editing code:
   ```python
   stale_days = cfg.get("data_quality", {}).get("stale_data_threshold_days", 120)
   ```

3. **`config.yaml` under `data_quality:`** — Add:
   ```yaml
   stale_data_threshold_days: 120  # Flag financials older than this (days)
   ```

4. **`run_screener.py`** — After scoring loop, print count of stale tickers:
   ```python
   n_stale = df["_stale_data"].fillna(False).sum()
   if n_stale:
       print(f"  WARNING: {n_stale} tickers have financial data > {stale_days} days old")
   ```

**Files touched:** `factor_engine.py`, `config.yaml`, `run_screener.py`
**Risk:** Low — only affects warning threshold, not scores.
**Tests:** Update any test that hardcodes `stale_days=200`.

---

## Track 2: ROIC Documentation Warning

**Problem:** The ROIC calculation already uses excess cash (Phase 4 fix: `max(0, cash - 2%*rev)`,
capped at 50% of cash) per `factor_engine.py:1352-1358`. This is CORRECT methodology.
However, the review flagged it as a "material bug" — the audit comment at line 1337 says
"Deducting ALL cash inflates ROIC" which was the OLD behavior. The current code is fine.

**Action:** No formula change needed. Add a clear inline comment + a note in the
`DataValidation` sheet (Sheet 6) warning users when a stock has ROIC > 40% and is
cash-heavy (i.e., `_excess_cash / _cash_bs > 0.5` was capped), so analysts can manually
verify. Also ensure `_roic_ic_floor_applied` flag is surfaced in the DataValidation sheet.

### Steps

1. **`factor_engine.py:1354-1358`** — Add flag when IC floor is hit:
   ```python
   _ic_floored = False
   if pd.notna(ta) and ta > 0 and ic < 0.10 * ta:
       ic = 0.10 * ta
       _ic_floored = True
   rec["_roic_ic_floored"] = _ic_floored
   ```

2. **`portfolio_constructor.py` — `write_data_validation_sheet`** — Add `_roic_ic_floored`
   as a diagnostic column in the raw-data table for top-N stocks, labeled
   "ROIC IC-Floor?" so analysts can see when the 10%-TA floor was applied.

3. **`run_screener.py`** — After scoring, warn if many stocks hit the ROIC IC floor:
   ```python
   n_floored = df.get("_roic_ic_floored", pd.Series(False)).sum()
   if n_floored > len(df) * 0.1:
       print(f"  NOTE: {n_floored} tickers had ROIC IC floored at 10% TA")
   ```

**Files touched:** `factor_engine.py`, `portfolio_constructor.py`, `run_screener.py`
**Risk:** Low — diagnostic only, no score change.
**Tests:** Add test that `_roic_ic_floored` is True for a company where IC < 10% TA.

---

## Track 3: Metric Correlation Matrix — Surface High-Correlation Pairs

**Problem:** `compute_factor_correlation()` already EXISTS in `factor_engine.py:2784`
and is already written to the Excel `FactorCorrelation` sheet (Sheet 5). However,
it is NOT surfaced in the run summary, and high-correlation pairs (>0.6) are not
flagged in the console output for the operator to notice.

**Fix:** After computing the correlation matrix, print a console warning for any
non-diagonal pair with |corr| > 0.70. Also add a `HIGH_CORR_PAIRS` summary to
`run_screener.py`'s final run summary.

### Steps

1. **`run_screener.py`** — After `corr = compute_factor_correlation(df)` (around line 1378),
   add:
   ```python
   if not corr.empty:
       # Find high-correlation off-diagonal pairs
       high_pairs = []
       cols = list(corr.columns)
       for i in range(len(cols)):
           for j in range(i + 1, len(cols)):
               val = corr.iloc[i, j]
               if pd.notna(val) and abs(val) > 0.70:
                   high_pairs.append((cols[i].replace("_pct",""),
                                      cols[j].replace("_pct",""), round(val, 2)))
       if high_pairs:
           stats["high_corr_pairs"] = high_pairs
   ```

2. **`run_screener.py` — `print_summary` equivalent** — Print high-correlation pairs:
   ```
   HIGH-CORRELATION METRIC PAIRS (|r| > 0.70):
     ev_ebitda ↔ ev_sales          r=0.82  ← expected overlap
     roic ↔ gross_profit_assets    r=0.71  ← expected overlap
   ```
   With note: "See FactorCorrelation sheet for full matrix."

3. **`config.yaml`** — Add threshold to make it configurable:
   ```yaml
   data_quality:
     high_corr_alert_threshold: 0.70  # Warn when metric pair |r| exceeds this
   ```

**Files touched:** `run_screener.py`, `config.yaml`
**Risk:** Zero — read-only on scores, console + Excel output only.
**Tests:** Add test that `compute_factor_correlation` returns expected high-corr pairs
for a known synthetic dataset.

---

## Track 4: Portfolio Construction — Markowitz Mean-Variance Optimization (Optional Mode)

**Problem:** Current construction is greedy (rank by composite score → iterate through
stocks → enforce sector cap). This is suboptimal: it can select two 0.85-correlated
Tech stocks over one uncorrelated Materials stock, even if the risk-adjusted portfolio
would prefer more diversification.

**Fix:** Add an *optional* Markowitz optimizer as a second portfolio mode, selectable
via `portfolio.weighting: 'markowitz'` in config. When selected:
- Use historical return covariance to minimize portfolio variance at target return
- Respect sector concentration caps as linear constraints
- Fall back to greedy if scipy is unavailable or optimization fails

**Important:** Keep greedy as default. Markowitz is opt-in and clearly labeled
"experimental" since it requires a stable covariance estimate (60-day minimum history).

### Steps

1. **`portfolio_constructor.py`** — Add new function `construct_portfolio_markowitz()`:

   ```python
   def construct_portfolio_markowitz(
       df: pd.DataFrame,
       cfg: dict,
       price_returns: pd.DataFrame  # (dates × tickers) daily returns, pre-fetched
   ) -> pd.DataFrame:
       """
       Mean-variance optimized portfolio. Minimizes portfolio variance subject to:
         - Target return = composite-score-weighted average expected return
         - Max sector concentration from config
         - Min/max position bounds (0.5% - max_position_pct)
         - Sum of weights = 100%
       Falls back to greedy if scipy.optimize unavailable or fails.
       """
   ```

   Implementation outline:
   - Filter candidates (same Steps 0-1b as greedy: trap flags, median composite, coverage, liquidity)
   - Take top 2×N candidates (e.g., top 50 for a 25-stock portfolio) to give optimizer room
   - Compute covariance matrix from `price_returns` for candidate tickers (60-day minimum)
   - Use `scipy.optimize.minimize` with `method='SLSQP'` to minimize `w.T @ Σ @ w`
   - Constraints: weights sum to 1, sector constraints as linear inequality constraints,
     position bounds (0 to `max_position_pct/100`)
   - Post-process: zero out positions < 0.5%, renormalize

2. **`portfolio_constructor.py` — `construct_portfolio()`** — Route to Markowitz when
   config requests it:
   ```python
   weighting = pcfg.get("weighting", "equal")
   if weighting == "markowitz":
       returns_data = pcfg.get("_price_returns_df")  # injected by run_screener
       if returns_data is not None:
           return construct_portfolio_markowitz(df, cfg, returns_data)
       else:
           warnings.warn("Markowitz requested but price_returns not provided; falling back to greedy")
   ```

3. **`run_screener.py`** — When `weighting == 'markowitz'`, extract price history
   DataFrame from the scored `df` (already fetched: columns like `_price_history_*`
   don't exist, so fetch separately):
   - Add helper `_fetch_price_returns(tickers, days=252)` that calls
     `yf.download(tickers, period="1y", auto_adjust=True)["Close"].pct_change().dropna()`
   - Inject into `cfg["portfolio"]["_price_returns_df"]` before calling `construct_portfolio`
   - On failure, log warning and remove key (triggers greedy fallback)

4. **`config.yaml`** — Add documentation comment:
   ```yaml
   portfolio:
     weighting: 'score'  # Options: 'equal', 'inverse_vol', 'score', 'markowitz' (experimental)
   ```

5. **`schemas.py`** — If `PortfolioConfig` exists, add `markowitz` to valid weighting enum.

**Files touched:** `portfolio_constructor.py`, `run_screener.py`, `config.yaml`, `schemas.py`
**Risk:** Medium — new code path. Guarded behind config flag + fallback. Never modifies
default behavior.
**Tests:**
- Unit test: `construct_portfolio_markowitz` with synthetic returns produces a portfolio
  that respects sector caps and position bounds.
- Unit test: falls back to greedy when price_returns is None.
- Unit test: Markowitz weights sum to 100%.

---

## Track 5: Analyst Revisions Data Coverage — Transparency Improvement

**Problem:** Analyst Revisions category (10% weight) has 30-60% missing data from
`earnings_history`. When coverage < 30% of universe, the category is auto-disabled and
weight redistributed — but this happens silently. A professional reviewer would want
to know exactly what fraction of the universe has each revisions sub-metric.

**Fix:** At end of scoring pipeline, print a per-metric coverage table for the Revisions
category. Add this to the Excel DataValidation sheet as a "Data Coverage" summary.

### Steps

1. **`run_screener.py`** — After scoring, compute and print revisions coverage:
   ```python
   rev_metrics = ["analyst_surprise", "price_target_upside",
                  "earnings_acceleration", "consecutive_beat_streak", "short_interest_ratio"]
   print("\nANALYST REVISIONS COVERAGE:")
   for m in rev_metrics:
       if m in df.columns:
           pct = df[m].notna().mean() * 100
           print(f"  {m:<30s}: {pct:5.1f}%")
   ```

2. **`portfolio_constructor.py` — `write_data_validation_sheet`** — Add a
   "Data Coverage Summary" block after the Sector Median Context table, showing
   coverage % for all revisions metrics and the 5 most-missing metrics overall.

3. **No score changes** — This is diagnostic only.

**Files touched:** `run_screener.py`, `portfolio_constructor.py`
**Risk:** Zero.
**Tests:** Verify coverage table is written to DataValidation sheet in Excel output test.

---

## Execution Order

Run these tracks in this sequence (each is independently mergeable):

| # | Track | Files | Risk | Time Estimate |
|---|-------|-------|------|---------------|
| 1 | Data staleness 200→120 | `factor_engine.py`, `config.yaml`, `run_screener.py` | Low | 30 min |
| 2 | ROIC IC-floor flag | `factor_engine.py`, `portfolio_constructor.py`, `run_screener.py` | Low | 30 min |
| 3 | Correlation matrix warnings | `run_screener.py`, `config.yaml` | Zero | 20 min |
| 5 | Revisions coverage transparency | `run_screener.py`, `portfolio_constructor.py` | Zero | 20 min |
| 4 | Markowitz optimization (opt-in) | `portfolio_constructor.py`, `run_screener.py`, `config.yaml`, `schemas.py` | Medium | 2-3 hr |

Tracks 1, 2, 3, 5 can be done in one session. Track 4 (Markowitz) is a separate session.

---

## What This Plan Does NOT Change

- **yfinance** — kept as-is per owner request
- **Factor weights** — no weight changes; these are methodology/diagnostic improvements
- **ROIC formula** — already correct (excess cash deduction implemented in Phase 4)
- **Scoring pipeline** — no changes to rank logic
- **Backtest** — biases already properly disclaimed; no changes needed
- **Any existing tests** — all changes are additive or tighten thresholds

---

## Test Checklist

After implementing all tracks, run:

```bash
cd "c:/Users/Caleb/OneDrive - Dordt University/Desktop/Screener-1"
python -m pytest tests/ -v --tb=short
```

Expected: all 372 existing tests continue to pass + new tests for Tracks 1-4.

---

## Notes for Presenter

If showing this system to a CFA or PM after these improvements:
- Lead with: "The ROIC excess-cash correction was already in Phase 4 — here's where."
- Show the FactorCorrelation sheet — explain that intentional overlap (EV/EBITDA + FCF Yield)
  is documented and justified; they diversify error sources.
- Show the data staleness summary — demonstrates active data quality monitoring.
- Markowitz is opt-in; default is composite-score-weighted (defensible and simple).
- Backtest biases are *documented* in the output header — transparency is a strength.
