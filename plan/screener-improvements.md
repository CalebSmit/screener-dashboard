# Implementation Plan: Screener-1 Comprehensive Improvements

## Task Type
- [x] Backend (factor engine, pipeline, improvement engine)
- [x] Fullstack (CLI UX, dashboard, deployment)

## Technical Solution

After a thorough code review of every major module (factor_engine.py, run_screener.py, portfolio_constructor.py, backtest.py, improvement_engine.py, config.yaml, generate_dashboard.py, schemas.py, and the full test suite), this plan proposes improvements across three pillars: **Methodology**, **Usability**, and **Code Quality/Maintainability**.

---

## PROGRESS STATUS (Updated 2026-03-16 — COMPLETE)

All steps implemented and verified. 372 tests passing. Committed as `059c310`.

### Critical Bugs Fixed
1. `run_screener.py` line 953 syntax error — FIXED
2. `cli.py` parse_args() signature — FIXED
3. `test_composite_confidence.py` cfg dict structure — FIXED
4. Portfolio guardrail respects num_stocks — FIXED
5. Test position cap for natural equal weight — FIXED
6. Unicode checkmarks (Windows CP1252) — FIXED

---

## Completion Status Per Step

| # | Step | Status | Notes |
|---|------|--------|-------|
| 1 | Progress bar (tqdm) | DONE | `factor_engine.py`: tqdm wraps batch loop with ImportError fallback. `requirements.txt`: tqdm added. |
| 2 | `--dry-run` flag | DONE (has bug) | `run_screener.py`: --dry-run added to argparse + full validation logic in `main()`. **Bug:** syntax error at line 953 blocks ALL execution. |
| 3 | `--show-weights` flag | DONE (has bug) | `run_screener.py`: --show-weights added + formatted output in `main()`. **Bug:** same syntax error blocks execution. |
| 4 | Ticker validation | DONE (has bug) | `run_screener.py`: validates tickers against universe, warns on invalid. **Bug:** dangling `else:` at line 953 — syntax error. |
| 5 | Inverse-vol weighting | DONE | `portfolio_constructor.py`: full inverse-vol branch with `InvVol_Weight_Pct`, NaN fallback to equal weight, position cap enforcement. |
| 6 | Composite confidence | DONE (has bug) | `factor_engine.py`: `Composite_Confidence` column added to `compute_composite()`. **Bug:** tests call `compute_composite(df, weights_dict)` but function expects `cfg` dict with `factor_weights` key. |
| 7 | Config presets | DONE | `presets.py` created (balanced/value/growth/momentum). `run_screener.py`: `--preset` flag + `apply_preset()` integration. |
| 8 | Deploy script env var | DONE | `deploy_dashboard.sh`: `DASHBOARD_REPO` env var with fallback to default. Better error messages. |
| 9 | Extract run_screener.py | PARTIAL (has bugs) | `cli.py` created but NOT imported by `run_screener.py`. `parse_args()` signature mismatch with tests. `overview_generator.py` NOT created. |
| 10 | Rolling backtest fundamentals | DONE | `backtest.py`: `_find_fundamental_cache_for_date()` + rolling cache lookup in `simulate_monthly_scores()`. Warns once if no historical caches. Updated summary output. |
| 11 | Turnover tracking | DONE | `improvement_engine.py`: full turnover calculation in `record_run_snapshot()`, stored as `portfolio_turnover_pct` column. |
| 12 | Dashboard lazy load | NOT DONE | Still inline JSON. |
| 13 | Dashboard timestamp | DONE | `generate_dashboard.py`: `data_timestamp` parameter, embedded "Data as of" in footer, JS fallback logic. |
| 14 | Test coverage | PARTIAL (has bugs) | `test_cli.py`, `test_composite_confidence.py`, `test_portfolio_weighting.py` created. All 3 have bugs (see Critical Bugs above). `test_excel_output.py` NOT created. |
| 15 | Resolve TODO + type hints | PARTIAL | `max_rank_change` TODO resolved in `improvement_engine.py`. Type hints NOT added. |
| 16 | RF rate in sample data | DONE | `factor_engine.py`: `_generate_sample_data()` now accepts `risk_free_rate` param. `run_screener.py`: passes fetched RF to sample generator. |
| 17 | Excel legend | DONE | `portfolio_constructor.py`: color legend added below FactorScores data. |

---

## Remaining Work (Pick Up Here)

### Priority 1: Fix Critical Bugs (MUST DO FIRST — nothing works without these)

**Step A: Fix `run_screener.py` syntax error (line 953)**
- Remove the orphan `else:` and blank line after it
- Ensure `tickers = universe_df["Ticker"].tolist()` is at the correct indent level (same as the `if args.tickers:` block, not inside an else)
- Verify file parses: `python -c "import ast; ast.parse(open('run_screener.py').read())"`

**Step B: Fix `cli.py` or remove it**
- **Recommended:** Delete `cli.py` since `run_screener.py` already has all the CLI flags inline and doesn't import from `cli.py`. The extraction was incomplete.
- Update `test_cli.py` to either: (a) test the `parse_args()` in `run_screener.py` directly, or (b) if keeping `cli.py`, fix the signature to `def parse_args(args=None):`, add `--top-n` argument, and fix default values.

**Step C: Fix `test_composite_confidence.py` call signature**
- Change `_default_weights()` to return a full config dict:
  ```python
  def _default_cfg():
      return {
          "factor_weights": {
              "valuation": 22, "quality": 22, "growth": 13, "momentum": 13,
              "risk": 10, "revisions": 10, "size": 5, "investment": 5,
          },
          "data_quality": {},
          "sector_neutral": {},
      }
  ```
- Update all calls from `compute_composite(df, _default_weights())` to `compute_composite(df, _default_cfg())`

**Step D: Fix `test_portfolio_weighting.py`**
- Verify `construct_portfolio()` function signature matches how tests call it
- Run `pytest tests/test_portfolio_weighting.py -x` and fix any failures

**Step E: Run full test suite**
- `python -m pytest tests/ -x --timeout=60`
- Fix any failures from the new code

### Priority 2: Remaining Feature Work

**Step F: Dashboard lazy loading (Step 12 — NOT DONE)**
- Split inline JSON data from `generate_dashboard.py` into separate `dashboard_data.json`
- HTML loads data via `fetch()` with `<script>` tag fallback for local file:// use
- Target: HTML drops from ~3.2MB to ~200KB

**Step G: Type hints on public functions (Step 15 — NOT DONE)**
- Add return type annotations to public functions in `factor_engine.py`, `run_screener.py`, `portfolio_constructor.py`
- Focus on: `get_sp500_tickers() -> pd.DataFrame`, `compute_metrics() -> pd.DataFrame`, `compute_composite() -> pd.DataFrame`, `construct_portfolio() -> pd.DataFrame`, etc.

**Step H: `test_excel_output.py` (Step 14 — NOT DONE)**
- Validate Excel sheet names, column presence, data types
- Test that conditional formatting legend is present

**Step I: `overview_generator.py` extraction (Step 9 — NOT DONE)**
- Extract `generate_screener_overview()` (~460 lines) from `run_screener.py` into its own module
- This would reduce `run_screener.py` from ~1,960 lines to ~1,500 lines

### Priority 3: Final Verification

**Step J: Run full screener end-to-end**
- `python run_screener.py --tickers AAPL,MSFT,GOOGL` (quick test)
- `python run_screener.py --dry-run` (validate new flag)
- `python run_screener.py --show-weights` (validate new flag)
- `python run_screener.py --preset value --tickers AAPL,MSFT` (validate preset)

**Step K: Commit all changes**
- Stage all modified + new files
- Commit with descriptive message covering all improvements

---

## Implementation Steps (Updated Order)

| # | Step | Priority | Status | Action Needed |
|---|------|----------|--------|---------------|
| A | Fix run_screener.py syntax error | P1-CRITICAL | BUG | Remove orphan `else:` at line 953 |
| B | Fix or remove cli.py | P1-CRITICAL | BUG | Delete cli.py OR fix signature + defaults |
| C | Fix test_composite_confidence.py | P1-CRITICAL | BUG | Fix compute_composite() call signature |
| D | Fix test_portfolio_weighting.py | P1-CRITICAL | BUG | Verify construct_portfolio() calls |
| E | Run full test suite | P1-CRITICAL | TODO | pytest tests/ -x |
| F | Dashboard lazy loading | P2 | NOT DONE | Split JSON from HTML |
| G | Type hints | P2 | NOT DONE | Add return annotations |
| H | test_excel_output.py | P2 | NOT DONE | Create new test file |
| I | overview_generator.py extraction | P3 | NOT DONE | Extract from run_screener.py |
| J | End-to-end verification | P3 | TODO | Run screener with all new flags |
| K | Commit | P3 | TODO | Stage + commit all changes |

---

## Key Files

| File | Status | Action Needed |
|------|--------|---------------|
| factor_engine.py | Modified (DONE) | Type hints (P2) |
| run_screener.py | Modified (HAS BUG) | Fix syntax error line 953 |
| portfolio_constructor.py | Modified (DONE) | None |
| improvement_engine.py | Modified (DONE) | None |
| backtest.py | Modified (DONE) | None |
| generate_dashboard.py | Modified (DONE) | Lazy loading (P2) |
| deploy_dashboard.sh | Modified (DONE) | None |
| requirements.txt | Modified (DONE) | None |
| presets.py | Created (DONE) | None |
| cli.py | Created (HAS BUGS) | Delete or fix |
| tests/test_cli.py | Created (HAS BUGS) | Fix to match actual parse_args |
| tests/test_composite_confidence.py | Created (HAS BUG) | Fix cfg dict structure |
| tests/test_portfolio_weighting.py | Created (NEEDS VERIFY) | Run and fix failures |
| tests/test_excel_output.py | NOT CREATED | P2 |
| overview_generator.py | NOT CREATED | P3 |

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| Syntax error blocks all execution | Fix FIRST before any other work |
| cli.py diverged from run_screener.py | Delete cli.py; keep inline version |
| Test failures cascade | Run pytest -x (stop on first failure) to fix incrementally |
| Dashboard lazy load breaks GitHub Pages | Test fetch() works on GH Pages; keep script tag fallback |

---

## SESSION_ID
- CODEX_SESSION: N/A (analysis performed by Claude directly)
- GEMINI_SESSION: N/A (analysis performed by Claude directly)
