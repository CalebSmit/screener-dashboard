# Review Changelog — Hedge-Fund-Grade Remediation (2026-07)

Maps each finding from [HEDGE_FUND_REVIEW_FINDINGS.md](HEDGE_FUND_REVIEW_FINDINGS.md) → the fix applied → the files changed. Grouped by the commit chunk that carried it.

**Baseline:** 450 tests pass / 1 fail (stale sample-data fixture). **After:** 492 tests pass / 0 fail.
**Owner gate:** F1 full fix, metric-bug batch, and F4 (config-gated OFF) approved; F17/F18 deferred.

---

## Environment fix (prerequisite)

The documented 4-env-var SSL workaround no longer works under Python 3.13 (the Avast root in `combined_ca.pem` has non-critical Basic Constraints, rejected by stdlib `ssl` and Windows schannel-curl). **Only `CURL_CA_BUNDLE` matters** for the live data path, because yfinance 0.2.66 uses `curl_cffi`, whose libcurl accepts the bundle. Both live runs (baseline 824.7s and final) verified genuinely live (Network OK, cache=0, 0 SSL cert errors). Documented in `SCREENER_OVERVIEW.md`.

---

## Track A — governance, robustness, reporting, docs (commit `6f3dc7f`)

| Finding | Sev | Fix | Files |
|---|---|---|---|
| **F2** kill switch was a no-op; auto-apply with no human | CRIT | Enforce `improvement.enabled` gate; new `allow_auto_apply` (default false = human-only) | `run_screener.py`, `improvement_engine.py`, `config.yaml`, `schemas.py` |
| **F3** confidence = obs count, not significance | CRIT | Auto-apply also requires composite IC IR ≥ `min_ic_ir_for_auto_apply` (0.5); block reason surfaced | `improvement_engine.py`, `config.yaml`, `schemas.py` |
| **F6** weight opt on 1-week IC via silent fallback | HIGH | `analyze_ic_trends(horizon, allow_horizon_fallback=False)`; refuse to optimize if target horizon empty | `improvement_engine.py`, `config.yaml` |
| **F7** 8 candidates, no multiple-comparisons control | HIGH | Benjamini-Hochberg FDR across candidates before activation | `improvement_engine.py`, `config.yaml`, `schemas.py` |
| **F28** shrinkage/EWM/regime magic numbers | MED | `regime_scale_factor` default 0.0 (off until validated); documented | `config.yaml` |
| **F29** change-log absent / no provenance | MED | Enriched log (auto_applied, confidence, ic_ir, horizon_used, backup_path); `applied_by` = human/auto | `improvement_engine.py` |
| **F30** no cumulative drift cap / feedback | MED | `_cumulative_change_ok` anti-drift cap over trailing window | `improvement_engine.py`, `config.yaml`, `schemas.py` |
| **F15** blanket `filterwarnings("ignore")` masks schema drift | MED | Targeted filter — screener's own UserWarnings pass | `factor_engine.py`, `run_screener.py` |
| **F34** dead config `winsorize_percentiles` | LOW | Pass config value through to `winsorize_metrics` | `run_screener.py` |
| **F39** silent `except: pass` in aux I/O | LOW | Replaced with `warnings.warn` | `factor_engine.py`, `improvement_engine.py` |
| **F41** legacy `factor_engine.main` penalizes banks | LOW | Bank-aware coverage filter (matches run_screener path) | `factor_engine.py` |
| **F10** no covariance risk model (reporting) | HIGH | New `portfolio_risk.compute_covariance_risk` (Ledoit-Wolf); surfaced in run summary | `portfolio_risk.py`, `run_screener.py`, `portfolio_constructor.py` |
| **F11** no turnover report for live top-25 | HIGH | `portfolio_risk.compute_turnover` vs prior snapshot + est. cost; in run summary | `portfolio_risk.py`, `run_screener.py`, `portfolio_constructor.py` |
| **F23** Markowitz mis-specified as risk model | MED | Ledoit-Wolf shrinkage; accurate "minimum-variance" docstring | `portfolio_constructor.py`, `portfolio_risk.py` |
| **F24** max_position_pct silently breached when n<20 | MED | Warn when cap infeasible / exceeded after redistribution | `portfolio_constructor.py` |
| **F26** sample-data / METRIC_COLS drift (the 1 baseline fail) | MED | Added 11 missing keys to `_generate_sample_data` | `factor_engine.py` |
| **F27** tautological / swallowed test assertions | MED | Removed swallows; fixed malformed fixtures; assert on computed output | `tests/test_defensibility_improvements.py` |
| **F12** doc drift (17 vs 44 metrics; min-max vs rank) | HIGH | README rewrite; SUPERSEDED banners; FORENSIC min-max correction; OVERVIEW canonical | `README.md`, `INSTITUTIONAL_AUDIT_REPORT*.md`, `FORENSIC_AUDIT_REPORT.md`, `SCREENER_OVERVIEW.md`, `SCREENER_DEFENSIBILITY_SPEC.md` |
| **F31/F6-doc** metric-count disagreement | MED | Canonical 44 (32+4+8) everywhere; env note | docs + `generate_dashboard.py` |
| **F37** `consecutive_beat_streak` mislabeled 0-4 | LOW | Corrected to 0-10 recency-weighted score | `config.yaml`, `schemas.py` |
| **F38** phantom `rebalance_frequency`/`gics_level` | LOW | Marked NOT ENFORCED | `config.yaml` |

New tests: `tests/test_governance.py` (14), `tests/test_portfolio_risk.py` (10).

---

## Track B — owner-approved ranking fixes (commit `4527782`)

| Finding | Sev | Fix | Files |
|---|---|---|---|
| **F1** composite = pure percentile rank, cardinality destroyed | CRIT | Cardinal weighted-average is the ranking key; percentile exposed as `Composite_Pct` | `factor_engine.py` |
| **F5** forward_eps_growth GAAP vs normalized EPS (45% of Growth) | HIGH | NaN when EPS ratio extreme (>2x / <0.3x) → weight redistributes | `factor_engine.py` |
| **F19** earnings_yield dual-definition fallback | MED | NaN instead of GAAP trailingEps/price fallback | `factor_engine.py` |
| **F20** revenue_cagr_3yr endpoint basis mismatch | MED | Annual-vs-annual endpoints (`totalRevenue_annual`) | `factor_engine.py` |
| **F21** sustainable_growth retention source | MED | Prefer cash dividendsPaid/NI over `.info` payoutRatio | `factor_engine.py` |
| **F22** operating_leverage EBIT sign flip + basis mismatch | MED | NaN on sign flip; annual-vs-annual EBIT & revenue | `factor_engine.py` |
| **F35** Sortino/MaxDD thin-history annualization | LOW | Require ≥200 daily obs (match volatility gate) | `factor_engine.py` |
| **F36** EBITDA reconstruction drops negative-sign D&A | LOW | Use `abs(_da)` | `factor_engine.py` |
| **F4** multicollinearity measured but never neutralized | HIGH | Optional Gram-Schmidt neutralization + effective-N diagnostic, config-gated **OFF by default** | `factor_engine.py`, `config.yaml` |
| **F17/F18** Risk redundancy / weight priors | MED | **Deferred** — need IC evidence; documented as priors | (docs) |

Golden file re-baselined deliberately (`tests/fixtures/golden_scores.parquet`): rank order **identical** on the 10-ticker fixture; composite now cardinal (70.20/58.07/57.27… vs old 100/90/80…). New tests: `tests/test_composite_cardinality.py` (5), `tests/test_metric_fixes.py` (6), `tests/test_neutralization.py` (4). Updated 8 tests that encoded the old contracts.

---

## Outputs & transparency (commit `6c2f47a`)

| Finding | Fix | Files |
|---|---|---|
| **F13** Excel had no disclaimers | ReadMe/Disclaimers as sheet 1 | `portfolio_constructor.py` |
| **F14** Composite_Confidence surfaced nowhere | Added as FactorScores column | `portfolio_constructor.py` |
| **F32** dashboard disclaimer only behind a click | Persistent footer disclaimer → Methodology | `generate_dashboard.py` |
| **F1/D7-7** outputs implied composite cardinality it lacked | Composite_Pct column; percentile-based composite coloring; clarified contribution/AI-prompt text | `portfolio_constructor.py`, `generate_dashboard.py` |

---

## Accepted / Deferred

- **F40** price_target_upside sign — ACCEPT (already deweighted to 12%; validate via live IC).
- **F42** backtest run-to-run reproducibility — MITIGATE (log cache manifest) — DEFER this cycle.
- **F25** decompose 3100-LOC files — DEFER (regression risk; guarded by golden test for a future cycle).
- **F17/F18** Risk-category redundancy & weight priors — DEFER (need IC evidence).

## Before/After portfolio comparison

Both snapshots are genuinely-live `--refresh` runs (Network OK, cache=0; baseline 824.7s, final 696.2s). "Before" = pre-remediation code; "After" = post-remediation.

**Headline:** 21/25 top names retained (Jaccard 0.72, 32% name turnover). The fixes refine the margins rather than upheave the selection. The most visible change is the **composite scale**: the old pure-rank ladder (100.0 / 99.8 / 99.6 …) is replaced by a **cardinal** score (77.8 / 75.1 / 74.3 …) that preserves conviction/magnitude.

| | Before (pure rank) | After (cardinal) |
|---|---|---|
| Top-6 Composite | 100.0, 99.8, 99.6, 99.4, 99.2, 99.0 | 77.8, 75.1, 74.3, 72.0, 70.2, 69.8 |
| #1 vs #6 gap | 1.0 (meaningless) | 8.0 pts (real edge) |

**Entered (4):** CAH, DLTR, EXE, FOXA  **Exited (4):** EA, MO, MU, TPR

Turnover reflects both the metric-definition corrections (F5 forward-EPS contamination NaN'd, F19 earnings-yield single-definition, F20/F22 annual-basis endpoints, F36 EBITDA abs) and minor data movement between the two runs. `Composite_Pct` preserves the old percentile view for anyone who wants it.

Full top-25 before/after table: see `REVIEW_CHANGELOG` git history / the run logs.

### Before (top-25, pre-remediation)
HST APA EXPE CF EIX NEM DECK BMY ACGL BBY INCY CINF TRV EXPD CNC HIG EG MU ALL JBHT EA MO TPR VLO NTAP

### After (top-25, post-remediation)
HST EXPE APA CF NEM EIX BMY DECK BBY ACGL INCY TRV CINF EXPD JBHT EG HIG CAH NTAP ALL FOXA EXE VLO CNC DLTR
