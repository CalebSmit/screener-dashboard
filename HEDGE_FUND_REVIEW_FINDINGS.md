# Hedge-Fund-Grade Review — Findings Report

**Target:** Multi-Factor Stock Screener (`Screener-1`) — 8 factor categories, 44 metrics in `METRIC_COLS`, self-improving engine, 451 tests.
**Review date:** 2026-07-28
**Method:** 7 independent dimension reviewers (D1–D7) run in parallel against the *current* code, followed by an adversarial synthesis pass that de-duplicated findings, reconciled them against live evidence, killed stale audit claims, severity-ranked, and assigned each a disposition.
**Data-source constraint:** yfinance is fixed by owner decision. No finding below blames yfinance for anything fixable in code; the grade is stated both *with* and *without* the yfinance data ceiling.

---

## 1. Executive Summary

The screener is **materially better than every prior audit doc claims.** The bulk of the historically-flagged methodology bugs (ROIC excess-cash and negative-pretax tax handling, Piotroski conditional weighting, Beneish min-data gate, accruals sign, LTM/MRQ time-basis split, EV cross-validation, 6-month/12-1 momentum skip-month) are **verified correctly implemented in current code.** The engineering fundamentals a quant pipeline lives or dies on — determinism (SHA-256 config hash, seeded RNG, sorted golden compare), immutability (zero `inplace=True`), a real value-comparing golden regression test, and a genuinely institutional-grade data-quality harness (coverage filter, EV cross-validation, beta-overlap gating, channel-stuffing/staleness flags) — are present and sound. Critically, we **verified the biased backtest IC does *not* feed the live improvement engine** (the one merge function that would do so is dead code) — killing the single scariest contamination hypothesis.

That said, the review surfaced **four issues a hedge-fund risk committee would treat as blocking**, none of which is a yfinance limitation:

1. **The final Composite is a pure percentile-rank transform** (`factor_engine.py:2445`) that overwrites the cardinal weighted-average score. A stock beating the field by 20 points and one beating it by 0.1 both map to 100.0. This is why the live top-25 is spaced exactly 100.0 / 99.8 / 99.6… (= 100/N). All conviction/magnitude information is destroyed before it reaches portfolio construction, and several outputs then present this rank as if it were cardinal. **(D2-1 / D3-5 / D7-7)**
2. **The self-improving engine can auto-rewrite production factor weights with no human in the loop**, gated only by an observation *count* (not statistical significance), and its one apparent kill switch (`improvement.enabled`) is **never read anywhere in the code** — it is a no-op. Auto-apply is not *reachable* today only because the live IC history has 3 rows; it becomes live the moment ~24 snapshots accumulate. **(D5-1 / D5-2)**
3. **~44 nominal metrics collapse to ~8–10 effective dimensions.** Multicollinearity is *measured* (an 8×8 Spearman matrix) but *never acted on* — no orthogonalization, no pruning, no effective-N. Valuation alone holds ~6 collinear "cheapness" metrics; the Risk category holds two redundant pairs (Vol~Beta, Sharpe~Sortino). The realized factor bet is far more concentrated than the nominal weights imply, and risk is understated. **(D2-2 / D3-6)**
4. **Documentation is fractured into contradictory sources of truth.** Three audit reports plus the README describe a superseded 17-metric, 6-category, "C+" model; even the two *current* docs disagree on the metric count (33 vs 34 vs 36 vs the actual 44). One audit actively misdescribes the composite as min-max scaling. Anyone doing diligence would draw materially wrong conclusions about scope, weights, and structure. **(D7-5 / D7-6 / D1-6)**

Below-blocking but real: no covariance-aware portfolio risk model and no disclosure of its absence (D3-1/D7-2); no turnover/rebalance-cost reporting for the live portfolio (D3-3); several time-basis mismatches inside individual metrics, the worst being GAAP-vs-normalized EPS contamination in the 45%-weighted `forward_eps_growth` (D1-1); silent suppression of schema-drift warnings (D4-3); stale-data flagged but never excluded (D4-4); and the `Composite_Confidence` score being computed but surfaced in zero outputs (D7-1).

### Letter Grade

| Framing | Grade | Rationale |
|---|---|---|
| **With the yfinance data ceiling** (i.e., "how good can this be on retail-API data?") | **B / B+** | Sound engineering, correct metric math, honest DQ harness. Held back from A- by the composite-rank cardinality loss, the ungoverned self-modifying engine, unactioned multicollinearity, and doc drift — all of which are fixable *without* better data. |
| **Without the yfinance ceiling** (graded as an absolute institutional process) | **C+ / B-** | The data source caps factor breadth, point-in-time integrity, and backtest validity below what an institutional shop (Compustat/CRSP PIT + I/B/E/S) would require. This ceiling is the owner's accepted constraint. |

The prior audits' flat "C+" reflected the *data ceiling*, not the code. On the axis the owner asked us to grade — **execution quality achievable on yfinance** — the current build earns a **B/B+**, and the FIX backlog below is what closes the remaining gap to A-.

---

## 2. Severity-Ranked Findings with Dispositions

Disposition key: **FIX** (implement now, in scope) · **MITIGATE** (can't fully fix within yfinance; reduce/flag/disclaim) · **ACCEPT** (deliberate tradeoff to document) · **DEFER** (valid but out of scope this cycle).
"Ranking?" = does the change alter the Composite score / top-25 selection (→ owner approval required per repo rule).

| # | Finding | Sev | Ranking? | Disposition |
|---|---|---|---|---|
| **F1** | Composite overwritten by pure percentile rank — cardinality destroyed (D2-1/D3-5/D7-7) | **CRITICAL** | **YES** | **FIX** (owner-gated) |
| **F2** | Self-improving engine auto-applies weight changes; `improvement.enabled` kill switch is a no-op (D5-1) | **CRITICAL** | YES* | **FIX** (governance) |
| **F3** | Auto-apply "confidence" = observation count, not significance (D5-2) | **CRITICAL** | YES* | **FIX** (governance) |
| **F4** | Multicollinearity measured but never neutralized; ~44 metrics → ~8–10 effective (D2-2) | **HIGH** | **YES** | **FIX** (owner-gated, config off by default) |
| **F5** | `forward_eps_growth` mixes GAAP trailing vs normalized forward EPS; 45% of Growth (D1-1) | **HIGH** | **YES** | **FIX** (owner-gated) |
| **F6** | Weight mutation driven by 1-week IC via silent fallback; strategy is monthly (D5-3) | **HIGH** | YES* | **FIX** (governance) |
| **F7** | Candidate-metric activation = 8 simultaneous tests, no multiple-comparisons control (D5-5) | **HIGH** | YES* | **FIX** (governance) |
| **F8** | Improvement-engine sample far below its own gates; not yet trustworthy (D5-6) | **HIGH** | YES* | **FIX** (governance) + MITIGATE |
| **F9** | IC is single-horizon (1-month); understates slow Value/Quality (D2-4) | **HIGH** | NO (indirect) | **FIX** |
| **F10** | No covariance/correlation portfolio risk model in default path (D3-1) | **HIGH** | NO | **FIX** (reporting) + **ACCEPT** (default weighting) |
| **F11** | No standalone turnover / rebalance-cost report for live top-25 (D3-3) | **HIGH** | NO | **FIX** |
| **F12** | Doc drift: 3 audits + README describe superseded model; counts disagree; one says min-max (D7-5/D7-6) | **HIGH** | NO | **FIX** |
| **F13** | Excel workbook carries no disclaimer/limitations on any sheet (D7-3) | **HIGH** | NO | **FIX** |
| **F14** | `Composite_Confidence` computed but in zero user-facing outputs (D7-1) | **HIGH** | NO | **FIX** |
| **F15** | `warnings.filterwarnings("ignore")` suppresses schema-drift NaN alerts (D4-3) | **MEDIUM** | YES (silent) | **FIX** |
| **F16** | Stale financials (>120d) flagged but never excluded or discounted (D4-4) | **MEDIUM** | **YES** | **FIX** (owner-gated) + MITIGATE |
| **F17** | Risk category redundancy: Vol~Beta, Sharpe~Sortino = 65/100 pts (D3-6) | **MEDIUM** | **YES** | **DEFER** (owner-gated; needs IC evidence) |
| **F18** | Within-category weights asserted "principled" but undocumented/arbitrary (D1-7) | **MEDIUM** | **YES** | **MITIGATE** (document as priors) |
| **F19** | `earnings_yield` silent fallback to GAAP EPS/price = two definitions in one rank (D1-2) | **MEDIUM** | **YES** | **FIX** (owner-gated) |
| **F20** | `revenue_cagr_3yr` mixes LTM current vs annual 3yr-ago endpoints (D1-3) | **MEDIUM** | **YES** | **FIX** (owner-gated) |
| **F21** | `sustainable_growth` prefers payoutRatio over cash dividends, inconsistent w/ ROE (D1-4) | **MEDIUM** | **YES** | **FIX** (owner-gated) |
| **F22** | `operating_leverage` sign-broken across EBIT zero-crossing; LTM/annual mismatch (D1-5) | **MEDIUM** | **YES** | **FIX** (owner-gated) |
| **F23** | Markowitz path mis-specified as risk model (no return term, raw cov, 40% cap) (D3-2) | **MEDIUM** | NO | **FIX** |
| **F24** | `max_position_pct` silently breached when portfolio has <20 names (D3-4) | **MEDIUM** | NO | **FIX** |
| **F25** | `factor_engine.py` 3127 LOC / `run_screener.py` 2096 LOC vs 800 ceiling (D6-1) | **MEDIUM** | NO | **DEFER** (large; regression risk) |
| **F26** | Sample-data/`METRIC_COLS` drift = the 1 failing baseline test (D6-2) | **MEDIUM** | NO (offline only) | **FIX** |
| **F27** | Tautological / `try/except:pass`-wrapped test assertions (D6-3) | **MEDIUM** | NO | **FIX** |
| **F28** | Shrinkage 0.5 / EWM 6mo / regime 0.10 are unvalidated magic numbers (D5-7) | **MEDIUM** | YES* | **MITIGATE** (document) + turn regime off by default |
| **F29** | Change-log audit trail absent + omits provenance fields (D5-8) | **MEDIUM** | NO | **FIX** (governance) |
| **F30** | No cumulative drift cap / post-change feedback / circuit breaker (D5-9) | **MEDIUM** | YES* | **FIX** (governance) |
| **F31** | Current docs undercount metrics (33/34/36 vs 44) incl. dashboard AI prompt (D7-6) | **MEDIUM** | NO | **FIX** |
| **F32** | Dashboard disclaimers only behind a click; default view has none (D7-4) | **MEDIUM** | NO | **FIX** |
| **F33** | Forensic flags visible only in top-10 Excel sheet, hidden for stocks 11–500 (D7-8) | **MEDIUM** | NO | **FIX** |
| **F34** | Config `winsorize_percentiles` is dead (scoring call hardcodes 1/99) (D2-5) | **LOW** | NO (matches default) | **FIX** |
| **F35** | Sortino/MaxDD use sqrt(252) annualization on thin history vs vol's ≥200d gate (D1-10) | **LOW** | YES (subset) | **FIX** (owner-gated) |
| **F36** | EBITDA reconstruction gates on `_da>=0` then falls back to distrusted reported EBITDA (D1-12) | **LOW** | YES (small) | **FIX** (owner-gated) |
| **F37** | `consecutive_beat_streak` is 0–10 score but config/schema say "0-4 count" (D1-6) | **LOW** | NO | **FIX** (doc) |
| **F38** | `rebalance_frequency` & `gics_level` config params never read (phantom) (D6-4/D6-5) | **LOW** | NO | **FIX** |
| **F39** | 5 silent `except:pass` in auxiliary I/O paths (D6-7) | **LOW** | NO | **FIX** |
| **F40** | `price_target_upside` higher=better is economically dubious (already deweighted) (D1-8) | **LOW** | YES (small) | **ACCEPT** (validate via IC) |
| **F41** | `factor_engine.main()` legacy coverage filter penalizes banks (D4-5) | **LOW** | YES (legacy path) | **FIX** |
| **F42** | Backtest not reproducible run-to-run (cache-date dependent) (D4-6) | **LOW** | NO | **MITIGATE** (log cache manifest) |

\* Governance findings (F2/F3/F6/F7/F8/F28/F30) are "ranking-affecting" only in the sense that they govern *whether/when the engine mutates weights.* Adding guardrails, a real kill switch, and human-approval-only mode does **not** change the current ranking — it prevents *future* un-owner-approved ranking drift. Per the repo rule these are therefore safe to implement now (they make the model *more* conservative), and I treat them as non-gated.

---

## 3. Killed Stale Claims (do not re-litigate)

Verified against current code; these prior-audit findings are **already fixed**:
- ROIC excess-cash deduction (`factor_engine.py:1355-1366`) and negative-pretax → 0% tax (`1343-1349`).
- Piotroski conditional weighting + growth-trap variant, correctly gated & signed (`2205-2276`); raw 0–9 not proportionally normalized (`1467-1473`).
- Beneish M-Score ≥5-of-8 min-data gate, bank-excluded (`524-525`, `1481`); sign correct.
- Accruals `(ni-ocf)/ta`, lower=better (`1476`) — correct per Sloan 1996.
- LTM (4-quarter sum) flows vs MRQ balance sheet, annual fallback (`668-776`).
- EV cross-validation vs computed MC+Debt−Cash with `_ev_flag` (`1257-1273`).
- Momentum skip-month: `return_12_1`/`return_6m` skip recent month; full `return_12m` reserved for Sharpe/Jensen (`1626-1687`).
- **Biased backtest IC does NOT feed the improvement engine** — `analyze_ic_trends` reads live IC only; `merge_historical_and_live_ic` is dead code (D4-2).
- Config phantoms `sector_cap_multiplier`, `min_position_pct`, `max_missing_metrics`, `peg_max_cap` — already removed (D6-6). Only `rebalance_frequency`/`gics_level` remain (F38).
- Determinism, immutability, Windows CP1252 handling, liquidity filter on live path — all sound (D6-8, D3-7).

Environment note (new): the documented 4-env-var SSL workaround no longer works under Python 3.13 (`combined_ca.pem`'s Avast root has non-critical Basic Constraints, rejected by stdlib `ssl` and Windows schannel-curl). The **only** var that matters for the live data path is `CURL_CA_BUNDLE`, because yfinance 0.2.66 uses `curl_cffi`, whose libcurl accepts the bundle. This should be documented (see F12/F38 doc reconciliation).

---

## 4. What Goes to Owner Gate vs Proceeds Now

**Requires owner approval (alters Composite/top-25):** F1, F4, F5, F16, F19, F20, F21, F22, F35, F36 — plus deferred F17.

**Proceeds now (non-ranking: governance guardrails, reporting, code, docs, error-handling):** F2, F3, F6, F7, F8, F9, F10 (reporting portion), F11, F12, F13, F14, F15, F23, F24, F26, F27, F28, F29, F30, F31, F32, F33, F34, F37, F38, F39, F41, F42.

The Phase 3 remediation backlog (separate section) orders these by severity × effort × ranking-risk and is presented for approval before any ranking-affecting code is written.
