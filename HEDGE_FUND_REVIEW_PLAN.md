# Hedge-Fund-Grade Review & Remediation Plan

**Target:** Multi-Factor Stock Screener (`Screener-1`) — ~19,100 LOC, 8 factor categories, ~33 metrics, self-improving engine, 451+ tests.
**Mandate:** Review and critique the screener the way a top quant hedge fund's research/risk committee would — then **fix everything actionable**.
**Hard constraint (owner decision):** The data source stays **yfinance**. Do NOT plan a data-vendor migration. Everything else — methodology, statistics, risk model, portfolio construction, code quality, validation, reproducibility, the self-improving loop — is in scope for critique AND remediation.

---

## 0. Guiding Principles

1. **Separate the data-source ceiling from everything else.** Every prior audit caps the system at C+/B- *solely* because of yfinance. Since yfinance is fixed, the review's job is to find every point where the methodology, statistics, or engineering falls short of what is *achievable on yfinance data* — and close that gap. "It's yfinance's fault" is not an acceptable excuse for any finding that could be fixed in-code.
2. **Critique like a committee, not a linter.** Findings must be framed as a quant PM, a risk officer, and a research-quality reviewer would frame them — with the *investment consequence* stated, not just the code smell.
3. **Every finding gets: severity, evidence (file:line), investment/statistical consequence, and a concrete remediation.** No vague findings.
4. **Fix everything actionable within the yfinance constraint.** The review is not advisory-only; Phase 5 implements the fixes, re-runs, and updates all docs/outputs.
5. **Do not break ranking logic without explicit rationale + tests.** Per repo constraints: small, reviewable changes; every metric precisely defined; golden-file regression must pass or be deliberately re-baselined with justification.

---

## 1. Scope — What Gets Reviewed (7 Review Dimensions)

Each dimension is reviewed by a dedicated pass. This is the critique surface.

### D1. Factor Methodology & Metric Definitions
- All ~33 metrics across 8 categories: is each formula correct, economically sound, and correctly signed (`METRIC_DIR`)?
- Known-questionable metrics flagged by prior audits — re-verify against **current** `factor_engine.py` (not stale audit claims): ROIC (excess-cash + negative-pretax tax handling), Piotroski conditional weighting, forward-EPS fiscal-year alignment, sustainable-growth dividend source, Beneish M-Score min-data gate, accruals sign, operating leverage.
- Within-category metric weights (e.g., FCF 45%, forward_eps_growth 45%, analyst_surprise 38%): are the specific numbers justified anywhere, or arbitrary? (FORENSIC §C4 says undocumented.)
- GAAP vs normalized EPS mismatch; EV / fundamental time mismatch (real-time MC ÷ MRQ balance sheet).

### D2. Statistical Rigor & Factor Construction
- **Multicollinearity / double-counting:** factors are *surfaced* via an 8×8 correlation matrix but never neutralized. Effective dimensionality ~8-10 vs 33 nominal. Evaluate orthogonalization / PCA / factor-neutralization vs. the intentional "keep it simple" stance.
- **Naive weight combination:** heuristic category + metric weights, no IC-derivation. Critique the tradeoff (overfitting risk vs. leaving signal on the table). Assess whether the improvement engine's IC path is a sound answer or a half-measure.
- **Composite double percentile-ranking destroys magnitude** (#1-by-20pts and #1-by-0.1pts both = 100). Evaluate whether cardinality should be preserved.
- **IC analysis quality:** backtest IC is self-labeled BIASED (survivorship + look-ahead); live IC is unbiased but data-starved. 1-month horizon only (understates slow Value/Quality). Per-category only, no per-metric IC in the biased path.
- **Winsorization / percentile / small-sector fallback:** are the thresholds (1/99, <10-value fallback) defensible?

### D3. Risk Model & Portfolio Construction
- **No forward portfolio-risk / covariance model in the default path.** Default weighting is inverse-vol that *ignores correlations* (author concedes this at `portfolio_constructor.py:444`). Markowitz path exists but is non-default/experimental. Critique: is a covariance-aware risk model achievable on yfinance returns, and should it be the default?
- **No standalone turnover / rebalance analysis** for the live top-25 (only a 10bps cost inside the biased backtest). Turnover is a first-order cost driver — flag as a gap.
- Sector-cap logic, min-position guardrail, max-position redistribution, liquidity filter — correctness and edge cases.
- Risk category (Vol/Beta/Sharpe/Sortino/MaxDD) — are these the right stock-level risk factors; any redundancy (Vol~Beta, Sharpe~Sortino)?

### D4. Backtest & Validation Integrity
- Survivorship bias (current constituents used historically) + look-ahead bias (static fundamentals). These make **every** performance/IC figure derived from the backtest scientifically invalid. Options within yfinance: (a) point-in-time-ish reconstruction of index membership, (b) restrict backtest to momentum/risk-only where PIT holds, (c) demote the backtest to a clearly-labeled illustrative artifact and stop feeding its biased IC into anything that matters.
- Validate the data-quality harness: coverage filters, EV cross-validation, beta-overlap, channel-stuffing/EPS-basis/staleness flags, silent `except Exception` swallowing in `_stmt_val()`/fetch (FORENSIC gap #9 — a yfinance schema change silently NaNs metrics).

### D5. The Self-Improving Engine (governance & soundness)
- The engine can **auto-mutate `config.yaml`** (self-modifying factor weights), gated only by a 2% change threshold + observation counts. Critique this as a **model-governance risk**: is auto-apply appropriate? Is there drift/overfitting protection, an audit trail (`change_log.csv`), rollback, and a kill switch?
- Metric-evolution (activate candidate metrics after 12 consecutive positive-IC obs): is the activation criterion statistically sound or noise-chasing?
- Shrinkage (0.5), EWM half-life (6mo), regime nudges — are these principled?

### D6. Code Quality, Correctness & Engineering
- `factor_engine.py` is 3,127 LOC — violates the repo's own 800-LOC file guideline. Assess decomposition.
- Silent `except Exception` handlers, mutation vs. immutability (repo style rule), error handling at boundaries, config parameters *declared but not enforced* (V2 Appendix B: `sector_cap_multiplier`, `min_position_pct`, `rebalance_frequency`, `max_missing_metrics`, `gics_level` always Sector).
- Determinism / reproducibility (`RunContext`, golden files), the Windows CP1252 / SSL-interception fragility already seen.
- Test *quality* not just count: are the 451 tests asserting correctness or just non-crash? Coverage gaps in scoring edge cases.

### D7. Outputs, Transparency & Defensibility
- Do the Excel workbook, dashboard, `SCREENER_OVERVIEW.md`, and disclaimer blocks accurately represent the methodology and its limitations to an end user?
- Is the "Composite_Confidence" score meaningful? Are caveat flags surfaced where a user actually sees them?
- Documentation drift: multiple audit docs describe different model versions (17-metric vs 36-metric). Reconcile to a single source of truth.

---

## 2. Execution Phases

### Phase 1 — Establish Ground Truth (baseline)
- Confirm the codebase is in a known-good state: run the full test suite (`pytest`), record pass count and any failures.
- Run the screener once (fresh yfinance data via the working CA bundle) to produce a **baseline** `factor_output.xlsx`, dashboard, and `runs/<id>/` artifacts. This is the "before" snapshot for measuring the impact of fixes.
- Snapshot current factor weights + golden files.

### Phase 2 — Multi-Perspective Critique (the review itself)
Run the 7 review dimensions (D1–D7). Use **parallel independent reviewers**, one per dimension, each producing a findings list. Then a **synthesis/adversarial pass** that (a) de-duplicates, (b) reconciles findings against the *current* code (kill stale audit claims), (c) severity-ranks, and (d) assigns each finding a fix disposition:
- `FIX` — actionable within yfinance, will be implemented in Phase 4.
- `MITIGATE` — can't fully fix (yfinance-limited) but can be reduced/flagged/disclaimed.
- `ACCEPT` — a deliberate design tradeoff to document, not change.
- `DEFER` — valid but out of scope / too large for this cycle.

**Deliverable:** `HEDGE_FUND_REVIEW_FINDINGS.md` — the committee-style critique report with an executive summary, a letter-grade assessment (with and without the yfinance ceiling), a severity-ranked findings table, and the fix dispositions.

### Phase 3 — Remediation Plan & Owner Gate
- From the FIX/MITIGATE findings, produce an ordered remediation backlog (severity × effort × ranking-risk).
- **Explicitly separate ranking-affecting changes from non-ranking changes.** Ranking-affecting changes (new neutralization, weight-combination changes, composite cardinality, risk-model default) require a stated rationale and will re-baseline golden files deliberately.
- **STOP for owner approval on any change that alters ranking logic** (per repo constraint). Non-ranking fixes (code decomposition, error handling, config enforcement, docs, turnover reporting, governance guardrails) can proceed without gating.

### Phase 4 — Implement the Fixes ("update everything")
Apply the approved remediations, TDD-style where logic changes (write/adjust tests first). Likely workstreams:
- **Statistics:** optional factor-neutralization/orthogonalization step (config-gated, off by default unless approved); preserve composite magnitude alongside the percentile; per-metric IC in the live path; extend IC horizons (1/3/6-month).
- **Risk/Portfolio:** covariance-aware weighting option and honest turnover report for the live top-25; enforce declared-but-ignored config params.
- **Backtest integrity:** demote biased IC from any live decision path; add prominent invalidity labeling; restrict PIT-unsafe categories or add best-effort membership reconstruction.
- **Governance:** tighten the self-improving engine (kill switch, drift caps, mandatory human-approval mode, richer audit trail).
- **Engineering:** decompose `factor_engine.py`, replace silent `except Exception` with typed handling + logging, immutability cleanups, harden the Windows/SSL fragility.
- **Correctness:** fix any metric-definition bugs surfaced in D1.
- After each logic change: run `pytest`; keep the suite green.

### Phase 5 — Re-run, Re-validate & Update All Outputs
- Re-run the full screener with fresh yfinance data (CA bundle env vars set).
- Regenerate `factor_output.xlsx`, dashboard, `dashboard_data.js`, `SCREENER_OVERVIEW.md`, validation CSVs, `runs/<id>/` artifacts.
- Re-run full `pytest`; re-baseline golden files with documented justification if ranking changed.
- **Update all documentation** to a single reconciled source of truth: update/supersede the stale audit docs, update `SCREENER_DEFENSIBILITY_SPEC.md`, `SCREENER_OVERVIEW.md`, and write a `REVIEW_CHANGELOG.md` mapping each finding → fix → file(s) changed.
- Produce a **before/after comparison**: portfolio turnover from the changes, top-25 diff, any grade-relevant improvements.

### Phase 6 — Commit, Push & Deploy
- Commit in logical chunks (conventional commits): review report, then remediation commits grouped by workstream, then the refreshed data/docs.
- Push `master:main`; deploy the dashboard to GitHub Pages via `deploy_dashboard.sh` (CA bundle env vars set).
- Verify the live site serves the updated data.
- Final summary: grade before/after (both framed against the fixed yfinance ceiling), findings fixed vs. deferred, commit hashes, dashboard URL.

---

## 3. Deliverables Checklist
- [ ] `HEDGE_FUND_REVIEW_FINDINGS.md` — the committee critique (exec summary, grades, severity-ranked findings + dispositions).
- [ ] `REVIEW_CHANGELOG.md` — finding → fix → files mapping.
- [ ] Updated methodology docs (SPEC, OVERVIEW; stale audits reconciled/superseded).
- [ ] Code + tests implementing all FIX-disposition findings; `pytest` green.
- [ ] Refreshed `factor_output.xlsx`, dashboard, validation CSVs, `runs/<id>/`.
- [ ] Before/after portfolio + turnover comparison.
- [ ] Committed, pushed to `main`, deployed to GitHub Pages, live site verified.

## 4. Constraints & Guardrails (from repo + owner)
- **yfinance stays** — no vendor migration.
- **No ranking-logic change without owner approval** and tests.
- Small, reviewable, well-justified changes; every metric precisely defined and sourced.
- Keep the test suite green; re-baseline golden files only deliberately.
- Set CA-bundle env vars (`CURL_CA_BUNDLE`/`SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE`) before any run — this machine's Avast TLS interception otherwise forces a silent stale-cache fallback.

## 5. Key Risks
- **HIGH — Ranking drift:** neutralization / weight / composite changes can materially reshuffle the top-25. Mitigate: owner gate, golden-file diffing, before/after report.
- **MEDIUM — Backtest can't be fully de-biased on yfinance** (no true PIT / historical membership). Mitigate: label as illustrative, stop feeding biased IC into live decisions.
- **MEDIUM — `factor_engine.py` decomposition** could introduce regressions. Mitigate: TDD, golden files.
- **LOW — Environment fragility** (SSL/CP1252). Mitigate: already-known workarounds documented.

## 6. Estimated Complexity: HIGH
Review (Phase 2) is bounded; remediation (Phase 4) scales with how many FIX findings the owner approves. Ranking-affecting fixes are the long pole and are owner-gated.
