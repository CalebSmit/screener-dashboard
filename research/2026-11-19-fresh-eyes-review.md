# A fresh-eyes review of the screener, 2026-11-19 (owner-run)

**What this is.** The owner asked for a review by a reader who had not read the previous sessions'
notes and carried on: use the product the way its users would, then read the code behind it. This
note is that review. Findings are ranked by how much each matters to the product's credibility and
usefulness. Each one says what was **seen** (with evidence) separately from what is **suspected**,
the fix, its rough size, and whether anything in `DECISIONS.md` or `METHODOLOGY_CHANGELOG.md` argues
against it.

**How it was done.** The live dashboard at 1440px (Chrome) and the committed page at 375px (local
preview of the same files Pages serves), `research.html` and `lab.html` at 375px. Four read-only
reviewers in parallel - methodology, number tracing, the runners and gates, and the last two weeks
of process - and every finding below that came from one of them was re-checked by me before it went
in; the measurement scripts are in the session scratchpad and the figures are reproducible from the
committed `dashboard_data.js` (run `3e881b94506b`) and the previous day's (`3aecb1a`).

**Proposal status: OPEN** - findings 1, 2, 3, 4, 5, 6b, 7 and 8 propose changes for the owner or a later
session to decide; `OWNER_FOCUS.md` carries them as one item opened today.

---

## The short version

The screener's arithmetic is unusually trustworthy: every published score reproduces from the
payload, and the page is careful, honest and works on a phone. **Its weaknesses are elsewhere:**

1. Today's #1 stock is #1 because its price download failed. The machinery that is meant to stop
   that does not, and the page's own sentence saying it does was false. (Fixed in part tonight.)
2. The 15-year test says the ranking has **no measurable edge**. The home page leads with a bold
   "+10.5%" and never says that; no session in the last two weeks worked on what follows from it.
3. The publish gate now refuses good data more often than it catches bad data.
4. The process has grown so heavy that it is crowding out the product: ~65-90k tokens of reading
   before work starts, four lines of tests/prose/measurement for every line of product code, and no
   user-facing feature from a scheduled session in 15 days.

---

## 1. The #1 stock is ranked on six of eight categories because one download failed - SAW

**What I saw.** On the live page this morning SanDisk (SNDK) is #1 at 79.1, 5.7 points clear of #2,
with **Momentum "—" and Risk "—"**. In the run's raw fetch (`runs/3e881b94506b/00_raw_fetch.parquet`)
every price-history field for SNDK is empty (`price_latest`, `price_12m_ago`, `volatility_1y`, all
`_ctx_*`), while its company and statement fields are present. It is the **only** ticker with no
history in any of the last 20 runs except KVUE on 11-03/04. Yesterday SNDK was **#7 at 71.02** with
Momentum 75.5 and **Risk 1.03** (it is the most volatile stock in its sector). Putting yesterday's
two scores back into today's composite gives **71.02 → rank 7**; with both at a neutral 50 it is #4.

Two mechanisms let it through:

- `factor_engine.compute_composite` (factor_engine.py ~3979-3988) divides by the weight of the
  categories present. That is the same as filling a missing category with **the stock's own average
  of the others** (here ~79), not with a neutral 50. It is only "neutral" if the missing score would
  have been near the stock's own average - and a stock whose Risk score is 1 is exactly the case
  where it is not.
- The coverage discount counts missing **metrics**, not missing **weight**: 6 of 27 metrics is 23
  points of composite weight, and the discount was **0.33%**.

Every gate passed (HEALTH: PASS); SNDK is in `signals.json`'s top 40, which the trading bot reads.
The drilldown does say "Not scored: none of its 3 metrics has data", and its input-churn caveat says
8 metrics computable a month ago are not now - but nothing tells the reader that the **rank** is the
artefact.

**Why it happened.** `_fetch_single_ticker_inner` (factor_engine.py:1683) skips history silently
when `Ticker.history()` returns an empty frame - yfinance does not raise, it logs "possibly
delisted; no price data found". The run's retry pass (`run_screener.py`, "Retry pass") only retries
records with an `_error`, so an empty history was never retried. The 02:00 logs show yfinance
printing "no price data found" for several tickers on several days (e.g. 11-12: KVUE, PSKY, WBD, EA,
EQR, CTRA, BK, AVB in the batch close fetch), so this is not a one-off source behaviour.

**What the records say.** `METHODOLOGY_CHANGELOG.md` (2026-09, the split-scale guard) says
renormalisation "is neutral by construction - the stock neither gains nor loses from it" and
expected "a couple of points, not tens". Today is +8 points and six places. The generated
methodology page (`SCREENER_OVERVIEW.md`, coverage discount bullet) said the discount "prevents
stocks with sparse data from ranking artificially high". **Both statements are false for this
case.** Settled row 1.5 (split-scale guard) does not cover a history that is simply empty.

**Fixed tonight (engineering, no methodology change):**
- `run_screener.missing_price_history` / `merge_recovered_history`: after the error retry, any
  successful fetch with no price history is re-fetched once (bounded at 5% of the universe, beyond
  which it is an outage the health check judges). A re-fetch replaces a record only if it has a
  history. `tests/test_history_retry.py`.
- The methodology page's coverage-discount sentence now says what the discount does and does not do,
  with this case; the old sentence is in `claims.FORBIDDEN` with a negative control.

**Proposed (methodology - needs a research note and a changelog entry, not done tonight):**
(a) when a whole category is missing for a stock, score it at the sector-neutral 50 rather than the
stock's own average - the practitioner convention for a missing factor exposure is the cross-sectional
mean (z = 0), and the research note should source that properly; or (b) make the coverage discount
count missing weight. (a) is simpler to explain to a student ("no data = average") and removes the
incentive for missing data to help. (c) A health-gate *note* when a stock in yesterday's top 40 loses
a whole category. Size: S-M plus one research session. Nothing in `DECISIONS.md` decides against it;
the changelog's "neutral by construction" is the claim this evidence refutes.

## 2. The page leads with a return figure and never states the test result - SAW

**What I saw.** The hero's KPI tile shows **"+10.5%"** (top 25 vs +7.5% equal-weight, 9-month
record) in the largest numerals on the page, with a small caveat "Too short to tell skill from luck:
0.3 standard errors". The answer to "has it been tested?" is a link. Following it, `research.html`
opens with **"Not measurably. ... No edge has been shown."** - a monthly rank IC of −0.001 over 176
months (95% interval −0.018 to +0.015), and none of 27 alternative weightings better. The
methodology page's "If you read nothing else" line does say this (claim
`overview.at_a_glance_tested`); the home page, which is what a student reads, does not.

The owner queue also records that **the trading bot's own rule lagged an equal-weighted S&P 500 in
every period tested** (−0.72%/month 2020-23, t −2.7).

**Why it matters.** The product's stated value is credibility. A teaching tool whose most prominent
number is an unproven return, while its own test says no edge, is the one inconsistency a sceptical
club advisor will notice first. The research page itself is exemplary; it just is not where the eye
lands.

**Fix (presentation, S):** put the history test's one-line verdict on the home page beside the track
record - generated from the same `research/experiments/summary.json` as `history_test_line`, under
the same claim - and give the track record and the test equal visual weight. Not done tonight
because nightly sessions are barred from design work and the owner should choose the wording.

**The larger question (research, M-L), which no session in 15 days has worked on:** if the ranking
does not rank next-month returns on public data, what is it for? Honest candidates, each researchable:
a *screen* (narrowing 500 to 40 worth reading) rather than a forecast; a longer horizon (the engine
tests one month; value and quality are slow); a long-only top-quintile vs. the index rather than a
rank IC; or fewer, better factors. Universe expansion (5 sessions of "byte-identical" plumbing in
two weeks) scales a ranking that has not been shown to rank. **I would put this ahead of universe
expansion.** `DECISIONS.md` has no entry against it; the project memory "No proven edge yet" agrees.

## 3. The publish gate refuses good data more often than it stops bad data - SAW

**What I saw.** In the 02:00 and 18:00 data-run logs from 2026-11-02 to 11-19 (14 weekdays) the
published-claims gate refused **9** runs; the health check refused **0**. **Four weekdays never
published** (11-02, 11-09, 11-11, 11-12). Of the refusals I read, the failing tests were:
`test_dashboard_printed_figures.py::test_the_whats_changed_footnote_prints_a_hand_worked_window`
(11-11 18:00 and 11-16 02:00), `test_page_fourth_pass.py::test_phone_side_by_side_fits_three_and_four_stocks[4]`
(11-12 02:00), and per the logs and commits a lab card's phone budget, the Best Buy name clash and a
rank read from a list position. At most two were real defects on the page (the 11-02 growth-neutral
sentence and the 11-06 capped `price_target_upside` row).

The gate's membership is the curated list **plus every module any claim names**
(`scripts/check_published_claims.py`), which has grown to ~85 modules and ~2,370 tests - about half
the suite. `data-run.ps1` still describes it as "narrow on purpose".

**Why it matters.** Each refusal costs a day of evidence (the IC history accrues roughly one
independent `1m` observation a month), leaves the site and the bot a day stale, and sends the next
code session to fix a layout test instead of the product. CLAUDE.md now has four "sibling" rows
(`data->red`, `cache->red`, `clock->red`, `paint->red`), each a tripwire written after the last one
fired.

**Fix (M):** split the gate. **Blocking:** payload parse and contract, calculation reproducibility
(row 0.8), the advice ban, the evidence files, and claims about *how a number is computed*.
**Warn, with an ATTENTION line in the brief:** layout, phone budgets, footnote wording, lab cards.
`DECISIONS.md` (the 2026-10-02 and 2026-10-16 entries behind settled row "gates") argues for a gate
that runs the claims' own checks; it does not argue that a 2px phone overflow should discard a day's
data - and the gate's own comment says it should not. **The retrospective may not weaken a gate on
its own (CLAUDE.md); this is for the owner to approve.**

## 4. The routine is spending itself on itself - SAW

From the last two weeks (2026-11-05 to 11-19; `NIGHTLY_LOG.md` lines 19197-22459, 188 commits):

- **No scheduled session shipped a user-facing feature or moved a published score.** 17 scheduled
  sessions: 6 on inputs-vs-filings correctness (rigorous, each touching 1-27 stocks and no score),
  5 on universe-expansion and payload plumbing (each "byte-identical"), 4 repairing the routine's own
  refusals, ~1.5 on the Factor Lab, 1 on backtest diagnosis. Both Tuesday product days were diverted.
  Every product change in the window came from owner-run sessions.
- **Lines added in work commits:** research notes 6,419, tests 6,344, log/brief 2,492, production
  source 2,018. Since 11-07: ~4 lines each of tests, prose and measurement script per line of product
  code.
- **Reading load before any work:** `CLAUDE.md` 1,110 lines / 98.6 KB (+22% in 15 days),
  `OWNER_FOCUS.md` 183 KB (its Open section is 1,170 lines for six items; one item is ~650 lines),
  `prompts/nightly.md` 30 KB, plus the newest log entries - roughly **65-90k tokens, a third or more
  of the context window**. `NIGHTLY_LOG.md` is 1.57 MB and 22,459 lines and has never been archived.
- **The 2026-11-13 retrospective was lost** (`logs/nightly-2026-11-13_060001.log`: "Focus:
  RETROSPECTIVE" then "Required tool 'claude' is not on PATH. Aborting.") and no later entry mentions
  it. The last retrospective ran 2026-10-30.
- **An ACCEPTED build was silently dropped.** The implied-volatility inversion was decided ACCEPTED on
  2026-11-04 (`ef9bc75`), was the Next item on 11-05, and has not appeared in any Next list since; no
  code for it exists. CLAUDE.md calls an unmentioned deferral indistinguishable from an ignored request.
- About **16% of the 4,395 tests** test the runners, harness or process documents; at least ~160
  test functions pin literal text in `.ps1` files or Markdown (e.g. that CLAUDE.md contains a phrase).

**Why it matters.** The mandate is "measurably better every morning". The machinery is now mostly
measuring and defending itself. Defect-fixing that is safe and research-shaped is crowding out the
uncomfortable product questions (finding 2; "how much"; the sell workflow; priority 7, a student on a
phone, which got zero mentions in 3,300 log lines).

**Fixes (S each, owner's call where they touch rules):** archive `NIGHTLY_LOG.md` monthly and keep the
last 30 days; cap each `OWNER_FOCUS.md` Open item at ~20 lines and move history to `DECISIONS.md`;
collapse the four "->red" rows into one constraint with four test names; run a missed retrospective
at the next slot; have the brief list ACCEPTED proposals with no build commit (as it already lists
OPEN ones); let only a stalled data loop pre-empt a Tuesday product day.

## 5. Jensen's alpha is momentum counted twice, without the skip month - SAW (measured), research needed

`jensens_alpha` (25% of Momentum) is computed on `return_12m`, the full 12 months with **no skip
month** (factor_engine.py ~2886; the page says so). Its sector percentile correlates **+0.87**
(Spearman, 499 stocks) with `return_12_1`'s; `return_6m` only +0.42. So ~65 of Momentum's 100 points
are one 12-month return, and 25 of them include the most recent month - the month the project's own
`research/2026-10-09-short-term-reversal.md` (citing Da, Liu & Schaumburg 2014) says tends to reverse
in large caps. Nothing in `DECISIONS.md` decides the design; the 2026-10 changelog only corrected the
page's wording about alpha. **Fix:** a Monday research note on 12-1 alpha or residual momentum
(Blitz, Huij & Martens 2011). Not a change tonight.

## 6. A timed-out claims gate published as if pytest were missing - SAW, FIXED

`check_published_claims.py` caught `subprocess.SubprocessError` - which includes `TimeoutExpired` at
its 600 s limit - and returned exit 3, which `data-run.ps1` reads as "pytest unavailable: WARN and
publish". A page whose script never settled would have hung a browser test and been published with
`published_claims=warn` in `signals.json`. Now a timeout returns 1 (discard; the 18:00 retry exists
for a transient one). `tests/test_published_claims_gate.py::test_a_timed_out_run_refuses_rather_than_warns`.

## 6b. Two input defects on the page, found by tracing numbers - SAW, not fixed tonight

The number trace (AAPL, JPM, SNDK, HONA/FDXF; one metric per category from raw fetch to page)
reproduced every figure to 4-5 significant figures. The defects are in inputs and words:

- **The "reported and normalised EPS disagree" caveat does not measure that.** `_eps_basis_mismatch`
  (factor_engine.py ~2695) fires when Yahoo's `forwardEps / trailingEps` is above 2 or below 0.3.
  `forwardEps` is a next-fiscal-year consensus, so the flag fires on **expected growth**: SNDK is
  flagged with GAAP trailing EPS 125.12 against the consensus-basis last four quarters 118.37 (they
  agree); MU 74.33 vs 75.51; AVGO 7.83 vs 9.76. Meanwhile WDC (28.43 vs 13.05) and CINF (17.04 vs
  9.76) - the insurer the methodology page itself uses as the example of the two bases diverging -
  are **not** flagged. The stock sheet prints "Its reported and normalised EPS disagree, so
  earnings-based metrics carry more uncertainty" (`stock_summary.py` ~821, claim `summary.confidence`)
  on all 93 flagged stocks, and the home page's Diagnostics tile reads "93 EPS flags". **Fix (S, plus
  a research-ledger rerun because `factor_engine.py` is a `RESULT_PATHS` file):** compare
  `trailingEps` with the consensus-basis trailing figure (`_feg_b12`) where it exists, keep the old
  ratio only for the few stocks on the fallback basis, and re-word the KPI.
- **Berkshire's year-ago balance sheet is missing again** (`totalAssets_prior`, `totalEquity_prior`
  NaN for BRK-B in every run 11-13..11-19), so its Investment and ROE are unscored and sustainable
  growth falls back to ending equity - the three effects `METHODOLOGY_CHANGELOG.md` (2026-10-19,
  year-ago by date) records as fixed. Re-run `research/measurements/2026-10-19-year-ago-by-date.py`
  on its current statements. S.
- Sustainable growth's "ROE uses average equity" silently falls back to ending equity when prior
  equity is missing or negative (BA, MTD, STX at the 100% clamp, NFLX, BRK-B). Apply the bank ROE's
  rule (both balances positive, else not scored) or say so on the row. S.

## 7. Growth is labelled 13% and drives 1.9% of the ranking - SAW (measured)

Each category's share of composite variance on today's payload (w·cov(S,C)/var(C)): Valuation 34.1%,
Quality 29.1%, Momentum 15.9%, Revisions 7.2%, Risk 7.0%, Investment 3.1%, **Growth 1.9%**, Size 1.7%.
Growth correlates **0.07** with the composite (it is negatively correlated with Valuation −0.35 and
Investment −0.45). **The weight is deliberately unchanged** (Growth 13 REFUSED 2026-10-13; the
"counterweight" design in `DECISIONS.md`) and I am not arguing to change it. The finding is that the
page tells a student "13% of the final score", which is true of the formula and false of what moves
the ranking. **Fix (S):** publish each category's variance share beside its weight in the
methodology. In one line: the screener is ~63% a value-plus-quality screen.

## 8. Other machinery findings - SAW in code, not observed failing

- **Health check passes when it cannot read the published payload.** `check_run_health.published_payload()`
  returns `None` on any git/parse error and rank continuity becomes a *note*; `fold_shards` returns the
  payload unfolded if `generate_dashboard` cannot be imported - the route by which the 0/0 market-value
  comparison went unseen on 11-08. These should be `cannot_evaluate` (exit 2). S.
- **A failed data push can jam both loops** until fixed by hand (`data-run.ps1` push failure leaves
  the data commit on local `main`, the brief is then published to origin, and the next `pull --ff-only`
  stops). Not seen in retained logs. S, but rule 2's wording needs a line first.
- **Losing Chromium silently removes ~20% of the suite** (38 modules skip) and pytest still exits 0;
  no runner checks the skip count. S.
- **A runner that gives up on the repo lock still publishes the brief** in the other loop's tree. S.
- **Duplicated gates:** the payload parse checks are written twice in PowerShell and a 20-test module
  exists to keep the copies in step; the 40% fetch-failure gate cannot fire before the health check's
  2% unscored limit. Moving gate logic into one Python module both runners call would retire the
  text-pinning tests. M.

## 9. Smaller methodology observations - for the next synthesis session

- **Piotroski "halved" applies to 47% of stocks** (235 of 499, triggered at Valuation < 50) and moves
  a composite by at most ~1.75 points; Piotroski (2000) is about the cheapest book-to-market quintile.
  I found no note sourcing the 50 threshold or the 0.5 factor. Either source it or retire it - a
  special case covering half the universe is a second scheme to teach.
- Price-target upside (in Revisions) correlates −0.52 with the Risk score within sector; it rewards
  volatility. Add to the open question in `research/2026-10-26-price-target-upside.md`.
- Dead switches (`percentile_transform`, `factor_neutralization`, `sector_relative_composite`,
  `gics_level`, `max_position_pct`) make `config.yaml` harder to read than the live rule set is.
- Accruals ×3, sustainable growth vs ROIC, valuation-ratio overlap: known and deliberately unchanged
  (`DECISIONS.md` 0.12 and the cited notes). No new evidence.

## What I looked at and found sound

The calculation trace (every category and composite reproduces from the payload); the advice-language
ban and its scoping; mobile layout at 375px (no horizontal scroll; the sheet and both side pages read
cleanly); the research page's honesty; the drilldown's labelling of unscored categories; the claim
register. The stock sheet is long (12,500 characters for SNDK) and dense with citations - good for an
analyst, heavy for a first-year student; a "short version" toggle is worth considering under priority 7.

## Fixed tonight

| # | Fix | Files | Test |
|---|---|---|---|
| 1 | Re-fetch a successful fetch whose price history came back empty | `run_screener.py` | `tests/test_history_retry.py` (5) |
| 1 | Methodology page no longer claims the coverage discount prevents sparse-data rankings; says what it does | `run_screener.py` (generator), regenerated `SCREENER_OVERVIEW.md`, page rebuilt | `claims.FORBIDDEN` + negative control in `tests/test_claims_register.py` |
| 6 | Claims-gate timeout refuses instead of warn-and-publish | `scripts/check_published_claims.py` | `tests/test_published_claims_gate.py` (+1) |
| - | The health line read the 18:00 retry's "already completed, nothing to do" log as a failed publish (any session after 18:00 would have logged "Data loop published? **no**") | `scripts/health_numbers.py` | `tests/test_health_numbers.py` (+2) |

No score, weight, metric or rank changes from these. SNDK will be scored on its price history again
on the next run in which Yahoo returns it, with or without the retry.
