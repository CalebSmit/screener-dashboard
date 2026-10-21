# Morning Brief - Wednesday 21 October 2026, 02:26

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-10-21T02:00:03.974680 |
| Stocks scored | 501 |
| With a price | 501/501 |
| With an analyst target | 497/501 |
| Top 5 | EXPE, APA, INCY, HST, VLO |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (33 rows, but overlapping windows are not independent; 89 rows across all horizons), newest 2026-10-13 |
| Context signals | Context signals: 10 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 425/426 within 0.5%; operating cash flow 407/411 within 0.5%; net income 398/421 within 0.5%; market value 447/487 within 5%; filings cached 5 days ago |
| Top 25, worth a look | EXPE (#1: equity 4.2% of assets); HST (#4: net income 1.46x after-tax operating income); CF (#11: net income 16% off the filing); MU (#14: a period behind its filings) |
| Context coverage | options usable 492/501 (9 quotes-closed) - insider from SEC 501/501 - earnings variability 443 |

## Waiting for a decision

A session researched these, proposed a change and left it for review. The next
session decides and records why; you can also answer in `OWNER_FOCUS.md`.

- **score each company once: in `run_screener.py`, after `compute_metrics` and before `compute_sector_percentiles`, group universe rows by SEC CIK (falling back to identical `longName`), keep the line with the higher `avg_daily_dollar_volume` (S&P DJI's Designated Listing / FTSE Russell's pricing vehicle; on 2026-10-19 GOOGL, FOXA, NWSA) and remove the other lines from percentiles, scoring and ranking; publish them as unranked entries pointing to the company's ranked line with their own price and the price gap; shipped with a no-two-rows-per-CIK test, the page sentence and its claim, the generated methodology sentence and a changelog entry (section 9). Measured on the 2026-10-19 run: top 25 unchanged; FOXA 47 -> 28, GOOGL 226 -> 197, NWSA 61 -> 52; other Communication Services stocks +1.56 composite points on average** - open 2 days - `research/2026-10-19-share-classes.md`
- **no weight or metric change; state the bank P/B-ROE pairing as the design and guard it: (1) a paragraph in `run_screener.build_screener_overview()`'s "Bank-specific scoring" section saying a bank's ROE is its earnings yield times its P/B, so its Valuation and Quality scores are read together - the 44 points rank it on ROE relative to P/B (to first order, its earnings yield) and on equity per dollar of assets, with no net weight on P/B by itself - registered in `claims.py` as `overview.bank_pb_roe_identity`; (2) the two table cells calling P/B and ROE "THE key" bank metrics reworded to say each is read against the other; (3) `tests/test_bank_pb_roe_identity.py`, in that claim's `checked_by`, failing if the first-order net loading on log P/B computed from `config.yaml` (factor weights x bank weights: -P/B + ROE + ROA, in composite points; 0.0 today) leaves +-1.5 points, or if on the published payload log ROE regressed on log earnings yield and log P/B has R-squared below 0.90 across the bank-like stocks (0.978 today); (4) a `config.yaml` comment above `bank_metric_weights` naming that constraint and this note.** - open 1 day - `research/2026-10-20-bank-quality-valuation-overlap.md`

## What changed in the repo

- `ab5c87e data: screener run 2026-10-21 - 501 scored, top: EXPE APA INCY HST VLO`
- `a937e34 OWNER_FOCUS: Tiingo's free plan is personal use only - nothing derived from it is published until Tiingo confirms in writing`
- `b6003ad research: Tiingo's public ticker list carries 84 of 152 acquired departed members - most of the survivorship hole`
- `0437184 OWNER_FOCUS: the tests that need runs/ should skip loudly in a fresh worktree`
- `395d5f8 log: 2026-10-20 owner-run night - nine agents, four batches`
- `534433a lab: experiment 15, Piotroski (2000) on Open Source Asset Pricing - 1 of 4 within`
- `24a4194 page: regenerate after experiment 16 - the methodology links the new lab experiment`
- `8cd9ce7 lab: experiment 4b - size among large companies, profitability held fixed (Asness et al. 2018): 10 of 17 within; the note's rule not refuted`
- `12e3b7b OWNER_FOCUS: the mutation audit's first round`
- `9ebdaeb tests: claims mutation audit - 38 mutations on 22 claims, 15 uncaught, all 15 now caught`
- `bb31e09 OWNER_FOCUS: experiment 15's second reading is done`
- `5758f80 lab: second reading of experiment 15 (JKP 2023) - all ten figures hold; keep the licensor's copyright notice`
- `39a6727 lab: experiment 4b tolerances and the note's refutation rule, written before the comparison (Asness et al. 2018)`
- `a8fa5d6 OWNER_FOCUS: queue the two free v2 data fixes and the extended-universe build`
- `e0109c8 OWNER_FOCUS: fine-tune and test what the 2026-10-20 night built (owner direction)`

## The session's own account

> 2026-10-20 (owner-run, night) - nine agents' work merged in four batches: three lab experiments, three research notes, a test-the-tests audit, 0.11 closed
> 
> **Health (rule 8):** last code session ran? **yes** - 13:00 shipped to main. Data loop published?
> **yes** - 02:00 2026-10-20, HEALTH: PASS. Evidence base at `1m` = **33 rows, newest 2026-09-18 (32 days,
> bound 40), 4 effective**. Priority 0: `allow_auto_apply` **false**, 4 of 8. Top open roadmap item:
> **0.11 is CLOSED tonight**, so the oldest open item is **0.13, the Factor Lab** (opened 2026-10-12, **age 8
> days**), worked tonight (experiments 15-17).
> 
> **Owner:** *"do as much work as you can tonight before the 2 am session please! Then have the sessions
> fine tune things and test things."* Answered with two waves of agents in isolated worktrees, each
> merged only after its own full suite, then batch-gated here (tags `good/2026-10-20-owner-2` .. `-5`).
> 
> ### Shipped
> - **Engine calibration test** (`tests/test_engine_calibration.py`, 7): planted IC differences of
>   0.03/0.05/0.10 come back within 4 SE through the engine's own `per_month`/`summarise`; 0.10 clears the
>   bar held out; no false alarm on ten null pairs; twice `detectable` always caught; a 70%-coverage screen
>   is measured on the shared stocks. Test-only - the ledger is untouched.
> - **Size: the weight stays at 5%** (`research/2026-10-20-size-weight.md`): no unconditional size premium
>   among large companies, but a within-sector small tilt beside Quality is documented practice (MSCI,
>   FTSE Russell, Dimensional) and Size is one of the counterweights holding the ranking growth-neutral.
>   **Lab experiment 16 tested the note's pre-registered rule and did not refute it** (0.166% a month, t
>   2.36, against 0.136% and 1.5) - with a thin margin (0.030, t 1.24) and negative since 2013. The page's
>   "small-cap premium" question line and the Size column's tooltip were overstatements; both reworded
>   at their generators.
> - **Bank Quality vs Valuation** (`research/2026-10-20-bank-quality-valuation-overlap.md`): ROE = EY x P/B
>   is not a double count - the 44 points net to earnings yield and equity per dollar of assets with no
>   net P/B weight, the deliberate "high ROE at a low P/B" bank screen. **OPEN proposal**: state the
>   pairing on the page and guard the zero net P/B loading with a test.
> - **Free survivorship sources** (`research/2026-10-20-free-survivorship-sources.md`): free data closes
>   **37%** of v2's 2012-2019 survivorship gap with monthly returns (the members that left but kept
>   trading), **1.2% of acquisitions** (half the gap). The Size test on free data can falsify the
>   survivorship explanation and cannot confirm it. Two free v2 data fixes found (pre-2014 renames, 167
>   member-months; predecessor CIKs, 773) - queued.
> - **Context layer 0.11 CLOSED**: a false live sentence fixed (mid-session quotes labelled "previous
>   close" for 485 of 501 stocks after the 13:00 publish), a mid-session run now refused from the context
>   log (and 10-20's log restored), `context_eval` uses the generator's skew rule, five claims registered.
>   The one remaining item is a date: first one-month readings with the 2026-11-09 run.
> - **Factor Lab 15-17**: Jensen, Kelly & Pedersen 2023 (10 of 10; **CC BY-NC 4.0 - teaching only**;
>   second-read the same night, all ten figures hold, copyright notice added); size inside profitability
>   (10 of 17); Piotroski 2000 on Open Source Asset Pricing (1 of 4, a miss the decile cut explains). 17
>   experiments, **145 of 217** within. Piotroski was moved after 15 and 16 at merge so their published
>   numbers hold.
> - **Claims mutation audit** (`plan/claims-mutation-audit.md`): 38 mutations on 22 claims; **15 got past
>   the registered checks** - 8 untested boundaries, 3 checks reading the committed payload instead of the
>   code, 1 tolerance looser than its claim. All 15 now caught; the next 12 claims are listed.
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

