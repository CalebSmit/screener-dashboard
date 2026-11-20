# Nightly Log

Append-only. Newest entries at the bottom. One entry per session.

Each entry records what changed and why, the evidence behind it, what was tried
and rejected, and where the next session should pick up. This is the only
memory that carries between otherwise-cold sessions - write it for a reader
with no other context.

---

**Archive.** Entries from 2026-08-05 to 2026-10-19 are in
`log-archive/NIGHTLY_LOG-2026-08-05-to-2026-10-19.md` (moved 2026-11-19, text unchanged). Keep
about the last month here; move older entries to `log-archive/` a month at a time.

---

## 2026-10-20 (06:00 session) - QUEUE. The bank price-to-book proposal reviewed and ACCEPTED: Berkshire's P/B was 0.0010, not 1.52, and had been since March

**Health (rule 8, all five):** last code session ran? **the 06:00 one ran and shipped**
(`logs/nightly-2026-10-19_060001.log` - its gates failed at the time and an owner-run session
finished and merged its work the same day); the 13:00 slot **stood down** for the owner marker, as
designed. Data loop published? **yes** - `logs/datarun-2026-10-20_020001.log` ends "Data loop
complete", HEALTH: PASS, 501 scored. Evidence base at `1m` = **33 rows, newest 2026-09-18 (32 days
ago, bound 40), 4 effective**. Priority 0: `allow_auto_apply` **false**, 4 of 8 observations. Top
open roadmap item: **0.11, the context layer** (opened 2026-10-08, **age 12 days**) - deferred
today, see below.
**Tests:** before **2,817 passed, 1 skipped**; after **2,830 passed, 1 skipped** (13 new).
**Owner queue / rotation:** `MORNING_BRIEF.md` listed **three** proposals waiting for a decision,
all opened 2026-10-19, so deciding one was the day's first job ahead of the Tuesday focus
(`prompts/nightly.md` section 1). Took the bank price-to-book one, because it was the only one of
the three carrying a **demonstrably false number on the live site**. The context-layer item
(0.11) and the other two proposals were not started; nothing there is half-done.

### Did

**Reviewed, re-measured and ACCEPTED the bank price-to-book proposal**
(`research/2026-10-19-bank-price-to-book.md` section 8, raised by the 2026-10-19 session). Bank P/B
is 60 of the bank Valuation table's 100 points - about **13.2 composite points for each of the 46
bank-like stocks** - and it read the data source's `.info` `priceToBook`, which is
`currentPrice / bookValue` with `bookValue` a **per-share** figure.

- **The live defect, confirmed on this morning's published payload before anything was changed:**
  BRK-B's `pb_ratio` was **0.0010191567** at the **98.9th** percentile of the bank P/B ranking -
  the cheapest bank-like stock by a factor of nearly a thousand - and it was ranked **65th** on the
  strength of it. Its book value is per **Class A** share ($522,226) against a **Class B** price
  (about $532). Checked against the company's own filing rather than another of the source's rows:
  filed stockholders' equity of **$747,910,000,000** (equal to the probe's common equity *and* to
  the run's own `totalEquity`) over a market value of $1,139,349,913,600 is **1.5234**. The
  published figure was **1,494.8x** too low, which is the 1,500:1 A-to-B ratio. It had been
  published every weekday since March: at least the 91.4th percentile in **all 52** improvement
  snapshots carrying it, median 98.3.
- **Now:** `pb_ratio = marketCap / commonEquity`, both > 0 required, where `commonEquity` is a new
  raw field - the quarterly balance sheet's "Common Stock Equity" at the same column `totalEquity`
  is read from. **No fallback** to `priceToBook` or `bookValue` (the Phase 13 F19 rule). A
  company-level ratio has no per-share unit to get wrong, and it puts the metric on the same
  balance sheet as the rest of the record - the source's book was a quarter newer than everything
  else scored for **18 of the 46**.
- **Shipped with its frontend, same commit** (CLAUDE.md 0.8c): `metric_lineage` LINEAGE /
  RECOMPUTE / EQUATIONS, `pit_record.FIELD_CLASSES`, `scoring_schema` 12 -> 13, the generated
  methodology paragraph, claim `overview.bank_price_to_book`, `METHODOLOGY_CHANGELOG.md`.
- **A full screener run was made in the session** so the new field is in the live payload rather
  than waiting for 02:00: 501 scored, all **46 of 46** bank-like stocks keep a P/B, every one
  rebuilds from its two published inputs, `check_run_health` **PASS**,
  `scripts/check_published_claims.py` **PASS (1,130 tests)**, `scripts/rehearse_publish.py`
  **PASSED**.

**What the review added, which is the reason a second session decides these.** The note's evidence
for the one thing that could sink it - availability - was a single morning's probe. Re-measured
today on a **fresh** probe read through the engine's own extraction path and on the **2026-10-20**
run (`research/measurements/2026-10-20-bank-pb-review.py`): "Common Stock Equity" present and
positive for **46 of 46**, probe balance-sheet date equal to the run's for **46 of 46**, the run's
`totalEquity` within 0.5% of the probe's stockholders' equity for 46 of 46. That is exactly the
test the earnings-definition proposal **failed** on 2026-10-16 - 23 companies with no adjusted
figure at all, re-creating the fallback-cohort defect - and it is the difference between the two
decisions. The stakes were re-measured on this run too (the baseline reproduces **501 of 501**
published ranks): top 25 **unchanged both ways**, bank-like median move **1.5** places, 7 moved 10
or more, **0** non-bank composites changed, universe composite Spearman **0.9996**, **BRK-B 65th ->
139th** (P/B percentile 98.9 -> 59.8, Valuation 81.7 -> 58.2, composite 59.93 -> 54.77).

**Looked at the page** (Tuesday's standing rule). BRK-B's and JPM's P/B rows opened at 1440 and
375px: *"$1.14T market value ÷ $747.9B common equity = 1.52"*, *"Ranked 19th of 46 in Financials,
lower being better; sector median 1.86"*, *"59.78 percentile × 60% weight = 35.87 points"*, and
*"✓ Rebuilt from the inputs above: the same 1.52 that was scored."* The definition block names the
construction and says what it replaced. Two-line card on the phone, nothing clipped.
`logs/design/bankpb-*.png`.

**The research engine was rerun complete, and the prediction about it was written first.**
`factor_engine.py` and `pit_record.py` are both in `v2_experiments.RESULT_PATHS`, so all 30 rows
went stale. Predicted in the changelog before running: nothing moves, because
`pit_record.statement_fields` sets `commonEquity` to the filed stockholders' equity and
`market_fields` already built `priceToBook` as price x shares / equity - the same ratio. Rerun in
eight foreground chunks with `scripts/rerun_experiments.py`, committing after each; `--status` now
reports **0 stale**. Checked rather than assumed
(`research/measurements/2026-10-20-engine-rerun-diff.py`): each experiment's newest row against the
one before it, every numeric statistic at full precision - **30 of 30 identical, largest difference
0.0** - and `research.html` and `summary.json` came out byte-unchanged.

### Evidence / research

- **MSCI, *Fundamental Data Methodology* (March 2021)**, section 1.2.5 (p. 16): book value excludes
  "minority interest, treasury shares and preferred shares"; section 1.1.1 (p. 10): per-share
  figures use "the total number of shares at the company level", all classes aggregated - i.e.
  market value over common equity.
  https://www.msci.com/eqb/methodology/meth_docs/MSCI_Fundamental_Data_Methodology_Mar2021.pdf
- **S&P DJI, *S&P U.S. Style Indices Methodology* (June 2022)**, p. 6: "Book Value to Price Ratio"
  computed "for each company".
- **JPMorgan Chase, 2022 Form 10-K**, MD&A p. 59: "TCE represents the Firm's common stockholders'
  equity (i.e., total stockholders' equity less preferred stock)".
  https://www.sec.gov/Archives/edgar/data/19617/000001961723000231/jpm-20221231.htm
- **Damodaran (2009), "Valuing Financial Service Firms"**, pp. 30-31: P/B with ROE as its
  "companion variable" - the pairing `bank_metric_weights` rests on.
  https://pages.stern.nyu.edu/~adamodar/pdfiles/papers/finfirm09.pdf
- **Fama & French (1992), p. 429** excludes financial firms from the book-to-market tests, so the
  published value premium is **not** evidence about bank P/B. This change is justified by practice,
  not by a premium.

### Methodology changed

- `METHODOLOGY_CHANGELOG.md` **2026-10-20 - Proposal ACCEPTED: bank P/B is market value over common
  equity, not the source's per-share ratio.** `scoring_schema` 12 -> 13. Rollback: revert the
  commit, schema back to 12.

### Tried and rejected

- **Market value / stockholders' equity (including preferred)** - needs no new field, so no
  refetch, and it was tempting for that reason alone. Rejected: it agrees with the source for only
  16 of 45, it makes preferred-heavy banks look cheaper by the amount of preferred they have issued
  (KEY, STT, WFC, GS, USB, BNY each 11-13%), **and it changes the top 25** (ACGL out, DECK in).
  Every source read today - MSCI, S&P DJI, JPMorgan's own 10-K - starts from **common** equity,
  because preferred stock is a senior claim the common price does not buy.
- **Market value / tangible book**, the bank analysts' own measure. Rejected on measurement, not on
  principle: the source's tangible-book row agrees with the filings for only 22 of 41 (filers tag
  intangibles inconsistently, and JPMorgan-style deferred-tax add-backs are not tagged at all), and
  it reorders the bank set by a median 20.5 places. 17 of the 46 are insurers, conventionally
  valued on book. A tangible-book metric for banks proper stays a research question.
- **Patching Berkshire alone** (the note's minimum option). Rejected: a ticker-specific correction
  on a figure the run cannot check is how the error survived seven months in the first place.
- **Deferring the full run to 02:00.** Rejected: without it the payload cannot carry the new input,
  three `test_metric_lineage` tests fail against the committed payload, and the false figure stays
  on the public site another day. The run's own gates were run by hand instead.

### Next

- **Decide the second of the three open proposals: bank ROE and ROA**
  (`research/2026-10-19-bank-roe-roa.md`). It is the natural pair to today's - the same 46 stocks,
  the same 18 with a newer source quarter, and section 7 of today's note explains how the two
  constructions sit together (P/B on common equity, ROE on total equity, the identity inexact by
  the preferred share of equity, which is under 5% for 24 of the 46 and at most 12.6%). Re-measure
  availability on a fresh probe first, as this session did; that is what separated this decision
  from the 2026-10-16 refusal. The share-class proposal is the third and the largest in scope.

---

## 2026-10-20 (13:00 session) - QUEUE. The bank ROE/ROA proposal reviewed and ACCEPTED: 60% of bank Quality rested on two ratios with no published definition

**Health (rule 8, all five):** last code session ran? **yes** - `logs/nightly-2026-10-20_060001.log`
ends "Run complete: shipped to main", tagged `good/2026-10-20`. Data loop published? **yes** -
`logs/datarun-2026-10-20_020001.log` ends "Data loop complete", HEALTH: PASS. Evidence base at
`1m` = **33 rows, newest 2026-09-18 (32 days ago, bound 40), 4 effective**. Priority 0:
`allow_auto_apply` **false**, 4 of 8 effective observations - unchanged, and nothing here rests on
it. Top open roadmap item: **0.11, the context layer** (opened 2026-10-08, **age 12 days**) -
deferred again today, see below.
**Tests:** before **2,830 passed, 1 skipped**; after **2,852 passed, 1 skipped** (+22, exactly the
new module; no pre-existing failures either side).
**Owner queue / rotation:** I am the day's **second** session, so I read the 06:00 entry first and
took its **Next** verbatim: decide the second of the three proposals opened 2026-10-19. That is
also an open-proposal obligation, which outranks the rotation (`prompts/nightly.md` section 1).
**Deferred:** the context layer (0.11) and the third proposal (share classes). Nothing there was
started, so nothing is half-done. I did not revisit or reverse anything the 06:00 session decided.

### Did

**Reviewed, re-measured and ACCEPTED the bank ROE/ROA proposal**
(`research/2026-10-19-bank-roe-roa.md` section 7; the review is now section 9 of that note).
ROE and ROA are 35 and 25 of the bank Quality table's 100 points - about **13.2 composite points
for each of the 46 bank-like stocks** - and both were read straight from the data source's
`.info` `returnOnEquity` / `returnOnAssets`.

- **Why that was wrong, and not merely untidy.** Neither the provider nor `yfinance` documents
  either field. The engine threw out `earningsGrowth` for exactly this reason - "a black-box input
  with no verifiable definition". Reverse-engineered, the source's ROE is *consolidated* net income
  over average *total* equity **including minority interests**.
- **The decisive measurement, against the companies' own SEC filings** (TTM `NetIncomeLoss` over
  averaged `StockholdersEquity` / `Assets` from the weekly `companyfacts` cache): the new
  construction agrees **within 0.5% for 37 of 37** comparable stocks (ROA 36 of 37, worst 0.79%);
  the source's ratios agree **within 2% for 21 (ROE) and 12 (ROA) of 35**. The change makes the
  metric agree with the primary source.
- **Now:** `roe = netIncome / ((totalEquity + totalEquity_prior) / 2)`, `roa` the same on total
  assets, both balances required positive, **no fallback** to the source's ratio and none to an
  ending balance (the Phase 13 F19 rule). The prior balances are the ones `_year_ago_position`
  matches by date (330-400 days).
- **Shipped with its frontend, same commit** (CLAUDE.md 0.8c): `metric_lineage` LINEAGE / RECOMPUTE
  / one **exact** EQUATIONS template each, `pit_record` (both fields leave `FIELD_CLASSES` and
  `market_fields` stops building them - they were a *third* construction, on ending balances),
  `scoring_schema` 13 -> 14, the generated methodology paragraph, `run_audit`'s field provenance,
  claim `overview.bank_roe_roa`, `METHODOLOGY_CHANGELOG.md`, `tests/test_bank_roe_roa.py` (22).
- **A full screener run was made in the session** so the fix is live now rather than at 02:00:
  **501 scored**, `check_run_health` **PASS**, `scripts/check_published_claims.py` **PASS (1,152
  tests)**, `scripts/rehearse_publish.py` **PASSED**, both payloads `node --check` OK.

**What the review added, which is the reason a second session decides these.**

1. **Availability re-measured, 46 of 46.** The note's evidence was one morning's probe. Re-measured
   on a **second day's run** through the engine's own extraction path plus a **fresh probe**:
   `netIncome`, `totalEquity_prior`, `totalAssets_prior` present and positive for **46 of 46**, the
   two balance sheets **exactly 365 days apart for all 46**. So the no-fallback rule leaves nobody
   unscored. This is the test the earnings-definition proposal **failed** on 2026-10-16 (23
   companies with no figure) and it is the whole difference between the two decisions.
2. **The note's own prediction was already stale, in the proposal's favour.** Section 6 says
   Berkshire "loses both ratios under any averaged construction". It does not:
   `_year_ago_position` (made late on 2026-10-19) finds its year-ago sheet at column **3**, and the
   probe shows BRK-B is the *only* one of the 46 whose prior sheet is not at column four - the only
   one taking it by position would have got wrong.
3. **The mixed-basis cohort had grown from one stock to two.** The note names MS as the only stock
   the source serves no ratio for; **WRB** had joined it, so two stocks were ranked on *ending*
   balances inside a percentile ranking of 44 on the source's construction. The proposal removes a
   live defect. Both rose on the fix (MS 447 -> 435, WRB 198 -> 197).
4. **The judgement call the note explicitly left open, decided with a measurement.** ROA's
   numerator is the shareholders' income and its denominator the whole consolidated balance sheet.
   Minority interests exceed 5% of equity for **6 of 46** and are large only for three names
   (IBKR 2.77x, KKR 1.59x, APO 0.97x), measured two independent ways that agree exactly. The
   alternative - counting the minority holders' profit too - was **refused**: the FFIEC's UBPR puts
   that income on its own line (item 14) and JPMorgan's 10-K agrees, so the regulators' numerator
   *is* the proposal's; and including it would flatter precisely the three companies whose
   shareholders own least of the assets being divided by (IBKR reads 2.28% on group profit against
   **0.53%** on its shareholders'), the wrong direction of error for a long-only screener. The
   asymmetry is real, so it is **stated on the page** rather than hidden.
5. **One ROE instead of two.** `sustainable_growth` already used average equity, so the lineage had
   to warn readers "this ROE is not the Yahoo ROE shown for banks". They now agree for **46 of 46**
   and the caveat is gone.

**The effect, isolated properly** (`research/measurements/2026-10-20-bank-roe-roa-shipped.py`):
the published run's own table scored both ways - composite Spearman **0.99973**, **top 25
unchanged**, **0 of 455 non-bank composites changed**, bank-like median move **5 places**, 13 moved
10+, 2 moved 25+; largest **IBKR 419 -> 449** (ROA 2.28% -> 0.53%, composite -3.2), PRU 150 -> 120,
C 198 -> 219, PNC 162 -> 181, STT 176 -> 195. Every bank-like stock keeps both metrics (46 of 46)
and each figure rebuilds from its two published inputs for **46 of 46**, worst error **0.0**.

**Looked at the page** (the standing rule for a scoring change). JPM's and IBKR's ROE and ROA rows
opened at 1440 and 375px: *"$65.1B net income / (($374.6B equity + $356.9B equity a year earlier) /
2) = 17.8%"*, *"Ranked 18th of 46 in Financials, higher being better; sector median 13.4%"*,
*"61.96 percentile x 35% weight = 21.68 points"*, and *"Rebuilt from the inputs above: the same
17.8% that was scored."* IBKR's ROA row carries the minority-interest caveat, which is exactly the
stock it is about. Two-line card on the phone, nothing clipped. `logs/design/bankroe-*.png`.
**One defect found by looking and fixed:** my first draft of the ROE caveat used backticks around
a field name, and this surface renders them literally - the page showed the backticks. It was the
only one of 17 lineage caveats using them. Reworded to prose and the dashboard regenerated, so the
published payload carries the corrected text.

### Evidence / research

- **FDIC, *Quarterly Banking Profile*, Notes to Users (Q2 2026)**, pp. 2 and 6: return on assets is
  net income "as a percentage of average total (consolidated) assets"; return on equity the same
  net income "as a percentage of average total equity capital".
  https://www.fdic.gov/quarterly-banking-profile/qbp-notes-users-second-quarter-2026.pdf
- **FFIEC, *UBPR User's Guide*, Summary Ratios page 1 (2026-08-20)**, item 16: "divided by average
  assets"; **item 14** is a separate line for income attributed to minority interests - the detail
  that settled the ROA judgement call.
  https://www.cdr.ffiec.gov/CDRDownload/CDR/UserGuide/v186/FFIEC%20UBPR%20User%20Guide%20Summary%20Ratios--Page%201_2026-08-20.PDF
- **JPMorgan Chase, 2022 Form 10-K**, MD&A p. 59: "ROA: Reported net income / Total average
  assets"; "ROE: Net income / Average common stockholders' equity".
  https://www.sec.gov/Archives/edgar/data/19617/000001961723000231/jpm-20221231.htm
- **MSCI, *Quality Indexes Methodology* (May 2025)**, Appendix I, and ***Fundamental Data
  Methodology* (March 2021)** section 1.2.5 - the *other* convention: **latest** book equity
  excluding minority interests and preferred. Named in the changelog as a disagreement with a
  reason, not ignored.
  https://www.msci.com/indexes/documents/methodology/2_MSCI_Quality_Indexes_Methodology_20250520.pdf
- **Asness, Frazzini & Pedersen (2019), "Quality minus junk"**, *Review of Accounting Studies*
  24(1), 34-112, Appendix: "ROE is net income divided by book-equity", book equity being
  "shareholders' equity minus preferred stock"; no averaging described.
  https://research-api.cbs.dk/ws/portalfiles/portal/60211462/lasse_heje_pedersen_et_al_quality_minus_junk_publishersversion.pdf
- **Damodaran (2009), "Valuing Financial Service Firms"**, pp. 30-31: of the determinants of
  price-to-book "the return on equity has the biggest impact" - the P/B-ROE pairing the bank metric
  set rests on, and the reason both halves of it changing today sharpens the double-count question.
  https://pages.stern.nyu.edu/~adamodar/pdfiles/papers/finfirm09.pdf
- **No IC series, no backtest number, and no research-engine figure was used** (rules 4 and 5).
  The justification is documented practice plus agreement with the companies' own filings.

### Methodology changed

- `METHODOLOGY_CHANGELOG.md` **2026-10-20 (13:00 session) - Proposal ACCEPTED: bank ROE and ROA are
  net income over AVERAGE balances, not the source's undocumented ratios.** `scoring_schema`
  13 -> 14. Rollback: revert the commit, schema back to 13.

### Tried and rejected

- **The common-equity basis** (net income to common over average common equity) - what banks and
  analysts actually quote, and MSCI's book value. Rejected on measurement: it ranks the bank set
  almost identically (rank correlation 0.987, median ratio 1.008), needs a fetch field whose rows
  are unreliable (Apollo's "Net Income Common Stockholders" is 23% of its net income, a data
  artefact that would collapse its ROE to 2.3%), and the preferred share of equity is under 5% for
  24 of the 46. Worth revisiting only if the common rows improve.
- **ROA including minority interests** - the note's named alternative. Rejected on the UBPR's own
  schema and on the direction of the error; see item 4 above.
- **Keeping an ending-balance fallback** for stocks missing a prior sheet. Rejected: it is the
  defect being removed, and at 46 of 46 availability it would never fire anyway - a branch that
  cannot execute is a second definition waiting for a bad data day.
- **Differencing the 06:00 and 13:40 published runs to state the effect.** This is what I first
  did, and it is wrong: the two runs are 7.5 hours and a full refetch apart, so it showed 434
  non-bank composites moving and two top-25 changes, none of it this change. The measurement script
  now carries the warning and the attribution comes from scoring **one** table both ways. Recorded
  because the number was nearly published in the changelog.

### Next

- **Decide the last of the three proposals: score each company once where it has several share
  classes** (`research/2026-10-19-share-classes.md`) - the only one still open, and the largest in
  scope of the three. It changes `run_screener.py`'s pipeline order rather than one metric, so
  re-measure its own stated stakes (top 25 unchanged; FOXA 47 -> 28, GOOGL 226 -> 197, NWSA
  61 -> 52) on a current run before trusting them: today's two accepted changes both moved the
  bank set, and the note's figures are from the 2026-10-19 table.
- **Then the context layer (0.11), deferred three sessions running now** (2026-10-19, and both of
  today's). Per the deferral-streak rule, the next session carrying no proposal and no broken loop
  must either advance item 4's track-record note or argue in writing for what outranks it.
- **A Wednesday question this change sharpened, not an open defect:** bank ROE now equals earnings
  yield x P/B on a near-common book, and both of those changed definition today, so whether 35
  points of bank Quality duplicating two Valuation inputs is a double count is worth the synthesis
  day (section 5 of the note).

## 2026-10-20 (owner-run, evening) - "should we run every backtest we can?" Measured: no. The overfitting check on the research page, and the engine rerun the 13:00 session left

**Health (rule 8):** last code session ran? **yes, both** - 06:00 and 13:00 shipped to main
(`logs/nightly-2026-10-20_130002-midday.log`). Data loop published? **yes** - 02:00, HEALTH: PASS,
claims gate 1,117 passed. Evidence base at `1m` = **33 rows, newest 2026-09-18 (32 days, bound 40),
4 effective**. Priority 0: `allow_auto_apply` **false**, 4 of 8. Top open roadmap item: **0.11, the
context layer** (opened 2026-10-08, **age 12 days**) - not worked; this session answered an owner
question instead (below), and the deferral streak the 13:00 entry records still stands.

**Owner:** *"should we be running more backtests, like all the backtests we possibly can to decide
what is best moving forward for the methodology, weightings, etc?"*, then *"do whatever you think
would be best"*, and whether the Data.gov API or other APIs would help.

### Shipped
- **`overfitting.py`** - the probability of backtest overfitting (Bailey, Borwein, Lopez de Prado &
  Zhu 2017, CSCV) over the ledger's 28 comparable weightings and 176 months, scored as each one's IC
  lead over the current weighting on shared stocks. Reads the committed ledger only, outside
  `RESULT_PATHS`, so it cannot stale a row. **Result:** PBO **0.21** over 2012-2026, but Size alone
  wins 68% of the in-sample trials and its lead sits in the heavy-survivorship 2012-2019 (+0.027) and
  is negative held out (-0.015); PBO **0.73** on 2020-2026 alone and **0.63** with Size dropped. So
  "keep the best backtest" would have picked the screen most exposed to the backtest's known defect,
  and otherwise noise. `research/2026-10-20-backtest-overfitting.md`.
- **`research.html`** gains "If you had kept the best backtest", every sentence generated from the
  three readings (the verdict sentence flips only if both checks fall below 0.45).
  `tests/test_overfitting.py`, 8 tests (noise ~0.5, a real edge -> < 0.1, all 12,870 splits, the page
  prints the ledger's numbers, no production import, not in `RESULT_PATHS`).
- **The research-engine rerun the 13:00 session did not do.** Its bank ROE/ROA change edited
  `factor_engine.py`, which left all 30 ledger rows stale on code. Rerun in committed chunks with
  `scripts/rerun_experiments.py`: **0 of 25,460 numeric values moved**; `summary.json` unchanged.
  (One chunk of 4 overran the Bash tool's 600 s and finished in the background; chunks of 3 after.)
- **`plan/factor-lab.md` "Free data to evaluate"**: Jensen, Kelly & Pedersen (2023) factor data
  (CC BY-NC 4.0) and Chen & Zimmermann's Open Source Asset Pricing returns - the second may let the
  lab rerun Piotroski, analyst revisions and earnings drift at the portfolio level. Terms to confirm
  before building. The api.data.gov key is not useful here (SEC EDGAR and FRED's CSV need none).

### Not changed
No weight, metric or threshold (rules 4-5). The PBO is a reason not to search harder, not a basis
for any weight.

### Next
Unchanged from the 13:00 entry: the share-class proposal, then the context layer (0.11).

## 2026-10-20 (owner-run, night) - nine agents' work merged in four batches: three lab experiments, three research notes, a test-the-tests audit, 0.11 closed

**Health (rule 8):** last code session ran? **yes** - 13:00 shipped to main. Data loop published?
**yes** - 02:00 2026-10-20, HEALTH: PASS. Evidence base at `1m` = **33 rows, newest 2026-09-18 (32 days,
bound 40), 4 effective**. Priority 0: `allow_auto_apply` **false**, 4 of 8. Top open roadmap item:
**0.11 is CLOSED tonight**, so the oldest open item is **0.13, the Factor Lab** (opened 2026-10-12, **age 8
days**), worked tonight (experiments 15-17).

**Owner:** *"do as much work as you can tonight before the 2 am session please! Then have the sessions
fine tune things and test things."* Answered with two waves of agents in isolated worktrees, each
merged only after its own full suite, then batch-gated here (tags `good/2026-10-20-owner-2` .. `-5`).

### Shipped
- **Engine calibration test** (`tests/test_engine_calibration.py`, 7): planted IC differences of
  0.03/0.05/0.10 come back within 4 SE through the engine's own `per_month`/`summarise`; 0.10 clears the
  bar held out; no false alarm on ten null pairs; twice `detectable` always caught; a 70%-coverage screen
  is measured on the shared stocks. Test-only - the ledger is untouched.
- **Size: the weight stays at 5%** (`research/2026-10-20-size-weight.md`): no unconditional size premium
  among large companies, but a within-sector small tilt beside Quality is documented practice (MSCI,
  FTSE Russell, Dimensional) and Size is one of the counterweights holding the ranking growth-neutral.
  **Lab experiment 16 tested the note's pre-registered rule and did not refute it** (0.166% a month, t
  2.36, against 0.136% and 1.5) - with a thin margin (0.030, t 1.24) and negative since 2013. The page's
  "small-cap premium" question line and the Size column's tooltip were overstatements; both reworded
  at their generators.
- **Bank Quality vs Valuation** (`research/2026-10-20-bank-quality-valuation-overlap.md`): ROE = EY x P/B
  is not a double count - the 44 points net to earnings yield and equity per dollar of assets with no
  net P/B weight, the deliberate "high ROE at a low P/B" bank screen. **OPEN proposal**: state the
  pairing on the page and guard the zero net P/B loading with a test.
- **Free survivorship sources** (`research/2026-10-20-free-survivorship-sources.md`): free data closes
  **37%** of v2's 2012-2019 survivorship gap with monthly returns (the members that left but kept
  trading), **1.2% of acquisitions** (half the gap). The Size test on free data can falsify the
  survivorship explanation and cannot confirm it. Two free v2 data fixes found (pre-2014 renames, 167
  member-months; predecessor CIKs, 773) - queued.
- **Context layer 0.11 CLOSED**: a false live sentence fixed (mid-session quotes labelled "previous
  close" for 485 of 501 stocks after the 13:00 publish), a mid-session run now refused from the context
  log (and 10-20's log restored), `context_eval` uses the generator's skew rule, five claims registered.
  The one remaining item is a date: first one-month readings with the 2026-11-09 run.
- **Factor Lab 15-17**: Jensen, Kelly & Pedersen 2023 (10 of 10; **CC BY-NC 4.0 - teaching only**;
  second-read the same night, all ten figures hold, copyright notice added); size inside profitability
  (10 of 17); Piotroski 2000 on Open Source Asset Pricing (1 of 4, a miss the decile cut explains). 17
  experiments, **145 of 217** within. Piotroski was moved after 15 and 16 at merge so their published
  numbers hold.
- **Claims mutation audit** (`plan/claims-mutation-audit.md`): 38 mutations on 22 claims; **15 got past
  the registered checks** - 8 untested boundaries, 3 checks reading the committed payload instead of the
  code, 1 tolerance looser than its claim. All 15 now caught; the next 12 claims are listed.
- **`OWNER_FOCUS.md`**: the owner's fine-tune-and-test direction as the top open item, with an ordered
  queue for the scheduled sessions.

### Lessons for the routine
- **Chain the gates.** A gate-2 failure did not stop a merge because the commands were joined with `;`.
  The failure was an artifact (`python -P` hides the repo's own modules from a repo script), and gate 2
  passed on the identical tree minutes later, but the chain must stop on any gate: use `set -e`/`&&`.
- **The scratchpad is shared with subagents.** One wrote `inspect.py` there, which shadowed the standard
  library for any script run from that folder. Give agents their own subfolder and run scratch scripts
  with `-P`; never run repo scripts with `-P`.
- **A fresh worktree lacks `runs/`**: three tests fail with FileNotFoundError. Every agent copied it in;
  worth making those tests skip loudly instead (queue it).
- **The full suite runs 6:07-7:20** against the Bash tool's 600 s. Watch it (OWNER_FOCUS item 7).

### Next
`OWNER_FOCUS.md`'s 2026-10-20 item, in its order: decide the two OPEN proposals (share classes; the bank
P/B-ROE guard), second-read experiments 16 and 17, continue the mutation audit, the positive-control
ledger row, the v2 data fixes and the extended build.

## 2026-10-21 (06:00 session) - QUEUE. The share-class proposal reviewed and ACCEPTED - but the designated line is declared, not re-picked: Alphabet is 179 on GOOGL and 101 on GOOG

**Health (rule 8, all five):** last code session ran? **yes** - 2026-10-20 13:00 ended "shipped to
main" (`logs/nightly-2026-10-20_130002-midday.log`), and two owner-run sessions followed it.
Data loop published? **yes** - `logs/datarun-2026-10-21_020001.log` ends "Data loop complete",
02:26. Evidence base at `1m` = **33 rows, newest 2026-09-18 (33 days ago, bound 40), 4 effective**
- inside the bound, no action. Priority 0: `allow_auto_apply` **false**, 4 of 8 effective
observations. Top open roadmap item: **0.13, the Factor Lab** (opened 2026-10-12, **age 9 days**) -
not worked; its day is Friday 2026-10-23, and the owner queue outranks it today.
**Tests:** before **2922 passed / 1 skipped**, after **2969 passed / 1 skipped** - +47, which is
exactly the two new modules, no failures and nothing removed. Gates 2 and 3 pass (dry-run exit 0;
`index.html` 642,631 bytes; both payloads `node --check` clean with their opening assignments).
**Owner queue / rotation:** took `OWNER_FOCUS.md`'s 2026-10-20 item **1**, the first OPEN proposal.
Wednesday's nominal SYNTHESIS focus was displaced by it, which the prompt requires - and the work
was a synthesis question anyway: what the *population* of a sector percentile should be. **Deferred:**
the second open proposal (the bank P/B-ROE guard, opened 2026-10-20) and the Factor Lab.

### Did

**Reviewed `research/2026-10-19-share-classes.md` and accepted it, with one change the review
measured and the note did not have.** The note is good: it inventoried every multi-class company,
established from MSCI, S&P DJI and FTSE Russell that fundamentals are company facts and prices are
line facts, rescored six candidate rules, and ruled out the cheap one (filling a line's *missing*
analyst fields puts FOX at **#6**, into the top 25, on a fallback growth figure of +49.6%). What it
proposed was S&P DJI's approach B - score the company once on its **Designated Listing** - with the
line picked from **each run's own `avg_daily_dollar_volume`**.

**That last part is not S&P DJI's rule and is not stable, which is what a second reader is for.**
S&P changes a Designated Listing only when the other line leads on *both* criteria by more than
20%. Measured over 89 month-ends (2019-06 to 2026-10, 63-day average daily dollar volume):

| | Alphabet | Fox | News Corp |
|---|---|---|---|
| Months the two lines are within S&P's 20% margin | **31 of 89 (34.8%)** | 0 | 0 |
| Closest the two ever came (volume ratio) | **1.003** | 2.077 | 2.193 |
| Times the busier line actually changed | **4** (2020-09, 11, 12; 2021-01) | 0 | 0 |

And it matters: rescoring the 2026-10-21 run both ways, **Alphabet ranks 179 scored on GOOGL and
101 scored on GOOG** - 78 places, with a different ticker in the published ranking - because the
two lines carry different analyst records (the note's own section 7: the source serves GOOGL's
quarterly "actual EPS" on a GAAP basis and GOOG's on the analysts' basis). A per-run recomputation
would have moved a company 78 places on a liquidity tiebreak that is a coin flip a third of the
time, with nothing in the changelog to explain it.

**So the choice of line is declared in `config.yaml` (`share_classes.designated: GOOGL, FOXA,
NWSA`)**, changed only with a changelog entry, and the run *reports* a crossing of the margin
rather than acting on it (`ATTENTION -` in `run.log` and the health note). An undeclared
multi-class company - a new one joining the index - falls back to volume and says so loudly, and
two declared lines of one company raise. Everything else shipped as proposed.

**What shipped:**
- **`share_classes.py`** - group the universe by SEC CIK (falling back to identical `longName`),
  keep the declared line, report the margin, and state each dropped line's price gap. `run_screener`
  drops the secondary lines after the outlier flagging and **before** `compute_sector_percentiles`,
  and records `runs/<id>/share_classes.json`. The dropped rows stay in `02_outliers_flagged`, so
  they remain auditable.
- **`run_context.config_hash`: `share_classes` added and `scoring_schema` 14 -> 15.** Without the
  bump a warm start would have served the cached 501-row scored table - built when the percentiles
  counted all three pairs - and the change would have silently not happened. This is the same trap
  the schema comments record for 2026-10-07, 10-08 and 10-20.
- **The page answers for a ticker it no longer ranks.** `generate_dashboard._share_classes_block`
  puts the second lines in the first payload (not the context file - the search palette and My
  Holdings run before context loads), and the search palette, the table's empty state and the
  holdings card now say "FOX is a second listed share class of Fox Corporation. The company is
  scored once, on its more-traded class FOXA, ranked #35 here ... It trades 10.8% below FOXA for the
  same claim on the same earnings." Before this, all three surfaces told a holder of FOX the company
  had left the S&P 500 or its data could not be fetched - false.
- **The methodology page** gained Step 3's opening paragraph, generated from `config.yaml` so it
  cannot name a line the run dropped, with the three providers' rules and the stability measurement.
- **Claims `overview.share_classes` and `drilldown.share_class_second_line`**, so both are in the
  data loop's publish gate. The page claims are checked by **reading the sentences out of a browser
  render of the generator's own page**, not by grepping the source for a string (CLAUDE.md 0.11).

**Verified end to end, and that is where the only real defect turned up.** I ran the whole
screener (run `4e31fc82a3d6`, 08:16) rather than trusting unit tests, into a throwaway run whose
published artifacts I then restored byte-for-byte - the 02:00 loop publishes this through its own
gates, which a code session should not bypass. **The first attempt crashed**: inside the pipeline
the fetch is a **list of per-ticker dicts** and only the run artifact is a DataFrame, so
`'list' object has no attribute 'columns'` - with all 42 unit tests green, because every one of
them had handed the function a DataFrame. Fixed as `share_classes.fetch_maps`, which reads both
shapes, with four tests that would have caught it. The second run:

- `Share classes (sec_cik): 3 multi-class companies, scored once on ['FOXA', 'GOOGL', 'NWSA'];
  3 line(s) not ranked: ['FOX', 'GOOG', 'NWS']`, no margin warning (GOOG/GOOGL 1.52x today).
- **498 composites and 3,979 category scores reproduced from the published payload alone**
  (`calc_trace`, CLAUDE.md row 0.8), and the ranks match the pre-shipping rescoring **exactly**:
  FOXA **#35**, GOOGL **#179**, NWSA **#54**.
- **`scripts/check_run_health.py`: HEALTH: PASS** on the 498-row run - price coverage 498/498,
  every category's dispersion within its band, no metric availability change. So the 02:00 loop
  will accept it.
- **Looked at the page at 1440 and 375** with that run's real payload (no console errors at either
  width): the hero now reads "RANKED THIS RUN **498**", and searching FOX, GOOG or NWS gives, for
  example, *"FOX is a second listed share class of Fox Corporation. The company is scored once, on
  its more-traded class FOXA, ranked #35 here - the way S&P DJI and FTSE Russell treat multiple
  share classes. It trades 10.8% below FOXA for the same claim on the same earnings."* The holdings
  card says the same and keeps its remove button. **One defect seen by looking and fixed**: the
  sentence printed "a second listed share class of Alphabet Inc.**.**" - a company name that ends
  in a stop now loses it, with a test.
- `scripts/rehearse_publish.py` **PASSED** (it had failed first time, for the ledger reason below).
- **One thing to know if you do this again:** the verification run's directory must be removed
  afterwards, not just its published artifacts. Leaving a 498-row run *newer* than the 501-row
  published payload is the mismatch CLAUDE.md row 0.5/0.7 forbids, and
  `tests/test_investor_profiles.py` caught it immediately (`KeyError: 'FOX'`) - the test reads the
  committed payload against `latest_run_dir`. Both throwaway runs are gone and the suite is green.

### Evidence / research
- **MSCI, *Fundamental Data Methodology* (March 2021), s1 p. 10 / s2.1 p. 19** - company-level
  aggregation of all classes, listed and unlisted; every ratio is the **security's** price over the
  company's per-share figure.
  https://www.msci.com/eqb/methodology/meth_docs/MSCI_Fundamental_Data_Methodology_Mar2021.pdf
- **S&P DJI, *Equity Indices Policies & Practices* (Aug 2026), pp. 17-18** - Designated Listing
  (highest one-year liquidity **and** largest float-adjusted cap), the **20%** switching margin, and
  approach **B**: "Each company is represented once by the Designated Listing."
- **FTSE Russell, *Russell US Equity Indexes* v7.2 (Aug 2026), rules 6.5.1-6.5.4 p. 18** - style
  probabilities "based on that of the pricing vehicle and assigned consistently across all
  additional share classes".
  https://www.lseg.com/content/dam/ftse-russell/en_us/documents/ground-rules/russell-us-indexes-construction-and-methodology.pdf
- **S&P DJI, *S&P U.S. Style Indices Methodology* (June 2022), pp. 5-6** - "Evaluating Growth and
  Value at the **Company** Level". The screener's sector percentiles counted **rows**.
- **Measured, re-derived independently on the 2026-10-21 run**
  (`research/measurements/2026-10-21-share-class-stability.py`; the rescoring baseline reproduces
  **501 of 501** published ranks): CIK coverage **503 of 503** with the `longName` fallback giving
  **identical** groups and membership (so the fallback a fresh checkout uses is corroborated, not
  assumed - `data/sec/` is gitignored); exactly three multi-line CIKs, **no false merge**; top 25
  **unchanged**; FOXA 61 -> 35, GOOGL 199 -> 179, NWSA 65 -> 54; 395 other rows moved, largest 55
  places; the 16 other Communication Services stocks **+1.565** composite points on average (min
  +0.69, max +2.52, none negative - they were being held down by duplicate rows in a 22-row peer
  group); **0.0** composite change outside that sector; and the earnings-yield ratio over the price
  ratio is **1.000000** for all three pairs, which is what lets the page state the whole valuation
  difference between two tickets as one price gap.

### Methodology changed
- **`METHODOLOGY_CHANGELOG.md` 2026-10-21** - "Proposal ACCEPTED, with the designated listing
  declared rather than re-picked: one company, one ranked line", with the four provider sources, the
  stability measurement, the full before/after table, the four rejected alternatives and the
  rollback (`share_classes.score_once_per_company: false`).

### Tried and rejected
- **Re-picking the designated line from each run's volume**, as the note proposed - refused on the
  89-month stability measurement above. It is the one change the review made.
- **MSCI's route** (rank every line, cap exposure by issuer) - needs an issuer cap the bot does not
  have, and leaves the peer-group double count. **FILL**, **CONSENSUS** and **ONCE** - all rescored
  and all worse; FILL would have put FOX in the top 25 at #6.
- **Narrowing the research engine's `base_config_hash`.** My `config.yaml` change made all 30 ledger
  rows stale, because `v2_experiments.base_config()` hashes the **whole** config and `config.yaml`
  is in `RESULT_PATHS` - yet backtest v2 never reads `share_classes`. That is the same over-broad
  fingerprint the 2026-10-18 session fixed for the dirty-tree marker ("a marker that fires in normal
  operation is a marker that gets ignored"). I **did not** narrow it: it is another subsystem's
  provenance guard, the rerun machinery exists for exactly this, and the honest outcome is the
  evidence. All 30 rerun in committed chunks: **0 of 25,460 numeric values moved**, and
  `summary.json` is byte-identical, so the public research page says the same thing. Worth a future
  session's attention as a *narrowing* with a test, not as a loosening.

### Lesson for the routine
**Unit tests cannot see a shape assumption; an end-to-end run can.** 42 tests passed on a module
whose single caller would have crashed the 02:00 data loop tonight, because every test handed it
the shape the run artifact has and the pipeline hands it another. The habit worth keeping is the
one that caught it: **run the thing, in full, before you push a pipeline change** - into a
throwaway run whose published artifacts you restore, so the verification costs nothing and does not
publish behind the data loop's gates.

### Next
**The second open proposal: the bank P/B-ROE pairing** (`research/2026-10-20-bank-quality-valuation-overlap.md`
section 8, opened 2026-10-20) - state the identity on the methodology page and guard the zero net
P/B loading with a test. Then `OWNER_FOCUS.md` item 2 (second-read lab experiments 16 and 17) and
item 3 (the next 12 claim mutations in `plan/claims-mutation-audit.md`).

---

## 2026-10-21 (13:00 session) - QUEUE. The bank P/B-ROE proposal reviewed and ACCEPTED - and the guard it asked for could not see the one change the note itself calls the mildest route to a P/B tilt

**Health (rule 8, all five):** last code session ran? **yes** - today's 06:00 session ends
"Run complete: shipped to main" at 08:48 (`logs/nightly-2026-10-21_060000.log`). Data loop
published? **yes** - `logs/datarun-2026-10-21_020001.log` ends "Data loop complete" at 02:26,
PUBLISHED CLAIMS PASS (1214 tests). Evidence base at `1m` = **33 rows, newest 2026-09-18 (33 days
ago, bound 40), 4 effective** - inside the bound, no action. Priority 0: `allow_auto_apply`
**false**, 4 of 8 effective observations. Top open roadmap item: **0.13, the Factor Lab** (opened
2026-10-12, **age 9 days**) - not worked; its day is Friday 2026-10-23, and the owner queue
outranks it today.
**Tests:** before **2969 passed / 1 skipped**, after ****2980 passed / 1 skipped**** - the delta is exactly the one new
module, no failures and nothing removed. Gates 2 and 3 pass (dry-run exit 0; `index.html` 642,631
bytes; both payloads `node --check` clean with their opening assignments).
**Owner queue / rotation:** took `OWNER_FOCUS.md`'s 2026-10-20 item **1**, which named this as "the
next session's first job", and it was the only OPEN proposal in the morning brief. **Deferred:**
item 2 (second-read lab experiments 16 and 17), item 3 (the next 12 claim mutations) and the
Factor Lab.

### Did

**Reviewed `research/2026-10-20-bank-quality-valuation-overlap.md` section 8 and accepted it -
with the guard replaced, which is the part only a second reader produces.**

The note's question was whether 35 points of a bank's Quality score double-count two of its
Valuation inputs. Since the two changes that shipped on 2026-10-20 put ROE and P/B on nearly one
book, a bank's ROE **is** its earnings yield times its P/B. The note's answer - no, because ROE
duplicates earnings yield but *cancels* price-to-book, so the four weights net to "high ROE at a
low P/B", the construction bank analysts and index builders use on purpose - holds up. I
re-measured it on a different run and it reproduces. **No weight, metric or threshold changed.**

**Re-measured on run `4dc4acf1d780` (today 02:00) against the note's `7c12b43a07b6` (yesterday
13:21).** Both measurement scripts rerun; the rescoring baseline reproduces **501 of 501**
published ranks either way:

| | Note's run | My run |
|---|---|---|
| **(a)** R-squared, log ROE on log E/P and log P/B (bar 0.90) | 0.978 | **0.978** |
| **(b)** measured loading on P/B's rank | -0.067 | **-0.069** |
| loading on E/P's rank - criterion (b)'s bar is half of it | +0.284 | **+0.283** |
| status quo against the P/B-ROE (Wilcox) residual | -0.890 | **-0.892** |
| the six alternatives against it | -0.85 / -0.72 / -0.84 / -0.79 / -0.84 / -0.88 | **-0.848 / -0.724 / -0.839 / -0.796 / -0.838 / -0.882** |

Every figure to the third decimal, and the status quo is still the closest of the seven
constructions to the combined signal. Both refutation criteria hold.

**Then I measured the guard the proposal asked for, and it is blind in the direction that
matters.** The proposal specified one failing condition: the first-order net loading on log P/B
computed from `config.yaml` (`-pb_ratio + roe + roa`, in composite points; 0.0 today), outside
+-1.5 points. That is exact algebra on log coordinates - but **the engine ranks percentiles**, and
**ROE and ROA both enter log P/B with coefficient +1**, so moving weight between them cannot move
the first-order figure *at all*. Measured on the engine's own scores
(`research/measurements/2026-10-21-bank-pb-roe-guard-review.py`):

| Candidate | First-order net on log P/B | Proposal's guard | Measured loading on P/B's rank | Criterion (b) |
|---|---|---|---|---|
| **S0 status quo** | 0.0 | passes | **-0.069** | holds |
| C1 drop ROE | -4.7 | FAILS | -0.211 | REFUTED |
| C2 drop P/B | +13.2 | FAILS | +0.216 | holds |
| **C3 ROE's 35 -> ROA** | **0.0** | **passes** | **-0.156** | **REFUTED** |
| C4 drop ROE and ROA | -13.2 | FAILS | -0.308 | REFUTED |
| P1 ROE 35 -> 25, equity ratio 15 -> 25 | -2.2 | FAILS | -0.119 | holds |

**C3 is not a corner case - it is the change the note's own refutation criterion (c) names as "the
mildest route measured" to a deliberate P/B tilt.** It reads exactly 0.00, identical to the status
quo, while its measured loading is **2.3x** today's and past the note's own criterion (b) bar of
0.141. The mechanism is that ROA also carries log B/A, which correlates **-0.34** with log P/B
across the bank set. So the proposed guard would have let that reweight through while the page
went on saying "no net weight on P/B by itself". And **P1 shows the same band erring the other
way**: a reweight nobody proposed trips it at -2.2 points while leaving the claim measurably true.
Fitted across the candidates, +-1.5 points is about +-0.03 of measured loading - a fifth of
criterion (b)'s bar.

**So the failing condition is the measured loading at criterion (b)'s own pre-registered bar**, and
the first-order check is kept at a quarter of the earnings-yield net as the arithmetic the page
*explains*. The general form is CLAUDE.md 0.11's lesson: **the first-order figure is the page's
explanation; the measured loading is the page's claim - guard the claim.**

**What shipped:**
- **The methodology page** gains "A bank's Valuation and Quality scores are one reading, not two":
  the identity, the three independent log coordinates, a table of the net composite points
  (**+22.0** earnings yield, **0.0** price-to-book, **+8.8** equity per dollar of assets) and the
  practice it follows. **Every figure is generated from `config.yaml`**, and if a reweight ever
  moves the block off neutral the sentence **states the measured tilt instead of claiming
  neutrality** - a page that cannot become false rather than a sentence with a constant in it.
- **The two cells calling P/B and ROE "THE key" bank metric** now say each is read against the
  other, and why.
- **A `CONSTRAINT` comment above `bank_metric_weights`** naming the coordinates, the net position,
  the note and the test - where someone editing one of the four weights will actually read it.
- **`tests/test_bank_pb_roe_identity.py`** (11 tests) and claim `overview.bank_pb_roe_identity`, so
  it is in the data loop's publish gate. One test asserts the blind spot itself, so nobody later
  deletes the measured check as redundant.
- **The note gains section 11** (the decision, the re-measurement, the guard finding) and its
  status line is rewritten. `write_brief.open_proposals()` now returns `[]`.

**Looked at the page**, at 1440 and 375, on the *regenerated* page from `rehearse_publish`, not the
published one: the section renders as an H3 inside Bank-specific scoring, the net-points table
picks up the page's own table styling with tabular figures, **no console errors at either width**,
and at 375 the table sits exactly inside its container (354px of 354px) and **wraps rather than
clips** - checked on the screenshot, not only with `scrollWidth`, which cannot see a clipped cell.
`scripts/rehearse_publish.py` **PASSED** with the published files restored byte-for-byte.
**One defect seen by looking and fixed:** the first placement left the
"All financial-sector stocks receive a `Financial_Sector_Caveat` flag" paragraph orphaned *under*
my new heading, where it belongs to the parent section. Moved to the end of the block.

**The cost worth knowing, and why I paid it.** Item (4) is a `config.yaml` comment - and
`v2_experiments` fingerprints `config.yaml` **by its bytes**, so a comment that cannot change a
number staled all 30 research-engine ledger rows, exactly as the 06:00 session's real config change
did this morning. I kept the comment (it is the one place a future editor of those weights will
look) and reran the 30: **every stored statistic of all 30 rows compared with its previous run at full precision, **0 of 25,460 values moved, largest difference 0.0**, and the research page's `summary.json` is byte-identical, so the public page says exactly the same thing**. The honest note for whoever picks this up: this is the second
time in one day that an over-broad fingerprint has forced a 47-minute rerun for nothing, which is
the "a marker that fires in normal operation is a marker that gets ignored" failure one level up.
The 06:00 session already flagged it as worth **narrowing with a test** - hashing `config.yaml` by
its parsed scoring-relevant content rather than its bytes. I did not do it here because it is
another subsystem's provenance guard and changing `v2_experiments.py` would stale every row again
anyway; it wants its own session.

### Evidence / research
All read in the note and checked against the page's wording before it shipped; nothing here is a
return claim, and the generated section carries a test forbidding one.
- **Damodaran, *Valuing Financial Service Firms* (2009), pp. 30-33** - ROE is the biggest
  determinant of a bank's P/B; the two correlate above 0.70 across U.S. banks (Feb 2009); his
  worked example regresses P/B on ROE across peers and reads the residual.
  https://pages.stern.nyu.edu/~adamodar/pdfiles/papers/finfirm09.pdf
- **Damodaran, *Investment Valuation* (2nd ed.), ch. 19, PDF pp. 6-7** - P/BV = (ROE - g) / (r - g);
  with no growth, r = ROE / (P/B) = E/P, which is the identity this page now states.
  https://pages.stern.nyu.edu/~adamodar/pdfiles/valn2ed/ch19.pdf
- **Wilcox (1984), "The P/B-ROE valuation model", *Financial Analysts Journal* 40(1)** - secondary
  (CFA Institute abstract + Damodaran ch. 19 PDF p. 23), and the page says only that it is the same
  line. https://rpc.cfainstitute.org/en/research/financial-analysts-journal/1984/the-pb-roe-valuation-model
- **Nissim (2013), *Review of Accounting Studies* 18, 324-359, pp. 326, 338** - conditioning P/B on
  ROE is the standard in insurance and banking and improves valuation accuracy. Accuracy, not
  return.
- **Mercer Capital, *Shareholder Value Drivers* (2020), slides 21-22** - a bank-valuation firm
  reading a persistently low P/TBV as depressed profitability, not cheapness.
  https://mercercapital.com/content/uploads/Shareholder-Value-Drivers.pdf
- **MSCI, *Diversified Multiple-Factor Indexes Methodology* (May 2022), pp. 4, 11** - book-to-price,
  earnings yield and profitability added in one score, **no carve-out for financials**. The closest
  institutional analogue to what this screener does.
- **Measured by me, independently, on run `4dc4acf1d780`**
  (`research/measurements/2026-10-21-bank-pb-roe-guard-review.py`, JSON committed beside it): the
  two refutation criteria, the first-order-against-measured table above, and the
  -0.34 correlation between log P/B and log B/A that makes a ROE/ROA reweight move the measured
  loading.

### Methodology changed
- **`METHODOLOGY_CHANGELOG.md` 2026-10-21 (13:00 session)** - "Proposal ACCEPTED: a bank's
  Valuation and Quality scores are one reading, stated on the page and guarded by what it claims".
  No weight, metric, threshold or input changed; **0 of 501 composites move** and the payload's
  numbers are byte-identical. The entry is in that file rather than a plain commit because it fixes
  the *explanation* of a scoring position and pins the position against future drift.

### Tried and rejected
- **The guard as the proposal specified it** - first-order net from `config.yaml`, +-1.5 composite
  points - **refused and replaced**, on the C3 and P1 measurements above. It is the one change the
  review made, and it is the same class of finding as the previous two refusals: the note measured
  one point, the review measured the thing the sentence actually claims.
- **Changing any bank weight.** Six alternatives rescored on my own run; every one is further from
  the documented P/B-against-ROE reading than the status quo, four of six change the top 25, and
  all move a median 18-42 bank-like places. Removing the "overlap" removes the *conditioning*.
- **Narrowing `v2_experiments`' config fingerprint** to dodge the 30-row rerun - deferred, with the
  reasoning above. Doing it would have staled every row anyway.
- **Dropping the `config.yaml` comment** to avoid the rerun - rejected: it is the proposal's item
  (4) and the single most likely place for a future session to read the constraint before breaking
  it. The rerun is cheaper than the defect it prevents.

### Next
- **`OWNER_FOCUS.md` item 2: second-read lab experiments 16 and 17** (size inside each
  profitability fifth; Piotroski on Open Source Asset Pricing's portfolios). Each needs its paper
  reopened, every reference figure re-read against `factor_lab.py`, and `git log` checked that the
  tolerances were committed before the comparison ran. Experiment 15 is already done.
- Then **item 3**, the next 12 claim mutations in `plan/claims-mutation-audit.md`.
- **Worth its own session when one is spare:** narrow `v2_experiments.RESULT_PATHS`' handling of
  `config.yaml` to the parsed scoring-relevant keys, with a test - it has now forced two 47-minute
  reruns in one day for changes that provably cannot move a number.

## 2026-10-22 (06:00 session) - QUEUE/BUILD. One pre-registered stop-loss on the bot's rule, and the prediction failed in two of three periods

**Health (rule 8, all five):** last code session ran? **yes** - 2026-10-21 13:00 ends "Run complete:
shipped to main" at 14:30 (`logs/nightly-2026-10-21_130001-midday.log`). Data loop published?
**yes** - `logs/datarun-2026-10-22_020001.log` ends "Data loop complete" at 02:24, PUBLISHED CLAIMS
PASS (1272 tests), 498 scored. Evidence base at `1m` = **34 rows, newest 2026-09-21 (31 days ago,
bound 40), 4 effective** - inside the bound, no action. Priority 0: `allow_auto_apply` **false**,
4 of 8 effective observations. Top open roadmap item: **0.13, the Factor Lab** (opened 2026-10-12,
**age 10 days**) - not worked; its day is Friday 2026-10-23, and the owner queue outranks it today.
**Tests:** before **2981 passed / 1 skipped** (364s), after **2988 passed / 1 skipped** (353s) -
+7, which is exactly the seven new tests in two existing modules, no failures and nothing removed.
Gates 2 and 3 pass (dry-run exit 0; `index.html` 651,277 bytes; both payloads `node --check` clean
with their opening assignments; `research.html` 55,107 bytes).
**Owner queue / rotation:** took `OWNER_FOCUS.md`'s **top open item, 2026-10-22** - one
pre-registered stop-loss test on the bot's rule. It is a BUILD day and this was a build item, so
nothing was displaced. No proposal was waiting for a decision (`write_brief.open_proposals()`
returned `[]`). **Deferred:** the 2026-10-21 item (the rank-continuity gate and the rest of the
screener's half of the path to real money), the 2026-10-20 item's parts 2-6, and the Factor Lab.

### Did

**The owner asked for exactly one thing and the value is in its being exactly one.** Register a
stop-loss variant of the trading rule before building it, build that one, report it beside the
unstopped rule, never sweep the level. All four held: `bot-rule-v1.1-stop10` and the code that runs
it are in **one commit** (513c364) and no ledger row for it existed before, so the git history is
the pre-registration.

**The rule was read from the paper, not from memory.** Han, Zhou & Zhu (2016), "Taming Momentum
Crashes: A Simple Stop-Loss Strategy", SSRN 2407199, DOI 10.2139/ssrn.2407199 - 52-page PDF
downloaded and read with `pypdf`; construction on printed pp. 5-6, Table I Panels A-C on pp. 6-7,
turnover and stop frequency pp. 8-9. **Two things the reading corrected:** the DOI resolves through
Crossref to SSRN posted content and **no journal** - it is a working paper, and the *Journal of
Financial Markets* in the owner's item is Kaminski & Lo (2014), a different paper. And the paper's
rule is not "sell at the close 10% down": it triggers on the day's **lowest** price, waits a day,
and sells at a limit of `(1-L/2)(1-L)P0` - 85.5% of the month's starting price at L = 10% - or at
that day's close if the whole day trades below it.

**So four differences are registered under `construction_note`, each because it changes when the
stop fires:** the trigger here is the close (the daily panel carries no intraday range, so this
fires *less* often than the paper's); the sale is at the triggering close, not the next day's limit
(so it exits *sooner and higher*, and exits names the paper's band would have kept); cash earns
nothing where the paper earns T-bills; and this is the long side of a 25-name screen, not long-short
momentum deciles.

**The prediction, registered before the run:** less negative worst month *and* maximum drawdown in
**every** period, excess about unchanged or slightly lower, turnover higher.

**It failed in two of the three periods, on the one thing a stop is for.**

| | 2012-2019 (96 mo) | 2020 - mid-2023 (42 mo) | mid-2023 - 2026 (38 mo) |
|---|---|---|---|
| Worst month, no stop -> 10% stop | -8.13% -> **-8.41%** | -17.27% -> **-13.96%** | -5.27% -> **-8.36%** |
| Max drawdown, no stop -> stop | -14.55% -> **-15.43%** | -20.62% -> **-18.65%** | -7.52% -> **-14.13%** |
| Excess vs the average scored stock | -0.148% -> -0.240% | -0.701% -> -0.690% | -0.220% -> -0.411% |
| Turnover a year | 1.42 -> **2.00** | 1.74 -> **3.05** | 1.35 -> **2.42** |
| Positions stopped | 180 | 164 | 125 |

Both figures improved only in 2020 - mid-2023, the period containing March 2020, where the stop
turned a -17.27% month into -13.96% with all 25 positions stopped. **In the held-out years it made
the deepest fall nearly twice as bad.** Across all 176 months it changed the result in 174 and
**helped in 82** - fewer than half; in the ten worst months for the unstopped rule it helped in 7,
in the 58 losing months in 36. The mechanism is the obvious one: a stop pays for the falls it avoids
with the rebounds it misses, and a run of ordinary 10% dips and recoveries is what it loses on.

**The construction was checked against the paper's own frequency figure rather than assumed
correct.** 469 positions closed over 176 months, median 1 a month, none at all in 58 of them, 25 of
25 in March 2020 - **8.1%** of positions a month in 2012-2019 and **18.5%** in 2020 - mid-2023,
against the paper's **10.05%** of winner positions over 1926-2013, higher where the period was more
volatile, which is also what the paper reports about volatility (its stopped share correlates 68.49%
with market volatility for the winners, p. 9).

**What shipped:**
- **`registry.yaml`: `bot-rule-v1.1-stop10`**, with the paper's figures, the four differences and
  the prediction. A test asserts there is **exactly one** registered stop level and that the arm's
  rule is otherwise identical to `bot-rule-v1.1`, so the comparison can never quietly become two
  changes at once or a sweep.
- **`backtest_v2_returns.daily_wide`** is now the one place the daily panel is reshaped, and
  `month_end_prices` takes its rows from it. That is what makes the stop measurable against **the
  same price series the forward return divides** - so a month in which no stop triggers reproduces
  the unstopped simulation **exactly**, which is asserted rather than hoped for. A stop rule handed
  no prices **raises** rather than silently simulating no stop under the stopped name.
- **`summarise_evaluation`** now records the worst month, the universe's worst month, the months
  with a stop and the positions stopped, for every evaluation - the figures a stop is judged on were
  not being kept.
- **The research page** carries the arm's table beside the unstopped rule's, with the construction
  and a **generated** verdict sentence. The test feeds it three fabricated pairs - better
  everywhere, worse everywhere, mixed - and checks it says a different thing for each; a sentence
  that could only ever report success would be checking plumbing, not the claim (0.11's lesson).
- `research/2026-10-22-stop-loss-on-the-bots-rule.md` and
  `research/measurements/2026-10-22-stop-loss-arm.{py,json}`.

**The 47-minute rerun was budgeted for and run in the session**, not left for someone else: editing
`v2_experiments.py`, `backtest_v2_returns.py` and the registry moved the code fingerprint, so all 31
rows went stale. The two evaluations were run first (so the owner's answer existed before the long
tail), then the 29 experiments through `scripts/rerun_experiments.py`. **Checked rather than
assumed, as the 2026-10-18 session established:** 30 of 31 ids compared with their previous run at
full precision - **0 fields differ, largest numeric difference 0.0**. The check is not vacuous: it
saw **192 new fields** on `bot-rule-v1.1` (the worst month, the drawdown and the per-month stop
count, which were not being kept) and no changed value. `allow_auto_apply` and `METHOD_VERSION` are
untouched - nothing about *how* an existing comparison is measured changed, which is what that
check establishes rather than asserts.

**Three existing guards fired on the first draft of the page, and all three were right - so the fix
went the other way round, twice.** (a) The no-advice guard (`BANNED_TERMS`, priority 4) refuses the
registered entry's wording, and this page's own tripwire refuses the phrase "in every period",
which the registered prediction contains. **A registered prediction may never be edited once it has
run** (`registry.yaml`'s own rule), and editing one after seeing its result is the single most
corrosive thing this project could do to its record - so the *page* states the construction and the
prediction in its own words, page-safe, and points at the registered text by path, while RESULTS.md
and the research note quote it verbatim. A test asserts the page does **not** quote it, still names
all four differences, and carries both paths. (b) `test_results_md_marks_every_known_outcome_row`
forbids "on the same months" anywhere in RESULTS.md - method version 1's description of the pairing,
corrected 2026-10-16. The stop arm's registered prediction says its two *portfolio rules* are
compared "on the same months", which for a portfolio simulation is simply true, there being no
per-stock intersection to take. The assertion is now **scoped to the report's own prose**: every
verbatim registered field is stripped first, and a second assertion **fails if no registered field
carries the phrase any more**, so the exclusion is deleted along with the wording that needed it.
Anything the report writes itself is still scanned. That is the narrower-but-not-weaker shape; a
blanket string ban that a quoted prediction can trip is a guard that will eventually be deleted
rather than scoped.

**Looked at the page at 1440 and 375** (`scripts/shot_research.py`, new - `shot_dashboard.py` only
knows about `index.html`): no horizontal overflow at either width; the two tables read as one
comparison; on mobile the turnover column sits one swipe right inside the page's existing `.tw`
scroll container. **Both tables gained a "Deepest fall" column** - the verdict sentence cites the
deepest fall and the page was not showing it.

### Evidence / research
- **Han, Y., Zhou, G. & Zhu, Y. (2016)**, "Taming Momentum Crashes: A Simple Stop-Loss Strategy",
  SSRN working paper 2407199, DOI 10.2139/ssrn.2407199 (Crossref: SSRN posted content, no journal).
  CRSP 1926-2013, equal-weighted six-month momentum deciles. At L = 10% (Table I Panel C): mean
  2.32% a month against the unstopped 0.99%, sd 4.61% against 6.01%, Sharpe 0.504 against 0.165,
  skewness +1.54 against -1.18, worst month -15.37% against -49.79%. Stop frequency 10.05% of
  winner positions a month; winners' monthly turnover 39.11% -> 49.08% (pp. 8-9). The paper
  attributes most of the loss reduction to the **losers** portfolio - the side this screener does
  not have.
- **Kaminski, K. & Lo, A. (2014)**, "When do stop-loss rules stop losses?", *Journal of Financial
  Markets* 18, 234-254 - the reference behind the bot's decision to carry no stop, and the correct
  owner of the journal name in the owner's item.
- **Measured here** (`research/measurements/2026-10-22-stop-loss-arm.json`, both rows on one code
  version and one data vintage): the table above. Backtest observation only, not Evidence for any
  change (rule 5, until 2027-02-11).

### Methodology changed
- **None.** No weight, metric, threshold or score moved, and nothing in the bot changed. This was a
  registered test whose result is reported, which is what the item asked for.

### Tried and rejected
- **Trying 15% or 20% as well.** The paper reports all three and 15% is its headline, so a sweep
  would have looked like thoroughness. It is the one way this test could have produced a number
  nobody should believe: a stop level chosen by what 176 months of one market rewarded is fitted,
  and the engine's rules already forbid registering a variation of a near-miss. The owner's item
  says "one variant only"; a test now asserts it.
- **Reading the failure as "stops are bad".** It is the long side of a 25-name screen; the paper's
  gain comes mostly from the short side it does not have, and 42 to 96 months cannot distinguish a
  0.1-0.2%-a-month difference from zero. The note says so in both directions.
- **Registering a claim in `claims.py` for the new page sentences.** The register's surfaces are
  enumerated (summary / overview / README / drilldown) and `research.html` is not one of them, so
  adding one is a real extension rather than a line - see **Next**. The sentences are checked by
  behaviour in `tests/test_research_page.py` meanwhile, which is in the publish gate.

### Next
- **Give `claims.py` a `RESEARCH` surface** so the research page's computational sentences are in
  the register like every other public page's, with
  `test_every_detect_pattern_still_matches_its_surface` reading the generated `research.html`.
  Today's two new sentences (the stop's construction and its verdict) are the first entries;
  `overview.research_engine` already exists for the methodology page's version.
- Then `OWNER_FOCUS.md`'s **2026-10-21** item 1: the rank-continuity gate in `check_run_health.py`
  (FAIL below 0.90 correlation with the last published run, more than 8 new top-25 names, or more
  than 3 top-25 names below #40). It is the first step of the screener's half of the path to real
  money and it has a test specified in the item.

## 2026-10-22 (owner-run, afternoon) - an improvement wave across both repos: the bot's safety and measurement, the screener's input gates, and which free data adds evidence

**Health (rule 8):** last code session ran? **yes** - 06:00 shipped (the stop-loss test). Data loop
published? **yes** - 02:00, HEALTH PASS, claims 1,272 passed, 498 scored. Evidence base at `1m` =
**34 rows, newest 2026-09-21 (31 days, bound 40), 4 effective**. Priority 0: `allow_auto_apply`
**false**, 4 of 8. Top open roadmap item: **0.13, the Factor Lab** (opened 2026-10-12, **age 10
days**) - not worked; the owner's request outranked it. The 13:00 slot stood down for the owner
marker, as designed.

**Owner:** *"go and find something in these tools to improve ... maybe even coming up with a better
plan ... so much data out there."* Answered with five agents in isolated worktrees and merged here.

### Screener (this repo)
- **The trading bot's input gates** (screener-bot ROADMAP 1.3-1.5, 1.9; `OWNER_FOCUS` 2026-10-21 items
  1-4): rank continuity, the unscored share, price dates (`stock_detail[t].price_date`,
  `scoring_schema` 16 - tonight's run refetches), and data-loop git hygiene (fail closed on a bad pull,
  push only the data commit). **Replayed against all 72 published pages since 07-28: the four
  known-bad runs fail, no ordinary day does.** `data-run.ps1` was parse-checked here
  (`Parser::ParseFile`: OK) - the agent could not run PowerShell. `tests/test_input_gates.py` (75),
  in the publish gate. CLAUDE.md row 0.5/0.7 and the data-loop gates paragraph updated.
- **Free data that adds evidence** (`research/2026-10-22-free-data-that-adds-evidence.md`): the scarce
  thing is months, not data. Three free, licensed sources add independent evidence or fix a recorded
  weakness - **SEC 13F** (quarter-end prices for departed members, acquisitions included), **FINRA
  short interest** (history from at least 2017-12, contradicting the "no free history" class for
  `short_interest_ratio`), **SEC XBRL 2009-2011** (about 22 untouched months for backtest v2). Three
  pre-registerable tests written, none registered: **Proposal status OPEN** for a later session.

### screener-bot (owner-run; its SESSION_LOG has the detail)
- Order safety (2.3-2.5), account safety (2.1, 2.2 built not switched on, 2.6, 2.7, 2.10, 2.11) and
  measurement (3.3, 3.4, 3.6, 3.7 part) merged and joined: sells send the bot's own quantity; the
  emergency stop keeps a wholesale close under its own name; the completeness check asks whether
  each planned buy went in (a 95%-invested floor fired on a correct rebalance and was replaced at
  merge). 475 tests, rehearsals clean, CI green. The setup panel now uses finished sessions only.

### Next
The data-scout proposal (register tests A-C or refuse them) is the open decision. Then the 2026-10-21
OWNER_FOCUS item's parts 5-6 (a data-loop-only `signals.json` the bot trades from; the audit trail).


---

## 2026-10-23 (06:00) - the open proposal decided: the registry cannot hold any of the three tests, and one of them would have manufactured a false null

**Health (rule 8, all five):** last code session ran? **yes** - 2026-10-22 06:00 ended "shipped to
main" (the 13:00 slot stood down for the owner marker, as designed). Data loop published? **yes** -
02:28 today, HEALTH PASS, 498 scored, claims gate 1,360 passed. Evidence base at `1m` = **35 rows,
newest 2026-09-22 (31 days ago, bound 40), 4 effective**. Priority 0: `allow_auto_apply` **false**,
4 of 8 effective observations. Top open roadmap item: **0.13, the Factor Lab** (opened 2026-10-12,
**age 11 days**) - **deferred**, see below.
**Tests:** before **3063 passed, 1 skipped**; after **3064 passed, 1 skipped** (one new guard).
**Owner queue / rotation:** today was the lab's Friday. A proposal was waiting for a decision
(1 day open), which the prompt and `OWNER_FOCUS.md` item 1 both put ahead of the focus, so **the
Factor Lab experiment is deferred**. Nothing lab-side was started, so nothing is half-done.

### Did

**Decided the open proposal in `research/2026-10-22-free-data-that-adds-evidence.md` - and the
decision is a refusal of all three registrations as written, with the research left standing.**
The note asked a later session to register three pre-registered tests in
`research/experiments/registry.yaml`, "each in its own commit before any of its data is built".
It was written by a different session, so this is the second reader the routine is built around.
Three things were measured that the note had not; both scripts are committed beside it.

**1. The FINRA claim is true - re-run, not taken.** Twelve settlement dates fetched
(`research/measurements/2026-10-23-finra-short-interest-probe.py` + `.json`). 2014-2016 return
**403**; **2017-12-29 onward return 200**, 15,495 rows rising to 20,516 by 2025. `marketClassCode`
on 2017-12-29: **NYSE 3,102, NNM 2,355, ARCA 1,433** beside OTC 7,152 - exchange-listed rows are in
the file four years before the June 2021 FINRA's own page states, so the note's correction of FINRA
stands. `daysToCoverQuantity` is numeric and positive on **15,494 of 15,495** rows then and
**20,516 of 20,516** in 2025, and all seven large members probed (AAPL, JPM, MSFT, XOM, JNJ, KO,
GE) are present on every served date. Two details the note should have carried: the files are
**pipe-delimited despite the `.csv` extension** (my first probe read one column), and the field
**floors at 1.00** - AAPL and JPM both read exactly 1.00 on 2018-12-31 - so it is censored below.

**2. The decisive finding: the registry cannot hold any of the three, so the instruction is not an
executable action.** Measured by running the engine's own call sites rather than reading the YAML
(`research/measurements/2026-10-23-can-the-registry-hold-these-tests.py` + `.json`). The committed
registry has exactly **two** shapes: `experiments:` - 29 entries, **29 of 29** carrying a
`transform` whose `kind` is one of seven known kinds - and `evaluations:` - 2 entries, each with a
`rule` and no transform. None of A, B or C is either. Shaped as the note writes them,
`apply_transform` raises **`KeyError: 'factor_weights'`** and the bare `e["transform"]["kind"]` read
raises **`KeyError: 'transform'`** - unguarded at `v2_experiments.py:1122`, `:1156` and `:1180`
(`report`, `summary_for_page`) and in three places in `tests/test_engine_calibration.py`. Following
the instruction would have reddened the suite and the public research page, not pre-registered
anything.

**3. Worse than an error, for test B: registering first would have manufactured a false null.**
`base_config()` is `v2_config(config.yaml)`, which zeroes every name in
`backtest_v2.UNMEASURABLE_METRICS` - `short_interest_ratio` is in it - and `backtest_v2._record`
NaNs every field in `pit_record.FIELD_CLASSES["no_free_history"]`, which holds `shortRatio` and
`shortPercentOfFloat`. A `kind: metrics` transform *could* put the weight back, and the entry would
then **run without error on an all-NaN column**: zero coverage, renormalised away, a "not
distinguishable from zero" row in the ledger. Pre-registration exists to stop data-dependent
choices; here it would publish a null that is a property of the missing panel and says nothing about
short interest. Build the panel, confirm the metric is scored, *then* register - the opposite order
to the proposal's.

**The three verdicts** (full reasoning in the note's new section 7):

- **Test A, 13F quarterly prices for the survivorship question: REFUSED.** Its own `void_if` needs a
  price for **>= 80%** of departed member-quarters and both coverage rates the note measured sit
  below it - **60 of 86** acquired members name-matched (69.8%) and **29 of 39** 13F prices within 1%
  of Yahoo (74.4%) - with the better identification routes it proposes left **unmeasured**. It would
  commit an "L"-effort build whose own stated outcome may be "void". The price panel also carries a
  long list of choices made while looking at the data (filer thresholds, splits inferred from
  aggregate share counts, hand-checked deal prices, Shumway's -30%), which a pre-registered
  *hypothesis* does not protect. And `only-size-extended` is already queued, is monthly, and can
  **falsify** the survivorship explanation - the decision-relevant direction. Do the cheap test first.
- **Test B, short interest: ACCEPTED in substance, REFUSED in form.** The only one of the three
  testing a metric that **carries live weight** - 13.5 of Revisions' 100 points, **1.35 composite
  points** (`config.yaml:125`, verified) - and which v2 scores at 0 as unmeasurable. Its data is
  real and now verified across ~104 months. Refused in form for findings 2 and 3 above and because
  its pre-registration is malformed: its `prediction` is a null while its `refuted_if` is a
  condition on the **sign** of the point estimate, so under its own predicted null it is declared
  refuted with probability **0.500** while the prediction holds with 0.466. The note now carries a
  five-step build order, licence check first, with the verdict bands rewritten disjoint and keyed to
  effect size.
- **Test C, the XBRL phase-in months: REFUSED** on the trade. The 22 months buy **9.8%** of
  precision on 2012-2019 (SE 0.01123 -> 0.01013; smallest IC at `t >= 3`, 0.0337 -> 0.0304) for a
  v2 record rebuild to 2009 plus a full rerun. Its confirm band fires with probability **0.808**
  under the null and it leaves **16.2%** of outcomes with no verdict at all (confirm at `|t| < 1.304`,
  refute at `|t| >= 2.174`). And the months are **selected on size** - the XBRL phase-in was mandated
  by public float above $5bn (SEC Release 33-9002, the note's own citation) and a member enters only
  if `pit_record` builds its record - so they are precisely the months that could not be pooled into
  the engine's one near-miss, Size alone in 2012-2019. What would reopen it is written in the note.

**Corrected a false statement on the project's own record, in four places, without creating the
leak the correction invites.** "Short interest has no free history" appears in `pit_record.py`'s
`no_free_history` comment, `backtest_v2.py`'s `UNMEASURABLE_METRICS` comment and
`research/2026-10-09-pit-fundamentals-census.md`, and is repeated in
`research/2026-10-15-pit-record-validation.md`. All four now say no free panel is **in use** and name
the measurement. **The class membership is deliberately unchanged:** `backtest_v2._record` NaNs
everything in that class, so moving a field out without supplying its history makes every historical
month read **today's** value - a look-ahead leak inside the very metric Phase 13 once fixed for a
fallback cohort. Because the correction is itself what invites a future session to reclassify, it
ships with a guard: `test_short_interest_stays_unmeasured_until_its_panel_is_built` requires the two
fields and the metric's v2 weight to move **together or not at all**. Mutation-tested as the routine
requires - unmutated passes, and all three mutations (both fields reclassified, one field
half-moved, the metric un-zeroed) fail with the right message.

### Evidence / research

- Measured today, both committed: `research/measurements/2026-10-23-finra-short-interest-probe.py`
  (12 FINRA settlement dates; the exact 403/200 boundary at 2017-12-29, row counts, market-class
  counts, per-symbol days-to-cover) and
  `research/measurements/2026-10-23-can-the-registry-hold-these-tests.py` (the registry's two
  shapes; the engine's own call sites against the note's entries; the probability each entry's own
  verdict bands fire under the null it predicts; the precision 22 months buy).
- SEC Release 33-9002 (2009), *Interactive Data to Improve Financial Reporting* - phase 1 covers
  large accelerated filers with public float above $5bn, periods ending after 2009-06-15. The note
  cites it; the consequence it did not draw is that test C's months are **size-selected**.
- Boehmer, Huszar & Jordan (2010), *JFE* 96(1), 80-97; Hong, Li, Ni, Scheinkman & Yan (2015), NBER
  WP 21166; McLean & Pontiff (2016), *JF* 71(1) - test B's priors, **not re-read today** and still
  flagged in the entry as from abstracts and secondary summaries. Reading them from the PDFs is step
  5 of the build order, not something this decision rests on.
- **No backtest or IC number from this system justified anything here** (rules 4 and 5). The only
  quantities used are data availability, the engine's code paths, and arithmetic on each entry's own
  stated standard error.

### Methodology changed

- **None.** No weight, metric, threshold or category moved; no score, rank or payload byte changed.
  The decision refuses registrations and corrects documentation. `METHODOLOGY_CHANGELOG.md` has no
  entry today, correctly.

### Tried and rejected

- **Registering test B as a `kind: metrics` experiment anyway**, which is the one registry shape
  that could nominally express "weight this single metric". Rejected because it is the trap, not the
  route: `v2_config` zeroes the metric and `_record` NaNs its inputs, so the entry would run clean
  and report a null caused by the absent panel. A test that cannot see its own variable is worse
  than no test, because its null looks like evidence.
- **Reclassifying `shortRatio` out of `no_free_history` on the strength of the FINRA finding**, which
  is the obvious-looking follow-through and is a look-ahead leak: `_record` NaNs that class, so a
  field removed from it without a panel is read at today's value in every historical month. Guarded
  by a test rather than left to the next reader's judgement.

### Next

- **Run the owed research-engine rerun: all 31 rows, in one session.** The ledger was **already 31
  rows stale at baseline**, from commit `502ca2b` (2026-10-22), which added `_price_bar_date`,
  `regularMarketTime` and `published_price_date` to `factor_engine.py` and `pit_record.py` - both in
  `RESULT_PATHS`. That change is additive and display/gating-only (`published_price_date` is called
  by `check_run_health`, never by scoring; the two fields are classed `price_history`, which v2
  restates per month), so the rerun **cannot move a number** - the same category as the 2026-10-18
  provenance rerun, which measured 0 differences across 29 rows, largest 0.0. It is provenance
  hygiene, not a result. `python scripts/rerun_experiments.py --status` reports them; `--limit 4` is
  about 8 minutes and fits one foreground call, so budget ~8 calls and commit the ledger after each.
  A *partial* rerun leaves rows at mixed fingerprints inside one method version, so it is all 31 or
  none - do not start what the session cannot finish.
- After that, the deferred items in order: **0.13, the Factor Lab** (age 11 days today); then
  `OWNER_FOCUS.md` 2026-10-20 item 6's two free survivorship fixes, which force the same rerun and
  so should be batched with it.
- **If someone wants test B built:** the five-step order is in the note's section 7.2. Step 1 is the
  licence, and it is genuinely first - my probe used `cdn.finra.org`, i.e. the **website**, whose
  terms forbid derivative works and building a database from it. The route that permits publication
  is the Query API under the Equity Specific Terms. If the API does not serve the exchange-listed
  rows, nothing from this test may appear on a public page.

## 2026-10-23 (13:00 session) - the lab's two unchecked experiments second-read: all 21 reference figures exact, every measured figure reproduced, and the number that keeps Size at 5% is now checked

**Health (rule 8, all five):** last code session ran? **yes** - today's 06:00 ended "Run complete:
shipped to main" at 06:35. Data loop published? **yes** - 02:28 today, HEALTH PASS, 498 scored.
Evidence base at `1m` = **35 rows, newest 2026-09-22 (31 days ago, bound 40), 4 effective**
(`_n_observations` 4, `_n_raw_observations` 35 - the raw count is the one not to quote). Priority 0:
`allow_auto_apply` **false**, 4 of 8 effective observations. Top open roadmap item: **0.13, the
Factor Lab** (opened 2026-10-12, **age 11 days**) - advanced, see below.
**Tests:** before **3064 passed, 1 skipped**; after **3064 passed, 1 skipped**. Gates 2, 3, 4 pass
(dry-run clean; `index.html` 651,844 bytes, both payload files parse under `node --check` and open
with their own assignment; working tree clean). `scripts/rehearse_publish.py` **not run and not
needed**: the diff is research notes, measurement scripts, `OWNER_FOCUS.md` and `plan/`, and
touches nothing the page is built from.
**Owner queue / rotation:** `OWNER_FOCUS.md` 2026-10-20 **item 2** taken and its outstanding half
**closed**; **item 5** closed as already-done. No proposal was waiting
(`write_brief.open_proposals()` returns `[]`). **Deferred: the research-engine rerun** the 06:00
session named as Next - the argument is below, under *Tried and rejected*, because deferring an
earlier session's Next on the same day needs one.

### Did

**Second-read Factor Lab experiments 16 and 17, the two that had no reader but their author.**
`OWNER_FOCUS.md` item 2 asks for exactly this and names what it means: reopen the paper, re-read
every reference figure against `factor_lab.py`, and check in `git log` that the tolerances were
committed before the comparison ran. Experiment 15 was done the night it shipped; 16 and 17 were
merged after it on the owner-run night of 2026-10-20 and have been uncovered since. Both were
checked three ways, and a fourth was added - recomputing every *measured* figure from the source
files with no import of `factor_lab`, so a construction error would disagree rather than be
reproduced. **Nothing needed correcting.** Both measurement scripts are committed.

**Experiment 16, `size-given-profitability`** (Asness, Frazzini, Israel, Moskowitz & Pedersen 2018)
- the one that matters most, because its verdict is what kept the Size weight at 5% on 2026-10-20.
The publisher's open-access PDF was obtained (the CBS Research Portal copy; ScienceDirect answers
403 to anything that is not a browser and that was **not** worked around) and **all seventeen
reference figures are exact** against Table 3 Panel A, printed p. 495: the Small - Big column
0.23/0.46/0.54/0.42/0.44 with t 1.21/3.02/3.44/2.82/2.87, the Quality - Junk ends 0.62 (t 4.78) and
0.42 (t 2.78), and SMBQ 42 bps t 3.18 quoted verbatim from the text on the same page. The single
**derived** figure is derived correctly - the paper's column "4" minus column "Big" runs
0.28/0.19/0.25/0.30/0.25 down the five rows and their mean is **0.254**. The second implementation
reproduces every measured figure (SMBQ **0.3918**, fourth-minus-biggest **0.2414**, the five rows,
both quality spreads), gives **10 within and 7 outside** on the committed tolerances - the published
count, with the misses exactly where the verdict note says - and recomputes the pre-registered
criterion at **0.1661 a month, t 2.3641** against its bar of 0.136 and 1.5, **-0.1066 (t -0.70)**
since 2013. So "not refuted" is right. `research/2026-10-20-size-weight.md` section 12.

**Experiment 17, `piotroski`** (Piotroski 2000, on Open Source Asset Pricing). **All four reference
figures exact** in Table 3 Panel B, printed p. 17: 0.078, 0.313, 0.235, t 5.594, with the firm-year
counts (14,043 / 396 / 1,448) and the abstract's "23% annual return between 1976 and 1996" as
quoted. Measured side reproduced over the same 252 months: **26.07 / 30.41 / 4.34 / t 1.5175**, the
authors' value-weighted cut **12.58 (t 3.64)**, and the end deciles' average scores - the stated
reason for the miss - **2.69** and **7.53**. Section 7 of that note also makes two claims about
periods no tolerance covers, so they were measured too: "t 1.38 and -0.55" after the paper is right,
and the value-weighted gap is the larger in **all four** of the page's presets.
`research/2026-10-20-lab-osap.md` section 8.

**Both pre-registrations are genuine, checked in `git log` rather than taken on the commit
message.** `39a6727` (22:05:47) carries experiment 16's seventeen figures, its tolerance reasoning
and the criterion with `"verdict_note": ""`, and touches only `factor_lab.py` and its tests - no
`lab_data.js`, so no comparison existed; `8cd9ce7` is the result 24 minutes later. `ce78f7d`
(21:09:10) does the same for experiment 17, with the message recording that Google Drive was
refusing the OSAP release at the time so not even a header had been seen; `534433a` is the result 90
minutes later. Each tolerance commit is an ancestor of its result commit.

**Three corrections, none of them to a number:**

- **`plan/factor-lab.md` row 7c called experiment 17 "experiment 15"** - which is the JKP one. Both
  of its commit messages say the same (history, so not fixable there). `EXPERIMENTS` and the code
  comment have always had it as 17. Fixed in the plan, with the slip recorded so the commit
  messages do not read as evidence of a different experiment.
- **OSAP's own `SignalDoc.csv` names the wrong panel** for `PS`: `Key Table in OP = 3A` beside
  `T-Stat = 5.594`. Those cannot both be right - 5.594 is Panel **B**'s; Panel A is one-year
  *market-adjusted* returns and reports High-Low 0.230 with t **5.590** and ends of -0.096 and
  0.134. The experiment cites Panel B, where all four of its figures actually are, so **the lab is
  right and more precise than the library it draws on.** Recorded in the note so a reader comparing
  the page with `SignalDoc.csv` does not conclude the panel was mis-cited.
- **One `open_details` figure is on a different basis from the comparison beside it.** "Among the
  largest third, high minus low 15.2%, t 1.88" is exact (Table 4, Large Firms, p. 19), but Table 4
  reports market-adjusted returns where the page's main comparison is raw; the right contrast is
  Panel A's all-sizes 23.0%, not Panel B's 23.5%. The sentence's point - the effect is concentrated
  in small and medium companies - holds on either basis and no number on the page changes, so this
  is recorded rather than edited (editing it would rebuild `lab_data.js` and bump `ASSET_VERSION`
  for a wording change).

**Also checked against the screener rather than the paper:** experiment 17's `in_the_screener`
sentence says the score is 16% of Quality, Quality 22% of the composite, "about 3.5 points", halved
where the valuation score is below the middle. `config.yaml` has `piotroski_f_score: 16`,
`quality: 22`, `piotroski_conditional.reduction_factor: 0.5`; 0.16 x 22 = 3.52. Correct as printed.

**All seventeen lab experiments are now second-read.** Item 2's second sub-item is clear, and item 5
(lab row 4b) was closed as already-done - it shipped as experiment 16 on 2026-10-20 and the line had
simply never been closed.

### Evidence / research

- **Asness, C., Frazzini, A., Israel, R., Moskowitz, T.J. & Pedersen, L.H. (2018).** "Size matters,
  if you control your junk." *Journal of Financial Economics* 129(3), 479-509.
  DOI **10.1016/j.jfineco.2018.05.006**. Read from the publisher's open-access PDF today, Table 3
  Panel A and its text, printed p. 495; sample July 1957 - December 2012. Figures as tabulated
  above. Two caveats re-confirmed: the paper gives **no** t-statistic for the fourth-size-group
  difference, and it does **not** state its quintile breakpoints.
- **Piotroski, J.D. (2000).** "Value investing: the use of historical financial statement
  information to separate winners from losers." *Journal of Accounting Research* 38 (Supplement),
  1-41. DOI **10.2307/2672906**. Read from the University of Chicago GSB Selected Paper 84 reprint
  (January 2002): Table 3 Panels A and B (pp. 16-17), Table 4 (p. 19), sections 2.3 and 3.1.
- **Chen, A.Y. & Zimmermann, T. (2022).** "Open source cross-sectional asset pricing." *Critical
  Finance Review* 11(2), 207-264. `SignalDoc.csv` for `PS` in the October 2025 release - the source
  of the panel-naming discrepancy above, and of the `signallag` column used to check the average
  scores of the two end deciles.
- **Kenneth R. French Data Library**, "25 Portfolios Formed on Size and Operating Profitability"
  (202608 CRSP vintage), value-weighted monthly block, 758 months from 196307.
- Both measurements are committed and rerunnable from the gitignored library files:
  `research/measurements/2026-10-23-size-given-profitability-second-reading.py` (+ `.json`) and
  `research/measurements/2026-10-23-piotroski-second-reading.py` (+ `.json`).
- **No backtest or IC number from this system was used** (rules 4 and 5). Every quantity here is a
  figure printed in a paper, a published portfolio return, or arithmetic on those.

### Methodology changed

- **None.** No weight, metric, threshold or category moved; no score, rank or payload byte changed.
  `METHODOLOGY_CHANGELOG.md` has no entry today, correctly. The session's result is that an existing
  decision - Size stays at 5% - is now resting on figures a second reader has checked, and that the
  twenty-one figures behind two published replications are confirmed.

### Tried and rejected

- **The research-engine rerun the 06:00 session named as its Next: deferred, deliberately, and
  this is the reason.** All 31 rows are stale against the tree (commit `502ca2b` touched two files
  in `RESULT_PATHS`), and the honest case for rerunning is provenance currency, not a number.
  Three things measured before deciding: `scripts/rerun_experiments.py --status` reports **"rows
  level with each other: yes"**, so the report is not flagging a mixture and nothing public is
  inconsistent; the **full suite is green at 3064** passed, so no provenance test is red; and the
  06:00 session's own argument is that the change *cannot move a number*. That is precisely the
  case the rerun script's docstring was written to exclude - *"A change that cannot move a number
  ... leaves every row stale and level - which is not by itself a reason to spend 47 minutes
  rerunning."* Against that, a 47-minute foreground wait (background jobs are forbidden; see the
  settled row) is most of this session. And the lab's day is **fortnightly** - 2026-10-30 is an
  even ISO week, so the next lab Friday is 2026-11-06 - whereas the rerun is resumable at
  four-row granularity by any later session and costs nothing to carry. **The one thing that would
  change this:** `OWNER_FOCUS.md` item 6's two free survivorship fixes force the same rerun, so
  whoever takes them should do the rerun in the same session - one rebuild, one rerun - rather
  than spending it twice.
- **Editing the Table 4 sentence on the lab page to name its basis.** Rejected for today: it is
  not wrong, no number moves, and it would force a `lab_data.js` rebuild and an `ASSET_VERSION`
  bump for a wording change. Recorded in the note instead, where the next session that touches the
  experiment will find it.
- **Starting experiment 18.** The lab's Friday was the nominal focus and item 2 outranked it, but
  the honest reason is narrower: with ~2.5 hours and two unchecked experiments already merged, a
  reader for those is worth more than an eighteenth result nobody has checked either. Said here so
  the trade is visible rather than inferred.

### Next

- **The research-engine rerun, batched with `OWNER_FOCUS.md` 2026-10-20 item 6** - the two free
  survivorship fixes (167 renamed/reformatted member-months, 773 `no_filings` months under
  predecessor CIKs). Both touch `RESULT_PATHS`, so they force the rerun the deferral above leaves
  owed: one rebuild, one rerun, all 31 rows, `python scripts/rerun_experiments.py --limit 4` per
  foreground call with the ledger and report committed after each. Do not start what cannot be
  finished in the session - a partial rerun leaves mixed fingerprints inside one method version.
- Then `OWNER_FOCUS.md` item 3, the claims mutation audit: `plan/claims-mutation-audit.md` lists
  the next 12, and the last pass found **15 of 38** mutations got through.
- The lab's next experiment (18) is unstarted; next non-retrospective Friday **2026-11-06**, and
  item 7 asks for `lab_data.js`'s first paint (~2.9 MB) to be measured in the same session.

## 2026-10-24 (06:00 weekend session) - the screener's half of the plan to real money is finished: only the gated run is tradable, and the run a page came from can now be checked against its own inputs

**Health (rule 8, all five):** last code session ran? **yes** - 2026-10-23 13:00 ended "Run
complete: shipped to main" at 13:40, tag `good/2026-10-23-1340`. Data loop published? **yes** -
2026-10-23 02:28, HEALTH PASS, 498 scored, all four of the 2026-10-22 input gates green (Spearman
0.994, 1 new in the top 25, 498/498 prices on the expected session). No 02:00 run today, correctly
- Saturday. Evidence base at `1m` = **35 rows, newest 2026-09-22 (32 days ago, bound 40), 4
effective** (`_n_observations` 4; the raw count of 35 is the one not to quote). Priority 0:
`allow_auto_apply` **false**, 4 of 8 effective observations. Top open roadmap item: **0.13, the
Factor Lab** (opened 2026-10-12, **age 12 days**) - deferred, see below.
**Tests:** before **3064 passed, 1 skipped**; after **3109 passed, 1 skipped** (+45: three new
modules - 22, 14 and 8 - plus one new contract test). Gates 2, 3, 4 pass. **`scripts/rehearse_publish.py` run and
PASSED** - the diff touches `generate_dashboard.py`, which the four gates cannot see.
**Owner queue / rotation:** `OWNER_FOCUS.md`'s **top open item, 2026-10-21, items 5 and 6** - both
built, and the item is now **closed and moved to Done**. **Deferred: the research-engine rerun**
batched with the 2026-10-20 item 6, which the 10-23 13:00 session named as its Next; the argument
is under *Tried and rejected*.

### Did

**`OWNER_FOCUS.md` 2026-10-21 item 5 - only the gated run is tradable.** The hole this closes is
an asymmetry, the same shape as the two previous retrospectives found. `dashboard_data.js` is
published by **two** paths: the 02:00 data loop, behind seven gates (no synthetic data, a
fetch-failure ceiling, `check_run_health.py` with its rank-continuity, empty-record and
price-freshness rules, a size floor, a real `node --check`, the opening assignment, and
`check_published_claims.py`); and any code session that runs the full screener, behind **none of
them** - its four ship gates are about the repository, not about the run. The bot reads the page as
Pages serves it and cannot tell the two apart.

So the gated run now writes a second, small file and the bot trades that. `scripts/write_signals.py`
is called from `data-run.ps1` after the published-claims gate and before the commit, and
`signals.json` is in `$DataArtifacts`. `scripts/nightly-screener.ps1` never writes or stages it -
**that asymmetry is the whole mechanism**, so one test asserts the code runner does not mention it
and another that no other module in the repo writes it.

Three decisions in it worth recording, because each could reasonably have gone the other way:

- **It is derived from the gated payload, not from a second read of the run directory.** The
  ranking in `signals.json` is then by construction the ranking those gates inspected. A second
  read could disagree with the page, and the disagreement would be invisible.
- **Gate verdicts are `pass` or `warn`, and there is no `fail`.** Two gates publish on a documented
  fallback when the tool they need is missing (node, pytest), and a consumer is entitled to know a
  check did not run - `all_gates_ran` is false then. A `fail` verdict could not be true in a file
  only reached after every gate accepted, so the parser refuses one.
- **It is not a gate.** A failed write leaves the previous `signals.json` in place, which is the
  safe state: it carries an older `run_timestamp` and the bot's own freshness rule declines it.
  Discarding a published run over a 10 KB side file would cost a day of evidence - the `node` and
  pytest fallback precedent read the same way.

Verified end to end against the real 2026-10-23 published run: 40 names in rank order beginning
**EXPE APA HST INCY SNDK** - the same five that run's commit subject names - 9,673 bytes.
**No `signals.json` is committed today**, deliberately: a hand-written one would assert a gate
stack that never ran. The first is written by the **2026-10-26** 02:00 run.

**Item 6, first half - the page now says which run it came from.** `kpis` carries `run_id`,
`git_sha` and `raw_sha256` beside the `methodology_version` published since 2026-10-22.
`run_fingerprint.py` is the one computation, and the rule in it that matters:
`generate_dashboard._find_raw_fetch` deliberately falls back to a **neighbouring run's**
`00_raw_fetch.parquet` so a cache-hit run can still show price targets - right for a price target,
a lie in an audit trail. The fingerprint hashes this run's own file or reports `None`; the same
rule as the health gate, which has judged only the run's own artifacts since 2026-10-16. Git's
literal `"unknown"` is published as `null` for the same reason: an absent identifier must not read
like a value.

**Item 6, second half - and the half that makes the hash mean anything.** `kpis.raw_sha256` names
a file in `runs/`, which is gitignored scratch that `prune_artifacts.py` empties after twenty runs
- **about four weeks**, after which the published hash points at nothing and a questioned trade
cannot be checked against its inputs at all. `scripts/archive_run.py` copies each **published**
run's raw fetch, log, config and universe to `%LOCALAPPDATA%\ScreenerArchive`, re-hashes the copy
and verifies it against the hash the page publishes, and appends a manifest line. Outside OneDrive
for three separate reasons (it rewrites folder mtimes - the reason `latest_run_dir` stopped
trusting them on 2026-10-16; Files On-Demand can leave a placeholder that needs the network months
later; a quota and a cloud-side recycle bin this project does not control) and outside the repo
because inside it would be either a 1.7 MB daily commit or a gitignored sibling of the very folder
the prune empties. **Never pruned**, on purpose - four-week retention is the problem being fixed -
so it reports its size each run instead and says so loudly past 5 GB. Measured: **1.9 MB a run,
about 0.5 GB a year.**

**Verified on the machine, not only in tests (rule 11).** The archiver was run against the
2026-10-23 published run: `C:\Users\smitc\AppData\Local\ScreenerArchive\2026-10-23_725915b457e4`
holds five files, the archived parquet re-hashes to
`4ffe57bab10221552028684c9939718817a6f120d839181e21d536a64480275d` - byte for byte the value the
regenerated payload carries and the value `sha256sum` prints for the source - and a second call
reports `skipped`. Nothing scheduled had to change: the archiver is invoked from `data-run.ps1`
after the push, so the existing task picks it up on Monday.

**Nothing on the page moved.** The rehearsal regenerated the next run's page with this tree's code,
and the rebuilt payload differs from the live one in **exactly three keys - `run_id`, `git_sha`,
`raw_sha256` - and nothing else**; every other top-level key is identical, including all 498 table
rows and all the stock detail blocks. No score, rank, sentence or visible element changed, and the
three new fields are not displayed anywhere. `rehearse_publish.py`: 355 page tests pass,
`check_published_claims` passes at 1,406, published files restored byte for byte.

All three new modules are in the data loop's publish gate (`check_published_claims.py` MODULES),
because `signals.json` is an artifact that very run writes and publishes. `conftest.py` protects
`signals.json` as a tripwire from the day it exists - a test that wrote it would be asserting a run
was gated when it was not.

### Evidence / research

- **A demonstrated failure, not a citation.** The justification here is structural and checkable,
  in the form CLAUDE.md names: *"a documented failure - a real user-facing problem you can
  demonstrate."* Two facts, both verifiable in this repo today: (1) `scripts/data-run.ps1` contains
  seven refusals before its publish and `scripts/nightly-screener.ps1` contains none of them, while
  both paths can rewrite `dashboard_data.js` - so "the page was gated" is false of one of the two
  publishers; (2) `scripts/prune_artifacts.py` keeps the newest **20** runs and the data loop makes
  five a week, so the file `kpis.raw_sha256` names is deleted about four weeks after it is
  published. Neither needed a backtest or an IC number to establish.
- **The asymmetry rule followed here is the project's own, twice-measured one** (CLAUDE.md, "The
  data loop's gates, and the asymmetry to keep checking"): the strongest checks guard the path that
  publishes least often, and when you compare the two runners the weaker one is usually the bug.
  2026-09-18 (the payload parse) and 2026-10-02 (the claims gate) are the two prior instances; this
  is the third, on the consumer rather than on the page.
- **No backtest number and no IC number was used** (rules 4 and 5). Nothing here touches a weight,
  a metric, a threshold or a score.

### Methodology changed

- **None.** `METHODOLOGY_CHANGELOG.md` correctly has no entry today: no weight, metric, threshold,
  category or rule that can move a score was touched. The rebuilt payload's only difference from
  the live one is three provenance keys.

### Tried and rejected

- **Writing `signals.json` as a gate of its own.** Rejected: a failure to write a 10 KB side file
  would then discard a run that passed every check that matters, costing a day of evidence for the
  improvement engine. The previous file staying put is already the safe state, because it is stale
  and the bot's freshness rule declines it.
- **Reading the run directory a second time to build `signals.json`.** Rejected: the point of the
  file is that it describes the run the gates accepted, and a second read of
  `05_final_scored.parquet` could disagree with the payload those gates inspected. Deriving it from
  the gated payload makes the agreement structural rather than something a test has to notice.
- **Committing a first `signals.json` by hand so the file exists in the repo today.** Rejected on
  the same ground the whole item rests on: it would carry gate verdicts for a gate stack that never
  ran.
- **Showing the new provenance on the page's Data Provenance panel.** Not done: the owner item asks
  for the identifiers in `kpis` and in an archive, and a visible surface is a separate question
  that would need a registered claim. Noted here rather than quietly added.
- **The research-engine rerun batched with the 2026-10-20 item 6 (two free survivorship fixes)** -
  what the 10-23 13:00 session named as Next. Deferred, and this is the reason: `OWNER_FOCUS.md`'s
  **top** open item was the 2026-10-21 one, whose items 5 and 6 were the only two things left in
  this repo's half of the owner's "make me money" plan, and both the prompt and that file put the
  top open item ahead of a previous session's Next. The deferral costs nothing that is accruing:
  `scripts/rerun_experiments.py --status` still reports the rows level with each other, so nothing
  public is inconsistent, and the rerun is resumable at four-row granularity. It stays the next
  session's work, and should be done in the same session as the survivorship fixes - one rebuild,
  one rerun.

### Next

- **The research-engine rerun, batched with `OWNER_FOCUS.md` 2026-10-20 item 6** - unchanged from
  2026-10-23. The two free survivorship fixes (167 renamed/reformatted member-months, 773
  `no_filings` months under predecessor CIKs) both touch `RESULT_PATHS`, so they force the rerun:
  one rebuild, one rerun of all 31 rows, `python scripts/rerun_experiments.py --limit 4` per
  foreground call, ledger and report committed after each. ~47 minutes. Do not start what cannot be
  finished - a partial rerun leaves mixed fingerprints inside one method version.
- Then `OWNER_FOCUS.md` 2026-10-20 item 3, the claims mutation audit
  (`plan/claims-mutation-audit.md` lists the next 12; the last pass had 15 of 38 mutations get
  through).
- **For the owner, in the other repo:** nothing in `screener-bot` reads `signals.json` yet, so
  until its `bot/signals.py` is pointed at it the file accrues beside the page and changes nothing.
  That is bot work and nightly sessions here do not do it.

## 2026-10-24 (13:00 midday session) - the two free data fixes backtest v2's record was missing, and the rule I first wrote for one of them was wrong in both directions

**Health (rule 8, all five):** last code session ran? **yes** - 2026-10-24 06:00 weekend ended
"Run complete: shipped to main" at 06:42, tag `good/2026-10-24`. Data loop published? **yes** -
2026-10-23 02:28, HEALTH PASS, 498 scored, all four input gates green; no 02:00 run today,
correctly - Saturday. Evidence base at `1m` = **35 rows, newest 2026-09-22 (32 days ago, bound
40), 4 effective** (`_n_observations` 4; 35 is the row count and not the one to quote). Priority
0: `allow_auto_apply` **false**, 4 of 8 effective observations. Top open roadmap item: **0.13,
the Factor Lab** (opened 2026-10-12, **age 12 days**) - deferred; the next experiment is 18 and
the next non-retrospective Friday is 2026-11-06, so nothing about it is time-critical today.
**Tests:** before **3109 passed, 1 skipped**; after **3129 passed, 1 skipped** (+20, one new
module). Gates 2 and 4 pass; gate 3 via `tests/test_payload_parse_gate.py` (19) and the fact that
no payload file is in this diff - `node --check` is refused by this session's tool guard, and the
runner re-runs gate 3 itself. **`scripts/rehearse_publish.py` run and PASSED**, because the diff
changes `SCREENER_OVERVIEW.md`, which `generate_dashboard` embeds in the page.
**One test needed changing and the reason is worth reading** (see *Tried and rejected*):
`test_every_headline_figure_carries_its_interval_from_the_ledger` pinned the 2026-10-20 audit's
literal confidence interval, which a legitimate data fix moves.
**Owner queue / rotation:** `OWNER_FOCUS.md`'s **top open item (2026-10-20), sub-item 6, first
half** - the two free data fixes - which is also what the 06:00 session and the 10-23 13:00
session both named as Next. No proposal is waiting for a decision
(`write_brief.open_proposals()` returns `[]`), so section 3's first job was clear.

### Did

**`OWNER_FOCUS.md` 2026-10-20, sub-item 6, first half: backtest v2's two free data fixes, done
together because both are in `v2_experiments.RESULT_PATHS` and so force one rebuild and one
rerun.** The write-up is `research/2026-10-24-v2-pre-2014-renames.md`; every figure below is in
`research/measurements/2026-10-24-v2-data-fix-effect.json`.

**Re-measured before touching anything (the prompt's rule), and nothing had moved since
2026-10-20:** 48,282 member-months in 2012-2019, 31,919 built, `no_current_record` 15,108,
`no_price` 434, `symbol_reused` 48, `no_filings` 773 - every figure identical. The 167 months the
note attributes to an uncarried rename reproduce symbol by symbol (TYC 56, ACE 48, SAI 21, MHP 16,
PCS 16, BFB 6, CMCSK 3, SAIC 1).

**Fix 1 - the renames the membership history cannot carry.** `Inputs.current_symbol` matches a
month's member to today's record by SEC registrant number and nothing else, which is right: only
the CIK separates a rename from a removal. But the revisions had no CIK column until **2014-05**,
so a company renamed before then has no registrant number anywhere in the history and the matcher
has nothing to match on. `universe_history.PRE_REVISION_CIKS` supplies three - BFB (BF.B written
without the dot, so it normalises to a symbol that never meets today's BF-B), MHP (S&P Global) and
SAI (Leidos) - each a *unique* name match against SEC's own `cik-lookup-data.txt` from the
2026-10-20 note, each verified here to resolve to a company in today's universe, to a target that
was **not itself a member in any affected month**, and to a price series that covers those months.

**The other five are refused, and the refusals are the rule rather than exceptions to it.** ACE→CB
(48 months) and TYC→JCI (56): the target symbol was itself a member in **every** affected month -
Chubb Corp and Johnson Controls Inc, both later absorbed by the registrant that now holds the
symbol - so one current record would have to stand for two members, and the symbol's price history
before the merger belongs to the other company. CMCSK→CMCSA (3): a second share class of a company
already scored, which the 2026-10-21 share-class decision says is scored once. PCS→TMUS (16): the
note's own identity check - SEC public float against the symbol's close - reports **MISMATCH**
(median ratio 0.353, 2 of 9 dates), because MetroPCS's 1:2 reverse split *and* $4.05 special
dividend are not a clean split and the filed share count cannot be reconciled with the adjusted
price. SAIC (1): the note resolves it to Leidos's registrant but takes its prices from today's
SAIC, a different company.

**Fix 2 - each member's predecessor registrant.** `facts.parquet` is fetched by today's CIK, so a
holding-company reorganisation or redomicile leaves every month before it asking a registrant that
had filed nothing - **773 member-months** in 2012-2019, among them the whole window for Apache,
BlackRock and Exxon. Read off the parquet the pattern is unmistakable: each symbol's filings begin
the day its current registrant started filing (XOM 2026-08-03, BLK 2024-11-06, APA 2021-05-07,
DIS 2019-05-08, CI 2019-02-28, DD 2017-11-06, AVGO 2018-06-14, GOOG/GOOGL 2015-10-29, MDT
2015-02-27). **Those filings were public at the time** and are the company's own under the name it
then had, so joining them is the opposite of look-ahead - leaving them out is a hole in the
point-in-time record, and the only reason it exists is a fetch keyed to a symbol's *current*
registrant. `research/measurements/2026-10-24-predecessor-facts.py` fetches every CIK the history
ever gave a current member (22 pairs, one `companyfacts` request each, 12 seconds) and
`backtest_v2._load_facts` concatenates the side file; `pit_record` then selects by filed date
exactly as before, so there is no handover date to maintain.

**The rule I first wrote for admitting a pair was wrong in both directions, and measuring it is
what showed that.** The draft refused any pair whose predecessor kept filing after the successor
started, on the theory that a real reorganisation hands over cleanly. On the 22 pairs it refused
**three genuine reorganisations** - Apache Corp (96 months), EIDP (68) and Dow Chemical all still
file as debt-only subsidiaries of their own holding companies, overlapping by 1,280, 548 and 2,639
days - and **admitted two different companies**, SanDisk Corp and Twenty-First Century Fox, each of
which stopped filing before today's holder of its symbol was created. What replaced it is a cut
with a proof: keep only rows filed **strictly before** today's registrant's first filing, because
`pit_record` takes the latest filing, so a kept row can only ever be chosen for a month in which
today's registrant had filed *nothing at all* - exactly the gap being filled. Strictly before, so a
same-day handover resolves to the successor (Alphabet and Google Inc both filed on 2015-10-29).
A second bound stops a registrant that lost the symbol years earlier - IQVIA held `Q` for three
months in 2017 and filed until 2025 - by dropping rows filed more than 120 days after the last
month the history named it. 24,365 of 30,063 fetched facts kept; `CB:20171` and `JCI:53669` cut to
nothing, which is the two whose predecessor is a company today's registrant absorbed.

**What the rebuild changed.** All 176 months rebuilt, four shards at a time, and compared against
the pre-fix census and scores copied aside first:

| | 2012-2019 | 2020-2026 |
|---|---|---|
| `no_filings` | 773 -> **91** | 177 -> **13** |
| `no_current_record` | 15,108 -> **15,065** (-43) | unchanged |
| Records built | +725 | +164 |
| Member-months **scored** | +398 | +98 |
| Month-stocks removed | **0** | **0** |

The -43 is exactly the three renames (SAI 21, MHP 16, BFB 6). The scored names are DIS 94, DD 52,
GOOG 49, AVGO 45, CI 31, MDT 25, GOOGL 22, LDOS 21, APA 17, SPGI 16, BLK 11, KHC 7, BF-B 6, FOX 1,
FOXA 1 in the earlier period; BLK 61, APA 25, BG 11, XOM 1 in the later one.

**Two things in that table needed explaining rather than reporting, and both check out.**

- **725 built but 398 scored.** The coverage filter declined the rest, and the largest single case
  is instructive: **Exxon gains a record in all 96 months and is ranked in none**, because CIK
  34088 files **no `OperatingIncomeLoss` tag at all** - measured: of the fetch's fourteen input
  concepts it is missing `operating_income`, `gross_profit` and `cost_of_revenue`, and operating
  income feeds ROIC, EV/EBITDA, operating margin and operating leverage together. BlackRock also
  lacks both current-asset items, Cigna those and operating income. This is a property of a
  filings-only rebuild, not of this fix, and the right response is that a filter declined a record
  it could not cover. It also names the next free improvement: more XBRL tags per input, no new
  data source (note section 5).
- **12,800 of 31,410 already-scored composites moved, the largest by 7.42 points.** That is the
  mechanism working. The engine ranks each metric **within the stock's sector**, so adding a name
  to a month re-ranks every other name in that sector. The symbols that moved most often are
  exactly the sector-mates of what was added - Netflix 96 months, Omnicom 95, AT&T 95, Comcast 94,
  Verizon 92, the Communication Services peers of Disney and the two Alphabet lines; then a block
  of banks beside Cigna and BlackRock. In 2020-2026 it is 6,779 moved and the top twenty-five are
  **all** financials, beside a BlackRock that was absent from 58 of 80 months. A version of this
  that left peers untouched would be the broken one.

**What has already been published rested on the old panel.** Until today v2's 2020-2026 rankings
contained no Exxon in any month and no BlackRock in 58 of 80, and its 2012-2019 rankings contained
no Exxon, Apache, BlackRock, Disney, Cigna, DuPont, Broadcom, Alphabet or Medtronic for most or all
of the window. Every research-engine result was measured on that panel, which is why the rerun
below is not optional bookkeeping.

**The rerun: all 31 rows, and nothing a reader is told has changed.** A rebuild of
`cache/v2_metrics` stales every row's data fingerprint, and the engine refuses to compare results
across one, so all 29 experiments and both trading-rule evaluations were rerun in the same
session - 45 minutes, `STALE REMAINING: 0`. **Every stored figure in all 31 rows moved**, which is
what a data change does and is the reason the rerun is compulsory rather than tidy. What did not
move is the conclusion:

- **0 of 27 alternative weightings beat the current one**, exactly as before.
- The baseline's rank correlation with next month's return is still indistinguishable from zero in
  both eras: **-0.00188** in 2012-2019 (was -0.00275) and **-0.00030** in 2020-2026 (was
  -0.00003), pooled t **-0.14** (was -0.18), against a design that can see about 0.032.
- **No verdict changed on any row.**

`SCREENER_OVERVIEW.md` was regenerated, because the methodology page quotes those baseline ICs
from `research/experiments/summary.json` (`run_screener._research_engine_sentence`) - the one
sentence that moves reads **-0.002** where it read -0.003. That is the whole public-facing effect
of the day's work: no weight, metric, threshold or score changed, and the live dashboard payload is
untouched.

### Evidence / research

- **A documented failure, demonstrable in the repo, which is the form CLAUDE.md names.** Not a
  citation and not a backtest number: `cache/v2_census_presample.json` records 773 member-months
  where v2 has the company and no filings, and `data/sec/pit/facts.parquet` shows why - XOM's
  earliest `filed` under today's CIK is **2026-08-03**, BLK's **2024-11-06**, APA's **2021-05-07**.
  The census's own 167 rename-months reproduce symbol by symbol. Nothing here needed an IC or a
  backtest figure to establish.
- **SEC EDGAR XBRL `companyfacts` and `cik-lookup-data.txt`** - the registrant numbers, the entity
  names that told the three real reorganisations from the symbol handovers (`IR`'s earlier CIK
  prints as *Trane Technologies plc*, `Q`'s as *IQVIA Holdings*), and every fact's `filed` date,
  which is what makes the join point-in-time rather than look-ahead.
- `research/2026-10-20-free-survivorship-sources.md` sections 1, 2 and 6 - the census of what is
  missing, each departure's cause, and the identity checks the five refusals rest on (PCS's float
  MISMATCH at median ratio 0.353 is the one that refused a rename on measured grounds rather than
  on judgement).
- `research/2026-10-15-pit-record-validation.md` - why this is worth a session: the fundamentals
  half of v1's look-ahead is the larger half, **72.5% of name-months**, so the completeness of the
  point-in-time record is the thing v2 exists to get right.
- **No backtest number and no IC number justified anything** (rules 4 and 5). The engine figures
  above are reported as what the rerun produced, not as a reason for any change; no weight, metric
  or threshold moved.

### Methodology changed

- **None.** No weight, metric, threshold, category or scoring rule was touched, and the published
  ranking is byte-identical - the only generated public file that moved is one sentence of
  `SCREENER_OVERVIEW.md`, and it moved because the research engine's own baseline number did.
  `METHODOLOGY_CHANGELOG.md` correctly has no entry.

### Tried and rejected

- **"Refuse a predecessor pair whose registrant kept filing after the successor started."** My own
  first rule, rejected because measuring it on all 22 pairs refuted it in both directions: it
  refused Apache (96 months), EIDP (68) and Dow Chemical, all three genuine reorganisations whose
  predecessor survives as a debt-only subsidiary, and it admitted SanDisk Corp and Twenty-First
  Century Fox, which are different companies that merely stopped filing early. Replaced by the cut,
  which has a proof rather than a theory behind it. `tests/test_v2_data_fixes.py` fails if the
  overlap refusal returns.
- **Carrying ACE->CB and TYC->JCI (104 of the 167 rename-months, the two largest).** Refused: the
  symbol they would claim was itself a member in every affected month, so one current record would
  have to stand for two members, and the symbol's price history before the merger belongs to the
  other company - which is the defect `check_price_series_integrity` exists to catch. The 104
  months stay unscored and the reason is written down.
- **Carrying PCS->TMUS (16 months).** Refused on a measured ground, not a judgement: the
  2026-10-20 note's own identity check puts the symbol's close against the filed public float and
  reports MISMATCH, 2 of 9 dates agreeing, because MetroPCS's reverse split *and* special dividend
  are not a clean split and `cache/splits.parquet` carries only the 0.5.
- **Adding ICE and ETN to the rename constant** (32 `no_filings` months from reorganisations before
  2014-05, which the general rule cannot see). Not done, and recorded in the note rather than left
  implicit: a hand-maintained list of companies is the thing this session was trying to avoid, and
  32 member-months does not pay for one. If a later session wants them, the note says exactly what
  it would take.
- **Loosening the coverage filter so the 327 built-but-unscored records would rank.** Rejected
  outright - a filter declining a record it cannot cover is the filter working. The right fix is
  more XBRL tags per input, which is a different and better-defined piece of work.
- **Leaving `tests/test_research_page.py` as it was.** One test went red, and it is the only test
  this session changed, so the reasoning matters. Its last two assertions pinned the **2026-10-20
  audit's own literal confidence intervals** - `"&minus;0.023 to +0.017"` and
  `"&minus;0.027 to +0.027"` - in the page text. Today's data fix legitimately moved the record
  those intervals are computed from, to -0.022..+0.018 and -0.028..+0.027, so both went red while
  the page was exactly right. **This is not weakening a test to pass a gate.** The same test's
  loop, five lines above, already asserts each era's interval string **computed from the ledger**,
  which is the strictly stronger check: it fails if the page disagrees with the ledger at *any*
  data version, where a literal can only say whether today's data is the data it was written
  against. What the literals uniquely covered was the number *formatter*, and that is now asserted
  directly on three values (a negative, a positive, and zero). The lesson is this project's own,
  from the 2026-10-18 dirty-marker bug: a tripwire that fires when nothing is wrong is one that
  gets ignored.

### Next

- **The extended universe, which is the rest of `OWNER_FOCUS.md` 2026-10-20 sub-item 6.** Register
  `only-size-extended` in `research/experiments/registry.yaml` **first and alone in its commit**,
  with the one-sided reading from `research/2026-10-20-free-survivorship-sources.md` section 7
  (falsified at >= +0.020; a fall below +0.014 does **not** confirm, because the acquired members
  are still missing) and the paired month-by-month statistic, then follow that note's section 8.
  Budget the rebuild and the ~45-minute rerun in the same session as today did: registering it
  stales all 31 rows.
- **The cheaper win first, if a session wants one:** more XBRL tags per input in
  `research/measurements/2026-10-09-xbrl-point-in-time-census.py`'s `INPUTS`. 725 records were
  built in 2012-2019 and 398 scored; the gap is `gross_profit`, `operating_income` and
  `cost_of_revenue` tags the filers did not use under the names the fetch asks for. It needs no new
  data source and it would lift the same records - **but it is also a data-version change, so it
  carries the same rebuild and rerun.** Do it in one session with something else that forces them,
  never on its own.
- Then `OWNER_FOCUS.md` 2026-10-20 item 3, the claims mutation audit
  (`plan/claims-mutation-audit.md` lists the next 12; the last pass had 15 of 38 mutations get
  through) - unchanged from the 06:00 session's Next, and still untouched.


## 2026-10-25 (06:00 weekend session) - the positive control is a ledger row: the page's "the test can see a signal" is now a number, rerun with everything else

**Health (rule 8, all five):** last code session ran? **yes** - 2026-10-24 13:00 midday ended "Run
complete: shipped to main" at 15:49, tag `good/2026-10-24-1549`. Data loop published? **yes** -
2026-10-23 02:28, HEALTH PASS, 498 scored, all four input gates green; **no 02:00 run on 10-24 or
today, correctly - Saturday and Sunday** (next is Monday 2026-10-26, which brings the first
`signals.json`). Evidence base at `1m` = **35 rows, newest 2026-09-22 (33 days ago, bound 40), 4
effective** (`_n_observations` 4; 35 is the row count and not the one to quote). Priority 0:
`allow_auto_apply` **false**, 4 of 8 effective observations. Top open roadmap item: **0.13, the
Factor Lab** (opened 2026-10-12, **age 13 days**) - deferred; the next experiment is 18 and the next
non-retrospective Friday is 2026-11-06, so nothing about it is time-critical.
**Tests:** before **3129 passed, 1 skipped** (369.6s) - identical to the 10-24 13:00 session's after
count, so no drift overnight; after **3149 passed, 1 skipped** (359.3s), **+20, all in the one new
module** (`tests/test_positive_control.py`). Gates 2 and 4 pass; gate 3 via
`tests/test_payload_parse_gate.py` (19) and the fact that no payload file is in this diff -
`node --check` is refused by this session's tool guard, and the runner re-runs gate 3 itself.
**`scripts/rehearse_publish.py` run and PASSED**, because the diff touches `claims.py`, which the
page is built from: 1,426 claims-gate tests, 3,979 category scores and 498 composites reproduced
from the regenerated payload, published files restored byte for byte.
**One existing test needed updating and it was the contract, not the assertion:**
`test_a_limited_rerun_does_experiments_first_and_evaluations_once_they_are_level` pinned
`rerun_experiments.plan`'s return as a 3-tuple. It is now a 5-tuple (controls, and the controls whose
arm is not built), so the test pins the new shape *and* the new ordering - experiments, then
evaluations, then controls, because a control is paired against the baseline and is worth nothing
until that row is current - plus that an unbuilt arm comes back named with the command that builds
it. Nothing was loosened.
**Owner queue / rotation:** `OWNER_FOCUS.md`'s **top open item (2026-10-20), sub-item 4** - the
positive control as a ledger row. No proposal was waiting for a decision
(`write_brief.open_proposals()` returns `[]`), so section 3's first job was clear. Sub-item 4 is the
one the 10-18 and 10-19 sessions both named as Next and nobody had taken since; sub-item 6's
remaining half (the extended universe) is the larger piece and is left as Next with its reason.

### Did

**`OWNER_FOCUS.md` 2026-10-20 sub-item 4 / `plan/research-engine.md` item 4(c): the positive
control is a ledger row.** The thing it fixes is a sentence. Since 2026-10-15 the public research
page has told readers, in the three-bullet answer at the top, that **"The test can see a signal when
there is one"** - with no number beside it, resting on a measurement from 2026-10-15 that was never
a ledger row and whose monthly records have since been rebuilt **twice** (operating income on 10-15,
the pre-2014 renames and predecessor CIKs on 10-24). It was the one claim on that page a reader
could not check, and the page's 28 flat verdicts are the claim that most depends on it: a row of
nulls is exactly what a broken measurement produces.

**Why a null needs this and cannot supply it.** Harvey, Liu & Zhu (2016), *RFS* 29(1) 5-68,
DOI 10.1093/rfs/hhv059 - whose t >= 3 bar this engine adopts, with a Bonferroni correction on top -
raise the threshold for a finding, which necessarily lowers the power to detect one. So the power
has to be shown separately or every "not distinguishable" verdict is uninterpretable. The synthetic
half of that shipped 2026-10-20 (`test_engine_calibration.py`: planted differences come back within
four standard errors). This is the other half: a real arm, built by the real pipeline, carried in
the ledger, rerun with everything else.

**What was registered, and when.** `controls:` in `research/experiments/registry.yaml`, id
`lookahead-filings`, committed in `e55d36a` **before** the arm was rebuilt and before the comparison
ran. The prediction: the arm's paired IC advantage over the baseline lands between **+0.030 and
+0.075** and clears p < 0.05/N with |t| >= 3. The failure condition, also written first: if it does
*not* clear that bar, it is **not** a finding about the screener - it means this design cannot see a
signal of the size look-ahead manufactures, and every flat verdict in the ledger must be re-read as
uninformative until the cause is found. A control's outcome is known by construction, so the entry
carries `outcome_known_at_registration` and `refuse_undisclosed_controls` makes an undisclosed one
impossible - a control that did not say so would be a false pre-registration wearing a real one's
clothes.

**The arm, rebuilt on the current records**, because the one on disk was from 2026-10-15 and pairing
a current baseline against a stale arm would put two data vintages inside one comparison - the thing
the engine refuses everywhere else. `backtest_v2.py --diagnostic-filings-as-of 2026-10-25`, four
shards, 81 months in 44 minutes: **35,470 stock-months against the old arm's 34,741**, so the 10-24
data fix reached this arm too.

**The result: the design sees it, inside the band.**

| Period | Months | The arm's IC | The baseline, same stocks | Difference (95%) | t |
|---|---|---|---|---|---|
| Discovery 2020-01..2023-06 | 42 | +0.0425 | -0.0043 | +0.0467 (+0.0135..+0.0799) | 2.84 |
| **Held out 2023-07..2026-08** | 38 | +0.0602 | +0.0041 | **+0.0561** (+0.0425..+0.0698) | **8.34** |
| **All 2020-01..2026-08** | 80 | +0.0509 | -0.0003 | **+0.0512** (+0.0330..+0.0694) | **5.59** |

Against a bar of p < 0.05/28 = 0.00179 with |t| >= 3, and a `detectable` floor of **0.0373** - the
smallest difference this comparison catches eight times in ten. So the signal is about **1.4x the
floor**, which is the honest version of "it can see this". Three things in that table are worth more
than the verdict: **the baseline on the same stocks is -0.0003** over 80 months, which is the
plainest statement of where the screener stands (with the filings as they actually stood,
approximately nothing; with today's filings, +0.05 a month - the gap *is* the look-ahead, measured);
**the held-out half is the stronger one**, and nothing was tuned to either; and it **reproduces the
2026-10-15 figures it replaces** (+0.0509 against the published +0.050) on records rebuilt twice
since, so that measurement was right - it just could not be checked by anyone, because it was not a
row.

**And the rerun moved nothing, checked rather than asserted.** Registering the control staled all 31
existing rows (`registry.yaml` is in `RESULT_PATHS`), so they were rerun and each newest row was
compared with its predecessor at full precision: **31 ids, 24,796 stored statistics, 0 differences,
largest 0.0** (`research/measurements/2026-10-25-control-rerun-effect.py` + JSON). That was the
expected answer - the diff adds a parallel path and never touches the experiment path - but the
public page quotes these figures and "it should be identical" is how a silent change gets published.

**What was built, not just measured.** Four pieces, each with the failure it prevents:

- **A control is not an experiment, in code.** No `transform`, its own registry section, never in
  the multiple-testing denominator, and no scoring module may read it. This is the 2026-10-23 lesson
  applied before it bit: that session refused three registrations because, written as `experiments:`
  entries, they raised `KeyError: 'transform'` in `report`. And counting the engine's own self-check
  as an alternative would make every real experiment's bar stricter because the engine checked
  itself.
- **The row names its arm by SHA-256.** `content_hash` covers `cache/v2_metrics/` - the
  point-in-time records, which are the *baseline* side of the pairing - and cannot see the arm's
  scores file at all. Without the hash, rebuilding the arm would move every number under a row that
  still looked current. `scripts/rerun_experiments.py` compares it with the file on disk.
- **An unbuilt arm is printed, never run.** `run_controls` raises on a missing arm, which inside the
  scheduler would take a whole 45-minute rerun down over an optional row; a scheduler that says
  which arm to build is better than one that dies.
- **The page states the outcome it got.** `research_page.control_sentence` has three states - seen,
  not seen, and not run on these records - and the "not seen" wording says plainly that every flat
  result on the page is then uninterpretable. A page that could only report the reassuring outcome
  would be worthless. Registered as claim `research.positive_control`, which added **`research.html`
  as a claims surface** (it had none), so `tests/test_positive_control.py` joins the data loop's
  publish gate by derivation - 54 modules now.

**Looked at the page.** The bullet now reads: *"The test can see a signal when there is one. Fed the
filings as they stand today for every past month - information nobody had at the time - the same
test puts that ranking +0.051 of rank correlation a month ahead of the real one over 80 months
(t +5.6), against the +0.037 it could reliably detect. So the flat results above are a finding about
the ranking, not a test that cannot see anything."* `RESULTS.md` gained a "Controls" section with the
arm's hash, its months and its stock-months.

### Evidence / research

- **Harvey, C. R., Liu, Y. & Zhu, H. (2016)**, "...and the Cross-Section of Expected Returns",
  *Review of Financial Studies* 29(1), 5-68, DOI 10.1093/rfs/hhv059. The engine's t >= 3 bar. Used
  here for the argument it implies and that nobody had acted on: a higher threshold for a finding is
  a lower power to detect one, so power has to be demonstrated separately.
- **Measured, this session:** the arm against the baseline, +0.0512 (95% +0.0330 to +0.0694,
  t 5.59) over 80 months, detectable floor 0.0373; held out +0.0561, t 8.34; the baseline on the
  same stocks -0.0003. `research/2026-10-25-positive-control.md`.
- **Measured, this session:** the forced rerun's effect on every existing result - 31 ids, 24,796
  statistics, **0 differences**.
- **Prior measurements this rests on:** the look-ahead's two halves, 63.2% of name-months (prices,
  `research/2026-10-01-lookahead-bias-size.md`) and 72.5% (fundamentals,
  `research/2026-10-15-pit-record-validation.md` s.3), 75.8% together.
- **Not evidence, and not used as any:** nothing here is decision-grade before 2027-02-11 (rule 5),
  and a control is further from it than anything else in the ledger - it is deliberately cheating.
  No weight, threshold or rule changed. What the control does **not** license is also written down
  in the note's section 5: it establishes power at the size it was run at and nowhere else, and a
  *typical* published multi-factor rank IC of 0.02-0.05 is still at or below this design's 0.0373
  floor - exactly as `research/2026-10-15-research-engine-first-batch.md` already said.

### Methodology changed

- **None.** A control cannot support a methodology change by construction, and nothing else in the
  session touched scoring. No `METHODOLOGY_CHANGELOG.md` entry: no score, rank, sentence or
  published number moved except the three figures that are new on `research.html`.

### Tried and rejected

- **A sharded rerun - three processes writing to their own ledger files, merged by a parent.**
  Written (`logs/_rerun_sharded.py`, with a validate-before-merge refusal) when the clock looked far
  worse than it was, and **not used**: I had mis-read the time by nearly an hour, and once there was
  room, the single-process path that is already tested and already documented was obviously the
  right tool. Recorded because the reasoning may be wanted: three processes cannot append to one
  ledger (a row carries 176 monthly figures, far past any atomic write, and the ledger is
  append-only, so a corrupted row is unrecoverable) - but `X.LEDGER` can be reassigned *in a child
  process* without touching the file on disk, so `code_hash` stays identical and the rows come out
  level. If a future session genuinely cannot fit a 45-minute rerun, that is the shape; and it must
  refuse as a whole if any shard is short, because a partial merge is worse than none.
- **Registering the control in `experiments:`.** Rejected before it was written, on the 2026-10-23
  precedent: everything downstream reads `e["transform"]`.
- **Widening `test_the_recorded_caveats_say_whether_fixed_or_which_stage_extends_them`** to accept
  the new claim's caveat. Rejected - that field carries limits a named T-stage will close, and the
  control's two limits (2020-2026 only; the hindsight shrinks across the window, so the difference
  is a *lower bound*) are permanent properties of the arm, not unfinished work. They are stated in
  the registration instead, and printed in `RESULTS.md`.
- **Running the two long jobs concurrently.** Not a rejection so much as a measured mistake, worth
  the next session's attention: the arm's build (4 shards) and the rerun ran together for 15 minutes
  and produced **one** ledger row between them, against ~1.6 minutes a row when the rerun runs
  alone. Run them in series.

### Next

- **The extended universe - the rest of `OWNER_FOCUS.md` 2026-10-20 sub-item 6, and now the only
  open piece of it.** Register `only-size-extended` in `research/experiments/registry.yaml` **first
  and alone in its commit**, with the one-sided reading from
  `research/2026-10-20-free-survivorship-sources.md` section 7 (falsified at >= +0.020; a fall below
  +0.014 does **not** confirm, because the acquired members are still missing) and the paired
  month-by-month statistic, then follow that note's section 8. **Budget the rebuild and the
  ~45-minute rerun in the same session, in series, not concurrently** (see above).
- **The cheaper win, unchanged:** more XBRL tags per input in
  `research/measurements/2026-10-09-xbrl-point-in-time-census.py`'s `INPUTS`. 725 records built in
  2012-2019 and 398 scored; the gap is `gross_profit`, `operating_income` and `cost_of_revenue` tags
  the filers did not use under the names the fetch asks for. It is a data-version change, so it
  carries the same rebuild and rerun - do it in one session with something else that forces them.
- Then `OWNER_FOCUS.md` 2026-10-20 item 3, the claims mutation audit
  (`plan/claims-mutation-audit.md` lists the next 12; the last pass had 15 of 38 mutations get
  through) - named as Next by three sessions now and still untouched. **A session with no owner item
  waiting and no broken loop should take it** rather than make it a fourth deferral.

## 2026-10-25 (13:00 midday session) - the claims register's second mutation pass: four bars nothing could see, a test named for a strictness it did not test, and one claim that was simply false

**Health (rule 8, all five):** last code session ran? **yes** - today's 06:00 weekend session
ended "Run complete: shipped to main" at 08:13, tag `good/2026-10-25`. Data loop published?
**yes** - 2026-10-23 02:28, HEALTH PASS, 498 scored; **no 02:00 run on 10-24 or today, correctly -
Saturday and Sunday**; the next is Monday 2026-10-26, which brings the first `signals.json`.
Evidence base at `1m` = **35 rows, newest 2026-09-22 (33 days ago, bound 40), 4 effective**
(`_n_observations` 4; 35 is the row count and never the one to quote). Priority 0:
`allow_auto_apply` **false**, 4 of 8 effective observations. Top open roadmap item: **0.13, the
Factor Lab** (opened 2026-10-12, **age 13 days**) - deferred again; experiment 18 is unstarted and
the next non-retrospective Friday is 2026-11-06, so nothing about it is time-critical, and this
session took an owner sub-item that outranks it.
**Tests:** before **3149 passed, 1 skipped** (364.6s) - identical to the 06:00 session's after
count, so no drift. After **3152 passed, 1 skipped** (+3: two in `test_stock_summary.py`, one in
`test_capex_lease_note.py`).
**Owner queue / rotation:** `OWNER_FOCUS.md` 2026-10-20, **sub-item 3, "Test the tests"** - the
claims mutation audit. No proposal was waiting (`open_proposals()` returns `[]`). This item had
been named as Next by **three** consecutive sessions and never taken, and the 06:00 entry said
plainly that a session with no proposal waiting and no broken loop should take it rather than make
it a fourth deferral. I did. The morning's own Next - the extended universe - is deferred and
stated below, because it needs a rebuild plus a ~45-minute serial rerun and this slot had under
three hours.

### Did

**`OWNER_FOCUS.md` 2026-10-20 sub-item 3: the second mutation pass over `claims.py`.**
`plan/claims-mutation-audit.md`, rows 39-62. Method unchanged: for each registered sentence that
states a number, make the smallest plausible edit that makes the sentence false, run **only** that
claim's `checked_by`, and record caught or gap. **24 mutations on 11 claims: 15 caught, 4 real
gaps, 5 gaps in the register, and 1 mutation that could not be made because the claim was already
false.** Every gap now fails a registered check, and every mutation was **re-run after its fix and
required to read CAUGHT** - which caught me out once: two fixes first came back as register gaps
because I had written the test and not yet named it in `checked_by`.

**One improvement to the harness, which is the reason the numbers above are legible.** It now runs
the claim's `checked_by` *and* the whole of each module those tests live in, so "nothing in the
suite checks this" and "the suite checks it but the register does not say so" are separate
verdicts. The 2026-10-20 pass had to tell them apart by hand and its rows 35 and 37 record the
distinction informally. Five of this pass's nine gaps were the second kind.

**The four real gaps** - nothing anywhere in the suite saw them (verified: only the one module
references each symbol):

1. **`summary.weakest`, `LARGEST_SIZE_SCORE` 10 -> 11.** The sentence says "at 10 or under, one of
   the largest"; the four registered checks probe Size at 2, 30 and 55, so the bar could have sat
   anywhere between 2 and 30. Now `test_the_largest_company_bar_holds_at_ten_and_not_above_it`.
2. **`summary.weakest`, the sector-median bar `n >= 50` -> `n > 50`.** The relaxed version prints
   the gloss *at* the median - telling a reader a sector-median company is "larger than most of
   its sector". Now `test_the_gloss_is_said_below_fifty_and_not_at_fifty`.
3. **The momentum panic rule's volatility bar, `vol > thr` -> `vol >= thr`.** This is the rule
   CLAUDE.md 0.12 says is set in advance and never re-fitted, and its edge could widen with
   nothing going red.
4. **The capex note's 5% reconciliation, widened to 8%.** The bar exists because Ford files 5.8x
   its capex in loan and lease purchases; without it the row would tell a reader that 577% of
   Ford's capex was vehicles for lease.

**The claim that was already false.** `summary.earnings` asserted - and `_sentence_earnings`'s own
docstring repeated - that "a date at or before the run date is dropped". **No version of the code
has ever done that:** it drops `days < 0` and renders `days == 0` as "the day of this run". So the
mutation that would make the sentence true is the one that *breaks* the behaviour, and a test in
the module (`test_near_horizons_read_naturally["2026-09-29"]`) fails on it - the register simply
did not name it. **The code is the right half and the sentence was corrected**, per the method's
rule 5: a company reporting on the run date genuinely is its next report, and by the paper the
feature is built on (Akepanidtaworn, Di Mascio, Imas & Schmidt 2023, *JF* 78(6)) announcement day
is the one day on which a sell beats its random-sell counterfactual, so dropping the date would
withhold the schedule exactly where the evidence says it matters most. **Nothing a reader sees
changed.** This is the second false claim the audit has found; the first was 0.11's
`context.expected_move`.

**The five register gaps** - the suite had cover, `checked_by` did not name it. Four are the
momentum rule: the three registered checks pinned 0.70, the equal split of freed weight and the
two-condition logic, while the **126-day window, the 24-month window and the configured 20.6%
threshold** - three of the four numbers the sentence names - were held only by tests the register
never mentioned. The module runs in the publish gate either way, because `module_paths()` is
derived rather than transcribed, so nothing was unprotected today. What was missing is the thing
the register exists for: telling the *next* session which test is load-bearing for a public
sentence, before it edits the sentence. The fifth is `summary.confidence`'s configured
stale-filing bar.

**And the limitation that blocked the last of those, fixed rather than worked around:**
`checked_by` can now name `Class::method`. The 2026-10-20 pass recorded "`checked_by` cannot name
a class method" as a known cost and wrote duplicate module-level tests instead; hitting it a
second time made fixing it cheaper, because the only two tests in the whole suite that pin the
configured stale bar are methods of `TestStalenessThreshold`. `_test_functions` in
`tests/test_claims_register.py` now emits both forms, and every other consumer of `checked_by`
already takes `ref.split("::")[0]`, so nothing else needed changing. This also makes the good
tests in `tests/test_metrics.py` registrable - the last pass listed their invisibility to the
register and to the publish gate as an open cost.

**What generalises, and is now written into the plan's standing rules:**

- **A check may not import the number it is checking.** The capex tolerance's only assertion read
  `approx(..., rel=X.LEASE_RECONCILE_TOL)`. Widening the constant widens the assertion with it, so
  the test passes at every value. It is rows 12-13's self-consistency failure in a new shape, and
  the plan's next pass now has it as a **mechanical grep** rather than more hand-picked mutations.
- **A test's name is read far more often than its body.**
  `test_the_threshold_is_strict_and_configurable` probed the bar 0.001 above and 0.001 below the
  reading and never *at* it - the single point that separates `>` from `>=`. A name that claims a
  property the body does not test is worse than no test: three sessions could read it and conclude
  strictness was covered.
- **Finishing a gap means registering the fix.** A new test `checked_by` does not name reads as a
  gap on the next pass and sits outside the publish gate's derivation.

### Evidence / research

- No methodology evidence was needed or used: this pass changed no score, weight, threshold or
  published number. The one judgement call that needed a source was whether to keep a same-day
  earnings date, and the source is the paper the feature already cites and the register already
  records - **Akepanidtaworn, Di Mascio, Imas & Schmidt (2023), "Selling Fast and Buying Slow",
  *Journal of Finance* 78(6)**: sells executed on a holding's earnings-announcement day beat
  non-announcement-day sells by more than +150 bp/year and are the only sells in the sample that
  beat a random-sell counterfactual. That is the case for keeping the date, so the code was kept
  and the sentence corrected.
- The pass's own evidence is the measurement: 24 mutations, each with its verdict in
  `plan/claims-mutation-audit.md`, and each fix shown to fail under its mutation and pass without
  it. That is a documented failure in the mandate's sense - four public sentences whose stated
  numbers could move with nothing going red.

### Methodology changed

- **None.** No weight, threshold, formula or published sentence moved. Three corrections are to
  the project's own record of itself: `summary.earnings`'s `asserts`, the matching docstring in
  `stock_summary.py`, and four caveats recording what each gap was. `METHODOLOGY_CHANGELOG.md` is
  deliberately untouched - there is no methodology change to record, and an entry for a test fix
  would dilute the file that exists to explain real ones.

### Tried and rejected

- **Writing a duplicate module-level test for the stale-filing bar**, which is what the 2026-10-20
  pass did when it hit the `Class::method` limitation. Rejected: the two tests that pin that bar
  already exist, are correct, and are the only ones in the suite that do; copying them to get past
  a parser would leave two copies to keep in step and would not help
  `tests/test_metrics.py`, which the same limitation hides entirely. Teaching the register the
  longer form is ~8 lines and every other consumer already tolerated it.
- **Loosening `test_the_recorded_caveats_say_whether_fixed_or_which_stage_extends_them`.** Two of
  my new caveats failed it because they began "corrected 2026-10-25" rather than "Fixed". The
  contract is right - a recorded caveat must say whether it is closed - so the caveats were
  reworded, not the test.
- **Mutating the trap-flag percentiles by editing their `.get()` defaults.** Equivalent mutants:
  `config.yaml` supplies all four, so the default is dead code and the mutation proves nothing. I
  shifted the effective threshold instead, and separately replaced one config read with a
  constant - which is the mutation that tests "read from config", and was caught.
- **Starting the extended-universe build** (the morning's Next). It needs a records rebuild plus a
  ~45-minute serial rerun of 31 ledger rows, and the ledger is append-only, so a half-finished one
  is worse than none. Under three hours was not a budget for it; see Next.

### Next

- **The extended universe - the only open piece of `OWNER_FOCUS.md` 2026-10-20 sub-item 6**, and
  unchanged from the 06:00 entry. Register `only-size-extended` in
  `research/experiments/registry.yaml` **first and alone in its commit**, with the one-sided
  reading from `research/2026-10-20-free-survivorship-sources.md` section 7 (falsified at
  >= +0.020; a fall below +0.014 does **not** confirm, because the acquired members are still
  missing) and the paired month-by-month statistic, then follow that note's section 8. **Budget the
  rebuild and the ~45-minute rerun in the same session, in series, not concurrently.** A session
  that cannot finish both should not start either.
- **The cheaper win, still unclaimed:** more XBRL tags per input in
  `research/measurements/2026-10-09-xbrl-point-in-time-census.py`'s `INPUTS` - 725 records built in
  2012-2019 against 398 scored, the gap being `gross_profit`, `operating_income` and
  `cost_of_revenue` tags filers did not use under the names the fetch asks for. Same rebuild and
  rerun, so pair it with the item above.
- **The mutation audit's next pass is six items, two of them mechanical** and worth more than four
  more hand-picked mutations: grep the suite for `rel=<module>.<CONST>` and `== <module>.<CONST>`
  where the constant is the number the claim states (row 59's shape), and re-probe every claim
  whose only check reads the committed payload (rows 35-37's shape, plus row 57, which is *caught*
  only by a payload-reading test - so a wrong constant there merges green and first appears when
  the data loop refuses to publish the next morning). `plan/claims-mutation-audit.md`.

## 2026-10-26 (06:00 session) - RESEARCH. The last unresearched weighted metric: `price_target_upside` is 71% price and 29% analyst at the horizon we score it over - and the screener is accidentally on the right side of the one thing the literature actually tested

**Health (rule 8, all five):** last code session ran? **yes** - the 2026-10-25 13:00 midday session
ended "Run complete: shipped to main" at 13:44, tag `good/2026-10-25-1344`. Data loop published?
**yes** - 2026-10-26 02:00, "Data loop complete", **HEALTH: PASS**, 498 scored, top EXPE MU APA HST
INCY; this is the run that wrote the **first `signals.json`**, as the 10-24 entry predicted.
Evidence base at `1m` = **36 rows, newest 2026-09-23 (33 days ago, bound 40), 4 effective**
(`_n_observations` 4 against a gate of 8; 36 is the row count and never the one to quote; the
whole file reads 95 rows / 2026-10-16, which is the number that cannot stand still). Priority 0:
`allow_auto_apply` **false**, unchanged. Top open roadmap item: **0.13, the Factor Lab** (opened
2026-10-12, **age 14 days**) - advanced today rather than deferred, see below.
**Tests:** before **3152 passed, 1 skipped** (376.8s) - identical to the 10-25 midday session's
after-count, so no drift overnight. After **3152 passed, 1 skipped** - no production code changed
today (Monday).
**Owner queue / rotation:** `OWNER_FOCUS.md`'s top open item is 2026-10-20, whose sub-items 1, 2, 4
and 5 are closed; **no proposal was waiting** (`open_proposals()` returns `[]`), which is the first
thing the prompt asks. So the day's focus stood, and I took the Monday research slot in the way
that also advances the top roadmap item: `plan/factor-lab.md`'s "From the lab to the weights" names
five ranked weight questions, and this is the metric-level residual of question 2's category. **One
thing deferred and stated:** sub-item 6's extended universe, which the last two sessions also
deferred - it needs a records rebuild plus a ~45-minute serial append-only rerun, and a research
day was not the session to start it. Sub-item 3's next mutation pass is likewise untouched.

### Did

**Picked the one weighted metric in the composite with no research note of its own.** Everything
around `price_target_upside` has been researched - the category was rebuilt on a 1,022-line note,
`earnings_acceleration` was zeroed, `short_interest_ratio`'s data history was probed. This metric
is mentioned four times across all of `research/` and examined nowhere; the category rebuild's own
table says only "role unchanged". It carries 13.5 of the Revisions category's 100 points, so
**1.35 composite points**, and it is the only metric in that category that is a *level* rather than
a change - with price in its denominator.

`research/2026-10-26-price-target-upside.md`, with
`research/measurements/2026-10-26-price-target-upside.py` + `.json`, and a second committed script
that re-verifies every quoted paper figure against the PDF bytes.

**1. The measurement, from data already on disk - no return, IC or backtest read anywhere.**
Implied upside in logs is exactly `log(target) - log(price)`, so its change decomposes additively.
Reading `pt_mean` and `price` out of **56 committed versions of `dashboard_data.js`**:

- Between consecutive published runs the median absolute log change in the consensus target is
  **exactly 0.0000** while the price moves 0.26-2.7%, and **60-77% of targets are byte-identical**.
  corr(d upside, d price) runs **-0.82 to -0.99**.
- The term structure, which is the part that matters, because daily is not the horizon anything is
  decided on. Beta is the slope of d log target on d log price - 0 means the target never moves,
  1 means it fully tracks the price: **~7d +0.246, ~30d +0.245, ~91d +0.540, ~182d +0.823**. At the
  **30-day** horizon the review cadence and the bot's rebalance both use, **the price move explains
  71.4% of the variation in implied upside** and analysts changing their minds explain 28.6%.
- Within sector the metric is **-0.403** with `jensens_alpha`, **-0.473** with the Risk score,
  -0.317 with the Momentum score, +0.318 with Growth - and **-0.007 with `fy1_revision_3m`**, the
  one metric in its own category that measures an analyst revision. A high implied upside
  identifies a volatile stock that has recently fallen.
- **What it buys: almost nothing.** Redistributing its 13.5 points in proportion moves the median
  stock **4 ranks** and changes the top 25 by **0 names**. The reconstruction is exact first - the
  rebuilt Revisions score is Spearman **1.0** and **0.0 points** from the published one, the
  rebuilt composite 0.9999 - so that is this screener's arithmetic, not an approximation of it.

**2. The literature, read from the PDF bytes, and it did not say what I expected.** Da & Schaumburg
(2011, *JFM* 14(1)) is the one paper that tests this exact construct - "TPER = TP/P - 1", PDF p. 2 -
and the decisive thing in it is a **comparison between two ways of ranking the same signal**: the
**absolute** level across all stocks returns **79 bp/month, t 0.86, not significant**, while the
**within-sector** sort returns **177 bp/month, t 3.55** (both PDF pp. 13-14). Their words: "it is
relative and not absolute valuations that convey the most information to investors."
**The screener scores a sector percentile, so it is on the supported side of the only distinction
anyone has tested** - arrived at by this project's general sector-relative design, not by anyone
reasoning about price targets, and worth writing down before someone "simplifies" that machinery.
Four findings in the same paper cut the other way, three of them about our own universe and
cadence: two-thirds of the first month's profit lands in **two weeks** with nothing in months 2-6
(p. 13); **Large growth is 0.23% risk-adjusted, t 0.37** (p. 28); it decays after decimalization
and Reg FD (p. 3); and the staleness channel is in their data too - "High (low) TPER stocks tend to
have a lower (higher) past return at the one-to-six month horizon" (p. 7).

**3. The mechanism, and an independent cross-check I did not go looking for.** Brav & Lehavy (2003,
*JF* 58(5), DOI 10.1111/1540-6261.00593) find the level is a persistent firm baseline - the long-run
target/price ratio averages **1.28** across 900 firms, 25th/75th 1.20/1.33 (PDF p. 27), so ranking
the level across all stocks largely ranks firm identity, which is why the absolute sort is t 0.86.
And when a gap is wide it closes because **the analyst cuts the target**: "Market prices, in
contrast, barely contribute to this correction phase" (p. 3), "predominantly made by analysts"
(p. 28). Their estimated correction rate of 9% of the gap per week implies **31% / 71% / 91%**
closed by 4, 13 and 26 weeks. **My betas at the matching gaps are 24.5% / 54.0% / 82.3%** - same
shape, slightly slower, from a different vendor and a sample 27 years earlier. That is the best
available evidence that section 1 is measuring analyst behaviour and not a Yahoo refresh cycle,
which was the obvious objection to my own measurement.

**4. Practice points the same way as the mechanism.** Barra USFAST's `Sentiment` factor has three
analyst descriptors and **all three are changes** (revision ratio, change in analyst-predicted E/P,
change in analyst-predicted EPS); there is no price-target descriptor anywhere in 24 style factors.
The Zacks Rank's four components contain no price target either - its "Upside" is estimate versus
consensus estimate, not target versus price.

### Evidence / research

- **Da, Z. & Schaumburg, E. (2011)**, "Relative valuation and analyst target price forecasts",
  *Journal of Financial Markets* 14(1), 161-192,
  <https://academicweb.nd.edu/~zda/TargetPrice.pdf>. First Call, Dec 1996-Dec 2004. Absolute TPER
  sort **79 bp/month (t 0.86)**; within-sector **177 bp/month (t 3.55)**; net of costs 148 bp
  (t 3.67, 17.8%/yr, p. 18); significant with all 15 characteristic controls (p. 21); **Large growth
  0.23% risk-adjusted, t 0.37** (p. 28); two-thirds of month 1 in two weeks (p. 13).
- **Brav, A. & Lehavy, R. (2003)**, *Journal of Finance* 58(5), 1933-1967,
  DOI 10.1111/1540-6261.00593, <https://webuser.bus.umich.edu/rlehavy/BL.pdf>. First Call 1997-99,
  223,016 targets on 6,544 firms. Long-run target/price ratio mean **1.28**, median 1.26, 25th/75th
  1.20/1.33 (p. 27); correction is "predominantly made by analysts" (p. 28); unexpected revision
  2.671 (t 27.0) but **"no reliable relationship" for the expected revision** (p. 29).
- **MSCI Barra USFAST datasheet (March 2015)** and the **Zacks Rank**'s four components, both via
  this project's own prior sourced reading in `research/2026-09-07-...` §2.1-2.2: every analyst
  descriptor either provider ships is a *change*; neither scores a price-target level.
- **Asquith, Mikhail & Au (2005)**, *JFE* 75(2) - **not obtained**; read second-hand from NBER
  w9246 through a text proxy, flagged as such in §2.3, and **nothing in the recommendation rests on
  it**.
- Every figure in section 1 is reproducible from
  `research/measurements/2026-10-26-price-target-upside.py`; every paper figure is re-checked
  against the PDF bytes by `research/measurements/2026-10-26-verify-target-price-papers.py`.

### Methodology changed

- **None, deliberately, and the rule that says so was written before the question was asked.**
  `plan/factor-lab.md`'s reading rules: *"A change needs the lab and the literature or practice
  pointing the same way. Where they disagree the note says so and the weight stays."* Here the
  literature partly supports the construct the screener actually implements (sector-relative, t
  3.55) and practice does not use it at all. That is a disagreement, so the weight stays.
  Materiality says the same from the other side: a cut moves the median stock 4 ranks and no top-25
  name, so it would spend the one-change-at-a-time budget on a rounding error.
- **One proposal left OPEN for a reader who did not write the note** (§6.2): replace the scored
  *level* with a **90-day change in the consensus target scaled by price**, taking its 13.5 points,
  with the level kept display-only. That is the object both camps point at, it is the same move this
  category already made successfully in 2026-09-10 (build `fy1_revision_3m`, do not just delete the
  broken thing), and it cannot be 71% price by construction. The note lists four things the
  reviewing session must do first, and says plainly that **if step 2 finds it duplicates
  `fy1_revision_3m`, the honest outcome is no change.**

### Tried and rejected

- **Cutting or zeroing the weight**, which is where the measurement alone pointed and where the two
  precedents in this category (`sharpe_ratio`/`sortino_ratio`, `earnings_acceleration`) went.
  Rejected by the pre-written reading rule above once Da & Schaumburg turned out to test the
  *sector-relative* level and find it significant on a sample including the S&P 500. I had expected
  to find the metric unsupported; the honest reading is that the screener is accidentally doing the
  supported version of it.
- **Asserting Brav & Lehavy's beta-by-size terciles (1.37 / 1.29 / 1.23).** Reported second-hand and
  directly useful to my argument that the level is mostly firm identity - and I could not find it in
  the PDF in the time available, so it is named in §7 as not asserted and the argument rests on the
  full-sample 1.28 instead. The verification script is what caught this.
- **Asserting a full provider sweep.** A background sweep of MSCI/FTSE/S&P/StarMine/AQR methodology
  documents was commissioned and did not return before the time budget; §3 therefore re-uses this
  project's own two sourced documents and §7 says explicitly that two providers establish that
  practice scores changes, and do **not** establish that no provider scores the level.
- **Starting sub-item 6's extended universe** (the last two sessions' Next). A records rebuild plus
  a ~45-minute serial rerun of an append-only ledger is not a thing to begin on a research day with
  a 09:00 deadline; a half-finished one is worse than none.

### Addendum, written after the above: the delegated reads returned late and they changed the note

Three background paper/practice reads returned **after** sections 1-2 were committed. Two of them
handed over PDFs, so I checked their load-bearing figures against the bytes myself; the rest is
marked unverified in the note's new §3.1 and nothing rests on it. What they changed:

- **The strongest citation in the note now confirms my own measurement, independently.** Green,
  Hand & Sikochi (2024, *Review of Accounting Studies* 29(1), 889-915, open access) run quantile
  regressions on target-price implied returns and report, **verified at PDF p. 18**: "The
  coefficient of **-0.92 on MOM-** in the 90th quantile IRET regression indicates that a 1% more
  negative MOM- is associated with an 0.92% higher IRET, **almost an inverse one-to-one relation**",
  concluding that "an important determinant of analysts' claimed undervaluation is **transitory
  declines in market prices**." That is my 71.4% restated as a slope, on 2001-2017 analyst-report
  text rather than 2026 Yahoo data. Their bound on what survives: the level predicts returns at
  **16-18 cents on the dollar**, only where analysts claim undervaluation, only 1-2 quarters out.
- **And the strongest counter-evidence to my own reading, which is now in the note.** Da &
  Schaumburg's Table 2 Panel B (**verified, PDF p. 9**): in their S&P 500 sample "less than 1.5% of
  these stocks have a target price during the current month and none during the previous two
  months", and the highest-TPER portfolio's recent changes "are dominated by **upgrades**" (55%
  against 22%). Bilinski et al. (2013, *TAR* 88(3), verified) point the same way - adjusting for
  revisions moves the met-target rate only 59.1% -> 43.4%. **So "the metric is stale data" is too
  strong and I am not claiming it.** Mine is a time-series fact (a month apart, the target moves a
  quarter as much as the price); theirs is a cross-sectional one (high-upside names are covered and
  recently upgraded). Both hold, and Brav & Lehavy's 9%-a-week correction is why.
- **Practice turned out to be unanimous on seven primary documents, not two.** Five more rulebooks
  were page-pinned (FTSE Global Factor Index Series v8.8, which governs the Russell Comprehensive
  Factor indexes; S&P Quality; S&P QVM; S&P Momentum): across all of them the strings "price
  target" and "target price" appear **zero times**, and S&P's "risk-adjusted momentum" is a
  realised 12-1 price return over its own realised daily volatility. The one analyst input either
  index house scores is Russell style's I/B/E/S two-year **growth-rate** forecast - and §8.10.1
  **shrinks it toward the sector mean when coverage is thin** (1 analyst -> 1/3 own score, 2 -> 2/3,
  3+ -> full), which is a reliability treatment worth copying whatever happens to this metric.
  Reported but unverified: MSCI's 2025 Analyst Sentiment index *does* use price-target data, as a
  revisions ratio and a change, and **defines no target-to-price descriptor** while defining
  exactly that for earnings, cash flow and sales.
- **The proposal's construction changed as a result.** My first draft scaled the target change by
  the lagged price; MSCI's `PTGF_C` scales by the **target's own mean absolute level** and S&P
  Global MI's documented signal is simply `TP_t / TP_{t-1m}`. Scaling by price would smuggle the
  price back in and partly re-create the defect. Also new: S&P MI reports **86% monthly turnover**
  on a 1-month target change, so the 90-day window is probably the right trade and Novy-Marx &
  Velikov's 50% bound is the check.
- **§4's conclusion survived but its reasoning did not.** I had written "literature partly supports
  it, practice does not, so they disagree and the weight stays". With the later and larger samples
  in, the literature mostly does *not* support the level either. The weight still does not move
  today - but because the change the evidence points to is a **substitution**, not a cut, and that
  needs its own measurement first. A cut would discard the one thing both camps call informative.
  **The honest version is that I got the right answer for the wrong reason in the first draft, and
  the note now says so in §4.**

One process note worth keeping: the sweep was commissioned at the start of the session and returned
near the end, so §7 of the first committed draft said it "did not return in time" - which was true
when written and false twenty minutes later. Rule 9 cuts both ways; a note is a dated artifact and
the fix is an addendum that says when each part was known, not a silent rewrite.

### Next

- **Verify §3.1 before anyone uses it.** Three items are marked **[VERIFY FIRST]** in the note:
  Palley, Steffen & Zhang (2024, *Management Science* 71(3) - reported long-top/short-bottom on
  implied return of **-4.60%, t -3.08**), Han, Kang & Kim (2022, *JFM* 59 - betting against
  optimism, reported **stronger in large caps**), and AQR's *Quality Minus Junk* Table X (reported
  implied upside falling 0.27 -> 0.16 from junk to quality while realized returns rise 0.08 ->
  0.13). The last is the one most specific to this screener: if it reads as reported, 22 points of
  Quality and 1.35 points of implied upside are pulling against each other, and my own -0.171
  within-sector reading is the same sign.
- **`short_interest_ratio` is the obvious next Monday**, and it is the same shape of gap: 13.5
  points of the Revisions category, -0.238 within sector against the metric researched today, no
  research note, and a *documented free data history* already probed on 2026-10-23
  (`research/2026-10-22-free-data-that-adds-evidence.md` §7 has the build order, licence check
  first). That would leave the category with every metric researched.
- **The open proposal above needs a second reader** - it will appear in the morning brief's
  "Waiting for a decision" from tomorrow. Its first step is cheap and decisive: measure whether a
  90-day target revision duplicates `fy1_revision_3m` on the committed payload history. If it does,
  close it as no change.
- **A session with a long slot should take `OWNER_FOCUS.md` sub-item 6's extended universe** -
  deferred by three consecutive sessions now, including this one. Register `only-size-extended`
  first and alone, then budget the rebuild and the ~45-minute rerun in series. Three deferrals is
  the point at which the rule says the next session carrying no owner item and no broken loop must
  resolve or demote it.
- **Finish the provider sweep** §7 leaves open: StarMine's Analyst Revisions Model and MSCI's
  multi-factor index rulebooks, to settle whether *any* documented provider scores a target level.

## 2026-10-26 (13:00 midday session) - QUEUE. The morning's proposal reviewed and REFUSED: the candidate is a real revision metric, and 94% of its overlap with the rest of the screener is momentum

**Health (rule 8, all five):** last code session ran? **yes** - the 06:00 session this morning
ended "Run complete: shipped to main" at 06:51, tag `good/2026-10-26`. Data loop published?
**yes** - 2026-10-26 02:00, "Data loop complete", **HEALTH: PASS**, 498 scored, top EXPE MU APA
HST INCY. Evidence base at `1m` = **36 rows, newest 2026-09-23 (33 days ago, bound 40), 4
effective** (`_n_observations` 4 against a gate of 8; unchanged since 02:00, as it should be -
no data run between the two sessions). Priority 0: `allow_auto_apply` **false**, unchanged. Top
open roadmap item: **0.13, the Factor Lab** (opened 2026-10-12, **age 14 days**) - advanced,
since the proposal decided today came off its "From the lab to the weights" route.
**Tests:** before **3152 passed, 1 skipped** (367.4s) - identical to the 06:00 session's
after-count, so no drift. After **3159 passed, 1 skipped** - the 7 new tests are the guard below;
no production code changed.
**Owner queue / rotation:** took `OWNER_FOCUS.md` **item 1, deciding the open proposals** - the
one the 06:00 session left this morning, which the brief and `prompts/nightly.md` both put ahead
of the day's focus. Carried on from the morning's **Next**, which named exactly this ("the open
proposal above needs a second reader ... its first step is cheap and decisive") and also asked
for §3.1's three **[VERIFY FIRST]** papers to be verified; both are done. **Nothing was
reversed.** Deferred: `OWNER_FOCUS.md` sub-item 6's extended universe, now **four** consecutive
sessions - see **Next** for why the deferral-streak rule has still not triggered and what would
trigger it.

### Did

**Decided the open proposal: REFUSED.** `research/2026-10-26-price-target-upside.md` §6.2 asked a
later session to take `price_target_upside`'s 13.5 points off the scored *level* of analyst
implied upside and put them on a 90-day *change* in the consensus target. The decision and every
figure are **§8** of that note, written by a session that did not write §1-§7. Two measurement
scripts: `research/measurements/2026-10-26-target-change-candidate.py` (the candidate built over
three windows on every end date the committed payload history supports - 49, 31 and 18 of them)
and `-target-change-span.py` (the §8.4 regression on the published run).

- **It passed its own first test.** The note made the duplication check step 2 and said "if (2)
  finds the duplication, the honest outcome is **no change**". It is not a duplicate: **+0.3475**
  within sector against `fy1_revision_3m` at the proposal's own 90-day window (16 end dates),
  +0.3028 at 63 days, +0.1978 at 30. Against the incumbent's **-0.007**, the candidate genuinely
  is the construct-validity improvement §6.2 claimed - it belongs to the category it would sit in
  and the incumbent does not. So the escape hatch did not fire and the proposal had to be judged
  on the test it deferred.
- **It fails the §8.4 spanning test, which is the one `config.yaml` requires.** With the 13.5
  points moved and nothing else touched, **`revisions ~ momentum` goes +0.2223 -> +0.3778**,
  through the **~0.35** bar `research/2026-09-07-...` §8.4 set *before seeing its own number* -
  the same bar that **admitted** `fy1_revision_3m` at +0.317, which is precisely why it is not
  available to be widened now that it refuses something. The rebuild the counterfactual rests on
  reproduces the published Revisions score at Spearman **1.0** and **0.0 points** mean absolute
  difference, so this is the screener's arithmetic and not an approximation of it.
- **The sharpest number, and the real argument:** of the candidate's overlap with everything the
  screener already scores, **94% is momentum** - R² **0.2598** on all eight category ranks of
  which **0.2431** is momentum alone. The incumbent's is **22%** (0.3507 and 0.0776). So the
  candidate is *more* independent overall and almost all of what it shares is the one thing the
  rule protects: the swap would **concentrate** the screener's duplication rather than reduce it,
  and flip a **-0.32** offset against Momentum into a **+0.53** alignment with it. Momentum's own
  unspanned share falls 83.5% -> 75.9% because Revisions starts carrying it.
- **What it would have bought:** top 25 unchanged, top-50 turnover 3, median 6.5 ranks. The same
  materiality argument that made the incumbent "incoherent but not loud" (§1.4) applies to its
  replacement.

**Shipped the rule as a tripwire: `tests/test_revisions_momentum_budget.py` (7 tests).** The
§8.4 rule did real work today and nothing enforced it - it was a comment in `config.yaml` a
session had to remember to read. It now holds the published run's `revisions ~ momentum` to the
0.35 bar, and holds the comment stating the rule in place. Two things make it a check rather
than plumbing:

- **Calibrated before shipping**, the way the bot's input gates were (`DECISIONS.md` row 0.5/0.7):
  replayed across **all 57 committed payload versions** - min **-0.0835**, median **+0.2498**, max
  **+0.3206**, that maximum on 2026-09-11..09-18 immediately after the 09-10 reweight, which is
  §8.4's own predicted +0.317 turning up in published data. **No ordinary day fires; the refused
  change reads +0.3778 and does.**
- **Mutation-checked four ways**, per `OWNER_FOCUS.md` item 3's discipline: replacing Revisions
  with Momentum (+1.0000), a 50/50 blend (+0.8394) and even an **80/20** blend (+0.4777) all fire;
  removing the column or flattening it to a constant fires the vacuity guard rather than passing
  silently. The bar is a **literal** in the test and in the 09-07 note, not an imported constant -
  the capex-tolerance defect from the 2026-10-25 audit, where the assertion read the constant it
  was checking and so passed at any value. And the boundary is probed **at** the bar, which is the
  one point separating `>` from `>=`. Added to the data loop's publish gate, because the figure it
  holds is a property of the scored table a data run rewrites every morning.

### Evidence / research

**The three [VERIFY FIRST] items of §3.1 are now read from the PDF bytes, with three
corrections.** §7 made this the deciding session's first job. Each was downloaded with `curl` and
a browser user-agent, checked to start with `%PDF`, and read with `pypdf`.

- **Palley, A.B., Steffen, T.D. & Zhang, X.F., "The Effect of Dispersion on the Informativeness of
  Consensus Analyst Target Prices", *Management Science*,
  DOI 10.1287/mnsc.2021.03549** (authors' copy, insights.som.yale.edu, 55pp). Their PRET is this
  screener's metric verbatim. **Figures exact:** deciles 1-8 "fairly stable around 12%" then
  **10.84%** and **7.28%**, long-top/short-bottom **-4.60% (t -3.08)** (PDF p. 16); dispersion
  moderator **+2.96% (t 2.54)** against **-11.44% (t -5.34)** (PDF p. 17).
  **CORRECTION 1:** the sample is **537,519 firm-months, July 1999 - December 2020, 258 months**,
  not "465,797 firm-months, Jul 1999 - Jun 2018" - the second-hand report understated it and
  ended it two and a half years early, so the correction *strengthens* the finding.
  **CORRECTION 2:** the positive region is only the two lowest dispersion deciles - "this positive
  correlation only manifests when dispersion is very low ... conditioning on low dispersion ... is
  not a panacea" (PDF p. 17). And their mechanism is this project's §1.2 in a peer-reviewed sample:
  "incentive-driven staleness in price targets by some analysts after bad news. As a stock performs
  poorly and some analysts are slow to update their target prices, dispersion increases, and the
  consensus target price becomes too high."
- **Han, C. & Kim, S.Y., "Betting Against Analyst Target Price"** - ScienceDirect serves a bot
  check, so per `prompts/nightly.md` it was not worked around; what was read is the **EFMA 2019
  conference version (January 2019, 48pp)**, a different draft and author list from the published
  Han, Kang & Kim (2022), *JFM* 59, 100677, and flagged as such everywhere. **Figures exact:**
  **0.98% per month (t 3.80)**, US 1999-2015, "more prominent among large, liquid, and high
  institutional ownership stocks", one-year cumulative 15.10% (t 8.61) (PDF p. 3).
  **CORRECTION 3, and it matters:** the signal is **NOA, the net number of optimistic analysts**,
  and the paper explicitly prefers it *over* the construct this screener scores - NOA "is
  considered to reflect the impact of target price releases better than other alternative measures
  such as the mean or median of target price implied returns". So it is strong evidence that
  sell-side target optimism is contrarian in large liquid stocks, and it is **not a direct test of
  `price_target_upside`**.
- **Asness, C., Frazzini, A. & Pedersen, L., *Quality Minus Junk*, DOI 10.1007/s11142-018-9470-2**
  (read from the **June 19, 2014** working draft, 65pp - flagged as pre-publication). **Table X
  Panel A, U.S., I/B/E/S 1999-2013, exact in every cell:** mean implied expected return
  0.27/0.23/0.21/0.18/0.18/0.18/0.18/0.17/0.16/0.16, **H-L -0.11, t -9.86**; realized future
  12-month return 0.08 rising to 0.13, **H-L +0.05, t +6.62** (PDF p. 45). The table's own header
  defines the construct as "the ratio of target prices to current prices minus 1" - this
  screener's metric. §1.3's within-sector **-0.171** of it against `quality_score` is the same sign
  on our own data, much weaker.

**What that does to the note's §6.1, recorded rather than acted on.** §6.1 kept the incumbent on
"literature and practice disagree", where the literature for it was **one** paper (Da &
Schaumburg 2011, 1996-2004, whose large-growth corner is t 0.37). The verified ledger on the
*level* is now **one direct test for** (+177 bp/month, t 3.55, within sector) and **two direct
tests against** (Palley's -4.60%, t -3.08, on a later and 538k-observation sample to 2020; AQR's
-0.11, t -9.86, of implied upside against +0.05, t +6.62, of realized return), plus Han & Kim
indirectly and in our own universe, against unanimous practice across seven primary rulebooks.
**So "disagree" is too generous to the incumbent.** No weight moved on that today - one change at
a time, and §1.4 measured that removing the metric entirely moves the median stock 4 ranks and no
top-25 name - but the question is now live and pointed, with criteria written in §8.7 before
anyone looks further, including a **sign flip refused in advance** unless someone can show the
reversal in a long-only large-company book (Palley's -4.60% is a long-short spread whose short leg
this tool cannot hold).

**Palley's moderator, and whether it is buildable here** (`-target-dispersion-feasibility.py`).
Their DISP is the standard deviation of individual targets over price. The payload carries
`pt_high`/`pt_low` for **495 of 498** stocks and an analyst count, and **no individual targets and
no standard deviation anywhere** - so only a high-low range proxy is possible, and it has a defect
their measure does not: **Spearman +0.3999** against the analyst count (median range 0.390 at 3-9
analysts against **0.626** at 25+), because a range widens with the number of estimates and an sd
does not. With that caveat stated, the decision-relevant reading: within sector, implied upside
and the proxy correlate **+0.3679**, and the **top implied-upside quintile sits at a median
dispersion percentile of 79.2 against 40.0 for the bottom** - **49.1%** of top-upside names are
also in the top dispersion quintile and **5.7%** in the bottom. *The stocks this screener scores
best on this metric are concentrated in the region where Palley et al. measure -11.44% and nearly
absent from the region where they measure +2.96%.* It reads no return and changes nothing today;
§8.7 makes it the next Monday in this category with a precondition written in advance (if the
proxy's coverage contamination cannot be brought under +0.10, the dispersion branch is closed).

### Methodology changed

**None.** No weight, metric or threshold moved - that is the decision, not an omission. What
shipped is a guard on an existing rule and a research record; `METHODOLOGY_CHANGELOG.md` is
untouched, and `tests/test_revisions_momentum_budget.py::test_no_weight_moved_with_the_refusal`
skips itself the day a changelog entry takes the question up.

### Tried and rejected

- **The substitution itself**, above - refused on §8.4's pre-registered bar, not on taste.
- **The Russell coverage-shrinkage rule.** §3.1 called it "worth copying regardless of what
  happens to this metric" (Russell US Equity Indexes v7.2 §8.10.1: 1 analyst -> 1/3 own score,
  2 -> 2/3, 3+ -> full). Measured on the published run: of 496 stocks with a target, **1** has
  fewer than 3 analysts and so already scores nothing at the metric's own floor, **1** has exactly
  3, **1** has 4-5, and **461 (92.9%) have 11 or more**; median analyst count **20**. It would
  touch no scored stock. Good practice on a Russell 2000 universe, a no-op on the S&P 500 - so it
  was measured rather than shipped, which is the cheaper outcome than shipping and discovering it.
- **§6.2's scaling argument**, which turns out to be moot. It spends a paragraph preferring MSCI's
  `PTGF_C` (change over the mean absolute target) to the first draft's change-over-price, citing
  S&P Global MI's `TP_t/TP_{t-1m}`. For positive targets both MSCI's form and S&P's ratio are
  strictly monotone functions of the same ratio, so they are **rank-identical** - verified equal to
  four decimals on **all 98 end dates** across three windows. A percentile screener cannot
  distinguish them, and the price-scaled variant differs trivially (+0.3546 against +0.3475).
- **The inference that made the proposal look safe, which is the lesson worth keeping.** §6.2 read
  §1.2's **30-day** beta of +0.245 as "a reason to expect the overlap to be *small*" for a
  **90-day** metric, where its own table four rows up says **+0.540**. More generally: **a level
  beta does not bound a rank correlation.** Measured on the same series at the same gaps, the beta
  is flat at ~0.245 from 7 to 30 days while the within-sector rank correlation of target change
  with price change runs **+0.116 -> +0.296 -> +0.526 -> +0.581**. A target that moves a quarter as
  far as the price still moves in the price's *direction*, and a screener that ranks keeps all of
  the direction and none of the attenuation. Any future note reasoning from a beta to an overlap in
  this screener should read that row first.

### Next

- **`short_interest_ratio`** - the next Monday, carried unchanged from this morning's Next and
  still right: 13.5 points of the Revisions category, -0.238 within sector against the metric
  researched today, no research note, and a free data history already probed
  (`research/2026-10-22-free-data-that-adds-evidence.md` §7 has the build order, **licence check
  first**). That would leave this category with every metric researched - and then §8.7's
  dispersion question, which is the heavier of the two.
- **`OWNER_FOCUS.md` sub-item 6's extended universe is now deferred by four consecutive
  sessions**, and the deferral-streak rule has **still not triggered**, because its precondition is
  a session "carrying no owner item and no broken loop" - deciding a waiting proposal *is* an open
  owner item, and it outranks the rotation by both `prompts/nightly.md` and item 1. **What would
  trigger it:** the next session that finds `write_brief.open_proposals()` returning `[]` and the
  data loop healthy. It returns `[]` now, so that is probably tomorrow's 06:00. Register
  `only-size-extended` first and alone, then budget the rebuild and the ~45-minute rerun in series;
  do not start it after about 14:00.
- **Two papers §7 still lists as not obtained are now the gating reads for §8.7**, in this order:
  **Dechow & You (2020)** - the paper whose title is closest to the question and still cited only
  from an abstract - then **Bradshaw, Brown & Huang (2013)** in its *published* 2000-2009 sample
  rather than the 2005 working paper §3.0 read. And on the practice side the specific question is
  narrower than the sweep already done: does **any** documented provider condition an analyst
  descriptor on **dispersion**? Practice has been unanimous that nobody scores the level; if it is
  also silent on dispersion-conditioning, the pre-written rule in §8.7 keeps the weight where it is.
- **The published Han, Kang & Kim (2022) version remains unread** (bot check, not worked around).
  It is not load-bearing for anything - the refusal does not use it and §8.5 flags the draft - but
  if a later session can reach the published paper through a library copy, the author list and the
  sample should be checked against the EFMA draft before anyone quotes 0.98%/month from this repo.

## 2026-10-26 (owner-run, night) - SPEED. The suite in 1:51 instead of 6-7 minutes, engine reruns in parallel, the lab's phone first paint 6.3 s -> 0.95 s, ~4 minutes off the 02:00 run, and three bot ROADMAP items

**Health (rule 8, all five):** last code session ran? **yes** - the 13:00 midday session ended
"Run complete: shipped to main" at 13:40, tag `good/2026-10-26-1340`. Data loop published?
**yes** - 2026-10-26 02:00, "Data loop complete", **HEALTH: PASS**, 498 scored. Evidence base at
`1m` = **36 rows, newest 2026-09-23 (33 days ago, bound 40), 4 effective**. Priority 0:
`allow_auto_apply` **false**, unchanged. Top open roadmap item: **0.13, the Factor Lab** (opened
2026-10-12, **age 14 days**) - advanced (its first paint, below).
**Tests:** before **3159 passed, 1 skipped** in 7:19 serial. After, on the merged tree: **3248
passed, 1 skipped in 1:51** through `scripts/run_tests.py` (parallel), and 6:09 when forced serial
on the same tree (3,249 collected either way). `python scripts/rehearse_publish.py`: PASSED,
published files restored byte for byte.
**Owner queue:** the owner's brief for the night was speed first. `OWNER_FOCUS.md` 2026-10-20 item
7 (suite runtime, `lab_data.js`, the fresh-worktree failures) is **done in all three parts**.
Nothing scheduled was reversed; the price-target decision from 13:00 is untouched.

### How the night was run

Measured first, then seven agents in isolated worktrees (two in the bot repo), each told not to
push or merge; every diff was read before it was merged onto `staging/2026-10-26-owner`, the gates
were run once on the combined tree, then `--no-ff` to `main`. The five screener branches merged with
no conflicts; the two bot branches conflicted only in `ROADMAP.md`/`SESSION_LOG.md`.

### Did - screener

**1. The suite runs in parallel** (`scripts/run_tests.py`, pytest-xdist 3.8.0 pinned). The slow
tail was a long tail - the slowest test is 16 s - so distribution, not test surgery, was the lever.
- **Gate 1** (`scripts/nightly-screener.ps1`) and the **data loop's claims gate**
  (`scripts/check_published_claims.py`) add `-n auto` only when xdist is importable; otherwise the
  byte-for-byte serial command. pytest exit 3/4 from a parallel run (xdist could not run the suite -
  not a verdict) reruns serially and that run decides; a failing test is never retried.
  `scripts/data-run.ps1` is **unchanged**.
- **The `conftest.py` tripwire works under xdist:** the controller snapshots the protected files
  before any worker starts and hands each worker the digests; a per-worker snapshot would have
  snapshotted another worker's write and "restored" it. Proved with a throwaway writer under `-n 4`
  and pinned by a real `-n 2` run in `tests/test_parallel_suite.py` (32 tests).
- **One real isolation defect:** `test_one_observation_per_run_date`'s fixture left the two
  improvement-history paths pointing at the repo, so every run re-read the 21 MB
  `performance_history.csv` three times and **rewrote the tracked `improvement/live_ic_history.csv`**
  (byte-identical, so no tripwire saw it - but two parallel writers could have dropped rows from the
  evidence base). Now `tmp_path`; the module went 29.4 s -> 2.4 s.
- Three git-sandbox modules build their origin+clone once per module and copy it (42.7 s -> 26.2 s).
- **Fresh worktrees:** the three tests that needed gitignored `runs/` skip with `MISSING DATA: ...`
  only when it is absent. **Copy data into a worktree with `cp -rp`** - two agents lost time to
  `test_investor_profiles` choosing the wrong "newest run" from a copy whose mtimes `cp -r` reset.
- Claims gate on the rehearsed page: **1,576 passed in 71 s, `[parallel, -n auto]`** (02:00 this
  morning: 1,445 in 165 s serial).

**2. Engine reruns in parallel** (`scripts/rerun_experiments.py --jobs N`, cap 4, ~2 GB a worker).
Workers are subprocesses writing private ledgers; only the parent appends to the real ledger, and
only a prefix of the serial order, so a cut-off run leaves exactly what a serial run would have.
Proved on 10 ids: serial and `--jobs 4` rows **byte-equal except `run_at`**. Then used for real on
the merged tree: **all 32 rows in ~18 min** under load (serial ~47-63), `STALE REMAINING: 0`, and
against each id's previous row **0 differences outside `run_at`/`code_hash`/`code_hashes`/
`code_version`/`commit`** - a provenance refresh (today's `config.yaml` edit), no published number
moved. **One Windows finding worth keeping:** a worker thread blocked reading stdin hung every
worker at its first `git` call (starting a child process while a thread holds a pending stdin
read blocks); the workers watch the parent's process handle instead, with a regression test.

**3. The 02:00 run is timed, and ~4 minutes shorter** (`phase_timing.py`; note
`research/2026-10-26-data-run-timing.md`). The breakdown of this morning's 22-minute screener step,
rebuilt from logs: Yahoo ticker fetch 452 s (12 requests a ticker, none duplicated), **SEC Form 4
refresh ~400 s** (the "missing" time - 503 filing-index requests, 92 MB, to find 57 new filings),
improvement snapshot 181 s, context pass 178 s. Shipped, each proven output-identical:
- the context pass makes **no Yahoo insider request for a stock whose rows the SEC refresh then
  overwrites** (503 of 503 at 02:00; ~178 s). The SEC refresh now runs before the pass and its rows
  are still set after it; with live option quotes nothing is skipped. Proof: two real 30-ticker
  runs back to back - identical insider and option fields, Yahoo requests 30 -> 0.
- **forward returns skip a download that cannot complete any horizon** (no weekday between the
  horizon end and yesterday): 4 of 6 downloads on 10-26. Proof: the pre-10-26 state replayed with
  old and new code - `performance_history.csv` and `live_ic_history.csv` byte-identical.
- every stage's seconds in `meta.json` and a `PHASE TIMING` line in the loop log.
**Runs for the first time at 02:00 tonight.** Read the `PHASE TIMING` line tomorrow and check the
context pass reports "503 more have insider rows from SEC Form 4".

**4. The Factor Lab's first paint.** Measured with a phone profile (375 px, 4x CPU, slow 4G): the
menu took **6.2-6.4 s**, almost all of it downloading a 1.18 MB gzipped `lab_data.js` of which JKP's
153 factors were 60%. The six series the first screen never reads are now `lab_data_<key>.js`,
loaded when their experiment opens: `lab_data.js` **100 KB gzipped, menu in 0.95 s**; opening "153
factors" the first time now costs 4-5 s on that profile (an honest trade, shown as "Loading...").
The reassembled object deep-equals the old one (Python and Node), and every experiment's rendered
text is identical at 1440 and 375 px. No publish path builds lab files, so no runner changed.

**5. Claims mutation audit, items 5 and 6** (`plan/claims-mutation-audit.md` rows 63-88). 45
mutations on 25 claims: **21 checks could not see the number they check** (10 real gaps, 11 gaps
in the register), all fixed and re-mutated to CAUGHT. The row-59 shape was everywhere: a test that
builds its expected page text *and* its probes from the module constant moves with the constant.
Five claims were checked only against the committed payload; each now has a test that runs the
code (the Beneish SGI coefficient 0.892 -> 0.982 and a median computed as a mean were invisible
before). **And the register itself:** `test_claims_register` accepted a class method's bare name,
which pytest cannot select - five registrations used it; now only `file::function` or
`file::Class::method` pass. No claim was false; no production code changed.

### Did - bot (`../screener-bot`, `main` at `9681f45`, CI green)

- **ROADMAP 3.5:** every Alpaca order journals the IEX bid/ask/mid read just before submit, SPY's
  mid and the ranking's price; the report splits cost into slippage vs arrival and delay net of SPY,
  with C1/C2 reading "not decided" until 100 fills. One 5 s attempt, never retried, never able to
  change or stop an order. First live use: the next rebalance.
- **ROADMAP 3.9:** the tear sheet against a hand calculation - two defects fixed (longest drawdown
  counted run dates, not closes; a trailing `live` mark was paired with the previous close's
  benchmark).
- **ROADMAP 3.8:** the replay rebuilt point-in-time - the old one used a post-slot version on **6**
  days (not the 3 the audit counted) and traded Labor Day; now 0 and skipped. Labelled machinery,
  never return evidence, never joined to the live record.
- Suite 491 -> **532**; both rehearsals exit 0. Seen in passing, not changed: `bot.rehearse` warns
  "no signals.json is published yet" although one is - on unchanged `main` too, so the rehearsal
  probably does not fetch it. Worth a look in the next bot session.

### Methodology changed

**None.** No weight, metric, threshold or `config.yaml` line moved; `METHODOLOGY_CHANGELOG.md` is
untouched. The engine rerun moved no statistic.

### Next

- **Read 02:00's `PHASE TIMING` line** - the first measured run - and confirm the claims gate says
  `[parallel, -n auto]`. If either new step misbehaves, the gates refuse and the site keeps today's.
- **`config.yaml`'s `caching:` block misleads:** it reads like a 7-day fundamentals cache; it only
  bounds reuse of the scored cache (effectively 1 day) and `estimate_data_refresh_days` is read by
  nothing. Fix the comment **together with the next session's other `config.yaml` change** - any
  edit there stales all 32 engine rows (`RESULT_PATHS`), and a `--jobs 4` rerun is now ~18 min.
- The data-run note's **`Proposal status: OPEN`**: run the SEC passes alongside the Yahoo fetch
  (est. 5-7 min), with an old-vs-new identity check. Its other ranked items need weeks of
  comparison data; leave fetch concurrency alone.
- `scripts/rehearse_publish.py --full` still runs its tests serially - a cheap follow-up.

## 2026-10-27 (06:00 session) - PRODUCT. A flaky ship gate fixed at the root, the open proposal refused on the number it asked for, and the drilldowns re-checked against the engine

**Health (rule 8, all five):** last code session ran? **yes** - the 2026-10-26 13:00 midday
session ends "Run complete: shipped to main", tag `good/2026-10-26-1340`. Data loop published?
**yes** - 2026-10-27 02:00, "Data loop complete", **HEALTH: PASS**, 498 scored, live at 02:20.
Evidence base at `1m` = **38 rows, newest 2026-09-25 (32 days ago, bound 40), 4 effective**.
Priority 0: `allow_auto_apply` **false**, unchanged. Top open roadmap item: **0.13, the Factor
Lab** (opened 2026-10-12, **age 15 days**) - not advanced today; the next non-retrospective
Friday is 2026-11-06 and nothing here competed for it.
**Tests:** before **1 failed, 3247 passed, 1 skipped** - the failure was *flaky, not mine*, and
became the session's first work (below). After: **3256 passed, 1 skipped** - 3,248 collected at
baseline plus the scheduler test split into two (+1) and 7 new brief tests, with the flake gone.
**Owner queue / rotation:** `OWNER_FOCUS.md` item 1 sub-item 1 ("decide the OPEN proposals") was
waiting with one, 1 day open, so it came before the Tuesday focus as the prompt requires. The
Tuesday focus was then done in its verification half (drilldowns against the engine) and found
nothing broken. Nothing deferred for a loop or a gate.

### Did

**1. Gate 1 was flaky, and the flake was a test asserting a race.** The session's baseline came
back `1 failed, 3247 passed` on
`test_session_ends_clean.py::test_a_failed_worker_stops_the_run_with_the_serial_prefix_appended`,
with `assert ['older'] == ['older', 'baseline', 'e1', 'e2', 'e3']`. The same test then passed
alone, passed under `-n auto` in its own module, and the full suite passed twice more on the
identical tree. **A nondeterministic gate 1 is worse than a red one**: the runner re-runs all four
gates itself, so this could refuse a merge for a reason unconnected to the day's work, and would
look like a different fault each time.

The cause is not in the scheduler. `run_parallel`'s documented guarantee is that the ledger holds
**a** prefix of the serial order; the test asserted the **maximal** prefix, which additionally
requires workers 0 and 2 to reach their first row before worker 1 reaches its failing one - a race
between three Python interpreter startups, held off only by a 0.3 s sleep on two unrelated rows.
The parallel suite shipped 2026-10-26 puts an xdist worker on every core, which is exactly what
makes a process lose that race. **Reproduced 100%** by delaying the other workers' first rows, the
thing CPU contention actually does to startup: the prefix collapses to `['older']` and the failure
is byte-identical to the baseline's.

Fixed by removing the race rather than the assertion. The fake engine gains a done-marker per
finished row and a barrier that waits on them, so e4 cannot fail until **every row another worker
owns has finished** (c1 excluded - worker 1 runs it *after* e4, so waiting on it would deadlock
rather than order anything). The maximal-prefix assertion is now a fact of the run instead of a
hope, and **strictly stronger than before**: e5, e6, v1 and v2 are all demonstrably finished and
still withheld, which is the "nothing after it even if it finished" clause the old sleep could only
hope for. A second test pins the outcome that actually fired on the live machine - a short prefix -
with assertions that are universal invariants (a prefix, never a hole, nothing at or after the
failing row), so it holds whoever wins and cannot itself flake. Verified against the adverse
schedule that broke the old test 100% of the time, plus twelve repeats under a live xdist load.

**Swept for the same shape elsewhere and found none.** The only other wall-clock tests are
`test_loop_mutual_exclusion`'s acquire loop (a 60 s polling deadline - a liveness bound, not a
race) and `test_a_worker_stops_when_its_parent_goes`, whose 1 s sleep asserts the worker is *still
running*, a claim slowness only reinforces.

**2. The open proposal: REFUSED as written, on the measurement it nominated itself.**
`research/2026-10-26-data-run-timing.md` section 5 item 1 asked for the SEC Form 4 refresh to be
overlapped with the Yahoo ticker fetch, worth **~5-7 minutes**, and - to its credit - named the
evidence that should decide it: *"read the first `PHASE TIMING` line in `logs/datarun-*.log`
before deciding."* That line existed for the first time this morning.

**`context.sec_insider` is 87.5 s, not ~400 s.** Not a short run either: 503 tickers, 566
requests, 63 new filings, 0 failures, `complete: true`. The note's own 335 s stopwatch did not
transfer because it timed EDGAR at **21:27-21:35**, where the median `submissions` response was
0.50 s. At 02:00 the step is **floor-bound, not latency-bound** - 566 x `MIN_INTERVAL` 0.13 s =
**73.6 s** of politeness this project imposes on itself, so EDGAR's share is ~14 s against ~262 s
in the evening. **Even an infinitely fast EDGAR leaves 74 s**, so there is almost nothing there to
overlap. Section 3's ~400 s was a *remainder* (`fetch_time` less every other stage) and so absorbed
what the list-scheduling model under-counted: it put the Yahoo fetch at 452 s and the measured
`fetch.tickers` is **659 s**. The "~500 s unaccounted" the note went looking for was mostly the
ticker fetch itself.

So item 1's ceiling is **~92 s**, and it would buy that by threading Form 4 XML parsing into the
phase the same run shows **rate-limited on 39 tickers with a 124.7 s retry pass** - the longest
phase and the one with a demonstrated failure mode. Wrong trade. **The ranking inverts with the
number:** item 2 (EDGAR's daily form index, ~566 requests -> ~2) takes the same step to ~2 s, a
*larger* saving than overlapping can reach at all, with no concurrency, no GIL contention, and 92
MB/day less downloaded to find 57 documents. Section 8 of the note has the account; the status line
is rewritten and `open_proposals()` returns `[]`.

**3. The count that decision says to watch now has somewhere to be watched.** Item 6 proposes
raising worker count only after the rate-limited count reads 0 "on every 02:00 run for a month" -
and that number lives in a run's `meta.json` and one line of a gitignored log. That is precisely the
rule-8 failure: the evidence-base line read "3 rows, newest 2026-02-22" for 183 days because nobody
wrote it down. The brief's At-a-glance gains one row, built like the filing-check row - parsed from
the line the run prints, **omitted entirely when a run printed none rather than guessed**:

    | Data run timing | **19 min** total; slowest `fetch.tickers` 659 s, `snapshot` 136 s,
      `fetch.retry` 125 s; **39 tickers rate-limited** |

A clean run still states "0 tickers rate-limited" - that is the reading the month is looking for,
and a row appearing only when something is wrong makes its absence ambiguous.
`tests/test_phase_timing.py`'s `_MUST_NOT_READ` bars modules that **score, publish or hand the
ranking to the bot**; the brief is none of those and timing is provenance about the machine, which
is what it reports. A test pins the prefix to the one `phase_timing.py` prints, so the row cannot
silently go blank.

**4. The Tuesday focus: four drilldowns re-checked against the engine, all reproduce.** Looked at
the live page at 1440 and 375px.

| | composite from its own panels | page says |
|---|---|---|
| HST #4 | 20.46+18.15+4.68+12.636+4.64+5.30+2.915+2.415 = **71.196** | 71.2 |
| JPM #339 (**bank profile**) | 8.382+8.58+7.904+8.502+7.13+3.79+0.10+1.745 = **46.133** | 46.1 |
| EXPE #1 | `sustainable_growth` 97.14 pct x 15% weight = **14.571** | 14.57 points |
| CF #20 | Valuation+Quality 19.9+19.3 = **39.2** of 65.4 | 39.2 of its 65.4 |

The rank-percentile sentences are right too (JPM 159/497 = **32%**, HST 1−3/497 = **99%**), and the
composite is still described as "a 0-100 score built from its 8 category scores and their weights,
**not a percentile**" - the 2026-10-07 fix holding.

**And the three names the health check flagged to the owner this morning all say so to the user**,
which is the thing worth checking on a product day. EXPE (#1) carries the `fign` note on its own
metric row - *"Shareholders' equity is only 4.2% of total assets ... this measure sits at its 100%
cap"* - one of 48 stocks with one. CF (#20) renders *"Checked against the company's own SEC filing
for the same quarters: total assets, equity, revenue and operating cash flow agree within 2%; net
income (16%) differs from it"*, with the standing caveat naming minority interests, which is what a
16% gap at CF Industries almost certainly is. **No defect found; the Tuesday verification is a
clean pass, not a skipped day.** I first thought EXPE's note was missing and was wrong - I had
grepped for a key containing "note"/"audit" and the field is `fign`.

### Evidence / research

- Not a literature day. The decision rests on direct measurement: `logs/datarun-2026-10-27_020002.log`'s
  `PHASE TIMING` line, `runs/3328243b902e/meta.json` (`phase_seconds`, `phase_detail`),
  `data/insider/filings.json` `_meta` (503/566/63/0, complete), and `insider_activity.MIN_INTERVAL`.
- The flaky-test diagnosis is a reproduction, not an inference: the adverse schedule fails the old
  test 100% of the time with the identical assertion text, and passes the new one.
- Run time 2026-10-26 -> 10-27 on the same stage boundary: **1,075.1 s -> 906.3 s**, 169 s, ~2.8 min
  against the ~4 min section 4 predicted. `context.options_pass` 1.0 s against 178 s removed.

### Methodology changed

- **None.** No weight, metric, threshold or input moved; `METHODOLOGY_CHANGELOG.md` untouched. All
  three pieces are infrastructure, evidence-visibility and verification.

### Tried and rejected

- **Overlapping the SEC passes with the Yahoo fetch** - the day's refused proposal. Rejected on its
  own nominated measurement: the step is 87.5 s and floor-bound, so the prize is ~1.5 min, and it
  would be paid for inside the phase that was rate-limited that morning.
- **Raising fetch concurrency** - item 6's pre-registered gate ("rate_limited 0 on every 02:00 run
  for a month") failed on day one with 39. Honestly read, today is ambiguous rather than decisive:
  the owner-run night merged 3h22m earlier with two real subset runs, which is inside the "within
  hours of another run" window item 6's own theory names. Policy unchanged either way; the claim
  "no 02:00 data run in the record was rate-limited" is corrected because it is now false.
- **Starting the per-category trend lines** (priority 2's residual, the north-star time-dimension
  gap). Visible again in JPM's "What changed" panel, which compares only to the Sep 28 baseline.
  Real work, too big to finish and verify before 09:00; starting it would have left the tree dirty.

### Next

- **Build item 2 of the data-run note: fetch Form 4 submissions from EDGAR's daily form index**
  instead of one `submissions` request per company (~566 -> ~2 requests, ~87 s -> ~2 s, and 92 MB
  a day less downloaded). It is now the largest remaining saving in the 02:00 run and carries no
  concurrency risk. Take its saving as **~85 s, not the ~5 min its own line claims** - that figure
  was derived from the same 400 s this session refuted. Its evidence bar is unchanged and is the
  real cost: two weeks of shadow runs finding exactly the same new filings as the per-company scan,
  so start the shadow run now and build on it later rather than waiting to begin.
- Watch the new brief row's rate-limited count. **A second rate-limited 02:00 run on a night with
  no preceding owner session** would distinguish "the 02:00 slot is not immune" from "it was still
  paying for the evening's requests", which is the question item 6 now turns on.

## 2026-10-27 (owner-run, day) - UI ELEVATION. All four tools redesigned, presentation only; last night's speed work checked on its first live runs

**Health (rule 8, all five):** last code session ran? **yes** - 06:00 today, "Run complete:
shipped to main", tag `good/2026-10-27`. Data loop published? **yes** - 02:00 today, "Data loop
complete", **HEALTH: PASS**, 498 scored. Evidence base at `1m` = **38 rows, newest 2026-09-25 (32
days ago, bound 40), 4 effective**. Priority 0: `allow_auto_apply` **false**, unchanged. Top open
roadmap item: **0.13, the Factor Lab** (opened 2026-10-12, **age 15 days**) - its page redesigned
(below), no experiment added.
**Tests:** **3258 passed, 1 skipped** at the start (staging, after the foundation commit); **3273
passed, 1 skipped** on the merged tree (+15: the page's 5, the drilldown's 10). Gates 2-4 green;
`python scripts/rehearse_publish.py` PASSED; `dashboard_data.js`, `dashboard_context.js`,
`lab_data*.js`, `signals.json` and `SCREENER_OVERVIEW.md` **identical to `origin/main`**.
**Owner queue:** the owner's request today - *"do a full in depth ui/ux update to all the tools ...
Be creative and look into absolutly everything"* - is `OWNER_FOCUS.md` Done 2026-10-27. Nothing
scheduled was reversed; the 06:00 session's decisions (the data-run proposal refused, the
rate-limit row) are untouched.

### Last night's work on its first live runs (owner asked: "make sure the runs last night worked")

- **02:00 data run:** published, PASS. `PHASE TIMING` present (total 1,118 s). The claims gate
  ran **`[parallel, -n auto]`: 1,576 passed in 65 s** (165 s serial the day before). The context
  pass made **0 Yahoo insider requests** - `insider_from_sec 503`, `context.options_pass` **1.0 s**
  against 178 s. The run was rate-limited on 39 tickers with a 125 s retry pass; the 06:00 session
  already recorded that (`research/2026-10-26-data-run-timing.md` 8.5).
- **06:00 session:** gate 1 parallel, 3256/1 in 1:42, shipped. **It found one flaky test that
  last night's parallel engine rerun introduced** (the scheduler's prefix test raced worker startup
  under xdist load) and fixed it at the root. Honest record: the owner-run night shipped it.
- **Bot:** CI green on the merged arrival-cost/replay work; today's trade run fires at 15:00 UTC.

### Did - the redesign (`plan/ui-elevation.md`, `good/2026-10-27-owner-1`)

Audited every surface first (100 screenshots, 1440 and 375). The tools were clean but generic, so
the direction is identity: **a research publication with an instrument inside it** - Newsreader
for headlines (a third family with one job; `plan/dashboard-design-system.md` extended), the data
as the decoration, depth through light. I laid the shared foundation (fonts, `--font-display`,
`--hairline-hi`, and **`scripts/rebuild_page.py`**, which rebuilds the page from the published
payload without being able to touch it - `tests/test_rebuild_page.py` fails if the committed page
is not what the generator builds). Then four agents in worktrees, one surface each; every diff and
its screenshots reviewed before merging; the two `generate_dashboard.py` branches conflicted only
on their include lines and the adjacent new functions, resolved by keeping both and rebuilding.

- **Dashboard page:** "498 companies, one dot each" (published composites; top 25 blue, trap flags
  amber rings, median line; hover/tap, click opens the stock); leader score fingerprints with a tick
  at 50; a small true chart on every home tile; heat-shaded table scores; phone list 96 -> 80 px.
  **A real defect fixed:** open sections were capped at 5,000 px with overflow hidden, so Reporting
  soon's 7-day list (8,857 px) hid ~100 companies. First-paint nodes 2,718 of 3,000.
- **Drilldown:** a masthead with the company in the serif and the stock's place among all 498;
  heat score tiles; the arithmetic as a receipt; folding header; keyboard on every card; drag to
  close on a phone; two small bugs fixed (the Data tab never lit; the rank-history label overprinted
  at 375). Sentences byte-identical - a test renders every stock's summary against the payload.
- **Factor Lab / research page:** the chip wall became a grouped, numbered index with verdict
  charts; the research headline drawn as intervals. Wordmark and eyebrow aligned with the dashboard
  by me after review.
- **Bot dashboard** (`screener-bot` `98f2e9e`, CI green; suite 532, both rehearsals exit 0): AA
  text inks, direction by triangle as well as colour, a pausable tape that stands still under
  reduced motion, one-row phone masthead, Replay named machinery. **I sent one change back:** the
  withheld Sharpe showed a big "2" (weeks so far) in the figure slot, which reads as a Sharpe of 2;
  it now shows a dash and "needs 26 weeks · 2 so far".

### Methodology changed

**None.** No weight, metric, threshold or sentence. `METHODOLOGY_CHANGELOG.md` untouched.

### Next

- The six residuals in `plan/ui-elevation.md` "Left for a later pass" are owner-run work (nightly
  sessions do not take design work); item 5 - `renderFlagsWarnings()` writing to a missing element,
  so its Beneish and channel-stuffing badges never render - is a correctness question a Tuesday
  session may take: register the thresholds in `claims.py` and wire it, or delete the dead code.
- Read the 02:00 run's first page built by this code: `PHASE TIMING`'s `dashboard` phase (24 s
  today) should not move materially with the new page functions.

### Same day, second owner pass - one typeface (`good/2026-10-27-owner-2`)

The owner looked at the live page on his phone: *"Please focus on the typography and consistent
fonts. I like the font of the EXPE word ... I don't love how the title font looks."* The display
serif (Newsreader) is removed from all three pages and from the font download; every headline that
used it (18 rules) is Inter 600 tracked tight - the leader cards' ticker style - and the two long
lead sentences are Inter 400. `plan/ui-elevation.md` and `plan/dashboard-design-system.md` now say
**do not reintroduce a serif or a second display face.** Presentation only; payload byte-identical.

### Same day, third owner pass - the header (`good/2026-10-27-owner-3`)

The owner photographed the live page on a wide monitor with the section links cut off after
"Mar". Reproduced on the generator's page at 1024, 1152, 1536, 1680 and 1920 px (five of eight
widths): the bar had accumulated width-specific patches, and above 1560 px a "Last updated" line
and the labelled search took the room. Replaced with one rule set - the links that fit stay, the
rest move in order into a "More" menu (keyboard and Escape work), re-measured on resize and font
load. `tests/test_header_nav_fit.py` (9 tests: seven widths, a live resize, the menu). Presentation
only; payload byte-identical.

### Same day, owner request - ideas from TradingAgents (`research/2026-10-27-tradingagents-ideas.md`)

The owner asked what the screener and the bot could take from `TauricResearch/TradingAgents` (an LLM
multi-agent trading framework). Its evidence is not usable (three stocks, one quarter of 2024, Sharpe
5.6-8.2, models trained after the window), so nothing proposes an LLM signal. What it does have is the
discipline of scoring a decision from the first close after it could be acted on - and checking that
here found **a live defect with a date on it**: `improvement_engine.compute_forward_returns` enters a
snapshot at the nearest close in either direction, so a holiday run enters at the close its ranking was
built from. Labor Day escaped by luck; **2026-11-26 and 2026-12-25 will not**. Three OPEN proposals in
the note: that fix (before Thanksgiving), a swap ledger for the bot, and a "case against" block in the
drilldown. No code changed.

### Same day, owner: "Yes make all changes necessary" - the three TradingAgents proposals built (`good/2026-10-27-owner-4`)

- **Evidence base, holiday entry** (`improvement_engine._entry_close`): a run on a market holiday now
  enters at the first close on or after its date, not the close its ranking was built from. No
  recorded row changes; Thanksgiving and 2026-12-25 would have been the first affected.
  `METHODOLOGY_CHANGELOG.md` 2026-10-27; `tests/test_evidence_integrity.py::TestEntryCloseOnAHoliday`.
- **"The case against"** in every drilldown (its own tab, after "Why it ranks here"): only what the
  sheet already shows against the stock - weakest categories, flags, analyst mean below price, below
  the 200-day average, insider net selling, earnings within 14 days, read-with-care caveats, recent
  move, filing-check differences, newer filings - each line opening its section; checks that ran clean
  are named. Claim `drilldown.case_against`; `tests/test_case_against.py` checks all 498 stocks against
  the payload. Payload byte-identical.
- **Bot swap ledger** (screener-bot ROADMAP 3.10, `17c03fe`, CI green): per rebalance, names bought
  minus names sold over 5/21/63 sessions from the first close after the order, net of cost, withheld
  under 26 non-overlapping rebalances. Measurement only. `stock_detail[t].price_date` now pinned by
  `tests/test_bot_payload_contract.py`.
- **Two flaky browser tests fixed** (fixed waits a loaded parallel suite overran; the rehearsal failed
  once on one). Assertions unchanged. Suite **3295 passed, 1 skipped**; rehearsal PASSED.

## 2026-10-27 (13:00 midday session) - QUEUE. The eleven badges that lived in dead code: four said nothing, one of them a false sentence on a top-25 stock, and one a false claim on the public methodology page

**Health (rule 8, all five):** last code session ran? **yes** - 06:00 today, "Run complete:
shipped to main", tag `good/2026-10-27`; five owner-run passes merged after it. Data loop
published? **yes** - 02:00 today, "Data loop complete", **HEALTH: PASS**, 498 scored, live at
02:20, claims gate 1,576 passed in 65 s `[parallel, -n auto]`. Evidence base at `1m` = **38 rows,
newest 2026-09-25 (32 days ago, bound 40), 4 effective**. Priority 0: `allow_auto_apply` **false**,
unchanged. Top open roadmap item: **0.13, the Factor Lab** (opened 2026-10-12, **age 15 days**) -
not advanced; the next non-retrospective Friday is 2026-11-06 and experiment 18 is a Friday's work.
**Tests:** before **3295 passed, 1 skipped**; after **3313 passed, 1 skipped** (+18, the new
module). `scripts/rehearse_publish.py` **PASSED**.
**Owner queue / rotation:** no proposal was waiting (`open_proposals()` returns `[]`). The 06:00
session's **Next** was the EDGAR daily-form-index build, whose own evidence bar is two weeks of
shadow runs - it cannot be finished end to end today, and the prompt asks for one item finished.
I took instead the item this morning's owner-run session flagged as available to a session
(`plan/ui-elevation.md` item 5), which is also what the 2026-10-15 session called **"the single
most valuable next thing"** and which had then sat for **12 days**. Nothing deferred for a loop or
a gate; nothing an earlier session decided today was reversed.

### Did

**The defect hunt the 2026-10-15 session left open.** That session found
`generate_dashboard.renderFlagsWarnings` dead - nothing called it, and its container
`#modal-flags` was never written into the page, so that morning's quarterly-rejection badge could
never have appeared. It rehomed the two provenance sentences and left the **other eleven badges**
with the note that one session should check each against the live page and either rehome it or
delete it. This is that session.

Measured on the 2026-10-27 published run, 498 stocks
(`research/measurements/2026-10-27-dead-badge-audit.py`, reproducible at any commit):

| Badge | Stocks | Reaches the reader? |
|---|---|---|
| value trap | 46 | yes - table flag column, `summary`, "The case against" |
| growth trap | 66 | yes - same |
| Beneish | 43 | yes - `summary`, "The case against" |
| channel stuffing | 14 | yes - `summary`, "The case against" |
| stale financials | 28 | yes - Filing Age badge, `summary`, "The case against" |
| EPS basis mismatch | 100 | yes - provenance badge, data-quality KPIs |
| bank-like | 46 | yes - the ROE/ROIC column label, methodology page |
| **LTM 3-of-4 annualised** | **4** | **no - and the panel said the opposite** |
| **enterprise value substituted** | **18** | **no** |
| **beta overlap under 80%** | **1** | **no** |
| **financial-sector caveat** | **0** | **no - dead at source, with a false claim on the public page** |

**1. A false sentence on the live site, one of the four stocks at rank 22.** The Data provenance
panel picks its lead on `fl.ltm_rolled || fl.q_rejected`. **CCL, EFX, TRV (rank 22) and V** have
both false and `data_source == "quarterly"`, so the page told readers their "income and cash-flow
figures are **the sum of the last four reported quarters**" while the engine had summed **three**
and scaled by 4/3. That is the same defect the 2026-10-15 session fixed for the other two
constructions - the third route simply had no sentence, because its badge was in the dead
function. TRV is inside the top 25, the set `signals.json` publishes to the bot.

**2. A substituted scoring input, 18 stocks, never disclosed per stock.** Where the source's
enterprise value is more than 10% (25% for financials) from market value + debt - cash, the engine
**replaces it** and scores EV/EBITDA, EV/Sales and free-cash-flow yield on its own sum. The rule
is on the methodology page and in the EV/EBITDA row's caveat; no stock's page said it had fired
*there*, so a reader checking the enterprise value in the workings against the provider's own
figure found two numbers and nothing to explain them. The rejected figure is now published
(`_ev_api`, display-only) so both appear and the 10% rule is checkable from the page.

**3. An unexplained gap, 1 stock.** FDXF's beta is not computed at all (42% of its trading days
overlap the index, against the engine's 80% bar), so the row said "no data" with no reason.

**4. A flag no run has ever set, and a false claim on the public methodology page.**
`Financial_Sector_Caveat` is written by `factor_engine.add_financial_sector_caveat` - which only
that module's **own legacy end-to-end path** calls, not `run_screener.py`'s pipeline. It is absent
from `05_final_scored.parquet` (verified on the published run), so the payload published
`fin_caveat: false` for all 498 stocks on every run, and the Excel column is empty too. Meanwhile
`SCREENER_OVERVIEW.md` said, and has been serving to the public: *"All financial-sector stocks
receive a `Financial_Sector_Caveat` flag in the output, reminding the user that financial
companies require additional scrutiny regardless of classification."* **Zero did.** The field is
deleted rather than wired - see "Tried and rejected" - and the paragraph corrected **in the
generator**, with the file regenerated (rule 10).

**What shipped:** three sentences in `renderProvenance`, which the drilldown does call; the dead
field and the dead function deleted; the methodology paragraph corrected. Claims
`drilldown.ltm_annualized`, `drilldown.ev_substituted`, `drilldown.beta_overlap`,
`overview.no_financial_sector_flag`. `tests/test_dead_badges.py`, **18 tests**, browser-rendered
against *the generator's own page* - and in the data loop's publish gate by derivation from
`checked_by`, without anyone adding it to a list.

**Looked at the page**, which is the half of this a test cannot do, on the page regenerated from
today's run with this code (the committed `index.html` is the 02:00 run's and predates it).
TRV's panel now opens "Income and cash-flow figures cover the last twelve months, built for this
company as described below" and then names the construction - **4 lines at 1440 px, 8 at 375 px,
no clipping, no horizontal overflow at either width**, in the same dimmed "limit" style as the
roll-forward and filing-check sentences beside it. **A wording defect I could only see by reading
it:** the EV sentence ended "The wider limit applies to financial companies, whose debt includes
customer deposits" on **every** stock - so AES, a utility told it was 10% over, was then told
about a limit that is not its own. The clause is now on the 25% branch only and says what it is
instead of; checked live on both (AES/SMCI/KDP at 10% with no clause, APO/KKR/IVZ at 25% with it).
Shots in `logs/design/prov2-*.png`.

**Two things the next full run completes.** The regeneration showed TRV reading "these figures are
the three quarters summed" rather than naming the lines, and the EV sentence without its two
figures - because `_ltm_annualized_labels` and `_ev_api` are new and today's stored run predates
them. Both degrade to a correct shorter sentence, tested in its own right
(`test_the_annualised_sentence_works_without_labels`,
`test_the_substituted_figure_survives_a_missing_counterpart`); the 02:00 run fills them in.

### Evidence / research

- **Four documented, reproducible defects** - the admissible kind (CLAUDE.md, "a documented
  failure"). Each is a statement the live site makes, or fails to make, measured against what the
  engine did: `research/measurements/2026-10-27-dead-badge-audit.py` prints the eleven counts, the
  four stocks the lead sentence misdescribed with their ranks, and the absence of
  `Financial_Sector_Caveat` from the scored frame.
- **No literature is needed and none is cited.** Nothing here changes a weight, a metric, a
  threshold or what a number means; all four are the page telling the truth about arithmetic the
  engine already does. No backtest or IC figure is used anywhere (rules 4 and 5).
- The bars the new sentences quote are asserted **against the engine's own constants**, so a
  change to either side fails the suite: `_overlap_ratio >= 0.80`, and the 10%/25% EV tolerance.
- The beta sentence's bar is probed **at** 80.0 as well as either side of it - the 2026-10-25
  mutation pass's lesson that a test probing 0.001 on both sides and never at the point is blind
  to the one difference that separates `>` from `>=`.

### Methodology changed

- **None.** No weight, metric, threshold or input moved; `METHODOLOGY_CHANGELOG.md` untouched.
  The one public-page correction is about a flag that never existed, not about how anything scores.

### Tried and rejected

- **Wiring `add_financial_sector_caveat` into the pipeline** instead of deleting the field - the
  other way to make the methodology sentence true. Rejected on measurement: `fin_caveat` would
  then be true for exactly the 46 stocks `is_bank` is already true for (`fin_caveat & !is_bank` is
  **0**, because `_is_bank_like` defaults to the bank set for any unseen Financials sub-industry),
  so it would add a second badge saying "look closely at this financial" beside one that already
  changes the stock's weight tables. A duplicate flag is a worse answer than no flag, and the page
  is clearer saying which set a stock was scored on. A test fails if the column ever does reach
  the scored output, because the corrected paragraph would then itself be out of date.
- **Taking the 06:00 session's Next** (EDGAR's daily form index). Sound work and still the right
  next build, but its own evidence bar is two weeks of shadow runs agreeing with the per-company
  scan, so no session can finish it today; starting it would have left a half-built fetch path.
- **Fixing `renderProvenance`'s dead early return**, found while reading it:
  `s.metric_count === null && s.metric_count === undefined` can never be true, so the guard never
  fires. Left alone deliberately - repairing it would *remove* the panel from any stock with no
  metric count, a visible change to the page with no defect behind it. Recorded here so the next
  reader does not take it for live code.

### Next

- **The 06:00 session's Next stands: EDGAR's daily form index for the Form 4 refresh**
  (~566 requests -> ~2, ~87 s -> ~2 s, 92 MB/day less downloaded). Its bar is two weeks of shadow
  runs, so **start the shadow run first** and build on it later; take the saving as ~85 s, not the
  ~5 min the note's own line claims.
- Read tomorrow's 02:00 page for the two sentences only a full run can complete: TRV should name
  its annualised lines, and the 18 EV stocks should print both figures.
- The generalisable lesson, for whoever next adds a surface: **a badge written into a container
  that does not exist cannot fail visibly**, which is why this one survived for months and why the
  fix is a test asserting the function is gone rather than a test asserting it renders.

### Same day, fifth owner pass - "as many improvements as possible" (`good/2026-10-27-owner-5`)

Owner: *"make as many improvements as possible before that next session. Be creative and find
improvements across all tools and dashboards."* Four agents in separate worktrees, one surface each,
each told to audit at 375 and 1280 px before changing anything; the coordinator read the live site
independently and sent the page agent five findings, reviewed every branch, merged, regenerated the
page and ran the gates on the combined tree. **Presentation and access only: payload byte-identical,
no weight, metric, threshold or scored sentence moved; `METHODOLOGY_CHANGELOG.md` untouched.**

- **Page** (`tests/test_page_fourth_pass.py`, 20): phone rankings 50 at a time with "Show the next
  50 / all 498" (filters, sort, stepping and deep links still see all 498); measured phone height
  44,339 -> 8,834 px. The four residuals in `plan/ui-elevation.md` done. One roving tab stop for the
  table (was ~500). A real defect found on the way: below 1180 px the spacer rows spanned 15 columns
  with 14 shown, so the table grew an empty column and at 768 px the company name was 0 px wide.
  Trap badges under the name on a phone; flags column 80 px on desktop so names stop truncating;
  the Menu pill steps aside while scrolling. The Track record tile now prints the panel's own
  standard-error reading ("too short to tell skill from luck: 0.7 standard errors") - registered
  under the existing claim `context.track_luck` with a check that tile and panel agree.
- **Stock sheet** (`tests/test_drilldown_reading.py`, 49): headings, focus-moving tabs, stepping that
  keeps focus and announces the stock, tab-strip overflow cues, "What it measures" for 38 metrics
  (`METRIC_PLAIN`; tested to carry no digit, computing verb or advice term, so they are definitions,
  not claims), phone labels on the workings figures, a keyboard research toggle, print the sheet.
  The coordinator checked every definition against `factor_engine.py` and corrected three
  (sustainable growth assumes leverage held constant, not no new borrowing; analyst surprise is a
  median over recent quarters; max drawdown is the past year). **How it adds up's labels** read
  "Very Weak" for a Risk score of 21 and would read "Strong" for a small company's Size score; they
  now describe the score - High / Above mid / Below mid / Low.
- **Factor Lab and research page** (`tests/test_lab_reading_and_access.py`, 18; `test_research_page`
  +2): bar charts readable by tap and keyboard, "%" on axes, a failed lazy part keeps the experiment
  and offers Try again, compact phone overview cards, the hero's figures as rows on a phone, a status
  region for Copy a link, a contents list and print styles on the research page. `lab_engine.js` and
  the lab data untouched. `tests/test_overfitting.py` now anchors on the section's `<h2>` (the new
  contents list names it first) - same assertions.
- **Bot** (screener-bot `7f4328f`, 613 passed, both rehearsals exit 0, CI green): one-row masthead,
  honest first-mark state, keyboard ladder, legible heat map without moving the locked colour, ops
  strip counting only due days, and ROADMAP 3.7's health ledger (`bot/health.py`; read only by the
  report and dashboard, no line says "pass" while the gates are an unsigned draft).

Gates on the combined tree: suite **3,402 passed, 1 skipped**; `rehearse_publish.py` PASSED; dry run
OK; `node --check` on both payloads and `lab_engine.js`; tree clean.

**Left, recorded in `plan/ui-elevation.md` item 7 and `plan/factor-lab.md`:** Factor analytics' sector
table on a phone, the Diagnostics chevron, the bare Delta header, an unscored category tile that cannot
say why (needs a per-category reason in the payload), and the lab overview's three long tables. The bot
agent noted the draft go-live gate U1's worked example disagrees with its own formula (-20% at 1 year,
not 2) - the owner should correct the example when he signs the gates.

### Same day, sixth owner pass - round two (`good/2026-10-27-owner-6`)

Owner: *"Keep finding improvements please!"* Four more agents, aimed at what nothing had measured
yet: an automated accessibility audit, speed on a throttled phone, how the tool explains itself to
a student, and the leftovers. **Payload byte-identical; no weight, metric, threshold or scored
sentence moved; `METHODOLOGY_CHANGELOG.md` untouched.**

- **Dashboard** (`tests/test_page_round_two.py`, 18): axe-core 4.10.2 over 20 states, serious
  67 -> 0 after the coordinator underlined the guide's links (the agent was kept out of `#guide`),
  moderate 6,293 -> 0 (a `<main>` landmark; `.dashboard-container > .section` no longer reaches the
  sections - write `.page-main > .section`). Phone scrolling: `shiftWindow` keeps on-screen rows
  instead of rebuilding ~37 per step - reading-pace p95 frame 183 -> 50 ms, layout objects over 40
  steps 79,914 -> 17,968 (4x CPU). Sector table as cards on a phone; Diagnostics chevron; the Delta
  header reads "Move"; methodology code blocks wrap (source text unchanged) and in-dialog links
  scroll to their heading, with a test that every cross-reference has a target. Not done: the
  ~530 ms relayout when Inter arrives (a font-loading decision), unscored-category reasons.
- **Methodology, guide, README** (`tests/test_overview_reading_order.py` 15,
  `tests/test_readme_front_door.py` 9; claims `overview.at_a_glance_flags`,
  `overview.at_a_glance_tested`, `overview.glossary`; both modules joined the publish gate's
  `MODULES`): the generated page opens with "If you read nothing else", "How to use it", "What it
  cannot do" and "Words this page uses"; three lists that rendered as run-on paragraphs are lists;
  the duplicate closing summary is gone; "a bank is compared with banks" corrected to "other
  financial companies" (percentiles group by GICS sector). The history sentence is built from
  `summary.json` and says "no better than chance" only while the interval holds zero. The
  coordinator checked the at-a-glance bullets against `factor_engine.apply_value_trap_flags` /
  `apply_growth_trap_flags` and corrected one: Compare holds up to four stocks (`COMPARE_MAX`), not
  two. README: seven stale statements corrected (the 10-25% failure rate, a nonexistent `--top-n`,
  the sheet count, and others), its figures tested against config and the engine.
- **Lab and research page** (`tests/test_lab_reading_and_access.py` +5, `test_research_page` +3;
  claim `research.plain_answer`): axe to zero on both (links by colour alone, landmarks); the three
  overview tables readable on a phone (8,313 -> 5,615 px); a defect fixed - the phone category map's
  "Not covered" text read as covered; `lab_data.js`/`lab_engine.js` preloaded and an experiment's part
  fetched on press or hover (never on touch-down, so scrolling past "153 factors" downloads nothing).
  "Has it been tested?" now says under "Not measurably." that the higher-ranked companies did not do
  measurably better and no edge has been shown - printed only while every interval includes zero.
  Left: "153 factors" takes 4.5-5 s to open on a phone (bytes; splitting the part by universe is a
  lab-session change, written into `plan/factor-lab.md`).
- **Bot** (screener-bot `e631973`, 644 passed, both rehearsals exit 0, CI green): axe 18 -> 0 over 72
  states; skip link past the ladder; REPORT.md opens with "Real money: not decided yet" beside the
  range luck alone gives; H4 reconciles the latest Alpaca mark (0.0000% today); a crash in
  `report.health_rows` on a missing beta fixed. Nothing new written by a run; the per-mark positions
  file and 3.6's after-tax line are designed in ROADMAP.md, not built.

Gates on the combined tree: suite **3,452 passed, 1 skipped**; `rehearse_publish.py` PASSED; dry run
OK; `node --check` both payloads and `lab_engine.js`; `rebuild_page.py --check` clean; tree clean.

**Two unexplained full-suite failures, recorded rather than retried away.** Of six full runs on this
tree, two failed one or two tests each and four were clean: once
`test_page_round_two.py::test_methodology_cross_references_land_on_their_heading` and
`test_loop_mutual_exclusion.py::test_exactly_one_of_several_simultaneous_starts_wins` together
(the first failure's name was not captured). Neither reproduced alone (3 runs) or under a 10-worker
browser load (2 runs). The new test's only timing dependence was a smooth scroll it had to wait out,
so the cross-reference handler now honours `prefers-reduced-motion` (an access fix in its own right),
that test runs with reduced motion, and a new test pins smooth-by-default and instant-on-request from
the call itself. The mutual-exclusion test is older and untouched tonight (no runner script changed);
**if it fails at a 06:00 baseline, that is the load-sensitive start race, not tonight's work** - read
its message before anything else.

### Same day, seventh owner pass - round three (`good/2026-10-27-owner-7`)

Owner: *"please keep making improvements!! anything you can find throughout these tools."* Five agents
(sheet, page shell, Factor Lab, bot, flaky tests); the coordinator reviewed, merged, fixed what the test
hunt found in production code, and ran the gates on the combined tree. No weight, metric, threshold or
scored sentence moved; `METHODOLOGY_CHANGELOG.md` untouched.

- **The repo lock had two real races** (`research/2026-10-27-flaky-tests.md`, reproduced by
  `research/measurements/2026-10-27-repo-lock-races.py`). (1) `Enter-RepoLock` created the lock and
  only then wrote its PID; a contender reading it in that 70-800 ms window found PID 0, took the live
  owner for dead and deleted its lock - 4 of 5 trials with the winner paused 0.5 s. (2) Racers
  reclaiming a crashed run's lock each read-then-deleted and could delete each other's fresh locks -
  2 to 5 holders in 6 of 16 six-racer trials. This is what
  `test_exactly_one_of_several_simultaneous_starts_wins` had been catching under load. **Fixed by the
  coordinator:** `Publish-RepoLockFile` writes a temp file and `File.Move`s it onto the lock (atomic,
  fails if it exists); a lock naming no owner counts as held for 10 minutes; `Invoke-RepoLockReclaim`
  deletes a dead or over-age lock only while holding a short reclaim token, after re-reading it.
  Re-measured with the same script: **0 of 8** and **0 of 6**. Two new behavioural tests; the module
  passed 3 of 3; all three runner scripts parse (`Parser.ParseFile`, 0 errors). **Tonight's 02:00 data
  run is the first to use it** - read its log's lock lines.
- **Flaky tests** (no assertion changed, nothing retried): six root causes - a pointer left over a
  menu firing the lab's hover prefetch, two mouse moves read as a hover under load, a 200 ms sleep
  against a 150 ms debounce, a held request that reached pytest late, the header fit measured after a
  fixed 400 ms, a single-sample sort timing (now best of five against the same 250 ms). All 18
  browser modules: 2 of 7 loaded runs failed before, 4 of 4 clean after.
- **Stock sheet** (`tests/test_sheet_round_three.py`, 12; claim `drilldown.unscored_category`):
  scroll-spy lights the chosen tab and every tab in order (APA lit "Data" for Price targets); a
  Financials tab; the first open is prewarmed at idle on the #1 stock, unseen, with no history, title
  or focus change (layout 337 -> 71 ms in a quiet 4x trace); an unscored category says "none of its N
  metrics has data", the engine's own rule, checked on every published stock; empty sections say so;
  peers open their own sheet and show their rank. **Payload: one display-only field,
  `stock_detail[t].flags.px_rejected`** (the engine's `_price_series_rejected` reason or null), tested
  never to reach raw/pct or a scoring module; the committed payload is untouched and the field
  publishes from the next data run. Rehearsed twice by the coordinator on the combined tree: both
  PASSED (claims gate 1,780 tests).
- **Page shell** (`tests/test_page_shell.py`, 22; claim `page.freshness`): a failed or offline data
  load names the failure with a reload button and hides the empty sections; "still loading" after 4 s;
  one notice and a retry when the context file fails; a freshness line when two or more weekday runs
  are overdue; **a real defect** - the run time was read in the reader's own zone, so London saw
  "2:00 AM GMT" for a 07:00 GMT run (`runInstant` reads America/Chicago); a metric-matched Arial
  fallback for Inter (layout shift at 375 px 0.0020 -> 0.0004); a print layout (all filtered rows,
  light, header repeated - was 37 rows and eleven blank pages); `404.html`; `manifest.webmanifest` and
  icons (`scripts/make_icons.py`), no service worker. New root files are source, not data artifacts;
  no runner change needed.
- **Factor Lab** (claim `lab.replication_check`): "153 factors" split by universe - 5.0 -> 2.0 s to
  usable on a phone; the eight sorts one part each (earnings yield 2.04 -> 0.99 s); a phone row chart
  for the 13 themes; real minus signs on the research page's t-statistics. Reassembled lab payload
  byte-equal to round two's; `ASSET_VERSION` 28. CLAUDE.md's lab line corrected (one part per
  `LAZY_KEYS` entry, not six).
- **Bot** (screener-bot `0311be6`, 710 passed, both rehearsals exit 0, CI green): `bot/runstate.py` - a
  run's outcome in one line for the alert, step summary, REPORT.md and dashboard banner (when and how
  often alerts fire unchanged); `journal.problems` names a malformed journal row loudly; REPORT.md and
  the page build separately and a failed build is written into the run record; 15 edge-case journals
  through both builds; the book table fits at 1280.

Gates on the combined tree: suite **3,496 passed, 1 skipped**; `rehearse_publish.py` PASSED twice; dry
run OK; `node --check` both payloads, `lab_engine.js` and every `lab_data*.js`; `rebuild_page.py --check`
clean. Left: self-hosting Inter (a binary font in the repo) would end the font relayout; the sheet's
strip now has 11 tabs; `test_row_click_to_drilldown_is_fast` failed once under load in an agent's run
and was deliberately not given best-of-five (it measures the first open a reader feels).

## 2026-10-28 (owner-run, morning) - last night's refused run fixed; the universe-expansion plan

**Health (rule 8, all five):** last code session ran? **no** - 06:00 today stood down: "Could not
stash the leftovers" (the cause is below). The 13:00 session of 2026-10-27 shipped
(`good/2026-10-27-1338`). Data loop published? **no** - 02:00 today, HEALTH: PASS (498 scored, 0
fetch failures), then **refused by the published-claims gate** (1 failed, 1,779 passed); the live
site kept 2026-10-27's figures, as designed. Evidence base at `1m` = **38 rows, newest 2026-09-25
(33 days ago, bound 40), 4 effective**. Priority 0: `allow_auto_apply` **false**, unchanged. Top open
roadmap item: **0.13, the Factor Lab** (opened 2026-10-12, **age 16 days**) - not advanced this pass.

Owner, in chat: *"How did the runs go last night? I'm assuming they went okay"* - they did not, and
both failures were one defect each, now fixed:

- **The refusal was right; the line it refused was wrong.** BXP closed **0.03%** under its 200-day
  average (60.89 against 60.91). The stock sheet's "case against" line formats that gap with
  `cPct` at one decimal, which rounds it to zero and drops the sign: *"Its last close is 0.0%
  against its 200-day average"* - a line listed as a reason against the stock saying the gap is
  nothing. `tests/test_case_against.py` checks every stock's line every run and failed on it. Fix in
  `generate_dashboard.py`: the line is always signed, and a gap under 0.1% takes two decimals
  (`−0.03%`). New test `test_a_close_just_below_its_200_day_average_still_reads_as_below` pins BXP's
  exact figures. Re-run against a regeneration of last night's run (17098ca9b290): the module passes
  9 of 9 (it failed before), and the rehearsal's claims gate with it.
- **One refused run cost two sessions.** Every discard in `data-run.ps1` restored the tracked
  artifacts and cleaned only `improvement/snapshots`, so the refused run's new
  `data/context_log/2026-10-28.parquet` and two `improvement/price_cache` files stayed untracked.
  The 06:00 runner stashed them - the stash **saved** - but `git stash` exited non-zero on
  *"failed to remove output/: Permission denied"* and the runner stopped on the exit code. Two fixes:
  (1) every discard now goes through one helper, `Undo-RunArtifacts`, which also cleans the run's new
  files under `improvement`, `validation` and `data/context_log` (the tree was clean or stashed when
  the run began, so they are the run's; ignored files are never touched); (2) both loops judge a
  non-zero stash by whether the tree is clean afterwards - a stash that failed outright still leaves
  it dirty and still stops the run. Tests: `test_undo_run_artifacts_leaves_a_clean_tree_and_nothing_else_touched`
  runs the real function, extracted from `data-run.ps1`, in a scratch repo under PowerShell; two
  source checks pin that every discard uses it and that both loops decide on `git status`;
  `test_published_claims_gate.py::test_a_failed_claim_discards_the_run` follows the helper. Both
  runners parse (`Parser.ParseFile`, 0 errors).
- **A third casualty, found by gate 1:** `tests/test_investor_profiles.py` compared the published
  payload with the *newest* `runs/` folder - the refused run, which the page never showed - and
  failed on a correct payload; the 13:00 session's gate 1 would have failed the same way. It now
  pairs the payload with the run it names (`kpis.run_id`) and skips loudly if that run is absent.
  Gates on the final tree: suite **3,500 passed, 1 skipped**; dry run OK; `node --check` both
  payloads; tree clean; `rehearse_publish.py` against last night's refused run **PASSED** (claims
  gate 1,781). Payload byte-identical; page rebuilt with `rebuild_page.py`.
- **Not republished today, deliberately.** `data-run.ps1` has no regenerate-only mode and a fresh
  full run now would take mid-session prices and be the second full fetch within hours - the one
  pattern that has been throttled (`research/2026-10-26-data-run-timing.md`). The 2026-10-28
  observation is lost; tonight's 02:00 run publishes with the fix. The refused run's three files
  are in `stash@{0}` ("auto-rescue 2026-10-28_060001"): **do not pop it** - under the new rule a
  refused run leaves no evidence behind, and those files are that run's.
- **Repo lock, first night:** the data log has no lock line at all, which is what an uncontended
  acquire looks like (it logs only on waiting or reclaiming). The 06:00 runner stood down before
  the lock mattered, so the race fix has not yet been exercised under contention.

**Owner item opened: beyond the S&P 500** (`OWNER_FOCUS.md`, `plan/universe-expansion.md`). Five
phases, each with an exit test; the S&P 500 ranking, `signals.json` and the bot unchanged
throughout; per-universe evidence series. Measured for the plan: `dashboard_data.js` 6.2 MB with
`stock_detail` ~90% of it, the Yahoo fetch 795 s for ~500 stocks, and the constants that encode
"about 500" (`MIN_SCORED = 490`, `$universe = 500`, `MIN_UNIVERSE = 495`, the 495-515 membership
band, the page's `|| 500`). Phase 0 is research and belongs to a Monday.

### Next

- Read tonight's 02:00 log: it should publish, and a refused run (if any) should now leave
  `git status` clean.
- Phase 0 of the universe plan on research Monday 2026-11-02.
## 2026-10-28 (13:00 midday session) - QUEUE. The research page's statistics mutation-tested: six of its eight unchecked claims were seen only by a tripwire that is green in the state the change ships in

**Health (rule 8, all five):** last code session ran? **yes** - 2026-10-27 13:00 shipped
(`good/2026-10-27-1338`); today's 06:00 stood down and the owner-run morning session fixed both
causes (its entry above). Data loop published? **no, correctly** - 02:00 today was HEALTH: PASS
and then refused by the published-claims gate on a real false sentence; the live site keeps
2026-10-27's figures and the fix is committed, so tonight's run publishes. Evidence base at `1m`
= **38 rows, newest 2026-09-25 (33 days ago, bound 40), 4 effective**. Priority 0:
`allow_auto_apply` **false**, unchanged. Top open roadmap item: **0.13, the Factor Lab** (opened
2026-10-12, **age 16 days**) - not advanced, and not drifting: the lab's day is a Friday, 10-30 is
an even ISO week so the runner swaps in the retrospective, and the next lab Friday is **2026-11-06**.
**Tests:** before 3,500 passed / 1 skipped, after **3,506 passed / 1 skipped**.
**Owner queue / rotation:** took `OWNER_FOCUS.md` item 3 of the 2026-10-20 block, "Test the
tests" - the fourth mutation pass, `plan/claims-mutation-audit.md`'s queue items 4 then 1-3. No
proposal was waiting (`write_brief.open_proposals()` returns `[]`). The top open item in that file
(universe expansion) is a Monday research item by its own Next line; nothing else deferred.

### Did

**17 mutations across 6 claims: 11 caught, 6 gaps. Every gap now fails a registered check.**
Rows 89-105 of `plan/claims-mutation-audit.md`, which closes its queue items 1-4. No production
code changed and no published sentence moved; the whole diff is tests, the register, and the
write-up.

**The research surface was the one no pass had touched, and it had a kind of unchecked claim this
audit had not met.** `research.html`'s sentences quote intervals, t-statistics and a
"smallest difference the test could reliably detect" figure. Ten mutations of that arithmetic - the
80%-power quantile, the power term itself, the t-distribution for the bar, the interval boundary at
zero, the intervals' confidence level in both places that builds them, the record the pooled
estimate covers, the multiple-testing bar, the alternatives count, and the rule deciding what
"cleared its bar" means - produced **two** caught and **eight** gaps.

**The thing worth carrying forward: for six of those eight, the only test that failed was a drift
tripwire** - `test_the_committed_page_is_what_the_generator_makes_from_the_ledger`, or
`test_the_methodology_page_quotes_the_engine_s_own_summary`, which asserts the committed
`summary.json` equals a recomputation. Both of those fail under *any* numeric change and assert
nothing about the arithmetic - **and both are green in exactly the state the change ships in.**
That is measured rather than argued: every mutation was re-run with `python v2_experiments.py
report` in between, which is how `research.html` and `summary.json` are actually published, and
**all seven gaps stayed green.** It is CLAUDE.md rule 10 one level up ("assert claims against the
generator's output, not the committed file"): a check that compares a generated artifact with its
own regeneration cannot see a change to the generator.

**Underneath that, two claims were checked by a test that recomputes the rule out of the code and
then asks the page to agree with it.** `research.plain_answer`'s only check opened with
`every_zero = all(lo <= 0 <= hi for lo, hi in ci.values())` - the generator's line, copied - and
`overview.research_engine`'s read `published["better"]` out of the same function that computes it.
Both are consistency checks between the code and itself. This is row 59's rule ("a check may not
import the number it is checking") applied to a *rule* rather than a constant, and it is much
harder to spot, because nothing in the test looks like a constant.

**The six gaps and their fixes** (each new test shown to pass unmutated, fail under its mutation,
and every mutation re-run afterwards and required to come back `CAUGHT`):

- **The interval boundary at zero** (`<=` -> `<`): nothing in the suite saw it, before or after a
  regeneration. The published intervals are rounded to four decimals, so an endpoint of exactly
  `0.0000` is a real state, not a contrivance - and under the mutation the page would drop "No
  edge has been shown" and print "Not reliably" on a record that includes zero.
- **The intervals' confidence level** (5% -> 10%, in the era block and the pooled block) and
  **the record the pooled estimate covers** (176 months -> 80, by moving its start to 2020):
  pinned now on a synthetic baseline against quantiles written as **literals** (2.0262 on 37
  degrees of freedom, 1.9905 on 79, 1.9736 on 175), with a month either side of the record that no
  interval may include. The same test pins the pooled detectable figure at 0.0029 - `(3 + 0.8434) x
  se`, not `3 x se` - which catches two of the three `detectable` mutations as arithmetic.
- **What "cleared its bar" means** (`or` -> `and`): an alternative clears by either route, the
  held-out judgement or 2012-2019 in the 2020-2026 direction. Two new tests count one row clearing
  by each route and one clearing by neither.
- **The Sahm trigger at exactly 0.50** (`>=` -> `>`): the only reading ever handed to `readings`
  in the whole suite is 0.6, so the bar itself was never probed. Now 0.49 / 0.50 / 0.51.
- **The window that makes the Sahm low a *prior* 12-month low** (dropping the `shift(1)`): this
  one is the instructive failure. Both existing fixtures do real arithmetic - the one in
  `test_club_readiness` even restates the window independently as `avg3.iloc[-13:-1]` - and both
  were blind, because **every Sahm fixture in the register is a *rising* unemployment series**,
  where the prior-12 low and the inclusive-12 low are the same month. The new fixture falls: 5.0%
  down by 0.1 a month gives -0.1 against the mutation's flat 0.00, and the mutated code could
  never print a negative reading at all.

**One register finding that is not about any single claim.**
`tests/test_engine_calibration.py` - seven synthetic tests that plant a known IC difference and
check the engine measures it back, which is the whole reason to believe "we found nothing" means
there was nothing to find - was named by **no** claim. So `scripts/check_published_claims.py`,
whose membership is derived from `checked_by`, had never run it before publishing the page that
quotes its result. It is in the gate now by derivation (37 modules; 23 s, synthetic, fixed seed,
no network, no run directory).

**Four of the six gaps change no number published today** - the boundary, the `or`, the Sahm bar
and the Sahm window are all equivalent mutants on today's data. They change what the *next* value
would be. The audit's Method notes that class in a paragraph; on this pass it was the majority,
which is the argument for mutating rather than reading.

**What was caught, for the record** (11 of 17, so the register is mostly doing its job): the
Bonferroni correction in `pass_bar`; the alternatives count excluding the baseline; all three
put-skew mutations (the 10-60 day window flag at its ends, the nearest-target pick with `abs()`
dropped, and the put leg's target moved to the band's midpoint while leaving the call leg alone);
and both `context.track_turnover` mutations (the half in one-way turnover, and the factor of 2 that
charges the break-even cost on both sides of a trade).

### Evidence / research

- This is test-of-tests work, so the evidence is the mutations themselves: 17 applied one at a
  time, each a single exact-string replacement that had to occur exactly once, each followed by
  running **only** that claim's `checked_by` (the coverage the register actually buys, and what
  the publish gate derives), then `git checkout` and a clean `git status` before the next.
  The harness is ~60 lines and deliberately lives outside the repo (`plan/claims-mutation-audit.md`,
  "Method"): a mutation tool in the tree is one bad `finally` away from committing a mutant.
- The regeneration experiment is the new measurement: mutate, run `v2_experiments.py report`,
  then run the claim's checks. 7 of 7 still green. That is what demotes a drift tripwire from
  "a check" to "a staleness alarm".
- `research/2026-10-15-research-engine-first-batch.md` and `research/2026-10-25-positive-control.md`
  for what the figures being pinned actually are. Nothing here uses or quotes an engine number as
  evidence for anything (rule 5); the subject is the tests, not the results.

### Methodology changed

- None. No weight, metric, threshold or published sentence moved. `claims.py` gained seven
  `checked_by` entries across four claims and no new `asserts` text.

### Tried and rejected

- **Registering the three remaining computational sentences on the research page** - "the ranking's
  monthly correlation swings by about X from one month to the next" (`se x sqrt(n)`), "the smallest
  difference the test could reliably see was about X over 2012-2019 and Y over the held-out months"
  (the **median** alternative's standard error through `detectable`), and "N months of record to
  show a monthly correlation of 0.05 at the usual 5% level" (`months_needed`). All three say how a
  number is computed and **none is in `claims.py`**, so they sit outside the publish gate's
  derivation and outside this audit. I checked rather than assumed: `grep` over `claims.py` finds
  only the control's "reliably detect". I did not register them today, because the rule for a new
  claim is that it ships with a test on both sides of its number, and the one that matters -
  pinning that the two headline figures come from the *median* row's standard error rather than the
  mean or the smallest - needs a synthetic `rows` structure through `research_page.build`, and
  writing it in the last hour of a session is how you get another test that restates the code.
  It is item 3 of the plan's next list with the work specified.
- **Treating rows 89-91 as closed by the tests that already existed.** They were caught by
  `test_the_bar_at_38_held_out_months...` and the calibration module, both unregistered, so the
  claim bought nothing. The 2026-10-25 rule is that finishing a gap means registering the fix;
  three registrations were the fix.

### Observed, not acted on

- **Gate 2 dirties the tree gate 4 inspects.** `python run_screener.py --dry-run` rewrites
  `sp500_tickers.json` - today a pure reordering (PEG and PTC swap), same content - so a session
  that runs the gates in order sees gate 2 pass and gate 4 report a modified tracked file. It is
  known and handled: both runners carry the file in their restore list
  (`nightly-screener.ps1:105`, `data-run.ps1:527`) and `prompts/nightly.md` tells sessions not to
  commit it. Recorded because it costs a minute of doubt every time and a session reading gate 4's
  output cold would not know.

### Next

- **The plan's next list, item 1**, widened by what today found: re-run the 2026-10-26
  payload-reader scan with the *generated pages* added to what counts as a payload read
  (`research.html`, `SCREENER_OVERVIEW.md`, `index.html`, `summary.json`, and `== _rebuilt()`-shaped
  assertions), then mutate the numbers whose only reader is one of those. Item 2 is a third
  mechanical search for the 2026-10-28 shape - a registered test that computes its expected value
  by calling the same function it is checking.
- Read tonight's 02:00 log: it should publish, and a refused run should now leave `git status`
  clean (the owner-run morning session's fix, first exercise tonight).
- Phase 0 of the universe plan on research Monday **2026-11-02**; the next lab Friday is
  **2026-11-06**.

## 2026-10-29 (06:00 session) - BUILD. Three sentences the research page publishes every run, that nothing in the project could see change

**Health (rule 8, all five):** last code session ran? **yes** - 2026-10-28 13:00 ended "shipped to
main", tag `good/2026-10-28`. Data loop published? **yes** - 02:00 today ended "Data loop complete"
(498 scored, 45 tickers rate-limited on the retry pass, 18 min). Evidence base at `1m` = **39 rows,
newest 2026-09-28 (31 days ago, bound 40), 5 effective**. Priority 0: `allow_auto_apply` **false**,
unchanged. Top open roadmap item: **0.13, the Factor Lab** (opened 2026-10-12, **age 17 days**) -
not advanced, and not drifting: the lab's day is a Friday, 2026-10-30 is an even ISO week so the
runner swaps in the retrospective, and the next lab Friday is **2026-11-06**.
**Tests:** before 3,506 passed / 1 skipped, after **3,510 passed / 1 skipped**.
**Owner queue / rotation:** took `OWNER_FOCUS.md` item 3 of the 2026-10-20 block, "Test the
tests" - the fifth mutation pass, and specifically the item the 2026-10-28 session deferred with
the work specified (`plan/claims-mutation-audit.md`, next-list item 3). No proposal was waiting
(`write_brief.open_proposals()` returns `[]`). The stash `auto-rescue 2026-10-28_060001` is
**empty** (`git stash show -p` prints nothing, `--stat` nothing): the 2026-10-28 morning session's
fix cleaned the refused run's files through `Undo-RunArtifacts`, so there was nothing left for the
stash to hold - and its own entry says not to pop it. Left in place, nothing lost. The top open
item in `OWNER_FOCUS.md` (universe expansion) is a Monday research item by its own Next line;
nothing else deferred. Thursday's focus is BUILD and the week's research justified no methodology
change (2026-10-26's proposal was refused on 10-26 and again confirmed 10-27), so by the rotation's
own rule this is the top open owner item instead of an invented change.

### Did

**The research page's "How long would it take to prove an edge?" section states three numbers, and
`claims.py` registered none of them.** They are registered now - `research.ic_swing`,
`research.months_to_prove`, `research.detectable_difference` - each with checks on both sides of
its figure. Rows 106-117 of `plan/claims-mutation-audit.md`. No weight, metric, threshold or
published sentence moved; the diff is the register, four tests, one test strengthened, and the
write-up.

The three sentences:

- **"The ranking's monthly correlation swings by about 0.12 from one month to the next"** - the
  standard *deviation* of the baseline's monthly rank correlation over 2020-2026, recovered from
  that ledger row's standard error as `se x sqrt(n)`.
- **"31 / 59 / 164 months of record"** to show a correlation of 0.05 at the usual 5% level, the
  same at a t of 3, and 0.03 at a t of 3 - `((z + 0.84) x the swing / the correlation)^2`, the
  record needed to see it 80% of the time.
- **"The smallest difference the test could reliably see was about 0.037 over 2012-2019 and 0.062
  over the held-out months"** - the engine's `detectable` applied to the **median** alternative's
  standard error *in each period*, over that period's own month count, corrected for the number of
  registered comparisons.

**12 mutations, 12 caught.** That is the expected result for tests written the same day and it is
not the measurement worth keeping. This is:

**Before today, one mutation per claim passes the entire suite.** Not "passes the claim's checks" -
they had none. Each mutation was applied, the research page regenerated the way
`python v2_experiments.py report` publishes it, and then `tests/ test_screener.py` run with only
today's four new tests deselected: **3,506 passed, 1 skipped, three times out of three.** So these
were not under-checked sentences. Nothing anywhere in the project could see them change. Taking the
swing off the wrong era moves the published figure from 0.10 to 0.12 and all three month counts
with it (45 / 85 / 236), and would have shipped.

**Two findings generalise, and both are about what "caught" means.**

- **A test the register does not name is not coverage.** Two of the twelve *were* caught before
  today, by `test_signs_and_zero_are_printed_honestly` - which no claim named, so
  `scripts/check_published_claims.py`, whose membership is derived from `checked_by`, had never run
  it before publishing. "Something in the suite fails" and "the loop refuses to publish" are
  different facts, and only the second protects a reader.
- **A copied expression can catch a mutation, and that is the trap.** The test that caught those
  two asserted `rp.months_needed(0.05, 0.12, 3.0) == round(((3.0 + 0.84) * 0.12 / 0.05) ** 2)` -
  the function's own expression, copied, so it passes whatever that expression becomes. It survived
  only because both mutations changed the code *beside* the copied line (`int(round(` and the
  `+ 0.84`) rather than the line itself; a mutation of `** 2`, or of both sides together, would
  have gone straight through. So the 2026-10-25 rule needs its corollary: a row-59 check does not
  reliably read as a gap, it reads as coverage until a mutation happens to land on the copied part.
  Do not take "this mutation was caught" as evidence a check is sound - look at whether the
  expected value is a literal. That line is literals now: 85 and 45.

**One structural fix: `claims.RESEARCH` is a `detect` surface at last, and the generated page.**
The register could not carry a pattern for a research-page sentence before today -
`test_every_detect_pattern_still_matches_its_surface` had text for two surfaces and asserts
`text is not None` - and nothing had forced the question, because both pre-existing research claims
are one of three mutually exclusive states each and legitimately carry `detect=None`. The new
`generated_research` fixture builds the page from the committed ledger, so a reword goes stale on
the session that makes it rather than on whichever session next regenerates the page (rule 10).

**How the checks are built, since this is the part that decides whether they are worth having.**
Every alternative's comparison statistics are replaced in a deep copy of the committed ledger and
everything else on the page is left real, so the tests exercise the function the 02:00 loop calls.
The standard errors are a *plateau* wide enough to cover the median index whatever the row count,
with two rows below it, two above, and a distinctive value on the row that sits in the middle of
the **mean** ordering - so the figure the test demands cannot be produced by the mean (0.0062), the
smallest (0.001), the largest (0.100), or the row `build` happens to have at that index (0.030).
The injected `detectable` returns a value encoding **both** of its arguments, which is what makes
"the wrong period" and "the wrong month count" separately visible: the page must print
0.400 / 0.603, against 0.600 / 0.403 and 0.403 / 0.600 for those two mistakes. A second test uses
the real `detectable` with the registered count pinned to a literal 29, so the ledger growing
cannot move the expected figures (0.017 / 0.025).

### Evidence / research

- The mutations themselves, 12 of them, applied one at a time as a single exact-string replacement
  that had to occur exactly once, each followed by running **only** that claim's `checked_by` (the
  coverage the register actually buys, and what the publish gate derives), then `git checkout` and
  a clean `git status` before the next. Harness outside the repo, per the plan's Method.
- The blindness measurement is the new one and it is the reason to believe the work was needed:
  mutate, regenerate, run the whole suite minus today's tests. 3 of 3 blind. Recorded with the
  exact commands in `plan/claims-mutation-audit.md`, "Results, 2026-10-29".
- `research/2026-10-15-research-engine-first-batch.md` and `research/2026-10-25-positive-control.md`
  for what the figures being pinned are. Nothing here uses or quotes an engine number as evidence
  for anything (rule 5); the subject is the checks, not the results.
- Gate arithmetic: the four new tests enter `scripts/check_published_claims.py` by derivation
  without adding a module - all four live in `tests/test_research_page.py`,
  `tests/test_v2_experiments.py` and `tests/test_engine_calibration.py`, which the gate already ran
  (68 modules, `EXCLUDED` empty). Measured: the gate is **1,798 passed** in 123 s, against 1,781 on
  2026-10-28, and the full suite 3:01 before and after - the four tests cost nothing in parallel.

### Methodology changed

- None. No weight, metric, threshold or published sentence moved. `claims.py` gained three claims
  and `tests/test_claims_register.py` one fixture; `research_page.py` is untouched.

### Tried and rejected

- **Registering the difference chart's geometry in the same session.** Each row of the headline
  chart maps its difference, its 95% interval, the two pass-bar ticks and the detectable band onto
  a fixed +/-0.06 axis through `_pos`, and the suite pins none of it: the only assertions are that
  the row count matches `summary["alternatives"]` and that there are two ticks per row
  (`tests/test_research_page.py:49,205`). A sign error in `_pos` would put every dot on the wrong
  side of zero while every printed number stayed correct. I did not register it today for a reason
  that is also a scope question worth leaving on the record: the register's stated scope is
  *sentences* that state the arithmetic of a published number, a chart is not a sentence, and
  deciding that question in the last hour of a session is how a register stops being maintainable.
  It is written into the plan's next list with what it needs, beside the two other unregistered
  research-page figures (the overlap counts, the bot table's intervals).
- **Treating "12 of 12 caught" as the result.** It is the weakest number in this entry: the tests
  were written against those mutations. The pass is only worth reporting because the *prior* state
  was measured against the whole suite, and a session that reports the first number without the
  second is reporting that it wrote tests.

### Next

- **The plan's next-list item 1, which this pass rewrote and now argues is the highest-value one:
  the sweep for computational sentences that are not registered at all.** This audit can only
  mutate what the register names, and the three added today were found by reading one section of
  one page. Sweep each generated surface the way `test_every_sentence_builder_is_registered` sweeps
  `stock_summary.py`: every f-string in `research_page.py`,
  `run_screener.build_screener_overview` and `generate_dashboard.py` that interpolates a number
  into prose needs a claim or an explicit exemption. Start with `research_page.py` - small, and all
  of its numbers are statistics.
- Phase 0 of the universe plan on research Monday **2026-11-02**; the next lab Friday is
  **2026-11-06**; 2026-10-30 is a retrospective (even ISO week).

## 2026-10-29 (owner-run, morning) - both overnight runs healthy; the claims gate's log line fixed

| Check | Reading |
|---|---|
| Last code session | 06:00 ran, all four gates passed, shipped to main (`good/2026-10-29`) |
| Data loop | 02:00 published run `a08c2741a3ca`, 498 scored, HEALTH: PASS, "Data loop complete" - the 10-28 fixes held |
| Evidence base, `1m` | 39 rows; 5 effective observations (data-run log); newest `run_date` not re-read this session |
| Priority 0 | unchanged - `allow_auto_apply` false |
| Top roadmap item | 0.13 Factor Lab, opened 2026-10-12, 17 days |

- **Fixed: `scripts/check_published_claims.py` logged the wrong line.** This morning's data loop
  printed `PUBLISHED CLAIMS: PASS (ConnectionAbortedError: [WinError 10053] ...)`: the summary was
  the last line of stdout+stderr, and a browser test's teardown noise on stderr came after pytest's
  count. The verdict (exit 0) was correct. `result_line()` now picks pytest's own "N passed ... in
  Xs" line. Real run on the live artifacts: `PASS (1798 passed, 312 warnings in 145.87s)`.
  `tests/test_published_claims_gate.py`, +5 tests.
- **One flaky observation, recorded rather than explained away:** the first full-suite run failed
  `tests/test_dead_badges.py::test_the_substituted_figure_survives_a_missing_counterpart` (a
  Playwright test). It passed alone (18/18) and the full rerun passed 3,515 / 1 skipped. Nothing in
  this change touches that module or the page. If it fails again under `-n auto`, it is worth a
  look at its wait conditions.
- `stash@{0}` (the 10-28 refused run's three files) is still there, untouched.

## 2026-10-29 (13:00 midday session) - QUEUE. The sentences nothing was watching: a sweep for computed numbers on the research page that no claim names

**Health (rule 8, all five):** last code session ran? **yes** - 2026-10-29 06:00 shipped to main
(tag `good/2026-10-29`), and an owner-run session merged a claims-gate log fix after it. Data loop
published? **yes** - 02:00 today ended "Data loop complete", HEALTH: PASS, run `a08c2741a3ca`, 498
scored. Evidence base at `1m` = **39 rows, newest 2026-09-28 (31 days ago, bound 40), 5
effective**. Priority 0: `allow_auto_apply` **false**, unchanged. Top open roadmap item: **0.13,
the Factor Lab** (opened 2026-10-12, **age 17 days**) - not advanced, and not drifting: the lab's
day is a Friday, 2026-10-30 is an even ISO week so the runner swaps in the retrospective, and the
next lab Friday is **2026-11-06**.
**Tests:** before 3,515 passed / 1 skipped, after **3,523 passed / 1 skipped**.
**Owner queue / rotation:** the third session today. Carried on from the 06:00 entry's **Next**,
which specified this work: `OWNER_FOCUS.md` item 3 of the 2026-10-20 block ("Test the tests") and
`plan/claims-mutation-audit.md`'s next-list item 3. No proposal was waiting
(`write_brief.open_proposals()` returns `[]`). The stash `auto-rescue 2026-10-28_060001` is still
empty and still untouched (`git stash show --stat` prints nothing) - left in place, as its own
entry says. No owner-session marker. The top open item in `OWNER_FOCUS.md` (universe expansion) is
a Monday research item by its own Next line; nothing else deferred.

### Did

**`claims.py` registers the sentences someone thought to write down. This sweeps for the ones
nobody did.** Rows 118-126 of `plan/claims-mutation-audit.md`. No production code, no weight, no
threshold and no published sentence changed; the diff is one new section in
`tests/test_claims_register.py`, five tests, one claim and the write-ups.

**The argument for doing it.** This morning's session registered three computational sentences on
the research page and measured that each was invisible to the whole suite. It found them by
*reading one section of one page*. That is attention, not a method - and the mutation audit this
item has been running since 2026-10-20 **cannot reach the class at all**, because mutation can
only mutate what the register names. The gap is the one `claims.py`'s own docstring describes:
"nothing enumerated the *set* of computational claims the site makes".

**So the register now sweeps a generated page the way it has always swept `stock_summary.py`.**
`test_every_sentence_builder_is_registered` works because a new `_sentence_*` cannot ship
unregistered; no generated page has a naming convention to hang that on. `computed_sentences()`
supplies one structurally: it walks `research_page.py`'s f-strings, reconstructs each as the text
the page prints, splits it into sentences, and keeps those that interpolate a number **this repo
computed** - a numeric format spec, arithmetic, or a call to one of the generator's own statistics
helpers. **32 sentences.** Each must carry a verdict - `claim:`, `checked:`, `reported:` or
`open:` - and a new or reworded one fails the suite until it gets one.

**Two rules in it that are worth more than the code.** `checked:` and `reported:` mean *every*
computed figure in the sentence is pinned by the named test; **one unpinned figure makes the whole
row `open:`**, because a row that reads as coverage it does not have is the exact failure the
register exists to catch. And every verdict was assigned by reading the named assertion, not its
name - which is how the trading-rule row below was found.

**What the sweep found: 12 of the 32 sentences have at least one computed figure pinned by
nothing.** Each is listed by a literal slug in `KNOWN_OPEN` with what it needs. The three sharpest:

- **The pass bar is printed twice on the page and nothing pins either copy.**
  `test_the_bar_at_38_held_out_months...` pins `X.pass_bar` as a function; nothing says the page
  calls it with the median row's own month count per era, and nothing says the two printings agree.
- **`_pbo_words` is probed at 0.21, 0.5 and 0.73 and never at its own bars, 0.45 and 0.55.** That
  is the fourth time this audit has found a threshold tested either side of itself and not at it.
- **The trading rule's monthly figure is a wildcard.**
  `test_the_trading_rule_sentence_follows_each_period_s_sign_and_t` matches it as `[0-9.]+%`, so
  the sign, the words and the t-statistic are pinned and the percentage - the number a reader
  actually takes away - is not.

**One closed the same session: the universe paragraph**, which the 06:00 entry named as a residual.
It states four figures and until today one of them was pinned (the baseline's stocks a month) plus
the words around the other three. It is `research.pairing_overlap` now, with five checks: the
narrowest alternative is the *minimum*, against a decoy ranked wider; the overlap is the mean of
that screen's own monthly counts **inside** the 2020-2026 record, against two months dated outside
it carrying 999; the two subtractions either side of the overlap are different numbers (44 and 4 in
the fixture, so a swap shows); the branch turns at **exactly** 99% of the screen's universe; and a
row with no monthly counts prints no overlap sentence rather than a mean of nothing.

**9 mutations, 9 caught by today's tests, and 9 of 9 blind to everything else.** Method as this
morning's: one exact single-occurrence replacement, then `python v2_experiments.py report` to
regenerate the page the way it is actually published - without that step the drift tripwire fires
and tells you nothing, the 2026-10-28 lesson - then the whole suite with only today's eight tests
deselected. **3,515 passed, 1 skipped, eight times out of eight.**

**The number to read is row 124.** Appending a brand-new sentence that states the arithmetic of a
published figure - "and that is what each comparison is measured on, N figures in all" - is
**invisible to 3,515 tests**. That is the class the audit could not reach, measured rather than
argued, and the sweep is the one thing that makes it impossible.

**Row 125 is why the key is the sentence with its expressions in place** rather than a `{}`
template: rewording a registered sentence is also blind to everything else once the page is
regenerated, and a placeholder key would additionally miss swapping *which* expression is printed -
which is exactly how row 106 took a figure off the wrong era.

**And row 126 is the sweep pointed at itself.** Break `computed_sentences` so it stops matching and
the `>= 25` floor fires with its own message ("a sweep that stops matching passes vacuously; fix
`computed_sentences`, do not trim the table"). A sweep with no floor is CLAUDE.md rule 8's
decoration.

**Two implementation facts, both found because the sweep's first output was wrong** - worth
carrying to the other two surfaces:

- **Inline tags are inside a sentence; every other tag ends it.** A figure is wrapped in
  `<span class="num">`, so breaking on all tags separates every number from its words - and
  stripping all tags without breaking produced one key running from the chart legend through the
  whole alternatives table to the next heading.
- **A stat card is a figure in one element and its caption in the next.** Block-level breaking
  leaves the figure wordless and the caption numberless, and **all three month counts registered
  this morning disappeared from the sweep** until a figure-only unit was made to take the following
  unit's words.

### Evidence / research

- The nine mutations, applied one at a time as a single exact-string replacement required to occur
  exactly once, each followed by a regeneration and then (a) only today's tests and (b) the whole
  suite minus them; `git checkout` and a clean `git status --porcelain` between every one. Harness
  outside the repo per the plan's Method - a mutation tool in the tree is one bad `finally` away
  from committing a mutant.
- The blindness measurement is the one that justifies the work: 9 of 9 caught now, 8 of 8 blind
  before (the ninth is the sweep's own floor and has no "before"). Commands and figures in
  `plan/claims-mutation-audit.md`, "Results, 2026-10-29 (13:00 session)".
- Coverage arithmetic for the publish gate: the new tests add **no module**. All five live in
  `tests/test_research_page.py` and the three structural ones in `tests/test_claims_register.py`,
  both already in `scripts/check_published_claims.py`'s derivation - and the new claim's
  `checked_by` keeps them there by construction, so the gate runs them whether or not the curated
  list ever mentions them.
- Nothing here quotes an engine or backtest number as evidence for anything (rules 4 and 5). The
  subject is the checks, not the results.

### Methodology changed

- None. `claims.py` gained one claim (`research.pairing_overlap`) and no `FORBIDDEN` or `asserts`
  text moved. `research_page.py` is untouched.

### Tried and rejected

- **Requiring a claim for every number in prose, with no `reported:` verdict.** The honest line is
  that a figure read verbatim out of the ledger with no formatting is a *reported* number, and the
  page-matches-its-source tests are what cover it. Collapsing that into "needs a claim" would have
  put ~20 more rows in the table for no gain in coverage, and a table nobody can read gets
  rubber-stamped - which is how a register becomes documentation (this file's first line).
- **Deriving `KNOWN_OPEN` from the table** (`frozenset(s for s, v in ... if v.startswith("open:"))`),
  which is what I wrote first. It is the table compared with itself and passes at any value - row
  59's rule applied to the ratchet instead of to a constant. The slugs are literals now, which also
  means rewording a sentence does not churn the ratchet; only opening or closing one does.
- **Bringing the bot table's 95% intervals (`_ex`) into the sweep.** "95%: X to Y" in a table cell
  is a published figure with no prose around it, so it is not a sentence, and the register's stated
  scope is sentences. Widening that scope to table cells is a real decision with a real cost in
  table size, and the last hour of a session is not where to make it. Recorded in the plan with
  what it needs; the chart's geometry, the other named residual, **is** now an enumerated `open:`
  row rather than out of scope.
- **Sweeping `generate_dashboard.py` in the same session.** 13,485 lines, and the value of this
  mechanism is entirely in the verdicts being read rather than guessed. One surface per session is
  in the plan as item 0.

### Next

- **The rest of the sweep: `run_screener.build_screener_overview`, then `generate_dashboard.py`,
  one surface per session** (`plan/claims-mutation-audit.md` item 0). `computed_sentences()` takes
  a path and needs nothing new; the work is the table, and a verdict means reading the assertion.
  The overview is the public methodology page, so expect the larger yield there.
- Or take from the twelve open rows, which are a queue with the work specified. `_pbo_words` at
  0.45 and 0.55 is the cheapest; `bot-monthly-excess-figure` is the one most likely to be wrong
  without anyone noticing.
- 2026-10-30 is a **retrospective** (even ISO week). Phase 0 of the universe plan on research
  Monday **2026-11-02**; the next lab Friday is **2026-11-06**.

## 2026-10-29 (owner-run, evening) - the evidence engine kept dropping stocks the price source stopped serving

| Check | Reading |
|---|---|
| Last code session | 13:00 ran, all four gates passed, shipped (`good/2026-10-29-1403`) |
| Data loop | 02:00 published `a08c2741a3ca`, HEALTH: PASS, "Data loop complete" |
| Evidence base, `1m` | 39 rows, newest 2026-09-28 (31 days, bound 40), 5 effective |
| Priority 0 | unchanged - `allow_auto_apply` false |
| Top roadmap item | 0.13 Factor Lab, opened 2026-10-12, 17 days |

Owner: "find some improvements you could make now before the next session".

- **Evidence-base correction (changelog 2026-10-29 evening).** Traced the yfinance "possibly
  delisted" lines every data run has logged since 2026-10-20 to `compute_forward_returns`: a
  snapshot ticker missing from the download was skipped, and the free source drops a renamed or
  taken-over company's whole history. 96 stock-dates were missing across all 39 measured `1m`
  dates (1-3 each), up to 8 on a `3m` date. The engine now calls the track record's own rules
  (renamed -> new symbol by CIK; otherwise recorded closes, held at the last one - the S&P exit
  rule from `research/2026-09-30-delisted-price-source-cost.md`). Forward-only: no recorded row or
  IC changes. Dry run on a copy of the real data with 09-28 re-measured: the 499 stocks measured
  before came out identical and PSKY (-32.8%, its real fall to its last trade on 10-16; Yahoo
  serves nothing after) and WBD (+0.2%) were added. Offline re-pricing of the 39 dates moves a
  date's IC by 0.002 on average, 0.0055 at most - supporting colour only.
  `tests/test_forward_returns_lost_tickers.py` (7; the two behavioural tests fail on the old engine).
- **Runner log line.** `nightly-screener.ps1` gate 1 logged "PASS ( -- Docs: https://... 3523
  passed ...)"; a pass now logs pytest's count line (parse-checked, 0 errors; pattern tested in
  PowerShell and in `tests/test_parallel_suite.py`).
- **Not fixed, on purpose:** this morning's single `test_dead_badges` failure. No earlier failure
  in any log and its error text was not kept, so there is nothing to diagnose; tonight's gate 1
  (3,531 passed, 1 skipped) passed it. If it recurs, keep the traceback.
- Still open: the second yfinance failure block (CTRA, DAY, MMC, BK, HOLX - "no timezone found")
  is printed twice per run and some of it is probably the track record's own price download; it
  costs a few seconds and changes nothing, so it was left.

## 2026-10-29 (owner-run, night) - sweeps of what the site actually renders: three defects, two false alarms

Same health readings as the evening entry two hours earlier (nothing ran in between). Owner: "keep
finding improvements. be creative". The method was to read what the site renders, not the code
that renders it.

- **The brief's evidence line read the wrong horizon.** `write_brief.ic_observations` printed the
  newest run date across all horizons against a 21-day bound ("newest 2026-10-21" while the `1m`
  date was 2026-09-28) - the tripwire CLAUDE.md rule 8 retired on 2026-09-18 because it cannot stand
  still, still live in the brief the owner reads. Now the newest `1m` run date, its age and STALE
  past 40 days. `tests/test_morning_brief.py::TestEvidenceLine` (4).
- **Every stock panel opened on the live page (498, no JS errors): two signed zeros.** AMD's
  five-year earnings-yield range "-0.0% - 3.2%" and INTC's median "-0.0%" - `cPlain`, the context
  panels' percent formatter, the BXP defect class in another function. One more decimal when a
  value rounds to zero ("-0.02%"). The sweep is now `tests/test_every_panel_reads_clean.py` (12 s,
  fails on exactly AMD and INTC without the fix), in the code-loop suite and **deliberately not** in
  the data loop's publish gate: a cosmetic glitch should cost a fix, not a day of evidence.
- **The methodology said "NaN" eleven times** - edited in `build_screener_overview`, overview
  regenerated, page rebuilt, payload byte-identical.
- **Checked and left:** 8 summary sentences read "Sustainable Growth (0.0%)" - true zeros: 35 of 441
  stocks sit at the metric's 0 floor (payout above earnings, mostly REITs), documented on the metric
  row; it belongs to the open Growth-composition question
  (`research/2026-10-12-growth-category.md` section 11), not a formatting fix. Insider titles with
  doubled spaces are the filers' own text. The full-text sweep of `research.html` and `lab.html`
  found nothing; an anchor sweep's two hits were false (the lab's hash routes, and the methodology's
  `#bank-specific-scoring`, which `methodAnchorClick` resolves by slug).
- Gates: 3,547 passed / 1 skipped; dry run, both payloads parse, clean tree; `rehearse_publish.py`
  PASSED.

## 2026-10-29 (owner-run, late) - the lab's signed zeros, and the quarter the source leaves empty

Health as the evening entry (nothing scheduled ran since). Owner: "keep finding improvements to make!"

- **Factor Lab: "-0.00" in 12 of 17 experiments.** A sweep opened every experiment and moved every
  control through its values (each select option, each range's ends, each toggle): "Sharpe ratio
  -0.00", "Momentum averaged -0.00% a month", decile tables, and four experiments opened on a
  "-0.00" difference beside "within". `f2`, `sf2` and `diffText` in `lab.html` took the sign from the
  raw value and rounded after; now from the printed figure. Re-sweep: 0 of 17.
  `tests/test_lab_reading_and_access.py` +5, all failing on the old page. (The sweep crashed the
  test browser twice at the 11th experiment after ~100 rapid control changes each; a reader
  clicking through all 17 holds about 12 MB of JS heap - measured - so it is the harness, not the page.)
- **Research finding, proposal OPEN: `research/2026-10-29-ttm-fills-the-empty-quarter.md`.** The 15
  stocks scored a quarter behind a filing over 30 days old: the source's own trailing-twelve-month
  income series reaches the filed quarter for 9 of 15, and the implied quarter matches the 10-Q for
  6 of 8 checkable to $0.01bn. Amazon is scored on $90.8bn of net income where its filings to June
  give $135.3bn (the June quarter alone $62.65bn). Five questions to answer before building; no
  code changed. The morning brief lists it with its age.
- **The brief's filings row now carries that count** ("15 scored a quarter behind a filing over 30
  days old"); it had kept the four agreement figures and dropped the one about stale scores.
- **Swept and clean:** every main-page control through every option and 89 column sorts at 1280 and
  375 px; My Holdings with six names including one that left the index and one unscored; duplicate
  ids / unnamed buttons / unlabelled controls / images without alt on all three pages with a stock
  dialog, the methodology and an experiment open; every published price dated the same session.
  DD's "period behind" is the engine's annual fallback from a source quarter with -$1.87bn revenue -
  the guard working.
- Gates: 3,553 passed / 1 skipped; dry run; both payloads parse; clean tree.
- **The `test_dead_badges` flake recurred, and this time the traceback was kept.** The final gate-1
  run on this tree failed `test_the_wider_tolerance_is_stated_for_a_financial` once ([gw4]); the
  run ten minutes earlier on code-identical files (only this log differed) passed 3,553. The panel
  the test read held **EXPE's real provenance** ("Metrics: 27/27 ... Filing Age: 121 days") - EXPE is
  the first key in `stock_detail`, i.e. the `other` stock the preceding test,
  `test_the_four_annualised_stocks_in_the_published_run_say_so`, renders last - not the synthetic
  financial the failing test had just rendered. Ruled out: `renderProvenance` is synchronous with one
  write (`container.innerHTML = html`) and no memo or early exit for that input; it has one caller
  (`openStockDetail`); nothing writes `#modal-provenance` otherwise; no timer, guide or startup path
  opens a stock without a hash; the module passes 8 of 8 runs at `-n 8` alone and in isolation
  renders the synthetic financial correctly. So the synthetic render did not land, or something
  restored EXPE, only under full-suite load - unexplained, and not patched: a guess would be a fix
  in name only. What changed instead is the test's ability to say which: `_prov` now renders and
  reads in one browser call, reads again after a task turn, and fails naming both texts if they
  differ (an overwrite after the render); a wrong first read still fails the sentence assertions as
  before (a render that did not land). Nothing is loosened - both cases still fail.

## 2026-10-29 (owner-run, last before the 02:00 run) - unscored members named; glossary abbreviations; brief polish

Health as the evening entry. Owner: "keep finding improvements please before the 2 am run".

- **An S&P 500 member the run could not score is named and told why.** HONA and VYLR (in the
  index, unscored on 2026-10-29: too little data from the source) were absent from the payload, so a
  search or a saved holding said a company "that has left the index, or whose data could not be
  fetched" is not in the ranking - the first half false. The run already records them
  (`universe.json` "failed"); the generator publishes them as `unscored` (`_run_unscored`, display
  only, `{}` on any failure) and `unscoredMemberText` feeds the search, the palette and My
  Holdings. `tests/test_unscored_members.py` (7). **This changes the payload the 02:00 run
  publishes (+72 bytes); `rehearse_publish.py` PASSED against run `a08c2741a3ca`**, and the
  regenerated block read `{"HONA": "Honeywell Aerospace Inc.", "VYLR": "Vylor Inc."}`.
- **The methodology glossary spells out its abbreviations** - EPS appeared 35 times, ROE 23, P/B 21,
  ROIC 14, most never expanded. One entry, names only (no formulas, so no new computational claim);
  generator edited, overview regenerated, page rebuilt, payload byte-identical.
- **Brief polish:** the run time reads "2026-10-29 02:00", not the raw ISO stamp with microseconds;
  the context row no longer repeats its own label.
- `plan/dashboard-inventory.md` updated (rule 9).

## 2026-10-30 - RETROSPECTIVE. Evaluate whether this routine is actually producing value, and change the process where it is not.

**Health (rule 8, all five):** last code session ran? **yes** - `logs/nightly-2026-10-29_130001-midday.log`
ends "Run complete: shipped to main" (tag `good/2026-10-29-1403`). Data loop published? **yes** -
`logs/datarun-2026-10-30_020001.log` ends "Data loop complete", HEALTH: PASS, run `24d9938b511b`,
499 scored, `signals.json` written with 8 gates all run. Evidence base at `1m` = **40 rows, newest
2026-09-29 (31 days ago, bound 40), 5 effective** - in steady state; the data log's own line
agrees. Priority 0: `allow_auto_apply` **false**, unchanged, 5 of 8 effective. Top open roadmap
item: **0.13, the Factor Lab** (opened 2026-10-12, **age 18 days**) - not advanced and not
drifting: today is an even ISO week so the runner swapped in the retrospective, and the next lab
Friday is **2026-11-06**. (0.10 is older - opened 2026-10-07, **age 23 days** - but its remainder
is Sharpe and Sortino at weight 0, which CLAUDE.md already calls "do them only if a session has
nothing better".)
**Tests:** before **2 failed / 3,560 passed / 1 skipped** - a red `main`, see below - after
**3,566 passed / 1 skipped** (+6: the 2 fixed, plus 4 new - 1 stand-in guard, 1 in the publish
gate's module, 2 in the gate-ordering module). Dry run exits 0.
**Owner queue / rotation:** retrospective, so no rotation item, and `OWNER_FOCUS.md`'s two open
items (universe expansion phase 0, a Monday; the 2026-10-20 fine-tune block) were read and left
open - a retrospective does not work on the screener. **One open proposal is deliberately left**
(`research/2026-10-29-ttm-fills-the-empty-quarter.md`, fill an empty newest quarter from the
source's trailing series, 1 day old): a retrospective is the wrong session to decide a methodology
question, the same call the 2026-10-16 retrospective made and recorded. `git stash list` shows
`auto-rescue 2026-10-28_060001`, still empty (`git stash show --stat` prints nothing) and left in
place as its own entry asks. No owner-session marker.

### Retrospective findings

- **Sessions reviewed: 38** (2026-10-16 to 2026-10-30) - **24 scheduled, 14 owner-run.**
- **Genuinely valuable: 35 | Churn: 0 | Lost: 1 | Stood down correctly: 2 | Failed gates: 2.**
- **And one finding that was not in the logs at all: `main` was red when this session started.**

**0. The red `main`, found at baseline and fixed first.** `tests/test_unscored_members.py` shipped
the evening of 2026-10-29 naming **HONA and VYLR**, the two S&P 500 members that morning's fetch
had left unscored. The 02:00 run on 10-30 served HONA, so it is in the payload,
`notInRunText('hona')` correctly returns `''` for a stock that *is* in the run, and two tests
failed on a page that was right. **`main` was red from 02:21 and nothing said so**: the data run
logged HEALTH: PASS, the brief reported a healthy run, and no commit had touched any code.

Without this session looking, the runner's gate 1 would have failed at ~08:00 on a defect the
session did not cause, the whole retrospective would have been stranded on a branch, and the log
would have read like its fault. Both halves are now fixed - the test, and the blindness (process
changes 1 and 2).

**1. What fraction of sessions produced something genuinely valuable? 35 of 38, with 3 that
were not sessions' fault.** Nothing in the churn column for the second retrospective running. The
losses: **2026-10-28 06:00 never started** - the 10-28 02:00 run was refused by the claims gate
(correctly: a real `cPct` rounding defect on BXP's "case against" line), its discard left three
untracked files, and the runner's auto-rescue stash *saved* but exited non-zero on "failed to
remove output/: Permission denied", so the runner stopped. One refused run cost two sessions. The
10-28 owner session found and fixed both ends. **2026-10-19 13:00 and 2026-10-22 13:00 stood down**
because an owner session was marked in progress - designed behaviour, not waste.

The scheduled twenty-four, named: four proposals decided (10-20 bank P/B ACCEPTED - Berkshire's
P/B was **0.0010, not 1.52**, and had been since March; 10-20 bank ROE/ROA ACCEPTED - 60% of bank
Quality rested on two ratios with no published definition; 10-21 share classes ACCEPTED, with the
designated line declared rather than re-picked after measuring that the rule would move Alphabet
**78 places on a coin flip**; 10-21 bank P/B-ROE pairing ACCEPTED), two refused (10-16 earnings
yield, 10-26 the revisions candidate at 94% momentum overlap), two Monday research notes, the
research engine's provenance and positive control, the tradable-signals work, the claims
register's mutation passes 2-6, the eleven dead badges (one a false sentence on a top-25 stock,
one a false claim on the public methodology page), the flaky ship gate fixed at the root, and
2026-10-26's speed night checked on its first live runs. Every one is a defect fixed or a question
answered.

**2. Which rotation day earns its place? Monday and Thursday clearly; Tuesday once of two;
and Wednesday did not run at all.** Both Wednesdays went elsewhere - 10-21 to the proposal queue,
10-28 lost to the stash failure. The 2026-10-16 retrospective checked Wednesday specifically,
found it earned its place, and recorded that so the next retrospective would not re-litigate it.
Two weeks later it ran **zero** times.

That is not a decorative day; it is a day whose **remit was too narrow to defend itself.** "How
does Monday's research fit the rest of the screener" assumes Monday produced the change, so on a
week when the changes came from the proposal queue Wednesday had nothing of its own and was the
cheapest thing to give away. And the cost is specific: **four proposals were ACCEPTED this
fortnight, three of them changing how banks are scored** - P/B on common equity, ROE and ROA on
net income to common, and the two guarded as a pair - and nothing asked whether one cohort's
Valuation and Quality are coherent after four changes in two days. Widened, in all three places
rule 9 requires (process change 4).

**3. Is the evidence standard holding? Yes - and the citation check I inherited was
miscalibrated.** 11 changelog entries since 10-16; **9 carry an Evidence section**, and the two
that do not are evidence-base corrections, which is right. 26 research notes. The three that are
literature reviews all carry DOIs, which is the 2026-10-16 rule working: macro backdrop 14,
revenue-vs-asset-growth 13, price-target-upside 8.

But `research/2026-10-20-size-weight.md` - 703 lines, 13 sources, the note that settled a weight -
carries **zero DOIs and zero URLs**, and it is the best-sourced note in the fortnight: Fama &
French (1992) *JF* 47(2) Table II Panel A, Fama & French (2008) *JF* 63(4) Table IV, Alquist,
Israel & Moskowitz (2018) *JPM* 45(1), MSCI's Diversified Multiple-Factor methodology sections
2.3-2.4, FTSE Russell's Ground Rules v8.8 rules 5.2/5.5/6.1-6.3 - **and four sources marked
`- not read`** where only an abstract was served. A rule that would fail that note while passing a
note with a DOI and no table numbers is measuring the wrong thing. Recalibrated and made net
stricter (process change 5).

**4. What keeps going wrong?** Two patterns, and the second is new.

- **The old one, fifth instance: a check written for a class of artifact, and the class grows.**
  Today's version is the sharpest yet because it is a *consequence of the 2026-10-16 fix.* That
  retrospective replaced the publish gate's hand-kept module list with a derivation from
  `claims.py`, which made a missing *page sentence* impossible by construction - and a missing
  *evidence file* permanent, because `claims.py` registers published sentences and an evidence file
  has none. **A derived membership rule is only as wide as the thing it derives from.** Closed
  (process change 3); the prompt now asks what the derivation cannot reach.
- **The new one: a test whose premise is a property of one morning's data.** Twice in three days -
  `tests/test_unscored_members.py` (above) and `tests/test_investor_profiles.py` on 10-28, which
  compared the published payload with the *newest* `runs/` folder, a refused run the page never
  showed, and failed on a correct payload. Both reddened `main` with no commit touching code.
  Written into `prompts/nightly.md` section 2 with the rule that closes it: a payload-reading test
  asserts its own premise.

**5. Is the tool closer to being the place you'd look before buying or selling? Closer in one
specific way, and the blocker is unchanged.** Since 10-16 the inputs got materially more correct
where they were worst - a bank's P/B had been wrong by three orders of magnitude for a FTSE-sized
company since March, and 60% of bank Quality rested on two ratios whose definition was nowhere
published - and one company is now ranked once rather than twice. Those are the kind of defect that
makes a screener untrustworthy to someone who knows the sector. But **the blocker is still that
there is no demonstrated edge**: the research engine's pre-registered experiments over 176
point-in-time months still beat nothing, and the baseline's own IC is ~0. The page says so, which
is the right answer, and rules 4 and 5 correctly keep it that way until 2027-02-11.

**6. What is the routine systematically blind to?** Three things.

- **Whether `main` is green between 02:21 and the next session's baseline.** Nothing looked. Now
  the runner does, and labels gate 1's failures against it. The deeper version is in "Flagged".
- **The second publish path's *non-text* artifacts.** Every one of the five holes found at a
  runner read has been on the data loop's side, and the enumeration kept finding page artifacts
  because the checks are about page claims. The evidence files - which feed the *next* run's
  scoring - had never been looked at. `scripts/retro_artifact_coverage.py` now prints the list.
- **The lab's own payload.** `lab_data.js` and its 15 parts are generated by `factor_lab.py
  --build`, served live, and **neither loop rebuilds them**; the tests check the committed payload's
  internal consistency and that `lab_engine.js` reproduces it, but nothing can check that the
  committed payload is what the current `factor_lab.py` would produce, because the library lives in
  the gitignored `data/lab/`. Not closable today; flagged.

### Process changes made

1. **The runner takes gate 1's baseline before invoking the session** (`scripts/nightly-screener.ps1`).
   `CLAUDE.md`'s gate 1 is "no **new** failures vs the baseline you took at session start", and
   the runner - the real gate since 2026-09-04 - never took one, so it could not distinguish a
   failure the session caused from one it inherited, and the session was never told either. It
   publishes nothing and decides nothing: the same command as gate 1 resolved **once** so the two
   cannot drift, `Restore-Artifacts` after it as gate 2 already does, and an unavailable baseline
   reported as unknown. **Gate 1's verdict is unchanged** - nothing merges onto a red `main`, which
   section 4 of the retrospective prompt requires - it now also logs how many failures were new and
   how many inherited. Two new tests pin exactly that: the baseline block may not merge, push, tag
   or `Stop-Run`, and the attribution block may not alter the verdict.
   `tests/test_gate_ordering.py` **11 passed, was 9.**

2. **`tests/test_unscored_members.py` no longer names the stocks one fetch failed on.** The page
   tests use stand-ins (`QZZA` named, `QZZB` unnamed, `ZZZZ` not a member) and a new test fails if
   a stand-in ever appears in the published run - the premise the originals asserted nowhere.
   **8 passed, was 5 passed / 2 failed.** `prompts/nightly.md` section 2 carries the rule and both
   instances.

3. **The evidence files the data loop appends to are gated.** `improvement/dispersion_history.csv`,
   `improvement/snapshots/` and `factor_vol_history.csv` are all in `$DataArtifacts`, all appended
   to every weekday, and the publish gate could see none of them.
   `tests/test_one_observation_per_run_date.py` holds the only tripwire on the real committed
   copies and now runs in the gate (11 tests, 1.5 s, offline). **Gating rather than reporting
   because `factor_vol_history.csv` is read by `factor_engine.adjust_momentum_weight`**, so a
   duplicated row changes the *next* run's momentum weight - a methodology change with no changelog
   entry, which rule 3 forbids - and a double-counted day in `dispersion_history.csv` moves
   `check_run_health`'s own trailing-median bar. Both files are written during the run, so they are
   on disk before any gate.

   Also corrected the converse test's docstring, which **asserted a check that does not exist**:
   "evidence and data files ... are guarded by `check_run_health.py`". It is not - that gate judges
   fetch evidence, coverage, dispersion, a vanished weighted metric and the 2026-10-22 input gates,
   and never one-row-per-run-date. A comment claiming a check exists is what this project calls an
   unchecked claim, and this one hid the gap.
   `tests/test_published_claims_gate.py` **36 passed, was 35.**

4. **Wednesday's subject is whatever changed the methodology since the last synthesis session**,
   not only Monday's note - `CLAUDE.md`'s weekly-cycle table, `prompts/nightly.md`'s Wednesday
   bullet and the runner's focus string, with the measurement behind it (four accepted proposals,
   three of them bank scoring, zero synthesis sessions).

5. **A research citation carries a DOI, a stable URL, *or* journal + volume(issue) + pages**, and
   **a source reached only as an abstract carries `- not read`.** Net stricter: the second half is
   the honesty requirement the corpus had already invented and the instruction had not asked for,
   and an unmarked citation reads as "I checked the table" whether or not anyone did.

6. **`CLAUDE.md`: 1036 -> 959 lines**, with +4 lines of new constraint in the Settled table, so
   ~81 lines of closed record moved verbatim to `DECISIONS.md` (1048 lines). What moved: the lab's
   experiments 14-17 (a chronology again, exactly as 2026-10-16 found), the four weight questions'
   reasoning, backtest v2's measurement record, T0b, and 0.14's three settled input definitions -
   capex refused, operating income kept, earnings yield refused - each now one line plus the
   section a reopening must answer. **Both sources of regrowth are now named in the retrospective
   prompt**, because they were the same two as last time: a per-item chronology, and a refused
   proposal whose full argument sits under Open.

7. **`prompts/retrospective.md`:** the fifth hole is in the score table with its distinct lesson;
   step 3 asks what the gate's derivation *cannot* reach; the evidence list now says to count the
   sessions that never started and to read the `BASELINE` lines; the scheduled/owner-run split is
   recorded as **24/14 against 11/44**, with the point that the ratio is not a stable fact and the
   rotation's job changes with it. `scripts/retro_artifact_coverage.py` is the throwaway that
   found today's hole, kept and documented as the tool step 1 names.

### Did

See "Process changes made" - that is the session's work, plus the red-`main` fix that preceded it.
Five commits, all process, gates and tests. Nothing in the screener's methodology was touched.

### Evidence / research

- No new literature; this session's measurements are of the project itself, all reproducible:
  38 log entries 10-16..10-30, 24 scheduled / 14 owner-run; 11 changelog entries, 9 with an
  Evidence section; 26 research notes, the 3 literature reviews carrying 14, 13 and 8 DOIs and
  `research/2026-10-20-size-weight.md` carrying 13 sources, 0 DOIs and 4 marked `- not read`;
  2 gate failures (10-18 and 10-19 06:00, both gates 1+4, both the background-job cause fixed by
  `prompts/continuation.md`, **none since**); 1 lost session, 2 correct stand-downs; 10 data runs
  logged, all completing, 1 correctly refused.
- `tests/test_one_observation_per_run_date.py`: 11 tests in **1.52 s**, which is what makes it
  affordable inside the data loop's gate.
- `scripts/retro_artifact_coverage.py` on today's tree: **16 `$DataArtifacts` entries against 69
  gate modules**; the rows that matter are `improvement` (21 modules naming it, all outside the
  gate before today) and `factor_vol_history.csv` (3, all outside).

### Methodology changed

- none. A retrospective does not change methodology, and nothing here touched a weight, a
  threshold or a formula. The one open proposal was deliberately left for a normal session.

### Tried and rejected

- **Running the full suite inside `data-run.ps1` after the publish, non-fatally, to catch a red
  `main` at 02:25 instead of 06:0x.** Attractive - no membership list to go stale, which is the
  lesson of all five holes - and rejected on the risk it creates: the suite would run *after* the
  day's commit, and anything it left in the tree would hand the 06:00 session a dirty tree. That is
  precisely the 2026-10-28 failure, which cost two sessions. Cleaning to `HEAD` afterwards would
  probably be safe, but "probably" is not good enough for an unattended loop that publishes to a
  live site, and the runner baseline buys most of the same information with machinery that is
  already proven. If this is ever revisited, the cheap intermediate step is to run only the
  payload-reading modules outside the publish gate.
- **Registering the unscored-member sentence in `claims.py`** so the module would join the publish
  gate by the derived mechanism. Coherent, and rejected because it changes the *consequence*: a
  claim failure **discards the run**, and a test pinned to a transient ticker would then have cost
  a day of evidence rather than four minutes of a session. The gate is for assertions whose failure
  means *this publish* is unsafe; a stale ticker in a test is not that.
- **Protecting Wednesday from the proposal queue** by making the queue wait a day. The queue
  pre-emption was the owner's own direction (`OWNER_FOCUS.md` 2026-10-20, "Decide the OPEN
  proposals" first) and it produced four accepted changes that fixed real defects, so the problem
  was never the queue - it was that Wednesday had no subject of its own when Monday did not supply
  one. Fixed on that side instead.
- **Changing the retrospective's fortnightly frequency.** Five of five retrospectives have found a
  live hole on the first runner read, which argues for *more often*, not less; but four of the five
  were found by the enumeration rather than by elapsed time, and that enumeration is now a script
  any session can run. Left at a fortnight.

### Next

**Decide the open proposal** in `research/2026-10-29-ttm-fills-the-empty-quarter.md` (fill an empty
newest quarter from the source's own trailing series; it names section 4 as the thing to answer
first) - it is the first job in Orient, and four of this routine's proposals have now been refused
on a measurement the note had not made, so measure before deciding. Then **Monday 2026-11-02 is
phase 0 of the universe-expansion plan**, which is the top open `OWNER_FOCUS.md` item and says so
in its own Next line. The next synthesis Wednesday (2026-11-04) now has a subject waiting: the four
accepted bank and share-class changes.

### Flagged for the owner

- **A red `main` is a four-hour blind spot, and the fix I shipped only shortens it.** The 02:00
  data run can turn the tree red with no commit touching code, and nothing finds out until a code
  session takes its baseline. On a weekday that is ~4 hours. The runner baseline tells the session
  immediately and labels the gate, but it does not tell *you* any sooner, and it cannot act. The
  honest closure is the one I rejected above - the data loop checking the tree it just created -
  and it needs a guarantee that the suite leaves the tree clean, which I was not willing to assert
  unattended on the publishing loop. Worth a decision: accept the window, or spend an owner session
  proving the suite is tree-clean so the check can move to 02:25.
- **The Factor Lab's payload has no drift tripwire, and cannot easily have one.** `lab_data.js`
  plus its 15 parts are served live and generated by `factor_lab.py --build` from the gitignored
  `data/lab/`. `SCREENER_OVERVIEW.md` and `research.html` both have tests asserting the committed
  file equals the generator's output; the lab cannot, because the inputs are not in the repo. So a
  session that changes the lab's arithmetic and does not rebuild publishes stale figures beside
  correct Python, and every test passes. Options are all costs: commit a small fixture of the
  library, hash the inputs into the payload and check the hash, or accept it and make rebuilding
  part of the lab's checklist (it is in `ASSET_VERSION`'s five-things rule, but that pins the
  *version*, not the *figures*).
- **24 scheduled against 14 owner-run, reversed from 11/44 a fortnight ago.** The 2026-10-16
  retrospective concluded the scheduled session's distinctive contribution was "deciding what the
  owner's sessions left open". That was right for 11/44; at 24/14 the rotation is most of the
  throughput and has to carry the roadmap itself, which is also why Wednesday's narrow remit
  started to matter. Nothing to decide unless you expect the split to move back.
- **`NIGHTLY_LOG.md` is 17,037 lines**, up from 11,779 a fortnight ago. Still not breaking
  anything - rule 8 asks a session for its last three entries - but a retrospective reads every
  entry since the last one, and the file is now 45% bigger than when the last one said this. The
  obvious move remains archiving by month with the live file holding the current month. I did not
  do it, for the same reason as last time: splitting the one file every session can find is how the
  memory it preserves gets lost. Your call, and it is becoming the more pressing of the two.

## 2026-10-30 (13:00 midday) - QUEUE. The open proposal decided, then the claims sweep over the public methodology page

**Health (rule 8, all five):** last code session ran? **yes** - today's 06:00 retrospective,
`logs/nightly-2026-10-30_060001.log`, ends "shipped to main". Data loop published? **yes** -
`logs/datarun-2026-10-30_020001.log`, "Data loop complete", HEALTH: PASS. Evidence base at `1m` =
**40 rows, newest 2026-09-29 (31 days ago, bound 40), 5 effective** - steady state, unchanged from
this morning. Priority 0: `allow_auto_apply` **false**, 5 of 8 effective. Top open roadmap item:
**0.13, the Factor Lab** (opened 2026-10-12, **age 18 days**) - not advanced and not drifting; the
next lab Friday is **2026-11-06** and no experiment is half-built. (0.10 is older - opened
2026-10-07, **age 23 days** - and its remainder is Sharpe and Sortino at weight 0, which CLAUDE.md
calls "do them only if a session has nothing better".)
**Tests:** before **3,566 passed / 1 skipped** (green), after **3,570 passed / 1 skipped** (+4,
all in `tests/test_claims_register.py`). Dry run exits 0.
**Owner queue / rotation:** `OWNER_FOCUS.md` item 1 sub-item 1 (decide the open proposal) and then
sub-item 3 (test the tests), in that order, which is the order Orient and the item both give.
Nothing deferred for a gate or a loop. `git stash list` shows `auto-rescue 2026-10-28_060001`,
still empty, left in place as its own entry asks. No owner-session marker.

**Second session today - read the 06:00 retrospective's entry first.** Its **Next** was "decide the
open proposal in `research/2026-10-29-ttm-fills-the-empty-quarter.md`", so that was this session's
first job; nothing it decided was re-opened or reversed. Its process change 1 had its **first live
use** on this session: `logs/nightly-2026-10-30_130001-midday.log` logs
`BASELINE main is GREEN on arrival (3566 passed, 1 skipped ... 0:03:53)`, which matched my own
baseline exactly, so for the first time a session could tell before starting that it had inherited
nothing.

### Did

**1. The open proposal, decided: REFUSED as written - and its own case was understated.**
`research/2026-10-29-ttm-fills-the-empty-quarter.md` §5 (new), measurements
`research/measurements/2026-10-30-trailing-series-review.py` and
`...-trailing-column-moves.py` with their JSON. The proposal: when a stock's newest quarterly
income column is empty (14 stocks on 2026-10-30, Amazon and Bank of America among them), score
its income-statement inputs from the source's trailing-twelve-month series instead. It named five
things to answer first; three are now measured.

**Two answer in its favour, and one of them decisively.**

- **The trailing series is definitionally the same rows under different names.** On the twelve
  highest-ranked stocks that are *not* flagged - so their quarterly statement is the one
  `_stmt_val_ltm` actually sums - across all nine income-statement labels the engine reads, the
  trailing series against the engine's own four-quarter sum: **median relative difference
  0.000000**, max 0.00067 (Total Revenue), 0.00061 (Operating Income), 0.00052 (Net Income); nine
  of twelve agree **exactly** on every label, and the single worst figure anywhere is EG's interest
  expense at 0.68%. So the substitution preserves the 2026-10-15 decision to keep the source's
  pre-special-items `Operating Income` row, which was the largest risk in the proposal. Coverage is
  fine too: of the eight flagged stocks whose trailing column sits on the newest filed period,
  **seven carry all nine labels** (CHRW misses Interest Expense alone).
- **Practice pairs each item at its own latest period, explicitly, for the exact ratio the note
  worried about.** MSCI uses EPS for "Return on Equity (ROE), which is calculated using the
  trailing 12-month earnings per share figure **and latest book value per share**" (*Fundamental
  Data Methodology*, June 2024, §1.2.2 p. 14; §1.2.5 p. 15 defines book value "at the latest period
  end date"), and S&P DJI: "ROE ... is calculated as a company's **trailing 12-month earnings per
  share divided by its latest book value per share**" (*S&P Quality Indices Methodology*,
  Appendix A, p. 16). A June income statement over a March balance sheet is the professional norm,
  and the engine already publishes wider mismatches (FDXF carries `cf 2026-05-31` against
  `is 2025-11-30`).

**What refuses the rule, not the idea.**

- **It keys "reaches the newer quarter" on the newest SEC filing, not on the column the engine
  reads.** Restated against what the stock is actually scored on, the trailing column is newer for
  **13 of 15, not 9** - so the opportunity is larger than the note claimed, and its 9/6 split was
  an artefact of comparing against the wrong reference. But **7 of those 13 are two quarters past
  anything the quarterly table can reconcile them against**: at a one-quarter gap the implied
  newest quarter is `TTM - (three populated quarters)`, which is how the note checked eight against
  the companies' own 10-Qs and found six exact; at two quarters there is nothing to check it with
  (for Amazon `TTM(Sep) - (Jun+Mar+Dec)` is not computable *because June is the empty column*), and
  FedEx's trailing column is two quarters past anything its statement carries. **That uncheckable
  half contains every stock whose fill would move a score materially.**
- **Amazon is the proof that the note's table is one morning's data.** Recorded there as a match on
  10-29; on 10-30 its trailing column has moved to **2026-09-30** while the project's weekly SEC
  cache still holds June, so under the rule as written the build would fill **nothing** for the
  case the proposal is written around. TFC, listed as a miss, is not flagged at all today - it is
  published scored to 2026-09-30, resolved by itself in a day. This is the class the 06:00
  retrospective named this morning, from the other side of the repo.
- **The prior-year half is impossible from this source.** The note guessed the same call returns
  the year earlier "for some stocks and not others"; measured, the trailing series returns
  **exactly one period end, 15 of 15 and 26 of 26**. For none. `_stmt_val_ltm_year_ago` anchors on
  the quarterly statement's newest *populated* period, so filling only the numerator turns
  `revenue_growth` - a weighted metric - and two other prior-year comparisons into **fifteen-month**
  growth rates for exactly the stocks being fixed.

**§5.5 pre-registers the narrow version before anyone builds it** (gap of exactly one quarter so
the implied quarter is computable; the existing negative-revenue guard still applied; the
prior-year anchor moved with the fill or no fill; one change at a time with its `metric_lineage`
and `claims.py` lines) and states the benefit honestly: **6 of 14 stocks**, ranks 52-352, **no
top-25 name**, and not "Amazon's earnings yield goes 3.31% to 4.9%". Whether that is worth building
is left open on purpose - this review deliberately did not build its own corrected proposal.
`OWNER_FOCUS.md` item 1 updated; `write_brief.open_proposals()` returns `[]`.

**2. The claims-register sweep over the public methodology page** - `OWNER_FOCUS.md` item 1
sub-item 3, item 0 of `plan/claims-mutation-audit.md`'s next list, one surface as that list
requires. `tests/test_claims_register.py` section 6 now sweeps
`run_screener.build_screener_overview` the way section 5 sweeps `research_page.py`: every sentence
the generator prints that puts a number this repo computed into prose must carry a verdict
(`claim:`, `checked:`, `reported:` or `open:`), and a new or reworded one fails the suite until it
gets one. **11 sentences; 4 `checked:`, 7 `open:`**, every verdict assigned by reading the
assertion rather than its name. Rows 127-130 of the plan.

**The number to read is that a brand-new sentence stating the arithmetic of a published figure is
invisible to 3,566 tests.** Applied, both published artifacts regenerated the way the 02:00 run
rewrites them, then the whole suite with only today's four new tests deselected: **3,566 passed, 1
skipped.** Same for rewording a registered sentence. `SCREENER_OVERVIEW.md` is regenerated on
every full run, so this is a sentence the loop would have published five mornings a week with
nothing able to notice it.

**The sharpest of the seven open rows:** `test_the_clamps_and_floors_on_the_page_are_the_ones_applied`
says in its own docstring that "the metric rows hard-code their clamps; the Data quality bullet
reads config. Both must be config's" - and then matches only the metric-table rows, so the bullet
is a second printing of four clamp figures that nothing reads. And **no test in the suite reads the
share-class switching margin off this page**: `tests/test_share_classes.py` never calls the
generator, so the 20% is pinned in config and in `share_classes.CHALLENGE_MARGIN` and nowhere on
the surface a reader sees. The bank P/B-ROE section's *table* is pinned cell by cell against an
independent recomputation from config; its three *prose* restatements are not, and one of them
prints a word chosen on `bank_net_pb < 0` that nothing probes at the boundary.

**Two corrections to the audit's own method**, both now in the plan's "Rules that carry forward",
and both found by this pass being wrong first:

- **Regenerate every artifact the publish rewrites, not just the one the generator is named for.**
  The reword first read as **CAUGHT**, by `test_rebuild_page::test_committed_page_is_what_the_generator_builds`.
  That was an artefact: the methodology page is embedded in `index.html`, so rewriting
  `SCREENER_OVERVIEW.md` alone leaves the page drifted and a drift tripwire fires - on a day when
  the real run rewrites both and is green. A mutation that reads as caught because a second
  artifact was left stale is a **false negative for this audit**.
- **A mutation must change the published output, not only the source the checker reads.** The
  new-sentence mutation's first form assigned its sentence to an unused variable: the AST sweep saw
  it, the page did not, and "blind" would have been trivially true. Re-run properly and verified -
  the regenerated page gains exactly one line, `The eight categories carry 100 points in total.`

**Two markdown facts the sweep needed**, both from its own output being wrong: a newline **is** a
block boundary in markdown where in html it is whitespace (the first run produced one key running
from a three-row table through the sentence after it), and **a markdown table row is not a
sentence** - `| Earnings yield (E/P) | **+{bank_net_ep:.1f}** |` has a figure and no three-word run,
so the stat-card merge rule chained the separator and the next two rows into a single "sentence".
Both are negative-controlled by `test_the_markdown_sweep_breaks_on_blocks_and_skips_table_rows`,
which asserts the exact key under both `markup` settings, and the scope is guarded by
`test_the_sweeps_scope_is_the_generator_not_the_file` plus a `>= 9` floor so the sweep cannot pass
vacuously.

### Evidence / research

- **MSCI**, *MSCI Fundamental Data Methodology*, June 2024, §1.1.5 (p. 11), §1.2.2 (p. 14), §1.2.5
  (p. 15). Read from the PDF; §1.1.5 is the completeness rule quoted in §5.2 of the note
  ("updating company information only upon availability of complete information, such as official
  filings ... or upon dual vendor validation").
  https://www.msci.com/indexes/documents/methodology/0_MSCI_Fundamental_Data_Methodology_20240625.pdf
- **S&P Dow Jones Indices**, *S&P Quality Indices Methodology*, Appendix A (p. 16). Read from the
  PDF. S&P's own `spglobal.com/spdji` copy answers **403** to a scripted request, so it was read
  from S&P's `spice-indices.com` copy rather than worked around;
  `data/papers/invesco-sp-quality-indices-methodology.pdf` from an earlier session is a third copy
  of the same document and agrees.
  https://www.spice-indices.com/idpfiles/spice-assets/resources/public/documents/methodology-sp-quality-indices.pdf
- Both providers **normalize** earnings (MSCI §1.2.2 excludes extraordinary and non-recurring
  items). This screener scores earnings as reported by its own settled decision, so practice
  supports the *period mixing* and does **not** support §3's Amazon figure, whose $62.65 bn quarter
  the note itself calls a probable one-off. Recorded in §5.2 rather than used as a refusal.
- Measurements, all re-runnable: `research/measurements/2026-10-30-trailing-series-review.py`
  (+`.json`) and `research/measurements/2026-10-30-trailing-column-moves.py` (+`.json`), read-only,
  ~75 requests between them.
- `tests/test_claims_register.py` is 1,068 lines, 33 tests, 7.0 s - still affordable inside the
  data loop's publish gate, which it is already a member of.

### Methodology changed

- **none.** The proposal was refused, so no weight, metric, threshold, formula or published
  sentence changed, and no changelog entry was due. The sweep is tests and a plan file.

### Tried and rejected

- **Building the corrected version of the proposal today.** Tempting - §5.5 is specific enough to
  implement - and rejected because a reviewer building its own corrected proposal is the route
  without its second reader, and because the honest benefit is 6 of 498 stocks with no top-25 name
  against a scoring change that needs `metric_lineage`, a claim and a prior-year re-anchor in one
  commit. It is written down with its criteria instead, which is what the route asks for.
- **Refusing the fill on the ground that a trailing figure cannot be verified against a filing.**
  Attractive, and wrong twice over: the whole payload comes from this source unverified, and
  gating a *scoring* input on the SEC cache is forbidden outright (CLAUDE.md 0.14 - a test fails if
  `sec_crosscheck` appears in a scoring module). The guard in §5.5 therefore uses the source's own
  data, which is also what MSCI's "complete information ... or dual vendor validation" rule
  actually asks for.
- **Registering the methodology page's seven open sentences as claims in the same session.** The
  enumeration is what makes a *new* one impossible to add unnoticed, and it is the part that cannot
  be reached by mutation; pinning the existing seven is ordinary work a later session can do from
  the queue, one or two at a time, each with a test on both sides of its number. Doing both in one
  session would have meant assigning verdicts I had not read the assertions for, which is the
  failure mode this audit exists to find.
- **Sweeping `run_screener.py` rather than `build_screener_overview`.** It enumerates progress
  lines and log messages as published sentences. `within=` scopes it to the generator, and a test
  fails loudly if the generator is renamed rather than silently sweeping nothing.

### Next

**`generate_dashboard.py` is the last of the three generated surfaces** (`plan/claims-mutation-audit.md`
item 0, now rewritten for it) - html like the research page, 13,485 lines, so decide up front
whether to scope it per generator function with `within=` rather than sweeping the file. Failing
that, the **nineteen `open:` rows** are a queue in their own right: twelve on the research page and
seven on the methodology page, with the cheapest and the most load-bearing named in the plan.
**Monday 2026-11-02 is phase 0 of the universe-expansion plan**, the top open `OWNER_FOCUS.md`
item, and **Wednesday 2026-11-04 is the first synthesis session under its widened remit** with a
subject already waiting: the four accepted bank and share-class changes.

## 2026-10-31 (06:00 weekend) - WEEKEND. A red `main` fixed at its root, and four of the audit's open rows

**Health (rule 8, all five):** last code session ran? **yes** - `logs/nightly-2026-10-30_130001-midday.log`,
ends "shipped to main". Data loop published? **yes** - `logs/datarun-2026-10-30_020001.log`,
"Data loop complete", HEALTH: PASS, signals written (40 names, 8 gates, all ran). Evidence base at
`1m` = **40 rows, newest 2026-09-29 (32 days ago, bound 40), 5 effective** - steady state, one day
further on than yesterday's 31 as expected. Priority 0: `allow_auto_apply` **false**, 5 of 8
effective. Top open roadmap item: **0.13, the Factor Lab** (opened 2026-10-12, **age 19 days**) -
not advanced; the next lab Friday is **2026-11-06** and no experiment is half-built. (0.10 is
older - opened 2026-10-07, **age 24 days** - and its remainder is Sharpe and Sortino at weight 0,
which CLAUDE.md says to do only with nothing better.)
**Tests:** before **3,569 passed / 1 failed / 1 skipped** (RED on arrival, not this session's
doing), after **3,594 passed / 1 skipped**. Dry run exits 0. `scripts/rehearse_publish.py` PASSED
(the diff touches the methodology generator): 362 page tests, published claims 1,834 passed,
published files restored byte-for-byte, all four artifacts unchanged.
**Owner queue / rotation:** the red baseline first, as Baseline requires - then `OWNER_FOCUS.md`
item 2 sub-item 3 ("test the tests"), whose plan names the work. No proposal is waiting
(`write_brief.open_proposals()` returns `[]`). Nothing deferred for a gate or a loop.
`git stash list` shows `auto-rescue 2026-10-28_060001`, **still empty** (`git stash show -p` prints
nothing) - left in place, as every session since has. No owner-session marker.

### Did

**1. The red `main`, and its cause was the calendar.** `tests/test_forward_return_download_skip.py::test_skipping_changes_no_output_only_the_downloads`
failed at baseline, with no commit since 2026-10-26 touching anything it covers. It built its
snapshot dates from `datetime.now()`, and **it cannot fire the skip it tests on a Saturday or a
Sunday**: with an empty performance history the only skippable download is the one whose 1-week
end is today, that needs a *market-day* snapshot exactly seven days earlier, and seven days before
a weekend is a weekend. Today is the first weekend since the test shipped on Monday 2026-10-26.

The scenario is now **a fixed week, Monday to Sunday, parametrized** - every date (snapshots,
prices, and `_close_possible_since`'s notion of today) comes from the frozen date. It also warms
the history up first, which is both what makes the skip reachable on all seven days and the shape
the 02:00 loop actually meets: a run date's shorter horizons were recorded weeks ago, so the
horizon still due is the one whose end is at hand. With an empty history the clock-built version
exercised the skip through **one of its sixteen snapshot dates**.

**Mutation-checked, and the first draft was found wanting.**
`research/measurements/2026-10-31-skip-test-teeth.py`, five mutations of the production rule: the
rule always True, always False, `min` -> `max` over the due horizon ends, the scan starting a day
late, and the weekend counted as a session. The first draft **missed `min` -> `max`** - the
warm-up leaves exactly one horizon due per date, so the two agree - and the whole suite missed it
too: **3,563 passed with the mutation applied.** So
`test_the_skip_reads_the_earliest_due_end_not_the_latest` covers the case that separates them, a
date first processed late with its 3-month end out of reach and its 1-week end long past (which is
what a backfilled snapshot looks like). **5 of 5 caught now.** One mutation was also wrong on its
first draft - it inserted its assignment *above* the real one, which overwrote it - a no-op that
read as a gap; the harness records that, because the same slip would make any future pass
over-report.

**2. The class, not the instance: the clock-read sweep.** This is the **third** red gate in three
weeks from a test whose premise was what the clock said when it ran, and each was fixed in its own
file with the lesson transferring nowhere: `test_forward_eps_growth` at 23:25 on 2026-10-11 (a
tolerance tighter than a day of rounding drift), `test_option_quote_hours` on Saturday 2026-10-17
(a cache fixture stamped with a day the market never opened - the first Saturday session, which
opened on a red gate 1 with neither loop at fault), and this one fourteen days later. The cost is
not the test: a red `main` strands the whole of a session's work on a branch, because gate 1
refuses to merge onto it.

`tests/test_tests_do_not_read_the_clock_unguarded.py` (9 tests) requires every `.now()`,
`.today()` and `.utcnow()` in a collected test module to carry a `# clock: <reason>` note saying
why the verdict is the same whatever the clock says. **AST, not a regex** - a regex counts the
`datetime.now()` inside a docstring explaining one of the faults. It has its own floor (a
synthetic module with four reads and one non-read), a check that the marker needs a reason and not
just the token, one that the comment block has to be contiguous with the call, and one that fails
if the three modules the faults came from stop being reachable. **All 11 existing reads are
annotated**, reasoned one at a time; none of the other nine turned out to be day-dependent, and
`test_forward_eps_growth`'s is hour-dependent-by-a-rounding-day with every tolerance set wider
than the drift, which is now written beside it rather than three paragraphs away.

It cannot check that a reason is *true*. What it does is make the question unavoidable at the one
moment anyone is in a position to answer it - the same argument as the `claims.py` sentence
sweeps, and the honest answer for both weekend faults was "it isn't", which is why both now take
their dates from a fixed parametrized day instead.

**3. Four of the claims-mutation audit's `open:` rows** - `OWNER_FOCUS.md` item 2 sub-item 3;
rows **131-138** of `plan/claims-mutation-audit.md`. The plan's next list calls the nineteen
`open:` rows "a queue in their own right" and names four: the two cheapest and the two most likely
to be wrong without anyone noticing. I took all four rather than starting the
`generate_dashboard.py` sweep, which that list warns is the biggest of the three surfaces and
wants a whole session for its verdicts - and with the morning spent on the red baseline it would
not have been finishable. **All four are closed on the value of the figure; 8 mutations, 8
catches.** `KNOWN_OPEN` drops twelve -> ten.

- **`clamp-bullet-second-printing`.** `test_the_clamps_and_floors_on_the_page_are_the_ones_applied`
  said in its own docstring that "the metric rows hard-code their clamps; the Data quality bullet
  reads config - both must be config's", and matched only the metric rows. The bullet is the
  second printing of all four figures. It is read now, pinned against config, and asserted to
  agree with the rows.
- **`share-class-switching-margin`.** The margin was pinned twice as a literal - `config.yaml` and
  `share_classes.CHALLENGE_MARGIN` - and **no test in the suite read it off the page**:
  `tests/test_share_classes.py` had never called `build_screener_overview`. Five checks do now,
  including the config-follows case (raise it to 35% and the sentence has to move), the fallback,
  and the `else` branch that withdraws the paragraph when the rule is switched off.
- **`bot-monthly-excess-figure`.** `test_the_trading_rule_sentence_follows_each_period_s_sign_and_t`
  matched the published percentage as `[0-9.]+%`, so the sign, the words and the t were pinned and
  **the figure itself was a wildcard in all three periods**. Each is derived from the ledger now,
  with the magnitude-plus-word pairing so a sign flip cannot leave the figure looking right.
- **`bot-coverage-by-year`.** The two survivorship coverage percentages - the point of the page's
  survivorship caveat - were read by nothing. Both are now the two years' means from the coverage
  measurement, and the sentence says "rising to", so the test asserts that too.

**One production change, and it is why the row was worth taking.** `build_screener_overview` fell
back to its own `0.20` literal for the switching margin while step 11 falls back to
`share_classes.CHALLENGE_MARGIN`. With `challenge_margin` present in `config.yaml` the two agree,
so nothing on the page was wrong today - but two literals for one number are free to drift, and
the page would then have stated a margin no run applied. The generator shares the constant now
(CLAUDE.md row 0.8: the page shows the engine's number and never its own re-derivation of it).

**4. A correction to what this audit counts as "caught."** Run with the row's own check deselected,
the whole suite **did** fail on three of these mutations - and **not one of the three failures was
a check of the number**:

- `test_overview_is_generated::test_committed_overview_is_what_the_generator_produces` - a **drift**
  tripwire, green the morning the 02:00 run rewrites `SCREENER_OVERVIEW.md`. The 2026-10-30
  correction, met for the third time.
- `test_claims_register::test_every_computed_sentence_on_the_methodology_page_has_a_verdict` - the
  **sentence sweep built the day before**. It keys a sentence by its text, so changing a figure
  inside a template changes the key and the sentence reads as unregistered. Useful - it forces a
  human to look - but it is satisfied by registering the new wording, so it detects *an edit*, not
  *a wrong number*.

Measured: mutation A gives `2 failed, 3589 passed`; C gives `1 failed, 3586 passed`; every failure
is in that class. **So a mutation can read as caught twice over and still ship. A sweep detects
edits and a drift tripwire detects staleness; neither is a check of arithmetic, and this audit no
longer counts them as one.** That is written into the plan, with the method note that both new
research-page checks read the **generated** page and not `rp.PAGE` - the first drafts read the
committed file, which would have made them drift checks wearing value-check names, the exact
defect rows 135-138 exist to fix.

### Evidence / research

- No new citations - this was a defect session and an audit session, and the mandate's third
  category applies throughout: **a failing test that now passes, and a measured number**. The
  measurements are `research/measurements/2026-10-31-skip-test-teeth.py` (5 mutations, 5 caught;
  `min` -> `max` blind to 3,563 tests before today),
  `...-open-row-mutations.py` (4 of 4) and `...-research-page-mutations.py` (4 of 4). Each
  restores the file it mutates in a `finally` and asserts it byte-for-byte.
- No methodology, weight, threshold or published number moved. The one production edit replaces a
  duplicated literal with the constant the run already uses; `rehearse_publish` confirms all four
  published artifacts are byte-identical.

### Methodology changed

- None. No `METHODOLOGY_CHANGELOG.md` entry is due: no weight, metric, threshold or published
  sentence changed, and the share-class margin the page prints is the same 20% it printed
  yesterday.

### Tried and rejected

- **Starting the `generate_dashboard.py` sentence sweep** (the plan's item 0, the last of the three
  generated surfaces). Rejected for today: its own list says to expect the biggest table of the
  three from 13,485 lines and to "decide up front whether to scope it with `within=` per generator
  function", and "the work is not the sweep, it is the verdicts". With the morning on a red
  baseline it could only have been half-done, and a half-registered surface is worse than an
  unregistered one - the sweep would pass with verdicts assigned by skimming. The four named
  `open:` rows were the finishable half of the same owner item.
- **Reading `rp.PAGE` in the two new research-page checks.** Written that way first, because every
  other test in the module does. Rejected once stated plainly: a check against a committed
  artifact is green until someone regenerates, so it would have been a drift tripwire with a value
  check's name - which is the defect the rows were opened for. Both read `_rebuilt()`.
- **A blanket ban on clock reads in tests.** Rejected: of the 11 sites, nine are legitimately
  day-independent (elapsed days, a two-minute agreement between two readings, a consensus year-end
  six months out), so a ban would force worse tests. The annotation is the proportionate form, and
  an unannotated new read fails the suite.
- **Re-measuring "blind to everything else" for the four research-page mutations across the whole
  suite.** Not run: ~15 minutes for a result already established in its exact form by mutations A
  and C above (the sweep fires on the rewording, the drift tripwire on the artifact). Stated as
  the measured general fact rather than re-measured, and the module-scope runs do show the new
  check as the only failure once the drift tripwire is deselected.

### Next

- **`_pbo_words` at its own bars, 0.45 and 0.55** - the cheapest remaining `open:` row and the one
  the plan has named twice: the overfitting section's words are probed at 0.21/0.5/0.73 and never
  at the two points that decide which word is printed. Then the three bank P/B-ROE prose rows,
  whose residual clause chooses a **word** on `bank_net_pb < 0` with nothing probing that boundary
  either, and `coverage-discount-worked-example` (five figures deep, read by none of them).
  `generate_dashboard.py`'s sweep still wants a session of its own and should not be started
  alongside anything else.
- **Monday 2026-11-02 is phase 0 of the universe-expansion plan** (`plan/universe-expansion.md`),
  the top open `OWNER_FOCUS.md` item, and **Friday 2026-11-06 is the Factor Lab's day** -
  experiment 18, unstarted, with `lab_data.js`'s first paint already measured and fixed on
  2026-10-26.

## 2026-10-31 (13:00 midday) - QUEUE. Nine more audit rows, and the morning's own two turned out not to have been closed

**Health (rule 8, all five):** last code session ran? **yes** - today's 06:00 weekend session,
`logs/nightly-2026-10-31_060001-weekend.log`, ends "shipped to main". Data loop published?
**yes** - the last weekday run, `logs/datarun-2026-10-30_020001.log`, "Data loop complete",
HEALTH: PASS, signals written (40 names, 8 gates, all ran); today is a Saturday, so no 02:00 run
is due. Evidence base at `1m` = **40 rows, newest 2026-09-29 (32 days ago, bound 40), 5
effective** - steady state, unchanged from this morning. Priority 0: `allow_auto_apply` **false**,
5 of 8 effective. Top open roadmap item: **0.13, the Factor Lab** (opened 2026-10-12, **age 19
days**) - not advanced; the next lab Friday is **2026-11-06** and no experiment is half-built.
(0.10 is older - opened 2026-10-07, **age 24 days** - and its remainder is Sharpe and Sortino at
weight 0, which CLAUDE.md says to do only with nothing better.)
**Tests:** before **3,594 passed / 1 skipped**; the runner's own baseline line agrees
(`BASELINE main is GREEN on arrival (3594 passed, 1 skipped ... 0:04:01)`), after **3,602 passed /
1 skipped** (+8, all new). Dry run exits 0. `scripts/rehearse_publish.py` **PASSED** (the diff
touches `claims.py`): 362 page tests, published claims 1,850 passed, all four published files
unchanged and restored byte-for-byte.
**Owner queue / rotation:** no proposal is waiting (`write_brief.open_proposals()` returns `[]`),
so `OWNER_FOCUS.md` item 2 sub-item 3 ("test the tests"), carrying on from the 06:00 entry's
**Next** exactly as the focus asks. Nothing deferred for a gate or a loop. `git stash list` shows
`auto-rescue 2026-10-28_060001`; `git stash show -p` prints nothing and
`--include-untracked --name-only` lists three gitignored data files (a context log and two price
caches), so there is **no tracked work in it** - left in place, as every session since has. No
owner-session marker.

**Second session today.** The 06:00 entry's Next named `_pbo_words` at its own bars, then the
three bank P/B-ROE prose rows and the coverage-discount worked example. All of those are done -
and so is something that entry did not know it had left behind: its own two rows.

### Did

**1. The two rows the 06:00 session closed were never closed in the register.** This audit's own
rule, met from the inside: *"Finishing a gap means registering the fix too. A new test that
`checked_by` does not name reads as a gap on the next pass and is outside the publish gate's
derivation."*

Commit `a306d32` wrote five real checks for `share-class-switching-margin` and extended one for
`clamp-bullet-second-printing`. It touches **no register file**: both sentences still read
`open:` in `OVERVIEW_SENTENCES`, `OVERVIEW_KNOWN_OPEN` still named both slugs, and the five new
`TestThePagePrintsTheMargin` checks were in no claim's `checked_by`. "`KNOWN_OPEN` drops twelve to
ten", written in that entry and in `OWNER_FOCUS.md`, was true of the **research** page's two rows
only - the 07:00 commit did move those.

**Measured rather than inferred**, with a harness that differs from the morning's in the one way
that matters: `research/measurements/2026-10-31-registered-row-mutations.py` runs **the claim's
own `checked_by`**, which is what `scripts/check_published_claims.py` derives and therefore the
only coverage that can stop a publish (the audit's method step 3). `--before` drops the nodes
added today.

- `--before`: mutations C and D - the margin typed into the generator instead of read from config,
  and the page's fallback drifting from `share_classes.CHALLENGE_MARGIN` - are **GAP, 6 of 6
  registered nodes passing**, while a test in the suite caught both.
- as committed now: **A, B, C, D all CAUGHT, 4 of 4.**

So the clamp row's gap was the **verdict label alone** (its check was already named by
`overview.clamps_and_floors`) and the margin row was a real `GAP-IN-REGISTER`. **"The row is
closed" is three facts, not one:** a check exists, the register names it, and the verdict says so.
That is a carry-forward rule in the plan now.

**2. The whole "If you had kept the best backtest" section - five rows (139-143).** Its stat cards
were pinned; its three prose paragraphs were not. They restate all three PBOs, add two
out-of-sample shares, the count of alternatives and the top winner's two era leads, and nothing
read any of them. `_pbo_words` was probed at 0.21/0.5/0.73 and **never at its own bars, 0.45 and
0.55** - a pair of probes that passes for any bars between 0.21 and 0.5 and between 0.5 and 0.73.

Four checks in `tests/test_overfitting.py`. Two read the figures off the **generated** page
against the committed ledger; two feed `overfitting_section` a reading of their own, which is the
only way to put a figure *at* a bar the committed ledger does not sit on, and the only way to
reach the ", where it fell behind." clause at its sign boundary. The three words are literals in
the test, not `rp._pbo_words`'s own output (audit row 59).

**8 mutations, 8 caught**, each by the test meant to see it
(`research/measurements/2026-10-31-pbo-prose-mutations.py`): both bars moved, an era's PBO copied
from another era, a decimal dropped, a share swapped for a different statistic, the count no
longer excluding the current weighting, the clause firing at exactly zero, and the two leads
reversed.

**The larger finding is not a row.** `tests/test_overfitting.py` was named by **no claim**, so all
eight of its checks - including the only reader of the stat cards and of the C(16,8) split count -
were **outside the data loop's publish gate**, whose membership derives from `checked_by`. Same
shape as `tests/test_engine_calibration.py`, found 2026-10-28 and flagged to the owner as the one
register finding worth their attention. `research.overfitting_reading` names them and the module
is in the gate now by derivation. **Two of the five rows carried a `checked:` verdict pointing at
that module**, which reads as coverage and was not.

**And the rest of the suite is blind to both classes.** Whole suite, today's four new checks
deselected, the page regenerated the way `python v2_experiments.py report` publishes it: the
0.55-bar mutation gives `1 failed, 3593 passed` and the configuration-count mutation `2 failed,
3592 passed`, and **every failure is a drift tripwire or the sentence sweep** - the 2026-10-30
correction, met for the fourth and fifth time.

**3. The methodology page's last four prose rows (144-147)** - the three bank P/B-ROE prose
figures and the coverage-discount worked example. **8 mutations, 8 caught, all 8 a GAP before**
(9 of 9 and 3 of 3 registered nodes passing;
`research/measurements/2026-10-31-overview-prose-mutations.py`).

**The bank residual clause is the blindest sentence this audit has met**, and the reason is worth
keeping: today's weights print the *neutral* wording, so the branch that states a measured
price-to-book tilt was on a page **no test had ever generated**. The sentence sweep found it
because `computed_sentences` reads the generator's f-strings and not its output - the one thing a
sweep does that a page reader cannot. It is reached now with weights of the test's own, at
`pb_ratio` 79 (net P/B -4.18 against a bar of 4.455, inside it) and 81 (-4.62 against 4.345, past
it), which brackets the 0.25 neutrality share to about +-0.013 - a move to 0.20 or 0.30 flips one
of the two - and on both sides of the `against` / `on` word, which turns on `bank_net_pb < 0`.

**Stated rather than papered over:** the `<=` in that bar is **not** probed at equality. At
`pb_ratio` 80 the two sides are 4.4 either way in decimal and `-17.6 + 7.7 + 5.5` is
`-4.400000000000002` in binary, so the inclusive boundary is unreachable from any weighting here.
The first draft asserted it and failed, which is how this was found; the test records the limit
beside the brackets.

**One correction to the audit's own record.** Row 147's note said the worked example's five
figures were "read by none of them".
`test_the_discount_example_is_a_coverage_a_ranked_stock_can_have` captures **seven** figures and
pins four - the loss, the deepest cut and the pair it is quoted at. Three were read and discarded:
the example's `k`, its denominator, and the coverage percentage between them, which was bound to a
regex group and **never asserted at all**. So the sentence could have quoted any of the eight
coverages the filter allows, and the discount beside it would have been recomputed from whichever
`k` it printed and agreed with it. A row's note is written before the work - re-read the assertion
before trusting it.

**The queue: nineteen open rows on 2026-10-29, six now.** `KNOWN_OPEN` ten -> five,
`OVERVIEW_KNOWN_OPEN` seven -> **one**.

### Evidence / research

- No new citations - an audit session, and the mandate's third category applies throughout:
  **a failing test that now passes, and a measured number.** The three scripts are
  `research/measurements/2026-10-31-registered-row-mutations.py` (4 mutations: 2 GAP before, 4
  CAUGHT after), `...-pbo-prose-mutations.py` (8 caught, 2 of them re-run against the whole suite)
  and `...-overview-prose-mutations.py` (8 caught, 8 GAP before, 1 equivalent mutant). Each
  restores the file it mutates in a `finally` and asserts it byte-for-byte.
- **One equivalent mutant, kept in the harness rather than dropped** (the plan's "Equivalent
  mutants"): halving the worked example's 0.05 margin cannot move it at n = 27, because the rule
  takes the largest `k` with `k / 27` below the bar and 0.775 x 27 = 20.9 selects the same k = 20
  as 0.75 x 27 = 20.3. The generated page is **byte-identical** (95,140 bytes, sha256
  `f7a46c07a6f3a6eb`, both ways), so nothing can see it and nothing should. Without the row the
  next pass picks the same mutation and reads the pass as a gap. Its fingerprint is **sha256, not
  `hash()`** - `str` hashing is salted per interpreter, and the first draft reported the page as
  CHANGED for that reason alone.
- No weight, metric, threshold, published sentence **or production line** changed. The diff is
  `claims.py`, four test modules, three measurement scripts,
  `plan/claims-mutation-audit.md`, `OWNER_FOCUS.md` and this log. `rehearse_publish` confirms all
  four published artifacts are byte-identical.
- One thing this session could not verify itself: **`node --check` is refused by the tool guard**
  here, so gate 3 was checked by file size and opening assignment in Python plus
  `rehearse_publish`'s byte-for-byte restore. The runner re-runs the real parse on both payload
  files before it merges, and no artifact in the diff is generated.

### Methodology changed

- None, and no `METHODOLOGY_CHANGELOG.md` entry is due: nothing a reader sees moved, and the one
  new claim records what the page already said.

### Tried and rejected

- **Taking the `generate_dashboard.py` sentence sweep instead.** Rejected for the reason the 06:00
  session gave, which still holds: the plan's item 0 calls it the biggest of the three surfaces,
  from 13,485 lines, and says "the work is not the sweep, it is the verdicts". Nine finishable rows
  that each close on the *value* of a figure are worth more than a half-registered surface, which
  is worse than an unregistered one.
- **Asserting the neutrality bar at equality** (`abs(net_pb) == 0.25 * net_ep`). Written first,
  because `pb_ratio` 80 gives 4.4 on both sides in decimal. It fails in binary and no weighting
  reaches it, so the test brackets the bar and **names** the gap - a test that claims a boundary it
  cannot reach is worse than one that says it cannot.
- **Deriving each prose word from `rp._pbo_words`.** That is a check importing the number it is
  checking (row 59): a moved bar would move the page and the expectation together. The three words
  are literals in the test and the bars are probed through the page.
- **Counting the three "caught" verdicts the morning's harness reported for rows 131-133 as
  coverage.** They were a drift tripwire and the sentence sweep, exactly as that session wrote -
  and re-running against the claim's `checked_by` showed two of them were gaps the publish gate
  could not see. The distinction is not academic: it is the difference between "a test somewhere
  fails" and "the loop that publishes the page refuses".

### Next

- **`pass-bar-printed` and `pass-bar-under-chart`, together** - the research page prints the same
  pass bar twice, nothing pins either copy and nothing pins that the two agree. That is the shape
  `clamp-bullet-second-printing` turned out to be, so expect it to go the same way. After that the
  two Size rows (the `short of` / `beyond` word needs probing **at** |t| = bar, the same shape as
  `_pbo_words`), then `coverage-sentence-metric-counts` (the other-financials count,
  `len(_w_gen) - 1`, is looked for by nothing) and `chart-axis-and-geometry`, which is the one
  needing geometry assertions rather than text matching. `generate_dashboard.py`'s sweep still
  wants a session of its own.
- **Monday 2026-11-02 is phase 0 of the universe-expansion plan** (`plan/universe-expansion.md`),
  the top open `OWNER_FOCUS.md` item, and **Friday 2026-11-06 is the Factor Lab's day** -
  experiment 18, unstarted.

## 2026-11-02 (owner-run, midday) - Unstalling the data loop: the growth-neutral claim, and two tests that read the wrong data

**Health (rule 8, all five):** last code session ran? **no** - `logs/nightly-2026-11-02_060000.log`
ends "SESSION DID NOT RUN: ... OAuth session expired and could not be refreshed", as did the
11-01 06:00 weekend and 11-01 13:00 sessions. `claude auth status` reports `loggedIn: false`;
`~/.claude/.credentials.json` was last written 2026-11-01 05:43. **Logging in is the owner's to
do** (credentials) - he said he will tonight; until then every scheduled code session fails the
same way within two seconds. Data loop published? **no** - `logs/datarun-2026-11-02_020000.log`:
HEALTH: PASS, both payloads parse, then "PUBLISHED CLAIMS: FAIL" (2 tests) and "Run discarded";
the live site and `signals.json` are still the 10-30 run. Evidence base at `1m` = **40 rows,
newest 2026-09-29 (34 days ago, bound 40), 5 effective**. Priority 0: `allow_auto_apply`
**false**, 5 of 8 effective. Top open roadmap item: **0.13, the Factor Lab** (opened 2026-10-12,
**age 21 days**) - not advanced; a stalled data loop outranks it.
**Tests:** before **1 failed** at the runner's baseline (`test_special_item_note::
test_the_published_payload_agrees_with_the_filings`, inherited); after **3,606 passed / 3 skipped**
(two of the skips are the new premise skips below, until tonight's payload carries the stamp).
Dry run exits 0, both payloads parse. `scripts/rehearse_publish.py` against the refused run
1957c294714c with this tree: **PASSED** (362 page tests, published claims, files restored).

### The three failures, and what each was

1. **`test_the_ranking_really_is_close_to_growth_neutral` - real.** The composite's Spearman with
   the Growth score was +0.058 against the page's "sits close to neutral" band of 0.05. Not one bad
   run: across the fifteen published runs since the 2026-10-13 review it drifted steadily from
   -0.009 to +0.049 (10-30), and the weight at which the ranking is neutral moved from 13.5 to
   about 10 (the review's own sweep script, rerun). The gate refused the run as designed. **The
   weight did not move** - re-fitting it to a point that moved four points in three weeks is a
   rule to design, not a weight to set. The paragraph was rewritten - the test's own named
   alternative: it now states the review's reading, the drift and the 11-02 reading, and points at
   **`research/2026-11-02-growth-exposure-drift.md`, a new OPEN proposal** (re-fit on a schedule,
   constrain the exposure as MSCI does, or state the lean). The old sentence is in
   `claims.FORBIDDEN`; the brief now carries a "Ranking's lean on growth" row every morning.
   CLAUDE.md and DECISIONS.md updated.
2. **`test_valuation_history::test_the_published_blocks_are_internally_consistent` - the test.**
   First run of a month, so October's month-end close is the run's own price and five stocks'
   "today" sat within half a basis point of their last month. The tolerance `now + 0.00005`, done
   in floating point, came out below the stored month (0.09985 + 0.00005 = 0.0998999...). Now
   compared in basis points with the stored precisions (+-0.5bp per month, +-0.05bp for today).
   0 of 743 blocks fail on the refused run, 0 of 694 on the published one; a one-month-off mutant
   is caught exactly as often as before (679 of 1,486 vs 681 - ties are the limit in both).
3. **`test_special_item_note::test_the_published_payload_agrees_with_the_filings` - the premise.**
   The data loop refreshed the weekly SEC cache at 02:16 and then refused its publish, so the
   committed 10-30 notes were compared with 10-Qs filed after them (11 notes "missing" - AON,
   BIIB, CARR...). The payload now records the cache it read (`sec_facts`: rows, newest filing
   date - `generate_dashboard._facts_vintage`), and this test and its twin in
   `test_capex_lease_note.py` compare only against that cache, skipping loudly otherwise. In the
   publish gate the payload and cache always match (verified on the rehearsal: both
   `{rows: 2428963, filed_through: 2026-10-30}`), so the check is never lost where it matters.
   Filtering the cache by filing date was tried first and does not reproduce the notes.

### Next

- **Owner: `claude auth login`** - nothing else restores the code sessions.
- Tonight's 02:00 run should publish (rehearsed against this morning's run). Check its log for
  the claims gate PASS and the brief's new growth row.
- **Monday research item:** the open proposal in `research/2026-11-02-growth-exposure-drift.md`.

## 2026-11-02 (owner-run, evening) - "make as many improvements as you can before the 2 am session"

**Health (rule 8, all five):** last code session ran? **no** - `logs/nightly-2026-11-02_130000-midday.log`:
SESSION DID NOT RUN (Failed to authenticate: OAuth session expired). **The owner signed in at
18:51** (`claude.cmd auth login` - PowerShell blocks the `.ps1` shim); verified the same evening by
the login check below and a one-turn `claude --print` from this folder (`READY`, 8.8 s). Data loop
published? **no** - `logs/datarun-2026-11-02_020000.log` (stopped deliberately; HEALTH: PASS; the
claims gate refused it, fixed at midday). Evidence base at `1m` = **40 rows, newest 2026-09-29 (34
days ago, bound 40), 5 effective**. Priority 0: `allow_auto_apply` **false**. Top open roadmap item:
**0.13, the Factor Lab** (opened 2026-10-12, **age 21 days**) - **advanced: experiment 18 shipped**.
(That line was printed by the new `scripts/health_numbers.py`.)
**Tests:** before **3,606 passed / 3 skipped**; after **3,647 passed / 3 skipped** (gate 1 on the
merged tree, 3:03 - it was 4:36 at the start of the evening). Every batch: all four gates; the
batches touching `claims.py` or the overview generator also `rehearse_publish.py` against the
refused 11-02 run - **PASSED**. Tags `good/2026-11-02-owner-2` .. `-7`.
**Owner queue / rotation:** the owner's request. Universe expansion phase 0 (the top `OWNER_FOCUS`
item, due today; the scheduled Monday could not run) done here.

### Did

1. **The login outage, made predictable.** The CLI keeps `refreshTokenExpiresAt` in
   `~/.claude/.credentials.json`: after tonight's sign-in it reads **2026-12-02 10:28** - about 30
   days, not extended by the daily refreshes (the sessions ran daily up to the 11-01 lapse). New
   `scripts/check_cli_login.py`, run by the 02:00 data loop as a note (never a gate, always exit 0),
   reads that one field - **no token, no subprocess**: `claude auth status` may rotate tokens, and
   the suite runs this from inside a live session. The brief opens with "log in" once it has lapsed,
   warns a week ahead with the date, and shows the days left in its glance table. The code runner's
   log now names the fix when a session dies on the login. `tests/test_cli_login_check.py` (14).
2. **The watchdog sees a loop that fires but does nothing.** `scripts/check_loop_health.py` gains
   "Session ran" (a code-session brief without "SESSION DID NOT RUN") and "Data published" (a
   `data: screener run` commit), same commits, same two-consecutive-weekday rule, each with its own
   advice in the GitHub issue. Replayed over 2026-09-24..11-02: warns on the three real days (10-06,
   10-28, 11-02), never alarms. The four sessions lost this weekend left every old row green. The
   real-history test asserts only the firing rows - a fortnight's streak ends with a heartbeat
   written *after* the gates, so asserting the job rows would make the fix fail its own gate.
3. **`scripts/health_numbers.py`** prints rule 8's line from the brief's own functions (the last
   *finished* code session, computed ages); `prompts/nightly.md` section 5 points at it.
4. **Universe expansion phase 0** - `research/2026-11-02-universe-expansion-phase0.md` (a delegated
   writer; code claims spot-checked here: `SECTOR_MIN_PEERS = 10`, `min_market_cap: 2e9`, 27/24
   weighted metrics, coverage minimum 16). S&P 1500 first; phase-4 floor at the NYSE 20th
   percentile from Ken French's June file ($1,224.13M); percentiles within (sector, size band).
   **Proposal OPEN for a session that did not write it.** Measured here: the no-analyst case scores
   Revisions on days-to-cover alone for **one** S&P 500 stock today (Loews), six more partly.
5. **Factor Lab experiment 18** (delegated, isolated worktree, merged): Chan, Jegadeesh & Lakonishok
   (1996) on OSAP `REV6` - **3 of 3 within** tolerances committed first (`4e1f6a7`, reviewed here:
   paper's Table 5 figures, tolerance from its own Table 9 standard error, direction predicted).
   Paper years 8.65% over six months vs 7.7%; since 1993 0.90%/month equal-weighted (t 4.54), 0.50%
   by market value (t 1.79). Lab: 18 experiments, 148 of 220 within. **Second read pending.** The
   OSAP value-weighted parquet was copied into the main checkout's `data/lab/osap/`.
6. **Claims audit: both open lists empty.** Research page: the pass bar printed twice
   (`research.pass_bar`, rows 148-153), the Size sentence (`research.size_sentence`, 154-162), the
   chart's geometry read off the page alone (`research.chart_geometry`, 163-173). Methodology page:
   the coverage counts against the engine's own denominators (`overview.coverage_counts`, 174-177).
   30 mutations, each GAP before and caught now; three were missed by the first version of a test
   and fixed by sharpening it (rows 161, 172, 173). Harnesses in `research/measurements/2026-11-02-*`.
7. **Filing cross-check range measured** (`research/2026-10-14-yahoo-vs-sec.md` s.5): 11 runs, total
   assets 0, equity 2, operating cash flow 1-3 over 10% vs a bar of 25. Promotion pre-registered: on
   or after 2026-11-16, if those three stayed <= 5, they become a gate at 25.
8. **Speed:** `overfitting.cscv` ranked with `pd.Series.rank` 12,870 times per check; a direct count
   gives the same average rank (0 mismatches in 20,000 random tie-heavy vectors; `check()` output
   identical) and `t_critical` is cached. Research page build 3.5 s -> 0.6 s; `v2_experiments.py
   report` regenerates `research.html` and `summary.json` byte-identical.
9. **Brief:** each open proposal listed by its sentence, not its first wrapped line.

### Evidence / research
- Phase 0 note: Fama & French (2008), Hou, Xue & Zhang (2020), Jensen, Kelly & Pedersen (2023),
  Hong, Lim & Stein (2000); S&P DJI, FTSE Russell, MSCI and Barra methodology documents (the note).
- Experiment 18: Chan, Jegadeesh & Lakonishok (1996), NBER w5375 Table 5 (the note).

### Methodology changed
- None. No number, rank, weight or published sentence changed (the lab page and its methodology
  paragraph gained experiment 18; the dashboard payload is byte-identical).

### Tried and rejected
- Calling `claude auth status` from the data loop and the suite - shipped in the first batch,
  replaced within the evening once the stored expiry was found: a status call may rotate the tokens
  of the very session running the suite.
- Making ROIC's equation exact tonight: it needs the intermediate values published from where the
  metric is computed, which only a full run can verify - not before the 02:00 run.
- EXPE's `sustainable_growth` at its cap on 4.2% equity: already found and surfaced by
  `research/2026-10-15-top-five-audit.md` and the brief's "worth a look" row.

### Next
- After 02:00: the run should publish (rehearsed). Its log should carry "CLI login: OK - valid until
  2026-12-02 10:28", the brief a "Code-session login" row, and the payload a `sec_facts` stamp.
- **Decide the phase 0 proposal** (not this session) and the growth-exposure one.
- **Second-read experiment 18** (Friday 2026-11-06 or any session that did not build it).
- **The dashboard's own claims queue:** `generate_dashboard.py`'s sweep merged tonight (delegated,
  isolated worktree; `tests/js_sweep.py` reads the JavaScript the generator emits, since every
  number in the page's prose is formatted there). **84 computed sentences: 7 claim, 77 open** -
  almost all open because the claims' checks pin the engine's or payload's number and no
  registered test runs the JavaScript that prints it. No false sentence found. Closed: the
  Concentration block in My Holdings (`dashboard.holdings_concentration`; its 34-test module had
  been outside the publish gate, which now runs it - 71 modules), rows 178-188, 11 of 11 caught.
  The 77-row queue, grouped in three, is in `plan/claims-mutation-audit.md`'s newest section:
  `DASHBOARD_KNOWN_OPEN` only shrinks from here.

## 2026-11-02 (owner-run, night, round 2) - "Keep finding improvements!"

**Health (rule 8):** as the evening entry above (no scheduled session or data run since); the line
now prints from `scripts/health_numbers.py` and the runner hands it to every session in its prompt.
**Tests:** 3,663 passed / 3 skipped at the last gate run. Tags `good/2026-11-02-owner-9`, `-10`.

### Did

1. **Gate 5: the runner rehearses the next publish** (`scripts/rehearse_publish.py --changed-since
   $BaseSha`) whenever a branch changed more than prose, before gate 4; a failure keeps the work off
   `main`; "no full run to rehearse from" (exit 3) only warns. **It earned its keep within the
   hour**: the stale-flag fix below passed gates 1-4 locally while the rehearsal failed
   (`test_the_baseline_weighting_is_hashed_separately_from_the_transformed_one`) - a config edit
   stales the research ledger. CLAUDE.md now says so up front (`config.yaml` and `factor_engine.py`
   are in `RESULT_PATHS`; budget ~20 min for `rerun_experiments.py --jobs 4`).
2. **The stale-filing flag waits for the SEC deadline** (display only; `config.yaml` 120 -> 165,
   METHODOLOGY_CHANGELOG 2026-11-02 night). At 120 it told readers **320 of 499** stocks had stale
   filings on 10-30 while their Q3 10-Qs were not yet due; every bar from 130 to 210 flags the same
   **21** genuinely a quarter or more behind. The 32 research rows were rerun on the new config hash
   - every difference is a provenance field; `summary.json` and `research.html` byte-identical.
3. **Experiment 18 second-read** by a different reader (isolated worktree): every paper figure
   (NBER w5375, rendered pages) and every measured figure reproduced without importing
   `factor_lab`; the published article's figures confirmed via Jegadeesh & Titman (2001); the
   pre-registration order verified; three wording fixes to the note, no figure changed.
4. **Weight question 3 researched** - `research/2026-11-02-investment-inside-quality.md`: keep
   Investment standalone and state the practitioner convention on the page (**proposal OPEN**); a
   weight-preserving fold leaves the top 25 unchanged but is not neutral where data is missing,
   under the panic rule, in the profiles and in the trap flags.
5. **Two load-sensitive browser tests in the publish gate** now measure the page, not the machine
   (best of five clicks; the reload gets `_open`'s allowance) - one slow sample at 02:00 would have
   discarded a night's run. No budget loosened.
6. **The brief's "Coming up"** lists dated commitments (2026-11-09 first one-month context
   reading, 2026-11-16 filing-check promotion, 2027-02-11 backtest) from `COMING_UP`, each tied by a
   test to the file that commits to it.
7. **Checked and left alone:** Loews (no analyst coverage in any published run; its Revisions move
   is real short-interest change), Yahoo rate-limiting (40-45 a run since 10-27, all recovered).

### Next
- The OPEN proposals now waiting for a session that did not write them: growth exposure, universe
  phase 0, Investment-inside-Quality.
- After 02:00: the stale flag should fire for about 21 stocks, not about 320.
- **A FALSE sentence on the live page, measured and queued (item 1 of the claims queue):** the
  expected-move card says the move is the at-the-money straddle divided by *this run's* close, but
  the 20:00 option refresh (`options_cache.refresh` -> `_fetch_one` -> `context_signals.options_context`)
  is handed the previous run's `price_latest` and picks the at-the-money strike against it. On the
  10-30 payload 273 of 359 divisors differ from the printed price by >1%, 16 by >5%; NOW's strike is
  150 against $109.67, so its +/-27.3% is mostly the put's intrinsic value.
  `research/measurements/2026-11-02-expected-move-divisor.py`. The fix is in the refresh - price
  the straddle against the underlying's price at quote time - and needs a live option pull to verify,
  so it was not attempted two hours before a data run. Rows `expected-move-price` and
  `expected-move-figure` stay open, marked FALSE.
- **The dashboard claims sweep, round 2 (merged):** 37 of the 77 open rows closed (rows 189-271,
  83 of 83 mutations caught; 82 of them invisible to every check before tonight), via the new
  `tests/test_dashboard_printed_figures.py`, which runs the page's own JavaScript for every stock;
  `DASHBOARD_KNOWN_OPEN` 77 -> 40; the publish gate 71 -> 73 modules. One generator fix with no
  text change today: an unscored category's 'its N% was shared out' now prints this run's weight
  (it would have printed 13% for a panic-state Momentum scored at 9.1%).

## 2026-11-03 - PRODUCT, THROUGH THE NUMBERS. Open the live dashboard as a user would and ask whether it answers what should I look at / should I buy this / should I sell what I hold / how much. The visual design is finished (owner, 2026-10-07), so presentation is not this session's job. When the page cannot answer something the gap is almost always data or methodology: a number the run does not keep, a question no metric answers. Fix that. Check two or three drilldowns still reproduce the engine. A broken or false surface is a defect; one that merely could look different is not.

**Health (rule 8, all five):** last code session ran? **no** - `logs/nightly-2026-11-02_130000-midday.log`: SESSION DID NOT RUN (session reported an error - Failed to authenticate: OAuth session expired and could not be refreshed). Data loop published? yes - `logs/datarun-2026-11-03_020000.log` (completed; HEALTH: PASS). Evidence base at `1m`: 5 of 8 needed at the 1m horizon (43 rows, but overlapping windows are not independent; 110 rows across all horizons), newest 1m run date 2026-10-02 (32 days ago, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab** (age **22 days**); also open: 0.10 27d; 0.14 20d.
**Tests:** before **1 failed / 3,721 passed, 1 skipped**; after **3,766 passed, 1 skipped**. Gate 5 rehearsal PASSED.
**Owner queue / rotation:** took the red baseline, which turned out to be a false figure on the live page - a defect, and ahead of the focus by the prompt's own rule. **Deferred: all three OPEN proposals** (growth-exposure drift, Investment-inside-Quality, universe phase 0, each opened 2026-11-02, age 1 day). Each needs a measurement of its own and the day went on the live defect; the next session should take one. The scheduled 13:00 slot and tomorrow are both eligible readers - none of the three was written by a scheduled session.

### Did

**The red baseline was a false number on the live site, and fixing it properly meant fixing the family.**

1. **What was red.** `tests/test_every_panel_reads_clean.py` failed on the payload the 02:00 run
   published: NVR's analyst-target card read **"$6262.00 -0.0%"** - a mean target of \$6262.00
   against a \$6265.10 close, so 0.05% below, printed as a signed zero. That test exists because
   the same family of defect hit twice in two days in October, and it is deliberately **not** in
   the data loop's publish gate, so it goes red for the next code session instead of discarding a
   day of evidence. It did its job.

2. **The family, not the instance.** Three of these have now reached the live public site and each
   was fixed inside the one formatter it was found in - BXP's unsigned "0.0%" for a close 0.03%
   *below* its 200-day average (2026-10-28, the data loop refused that run), AMD's and INTC's
   signed "-0.0%" (2026-10-29, fixed in `cPlain` alone), and NVR this morning. So the rule is one
   function now - `pctNearZero` / `pctPlain` in `generate_dashboard.py` - and **fourteen call
   sites** use it: the analyst-target card, the snapshot and peer tables, the metric workings
   (`fmtMetric`, `fmtInput`, the earnings-surprise and ROE tables), the compare table's upside
   column, `cPct`, `cPlain`, `cPp`, the share-class price gap, the 10y-3m curve in both places it
   is printed, and the track chart's axis ticks and end labels.
   **The rule:** a value that is **not** zero takes one more decimal rather than printing as a
   zero; if it is still zero there it is zero to the reader and prints **unsigned**, because a
   sign on a zero is the glitch and an unsigned zero is true to the precision shown. Both halves
   are the project's own precedent (the case-against trend line, 2026-10-28; `cPlain`, 10-29).

3. **Measured, on today's published payload: 11 figures on the live page were wrong**, out of
   29,247 percentage lines across all 499 drilldowns, and **no other line moved**
   (`research/measurements/2026-11-03-percent-lines.py`, run before and after and diffed):

   | | before | after |
   |---|---|---|
   | NVR, analyst target | `$6262.00 -0.0%` | `$6262.00 -0.05%` |
   | ESS, vs 200-day average | `0.0%` ("Mixed trend 0.0% vs 200-day") | `−0.03%` |
   | VTR | `0.0%` | `−0.01%` |
   | ABNB / AMT / PTC, 5 days | `0.0%` | `+0.04%` / `+0.01%` / `+0.03%` |
   | EQT / TECH, 5 days | `0.0%` | `−0.02%` / `−0.03%` |
   | BKR, rate sensitivity | `0.00%` | `+0.003%` |
   | MRSH, analyst target | `+0.0%` | `0.0%` |

   **Eight of the eleven are the BXP direction** - an unsigned zero stating an equality that is
   not true - and the all-panel sweep **cannot see those**: its pattern looks for a *signed* zero.
   ESS is the sharpest: its close is 0.03% below its 200-day average and the page said the gap was
   zero, which is the exact statement a data run was refused over six days ago, still live in
   `cPct` because the fix went into `cPlain` beside it.

4. **The tripwire, so there is no fourth.** `tests/test_percent_near_zero.py`, **44 tests**: the
   rule at its boundaries (including the three live figures above), every formatter's own output,
   and a scan of the **generated** page - every line that builds a percentage must use the rule or
   be named in `EXEMPT` with the reason a zero cannot mislead there. **34 exempt** (bar and SVG
   geometry, shares printed beside the count they are taken from, category and metric weights, two
   macro levels), **0 unexplained**. A floor test confirms the tripwire fires on a formatter added
   without the rule, so the exemption list cannot pass vacuously.

5. **One real coverage regression, caught by the suite and worth recording.** Routing the
   share-class sentence through the rule made it **stop being a computed sentence to
   `tests/js_sweep.py`** - the register had a verdict for it, and after the change the sweep found
   no sentence there at all. `pctNearZero`/`pctPlain` are number helpers now, and `FORMATTER_SHAPE`
   covers `pct[A-Z]`, which is the part that generalises: the sweep's own
   "every formatter is classified" test could not see the gap, because it only looks at names
   shaped `fmt*`/`cX`/`mkX`. A new helper with a new prefix was invisible by construction.

6. **The rate card's check now asserts the rule instead of tolerating it.**
   `test_every_rate_card_...` pinned exactly two decimals, so BKR's corrected `+0.003%` failed it.
   `_check_near_zero` bounds the error by the decimals actually printed, and adds two assertions
   that were in no test anywhere: a figure that prints as zero carries **no sign**, and a value
   that is not zero does **not** print as a zero while a decimal would show it.

7. **Looked at the page, 1440 and 375** (`research/measurements/2026-11-03-shot-fixed-cards.py`;
   `logs/design/2026-11-03-*.png`). NVR's Avg Target card reads `$6262.00 -0.05%` in the
   down-colour at both widths, same card, same line-up, nothing reflowed - the extra decimal costs
   one character. ESS's Before-you-decide trend row reads `vs 200-day average −0.03%` beside
   `$265.65, rising over the last month`, where it read `0.0%` this morning; the rows are
   identical otherwise at both widths.

**Gates:** 1 green (3,766 / 1 skipped, no new failures); 2 `--dry-run` passed; 3 artifacts intact
and the payload byte-identical to what the 02:00 run published (`scripts/rebuild_page.py` rebuilt
`index.html`/`dashboard.html` only); 5 `rehearse_publish.py` **PASSED** - page tests 366 passed,
check_published_claims 2,025 passed, published files restored byte for byte.

### Evidence / research

- **A documented failure, demonstrated and measured** - the category the mandate names. The
  defect is in the published payload, the count is 11 of 29,247 printed percentages, and the
  before/after diff is reproducible from the committed measurement script.
- The rule itself is this project's own precedent rather than an invention: the two decimals
  under 0.1% come from the case-against trend line (2026-10-28), the unsigned fallback from
  `cPlain` (2026-10-29). This session's contribution is that there is now **one** copy of it.

### Methodology changed

- **None.** Every change is display-only: no weight, metric, threshold, score or ranking moved,
  and the payload is byte-identical. `METHODOLOGY_CHANGELOG.md` is deliberately untouched.

### Tried and rejected

- **Extending the all-panel sweep's `BROKEN` pattern to catch an unsigned "0.0%"** - the obvious
  first idea, and it cannot work: a true zero legitimately prints "0.0%", so the pattern would
  fire on correct sentences. The defect is only visible against the *value*, which is why the fix
  had to be in the formatter and the check had to be a source-level tripwire plus per-formatter
  boundary tests.
- **Routing the case-against trend line through the shared rule.** It already carries the rule
  inline and its sentence template is pinned in `claims.py`; changing its text would have moved a
  registered claim for no gain. It is in `EXEMPT` with that reason, printed where the next session
  reads it.
- **Touching bar widths, SVG geometry, and shares printed beside their own count** (34 lines).
  A 0.4%-wide bar and a trap rate shown as "0 of 12" are not the defect; each is listed with the
  reason, so the decision is re-readable rather than silent.


### Did, second half - the open proposal decided (added after the branch was first pushed)

With the defect shipped and the gates green, took the highest-leverage thing a scheduled session
can do: **decided one of the three OPEN proposals**, as a reader who did not write it.
`research/2026-11-02-growth-exposure-drift.md` - **DECIDED: option 3 accepted, options 1 and 2
refused as responses to this evidence, and the premise all three rest on does not survive the
longer record.** The reasoning is the note's new section 6, not only here.

8. **The question the note set aside is the one that decides it.** Section 3 says in writing:
   "Five changelog entries changed how scores are computed between 10-14 and 10-21, mostly for
   banks; they may account for part of it. **This note does not separate the two.**" Separating
   them is all this review did.

9. **It is not the banks**
   (`research/measurements/2026-11-03-growth-drift-decomposition.py`, every committed payload).
   With Financials removed - the cohort those entries rescored - the rise over the same window is
   **larger, +0.0949 against +0.0814**. And it continues through 10-23..10-30, five runs sharing
   one `kpis.methodology_version`, so no scoring rule changed underneath them.

10. **It is not a drift either, and that is the finding.**
    `research/measurements/2026-11-03-neutral-weight-history.py` runs the note's **own** sweep
    script, unmodified, over all 60 published payloads rather than the three it used. The Growth
    weight at which a run would be exactly growth-neutral:

    | period | runs | neutral weight |
    |---|---|---|
    | 2026-08-12 .. 2026-10-09 **morning** | 40 | **8.8 - 12.0**, mean 9.9 |
    | 2026-10-09 **evening** | 1 | **13.6** |
    | 2026-10-12 .. 2026-10-16 | 5 | 13.2 - 13.7 |
    | 2026-10-19 .. 2026-10-30 | 8 | 12.9 down to 10.2 |
    | 2026-11-03 | 1 | **8.8** |

    The step is **one run wide**: 10.2 on the 2026-10-09 morning run, 13.6 on the evening run the
    same day. Between them sit the five `METHODOLOGY_CHANGELOG.md` entries of 2026-10-09 -
    `operating_leverage`, `revenue_growth`, Piotroski 3/8/9, EBITDA, `forward_eps_growth`, four of
    the five inside Growth or Quality. Growth against Valuation goes -0.322 -> -0.443 across the
    same pair and is -0.359 today. **So 13% was never the neutral point.** It was neutral for about
    six runs, beginning four days before the review that measured it; the "drift" is that step
    decaying back to the level that held for the two months before.

11. **The decisions.** *Option 1, re-fit on the review cadence - **REFUSED**:* it would have set
    Growth to ~13.5 in mid-October, ~11.4 in late October and ~8.8 today, **three moves in a
    month, every one following the decay of this screener's own edit**. A weight that follows a
    correlation is not a weight, and this one would have been following an artifact. *Option 2,
    neutralise the composite after it is built (MSCI's instrument) - **REFUSED as a response to
    this evidence**, explicitly not on its merits:* at the configured weight the exposure is
    stationary in +0.02..+0.08 across the 40 pre-change runs and has returned into that band, so
    there is nothing persistent to constrain, and the step touches row 0.8 (the page must
    reproduce it). It stays open as a design question about the instrument. *Option 3, state the
    lean and leave the weight - **ACCEPTED***, which is what shipped on 11-02, so this ratifies
    rather than builds.

12. **One page correction it forced.** The methodology page said the balance "moves as the scores
    Growth offsets move **with the market**". The record does not support that attribution. The
    paragraph now gives the measured range, the one-run step, and what caused it, and says the
    weight is not re-fitted run by run **because** a re-fit would have chased that decay.
    Generated (`build_screener_overview`), so `SCREENER_OVERVIEW.md` was regenerated from the
    generator and `index.html`/`dashboard.html` rebuilt - never hand-edited (rule 10).
    **Read back out of the Methodology dialog as a reader opens it**, at 1440 and 375
    (`research/measurements/2026-11-03-shot-methodology-growth.py`): the paragraph renders in full
    at both widths, figures and both script paths intact, nothing clipped.

13. **Tests, not prose.** `tests/test_growth_category.py` (17): every dated figure the page gives
    is asserted against the table in the note that measured it; "with the market" must **not**
    appear; all three option verdicts must be in the note; and
    `test_the_neutral_weight_history_runs_and_reports_a_weight_per_run` reruns the sweep over the
    whole record and fails unless the 2026-10-09 evening run is above 11 and the morning run below
    it - so the claim the page makes about 40 runs is re-measured by the suite, not quoted from
    this log.

**What this does not do:** it does not reopen Growth's weight. The long-run neutral level is about
10 and the refused 2026-10-12 proposal asked for exactly 10, which is recorded in section 6.4 and
deliberately not acted on - fitting a weight to a correlation is what option 1 was just refused
for, and the 2026-10-13 refusal rested on what growth *earns*, which this measurement says nothing
about. No weight, metric or threshold changed today.

**Gates, re-run after this half:** 1 green (**3,769 passed, 1 skipped**); 5 rehearsal **PASSED**
again after the page change (page tests and `check_published_claims` both pass on the regenerated
page; published files restored byte for byte).

### Next (revised)

- **Two proposals still open**, both from 2026-11-02 and neither written by a scheduled session:
  Investment-inside-Quality (`research/2026-11-02-investment-inside-quality.md`) and universe
  phase 0 (`research/2026-11-02-universe-expansion-phase0.md`). The 13:00 slot can take one.
- **The expected-move divisor is still FALSE on the live page** - unchanged from the first half of
  this entry; it needs a session that can make a live option pull.
- **One habit worth adding to the changelog's measurement routine**, from section 6.4: when a
  metric inside Growth or Quality changes, record the composite's correlation with each category
  score before and after. The 2026-10-09 evening entry measured rank and score changes carefully
  and the exposure shift it caused stayed invisible for three weeks.

### Next (written in the first half, superseded by the section above)

- **Decide one of the three OPEN proposals** - growth-exposure drift
  (`research/2026-11-02-growth-exposure-drift.md`), Investment-inside-Quality, universe phase 0.
  All three were written by owner-run sessions on 2026-11-02, so any scheduled session is an
  eligible second reader, and the brief lists them with their ages.
- **The expected-move divisor is still FALSE on the live page** (queued by the 2026-11-02 night
  session as item 1): the card says the move is the at-the-money straddle over *this run's* close,
  but the 20:00 refresh prices the straddle against the previous run's `price_latest` - 273 of 359
  divisors differ by >1% on the 10-30 payload. It needs a live option pull to verify, so it wants
  a session that can run one, not a morning slot. Rows `expected-move-price` and
  `expected-move-figure` stay open, marked FALSE.
- Worth one measurement some morning: this session's defect and that one are the same shape - a
  figure whose *inputs* the page cannot check. The near-zero family is now closed by a tripwire;
  nothing yet holds a printed figure to the price it was actually computed against.

## 2026-11-03 (13:00 midday) - QUEUE. A second session today - more of the work, not a different kind. Read today's earlier NIGHTLY_LOG.md entry first and carry on from its Next; if it finished, take the next open item in OWNER_FOCUS.md, then the top of CLAUDE.md "Current priorities", then the queues in plan/factor-lab.md, plan/context-layer.md and plan/backtest-v2.md. Finish one item end to end, to the same evidence standard as any other session. Do not repeat or reverse a decision an earlier session made today without new evidence, and write down why if you do.

**Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-03_080346.log`
(completed). Data loop published? yes - `logs/datarun-2026-11-03_020000.log` (completed;
HEALTH: PASS). Evidence base at `1m`: 5 of 8 needed at the 1m horizon (43 rows, but overlapping
windows are not independent; 110 rows across all horizons), newest 1m run date 2026-10-02 (32 days
ago, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor
Lab** (age **22 days**); also open: 0.10 27d; 0.14 20d.

**The line above is not the one the runner handed me, and that is this session's second finding.**
The prompt's copy said *"last code session ran? **no** - `logs/nightly-2026-11-03_080346.log`:
SESSION DID NOT RUN ()"*. That log ends `=== Run complete: shipped to main ===`, merged six commits
and tagged `good/2026-11-03`. See item 2.

**Tests:** before **3,769 passed, 1 skipped** (green baseline, nothing inherited); after
**3,788 passed, 1 skipped** - 19 new (15 pinning the pre-registrations, 4 on the health line).
Gate 5 rehearsal PASSED (page tests 366 passed, check_published_claims 2,028 passed, published
files restored byte for byte).
**Owner queue / rotation:** carried on from the 06:00 session's **Next**, which asked a later
session to decide one of the three OPEN proposals. Took the **universe-expansion phase 0**
proposal, because it is the next step on the top item under `OWNER_FOCUS.md` **Open** (2026-10-28,
beyond the S&P 500) - it is now **decided and closed**, and `plan/universe-expansion.md`'s next
stage is phase 1. **Deferred: the Investment-inside-Quality proposal** (`research/2026-11-02-investment-inside-quality.md`,
opened 2026-11-02, age 1 day), still the only one on the brief's "Waiting for a decision" list. It
needs a measurement of its own - its section 4 lists four places a weight-preserving fold is not
neutral - and one proposal decided properly is worth more than two skimmed. Any session from the
06:00 slot tomorrow on is an eligible reader; neither of today's sessions wrote it.

### Did

**1. Decided the universe-expansion phase-0 proposal: all three pre-registrations ADOPTED, two of
them corrected by measurements the note did not make.**

The note (`research/2026-11-02-universe-expansion-phase0.md`, owner-run, 2026-11-02) is the
best-sourced thing in the queue and the review did not find it wrong. What it found is that two of
its arguments rest on one month's data and one of its instructions contradicts itself. Record:
note **section 9**; the operative decisions are now in `plan/universe-expansion.md`, pinned by
`tests/test_universe_expansion_preregistration.py` (15 tests). Three measurement scripts are
committed so every number can be re-derived.

- **PR2, the phase-4 floor: the headline number is exact.** Ken French's `ME_Breakpoints` was
  downloaded again (CRSP vintage 202609, 1,210 monthly rows) and the June 2026 row read
  independently: **n = 1,079 NYSE stocks, 20th percentile $1,224.13M** - both match the note to the
  digit, as do all four of its "recent Junes for scale".
- **But two of the five reasons given for it are struck.** Over 2015-2026 the line moved **+112.1%
  (2021), -34.4% (2022) and +40.7% this June**, so "it moves with the market, so it never needs
  re-tuning" is true of the *rule* and misleading about the *line*; and its agreement with S&P's own
  $1.2B small-cap entry floor is **this June's coincidence** (June 2025: $870M; June 2020: $484M -
  40% of it). Adopted on the one durable reason: it is the breakpoint Fama & French (2008),
  Hou/Xue/Zhang (2020) and Jensen/Kelly/Pedersen (2023) all draw, so their findings apply above it.
  **The consequence the note did not draw is now in phase 4:** the universe churns at its bottom
  edge every July, so the bot-style continuity gates must be re-derived **from a replay that spans
  a July**, never an average month.
- **PR3, percentiles within `(Sector, size_band)`: the exit test is a measurement now, not an
  argument.** The note called the S&P 500's byte-identity "met for free ... by construction". Run:
  `compute_sector_percentiles` against a transcription with exactly the two proposed changes, one
  constant band, on run `3b24b8eb8c2a` - **46 percentile columns x 502 rows = 23,092 values, 0
  differ.** With a floor test, because an identity check that cannot fail proves nothing: a genuine
  two-band split moves **20,151 of them, by up to 68.93 percentile points**.
- **And the fallback's cost is a number the note did not have.** Its one example ("the S&P 400 has 6
  Communication Services members") counts *members*; the code counts **valid values for one metric**
  (`valid = grp[col].dropna()`), so the unit is a (sector, band, metric) cell. Counting only cells
  where the metric can reach a composite - the four bank-only metrics (`equity_ratio`, `pb_ratio`,
  `roa`, `roe`, availability 9.2%) carry no weight outside Financials, so their fallbacks change no
  score - **today 1 of 301 weight-bearing cells falls back** (`beneish_m_score` in Financials), and
  **27 of 903 after the split, 35 with availability applied**. Every guaranteed one is in **S&P 400
  Communication Services (6 members)**, the only sector-band group under ten members anywhere in the
  S&P 1500. **The thin group is in the middle band**, so phasing the 600 in last does not defer it.
- **One self-contradiction in the note, resolved by measuring.** Its section 7 says both "keep 2e9
  for the S&P 500" *and* "set 0 for S&P index bands" - and the S&P 500 *is* an S&P index band, so it
  gives two answers for the one band whose byte-identity is the phase-1 exit test. Measured on the
  published raw fetch: **503 members, 0 below $2B, smallest MOS at $6.050B**, p1 $8.796B - 3x
  headroom. So one rule, `min_market_cap: 0` for every S&P index band, is byte-identical today and
  needs no special case. Reopening condition written down (a member within 2x of the floor).
- **Five of the note's Fama & French figures re-read from the paper, from a different copy**
  (`data/papers/ff-2008-dissecting-ivey.pdf`, the working-paper version, against the note's
  published PDF) - all five exact. One wording note, no figure affected: **FF's own term is "tiny",
  not "microcap"**.
- **Section 6's unresolved figure stays unresolved, but is now known not to be what the note
  guessed.** The note offers "probably a CRSP vintage difference" for FF's $2.3B end-June 2005 NYSE
  median against the file's $1.79B. A vintage change moves both percentiles and it does not: the
  **20th** reads $604.19M against FF's published $610M (**-1.0%**) while the 50th is **-22.3%** off,
  and the file's 50th runs smoothly $1,746M -> $1,786M -> $1,865M -> $1,896M -> $2,062M across
  2005-06, so it is not a one-month artefact. It touches no pre-registration: only the 20th
  percentile is used, and it agrees with FF's own published value to 1.0%.
- **Left open on purpose, not deferred:** section 4's choice of coverage rule (a)/(b)/(c) for a
  stock with no analysts. It is a **present-day** methodology change - one S&P 500 stock today
  (Loews, whose whole Revisions category rests on days-to-cover) and six more partly - so it gets
  its own changelog entry on its own merits, not as a side effect of phase 3.

**The guard, and the audit of the guard.** The decisions govern universes that do not exist yet,
which is exactly the condition under which a number drifts unnoticed: nothing in the pipeline reads
them, so nothing would fail if one were quietly re-fitted to whatever the first shadow run made
convenient. So `tests/test_universe_expansion_preregistration.py` pins them, the way
`tests/test_momentum_panic_rule.py` pins three numbers set in advance.

Then the guard was mutation-tested (`research/measurements/2026-11-03-preregistration-mutations.py`),
and **the first version had 5 gaps of 19 - all one defect.** The tests asked `needle in text`, and
**the plan states each decision twice** (once in the decisions block, once in the phase section that
acts on it), so mutating one copy left the other to satisfy the test: **a floor re-fitted from
$1,224.13M to $1,200M passed.** That is `DECISIONS.md`'s share-class lesson (2026-10-31, "two
literals for one number are free to drift") one level up - here it was the *check* that tolerated
the drift. `pinned()` counts copies now, every duplicated statement is mutated in **both**
positions, and the result is **25 of 25 caught**, with both files verified clean after each restore.

**2. Rule 8's health line could not leave the alarm state - a self-perpetuating false alarm on the
first line every session reads.**

Found by checking the line the runner handed me against the log it names, which rule 8 asks for
("check it, act on anything it shows").

`health_numbers.last_code_session()` asked `"SESSION DID NOT RUN" in text`. **The runner logs this
module's own output into every run log** (`[INFO] Health: **Health (rule 8, all five):** ...`), so
the phrase arrives in the log of a session that ran perfectly, carried there by the *previous*
session's verdict - and the next run reads it back as its own. From the first genuine failure
onward, **every later session is reported as dead, for ever, however healthy.**

- The genuine marker, 2026-11-02 13:00: `[13:04:04] [ERROR] SESSION DID NOT RUN: session reported
  an error - Failed to authenticate: OAuth session expired...`
- The echo, in the healthy 2026-11-03 08:03 log: `[08:07:01] [INFO] Health: ... SESSION DID NOT RUN
  (session reported an error - ...)`. Note **`RUN (`** where the runner writes **`RUN:`** - which is
  why the reason printed as an empty `()`.
- `logs/nightly-2026-11-03_080346.log` ends `=== Run complete: shipped to main ===`, merged six
  commits, tagged `good/2026-11-03`, and was reported **"no - SESSION DID NOT RUN ()"**. Both of
  today's sessions were handed that false line.

**This is the stuck-alarm twin of the stuck-green defect rule 8 exists for** (2026-09-18: "a
tripwire wired to a number that cannot stand still is decoration"). A line that can only say
"broken" gets ignored, and an ignored line still looks like coverage - the watchdog row's own
argument. The fix matches the runner's line shape (`[ERROR] SESSION DID NOT RUN: <reason>`).

**The number worth reading is the second one.** Four tests were added, and tests written the same
day as a fix pass by construction - so the question asked instead was whether anything else would
have noticed (`research/measurements/2026-11-03-health-echo-mutations.py`): with the fix reverted
and **only those four deselected, 3,784 passed, 1 skipped.** Not under-checked; **invisible.** And
the four are classified by what the mutation measured rather than by their names - **2 detect, 2 are
guards that pass either way** (they pin that a narrower pattern still sees real failures, which is a
real job but is not coverage of this defect). One was renamed because its name claimed more than its
body delivered - 2026-10-25's lesson, that a test's name is read far more often than its body.
`check_loop_health.py` was checked and is **not** affected: it reads commit subjects, not logs.

### Evidence / research

- **PR2:** Fama & French (2008), *JF* 63(4), 1653-1678, doi:10.1111/j.1540-6261.2008.01371.x -
  re-read from `data/papers/ff-2008-dissecting-ivey.pdf`: tiny = below the NYSE 20th percentile,
  ~60% of stocks and ~3% of market cap, 48-69% of firms and 1.0-5.0% of market cap after 1972,
  momentum slopes 0.41 (t 2.51) / 0.82 (t 4.65) / 0.78 (t 3.92). Hou, Xue & Zhang (2020) and
  Jensen, Kelly & Pedersen (2023) as cited in the note (not re-read today; the note marks its own
  second-hand sources). Breakpoints: Ken French Data Library, `ME_Breakpoints_CSV.zip`, CRSP 202609.
- **PR3:** FTSE Russell rule 8.2.3-8.2.4 and MSCI's Value/Growth section 2.2, as the note read
  them - documented practice, both separating size bands before scoring style.
- **Item 2 is a documented failure**, demonstrated against the committed log and reproducible from
  the committed mutation script.
- **No IC series, backtest or engine number is used anywhere above** (rules 4 and 5).

### Methodology changed

- **None.** No weight, metric, threshold, score, ranking or published sentence moved, and
  `METHODOLOGY_CHANGELOG.md` is deliberately untouched: the floor and the band split govern
  universes that do not exist yet, and the band split is *measured* to leave the S&P 500
  byte-identical. The changelog entry falls due when phase 3 publishes, one change at a time, as
  the plan already requires.

### Tried and rejected

- **Re-fitting the floor to the S&P 600's own $1.2B addition threshold**, which looks attractive
  because the two agree to 2% today. Rejected on the June series: they agree in 2026 and in no
  other year measured (2025 $870M vs $1.2B; 2020 $484M). A floor that happens to match a vendor's
  this year is a coincidence, not a rule.
- **Writing the adopted pre-registrations into code this session** (a `size_band` column, the
  per-band `min_market_cap`). That is phase 1, the stage *after* the one the plan's progress line
  named, and the prompt's rule is one stage per session. Deciding was the stage; phase 1 is the
  next session's.
- **Fetching small-cap data to measure availability below the S&P 500.** The note's 40-name probe
  stands as the only such measurement; this review reused the S&P 500's own availability rates as
  an explicit upper bound and said so at every use. A 600-ticker probe at 13:00 would also have
  spent Yahoo quota the 20:00 option refresh needs, for a number phase 2's shadow run produces
  properly.
- **Making the pre-registration tests tolerate the plan's duplicated statements** (the easy fix for
  the 5 gaps). The duplication is real and useful to a reader; what was wrong was a check that only
  looked at the first copy. Counting copies keeps both.

### Next
- **Decide the Investment-inside-Quality proposal** (`research/2026-11-02-investment-inside-quality.md`,
  opened 2026-11-02) - the only one left on the brief's "Waiting for a decision" list, and neither
  of today's sessions wrote it. Its section 4 names four places a weight-preserving fold is *not*
  neutral (missing-data renormalisation, the panic rule, the profiles, the trap flags); measure
  those before deciding.
- Then **phase 1 of `plan/universe-expansion.md`** - make the universe a parameter, S&P 500 payload
  byte-identical. The adopted decisions give it its work order; `min_market_cap: 0` for every S&P
  index band and the `(Sector, size_band)` grouping are both measured to change nothing today.
- **A worthwhile sweep, from item 2:** `health_numbers.py` is not the only reader whose input
  contains its own previous output. The runner writes the health line, the baseline verdict and the
  gate results into each log, and the brief and watchdog read logs. Check whether any other
  substring match can be satisfied by an echo - `data_loop()`'s `"HEALTH: PASS"` was checked today
  and is safe only because it reads `datarun-*.log` while the echo lands in `nightly-*.log`, which
  is luck rather than design.


## 2026-11-03 (owner-run, evening) - "Keep finding improvements ... an elite level of detail and precision"

**Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-03_130000-midday.log`
(completed). Data loop published? yes - `logs/datarun-2026-11-03_020000.log` (completed; HEALTH:
PASS). Evidence base at `1m`: 5 of 8 needed (43 rows; overlapping windows are not independent),
newest 1m run date 2026-10-02 (32 days ago, bound 40). Priority 0: `allow_auto_apply` **false**.
Top open roadmap item: **0.13, The Factor Lab** (age **22 days**); also open: 0.10 27d; 0.14 20d.
**The day's sessions:** the 06:00 run was lost to "Could not resolve host: github.com" at 06:00:01
and its unpublished brief left the tree dirty, which would have stood the 13:00 slot down; restored
by hand at 08:00, the 06:00 task re-run by hand at 08:03 (shipped, `good/2026-11-03`), and the 13:00
ran normally (`good/2026-11-03-1354`). Both entries are above.
**Tests:** 3,850 passed / 2 skipped at the last gate run (from 3,788). Tags `good/2026-11-03-owner`,
`-2`, `-3`; worked in a worktree outside OneDrive and pushed only while no session was running.

### Did

1. **The expected-move card's FALSE divisor - fixed at the source** (queued first by last night's
   sweep; changelog 2026-11-03 evening). The 20:00 refresh divided the straddle by the newest run's
   close, a session *before* its quotes, and chose the at-the-money strike against it, while the
   card named this run's close (273 of 359 divisors off by >1%; NOW's strike 37% away). Both option
   paths now take the stock's price from **the quote response itself** (`context_signals.chain_spot`
   - Yahoo's chain carries `underlying.regularMarketPrice`, the 16:00 ET close at 20:00, no extra
   request). The divisor is recorded (`_ctx_opt_spot`), published (`ctx.osp`) and printed; a move
   without one is withheld with its reason by the generator **and** refused by the page
   (`moveShown`, one predicate for the card, the summary chip and Reporting Soon - so the rebuilt
   page could not print the old payload's moves against "-" before tonight's run); `context_eval`
   counts the same readings the page shows, so the expected-move signal's first `1m` reading moves
   to about 2026-12-04 (readings before the fix are a different construction - excluded, not
   spliced). Checked live after the close on AAPL, JPM, NOW, VLO (NOW: strike 110 at $109.16,
   +/-11.5%, where the old rule priced the 150 strike). Eight behaviour checks on
   `context.expected_move`; **9 of 9** mutations caught. Pushed before 20:00 so tonight's refresh
   runs it - see item 7.
2. **The code loop waits for the network** (10 tries, 30 s apart - the data loop's own budget) before
   skipping the day, and **a brief that cannot be published is set aside**
   (`logs/MORNING_BRIEF.unpublished.md`, only where git ignores it) and the file restored, so a
   failed publish never leaves the tree dirty for the 13:00 slot. Behaviour tests run the runner's
   own block in PowerShell with git stubbed and the publisher in its origin+clone sandbox; **6 of 6**
   mutations caught. Settled row -1 says so.
3. **The midday's queued echo sweep: every log reader now matches the runner's own lines.**
   `write_brief.outcome()`, `newest_session()` and `health_numbers.data_loop()` asked whether a
   phrase appeared *anywhere* in a log that also carries echoed commit subjects, merge stats and the
   health line. On four echo cases built from real lines, a failed run read "completed" twice and
   "stopped deliberately" once, and an echoed "Standing down" subject hid a real session. Now a
   marker counts only as the start of a message the runner wrote (`write_brief.messages`/`said`), and
   a test ties each constant to the runners' own `Write-Log`/`Stop-Run` lines. **39 real logs, 0
   classified differently** (`research/measurements/2026-11-03-log-marker-echoes.py`). The same sweep
   found the health line's "finished" test keyed to "Morning brief published", which a run that cannot
   reach GitHub never writes - so this morning's skipped 06:00 run was passed over and the line
   reported the day before's login failure instead; it now keys on the runner's `finally` line and
   gives a failed run's reason.
4. **Claims audit: queue items 3 and 4 closed, plus rows of items 5 and 6 - 12 rows (38 -> 26 open).** The Insider trades
   card, the case against's insider sentence and the Trend card's 52-week line are rendered by the
   generator's own page for every stock and compared slot by slot with the payload, with the plan
   share and the net-sale rule worked out independently and hand-worked fixtures for branches a day
   may not take; the Market tile's percentile label against the published `pct_10y`, with
   hand-worked ordinals at the edges. **14 of 14** mutations caught. And the track record's period,
   changelog-tick and left-out-picks counts, with `methodology_dates` pinned on a hand-written
   changelog (**6 of 6**). And the percentile-convention note's "N of M metrics marked lower is better
   (EV/EBITDA, Beta and PEG among them)" - a new claim, `drilldown.percentile_convention`, on the
   scorer's own directions (a stale code comment there said 15 of 37; it is 14 of 38). And the
   rankings strip's count, range and median, the median checked to be the ranking's own (one
   mutation survived on today's fully scored data until an unscored row was built by hand).
5. **A data->red test fixed** (flagged by last night's session): `test_sheet_round_three` asserted
   `unscored >= 1`, which a fully-scored day would have turned red with no code change, and named JPM.
   The browser test now uses a real unscored category when the day has one and builds one otherwise
   (both branches run green).
6. **Research, at the owner's request, from primary sources:**
   - `research/2026-11-03-kronos-candlestick-foundation-model.md` - Kronos (Tsinghua, AAAI 2026) and
     "Day Trader". **No change; not a context candidate.** Never tested at a one-month horizon; its
     return IC is pooled, not cross-sectional; on US stocks the released sizes are level with a
     linear baseline (XNAS RankIC 0.0513-0.0601 vs 0.0518, re-read from the PDF); the only trading
     test is one year of A-shares. The post's "1,000 futures" are one model's samples, not outcomes.
   - `research/2026-11-03-open-source-quant-stack.md` - ten repos (OpenBB, TradingAgents, qlib,
     RD-Agent, gs-quant, nautilus_trader, Chronos, QuantLib, skfolio, vectorbt). **One fit:**
     QuantLib could invert our own option mids the way Cboe and OptionMetrics do, instead of
     Yahoo's undocumented `impliedVolatility`, which the ATM IV and the put skew take as given -
     **opened as a pre-registered paired measurement (Proposal status: OPEN)**. qlib's point-in-time
     store imputes lags where `pit_record.py` uses SEC's real `filed` dates; RD-Agent and "backtest
     10,000 versions, keep the survivors" are the selection practice the research engine exists to
     prevent; skfolio's weights beside named stocks would recreate the removed Model Portfolio.
   - Corrected an unsupported sentence in `research/2026-10-27-tradingagents-ideas.md` (the paper
     states no training cut-offs).

7. **The divisor rule, verified on tonight's 20:00 refresh** (ran tonight's code: the main folder was
   fast-forwarded before 20:00). 178 s, 467 of 503 kept, **all 467 with their divisor from the quote
   response, 0 strikes more than 5% from it, 0 moves different from straddle / divisor**. Against the
   same quotes the old divisor would have been off by >1% for 278 (median 1.27%) and >5% for 28 (IT
   13.2%, FISV 12.5%, SYY 9.3% - report-day moves). 30 older cached readings carry no divisor and are
   withheld until refreshed. Tomorrow's 02:00 page is the first built under the rule; the payload
   check `test_every_published_move_reproduces_from_its_published_divisor` stops skipping then.

### Not done, and why

- **The Investment-inside-Quality proposal** - this conversation wrote it on 2026-11-02, so it may
  not decide it; it waits for a scheduled session (any from 06:00 tomorrow).
- **The other 26 open rows** - queued in `plan/claims-mutation-audit.md` (equation lines next).

### Next

- **Run the QuantLib paired measurement** (`research/2026-11-03-open-source-quant-stack.md` section 1)
  - pre-register the bar, then measure one evening's quotes against Yahoo's field.
- **Decide Investment-inside-Quality** (still the oldest open proposal).
- Read tomorrow's brief for the first payload built under the divisor rule:
  `test_every_published_move_reproduces_from_its_published_divisor` stops skipping and checks it.


## 2026-11-03 (owner-run, night) - "keep finding improvements!! be creative. work until the 2 am session"

**Health (rule 8, all five):** as the evening entry above - no scheduled session or data run since.
Last code session ran? yes - `logs/nightly-2026-11-03_130000-midday.log` (completed). Data loop
published? yes - `logs/datarun-2026-11-03_020000.log` (HEALTH: PASS). Evidence base at `1m`: 5 of 8
needed (43 rows), newest 1m run date 2026-10-02 (32 days, bound 40). Priority 0: `allow_auto_apply`
**false**. Top open roadmap item: **0.13, The Factor Lab** (age **22 days**).
**Tests:** 3,879 passed / 3 skipped at the last gate run. Tags `good/2026-11-03-owner-6` to `-12`
and this entry's. Worked in a worktree outside OneDrive; pushed only while no session ran; the
20:00 option refresh ran tonight's code and was measured (evening entry, item 7).

### Did

1. **QuantLib against Yahoo's implied volatility: pre-registered, measured, bar MET, decision OPEN**
   (`research/2026-11-03-open-source-quant-stack.md` sections 1a-1b; pre-registration committed as
   `57958d5` before any number). On 51 stocks: our American IVs from closing mids sit a median
   **3.16 points below** Yahoo's at the money (90th pct 6.09); an independent CRR tree reproduces
   QuantLib on **10 of 10** contracts; the put skew's **sign differs for 17 of 30**. Diagnostic: no
   simple construction reproduces Yahoo's field (last trades a median 9.8 h old), and Yahoo's skew is
   negative for **28 of 48** stocks against **5 of 48** from the closing quotes. ~14 s a refresh if
   adopted. This session wrote the proposal, so a scheduled session decides; the brief lists it.
2. **The options panel's anchoring, completed and seen on the live page.** After the evening fix the
   card withheld a pre-fix move but still printed its "at the money" volatility and skew - NOW's
   from the 150 strike with the stock near $109. The move, the ATM IV and the skew legs share one
   anchor, so a reading without its recorded price withholds all three, in the generator, on the page
   (`optAnchored`) and in `context_eval`; the skew's explanation and effect sizes now appear only
   beside a skew. Looked at the live card at 375px after each change. Simulated tomorrow's 02:00 read
   of tonight's cache: **467** stocks will show option figures with their divisor, 12 older readings
   withheld; NOW will read +/-11.5% divided by $109.16 at the 110 strike. 12 of 12 mutations.
3. **The claims audit's dashboard sweep is closed: 38 -> 0 open rows.** Every computed sentence on the
   page now carries a verdict a test enforces - the workings' weight lines and points equation for
   every row of every stock (the first draft rebuilt the line from its own copy of the template and a
   mutation survived it), the Piotroski / surprise / ROE lines redoing their arithmetic from the
   inputs published beside them, the history and Reporting Soon counts, the price-target strip (its
   mean-target upside equals the scored one for 496 of 496), the share-class gap, the compare
   sentence, the holdings strip, the phone table's pages, and the snapshot's net-income growth and
   days to cover against the run's own fetch. New claims: `drilldown.percentile_convention`,
   `drilldown.price_targets`, `history.rank_counts`, `dashboard.reporting_soon_counts`,
   `dashboard.review_cadence_turnover`, `dashboard.phone_table_pages`. Four mutations survived first
   drafts on today's data (every stock ranked today, no report exactly 8 days out, every company
   scored, a template copied into the test) and were caught by hand-built cases.
4. **A stale figure found and fixed: the review-cadence turnover.** The note printed September's
   "122% at every run against 24% monthly" as "this screener's own history"; re-measured the same way
   on the longer record it is **149%** and **28%**. `history.review_turnover` now measures both at
   every build over every stock each run ranked, and the note names its basis (changelog 2026-11-03
   night, later). The conclusion stands and is stronger.
5. **Research, at the owner's request: "follow the smart money"**
   (`research/2026-11-03-following-the-smart-money.md`). Congress: **not built** - post-2012 trades
   show no outperformance, 2004-08 portfolios trailed an index ($69 vs $80, checked), the data are
   late, ranged and partly scanned, and 5 U.S.C. 13107(c) restricts commercial use. Hedge-fund 13F:
   copying earns the fund's return minus its fees; large-cap evidence after the 45-day lag is weak
   and its sign contested - one log-only line left as an **OPEN** proposal with its direction
   pre-registered. The insider panel already carries the best-evidenced "informed money" signal.
6. **Smaller:** the brief's "Coming up" gains 2026-12-04 (the option signals' first one-month reading);
   two tests updated to the anchoring premise, one of them building an anchored reading rather than
   skipping.
7. **Universe phase 1, first slice** (`plan/universe-expansion.md`, the top item under `OWNER_FOCUS.md`
   Open). `universe.py` defines the S&P 500 (members 495-515, ranked >= 490) and the health gate's
   floor, the rehearsal's full-run floor, `universe_history`'s plausible band and two tests' floors
   read it instead of four literal copies. Values unchanged: `check_run_health`'s output on the latest
   run is byte-identical before and after. The research engine's 32 rows were rerun because
   `universe_history.py` is hashed. Left literal on purpose and named in
   `tests/test_universe_definition.py`: `data-run.ps1`'s `$universe = 500` (not edited hours before a
   run) and the page's `|| 500` fallback.
8. **The publish gate audited for checks a good run could fail.** Tonight's new page checks run in
   the 02:00 loop's publish gate, where a failure discards the night. Three assumptions only today's
   data satisfied were found and removed before they could: the published-turnover check now mirrors
   `build_history` exactly (the run being published defines its own date's point - recomputing from
   the snapshot files alone would have failed on any night an earlier same-day run was on disk); the
   price-target check skips a clamped scored upside and handles a missing price; and count floors
   ("> 400 stocks", ">= 100 per metric") are gone - coverage is `check_run_health`'s to judge, and a
   floor in a claims check would discard a run over a source dip with nothing misprinted. **Lesson for
   whoever adds a check to that gate: it must hold on every good run, not on today's.**
9. **Investor profiles M4's prior question answered by measurement** (`plan/investor-profiles.md`):
   a client-side reweight reproduces the engine - every rank exactly and 1,995 of 1,996 composites
   under all four published weightings, from `dashboard_data.js` alone - **if** it follows the
   engine's order: round the composite to two decimals, *then* rank with ties sharing the lower rank.
   Ranking the unrounded composite gets 50 of 499 Balanced ranks wrong, because 50 stocks share a rank
   today. A port must also carry the momentum regime rule. Whether sliders belong on a decision-support
   page is the owner's call.
11. **Research: days to cover, Revisions' last unresearched leg** (`research/2026-11-03-short-interest-days-to-cover.md`,
   lab question 8). Weight **kept** by the project's rule (no lab evidence; literature marginal where
   the screener lives: value-weighted DTC 0.67% a month, t 2.24 all stocks, no S&P 500 test; Asquith et
   al.: large heavily-shorted stocks "do not reliably underperform"). Practice keeps short interest out
   of revisions - a placement question for a synthesis day. Not analyst data, so testable: a
   pre-registered research-engine experiment on FINRA's free days-to-cover is proposed (OPEN).
10. **Two stale statements corrected:** CLAUDE.md said experiment 18 had no second reader (it had one
   on 2026-11-02), and priority 6's M4 question is now answered. The page's last literal universe size
   (`|| 500` in the rank chart) reads the universe being run.

### Not done, and why

- **The three open proposals** (Investment-inside-Quality, QuantLib IVs, 13F log line) - this
  conversation wrote all three, so a scheduled session decides them.
- One research agent sent the owner's email address in a browser header while downloading a paper
  from nber.org. Reported to the owner; nothing to undo.

### Next

- **Decide the QuantLib IV proposal** - the measurement is done and the bar met; the decision is
  whether to adopt, and if so to log which construction each reading used and exclude the Yahoo-field
  skew from `context_eval` (the 2026-10-14 precedent).
- **Decide Investment-inside-Quality** (oldest open proposal).
- After the 02:00 run: `test_every_published_move_reproduces_from_its_published_divisor` and
  `test_the_published_turnover_is_the_snapshots_own` stop skipping and check the first payload
  built under tonight's rules.
- **37 references to named tickers in 6 payload-reading test modules** (JPM, EXPE, APA, MSFT, plus HST
  and BBY in one compare tray; `research/measurements/2026-11-03-named-tickers-in-payload-tests.py`) -
  row "data->red" in waiting: each turns `main` red the morning that company leaves the index. The
  recipe: pick by property from the payload (the highest-ranked stock with every category scored; the
  highest-ranked stock scored with bank weights for JPM's uses; for APA's short-sheet test, the
  property it was chosen for), with the named ticker kept as the first choice so today's behaviour is
  unchanged. Not done tonight on purpose - it rewrites JavaScript inside six UI modules, which is
  daytime work, not a midnight edit two hours before a data run.
- Universe phase 1 continues: `data-run.ps1`'s `$universe = 500` (a daytime edit to the data loop's
  own script, with a dry run), then `size_band` and the per-band gates.

---

## 2026-11-04 - SYNTHESIS. Whatever has changed the methodology since the last synthesis session: how does it fit the rest of the screener, what does it overlap with, what does it make redundant, what does it imply for the other seven categories? Design the coherent whole, not the isolated tweak. Record any methodology change in METHODOLOGY_CHANGELOG.md with its sources.

**Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-03_130000-midday.log`
(completed). Data loop published? yes - `logs/datarun-2026-11-04_020001.log` (completed; HEALTH:
PASS). Evidence base at `1m`: 5 of 8 needed at the 1m horizon (43 rows, but overlapping windows are
not independent; 111 rows across all horizons), newest 1m run date **2026-10-02 (33 days ago, bound
40)**. Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab**
(age **23 days**); also open: 0.10 28d; 0.14 21d.
**Tests:** before 3,887 passed / 1 skipped; after **3,893 passed / 1 skipped**. Gate 5 (publish
rehearsal) run twice - failed first, see "Did" item 4 - and passes.
**Owner queue / rotation:** the morning brief listed **four** open proposals, so deciding the
oldest was the day's first job (prompt section 1). It is also squarely a synthesis question -
*should a 5-point category be folded into another one* - so the rotation's focus and the owner
queue pointed at the same work, and nothing was deferred for it. The deferral-streak rule did not
fire: the top open roadmap item (0.13) is not the thing three consecutive sessions have passed
over, and this session carried an owner item.
**I am today's first session.** No earlier entry dated today.

### Did

**Decided the Investment-inside-Quality proposal: the fold REFUSED, the recommendation ACCEPTED,
and the page work it asked for shipped.** `research/2026-11-02-investment-inside-quality.md`, open
2 days, written by an owner-run session. **This is the first of seven reviewed proposals whose
recommendation was accepted as written** - the six before it were refused - and the reason is worth
recording, because it is not that this note is better argued than the others: it is that it
recommends *no change to any number*, and it measured the costs of the change it declines before
declining it. A reviewer's job is easier when the author has already tried to refute themselves.

1. **Re-measured section 4 from scratch, on eight runs instead of one.** The note's own script was
   not committed, so the review could not read it; it was rebuilt from `factor_engine` as
   `research/measurements/2026-11-04-investment-fold-review.py` (committed) and run over the eight
   published payloads 2026-10-22..2026-11-04. The reproduction is faithful - published minus
   recomputed is 3.5e-07 on Quality, 0.0 on Investment, 4.99e-03 on the composite (the payload's
   own 2dp rounding). **Every claim held**, and the three counts that were "one cross-section" are
   now ranges: **455 of 499 composites unchanged on all eight runs, the top 25 untouched (0 in,
   0 out) on all eight**, Quality's dispersion down **15.0-15.4%** against `check_run_health`'s 20%
   refusal bar. So the fold buys no ranking change at all, and would spend three-quarters of a
   safety margin on methodology rather than on data.
2. **One of the note's listed costs is struck, in the cheaper direction.** §3.2 says
   `scripts/check_run_health.py` reads `dispersion_history.csv` positionally, "so dropping a column
   shifts every later one". The positional read is real; the consequence is not, twice over -
   `investment_disp` is the file's **last** column, and `check_run_health.CATEGORIES` is **five**
   categories (valuation, quality, growth, momentum, risk), so the gate never reads that column at
   any position. The verdict is unchanged, because the gate *does* read Quality, which is where the
   dispersion cost actually lands. (Noticed in passing and not a defect: the CSV **has** a header
   row, which `header=None` loads as data and `to_numeric(errors="coerce").dropna()` then discards.
   It works by accident.)
3. **Every load-bearing figure re-read from the PDF bytes**, not from the note. Fama & French
   (2015) Table 6 and Table 4 - fourteen figures, **all exact**, including the big-company CMA at
   0.22/t2.00, 0.11/t1.50, 0.07/t1.03 and the p. 8 sentence *"almost all the average value of CMA
   is from small stocks"*. MSCI FaCS's 25% Quality-group share **and its worked Microsoft example**,
   exact. Scientific Beta Exhibit 14, Hsu/Kalesnik/Kose p. 57, Asness p. 5: exact. **Two one-page
   citation slips and no wrong number** - the Scientific Beta quote is on p. 34 not p. 35, the MSCI
   worked example on p. 9 not p. 8. One of the note's own figures has drifted (the lab's
   since-2003 big-company investment factor, -0.06/t-0.35 on 11-02, **-0.049/t-0.312** today)
   because the lab's data gained months; the sign and the argument are unchanged, and the page
   states that one as a band held to the lab's live data rather than as a literal that goes stale.
4. **Shipped the sound part: four paragraphs on the methodology page**, closing items 1-3 of the
   page work `research/2026-10-19-revenue-growth-vs-asset-growth.md` §11.2 queued on **2026-10-19**
   and open 16 days. The Investment section had **no large-company caveat** where Size and Momentum
   both carry thorough ones - the same standard applied unevenly, which is how that note put it. It
   now says which asset-growth measure the metric is and why that matters, what the large-company
   evidence actually shows, what the category does jointly with Growth, and why it is standalone
   when most providers fold it. Claim `overview.investment_standalone`,
   `tests/test_investment_standalone.py` (6 tests, in the 02:00 publish gate by derivation).
   **The rehearsal caught the thing the four gates cannot see**, exactly as designed: editing the
   generator left the committed `SCREENER_OVERVIEW.md` stale and gate 5 failed on the drift
   tripwire. Regenerating it needed a full run or a hand-edit, so there is now
   `scripts/regenerate_overview.py` (`--check` to test, no fetch, no scoring, writes only that
   file). Second rehearsal passes.
5. **Looked at the page, at 1440 and 375** (`scripts/shot_methodology.py`, pointed at the
   rehearsal's regenerated page). Desktop: the four paragraphs sit in the Methodology dialog's body
   beside the "On this page" nav, bold lead-ins, italic quotes, the inline `research/...md` code
   chip and the en dashes all rendering correctly. Phone: single column, paragraph box **307px
   wide, no right overflow, visible**; the bold lead-in wraps onto two lines and reads fine. The
   payload is byte-identical - this is page text, and `dashboard_data.js` and
   `dashboard_context.js` were unchanged in the rehearsal. The shot script was hardcoded to the
   Growth section, so `--section` and `--marker` are arguments now, defaulting to the old values.

6. **The `auto-rescue` stash is not empty, and every session since 2026-10-28 has been told it
   is.** Orient says to inspect it; `git stash show -p stash@{0}` prints **nothing**, which reads
   as "no work to recover" and is how it has survived seven days. It has **three parents** - it was
   made with `-u`, and `git stash show` does not show the untracked-files parent. That parent
   (`f2a4a40`) holds three files: two `improvement/price_cache/` parquets, both since regenerated
   and present, and **`data/context_log/2026-10-28.parquet`** - 503 rows, 46 columns, schema
   identical to its neighbours, and **the only weekday absent from `data/context_log/`**, whose
   other 18 days are *tracked in git*, not ignored.
   **It is deliberately not committed**, which is the judgement call here rather than the find.
   `logs/datarun-2026-10-28_020001.log` ends `Run discarded: published-claims check failed`
   (HEALTH: PASS, then `tests/test_case_against.py` failed in the publish gate) - so 2026-10-28
   published nothing, and there is no payload commit for that date. Committing its context log
   would splice an **unpublished run's** signals into the series `context_eval` reads, and this
   project's standing rule runs the other way ("the skew before 2026-10-16's log is excluded from
   `context_eval`, never spliced"; CLAUDE.md 0.11). Whether a refused run's readings belong in that
   series is a real question with an evidence consequence, and it is not a loose end to settle in
   the last ten minutes of a session that was deciding something else. **Nothing is lost**: the
   stash is untouched and the blob is
   `git cat-file blob f2a4a402a27756717bb9aaf3ed9b9481bdf0d9d1:data/context_log/2026-10-28.parquet`.
   Restored it, validated it, then removed it again so the tree is clean. **Do not drop the stash
   until that question is answered.**

### Evidence / research

The decision rests on published research and documented practice (rules 4 and 5; no return, IC or
backtest figure is used anywhere). Fully cited in `METHODOLOGY_CHANGELOG.md` 2026-11-04 and the
note's new section 6.4; the four that carry it:

- **Fama & French (2015)**, *JFE* 116(1), 1-22, doi:10.1016/j.jfineco.2014.10.010. Table 6 (p. 11):
  regressed on the other four it is **value** that is spanned (HML intercept -0.04, t -0.47) and
  investment that survives (CMA 0.28, t 5.03) - the academic models keep investment separate
  because it is the leg that holds up, which is the opposite of the overlap worry. Table 4 (p. 7)
  and p. 8: among big companies it is 1.03-2.00 standard errors from zero, 0.07%/month (t 1.03)
  under joint controls.
- **Asness (2014)**, AQR, p. 5: with value on the current price and momentum added - this
  screener's setting - *"it's actually CMA that's mostly redundant."*
- **MSCI FaCS Methodology** (Jan 2018), p. 8: Investment Quality is its **own** risk-model factor,
  given a 25% share of the Quality *group* for reporting. The structure closest to a screener that
  publishes category scores keeps the factor visible and groups it one level up.
- **ERI Scientific Beta** (May 2017), Exhibit 14 (p. 35) and the conclusion on p. 34: on the **500
  largest US stocks**, 1975-2015, investment and profitability correlate **-0.18** and *"cannot be
  combined in a composite manner to represent a single risk factor"*.

**The synthesis finding, which the note did not make and which is this session's own contribution.**
Reading Exhibit 14 rather than only the sentence quoted from beside it: among large companies the
investment factor's strongest documented relationship by a wide margin is **+0.68 with low
volatility**, not with profitability (-0.18). This screener's own scores agree in direction, far
more weakly: Investment against Quality **-0.068**, Risk **+0.166**, Valuation **+0.228**, Size
**+0.259**. **So the fold would house asset growth under the category it is least related to** -
an argument against it that is independent of explainability, and the one that makes the decision a
coherence decision rather than a presentational one. It also reframes what is still open: if these
5 points overlap anything here, it is Risk's 10 and Valuation's 22, not Quality's 22. Exhibit 15,
on the facing page and uncited by the note, says the same by overlap - the two factor indices share
only ~50% of their stocks (110 of 250), the same as value, size and momentum share with each other.
**Nothing here authorises moving a weight**: that is a correlation between long/short factors, not
between this screener's categories, and +0.166 on one run is weak. It is a Monday.

### Methodology changed

- `METHODOLOGY_CHANGELOG.md` **2026-11-04 - "Investment stays its own category, and the page now
  says why"**. A decision entry: **no weight, metric, threshold or scoring line changed** and
  `CAT_METRICS` keeps eight categories. What changed is the published methodology text, with its
  sources and the structural measurement behind the refusal.

### Tried and rejected

- **The fold itself** - refused on the measurement above and on Scientific Beta's -0.18 / MSCI's
  decision to keep the factor visible. The clincher beyond the numbers: **ERIE misses all six
  weighted Quality metrics and has no Quality score at all**; folded, renormalisation would hand
  asset growth the whole 27 points and publish a "Quality" score for it composed **entirely of the
  one metric that is not about quality**.
- **A first draft of the page paragraph said the screener's total-asset-growth measure is "not the
  per-share version Fama & French use to sort CMA".** That is wrong - FF2015's `Inv` *is*
  total-asset growth, confirmed from the paper (*"Inv is the growth of total assets for the fiscal
  year ending in t-1 divided by total assets at the end of t-2"*), and the 10-19 note says so at
  its line 319. The per-share construction belongs to their earlier anomaly work. Corrected before
  commit; recorded in the note's §6.5 because the near-miss is the instructive part - the
  distinction is real and worth stating, and getting its attribution backwards would have put a
  false sentence **about a cited paper** on the public methodology page.
- **Asserting the fold's counts of every run.** "455 of 499" was true on all eight runs measured,
  but a count that moves with coverage must not become a standing claim: a claim's check runs in
  the 02:00 publish gate, where failing **discards the night** (CLAUDE.md row `data->red`, and
  2026-11-03's own lesson). The page states it as what a named measurement found over named runs,
  and the check holds it to the committed script and note rather than to today's payload.

### Next

- **Decide the three remaining open proposals**, all opened 2026-11-03 and all written by the same
  owner-run conversation, so any scheduled session is an independent reader: the **QuantLib
  implied-volatility construction** (measurement done, its pre-registered bar met - the decision is
  whether to adopt, and if so to log which construction each reading used and exclude the
  Yahoo-field skew from `context_eval`), the **13F-filer log line**, and **registering the
  `short_interest_ratio` research-engine experiment** (note that registering one forces a rerun of
  all 32 rows, ~20 min with `--jobs 4` - budget it).
- **A Monday on Investment's 5 points and Risk**, which §6.3 opened: the documented large-company
  correlation of the investment factor with low volatility is +0.68, and this screener carries
  Investment 5 beside Risk 10 whose heaviest metric is volatility at 42.86%. The question is
  overlap, not the heading. Pre-registration before any measurement, per the lab-to-weights route.
- **Decide whether a refused run's context log belongs in `data/context_log/`** (item 6 above),
  then either commit `2026-10-28.parquet` from the stash or drop the stash saying why. While it is
  undecided, **two things should change regardless**: Orient's instruction to read a stash with
  `git stash show -p` is wrong for any `-u` stash and should say to check the parent count, and a
  data run that is discarded currently leaves its context log on disk, uncommitted and invisible -
  if those readings are not wanted, the run should not write them.
- Still waiting from 2026-11-03: the **37 named-ticker references in 6 payload-reading test
  modules** (daytime work, rewrites JavaScript in UI modules), and **universe phase 1**'s next
  slice (`data-run.ps1`'s `$universe = 500`, then `size_band` and the per-band gates).

## 2026-11-04 (owner-run, morning) - the 06:00 work landed, and a refused branch is no longer invisible

**Health (rule 8, all five):** last code session ran? yes, but **held back by its gates** -
`logs/nightly-2026-11-04_060000.log` (gate 5 failed once: "FAIL  check_published_claims").
Data loop published? yes - `logs/datarun-2026-11-04_020001.log` (completed; HEALTH: PASS;
published claims 2,141 passed, the first payload built under the 11-03 anchoring and turnover
rules). Evidence base at `1m`: 5 of 8 needed (43 rows), newest 1m run date 2026-10-02 (33 days,
bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor
Lab** (age **23 days**). **Tests:** 3,896 passed / 1 skipped on the merged tree; all five gates
passed here.

### Did

1. **The 06:00 session's work is on `main`.** Its decision (Investment stays standalone, the fold
   refused, four paragraphs on the methodology page) passed gates 1-4 and failed gate 5 once at
   06:51-06:56. The same gate on the same commit then passed **three** times: in a scratch worktree,
   in the main folder with the runner's exact command at ~08:30, and on the merged tree here. The
   session's own rehearsal had also passed. Merged unchanged, all five gates re-run, pushed while
   no session ran.
2. **Why nobody could say what failed, fixed.** The runner logged only lines starting FAIL/FAILED;
   `check_published_claims.py` prints its verdict as "PUBLISHED CLAIMS: FAIL (pytest exit N)" and
   a collection or teardown failure as `ERROR ...` lines, and the full output was kept nowhere. Now
   a failed gate 1 or 5 writes its whole output to `logs/nightly-<stamp>.gate1.txt` / `.gate5.txt`
   (never fatal), and gate 5 logs every ERROR and PUBLISHED CLAIMS line. The likeliest cause - a
   synced-folder lock during the rehearsal's install, seen before on 2026-10-15, or a parallel
   browser-test timeout - is a guess until the next failure leaves its output.
3. **A refused branch is named to the next session.** `health_numbers.held_back` lists every
   `origin/nightly/*` branch holding commits `origin/main` lacks, in the health line the runner puts
   in every prompt; `prompts/nightly.md` "Work the gates held back" says to read the saved output,
   merge the branch into the session's own, and let the runner's gates judge it again. Before this,
   the 13:00 session would have started from a `main` without the morning's work or its log entry.
4. **The brief says what happened.** A failed ship gate read "stopped deliberately", which sounds
   like the session chose to stop; it now reads "held back by its gates (work kept on its branch,
   not on main)". "Stopped deliberately" stays for a discarded data run.

### Next

- When a gate next fails, read its `.gate5.txt` / `.gate1.txt` before re-running anything - the
  first saved output is the evidence the 11-04 failure lacked.
- The 06:00 entry's Next stands (the stash parent-count note; the 37 named-ticker references;
  universe phase 1).

## 2026-11-04 (13:00 midday) - QUEUE. A second session today - more of the work, not a different kind. Read today's earlier NIGHTLY_LOG.md entry first and carry on from its Next; if it finished, take the next open item in OWNER_FOCUS.md, then the top of CLAUDE.md "Current priorities", then the queues in plan/factor-lab.md, plan/context-layer.md and plan/backtest-v2.md. Finish one item end to end, to the same evidence standard as any other session. Do not repeat or reverse a decision an earlier session made today without new evidence, and write down why if you do.

**Health (rule 8, all five):** last code session ran? **the runner's line said "held back by its
gates"** - `logs/nightly-2026-11-04_060000.log` - but **nothing is actually held back**:
`health_numbers.held_back` named no branch and `git log origin/main..origin/nightly/2026-11-04` is
empty, because the owner-run morning session merged that work after re-running all five gates (its
entry above). So there was no refused branch to merge, and the line is reading the 06:00 run log's
own ending rather than where the work ended up - worth knowing before the next session spends time
looking for it. Data loop published? yes - `logs/datarun-2026-11-04_020001.log` (completed; HEALTH:
PASS). Evidence base at `1m`: **5 of 8 effective** observations needed at the `1m` horizon (43 rows,
but overlapping windows are not independent; 111 rows across all horizons), newest `1m` run date
**2026-10-02 (33 days ago, bound 40)**. Priority 0: `allow_auto_apply` **false**. Top open roadmap
item: **0.13, The Factor Lab** (age **23 days**); also open: 0.10 28d, 0.14 21d.
**Tests:** before **3,896 passed / 1 skipped**; after **3,901 passed / 1 skipped**.
**Owner queue / rotation:** I am today's **third** session. The 06:00 entry's Next named the three
open proposals as the first job, and prompt section 1 says deciding one outranks the focus, so I
took the one it named first. Nothing was deferred for a broken loop or a gate. The deferral-streak
rule did not fire: this session carried an owner item (`OWNER_FOCUS.md` item 1, "Decide the OPEN
proposals"). No decision made earlier today was reversed.

### Did

**Decided the implied-volatility proposal: ACCEPTED - the options panel should invert its own
bid/ask mids rather than take the quote response's `impliedVolatility` field as given.**
`research/2026-11-03-open-source-quant-stack.md` section 1, open 1 day, written by an owner-run
session; section 1d is the review. This is the **ninth** proposal reviewed and the **second running**
to be accepted after six refusals. I did not write the note or its measurement.

1. **Pre-registered my own reading before computing anything** (section 1c, commit `fb27f1c`): a
   disjoint sample, the candidate construction the author could not test, the dividend gap in its
   cross-check, the literature check, and the adopt/refuse rule. The author's own pre-registration
   checks out in `git log` too - section 1a in `57958d5`, the result in `741215b`.
2. **Replicated on a disjoint sample, mid-session** (`recs[5::10]`, no overlap with the author's
   `recs[::10]`): 40 of 50 stocks priced, QuantLib minus the field **median -1.78 volatility points,
   median |difference| 2.00, max 7.51**. The 1-point bar holds on a second sample; the gap is about a
   third narrower at 14:10 ET than the author's 3.16 at 21:00.
3. **Closed the gap in the author's cross-check.** Its printout said 10 of 10 ATM calls reproduced an
   independently written binomial tree but not how many carried a **dividend** - the part the note
   itself calls hardest. Measured on two sets chosen by the same rule: **10 of 10 with no dividend
   and 9 of 9 with one**, 19 of 19 within half a point (largest CF, 60.46% against 60.82%).
4. **Found the deciding evidence already in this repository.** The note marks its key inference -
   that a negative put skew is the unusual case - as uncited. It needn't be: **Xing, Zhang & Zhao's
   Table 1 (p. 647)** puts the 5th percentile of their skew at **-0.99** points, so a negative skew
   is about a **one-in-sixteen** event in the data the measure comes from. It was read from the paper
   by the 2026-10-14 session and sits in
   `research/measurements/2026-10-14-skew-distribution-vs-paper.py`.
5. **Measured our published record against it on the whole universe, not 48 stocks.** From the
   committed context logs: the field puts the skew negative for **42-60% of stocks (median 47%) on
   every one of the 12 logged days** under the current construction, 476-493 stocks a day - about
   **5,800 stock-days** - with a median skew straddling zero against the paper's +4.76 points. I also
   dated the construction boundary from the data (the two-leg columns first appear in the **10-16**
   log, six old-construction days before it) and confirmed `context_eval`'s existing predicate keys
   on those legs rather than on a date, so it already puts the boundary in exactly the right place.
6. **Shipped the piece the switch needs first**, which is also the change to the proposal:
   `_ctx_opt_iv_src` on every reading, `IV_CONSTRUCTIONS` and one `PUBLISHED_IV_CONSTRUCTION`
   constant, and `context_eval.iv_is_the_published_construction` masking the two IV-derived signals
   so the record can never **pool** two constructions - the 2026-10-14 splice one level down. An
   unstamped reading counts as `source`, because nothing here has ever inverted an option price. The
   expected move is deliberately **not** masked: it is mids over the spot with no inversion in it,
   and a test fails if that distinction is lost. `METHODOLOGY_CHANGELOG.md` 2026-11-04 (13:00); 5
   tests. **No published number, score, rank or sentence changed** - every reading today is `source`,
   so the mask passes exactly what it passed yesterday, and the stamp is not in `CTX_FIELDS`.

### Evidence / research

- **Xing, Zhang & Zhao (2010)**, *JFQA* 45(3) 641-662, **Table 1, p. 647**: `SKEW` mean **+6.40**
  volatility points, **5% -0.99**, 25% +2.40, 50% +4.76, 75% +8.43, 95% +19.92; 1996-2005, ~840
  firms a week, OptionMetrics. Conditions matter and are stated in the note: all optioned US stocks,
  and skew is a small-company phenomenon in their own data (steepest-skew quintile $4.29bn of market
  value against $13.8-14.7bn in the middle), so an S&P 500 screen should sit flatter - which is why
  the 10-14 session's ~5-point level gap is reported rather than treated as an error. Read from the
  paper by that session, not by me; I used its recorded figures.
- **Cboe**, *American and European-Style Theoretical Options Calculation Methodology*, and
  **OptionMetrics IvyDB** product literature (both cited in the note): a binomial tree with discrete
  dividends, inverted against the **mid of the best bid and offer**, off the Treasury
  constant-maturity curve. XZZ's volatilities are IvyDB's. Our proposed construction is theirs with a
  linear rather than cubic-spline interpolation of the same free curve; the field we publish today is
  documented nowhere.
- **A documented failure, measured:** the field reproduces **none of eight candidate constructions**
  across two sessions. Mine were the four the author could not reach after the close, including the
  obvious remaining one - the last trade inverted against the underlying's price **at the minute of
  that trade**, from the day's 1-minute bars (American 2.15 points, European 2.13, against ours at
  2.30; 11-12% of contracts within half a point in all three). So the argument for adoption is not
  "our sign is more plausible" but this project's own rule: **a number nobody can rebuild cannot be
  checked from the page** (row 0.8).

### Methodology changed

- `METHODOLOGY_CHANGELOG.md` **2026-11-04 (13:00) - "Proposal ACCEPTED: the options panel will
  invert its own mids, and a reading now records which construction it used"**. Display-only area
  (settled row "ctx"), plus the evidence record. **No weight, metric, threshold or published figure
  moved.** The inversion itself is the next session's build; its order is section 1d's last
  subsection.

### Tried and rejected

- **Building the inversion in the same session.** I checked whether it was tractable rather than
  assuming either way, and the answer is no, for one measured reason: the dividend *schedule* needs
  an ex-date, and the newest `00_raw_fetch.parquet` carries `dividendRate` (403 of 503) and
  `trailingAnnualDividendRate` (503 of 503) but **no `exDividendDate`, `lastDividendDate` or
  `lastDividendValue`**. The 11-03 measurement got dates from `yf.Ticker.dividends`, a request per
  stock inside the pass that tripped the rate limiter when it ran inside the core fetch. So the build
  starts with a **fetch-schema change** (`exDividendDate` onto the already-pulled `.info`), which
  cannot be verified without a throwaway full run. Written into the build order as its first step.
- **The note's own account of why the field disagrees** - "computed at some earlier moment of the day
  against a different underlying price". Not confirmed; see above. Left in the note as the author's
  hypothesis with my refutation beside it.
- **My own first sizing of what the switch costs.** The draft said "18 logged days of
  `options_atm_iv` and 12 of the skew", read off the log files. Run through
  `context_eval.signals_frame` itself: **one day, 467 and 466 readings** - all seventeen earlier days
  are already excluded by the 2026-11-03 (night) anchoring rule. **Fifteen times too large**, and it
  inverts the conclusion about timing: there is no month of evidence to protect, so delay is the
  expensive option. Corrected in the note, the changelog, the `context_eval` docstring and the test
  that repeated it. The general lesson: the cost of a change to an evidence series is a number to get
  from the code that builds the series, not from the files underneath it.
- **A full-universe measurement of *our* construction.** It would need ~500 option chains re-pulled
  mid-session, and the one documented cause of a rate-limited run is a second full pass within hours
  of another. Both our figures (48 stocks after the close, 26 mid-session) are small samples and the
  note now says so; the field's side is the one measured on 5,800 stock-days.

### Next

- **Build the inversion**, order in `research/2026-11-03-open-source-quant-stack.md` section 1d:
  `exDividendDate` onto the `.info` fetch first (verified with a throwaway full run), then one
  inversion feeding all three contracts, the cached FRED curve with a fallback, a `source` fallback
  and stamp wherever QuantLib or a mid is missing, then flip `PUBLISHED_IV_CONSTRUCTION` **with the
  generator's matching guard in the same commit**, pin QuantLib 1.44 in `requirements.txt`, say the
  construction on the panel with a registered claim, and measure coverage before and after. The one
  unsized cost is coverage: **160 of 196 contracts (82%)** carrying a source IV had a usable mid.
  Do not loosen `MAX_REL_SPREAD` or the IV bounds to win it back.
- **Two proposals still waiting**, both opened 2026-11-03: the 13F-filer log line
  (`research/2026-11-03-following-the-smart-money.md`) and registering the `short_interest_ratio`
  research-engine experiment (`research/2026-11-03-short-interest-days-to-cover.md` - registering one
  forces a ~20-minute rerun of all 32 ledger rows, so budget it).
- The 06:00 entry's Next still stands: the refused-run context-log question (and the
  `auto-rescue` stash with it), the 37 named-ticker references in 6 payload-reading test modules, and
  universe phase 1's next slice.

## 2026-11-04 (owner-run, evening) - "keep finding improvements please, be creative": the data loop's blind spots

**Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-04_130001-midday.log`
(completed). Data loop published? yes - `logs/datarun-2026-11-04_020001.log` (completed; HEALTH: PASS).
Evidence base at `1m`: 5 of 8 effective observations needed (43 rows), newest 1m run date 2026-10-02
(**33 days**, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The
Factor Lab** (age **23 days**). **Tests:** 3,896 passed / 1 skipped at the morning's merge; **3,908
passed / 1 skipped** on this tree (parallel, 3:15). All five gates passed in a worktree outside
OneDrive; pushed while no session ran; tag `good/2026-11-04-owner-2`.

### Did

1. **The 02:00 run rate-limits itself, every night - and nothing recorded where.** Seven consecutive
   02:00 runs: 39-51 tickers rate-limited, a 125-155 s retry pass, every ticker recovered. The
   2026-11-02 run had **no fetch of any kind for 72 hours before it** (no `runs/` folder since 10-30,
   no owner session, no weekend option refresh) and still read 45 - the control the timing note's
   section 8.5 asked for, so the "leftover budget from an earlier run" reading is refuted
   (`research/2026-10-26-data-run-timing.md` section 8.7). The data loop logged only the last 25 lines
   of the screener's output, so the batch-by-batch throttle was discarded every morning. Now:
   - `data-run.ps1` saves the whole output as `logs/datarun-<stamp>.screener.txt` (never fatal);
   - `meta.json` records `rate_limited_positions` / `_tickers` / `rate_limit_errors`
     (`run_screener.rate_limit_detail`; provenance only, `factor_engine.py` untouched so no ledger
     rerun).
   No setting changed: the note says what a week of both must show before one does.
2. **A refused session whose work later landed is no longer reported as held back.** The 13:00
   session's health line said "held back by its gates" about the 06:00 run whose work this morning's
   owner session had merged, and that session spent its opening looking for a branch. The verdict now
   asks git whether the newest commit the runner listed is on `origin/main` ("held back by its gates,
   and that work has since reached main"; the health line says **yes**). A sha git cannot place keeps
   the warning. Tested in a bare-origin sandbox.
3. **Log retention counts runs, not files.** `prune_artifacts.prune_logs(keep=60)` counted every file,
   so each new sidecar (transcripts, the morning's `.gate1/.gate5.txt`, tonight's `.screener.txt`)
   silently shortened the history kept and could prune a run's saved gate output while its log
   survived. Now a run's sidecars live and die with its `.log`.
4. **An index change can no longer discard a night's data through a browser test.** Four modules in
   the 02:00 publish gate opened EXPE, APA, HST and BBY by name (the queued "37 named-ticker
   references"); a deletion from the S&P 500 would have failed the gate on every run. `tests/examples.py`
   returns the named stock unchanged while the payload carries it - every test reads exactly as before
   today - else the best-ranked stock with a full sheet. **Measured on a copy of the payload with those
   four removed: the old modules failed 6 tests, the new ones none that open a stock** (two others
   failed on that doctored copy because removing stocks moves the median and the profile ranks - an
   artifact of the surgery, not of a name). JPM and MSFT stay named: their tests assert what only
   those companies are. Tripwire: `tests/test_browser_examples.py` (caught the old APA line when put
   back; one false positive, a test that deletes FOX itself, exempted by rule).

### Not done, and why

- **4 GB of stale agent worktrees sit inside the OneDrive folder** (`.claude/worktrees/agent-*`, from
  2026-11-02, none holding unmerged or uncommitted work - checked). `git worktree remove` unregistered
  the two git still knew, but the sandbox refused deleting their files. **Removed later the same
  evening at the owner's request**, after checking every file git would track in all four against the
  repository's history (each one's content is in a commit; only the `.git` pointer differed): 4 GB, the
  dead `.git/worktrees/` records and two merged `worktree-agent-*` branches; `git fsck` clean after. OneDrive syncing 4 GB of churn is a plausible contributor to the file locks suspected in
  this morning's unreproducible gate-5 failure - unproven.
- The three proposals and the inversion build stay with the scheduled sessions (this conversation
  wrote two of them).

### Next

- **After tomorrow's 02:00 run, read `logs/datarun-2026-11-05_*.screener.txt` and the run's
  `phase_detail.fetch.tickers`**: which batch the first `Rate limit detected` arrives in, and whether
  the positions cluster. A week of that is the evidence section 8.7 asks for before any fetch setting
  moves.
- The midday entry's Next stands: build the implied-volatility inversion (fetch `exDividendDate`
  first), then the two waiting proposals.

## 2026-11-05 - BUILD. Implement what the week's research justified. Write tests alongside the code.

**Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-04_130001-midday.log`
(completed). Data loop published? yes - `logs/datarun-2026-11-05_020000.log` (completed; HEALTH: PASS).
Evidence base at `1m`: **5 of 8 effective observations needed (44 rows; 112 across all horizons),
newest 1m run date 2026-10-05 - 31 days ago, bound 40.** Priority 0: `allow_auto_apply` **false**.
Top open roadmap item: **0.13, The Factor Lab** (age **24 days**); also open 0.10 (29d), 0.14 (22d).
No branch held back by the gates. `git stash list` carries one `auto-rescue 2026-10-28_060001` entry
that `git stash show` reports as empty (no diff, no stat) - left alone, as four prior sessions did.
**Tests:** after **3,926 passed / 1 skipped** (parallel, 3:04). **The baseline was not a separate
run** - it is read off the mutation check in item 2, which ran the whole suite with this session's
first new module excluded and reported 3,906 passed with the 2 failures the mutation itself caused,
i.e. **3,908 passed / 1 skipped** on the arriving tree, matching the 11-04 evening entry. The 18 new
tests (11 + 7) account for the difference exactly. All five gates pass on this tree: suite, dry-run,
both payloads parse, clean tree, and `scripts/rehearse_publish.py`.

**Owner queue / rotation:** `OWNER_FOCUS.md` item 1 - **decide the OPEN proposals** - outranked the
rotation, and both were listed in the brief at 2 days old. **Both decided; the queue is empty for the
first time since 2026-11-03.** Thursday's build is the guard the first decision needed, so this is a
build day as well as a decision day. Deferred: the implied-volatility inversion, named in the 11-04
13:00 entry's Next - it is the accepted ninth proposal's build and needs a session that is not also
clearing two decisions; it is this entry's Next.

### Did

1. **Proposal decided - `research/2026-11-03-short-interest-days-to-cover.md` section 5: REFUSED as
   written. The experiment it asks for cannot measure anything.** It wanted a research-engine
   experiment dropping `short_interest_ratio` from Revisions. The engine has never scored that
   metric: `v2_config` zeroes it and the record leaves its column **NaN in 176 of 176 months**
   (68,197 stock-months), which makes **Revisions the only category backtest v2 does not score at
   all**. Measured over every month on file
   (`research/measurements/2026-11-05-short-interest-registration.py`): the arm, written the way the
   registry's own metric-level precedent writes it (`drop-revenue-growth`, `kind: config`), is
   **dict-identical to the baseline**, and ranks **0 of 67,113 stock-months differently - maximum
   composite difference 0.0**. Read as `drop_category revisions` it is identical too. So the row
   would publish "no difference, not distinguishable from zero" - the note's own prediction - which
   a reader takes as *the leg is not earning its 1.35 composite points* and is in fact an identity.
   The third expression the registry allows (`kind: metrics`) does move the ranking, but only by
   putting weight back on five all-NaN analyst columns.
   **The second half is blocked by the engine's own rule, not only by the missing panel:** FINRA's
   files begin 2017-12-29, which is **80 of 80** months after 2020 and **25 of 96** in the presample,
   and nothing is CONFIRMED without a presample arm pointing the same way - so a built panel could
   only be judged on 2018-2026, a `METHOD_VERSION` change that reruns everything.
   The note's research and its verdict (the weight stays; placement inside Revisions is a synthesis
   question) are untouched. Note section 6.
2. **Built: `tests/test_registry_can_measure_what_it_registers.py`** (11 tests). This was the
   **third** proposal refused on the same fact - 2026-10-23 was asked to put weight *on* the metric,
   this one to remove it - and each refusal was derived by hand from a comment in `backtest_v2.py`.
   The module fails the suite on any registered transform that (a) leaves v2's configuration
   unchanged, (b) weights a column `pit_record` classes `no_free_history`, or (c) leaves nothing
   measurable carrying weight. `baseline` and `whole_market` are exempt **by name with their
   reasons**, and a test fails if a third kind joins them. All four refused entries are fixtures, so
   a guard that passes on what it was built to catch is impossible.
   **Mutation-checked**: each of the three refused entries appended to a copy of the registry makes
   the module fail (2 of 11 each time), and the registry is restored.
   **And measured: before today nothing in the suite checked this.** With the entry registered and
   the new module excluded, the full suite reports **2 failed, 3,906 passed, 1 skipped** - and
   neither failure is a check of the arm: both read `summary.json`'s **count** of registered
   alternatives (a drift tripwire, and the methodology page's quote of it), so both go green the
   moment the registering session runs the engine and regenerates the page, which the rules require
   it to do. Same distinction the 2026-10-31 pass drew: a mutation can read as caught and still ship.
   A runtime refusal in `v2_experiments.run` would be better and is deliberately **not** built -
   `v2_experiments.py` is in `RESULT_PATHS`, so the edit stales all 32 ledger rows for a ~20-minute
   rerun whose numbers provably cannot change. Queued in `plan/research-engine.md` item 6 for a
   session already paying for a rerun, with the helper to move named.
3. **Proposal decided - `research/2026-11-03-following-the-smart-money.md` section 3: REFUSED as
   written, on a defect in how the context layer would count it rather than on the note.** It
   proposed logging each stock's 13F holder count and its quarterly change. A 13F reading changes on
   **four filing dates a year** (45 days after each quarter end) and is the same cross-section in
   between; `context_eval` counts effective observations over **spaced run dates**, which is right
   for a signal that moves daily and blind to one that does not. Measured against the data loop's
   own weekday cadence (`research/measurements/2026-11-05-quarterly-signal-accrual.py`): the gate
   would read **13 effective observations a year against 4 genuine vintages (3.25x)** and **25
   against 8 over two years (3.12x)**. That is the 2026-08-10 independence defect - measured at
   2.35x then - in a form `_effective_observations` **cannot** see, because the overlap is in the
   *signal* rather than the window and nothing in the log says when the reading last changed. On the
   one number rule 4 and priority 0 exist to protect. So the counting changes **before** such a
   signal is logged: recorded in `plan/context-layer.md` under "Things not to do", and the note's
   section 4.5 pre-registers it as condition 1 (condition 2: the identifier route settled without
   OpenFIGI's unread terms, match rate measured).
   Counted honestly the decision point is **2028-08-14** - eight vintages - for a signal the note's
   own section 1 finds absent in large companies after the lag.
   **Two things measured beside it.** `heldPercentInstitutions`, already in the fetch, is not a
   substitute and must not be logged as one: median **0.9167**, **91.1%** of the universe above
   0.70, and **24.5% above 1.00**, so it is not a share of the company at all. But it does **not**
   settle the note's "little to sort on" for the proposal's own construct - breadth is a *count*,
   and this project's own 2016 Q1 13F probe found a median of **598 filers per security**. That
   reading is left open, which is weaker than the note's.
4. **Carried out the 11-04 evening session's Next: the first rate-limit position record.**
   `runs/7bf31db85519/meta.json`, `phase_detail.fetch.tickers`, plus the sidecar the same session
   shipped. 38 rate-limited, positions **352-389 inclusive - exactly 38, perfectly contiguous**
   (`max - min + 1 == len`), OTIS..RF, one error string for all 38 ("Too Many Requests"), 38 of 38
   recovered in a 120.9 s retry pass. The sidecar shows the first backoff after **batch 12 of 17**,
   the throttle stepping 3 workers/3 s -> 2/5 s -> 1/7 s, and **no further rate-limited ticker** in
   batches 14-17. So for this night it is a **wall at a cumulative request count** (~4,200 requests
   at section 3.1's 12 per ticker) rather than a burst or a scatter - of the three readings section
   8.7 set out, the one pointing at pacing the whole pass. **One night is not the evidence**; section
   8.7's week stands and nothing moved. `research/2026-10-26-data-run-timing.md` section 8.8, with
   the four facts to record each morning.

5. **Found and fixed a live defect the decisions walked into: the research engine's published
   results had been produced by code that no longer existed, and nothing could see it.** While
   checking what the next session's build would cost (the build order's step 2 edits
   `factor_engine.py`, which is in `v2_experiments.RESULT_PATHS`), the claim in `CLAUDE.md` that
   such an edit "stales every ledger row, and
   `test_the_baseline_weighting_is_hashed_separately_from_the_transformed_one` stays red" did not
   reproduce: with `factor_engine.py` edited, `tests/test_v2_experiments.py` reports **57 passed**.
   `base_config_hash` is computed from `config.yaml`'s **parsed** contents, so that test is blind to
   a comment there and to all seven other result-affecting files; the other provenance test asserts
   the newest rows agree **with each other**, which is a different fact.
   **So I asked the tooling instead: `python scripts/rerun_experiments.py --status` reported
   `STALE REMAINING: 32` on a green `main`.** All 32 rows had been stale since **2026-11-03**, when
   `universe_history.py` was edited by the universe phase-1 slice (commit f02cf1b), and the 11-04
   and 11-05 data runs both published `research.html` from them with every gate passing.
   **Rerun and levelled**: `--jobs 4`, **15.5 minutes**, `STALE REMAINING: 0`. The numbers did not
   move, as predicted before it ran (that file is read when the monthly records are *built*, not
   when they are ranked, and `content_hash` was identical): **32 ids, 27,225 statistics, 0
   differences** (`research/measurements/2026-11-05-ledger-rerun-moved-nothing.py`). `summary.json`
   and `research.html` are byte-identical; `RESULTS.md` moved two provenance lines.
6. **Built the check that would have caught it, with its blast radius chosen deliberately.**
   `tests/test_ledger_is_level_with_the_code.py` (7 tests) compares the newest row per id against
   the live `code_hash`, `content_hash`, `base_config_hash` and `method_version`, reading
   **`scripts/rerun_experiments.py`'s own `current_provenance`** so the suite cannot be green while
   `--status` says stale - one definition, which is what was missing. A fourth test proves the
   fingerprint really sees each of the eight `RESULT_PATHS` files, by hashing a copy of the tree
   with one file changed, for each file in turn.
   **Mutation-checked**: with `universe_history.py`, `factor_engine.py` or `backtest_v2.py` edited,
   the new module fails (1 of 7) and `tests/test_v2_experiments.py` passes (57) - every time.
   **It is deliberately *not* in the 02:00 publish gate**, and a test fails if a claim ever puts it
   there. `tests/test_v2_experiments.py` is named by claim `overview.research_engine`, so a failure
   in it **discards the night's data**; a stale ledger is a code session's missed rerun and must
   block that session's merge (gate 1) rather than throw away evidence the data loop collected.
   `CLAUDE.md` priority 3 and `plan/research-engine.md` rule 7 now say what was measured.

### Evidence / research

- No new literature was read; two notes' literature was reviewed and **not** re-opened. The
  decisions rest on measurement of this system: 176 monthly records, 67,113 stock-months, the
  registry's own transform kinds, `context_eval`'s counting, and the live run's raw fetch.
- The two sources the decisions lean on were already in the repository: the FINRA start date
  (`research/measurements/2026-10-23-finra-short-interest-probe.py`, 2017-12-29) and the 13F data
  set's own numbers (`research/2026-10-20-free-survivorship-sources.md` section 5(d): 2016 Q1,
  median 598 filers per security, issuer-name matching 118 of 172).
- **One finding reported rather than worked**, because it belongs to the claims audit:
  `computed_sentences()` walks `ast.JoinedStr`, so a computed figure the generator **hard-codes**
  is outside the sweep by construction and `KNOWN_OPEN` being empty says nothing about it. The
  instance: `research_page.py:969` tells the public the engine omits "about a sixth of the live
  weighting". Measured **15.85 composite points, 1/6.31** - true today, read by nothing, and it
  would go false silently on a weight change. Queued as item 4 in
  `plan/claims-mutation-audit.md`'s next list, with the route in and an instruction not to close
  the class by registering the one sentence.

### Methodology changed

- **None.** No weight, metric, threshold, published sentence or production line moved. Both
  decisions were refusals of registrations and of a new signal; the only code added is a test
  module.

### Tried and rejected

- **Registering the short-interest experiment anyway, on 2018-2026 only.** Ruled out by the
  registry's own confirmation rule plus `METHOD_VERSION`: changing how an experiment is judged
  reruns every row, and the presample arm would measure coverage rather than the metric for the 71
  uncovered months.
- **Logging `heldPercentInstitutions`'s change as a cheap stand-in for 13F breadth.** Rejected on
  construct grounds (breadth is a count, not a percentage) *and* on the field itself - 24.5% of the
  index above 1.00 means it is not a share of shares outstanding.
- **Folding the three registry conditions into `v2_experiments.run` today.** Correct destination,
  wrong session: it stales 32 ledger rows for a rerun that cannot change a number. Queued.

### Next

- **Build the implied-volatility inversion** (the accepted ninth proposal;
  `research/2026-11-03-open-source-quant-stack.md` section 1d has the order, `exDividendDate` first).
  It is the one open build with a decision already behind it, and the proposal queue is now empty,
  so nothing outranks it. **Budget two things the note does not mention:** step 2 edits
  `factor_engine.py`, which is in `RESULT_PATHS`, so it forces a full ledger rerun (15-30 min at
  `--jobs 4`) and `tests/test_ledger_is_level_with_the_code.py` stays red until the ledger and the
  report are committed; and the step needs a throwaway full run to verify the fetch schema, which
  is a second Yahoo fetch in a day - note in `research/2026-10-26-data-run-timing.md` section 8.8
  that it removes tomorrow's 02:00 run from the clean-control set for the rate-limit record.
- Keep adding a row to `research/2026-10-26-data-run-timing.md` section 8.8 each morning: position
  range, contiguous or not, the error strings, the batch the first backoff follows. Six more nights.

## 2026-11-05 (owner-run, morning) - "the market backdrop's visuals could do more, explain more"

**Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-05_060000.log`
(shipped to main, `good/2026-11-05`). Data loop published? yes - `logs/datarun-2026-11-05_020000.log`
(Data loop complete; HEALTH: PASS). Evidence base at `1m`: **44 rows, newest run date 2026-10-05,
5 effective observations** (31 days, bound 40). Priority 0: `allow_auto_apply` **false**. Top open
roadmap item: **0.13, The Factor Lab**, opened 2026-10-12 - **24 days**.

**Owner request, in his words:** *"work on the ui and ux of the market backdrop dropdown ... it is
not very informative ... The visuals i mean, the words are good. But the visuals could do more,
explain more."* Owner-run design work; the standing "nightly sessions do not take on design" rule is
about the scheduled sessions and is unchanged.

### What the panel showed before (looked at live, 1280px)

Five reading cards that were text plus a state pill; ten series cards each with a 120x34 spark line
(no axis, no dates, no reference line, no latest point) and a 2px tick on a bare line for the
ten-year percentile, with no low, high or median shown. Nothing on the page let a reader see *why* a
reading said "Tight" or "Positive", or how far today was from normal.

### Did

- **A gauge on every reading** (`mkGauge`). The zones that set the reading's word, today's value on
  them (white line, value above), the ten-year median (tick beneath), the cut values beneath, and
  the ten-year low and high. The VIX's 2020 high (82.7) would squeeze every zone into a corner, so a
  long tail is cut past the last cut and the end label says the high is "off this scale". Credit is
  judged on its percentile, so its gauge runs on the percentile scale. Caution words light amber,
  the others the accent - colour for meaning, as the state pills already did.
- **One table for the word and the picture.** The five reading cuts moved out of inline comparisons
  in `market_context.readings()` into `READING_BANDS` + `band_state()`, and the generator writes the
  same table into the page (`MK_BANDS`). A test pins the old comparisons at both sides of every edge,
  so **no published word changed**; another checks the page is built with the table itself.
- **Real charts on the ten series** (`mkChart`): three years, the ten-year median as a dashed line,
  zero for the two curve spreads with the inverted part shaded amber (the 2022-24 inversion the
  curve sentence calls "the open case" is now visible), the 2% goal on CPI, start and end months,
  the latest point, and a hover/drag tooltip that reads any week off the line.
- **The ten-year range bar** now prints its low and high at the ends and the median under it; the
  marker still sits at `pct_10y`, so the middle of the bar is the median - the page says so.
- **Grouped** the ten cards as Interest rates / Risk appetite / The economy, each with a one-line
  plain-words description; the sector rate chart now labels its sides.
- Registered: the range bar and the gauge are in claim `context.macro_series`'s `asserts`, with
  three checks from the new module, so they run in the 02:00 publish gate. The browser checks assert
  relationships only (lit zone == the reading's own word, today inside it, marker == its own
  percentile) - no date, level or state is named (rows "data->red", "clock->red").
- `tests/js_sweep.py`: classified the new helpers (`mkAt`, `mkCut` number; `mkG` geometry; the rest
  dates or markup). The register reports no new unchecked computed sentence.

**Verified in a browser** (the worktree's page served locally, both published payloads): desktop at
1280 and 1024, phone at 375 (no horizontal scroll, `scrollWidth` 375), hover tooltip, no console
errors. Two defects found by looking and fixed before shipping: today's line struck through the zone
name it sat in (the name now moves to the other side of its zone), and the median tick inside the
bar did the same (moved beneath).

**Payload byte-identical** (`scripts/rebuild_page.py`); `index.html`/`dashboard.html` rebuilt.

### Methodology changed

- **None.** No weight, metric, threshold or published word moved. `market_context`'s cuts were
  moved, not changed, and a test proves it value by value.

### Next

- Nothing queued from this. If the owner wants more here, the obvious next step is a longer window
  (the payload carries three years of points; the ten-year range is summarised, not drawn).

## 2026-11-05 (owner-run, late morning) - "displayed to tell a story so people can understand"

**Health (rule 8):** unchanged from this morning's owner-run entry an hour earlier (code session
shipped `good/2026-11-05`; data loop PASS; `1m` evidence **44 rows, newest 2026-10-05, 5 effective**;
`allow_auto_apply` false; top open item **0.13, The Factor Lab, 24 days**).

**Owner request:** *"keep trying to make improvements on the display of things ... it needs to be
displayed to tell a story so people can understand. go and look for improvements in that area."*

### How the work was chosen

Walked the live page at 1280px, every section open, plus a stock sheet. The drilldown already tells
its story well (score tiles, How it adds up, rank history, analyst range). The gaps were all of one
kind: **the section's main point lived only in a sentence or in a closed table**:

- What changed: "Growth accounted for 21%, Valuation 20% and Quality 17%" - the eight shares were in
  the payload and never drawn.
- Track record: "5 of 8 periods ahead of RSP" - *which* periods, and by how much, was a closed table;
  and the caveats were a single ~250-word paragraph that buried its two important sentences.
- Factor analytics: the 2026-10 redesign moved the matrix to ink-only shading, so a standout sector
  had to be found by comparing numbers by eye.
- A defect seen on the way: the analyst-range bar cut its own end label ("ow $361.00").

### Did

- **What moved the scores** beside the overview sentence: all eight published shares as bars,
  largest first, the three fundamentals in the accent (the set is `run_overview.FUNDAMENTAL`, written
  into the page, so the bars and the sentence's "together 58%" use one definition).
- **Period by period** under the track chart: one bar per holding period, top 25 minus RSP, green
  exactly where `track_record` counts a period ahead (`top > RSP`; a tie is red, tested), captioned
  with the published count. **Caveats folded:** the luck sentence and "this record is not used to set
  the methodology" stay visible; the method notes move into *How the record is measured*. Every
  sentence is unchanged except one plain closing line added inside the fold so the sentence sweep still
  sees the cost sentence (*"The record shows what the ranking did, not what a reader should do."*).
- **Sector standouts:** each factor's single highest sector in the accent, single lowest underlined,
  with a key; a tie marks nothing.
- **Analyst range labels** read inward from their ticks; checked geometrically.
- Registered: the shares test on `history.run_overview`, the green-count and tie tests on
  `context.track_record`; the sentence register (`DASHBOARD_SENTENCES`) re-keyed for the split
  caveats. All checks assert relationships, never values; the target-label check opens its stock
  through `tests/examples.py`.

**Verified in a browser**: desktop 1280 and phone 375 (no horizontal scroll), no console errors.
Payload byte-identical (`scripts/rebuild_page.py`).

### Methodology changed

- **None.**

### Next

- Candidates seen and not done: Reporting soon is a plain table (the implied move could be drawn as a
  range, but its "Last quarter" column is an EPS surprise, not a price reaction, so there is no honest
  pairing to draw yet); the Diagnostics correlation shading is by magnitude only, which is the right
  question there (independence), so it was left.

## 2026-11-05 (owner-run, midday) - "professional and expensive": nothing cut short, no holes, one false 100%

**Health (rule 8):** unchanged from this morning's owner-run entries (code session shipped
`good/2026-11-05`; data loop PASS; `1m` evidence **44 rows, newest 2026-10-05, 5 effective**;
`allow_auto_apply` false; top open item **0.13, The Factor Lab, 24 days**).

**Owner request:** *"keep finding improvements with the presentation and front end please. Make
sure it is looking professional and expensive."*

### The audit (measured, the generator's page on the published payloads)

At 1440, 1280, 1100, 1000 and 375px: every leaf element whose text is wider than its box, the type
sizes in use, the corner radii, and digits not set in tabular figures. Radii were already a clean
two-level system (14px sections, 8px cards inside). What read as unfinished was text cut short and
an empty grid slot:

- **leaders, 761-1100px:** three cards and two with an empty slot;
- **leaders, 1280px:** "Valero Energy Cor...", "Marathon Petroleum Cor...", "Micron Technolog...";
- **trap rate by sector:** five sector names cut ("Consumer Discr...") at every desktop width, and
  below ~1240px the sector table scrolled sideways inside its panel;
- **Market, phone:** three sector names cut in the rate chart.

**And one false statement found by reading a sheet:** Expedia, ranked 3rd of 499, said *"ahead of
100% of the other 498 stocks"*. 496/498 is 99.6%; whole-percent rounding turned it into "100%",
which reads as first. Ranks 2 and 3 have printed that every run.

### Did

- **Leaders:** a 6-column grid from 761 to 1100px (three cards, then two sharing the row); the
  company name wraps to two lines and every card keeps two lines' height, so the five sets of bars
  start level (measured: one top value).
- **Trap rate by sector:** name and figure on one line, the bar beneath at full width; below 1240px
  the sector table and the trap bars stack at full width, the bars in two columns.
- **Market rate chart on a phone:** the same stacked row.
- **Tabular figures** on the mover lines, KPI sub-lines, chart reference labels and the
  diagnostics badges.
- **`stock_summary._share_text`:** whole percent, except a share that is not 100 or 0 but rounds
  to one takes a decimal (99.8%, 99.6%; 0.2% for the second-last), cut toward the truth if one
  decimal still rounds to the extreme. The page's near-zero rule, applied at both ends. Claim
  `summary.rank` updated; `tests/test_claims_register.py::test_rank_share_is_exact_for_every_published_stock`
  now checks the decimal case and refuses an untrue extreme; 8 hand-worked edges in
  `tests/test_stock_summary.py`. **This sentence is baked at run time, so the live page shows it
  after the 02:00 run.**
- **Gate 5 caught a test asserting the false sentence.** `tests/test_dashboard_browser.py::test_the_rank_sentence_is_true_for_the_top_stock`
  opened EXPE by name - #1 when it was written, #3 today - and asserted "ahead of 100%", which the
  rounding made true on the page. It now opens whichever stock is first. Its four sibling tests
  open `examples.stock("EXPE")`, and the tripwire in `tests/test_browser_examples.py` now matches
  `_open_stock(page, "X")`, the form that let five named opens in a publish-gate module through
  yesterday's sweep.
- `tests/test_layout_polish.py` (8 browser tests, geometry only): **7 of 8 fail on the page as it
  was this morning**, all pass now.

Payload byte-identical (`scripts/rebuild_page.py`).

### Methodology changed

- **None.** The rank-share fix changes how one figure is printed, not how anything is computed.

### Next

- The page uses **21 font sizes**, five of them half-pixel steps (10.5, 11.5, 12.5, 13.5, 14.5). A
  tighter type scale would look more deliberate; it touches most of the CSS, so it wants a session
  of its own with before/after screenshots of every surface.
- On a phone, long company names on the What changed mover lines end in an ellipsis; the ticker is
  beside them, so it was left.

## 2026-11-05 (owner-run, early afternoon) - "keep finding improvements! make something great"

**Health (rule 8):** unchanged from this morning's owner-run entries (code session shipped
`good/2026-11-05`; data loop PASS, morning brief published 02:21; `1m` evidence **44 rows, newest
2026-10-05, 5 effective**; `allow_auto_apply` false; top open item **0.13, The Factor Lab, 24 days**).

**Owner request:** *"keep finding improvements! make something great."*

### The gap

Walked the page at 1440 and 375px as a first-time reader. The story it tells runs: what is ranked
(the hero and the dot plot), who leads (the leader cards), what changed, has it worked. The question a
newcomer should ask between the second and third - **is this leader a fluke?** - had no answer on the
cards, and My Holdings, the sell-side surface, was all prose: no picture of where a held name had been.
Both answers were already in the first-paint payload (`history.series`, every stock's published rank in
every comparable run), so this is presentation only and the payload is byte-identical.

(A false alarm on the way, recorded so nobody chases it: screenshots taken straight after a scripted
`scrollTo` showed the nav and the sheet's tabs lit one section behind. Measured after a frame, both
scroll-spies are right - the screenshots were taken before the next animation frame.)

### Did

- **Leader cards: how long each has led.** Under the eight category bars, one column per run for the
  last 20 comparable runs: height is the published rank on a #1-#50 scale, accent where it ranked in the
  top 25, a dashed line at #25, and "Top 25 in N of 20 runs since Oct 7". Today it reads as the data
  does: Expedia #1 for 17 straight runs, Micron climbing from #48 to #4, Marathon Petroleum #26 the run
  before it jumped to #2. The key under the cards says what the columns are.
- **My Holdings: where each name has been.** A small rank line in each card's header: every comparable
  run, #1 at the top and the whole universe at the bottom, the same scale on every card (so a drift from
  #40 to #60 looks as small as it is), a dashed line at #25, and the end dot coloured only when the
  change beside it clears the material threshold - the movers' rule, so the colour never asserts more
  than the measurement does. At 800px the extra width pushed the composite onto a row of its own; the
  line, the change, the score and the remove button now wrap as one group, and on a phone the line
  spans the card beneath them (the end dot is a zero-length stroke, so it stays round when stretched).

### Checks added

- `tests/test_dashboard_printed_figures.py`: every stock's leader caption and columns recounted from its
  own series (the window, the count at <= 25, the date, filled vs other columns); a hand-worked window
  with an unranked run, a rank exactly on the line and a history longer than the window; every stock's
  holdings line point-by-point against its published ranks on the fixed scale, the #25 line, and the
  dot's colour against the material threshold. Claim `history.recent_ranks`, so the 02:00 publish gate
  runs them.
- `tests/test_layout_polish.py`: the leader caption on one line and the columns level (1440, 1280,
  375); the holdings group never split and no sideways scroll (1440, 1100, 800, 375).
- Mutation-checked: `<= 25` turned into `< 25` fails both leader tests; the holdings group made
  `display: contents` fails the layout test at 1440, 1100 and 800.

### Left

- The leader columns use a #1-#50 scale, so day-to-day moves inside the top ten look small. That is
  deliberate (they are small: the material threshold is 37 ranks), not an oversight.
- Still open from earlier today: the 21-size type scale, and mover names ellipsizing on phones.

**Caught on the live page after shipping (`good/2026-11-05-owner-4`):** at 1280px the leader caption
broke inside the date ("since Oct / 7"). The layout test had asserted the caption's height and passed,
because the test browser has no web font and its fallback is narrower than Inter. Fixed by splitting the
caption into two halves that never break inside themselves (the date drops to its own line, whole, on
all five cards at once), and the test now pins that rule instead of measuring a height the font decides
(mutation-checked: without the rule it fails at 1440, 1280 and 375). The general lesson is already in
memory - a geometry test in a font-less browser cannot see a wrap the real font causes; pin the rule.

## 2026-11-05 (13:00) - QUEUE. A red baseline came first, and it turned out to be a real defect

**Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-05_060000.log`
(shipped to main, `good/2026-11-05`); four owner-run sessions followed it, `good/2026-11-05-owner-1..4`.
Data loop published? yes - `logs/datarun-2026-11-05_020000.log` (Data loop complete; HEALTH: PASS).
Evidence base at `1m`: **44 rows, newest run date 2026-10-05 - 31 days ago, bound 40, 5 effective
observations of the 8 the engine's gate needs.** Priority 0: `allow_auto_apply` **false**. Top open
roadmap item: **0.13, The Factor Lab**, opened 2026-10-12 - **24 days**; also open 0.10 (29d, opened
2026-10-07) and 0.14 (22d, opened 2026-10-14). No branch held back by the gates. `git stash list`
still carries the one `auto-rescue 2026-10-28_060001` entry that `git stash show` reports as empty -
left alone, as five prior sessions did.

**Tests:** before **3,976 passed / 1 skipped** (parallel, 3:02 - my own run; the runner's baseline at
13:03 reported 1 failure, see below), after **3,982 passed / 1 skipped** (parallel, 3:04). The six
new tests are 1 in `tests/test_gate_ordering.py` and 5 in `tests/test_sheet_warm_is_settled.py`.
All five gates pass on this tree.

**Owner queue / rotation:** `write_brief.open_proposals()` returns `[]` - no proposal waiting.
`OWNER_FOCUS.md`'s open items are the universe expansion (next: phase 1) and the 2026-10-20
fine-tune item. The 06:00 entry's **Next** was the implied-volatility inversion build. **Neither was
taken, and the reason is the first line of the prompt**: the runner reported `main` **RED on
arrival**, which outranks the focus - and chasing it down produced the session's work. See "Why not
the IV build" below; nothing was deferred silently.

### Did

1. **A red baseline now keeps its transcript, like gates 1 and 5 do.** At 13:03:29 the runner logged
   `BASELINE main was RED on arrival: 1 pre-existing failure(s)` and named
   `tests/test_context_layer_claims.py::test_the_page_says_when_each_kind_of_quote_was_taken` - a
   browser module that `claims.py` also puts in the 02:00 publish gate, where a failure **discards
   the night's data**. And nothing else: no assertion, no traceback. Gates 1 and 5 have saved their
   whole output beside the log since 2026-11-04, for exactly this reason ("the log said only FAIL
   check_published_claims"); the baseline, which the session reads *first* and which the prompt makes
   its first job, did not. Measured cost: about **40 minutes** of this session establishing that
   `main` was in fact green.
   `Save-GateOutput` gains a `$Level` parameter defaulting to `'ERROR'`, so gates 1 and 5 are
   byte-unchanged, and the baseline calls it at `WARN` - it is not a gate and decides nothing - on
   **both** red branches, including "could not be taken", the most opaque of the three states.
   `tests/test_gate_ordering.py` (1 new test, 14 total); **5 mutations, 5 caught**
   (`research/measurements/2026-11-05-baseline-output-mutations.py`), including the save moved inside
   the named-failures branch and either gate made to pass its own level.
2. **Then the flake itself, found and fixed - it was a real defect in the harness, not luck.**
   It never reproduced: 8 runs across 4 scenarios, including **two whole-suite parallel runs of all
   28 Chromium modules, 737 tests each**, all green
   (`research/measurements/2026-11-05-context-claims-flake.py`).
   The cause, measured rather than inferred
   (`research/measurements/2026-11-05-prewarm-clobbers-the-panel.py`): `generate_dashboard`'s
   `sheetPrewarm` lays the top-ranked stock's sheet out unseen once the page is idle - correct, and
   invisible to a reader, because `openStockDetail` stops it and rewrites the sheet - and it gets
   there by calling `renderSheet(top, ...)`, **which rewrites `#modal-context`**. The test's helper is
   two round-trips, a render then a read, and the fixture's own `wait_for_function` returns with
   `SHEET_WARM.state === 'wait'`. Forced, the panel comes back as **VLO's**: the synthetic sentence
   gone, `"close of"` present - **exactly the two assertions that test makes first**, and exactly the
   pair that failed at 13:03.
   Why it surfaced today is the part worth keeping: the flake needs the pre-warm to land in the few
   milliseconds between one render and its read, and the number of such windows per run rose sharply
   this morning when an owner-run session added `tests/test_dashboard_printed_figures.py`, which
   renders per-stock context **in a loop** over the whole universe.
   The fix is in the harness, since production is right: `tests/sheet_warm.py`'s `settle(page)`,
   called in the fixtures of the three exposed modules (`test_context_layer_claims`,
   `test_context_cards_printed`, `test_dashboard_printed_figures` - the three that write into the
   sheet without opening a stock; an open settles it in the page).
   `tests/test_sheet_warm_is_settled.py` (5 tests) is the tripwire: a **sweep** over every Chromium
   module, so the next one written without it fails the suite, plus the behaviour on a real page both
   ways.
   **6 mutations, 6 caught** (`research/measurements/2026-11-05-sheet-warm-tripwire-mutations.py`) -
   and the sixth is the one worth reading: with `settle` reduced to a **no-op** the module first
   passed, because its single test settled a pre-warm that had **already finished**. Splitting it so
   the helper is exercised on a *pending* pre-warm is what gave it teeth. Generalised: *a test of a
   "wait until X is safe" helper must start from the unsafe state, or it passes whatever the helper
   does.*
3. **Recorded as a third failure class in `CLAUDE.md`**, beside `data->red` and `clock->red`:
   **`paint->red`** - a test whose premise is that the page has stopped changing. The two existing
   rows cover a premise set by the day's data and one set by the clock; this one is set by deferred
   rendering, and the lesson had nowhere to live.
4. **Checked, and it is reassuring: the 02:00 publish gate has never been tripped by a flake.**
   Two `PUBLISHED CLAIMS: FAIL` in the whole datarun log history, both genuine, data-dependent claim
   failures that refused a run as designed - 2026-10-28 (`test_case_against`) and 2026-11-02
   (`test_growth_category` + `test_valuation_history`, the growth-exposure drift already in
   `OWNER_FOCUS.md`). And **4 red baselines ever recorded**, of which **2 are Chromium modules**
   (2026-11-03 `test_every_panel_reads_clean`, today's). So the risk this fix removes is real and the
   damage so far is one session's time, not a lost night of evidence. **Nothing was weakened to get
   there**: no retry, no rerun plugin, no loosened assertion - the flake's cause was found, so none
   was needed.

### Why not the IV build

The 06:00 **Next** was the implied-volatility inversion (`research/2026-11-03-open-source-quant-stack.md`
section 1d, ACCEPTED 2026-11-04). It was read in full and **not started**, deliberately, because a
red baseline outranks the focus and then the work it turned into was real. It is also the wrong size
for what was left: its own build order makes step 2 - adding `exDividendDate` to the core `.info`
fetch - "a session of its own", because a fetch-schema change needs a **throwaway full run** to
verify (~15-20 min, and a second Yahoo fetch in a day), and the edit lands in `factor_engine.py`,
one of `v2_experiments.RESULT_PATHS`, which **stales all 32 ledger rows** and forces a `--jobs 4`
rerun (15-30 min). Starting either after 15:00 would have ended the session dirty, which the settled
row forbids. Three things measured for the session that takes it, so it starts ahead:

- **QuantLib 1.44 is already installed** and imports; step 5 is a one-line pin in `requirements.txt`.
- **The dividend schedule needs more than `exDividendDate`.** The accepted measurement's
  `projected_dividends` (in `research/measurements/2026-11-03-implied-vol-quantlib-vs-yahoo.py`) takes
  the **last ex-date and the median past interval** from `yf.Ticker.dividends` - a request per stock,
  in the pass that tripped the rate limiter. From `.info` alone the frequency has to come from
  somewhere else; `lastDividendValue` with `dividendRate` would give it (`rate / value` = instalments
  a year). **Which of those three fields the source actually serves is unmeasured** - the note says
  only that they are absent from the *parquet*, which is what you would expect when the fetch does
  not ask for them. Probe `.info` before editing the fetch, as the 2026-10-14 dividend-field probe did.
- **Step 7's coverage number cannot be measured from the cache.** `data/options/quotes.json` stores
  only derived readings, no bid/ask, so "measure it on the whole universe from one 20:00 refresh"
  means instrumenting the 20:00 pass, not reading what it already wrote.

### Evidence / research

- No literature read; none needed. Both changes rest on demonstrated user-facing failures and on
  measurement of this system: the 13:03 baseline, the 8-run non-reproduction, the forced clobber, the
  2 publish-gate failures and 4 red baselines in the log history, and 11 mutations across the two
  tripwires.

### Methodology changed

- **None.** No weight, metric, threshold, published sentence or scoring line moved. The diff is the
  runner, the test harness, two tripwire modules, four measurement scripts and `CLAUDE.md`.

### Tried and rejected

- **Making the suite's browser tests more robust by changing xdist's distribution** (`--dist loadfile`,
  so a module's tests share one worker and one Chromium instead of up to N). Attractive and
  **rejected**: it is a change to gate 1's own command justified only by a guess that contention
  caused the flake. Once the cause was measured, contention turned out not to be the mechanism at all
  - the clobber happens in a single-page, unloaded run too - so the change would have bought nothing
  and obscured the real defect. Recorded in case a future session reaches for it.
- **A retry or rerun plugin on the publish-gate browser tests.** That is weakening a gate to pass it,
  and it would have hidden this defect permanently.

### Next

- **Build the implied-volatility inversion** - still the one open build with a decision behind it and
  an empty proposal queue. Start with the `.info` probe named above (cheap, no pipeline run), then
  step 2 and its throwaway full run, and **budget the 15-30 minute ledger rerun in the same session**;
  `tests/test_ledger_is_level_with_the_code.py` stays red until the ledger and report are committed.
- Keep adding a row to `research/2026-10-26-data-run-timing.md` section 8.8 each morning: position
  range, contiguous or not, the error strings, the batch the first backoff follows. Five more nights.
- If another red baseline arrives, **read `logs/nightly-<stamp>.baseline.txt` first** - it exists now.

## 2026-11-05 (owner-run, evening) - the S&P 500 fell 4.4%: will tomorrow's run publish?

**Health (rule 8):** unchanged from today's earlier entries (data loop PASS 02:16; `1m` evidence
unchanged; top open item **0.13, The Factor Lab**). No scheduled session runs before 02:00.

**Owner:** *"the market just took a beating... excited to see if everything still holds up even when a
bunch of stocks go down, and how the rankings change if any."*

**The day:** S&P 500 -4.36% (total return), the technology sector ETF -8.0%, staples -0.8%, VIX 16.6 ->
28.2. SanDisk -22%, Lumentum -19%, Micron -17%. The input gates (`check_run_health`, 2026-10-22) were
replayed only over calm days - no day since 2026-07-28 moved the index by more than ~2% - so the
first question was whether the 02:00 run's rank-continuity limits (Spearman >= 0.90, <= 8 new top-25
names, <= 3 fallen below #40) would refuse a real market move as a scrambled run. A refused run
leaves the published ranking standing, and every later run is compared with it.

### Rehearsed before it happens

`research/measurements/2026-11-05-selloff-rehearsal.py` (+ JSON): the 11-05 run's own metric table,
re-scored with the engine's own scoring functions under its own effective weights - **the control
reproduces all 499 published ranks exactly** - then every price-moved metric moved to the 11-05 close
by its own definition: valuation, size and target upside through `lookahead.restate_at_price` (EV by
the engine's own fallback where the quote has none - the first draft skipped it and dropped Pfizer
166 places for that reason alone); volatility and beta with the day appended to the engine's own
published moments; drawdown from the 52-week high (not `_mdd_peak`, which is the peak before the worst
trough); Jensen's alpha from its own CAPM terms. One bulk chart request per ticker, no `.info`.

**Result: PASS, by a distance.** Spearman **0.998** against the page the site serves, **1** new name in
the top 25 (WDC), **0** fallen below #40. Mean absolute category-score change: Risk 1.3, Valuation 1.3,
Size 0.9, Revisions 0.8, Momentum 0.6, Quality/Growth/Investment ~0.

**Why so little:** both momentum measures end a month ago (the skip-month convention, metrics 13-14),
so a one-day fall cannot reach them; and every score is a percentile within the stock's sector, so a
fall that hits a sector together barely reorders it. The stocks that fell hardest got cheaper and
further below their price targets at the same time as they got riskier - SanDisk, down 22%, stays 5th
(composite 72.3 -> 72.3), Micron 4th -> 3rd. **That is the methodology working as written, and a
teaching point rather than a defect:** the screener ranks businesses against their peers; it does not
react to the tape. A sustained fall would reach momentum a month later.

Approximations, stated: one day appended to ~250 without dropping the oldest; nothing for news that
moves analyst estimates overnight; risk-free rate unchanged. None of these can plausibly move
Spearman from 0.998 to the 0.90 limit. Not decision-grade for methodology (rule 5) and not used as such.

### Built: the market and the ranking side by side (What Changed)

The question the owner asked - did the ranking hold when the market fell? - had no answer on the page.
Now What Changed shows, for each comparison:

- **how far the S&P 500 moved between the two runs' prices** - the total-return index the run already
  fetches for the momentum rule, now saved with each run (`runs/<id>/market_closes.json`, 400 sessions,
  context only, nothing scores from it), from the close the comparison run was priced at to the close
  this one was. A run is priced at the last session whose 16:00 New York close preceded its start; a
  run started mid-session (2026-10-20, 13:21) names no close at all;
- **how far the ranking's order moved with it** - the Spearman rank correlation of the two published
  rankings (the statistic the publish gate and the history's acceptance test already use), on a
  0.80-1.000 gauge, beside the median between consecutive runs (0.996 today);
- one more sentence in the overview carrying the same two numbers.

Start times come from the snapshots' `run_started_at` (present since 2026-10-20), so the market figure
appears for "since last run" from tonight and for "~1 month" from about 2026-11-19. The correlation
appears today. **Tonight's 02:00 run is the first to save closes, so tomorrow's page should read: the
S&P 500 fell 4.4% (Nov 4 close to Nov 5 close), rank correlation ~0.998.**

**Payload regenerated from the same run** (the 2026-10-07 precedent): diffed field by field, only
`history` (`compare.*.started_at`, `current_started_at`, the overview sentence and `rho`) and four
drilldown summaries differ - those four are the "ahead of 100%" fix from earlier today (MPC now "ahead
of 99.8%"), which would otherwise have waited for tonight. `table_data`, every score and rank, and
`dashboard_context.js` byte-identical. The run folder's own market file was *not* back-filled: the run
did not write one, so today's page states the correlation and no market figure.

**Checks:** claim `history.market_move`, 8 checks in the 02:00 publish gate - session mapping (02:00,
evening, 16:00 exactly, 15:59, lunchtime, Sunday, holiday), the ratio, the Spearman, the median, the
sentence word for word, the published figures against the run's own closes file and the history's own
series, and the two printed figures in the browser (sign, colour, dot position, below-0.80 pinning,
nothing drawn without a correlation). Mutation-checked: `<=` -> `<` at the close, dropping the
same-session guard, swapping the colours and moving the gauge's floor are each caught; dropping the
"ranked on both" filter is not, because pandas' Spearman already drops the missing pairs (equivalent).

### Next

- **Read tomorrow's page and the 02:00 health log against the rehearsal**: Spearman, new top-25 names,
  the market sentence. If the run is refused, that refusal is the finding - read its reason before
  anything else.

## 2026-11-05 (owner-run, late evening) - every stock says what kind of company it is

**Health (rule 8):** unchanged from today's earlier entries (data loop PASS 02:16; `1m` evidence
unchanged; top open item **0.13, The Factor Lab**). Pushed before the 02:00 data run.

**Owner:** *"Keep finding improvements, specifically more on the ui. Focus on having it always tell a
story and is easy to understand."*

**What a first-time reader could not get from the page.** Toured the live site at 1440 and 375 as a
newcomer. Every number is explained somewhere - the key, the workings, the sentences - but nothing says
in words what *kind* of company the eight scores describe. A leader card read "Momentum 98, Valuation
90, Size 21"; to learn that this is a company with a strong price trend, priced low for its profits and
larger than most of its sector, a student had to translate each number through the key, one at a time,
and remember that Size runs backwards.

**Built: Stands out / Trails.** For every stock, the categories scoring **70 or more** (highest first,
at most three) and **30 or less** (lowest first, at most two), each as a short phrase in the score key's
own direction (`generate_dashboard.TRAIT_WORDS`; "Lower price for its profits", "More volatile price",
"Larger company"). 70 is the line the leader cards already highlight a bar at; 30 mirrors it about the
sector median. Shown on the five leader cards, between the name and the bars (the key under the cards
says what the two words mean), and opening every stock sheet's Scores, where each chip also carries its
category and published score ("Momentum 97.6 - Stronger price trend") so the word and the number are
read together. **My Holdings cards carry the same strip** under their eight scores, so a holder reads
what kind of company they own before the run-to-run sentences (Apple: stands out for a calmer price and
profitability; trails as a larger company with weakening analyst views). A stock with nothing at either
end says so. Display only: it reads the published scores
and changes no score, rank or weight, so no payload change (`scripts/rebuild_page.py`, payload
byte-identical) and no changelog entry.

**Checks:** claim `dashboard.traits`, 9 tests in `tests/test_stock_traits.py` (in the 02:00 publish
gate through the claim): the JS rule against the Python one on all 499 published stocks; each leader
card's words against its own scores; the sheet's chips (category, score as the page prints it, words)
through `openStockDetail` for the first, median and last stocks; the hand-worked boundaries (70 and
30 count, 69.99 and 30.01 do not), ties and unscored categories in the browser and in Python; the
words against the score key's direction; no advice term and no digit in any phrase. Mutation-checked:
`>=` -> `>`, the sort direction, swapped words, a rounded score on the chip and a reversed Risk pair are
each caught; dropping the tie-break is equivalent (the list is built in the key's order and JS sort is
stable).

**Seen:** desktop and 375px, leader cards and two sheets (#1 VLO; #496 WMB, which has nothing at 70 or
more and says so). No horizontal scroll at 375, no page errors. The bars stay aligned across the five
cards whatever the number of chips. Holdings checked with VLO, AAPL and WMB saved, both widths.

### Next

- Tomorrow: read the 02:00 page and health log against the sell-off rehearsal (previous entry).
- ~~The same strip in **Compare**~~ - done 2026-11-06, next entry.

## 2026-11-06 (owner-run, 00:00) - Compare says what sets each stock apart

**Health (rule 8):** unchanged from 2026-11-05's entries; pushed well before the 02:00 data run.

**Owner:** *"Do that now. Do everything you can before the 2 am run."* - the Compare view, named as the
next place for the Stands out / Trails strip in the previous entry.

**Built:** Compare (Side by side) gains a **What sets it apart** row directly under Composite, one cell
per stock, the same `traitsHtml` strip as the leader cards, so two stocks are told apart by kind before
the eight rows of numbers (VLO: stronger price trend, lower price for its profits; AAPL: calmer price,
trails as a larger company with weakening analyst views; WMB: nothing at 70 or more). Chips that wrap
on a phone get a 10px corner rather than a pill. Display only; payload byte-identical.

**Checks:** `tests/test_stock_traits.py::test_the_compare_view_carries_the_words_for_each_stock`
(added to claim `dashboard.traits`): the first, median and last stocks opened together, each column's
words against its own published scores and in the view's rank order. Seen at 1440 and 375 with three
stocks; no horizontal scroll, no page errors.

**Caught by gate 5, and fixed before 02:00: a flake the previous push introduced.** The rehearsal failed
`tests/test_club_readiness.py::test_a_phone_reader_reaches_the_flag_definitions_from_the_tile` - on a
phone, "What a flag means" scrolls to the flag definitions and the test read their position at a fixed
700 ms. Measured: **0 of 10** failures on the page before last night's Stands out / Trails change, about
**1 in 3** after it (1 of 6 on the live `main`, 2 of 5 here). Cause, measured frame by frame: the leader
cards grew a line of words, the phone's scroll to the ranking got ~400px longer, and at 700 ms the smooth
scroll was still in flight (y = 703-806 against a 812 viewport); it landed at **y = 352 every time by
1.2 s**. The reader was never affected; the test's premise was. **This module is in the 02:00 publish
gate** (claim `dashboard.score_key`), so left alone it could have discarded tonight's data run on a bad
roll. Fix: read the position once the page has stopped scrolling (ten unchanged frames) - the same final
assertion, the premise of settled row paint->red. 0 of 15 failures after. It was the only test in the
suite reading a position after a fixed wait.

## 2026-11-06 - THE FACTOR LAB (deferred: the data loop was not publishing)

**Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-05_130000-midday.log`
(completed). Data loop published? **no** - `logs/datarun-2026-11-06_020000.log` (stopped
deliberately; HEALTH: PASS, the publish gate refused). Evidence base at `1m`: 5 of 8 needed at the
1m horizon (44 rows, but overlapping windows are not independent; 112 rows across all horizons),
newest 1m run date 2026-10-05 (**32 days ago, bound 40** - inside it). Priority 0:
`allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab** (age **25 days**);
also open: 0.10 30d; 0.14 23d.
**Tests:** before 4011 passed / 2 skipped (green baseline, nothing inherited), after 4020 / 2.
**Owner queue / rotation:** no open proposal (`write_brief.open_proposals()` -> `[]`); no owner
item taken. **Friday's lab focus deferred** - the data loop had stopped publishing, which outranks
it (CLAUDE.md, "The two loops"). No held-back branch in the health line.

### Did

**The data loop was discarding its run every morning's work, and the live site was a day stale.**
The 02:00 run passed every health check (`HEALTH: PASS - safe to publish`) and was then refused by
`scripts/check_published_claims.py` on two tests. Both are now fixed, and
`python scripts/rehearse_publish.py` **PASSES** on the page the next run will publish - it was red
on both counts before. Measured in `research/measurements/2026-11-06-publish-gate-failures.py`,
which runs each failure against the discarded run's payload and the committed one side by side.

**1. The drilldown printed an equality its own figures refute** - the substantive one, and a live
defect on the public site rather than only a test problem. `price_target_upside` is clamped to
-50%..+100% by the scorer and its Calculation line is *exact*, so for a stock on a bound the row
read `($2023.63 mean target - $806.43 price) / $806.43 price = 100.0%` where the arithmetic gives
**150.9%**. That is precisely what priority 0.8 exists to stop (owner, 2026-10-06: every number on
the page must be checkable from the page). **11 of 496 stocks** sat on a bound in the 11-06 run
(CIEN, COHR, FLEX, LITE, MRVL, MU, NRG, ORCL, SNDK, STX, WDC) and **2 of 496** on the page that is
live right now (NRG, FLEX).

The row now shows the cap as a step - `= 150.9% capped to 100.0%`, bounds in the title - so the
line is true and the clamp is taught where it applies instead of only in the definition beneath.
The uncapped figure is computed once in the build (`_capped_equations` -> `stock_detail[t].xcap`),
only for the stocks whose bound binds, and the page displays it rather than re-deriving it (0.8).
`metric_lineage.CLIPS` is pinned against `config.yaml`'s `metric_clamps` in **both directions**, so
a third clamp cannot reach the scorer without the page learning about it; a clamped metric has
exactly two honest presentations and no third (exact + published bounds, or prose that says the
value is clipped - which `forward_eps_growth` already was).

**Looked at it** on the regenerated page (`logs/shot_capped_row.py`), desktop 1440 and 375px, SNDK
capped and VLO uncapped. Desktop: one line, no overflow. 375px: wraps to two, `150.9%` in white and
`100.0%` in blue, "capped to" in the muted label colour - so the emphasis stays on the scored
value. The uncapped row is byte-unchanged (`= 21.6%`).

**2. The holdings-strip test read a rank off a list position.** It sorted the ranked rows, took
`rows[299]`, and asserted the strip says "#300". The composite is rounded to two decimals before
ranking, so **ties are ordinary and the ranks are not contiguous**: that row was #300 on 11-05 and
**#299** on 11-06. The page was right both mornings. Its premise assertion did not cover it -
`n == len(rows) == rows[-1].Rank and rows[0].Rank == 1` are all true with ties present. Ranks now
come from the picked rows, with an assertion that the three stand-ins are distinct. Swept the other
test modules for the same shape: `tests/test_page_fourth_pass.py` already guards it
(`r.Rank === 300 || r.Rank > 250`); nothing else maps a position to a rank.

### Evidence / research

- No literature needed: both are demonstrated, reproducible, user-facing failures, which is an
  acceptable ending under "I know this is an improvement because ___" (a concrete failure I can
  demonstrate; a failing test that now passes).
- The arithmetic is measured, not argued: equation-vs-scored agreement **485/496 = 0.9778** on
  2026-11-06 against a 0.99 bar, and **494/496 = 0.9960** on 2026-11-05 - the bar was held the
  morning before **by two stocks**. Every one of the 11 failures sat exactly on a clamp.
- `scripts/rehearse_publish.py`: `PASS pytest (page tests)`, `PASS check_published_claims`,
  `PASS published files restored byte-for-byte`.

### Methodology changed

- `METHODOLOGY_CHANGELOG.md` 2026-11-06, marked plainly: **no weight, metric, threshold or score
  moved** - the clamp is exactly what it was. What changed is a published sentence, so it is
  recorded rather than left to the commit message.
- `claims.drilldown.row_equation` asserted "...gives the value that was scored" full stop, which
  **was the false sentence**. It now describes the clipped form and names nine new checks. (The
  recurring lesson from the mutation audit again: when you correct a published claim, grep the
  tests as well as the prose - here the claim itself asserted the defect.)

### Tried and rejected

- **Handling the clamp inside the test and leaving the page alone.** Rejected: `prompts/nightly.md`
  section 3 forbids loosening the 99% bar or flipping a template to `exact=False` to get past it,
  and more to the point it would have left a false equality on the public site for the 2-11 stocks
  a day that hit a bound. The check was right that something did not add up; it was wrong about
  which side was broken.
- **A payload-reading test for the new `xcap` field.** Written, then rejected and rewritten: only a
  full run rewrites `dashboard_data.js`, so it failed gate 1 against the committed 11-05 payload
  while passing the rehearsal - the exact split rule 10 warns about ("assert claims against the
  generator's output, not the committed file"). It now reads `_capped_equations` directly, plus one
  end-to-end test that regenerates from the newest run to check the wiring.
- **A tripwire for "a test must not map a list position to a rank".** Considered and not built: the
  honest versions are either a brittle regex sweep or a test asserting the published ranking *has*
  ties - which is itself a premise that could flip on a quiet day, i.e. the same bug one level up.
  The lesson went into `CLAUDE.md`'s `data->red` row instead.

### Next

**The Factor Lab, 0.13 (age 25 days, deferred today and the reason is written above).** The next
Friday that is not a retrospective owes it one finished experiment, tolerances committed before the
comparison runs; `plan/factor-lab.md` has the ordered queue. If a session reaches it sooner, the
other standing item is experiment 19's second reader.

## 2026-11-06 (owner-run, late morning) - the page says how old its prices are

**Health (rule 8):** as this morning's 06:00 entry, with one change: the 02:00 data run was
**discarded** by the publish gate (two failures, both fixed and shipped by the 06:00 session), so the
live page is still the 2026-11-05 run. On the owner's word ("Yes, after close") a one-time task,
`Screener Data Run catch-up 2026-11-06`, runs `scripts/data-run.ps1` at **16:00 CST** today, after
the close, so the site does not stand on Thursday's prices all weekend; it is deleted once the run is
read. `1m` evidence, Priority 0 and the top open item (0.13, The Factor Lab, 25 days) unchanged.

**Owner:** *"keep finding improvements and updates you can make!"*

**What the 02:00 run showed, which the rehearsal had predicted:** HEALTH PASS on the first run after
the S&P 500's 4.4% fall - Spearman **0.987** against the published ranking (rehearsal 0.998; the real
run also carries a day of estimate and filing changes), **2** new names in the top 25 (WDC, EME), **0**
fallen below #40. The What Changed sentence read "the S&P 500 fell 4.4% with dividends included (Nov 4
close to Nov 5 close)" from the run's own `market_closes.json`.

### Built (presentation only; payload byte-identical)

1. **The header names the session the prices are from.** It said "Data as of Nov 5, 2026, 2:00 AM CST"
   and nothing else, and that run was priced at the **Nov 4 close** - the day before the fall. On the
   one day a reader most needed to know how old the prices were, the page could not say. The eyebrow
   and the footer now add "prices at the Nov 4 close": the `stock_detail[t].price_date` most stocks
   carry (the field the input gate already reads), ties to the later session, and "for N of M stocks"
   when not all of them do. `pricedSession()` in the page, `priced_session()` in Python, held to each
   other on the payload and on hand-worked cases; claim `dashboard.priced_session`,
   `tests/test_priced_session.py` (15).
2. **The Diagnostics tile says what its bars are.** Eight grey bars of each weight's top-20 stability on
   a 0-1 scale: 0.70 and 0.86 are a few pixels apart in a 40px tile, so they read as one height, and no
   caption said what they measured. Bars at or above the 0.85 line (the view's own "Robust") are now
   blue, and the caption reads "Top-20 overlap when each weight moves 5 points; 1 of 8 reaches the dashed
   0.85 line, where the list barely changes". *5 points, not 5%*: `run_weight_sensitivity` adds 5 to a
   weight out of 100. **The Diagnostics view itself still said "moved &plusmn;5%"** - the wording the
   methodology page dropped on 2026-10-19 (claim `overview.weight_sensitivity`: for Size's 5% it reads as
   4.75-5.25, where the run tries 0 and 10). The `FORBIDDEN` pattern only reads the methodology text and
   the view spells it `&plusmn;`, so nothing saw it; corrected to "5 percentage points up and down, one
   at a time, the others rescaled", pinned against `run_screener`'s own constants. Claim
   `dashboard.stability_tile`, `tests/test_stability_tile.py` (4).
3. **The What Changed tile carries the panel's two figures for the last run.** The owner's question on
   the day of the fall - did the ranking hold? - was answered inside the panel and nowhere on the front
   page. The tile now ends "Since the last run: S&P 500 &minus;4.4%, rank correlation 0.987 (typically
   0.996)" (red for a fall, green for a rise, unsigned at 0.0%), and only the correlation where the run
   has no market figure - which is today's live page, until the 16:00 run. Claim `history.tile_pulse`,
   `tests/test_changed_tile_pulse.py` (7).

**Looked at it** at 1440 and 375 (header, both tiles; the tile with the discarded 11-06 run's payload
for the market figure): no page errors, no horizontal scroll; on a phone the price session sits on its
own line under the date.

**Mutation-checked:** `>=` -> `>` at the 0.85 line (both the count and the mark), the tie rule, the "for
N of M" branch, an empty date counting as one (missed at first - it only matters where empty dates
outnumber real ones; a case added), the tile's colours swapped, the tile reading the month instead of
the last run, and the 0.0% sign rule - each caught.

### Next

- Read the 16:00 run's log and page: it is the first published run since the fall, and the first with
  the capped-row fix and today's three lines live. Delete the one-time task afterwards.

## 2026-11-06 (13:00 session) - QUEUE: the Factor Lab, experiment 19 (Growth's first rerun)

**Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-06_060000.log`
(completed). Data loop published? **no** - `logs/datarun-2026-11-06_020000.log` (stopped
deliberately by the publish gate; HEALTH: PASS; both causes were fixed and shipped by this
morning's 06:00 session, and the owner-run session scheduled a one-time catch-up run for **16:00
CST today**, after the close - reading it is the next session's first job). Evidence base at `1m`:
**44 rows, newest 2026-10-05 (32 days ago, bound 40), 5 effective of the 8 needed**. Priority 0:
`allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab** (age **25 days**) -
**advanced today**; also open: 0.10 30d, 0.14 23d.
**Tests:** before 4046 passed / 2 skipped (green baseline, nothing inherited), after 4046 / 2 -
the same count: today added assertions inside existing tests, not new test functions.
**Owner queue / rotation:** no open proposal (`write_brief.open_proposals()` -> `[]`). No
held-back branch. Today's 06:00 entry ended "the next Friday that is not a retrospective owes the
lab one finished experiment" - today **is** that Friday (ISO week 45, odd), so this session
carried on from its Next rather than starting a fresh owner item. `git stash list` holds one
`auto-rescue 2026-10-28` entry: **it is empty** (`git stash show -p` prints nothing), so there is
nothing in it to recover.

### Did

**Experiment 19: Mohanram (2005), the growth score, on Open Source Asset Pricing's `MS` groups.
4 of 7 figures within tolerances committed before any return was read (`4a55141`).**
`research/2026-11-06-lab-mohanram.md`.

**Growth was the last of the eight categories with no rerunnable experiment - and the lab's own
category table said why, falsely:** "Mohanram (2005), which it cites, needs company accounts, and
the library has no sort on sales or earnings growth." The second half is still true; the first is
not. Chen & Zimmermann publish portfolio returns for exactly this score, and their construction
code **pools the score's ends the way the paper does** (6-8 to 6, 0-1 to 1), so their six
portfolios *are* the paper's six groups - which is why this comparison could be row by row where
experiment 17's, on deciles of a discrete score, could only be end to end. That is the lab
auditing the screener's own citations in the direction nobody plans for: the claim about what
*cannot* be checked was the thing that needed checking.

**The result, and the part that matters to a screener that only buys.** The strong end reproduces
(16.5% a year against the paper's 17.4%); the weak end does not (+3.2% against -4.0%), so the gap
is 13.3% a year (t 4.96) against 21.4% and outside its tolerance. The whole shortfall is the
losers the public construction does not find - and the paper's own size-adjusted panel says that
is where its effect lives (+3.3% strong against -17.9% weak; its words, "more effective in
identifying potential losers ... than winners"). So the half that reproduces is the half this
screener could act on, and the half that does not is the half it never could. Score 5 sits below
score 4, so the monotonic rise is not reproduced in the middle.

**It survived its own sample, which almost nothing in this lab has:** 5.0% a year equally weighted
since 2002 (t 2.3), 6.9% since 2007 (t 2.7), and 6.0% by market value since 2007 (t 2.2). Against
experiment 14's asset-growth sort - a *growth rate* - at -0.07% a month (t -0.29) since its paper.

**No weight proposed and none moved.** What this sharpens is the open **Growth** question, which
CLAUDE.md already states is composition rather than weight: the category scores growth *rates*,
and the evidence it cites is about sorting growth companies on fundamental strength. Note section
6 lists the three things a Monday would have to answer first (where the steadiness signals belong,
whether any of it holds inside the S&P 500, and what practitioners do - they disagree).

**Two defects fixed in the test that bounds the overview's three tables on a phone.** They cost
most of the session's second half and both are families CLAUDE.md already names:

1. **It measured before the page had stopped changing** ("paint->red"). The three tables stand
   **157 px taller once Inter has loaded**, and `document.fonts.status` reports "loaded" before
   the faces this page uses have arrived - measured three runs in a row. The same commit passed
   serially (5,966) and failed under xdist (6,123). It now polls until the height holds still.
2. **The bound was absolute where the content is not.** After-the-paper is one row per experiment,
   so a fixed 6,000 px ceiling can only be met by having **fewer experiments** - the opposite of
   what the page is for, and it would have failed on experiment 20 whatever the prose said. It is
   now **150 px a row**, against a measured 139 (6,123 over 44 rows) and the 198 a row the
   2026-10-27 pass started from. The intent - the overview reads as rows a phone can scroll - is
   unchanged, and the Growth cells were shortened first, before the bound was touched.

**Looked at the page** at 1440 and 375 px (`logs/shot_exp19.py`): the GROWTH group in the index
with "19 Mohanram G-score 4/7", the verdict box with its four-of-seven meter and the published
null line ("A result of no effect at all would match 3 of these 7" - the build computes that
count, and it agrees with the one the record worked out by hand before the run), all seven
comparison rows and the six-bar chart at both widths, no page errors, no sideways scroll.

**Moved together, as the lab's rules require:** `EXPERIMENTS`, `TAKEAWAYS`, `CATEGORY_MAP`, the
methodology's "Rerun the research" paragraph (**regenerated** from `build_screener_overview`,
never hand-edited - rule 10), `ASSET_VERSION` 30 with both `?v=` tags, the scorecard's Growth row,
a Growth group in the menu, and `index.html` / `dashboard.html` rebuilt with
`scripts/rebuild_page.py` (payload byte-identical).

### Evidence / research

- **Mohanram, P. S. (2005)**, "Separating winners from losers among low book-to-market stocks
  using financial statement analysis", *Review of Accounting Studies* 10(2-3), 133-170,
  doi:10.1007/s11142-005-1526-4. Read with `pypdf` from the **accepted manuscript** (University of
  Toronto TSpace, April 2004 draft, 47 pages); Table 4 Panel A, PDF p. 39; the pooling rule and
  the "torpedo stocks" caveat from pp. 18-19. **The version of record was not obtained** and that
  is said on the page and in the note.
- **Chen, A. Y. and T. Zimmermann (2022)**, "Open source cross-sectional asset pricing",
  *Critical Finance Review* 11(2), 207-264 - the portfolio returns. Their `MS.py` and
  `01_PortfolioFunction.R` at tag **v2.0.0** were read before the tolerances were written; the two
  portfolio files only had their metadata looked at (signals, ports, months, how many companies a
  port holds).
- **Tolerance 4.0 percentage points a year**, two standard errors of the paper's own spread
  (21.4 / 10.71 = 1.998). The t-statistic is **deliberately not compared**: the paper's 10.71 is a
  two-sample t across 5,585 firm-years, and anything computed here is a t across monthly spreads.
- **Rule 8, from the record before the run:** a null matches **3 of 7**. `fl.null_within` computes
  3 independently.

### Methodology changed

- **None.** No weight, metric, threshold or published score moved. What changed on the public
  methodology page is one sentence of the "Rerun the research" paragraph, which now names
  experiment 19 - regenerated from the generator, with `SCREENER_OVERVIEW.md` rewritten by it.

### Tried and rejected

- **Comparing the paper's t-statistic, as experiment 17 did.** Rejected before the run: a
  cross-sectional two-sample t and a time-series t on monthly spreads are different quantities,
  and pairing them with a tolerance buys a guaranteed miss that says nothing. Reported beside the
  comparison instead (4.96).
- **Trimming prose until the phone budget passed.** Tried first, and it is why the Growth cells
  are as short as they are - but it does not work: the last two trims moved the settled height by
  0 px, because the overage is a whole new table row, and the next experiment would meet the same
  wall. The budget's *shape* was the defect, not the sentences.
- **A peek that turned out not to be one.** The record discloses that `SignalDoc.csv`'s summary
  for `MS` (1.575%/month, t 9.14) was read while reading the documentation, and says the spread
  was therefore expected inside tolerance. It was not: 13.3% measured. Checking afterwards against
  `PS` (1.958333 x 12 = 23.5%, Piotroski's published spread exactly) shows that column holds
  **the paper's** figure as the library transcribed it, not a measured one - so the
  pre-registration was blinder than its own disclosure claimed. The committed comment stands as
  written; the correction is in the note (section 4), not in the pre-registration.

### Next

- **Read the 16:00 catch-up data run's log and page** (the owner-run session's standing Next): it
  is the first published run since the 4.4% fall, and the first carrying the capped-row fix and
  this morning's three presentation lines. Delete the one-time task afterwards.
- **Experiment 19 needs its second reader** - a session that did not write it: the published
  article's Table 4 Panel A if a copy can be had, every measured figure recomputed without
  importing `factor_lab`, and the section-4 claim about `SignalDoc`'s `Return` column tested on
  two or three more signals.
- The lab's next experiment is **`EarningsSurprise`** (Foster, Olsen & Shevlin's SUE), which puts
  a rerun behind the Revisions category's 25 surprise points.

## 2026-11-06 (owner-run, evening) - the catch-up run could not publish, by construction. Fixed.

**Health (rule 8, all five):** last code session ran? yes - the 13:00 session shipped to main.
Data loop published? **no, twice** - 02:00 stopped by the publish gate (fixed by the 06:00
session), and the 16:00 catch-up run discarded as below; the live page is still the 2026-11-05
02:00 run. Evidence base at `1m`: **44 rows, newest 2026-10-05 (32 days ago, bound 40), 5
effective of the 8 needed** (unchanged since 13:00: neither run today recorded a snapshot that
survived). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor
Lab** (age **25 days**).
**Tests:** before 4046 passed / 2 skipped (13:00's count); after **4067 passed / 1 skipped** (+11
`tests/test_same_day_rerun.py`, +9 `tests/test_summary_percent_near_zero.py`, and one fixture line
in `tests/test_rehearse_publish.py`; the skip count moved with the new run's data).

### What happened at 16:00

The owner approved a one-time after-close data run so the site would not sit on Wednesday's
close all weekend. It ran at 16:00:01 and took 90 seconds, because it never fetched: it loaded the
02:00 run's scored cache (`[CACHE HIT] Loading scores from factor_scores_3ed2aee7f770_20261106`).
Dashboard generation then refused it - `CalculationMismatch: 286 of 499 stocks do not reproduce
their published scores` (ALL valuation 63.3 published, 93.3 recomputed, "profile generic") - and
the health check discarded the run for having no payload of its own. **The gate was right.**

**Cause.** `factor_engine.write_scores_parquet` drops every column whose name begins with `_`
(since Phase 0, 2026-02-17; the overview even says so). Among them is `_is_bank_like`, which the
generator's weight resolution reads, so every insurer was checked against the generic weights.
The scores themselves were byte-identical to 02:00's. And because the run fetched nothing it had
no `00_raw_fetch.parquet` either - which `check_run_health` requires - and its prices were the
02:00 fetch's, the previous session's close. **So a second full run on the same day could never
publish, and could never bring after-close prices even if it had.** Proposing the catch-up run
without checking this was this session's mistake.

**A second-order fault, the one that mattered for the weekend.** The discarded run was now the
newest folder in `runs/`, and `scripts/rehearse_publish.py` (gate 5) rehearses from the newest
full run - so every rehearsal failed with the same `CalculationMismatch`, whatever the branch.
Every weekend session's work would have been held off `main`. Confirmed by running the rehearsal
(failed), moving the folder to `logs/discarded-run-bce9b3ce6629` (not deleted), and running it
again (passed - which also confirms the 06:00 fixes clear the 02:00 run's data).

### Did

- **`run_screener.py --fresh`**: fetch and score afresh even when today's scored cache exists,
  keeping every cache file (`--refresh` would delete all `factor_scores_*.parquet`, which
  `backtest.py` reads as its rolling history). The decision is one function,
  `reuse_score_cache()`.
- **`scripts/data-run.ps1` always passes `--fresh`.** The publishing run never reuses the cache;
  hand runs and trials still may.
- **`rehearse_publish.newest_full_run` skips a run without `00_raw_fetch.parquet`** - a run that
  fetched nothing is not what the next data run will publish.
- `tests/test_same_day_rerun.py`, 11 tests; each of the three fixes reverted in turn fails one.
  `tests/test_input_gates.py` anchored on the literal `@('run_screener.py')`; it now anchors on
  the call's prefix, asserting the same ordering.
- Deleted the one-time task `Screener Data Run catch-up 2026-11-06`; `schtasks /query` confirms it
  is gone and the five standing tasks are untouched.

### Methodology changed

- **None.** No weight, metric or score moved; the cache window sentence on the methodology page is
  still true (a hand run may reuse the day's cache; the data loop now never does).

### Verified - the data loop rerun with the fix (18:34, `logs/datarun-2026-11-06_183427.log`)

With the fix on `main`, the standing `Screener Data Run` task was run once by hand (the owner's
after-close approval, which the 16:00 run had failed to deliver). It ran `run_screener.py --fresh`,
fetched (run `84a6a8b4c9dd`, `00_raw_fetch.parquet` present), and **published at 18:59**:
HEALTH: PASS - price dates **499/499 on the 2026-11-06 session**, rank continuity Spearman 0.979
against the 11-05 page with 2 new names in the top 25, 1 of 503 unscored (VYLR); PUBLISHED CLAIMS:
PASS (2252 passed); `signals.json` names the same `run_id` and `run_timestamp` as `kpis`. Evidence
at `1m` is now 45 rows, 5 effective. The live page reads "PRICES AT THE NOV 6 CLOSE"; What Changed
reads "S&P 500 −4.0%, rank correlation 0.979 (typically 0.996)" (11-04 to 11-06 closes); the live
page's own `equationHtml` prints "capped to 100.0%" for all 11 stocks the scorer capped and for no
other.

### The run turned `main` red - fixed the same evening

Gate 1 on the log commit after it: **1 failed** - `tests/test_every_panel_reads_clean.py`, LNT's
summary "the 60th on 12-1M Return (-0.0%)". Not inherited and not a flake: a return of about
-0.01% met a formatter with no near-zero rule. `stock_summary._fmt_metric` is the Python mirror of
the page's `fmtMetric`, and when the JS formatters were brought under `pctNearZero` on 2026-11-03
the mirror was not. It now follows the same rule (`_pct_plain`: not zero -> one more decimal; still
zero -> unsigned); `tests/test_summary_percent_near_zero.py`, 9 cases worked by hand, 6 of which
fail on the old line. The payload was regenerated from the same run (`84a6a8b4c9dd`, via
`rehearse_publish.regenerate` into an ignored folder) and **diffed field by field against the
published one: exactly one difference**, that sentence, now "(-0.01%)"; `dashboard_context.js`
byte-identical; the page differs only in the payload's `?v=` hash; `signals.json` untouched (same
`run_id`, timestamp and ranking). Suite after: **4067 passed, 1 skipped.**

**Pages did not deploy it.** Commit `84a4cd4` built fine, but its `pages-build-deployment` deploy
step hung in `syncing_files` and timed out at 10 minutes - three times (the original, a rerun, and
a fresh `POST /pages/builds`), while the five deploys before it that day took ~50 s. The uploaded
artifact was the same size as the last good one (37,180,568 bytes against 37,178,882), and
githubstatus.com showed Pages operational with no open incident - so GitHub's side, not the
commit. The live page therefore still served `9f92130` (the 18:59 publish: right data, LNT's one
"-0.0%" sentence) when this was written; this log commit is the next deploy attempt.

### Next

- **Check the live site serves `main`**: `D.stock_detail.LNT.summary` should read "(-0.01%)". If
  Pages is still timing out, the data is right and only that sentence is stale - not worth more
  than a rerun of the latest `pages-build-deployment`.
- Monday's 02:00 run is ordinary (no same-day cache exists then); nothing to watch for.
- The same-day cache is still unsafe after a *code* change: its key is the config hash, so a hand
  rerun after a scoring fix reuses scores from the old code. Hand runs only; not urgent.

## 2026-11-06 (owner-run, night) - a UI audit of every surface: ten fixes, each found by measuring

**Health (rule 8, all five):** last code session ran? yes - 13:00 shipped to main. Data loop
published? **yes** - the 18:34 rerun (`logs/datarun-2026-11-06_183427.log`, HEALTH: PASS, claims
PASS, the Nov 6 close). Evidence base at `1m`: **45 rows, newest 2026-10-06 (31 days ago, bound
40), 5 effective of the 8 needed**. Priority 0: `allow_auto_apply` **false**. Top open roadmap
item: **0.13, The Factor Lab** (age **25 days**).
**Tests:** before 4067 passed / 1 skipped; after **4106+ passed** (each batch gated in full).
**Owner direction (chat):** *"keep finding as many improvements as you can. Specifically to the ui
and frontend of everything... Check every single thing."* Owner-run, so the rule against nightly
design work does not bind; every change below is display-only unless it says otherwise.

### How the audit was run (re-runnable; the scripts were scratch, the method is the point)

Screenshots of every surface at 375, 768, 1024 and 1440 (page, all 499 sheets' text, every
panel, dialogs, lab, research, print), plus five automated sweeps: contrast/labels/tap targets/
overflow on every element; **every media-query rule compared with what the browser computes**
(this found two phone layouts that had never applied); truncated text without its full form;
focus visibility on every control (all pass); and all 499 sheets' text for "1 <plural>",
NaN/undefined, and any percentage of 500% or more.

### Shipped (tags good/2026-11-06-owner-5, -6, -7 and the next)

1. **Insider list marks its 90 days** - 253 of 444 lists showed older trades unmarked, 70 under
   "None"/"None". Divider at `summarise_rows`' own cut (`insWindowStart`).
2. **Five-year yield panel: share counts filed in the wrong units** - HST's range read
   -84764.2% to 5455967.6%; 15 stocks had such months. `valuation_history.SHARE_UNIT_RATIO = 10`,
   from a measured gap (genuine <= 4.24x, errors >= 914x, nothing in 5-100x;
   `research/measurements/2026-11-06-valhist-share-units.py`). Rebuilt from the same caches: 428
   of 428 blocks kept, absurd ranges 10 -> 1 (EXE: Chesapeake's 2021 fresh-start accounting, shares
   in bounds, left). **Reaches the page with Monday's data run.** Context only, but it is a
   logged context signal: 14 blocks' own-history percentiles move by up to 6 points, before any
   `1m` reading of that signal exists.
3. **"1 ranks" -> "1 rank".** A sweep of all 499 sheets finds no other count-of-one plural.
4. **Track record's open period** - the Nov 3 period (3 days) was drawn and counted like the
   month-long ones; now hatched, "so far", and the caption says the count includes it. Period
   dates wrap on a phone instead of truncating.
5. **Diagnostics on its own scale** - "+5%: 82%" -> "+5pp: 0.82"; badge "(0.77)".
6. **Phone comparison tray** - a later bare rule beat the phone rule; the second chip read "MI".
7. **Phone workings rows** - two more specific rules beat the phone ones; every row carried an
   82px "Calculation" pill. Now the chevron, as designed.
8. **Printed sheet** - "every section open" printed the workings' category headers with no
   metric under them; now the tables print and on-screen controls do not.
9. **Printed rankings** - a landscape page lays out at ~1010px, inside the 3-and-2 card spans;
   the five leaders now take one row (the print test now runs at 1010).
10. **Summaries name inputs from the stock's own weight table** - JPM's "strongest inputs" were
    Piotroski at the 14th and Accruals at the 2nd percentile; banks score Quality 75% on ROE,
    ROA, equity ratio. `stock_summary.stock_metric_weights` looks up the published profiles and
    `wp` (row 0.8). Payload regenerated from the same run and diffed: 43 bank-scored stocks'
    sentences change, nothing else. Claims `summary.best_inputs`/`worst_input` updated.
11. **Links**: `#stock=GOOG`/`FOX`/`NWS` open the company's ranked line and say why; a link to an
    unscored member gives its reason; `#sec-...` opens the section, as the header's links do.
12. **Renamed and departed names** - MMC, BK, EQR, SATS were told they had "left the S&P 500" while
    ranked as MRSH, BNY, VMRK, ECHO. `history.renamed` (by SEC registrant, the track record's own
    `renamed_symbols`) routes links, search and saved holdings to the new line; `history.last`
    gives a departed name's last rank ("It last ranked #55 on Jul 29"), which the holdings card's
    code had always tried to print and never could. Claim `dashboard.departed_names`. Payload
    diffed: those two keys only (~1 KB).
13. **The five-year panel's method note** now names the share-unit bound, tied to the constant.
14. **Partial option readings say where their figures come from** - 64 stocks' at-the-money bid and
    ask failed, so no move is shown "rather than built from a quote that would not stand up", while
    the source's last-trade implied volatilities sat beside that sentence unmarked (HST 90% implied
    vs 25% realised). Measured: IV/realised median 1.43 (partial) vs 1.19 (ok); >2x in 9% vs 1%;
    |skew| > 15 pts in 21% vs 2%. One caveat line; the figures stay (logged, never scored).
15. **research.html's share preview** said "26 pre-registered alternatives" (hardcoded) under a page
    saying 27 (from the ledger); both read `summary['alternatives']` now.
16. **Track record's changelog entries are listed** under "How the record is measured" (the ticks'
    hover titles were unreadable on a phone); `track_record.changelog_title` cleans session tags,
    backticks and mid-word cuts. `data/track_record.json` re-titled in place (dates asserted equal).
    Both methodology-adjacent changes are in `METHODOLOGY_CHANGELOG.md` (2026-11-06, owner-run, night).

### Looked at and left, with the reason

- Tap targets under 24px on phones: all pass WCAG 2.2's spacing exception or sit in sentences.
- Mixed `-`/`−` in negative numbers across the sheet (762 true minus vs hyphens in the workings,
  targets, rank deltas and baked sentences): cosmetic, and dozens of tests pin the characters.
- Very large real readings (SNDK/MRNA 12-1M returns > 700%, VRSN ROIC 699%, ECHO surprise
  1594%): real data, scored by percentile.
- "Reporting soon" lists the run's own day as upcoming; right for the 02:00 run, odd only for an
  evening rerun.
- 320px (smallest phones, 400% zoom) and all 499 sheets at 375: no horizontal overflow anywhere.
- Tap targets the sweep flagged at 375 (Hide, Show more) already get 40px+ under `pointer: coarse`;
  the sweep ran without touch emulation.
- Every lab control in all 19 experiments clicked: no page error, no NaN/undefined. Every in-page
  link resolves (the methodology's slug links via `methodAnchorClick`, the lab's via its router).
- Focus is visible on every focusable control on the page and in the sheet.

### Next

- **Monday 02:00:** HST, CCL, CHD, EG, COP, GRMN, WRB, TER, BRO five-year ranges should read as
  yields (EXE excepted). If any still reads over 100%, the bound missed a case - investigate.

## 2026-11-07 (06:00 weekend session) - the universe becomes a parameter where it decides what a stock is compared with

**Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-06_130000-midday.log` (completed). Data loop published? yes - `logs/datarun-2026-11-06_183427.log` (completed; HEALTH: PASS). Evidence base at `1m`: 5 of 8 needed at the 1m horizon (45 rows, but overlapping windows are not independent; 114 rows across all horizons), newest 1m run date 2026-10-06 (32 days ago, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab** (age **26 days**); also open: 0.10 31d; 0.14 24d.
**Tests:** before 4117 passed / 1 skipped (195 s, parallel); after 4130 passed / 1 skipped.
**Owner queue / rotation:** `write_brief.open_proposals()` is empty - no proposal waiting. Took
the **top open owner item** (2026-10-28, beyond the S&P 500), whose named plan is
`plan/universe-expansion.md` and whose Progress line said phase 1, `size_band` bullet. One stage,
finished end to end. Nothing deferred for a gate or a stalled loop. The stash `auto-rescue
2026-10-28_060001` holds an empty diff (`git stash show -p` prints nothing) - left alone.
0.13 (the Factor Lab, 26 days) deferred: Friday is the lab's day and the owner queue outranks the
roadmap; experiment 19 still wants its second reader.

### Did

**Percentiles are ranked within `(Sector, size_band)`, with a band-wide fallback.** This is
pre-registration 3 of the universe expansion, adopted 2026-11-03 and until today only written
down. Once the S&P 400 and 600 are in the pool, "cheap for a utility" has to mean cheap against
utilities *of about this size* - the size effect runs through most of these metrics - and a
sector-band group below `SECTOR_MIN_PEERS` has to fall back to its **band**, because falling back
across bands puts exactly the stocks with the thinnest peer group (small companies in a small
sector) back in a distribution of mega caps.

- `universe.Universe` carries a `size_band`; `universe.assign_size_bands()` sets the column,
  never overwriting one a caller set, and fills blanks because `groupby` drops a NaN key and a
  stock dropped out of its own distribution is the failure this whole exercise is meant to catch.
- `run_screener` calls it once the universe is settled, so `01_raw_metrics` onward carry it.
- `factor_engine.compute_sector_percentiles` groups by `(Sector, size_band)`; the fallback rank
  table is per band. `SECTOR_MIN_PEERS` stays 10 - only its scope changed.
- `Composite_Pct` is ranked within band.
- `universe.py` joined `v2_experiments.RESULT_PATHS`: it now decides what each stock is ranked
  against, so an edit there can move every engine number, and the hash could not see it.

**Nothing published moved, and that is measured rather than argued** - `tests/test_size_bands.py`,
13 tests. The module keeps a transcription of the sector-only code it replaced and compares the
two on the published run: **0 of 23,092 percentile values differ**. Its floor is a half-and-half
cap split, which moves **20,140** - so the zero means something. (Phase 0's script reported 20,151
on the 2026-11-03 run; that figure is a property of a morning's data, not a constant, which is
worth knowing before someone treats it as one.) The composite percentile has the same pair:
unchanged with one band, **492 of 499** moved by the split. The band-wide fallback has a fixture
built so the two fallbacks *disagree* - a thin cell that is the bottom of its own band and the top
of the universe - because with one band or one sector nothing can tell them apart, which is how a
universe-wide fallback could have shipped under this name.

**Verified past the suite, because unit tests cannot see a shape assumption:** a five-ticker live
`run_screener.py` (the band reaches `01`/`02`/`05`, the calc trace reproduces 40 category scores
and 5 composites, `SCREENER_OVERVIEW.md` regenerates byte-identical), the run directory and the
cache files it wrote deleted afterwards and the four touched tracked files restored; then
`scripts/rehearse_publish.py` - all four published files unchanged byte for byte, 380 page tests
and 2,323 claim checks green.

**The rerun this cost, paid in full:** editing `factor_engine.py` (and `v2_experiments.py`) stales
every ledger row. All 32 were rerun with `--jobs 4` in four foreground chunks (~25 min) and
`python v2_experiments.py report` regenerated. **No statistic moved** - the only changes in
`RESULTS.md` are the run count (616 -> 648) and the provenance line's code hash; `summary.json` and
`research.html` are byte-identical, so the public research page says exactly what it said.

### Also did - Factor Lab experiment 19, second-read (owner item 2026-10-20, sub-item 2)

The universe plan says one stage a session, so the phase-1 stage stopped where it stopped. The
remaining time went to the other open owner sub-item: **experiment 19 (`mohanram`) had not been
second-read**, and `OWNER_FOCUS.md` item 2 says an experiment is not checked until someone who
did not build it has read it. All three of the note's own open items were taken.
**It stands - nothing needed correcting.** Record: `research/2026-11-06-lab-mohanram.md` section
8; `research/measurements/2026-11-07-lab-mohanram-second-reading.py`, committed, which does not
import `factor_lab`.

- **The pre-registration is genuine.** `4a55141` (13:14) adds 173 lines to `factor_lab.py` and
  nothing else - no `lab_data.js`, so no return had been written - with `verdict_note` empty;
  the result is `3f3cb98` at 13:26. All seven paper figures and tolerances are byte-identical
  across the two commits. **Twelve minutes is the shortest gap of any experiment** (16 and 17
  had 24 and 90); the ordering is sound and the gap itself carries little weight. Said so.
- **All seven reference figures exact**, re-read from PDF bytes - and in **two independent
  manuscripts**, Columbia's as well as the TSpace one the record cites, whose Table 4 Panel A is
  character-identical. The **version of record is still not obtained** (Springer's PDF link
  returns HTML), so that caveat stands rather than closing. Panel B's two ends (+3.3%, -17.9%)
  and the 5,585 firm-years behind the paper's two-sample t are exact too.
- **Every measured figure reproduces to the second decimal** from the parquet files: +3.22,
  +5.40, +6.48, +12.75, +7.58, +16.48, spread +13.26, t 4.96, 4 of 7 within; the four
  later-window spreads and their t-statistics; and the strongest group the calmest (6.24 against
  7.18-7.89). A cross-check the note did not have: the library's own `LS` port averages
  1.1047%/month = 13.26% a year, so the lab's spread *is* the library's.
- **The note's weakest claim is now its strongest.** Section 4 said `SignalDoc.csv`'s `Return`
  column is the paper's figure rather than a measured one, resting on one check. Tested on seven
  signals, **five differ from the library's own measured long-short** by more than 0.05%/month,
  two by over 40% - and the values are recognisable as published headlines: `Mom12m` at
  **1.31% a month** is Jegadeesh & Titman's own number.
- **One slip found, and it is the paper's.** Mohanram's text on PDF p. 26 cites "the 21.2%
  difference reported in Table 4" while discussing raw returns. 21.2% is **Panel B's** spread;
  Panel A's is 21.4%, which is what the record uses. Close enough that a reader meeting only
  p. 26 would carry away the wrong number.
- **One observation, not a defect:** the library's six `MS` ports do not start together (port 06
  only from 1974-09), so 163 months before 1976-07 have no high group. None is inside the
  compared window - 1978-01 onward is complete for all six - but the page's
  "Everything · FIRST-LAST" preset draws each group over its own span while the spread uses the
  587 months both ends exist.

No weight, metric, threshold, tolerance, verdict or published sentence changed.

### Evidence / research

- Pre-registration 3 of `research/2026-11-02-universe-expansion-phase0.md`, decided and pinned
  2026-11-03 (`tests/test_universe_expansion_preregistration.py`). Fama & French, Hou/Xue/Zhang and
  Jensen/Kelly/Pedersen all form portfolios inside size breakpoints rather than across them; MSCI,
  FTSE Russell and S&P rank within size segment. This session implements that decision; it does not
  reopen it.
- The *failure* this prevents is the one phase 0 measured: with a genuine two-band split, 20,140 of
  23,092 percentile values change (up to ~69 points). A universe-wide fallback would have left the
  thinnest cells - the ones most in need of a sensible peer group - ranked against mega caps.

### Methodology changed

- **None.** No weight, metric, threshold or published sentence moved. The grouping is a different
  expression of today's rule, byte-identical on today's universe, which is the phase's exit test.
  It becomes a methodology change on the day a second band is published, and
  `METHODOLOGY_CHANGELOG.md` gets its entry then, one change at a time.

### Tried and rejected

- **Pinning `Composite_Pct` by recomputing it from the published composite.** It fails, and the
  claim is wrong rather than the code: the percentile is taken *before* `compute_composite` rounds
  the composite to 2dp (factor_engine.py:4059), and **58 of 499** stocks on the 2026-11-06 run are
  tied after that rounding and not before. The settled row `data->red` says a premise is not
  asserted until it is asserted; this is the same thing a level down - a test that recomputes a
  published figure is asserting a *pipeline order* it has not checked. The test now pins the claim
  the edit actually rests on (ranking inside one constant group is the ungrouped ranking, with a
  check that the run really does carry ties), and the production path is pinned by the two-band
  floor test instead.
- **Starting the `config.yaml` `universe:` block.** The 2026-11-03 session left it deliberately -
  the definition lives in Python until a second universe needs a switch - and nothing today needed
  it. Adding a config knob with one legal value is a surface to maintain, not a step forward.

### Next

- **Phase 1's remaining bullets**, in this order: the per-band health gates
  (`check_run_health.MIN_TARGET_COVERAGE`, `run_screener.CATEGORY_AUTO_DISABLE_PCT`, per-index
  `universe_history.MIN/MAX_PLAUSIBLE_MEMBERS`) - only the last touches `RESULT_PATHS`, so most of
  it costs no rerun - then the `stock_detail` shard, which is the one that keeps first paint flat
  as the universe grows and wants a session of its own.
- **A phase-3 obligation recorded today:** `drilldown.peer_rank`, `overview.percentile_rule` and
  `overview.glossary` all still say a thin sector falls back "across the universe". True today
  because the band *is* the universe; the session that publishes a second band must reword all
  three in the same commit.
- The **next** lab experiment built needs its own second reader; 19 was read this session and
  stands, so that sub-item is clear.
- If a copy of Mohanram (2005) as published (Springer, *Review of Accounting Studies* 10(2-3))
  ever becomes reachable, re-read Table 4 Panel A from it: two manuscripts agree, the version
  of record is still unseen.

## 2026-11-07 (13:00 session) - QUEUE: the per-band health gate, and the one of its three thresholds that is a methodology decision rather than engineering

**Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-07_060001-weekend.log` (completed). Data loop published? yes - `logs/datarun-2026-11-06_183427.log` (completed; HEALTH: PASS). Evidence base at `1m`: 5 of 8 needed at the 1m horizon (45 rows, but overlapping windows are not independent; 114 rows across all horizons), newest 1m run date 2026-10-06 (32 days ago, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab** (age **26 days**); also open: 0.10 31d; 0.14 24d.
**Tests:** before 4130 passed / 1 skipped (211 s, parallel); after 4194 passed / 1 skipped.
**Owner queue / rotation:** `write_brief.open_proposals()` is empty - no proposal waiting.
Second session today. Read the 06:00 entry and **carried on from its `Next`**, which named phase
1's remaining bullets "in this order: the per-band health gates ... then the `stock_detail`
shard". Took the per-band gates. Nothing deferred for a gate or a stalled loop. The stash
`auto-rescue 2026-10-28_060001` still holds an empty diff (`git stash show -p` prints nothing) -
left alone, as this morning did. 0.13 (the Factor Lab, 26 days) deferred again: the owner queue
outranks the roadmap, and this morning cleared the lab's one outstanding sub-item (experiment
19's second reading), so the lab has nothing waiting on a reader.
**Did not reverse this morning's decision** to leave the `config.yaml` `universe:` block alone -
it is still a knob with one legal value, and nothing today needed it.

### Did

**`check_run_health.MIN_TARGET_COVERAGE` (0.50) now applies inside each size band as well as to
the universe.** This is the first of phase 1's three remaining bullets
(`plan/universe-expansion.md`: per band, "because a universe-wide average hides a thin bottom
band"). `check_target_coverage_by_band` fails a run naming any band below the floor; the band is
read from the run's **own** `01_raw_metrics.parquet` (`run_size_bands`), an artifact this gate
already requires as fetch evidence, so the check needs no new input.

- **The band is not published.** Phase 1's exit test is a byte-identical S&P 500 payload, so the
  band stays out of it and the gate reads a parquet - which is also the rule the rest of the
  module has followed since 2026-10-16: judge the run's own artifacts, never the last published
  ones.
- **A run with no band column is a note, never a failure** - every run before this morning. Same
  trade as the unreadable-weights case beside it and the node/pytest fallbacks in both publish
  paths: a gate that can only fail closed jams an unattended loop.
- **It is strictly stronger than the universe-wide check it sits beside, and that is what let it
  ship with no shadow run behind it.** The universe rate is a weighted mean of the band rates, so
  a universe-wide failure implies some band below the same floor. `TestDominance` parametrises
  that over five splits x seven coverage rates rather than asserting it in a comment.

**`MIN_PRICE_COVERAGE` and `MIN_CATEGORY_COVERAGE` stay universe-wide, and the reason is not
caution.** Analyst coverage is the one input to this screener whose *availability* is a function
of company size. A price is served for every listed stock and a price history for every one with
a year of trading, so both of those guard a **feed failure** - which appears across the whole
universe or not at all. Banding them would add ways to fail and no ability to see anything. Two
tests pin the scope, so it reads as a decision rather than an oversight.

**`CATEGORY_AUTO_DISABLE_PCT` (30) is measured per band and still triggered universe-wide. That
is the part of the bullet I did not build, on purpose.** The plan lists it beside the health gate,
but the two are not the same kind of thing: the health gate's action is *discard the run*, so per
band is engineering. The auto-disable's action is *change the weights*, and there are three
incompatible per-band rules - **(a)** the universe mean decides, one weighting for everyone;
**(b)** any band below the floor disables the category for the whole run, throwing a category away
for 500 well-covered large companies because 600 small ones lack it; **(c)** disable per band,
which gives the bands different weight vectors, so a composite means something different in each,
on top of percentiles that are already within band. Which is right depends on how far Revisions
and Investment coverage actually falls in the S&P 400 and 600 - **phase 2's** measurement, which
does not exist yet. So `run_screener.weighted_coverage_pct_by_band` records the per-band figures
now (into `stats`, printed only from the band a second one exists, so no run log gains a duplicate
line), and choosing between (a)/(b)/(c) is a methodology change with its own
`METHODOLOGY_CHANGELOG.md` entry. `test_the_trigger_takes_no_band_and_so_cannot_have_become_per_band`
fails if it arrives as a keyword argument instead.

**Nothing published moved**, and the rehearsal says so rather than me: `scripts/rehearse_publish.py`
regenerated the page with this tree's code and all four published files came out **unchanged byte
for byte** (380 page tests, 2,323 claim checks green).

### Evidence / research

- **Why analyst targets and not the other two.** Analyst following is increasing in firm size:
  Bhushan 1989, "Firm characteristics and analyst following", *Journal of Accounting and
  Economics* 11(2-3), 255-274 - market value is among the strongest determinants of the number of
  analysts; and Hong, Lim & Stein 2000, "Bad News Travels Slowly", *Journal of Finance* 55(1),
  265-295, whose design rests on small firms being the thinly-covered ones (their
  residual-analyst-coverage sort is taken within size groups for exactly that reason). Both cited
  from their published bibliographic details - **not read this session**; what they support is the
  uncontroversial direction of the relation, not an effect size. The screener's own number is the
  one that decides anything here.
- **Measured on the published run** (`research/measurements/2026-11-07-target-coverage-by-band.py`,
  committed): analyst-target coverage by market-cap decile of today's S&P 500 is 98-100%, the
  **thinnest decile 98.0% - 1.96x the 50% floor**. So the gate cannot fire on the universe we
  publish however it is banded, which is what phase 4 needs to be true before it bands the 500
  itself by NYSE breakpoints. And it is **not vacuous**: with the 40 smallest stocks stripped of
  their targets and made their own band, the universe-wide check passes at **91.6%** while the
  per-band check fails.
- **The mutation standard, because a test module written the same day as its feature is the
  easiest place for a check to read as coverage and not be one** (`plan/claims-mutation-audit.md`,
  nine passes). Six mutations a reader could plausibly ship - the check as a no-op, the floor
  becoming exclusive, every run taking the one-band door, blank bands kept as bands, `df` for `g`
  in the per-band figure (one character), and the auto-disable gaining a `by_band=` argument -
  **6 of 6 caught** (`research/measurements/2026-11-07-per-band-gate-mutations.py`, committed; it
  restores both files in a `finally`, and refuses to run on a dirty tree). And the figure that
  says what the module is worth: with it deselected, the whole suite reports **4,130 passed, 1
  skipped** against the feature being a complete no-op. Nothing else in the project can see it.
- **Verified past the suite**, because unit tests use synthetic frames: the published run copied
  to a temp directory with two *real* bands written into its own `01_raw_metrics.parquet` (the
  bottom 40 by market cap against the other 462), run through `check_run_health.py --run-dir`.
  Both bands were judged, and **457/459 + 40/40 = 497/499** reconciles exactly with the
  universe-wide line - so no stock is lost between the two checks, which is the failure a
  `groupby` on a band column invites. (Exit 2 there is the copied directory having no snapshot of
  its own, not a coverage verdict.) Temp directory removed; the tree was clean afterwards.

### Methodology changed

- **None.** No weight, metric, threshold or published sentence moved. The gate gained a check
  that is strictly stronger and that cannot fire on today's universe (1.96x headroom, measured).
  The auto-disable's threshold and trigger are untouched - see above for why that is the decision
  and not the unfinished half.

### Tried and rejected

- **Making the auto-disable per band in this commit**, which is what the plan's bullet literally
  asks for. It changes the weights, so under (b) or (c) it would be a methodology change shipped
  as a phase-1 refactor with no number behind the choice: the thing that would decide between them
  is the S&P 400/600 coverage phase 2 exists to measure. Did the measurement half, wrote the three
  options into the plan and the docstring, and left the trigger alone with a tripwire on it.
- **Putting the per-band rule on the methodology page.** `SCREENER_OVERVIEW.md` says the gate
  "refuses a run with ... analyst-target coverage below 50%", which is exactly true today because
  the band *is* the universe. Adding "per band" would publish a sentence describing a rule with no
  effect, to a reader who has one universe. Recorded as a phase-3 obligation in the plan instead,
  beside the three claims this morning flagged.
- **`universe_history.MIN/MAX_PLAUSIBLE_MEMBERS` per index**, the bullet's third threshold. The
  bound is already a property of a `Universe` (2026-11-03); what is missing is an S&P 400 and an
  S&P 600 `Universe` to carry one, and those arrive with phase 2's constituent-list source. Giving
  `validate_membership` a `universe=` argument today would edit `universe_history.py`, which is in
  `v2_experiments.RESULT_PATHS` - a 32-row, ~25-minute research-engine rerun for a parameter with
  one legal value, and a second rerun when phase 2 defines the universes. Do it in that commit and
  pay the rerun once. Written into the plan so the next session does not have to infer it.

### Next

- **The `stock_detail` shard** - phase 1's last bullet, and the one that keeps first paint flat as
  the universe grows (the Factor Lab's `LAZY_KEYS` pattern). It wants a session of its own, and the
  bot's `stock_detail[t].price` has to stay where `tests/test_bot_payload_contract.py` says it is,
  or move with the bot's tests.
- Then phase 2: the S&P 400/600 constituent source. That is also the commit that should carry
  `universe_history`'s per-index bounds (one rerun for both) and the S&P 400/600 `Universe`
  objects the per-band gate will then see real bands from.
- The next lab experiment built needs its own second reader; 19 was read this morning and stands.

## 2026-11-08 (06:00 weekend session) - the sheet-only half of `stock_detail` leaves first paint: -19.8% on the wire

**Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-07_130001-midday.log`
(completed). Data loop published? yes - `logs/datarun-2026-11-06_183427.log` (completed; HEALTH: PASS).
Evidence base at `1m`: **45 rows, newest 2026-10-06 (33 days ago, bound 40), 5 effective** of the 8
the engine's gate needs. Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13,
The Factor Lab** (age **27 days**); also open: 0.10 32d, 0.14 25d.
**Tests:** before 4194 passed / 1 skipped (213 s, parallel); after **4218 passed / 1 skipped**
(211 s) - the 24 new ones are this work's.
**Owner queue / rotation:** `write_brief.open_proposals()` is empty - no proposal waiting. Took the
top open `OWNER_FOCUS.md` item (universe expansion, `plan/universe-expansion.md`) and the stage its
Progress line named next, which the 13:00 session's `Next` also named: **the `stock_detail` shard**.
Nothing deferred for a gate or a stalled loop. 0.13 (the lab, 27 days) deferred again - the owner
queue outranks the roadmap and the lab has no experiment waiting on a reader. The stash
`auto-rescue 2026-10-28_060001` still shows an empty diff; left alone, as the last three sessions did.

### Did

**`about` and `industry` now ship in a third payload file, `dashboard_detail.js`, fetched when the
first sheet is laid out instead of at first paint. Measured: the gzipped first-paint payload goes
1.547 -> 1.240 MB, -307 KB, -19.8%.** Nothing published moved - no number, score, rank or sentence.
`generate_dashboard.split_detail()`, `DETAIL_KEYS = ("about", "industry")`;
`tests/test_detail_payload_split.py`, 21 tests.

- **The shape of the shard was decided by measurement, and it is not the obvious one.** The plan's
  bullet says "sharded out of `dashboard_data.js` and loaded per stock". Per *stock* cannot be done
  without rewriting the page: **~85 call sites read `D.stock_detail[t]` synchronously for an
  arbitrary ticker** - search, peers, holdings, movers, compare, the hash router - and each would
  have to await a fetch. Enumerating those sites by enclosing function gives a rule that works
  today: **one read site, inside the sheet, and no number.** `about`/`industry` are read only by
  `renderAbout`. `financials` is also read by `buildPeerRow`, `eq4` by `renderReporting` (a
  first-paint strip), and `raw`/`pct`/`inp`/`summary` by the compare drawer, the movers strip and
  the calculation traces - 42.6% more of first paint, each needing its own measurement. The
  mechanism is now there for them.
- **Guarded in the commit that created it**, which is the 2026-10-16 lesson: both runners'
  size/header/`node --check` specs, `$DataArtifacts`, `check_published_claims.PUBLISHED_ARTIFACTS`,
  `rehearse_publish.PUBLISHED`, `rebuild_page.PAYLOADS` and `run_screener`'s publish copy. Two new
  tripwires generalise it: the rehearsal's install list is **derived from `$DataArtifacts`** (a
  payload file it does not install is one gate 5 rehearses *as committed*), and the page's
  cache-busting token must be the hash of the committed detail file (else a returning reader is
  served yesterday's descriptions forever).
- **Gate 5 passed with all five published files byte-identical** to what the next data run would
  write, which is also the proof that the committed artifacts are exactly what this tree's
  generator makes.

### Evidence / research

- **Where first paint's weight actually is**
  (`research/measurements/2026-11-08-stock-detail-field-weights.py`, committed; it prices each field
  by removing it from the whole payload and re-gzipping, so gzip's non-additivity is priced against
  the same whole). `stock_detail` is **84.3%** of the 1.547 MB gzipped payload, and the heaviest
  single field in the entire payload is **`about`, 298.6 KB, 19.3%** - ahead of `inp` (13.8%), `raw`
  (12.5%), `pct` (8.4%) and `summary` (7.9%). Twelve fields of 43 carry 83.2% of first paint.
- **A false number corrected in `plan/dashboard-inventory.md`, which would have argued against doing
  this at all.** It said the business descriptions "add ~0.71 MB raw but only **~60 KB** on the
  wire". Measured: removing them takes **298.6 KB** off the wire, and the text alone is 0.712 MB ->
  **215.2 KB, 3.31x**. 499 different companies' prose, ~1.4 KB each, gives gzip's 32 KB window
  nothing to match far back; the ~11x the old figure implies belongs to repetitive text, not to
  English. The paragraph now carries the measurement and the method.
- **The obvious claim about *when* it loads is also false, and my own comment said it before I
  checked.** "A reader who never opens a stock never pays for it" - the idle pre-warm
  (`sheetPrewarm`) lays the top stock's sheet out on an untouched page, so it pulls the file.
  Measured request order: `dashboard_data.js`, `dashboard_context.js`, `dashboard_detail.js`. Kept,
  because the pre-warm exists to make the first open instant and that now includes the text; the
  code, the docstring and the test say so, and the test asserts an **order** rather than a timing,
  so it is true on any connection.
- **The mutation standard** (`plan/claims-mutation-audit.md`, nine passes; the 2026-11-07 per-band
  gate): eight mutations a reader could plausibly ship - the split copying instead of moving, every
  stock carried including the empty ones, the file loaded from the start-up block, the once-only
  guard lost, the waiting state back to an empty block, the loader not merging into `stock_detail`,
  and each runner keeping the file while turning its parse off - **8 of 8 caught**
  (`research/measurements/2026-11-08-detail-split-mutations.py`, committed; it restores every file in
  a `finally` and refuses to run on a dirty tree).
- **One of those eight got through on the first pass, and it is a hole one level inside the
  2026-10-16 tripwire.** That tripwire asks whether every `.js` file the data loop publishes
  *appears* in both runners. Keep `$detPath` in the spec and set its expected-assignment regex to
  `''` and both runners read it as "not a payload": no header check and **no `node --check`**.
  Measured with that one change: **40 passed**. The runners' checks are now derived from the
  generator's own write lines - the only place that decides what a payload file opens with - and
  each global must appear as an anchored regex in both specs. `tests/test_payload_parse_gate.py`.

### Methodology changed

- **None.** The business description is display-only prose by the 2026-08-26 decision; nothing
  scores or ranks it, and a test in the new module re-asserts that on the new file (no `DETAIL_KEYS`
  member in `METRIC_COLS`, `raw` or `pct`, and every value a string).

### Tried and rejected

- **Moving `inp`, `raw`, `pct` and `summary` in the same commit** - 42.6% more of first paint, and
  the temptation. Each is read outside the sheet (compare, movers, the calculation traces), so each
  needs its own read-site audit and its own browser test; shipping them together would have put the
  numbers the page must prove it can reproduce behind an async fetch on the day the mechanism was
  built. Next slice is `financials` + `self_metrics`, written into the plan.
- **One file per stock (~499 small files) instead of one file.** It is the right end state - per-open
  cost flat as the universe grows - and wrong today: the page reads *other* stocks' detail inside the
  sheet (`buildPeerRow`, `peerContext`), so per-stock parts would need several fetches per open, and
  499 artifacts rewritten every weekday is a publish surface this had no measurement for. One file
  now; **bucketing (~32 stocks a part) is the step that makes the shard flat in the universe**, and
  it is a localized change once every read site goes through the loader. Recorded in the plan, with
  the honest statement that today's shard is "off first paint", not yet "flat in the universe".
- **Putting the prose in `dashboard_context.js`**, which already loads after first paint and is
  already gated - no new publish surface, same -307 KB. Refused: that file *is* the context layer
  (CLAUDE.md row "ctx", `plan/context-layer.md`), it loads on every page view whether or not a sheet
  opens, and it would have had to be undone to build the shard the plan actually asks for.

### Next

- **The shard's next slice: `financials` + `self_metrics`** (4.8% + 1.6% of first paint), which are
  sheet-only once `buildPeerRow` is counted as sheet - audit its read sites, move them through
  `DETAIL_KEYS`, and keep the bot contract's `stock_detail[t].price` where it is.
- Then phase 1's `config.yaml` `universe:` block, and **bucketing** the detail file (~32 stocks a
  part) - the step that turns "off first paint" into "flat in the universe" for phase 4.
- The next lab experiment built needs its own second reader; 19 stands.

## 2026-11-08 (13:00 session) - QUEUE: the shard's second slice, and the half of it that needed deleting rather than moving

**Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-08_060001-weekend.log`
(completed). Data loop published? yes - `logs/datarun-2026-11-06_183427.log` (completed; HEALTH: PASS).
Evidence base at `1m`: **45 rows, newest 2026-10-06 (33 days ago, bound 40), 5 effective** of the 8
the engine's gate needs. Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13,
The Factor Lab** (age **27 days**); also open: 0.10 32d, 0.14 25d.
**Tests:** before 4218 passed / 1 skipped (209 s, parallel); after **4232 passed / 1 skipped**
(206 s) - the 14 net new ones are this work's. Gate 2 clean, gate 5 **PASSED** with all five
published files byte-identical to what the next data run would write.
**Owner queue / rotation:** `write_brief.open_proposals()` is empty - no proposal waiting. Nothing
held back by the gates (`origin/nightly/2026-11-08` was level with `main`). Carried on from the
06:00 session's **Next**, which named the shard's next field group, under the same top open
`OWNER_FOCUS.md` item (universe expansion, `plan/universe-expansion.md`). Nothing deferred for a
gate or a stalled loop. 0.13 (the lab, 27 days) deferred again - the owner queue outranks the
roadmap and the lab has no experiment waiting on a reader. The stash `auto-rescue 2026-10-28_060001`
still shows an empty diff; left alone, as the last four sessions did.

### Did

**First paint fell 1.240 -> 1.161 MB gzipped (-6.4%) in two steps, and 1.547 -> 1.161 MB (-25.0%)
over the day with the morning's split.** Nothing published moved: no number, score, rank or
sentence. The two steps are different in kind and the first is the one worth reading.

**1. The peer table's rows all go through one function; `self_metrics` is deleted, not sharded.**
The morning's `Next` named `financials` + `self_metrics` as the next slice. `self_metrics` did not
need moving, because it held nothing of its own. The nine-column peer table built the open stock's
row from that precomputed block and **every peer's row from `buildPeerRow` in JS** over `raw` +
`financials` - and every stock sits on both sides of that split, being the subject of its own sheet
and a peer in its sector-mates'. So one company's figures could print two ways on two sheets.

- **Measured before touching it** (`research/measurements/2026-11-08-peer-row-two-paths.py`,
  committed): **0 of 4,491 cells disagreed** on the 2026-11-06 payload - all nine columns, all 499
  stocks - so no published figure moved. The reason to remove it anyway is CLAUDE.md row 0.8 (the
  page shows the engine's numbers, never a second derivation) and the fact that **one path cannot
  disagree with itself**. The two expressions really were different code: `round(1.0 /
  earnings_yield, 1)` on the unrounded yield in Python, round-half-to-even, against
  `Math.round(10.0 / ey) / 10` on the published 4dp yield in JS, half away from zero.
- **-17 KB of first paint for figures the payload already carried**, with no fetch, no waiting
  state and nothing new served on demand - the cheapest slice in phase 1's queue, and the only one
  that makes the page *simpler* rather than more conditional.
- The panel's empty state now asks the question its sentence makes - whether any of the nine
  columns has a figure, not whether one block exists. Unreachable on a good run either way, which
  the new module asserts as its own premise (0 of 499; the thinnest stock carries six columns).
- `tests/test_peer_row_one_path.py`, 8 tests. The one that would have caught the duplication reads
  one company's row from **its own sheet** and the same company's row from **another stock's
  sheet** - added there the way a reader adds a name - and requires the printed cells to be
  character-identical.

**2. `financials` moves into `dashboard_detail.js`: -62 KB, -5.1%.** What this slice actually built
is a **mechanism**, and that is the part the next session needs: the shard could not carry a
*number* until the loader could redraw the panels that print one.

- The loader called `renderAbout` and nothing else, so a figure the Company snapshot or the peer
  table reads would have sat in its waiting state until the reader closed and reopened the sheet.
  It redraws both now; the reader's custom peer list survives.
- Both panels say which state they are in. The snapshot says "Loading the financial figures..." and
  keeps "No financial figures were published for this stock in this run" for the case where the
  file landed without the block. The peer table's three `financials`-fed cells show **one
  ellipsis** rather than an em-dash, which in a column of figures means "this run published none";
  the other six columns come from `raw` and are on screen at first paint.
- So the membership rule is **restated, not broken**. It was "one read site, inside the sheet, and
  no number", which described what made `about` safe. It is now: *a field may move when every
  surface that reads it is the open sheet, and every panel that prints it can say honestly that it
  is waiting.* `financials` passes both - and only became a single-reader field **today**, because
  until step 1 the subject's peer row read `self_metrics`.

**Looked at the live page, desktop 1440 and 375px, before and after.** JPM's sheet: the Sector
peers table draws its YOU row (#402, $888.3B, 13.5x, +8.1%, 33.2%, ROE 18.0%, D/E em-dash, 2.0%,
FCF em-dash) above V, MA, BRK-B, BAC, MS and a peer median, every figure present, no ellipsis left
behind once the file lands, and the em-dashes are the ones the payload genuinely has no value for.
The Company snapshot above it shows Dividend Yield 2.0% "declared annual rate", Payout 25.8%,
Shares out 2.66B, Avg daily $ vol $2.4B, Short interest 2.9 days to cover. At 375px the table
scrolls horizontally as before and the YOU row keeps its marker. No page errors at either width.

### Evidence / research

- **The duplication measured rather than argued** (above): 0 of 4,491 cells, and a separate check
  that Python's own `round()` on the *published* 4dp earnings yield reproduces the published P/E
  for all 499 - so the second input was not diverging either. An honest "no live defect, real
  duplication", which is why the commit leads with row 0.8 and not with a bug.
- **Where first paint's weight is now** (re-measured with this morning's committed script): after
  both steps, `inp` 19.9%, `raw` 19.0%, `pct` 12.1%, `summary` 11.4% - the four that are 42.6% of
  what is left and are read outside the sheet.
- **The mutation standard** (`plan/claims-mutation-audit.md`, nine passes):
  `research/measurements/2026-11-08-peer-row-mutations.py`, committed - **7 of 7 refused**. Each
  row regenerates `index.html` and the payload in a subprocess before the tests run, because
  mutating the committed page is caught by the drift tripwire and the 2026-10-31 pass established
  that a drift catch is **not** arithmetic coverage: it is green in exactly the state the change
  ships in.
- **The one that got through first, and why it was right to.** `const self = s.self_metrics ||
  buildPeerRow(ticker)` - the JS half alone - is an **equivalent mutant**: with no such block in
  the payload the fallback is all that ever runs. It is also a loaded one, because a later session
  reintroducing the block would find the page already preferring it. Closed by a check that reads
  the generated page for a property *access* (the code comments name the removed block on purpose,
  so a regex on the word would have been satisfied by prose), and the pair applied together is now
  its own row: 5 tests refuse it.

### Methodology changed

- **None.** No weight, metric, threshold, score, rank or published sentence moved. `financials` is
  display-only by construction and a test re-asserts it on the new file (no `DETAIL_KEYS` member in
  `METRIC_COLS`, `raw` or `pct`).

### Tried and rejected

- **Sharding `self_metrics` as the plan's `Next` said.** It would have put a second copy of nine
  figures behind a fetch while the first copy stayed at first paint in `raw` and `financials`. The
  measurement is what turned the step from "move it" into "delete it" - recorded in
  `plan/universe-expansion.md` so the correction is not lost.
- **Moving `inp`/`raw`/`pct`/`summary` in the same session.** 42.6% of what is left and the
  temptation. Each is read by the movers strip, the compare drawer or the calculation traces -
  surfaces that are not the sheet and are live at first paint - so each needs its own read-site
  audit. The mechanism built today is what they move through.
- **Putting the detail wait inside `tests/sheet_warm.settle()`**, which would have fixed every
  module at once. Refused: `settle()` *stops* the pre-warm, which is the only thing that would have
  requested the file, so a module that does not copy `dashboard_detail.js` would hang for 60 s
  instead of failing. Fixed in the one fixture that needed it, which has to **ask** for the file
  explicitly - the first attempt turned 11 failures into 74 fixture errors by waiting without
  asking.

### One thing the next session should know: the browser tests are load-flaky under xdist

Five full parallel runs were taken today on **identical code** - the only commit between them was
prose (the log and the plans). They came back: **clean (4218, baseline), clean (4232), 3 setup
errors in `tests/test_track_open_period.py`, 1 failure in
`tests/test_dashboard_interaction.py::test_a_saved_name_outside_this_run_renders_and_is_kept`,
1 failure in
`tests/test_lab_reading_and_access.py::test_on_a_phone_the_overview_scans_as_a_list_with_every_experiment_on_it`,
clean (4232).** Three different modules, one occurrence each, and **every one passed on re-run both
serially and under `-n 4` in isolation.**

It is not this change. The third module is the Factor Lab's overview page, which neither reads nor
renders anything touched today - it does not load `dashboard_detail.js` at all - and CLAUDE.md
already records its font-timing sensitivity ("`document.fonts.status` reports before it is true -
reading early made the same commit pass serially and fail under xdist"). The common factor is
machine load: runs 3-5 were taken while this session was also driving Chromium for screenshots and
for the mutation harness.

**Why it is worth writing down anyway.** A one-in-three chance of a red gate-1 baseline that nobody
can reproduce costs the next session most of an hour - which is exactly what happened on 2026-11-05
and is why a red baseline now keeps its output. Three distinct modules in three consecutive runs
says the exposure is broader than the two faults already fixed by name. The measurement a session
could take: run the browser modules alone at increasing `-n` and find where the first timeout
appears, then either raise those fixtures' timeouts or mark the browser modules to one xdist group.
Not started today - it is a session's work, not a tail-end fix.

### Next

- **Phase 1's `config.yaml` `universe:` block**, then the shard's next group (`inp` + `raw`, or
  `summary` alone - whichever's read-site audit comes back cleanest). **Grep `tests/` for the field
  before moving it:** `tests/examples.py` reads `financials` to pick the stock four publish-gate
  modules open, and three helpers had to start reading the union of the two payload files today.
- Then **bucketing** the detail file (~32 stocks a part) - the step that turns "off first paint"
  into "flat in the universe" for phase 4.
- The next lab experiment built needs its own second reader; 19 stands.

## 2026-11-09 - RESEARCH (deferred to the stalled data loop first, then written)

**Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-08_130001-midday.log`
(completed). Data loop published? **NO** - `logs/datarun-2026-11-09_020001.log`: HEALTH: PASS, then
the published-claims gate failed on 2 tests and the run was discarded, so the live site still serves
the page of **2026-11-06 18:34**. Evidence base at `1m`: **45 rows, newest 2026-10-06 (34 days ago,
bound 40), 5 effective of the 8 needed**. Priority 0: `allow_auto_apply` **false**. Top open roadmap
item: **0.13, The Factor Lab** (age **28 days**); also open 0.10 (33d), 0.14 (26d).
**Tests:** before **1 failed, 4,231 passed, 1 skipped** (a RED baseline I did not cause); after
**4,243 passed, 1 skipped**. Gates 2, 3, 4 pass; gate 5 (`rehearse_publish.py`) PASSED.
**Owner queue / rotation:** `OWNER_FOCUS.md` item 1 is "decide the OPEN proposals" and there were
**none** at session start (`write_brief.open_proposals()` returned `[]`), so nothing was deferred
there. The data loop outranked the Monday focus (CLAUDE.md: a stalled loop is the highest priority
work available); the focus was then served by writing the diagnosis up properly. **Deferred: 0.13,
0.10 and 0.14, for the stalled loop** - and 0.13 has now been deferred by three consecutive
sessions, so the deferral-streak rule binds on the next session carrying no owner item.

### Did

**1. Diagnosed and fixed the two gate failures that discarded the 02:00 run.** Both were tests
whose premise had broken, not production defects - but a stalled publish is a stalled publish.

**(a) `test_share_classes_page.py::test_the_price_gap_is_the_only_valuation_difference`.** It
asserted FOXA's and FOX's `earnings_yield` differ by exactly the ratio of their prices, and read
0.6650 against 0.8929. This one was **also red in the baseline**, because it reads the newest `runs/`
folder. Per CLAUDE.md's "data->red" rule I asked first whether the claim was false. Established from
the filings, not inferred: **Fox completed its acquisition of Roku on 2026-11-06** (8-K of that date,
"the Closing Date"; S-8 and S-8 POS the same day), the 10-Q filed 2026-11-02 puts Class A at
201,552,650 and Class B at 220,248,764, and the source added the **144,500,000** newly issued Class A
shares to FOXA (market value on 566,301,414 shares) while leaving **FOX's market value on the
pre-merger 421,801,414**. Two records, two share counts, so the price ratio stopped being the
market-value ratio.
**The claim was not false.** `02_outliers_flagged` keeps every fetched line and the second line is
dropped before the percentiles, so FOX's `earnings_yield` is never scored, ranked or published, and
the page's sentence is about the two prices, both current. The test was asserting a property of a
**discarded intermediate**. Rewritten to assert the claim and its premise: `earnings_yield`'s ratio
must be the market-value gap and `dividend_yield`'s the price gap (both hold on either basis - they
are what says the filings behind the two lines are one, and only one metric was checked before);
those two are one number only while both records count the same shares, which is now tested rather
than assumed; and the fact that makes a divergence harmless is asserted rather than tolerated -
**exactly one line of the company is ranked**. 7 new tests, 27 in the module, every branch exercised
on chosen inputs because the run-reading test can only reach the branch today's data is in. The six
audit tests joined the claim's `checked_by`, so the publish gate runs them.

**(b) `test_lab_reading_and_access.py::test_on_a_phone_the_overview_scans_as_a_list_...`.** Passed
in my baseline, failed in the publish gate - the flake class CLAUDE.md already records for this
module's sibling test. **Measured, and it is not a flake, it is a wrong reading:** the bound reads
**4,429 px** and the settled height is **4,518 px**, the same 89 px in **6 of 6 trials**, with
`document.fonts.status` already "loaded" at the early read in 5 of them. So the fixed `< 4500`
ceiling **had been breached since experiment 19 shipped on 2026-11-06 and nothing could see it**;
the gate's xdist run landed after the reflow. The 2026-11-06 session fixed exactly this in one test
and left its sibling unsettled. Fixed both ways: the settle loop is now a shared `_settled` helper,
and the fixed ceiling becomes a **per-card budget** for the reason the three tables carry one - the
list is one card per experiment, so an absolute ceiling bounds how many experiments the lab may
hold. 4,518 px settled over 19 cards is 4,049 of cards (213 a card, tallest 247) and 469 of chrome;
bound 300 a card, between today's figure and the ~412 the 2026-10-27 pass removed, with the chrome
bounded separately so a growing list cannot hide in an average. **Both halves exercised**: `_settled`
is tried on a card that grows 400 ms after the first read (a bare read still returns the old
figure), and the budget is shown to break under 330 px of minimum card height - settling a page that
has already finished passes whatever the helper does.

**Verified the loop will publish:** `scripts/rehearse_publish.py` PASSED end to end -
`check_published_claims` 2,334 passed / 1 skipped, page tests 381 passed, all five published files
restored byte for byte. **I did not republish the payload myself**: that is `data-run.ps1`'s job and
the code loop must never write or stage `signals.json` (settled row "bot" - the asymmetry *is* the
mechanism). The next 02:00 run publishes; the bot declines a stale `signals.json` by design in the
meantime.

**2. The research note** - `research/2026-11-09-share-counts-and-corporate-actions.md`, with
`research/measurements/2026-11-09-post-acquisition-market-value.py` committed beside it.

### Evidence / research

**Measured, from this project's own runs and from EDGAR:**
- FOXA across the two runs either side of the close, **price identical**: market value +34.3%,
  `earnings_yield` **-25.5%**, `fcf_yield` -23.1%, `ev_ebitda` and `ev_sales` **+30.1%**, composite
  63.19 -> 60.98, **rank 34 -> 60**. A 26-place move from one input, no price change, no new filing.
- The signature (implied share count `marketCap/currentPrice` moving >5%) occurs **6 times in 19 run
  pairs, 0.32 a run, and in none of thirteen consecutive pairs** - episodic, not noise. Three line up
  with an 8-K read on EDGAR (KMB +84.5%, CHTR +37.8%, FOXA +34.3%); **three I did not establish**
  (CVNA, PCG, CPT) and two of those appeared between two runs on the *same day*, which points at the
  source. Marked unestablished rather than assumed.
- **Nothing saw it.** That run's `check_run_health` was `ok` on every line including rank continuity
  (Spearman 0.997, 2 new in the top 25) - correctly: one stock moving 26 places does not dent a
  499-name Spearman and renormalisation preserves dispersion.

**Documented practice, read from the PDFs rather than recalled:**
- **MSCI, *Corporate Events Methodology*, February 2026** (105 pp.): p. 5, corporate-event changes -
  "changes in number of shares (NOS)" among them - are implemented "**simultaneously with the
  event**"; s2.6 "**Pro Forma Float Calculation for M&As**" pp. 16-17; s2.7 p. 17 - a cash-only deal
  leaves the acquirer's weight unchanged, a share-issuing one does not; the `NOS Inflow Ratio`
  formula p. 18. Materiality threshold **+-1%** (version history, p. 93).
- **S&P DJI, *Equity Indices Policies & Practices*** pp. 28-29: "M&A activity is implemented when
  the target company is deleted"; off-cycle share changes "of **5% or more** ... are made weekly".
  **Version caveat stated in the note:** S&P's own host served an HTML bot wall instead of the
  current August 2026 file and that was **not** worked around, so an older public copy was read.
- Same document p. 29, "**Multiple Share Classes**": in US indices "each company is represented only
  once ... The stock price is based on one class, usually the most liquid class, and the share count
  is based on the **total shares outstanding**." That is this project's own 2026-10-21 rule, now
  quoted at a page number - **and it explains the failure's root**: S&P applies one share count to
  one price, while the source publishes total-shares x *each line's* price on *both* lines, so the
  two are independent estimates of one quantity and a corporate action reaching one first makes them
  disagree. Structural to the source, reproducible by any dual-class company.

**So the new share count is right and is what any provider would use.** What is missing is that the
**numerator has not caught up**: FOXA's earnings yield is standalone Fox earnings over
Fox-plus-Roku market value for about three months, and the page says nothing - every equation on the
row reproduces and the *reason* is invisible, which is the 2026-10-06 rule ("every number on the page
must be checkable from the page") failing from the other side.

**Academic half marked honestly:** Loughran & Vijh (1997), Rau & Vermaelen (1998) and Shleifer &
Vishny (2003) are cited with full locators and **"- not read"** on all three, and the single sentence
that leans on them is hedged to "a thing the literature has studied". A session that fetches them
can turn the note's first option into a researched answer.

### Methodology changed

**None.** No weight, metric, threshold, score or published sentence moved. The diff is two test
modules, six entries in one claim's `checked_by`, a research note and a measurement script.

### Tried and rejected

- **Widening the failing tolerance** (`rel=1e-4`) to let the run through. Refused: the tolerance was
  not wrong, the thing being measured was. The identity it asserted is exact and is still asserted
  exactly; what changed is that it is now applied to the quantity the claim is about.
- **Option 3 of the note** - dropping the market-value-based Valuation metrics for a stock's
  post-deal stub window. Argued against rather than deferred: it would drop a stock's largest
  Valuation inputs for a quarter on a share-count heuristic, `MIN_CATEGORY_COVERAGE = 0.90` and the
  coverage discount carry the consequences past the one metric, and it discards the correct current
  market value that section 3 shows every provider uses.
- **Re-measuring the lab budget to fit 19 experiments** under the old fixed ceiling. The fixed
  ceiling was the defect - it bounds how many experiments the lab may hold - and the project had
  already reached that conclusion for the three tables on 2026-11-06; this applies the same answer to
  the sibling rather than inventing a second one.
- **Republishing the payload from this session** to get the site current. Refused on settled row
  "bot": only the gated 02:00 run is tradable and a second writer of `signals.json` reopens the hole
  the asymmetry closes.

### Next

- **Decide the open proposal** - `research/2026-11-09-share-counts-and-corporate-actions.md`,
  options 1-3 for a stock whose implied share count moved more than 5% while its earnings predate
  the move. The deciding session must not be this one. My preference is **option 2** (a
  `check_run_health` note, not a gate), because it is the option that produces the counts needed to
  choose between the others and it has the `sec_crosscheck` precedent exactly. Section 5
  pre-registers the detector (`marketCap/currentPrice`, **not** `sharesOutstanding` - that field is
  per class and the market value is not), the 5% threshold taken from S&P rather than from this
  data, counting per run date, and a void condition at >3 stocks a run date over four weeks.
- **0.13 (The Factor Lab) has now been deferred three sessions running.** The deferral-streak rule
  binds: the next session with no owner item and no broken loop must advance it or demote it in
  "Current priorities" with the argument for what outranks it. A fourth defensible deferral is not
  a result.
- Confirm tomorrow's 02:00 run publishes (`logs/datarun-2026-11-10_*.log` ending "Data loop
  complete"). If it does not, the first thing to read is whether the claims gate names a *different*
  test - the two fixed today were independent faults in one night, which is itself worth noticing.

## 2026-11-09 (13:00 session) - QUEUE. The share-count proposal decided, and the detector it asks for found already built and silently broken

**Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-09_060001.log`
(completed). Data loop published? **no** - `logs/datarun-2026-11-09_020001.log` (stopped
deliberately; HEALTH: PASS) - this morning's session diagnosed and fixed both gate failures and
verified the rehearsal, so the next 02:00 run publishes; nothing further was needed today.
Evidence base at `1m`: **5 effective of the 8 needed, 45 rows, newest 1m run date 2026-10-06 (34
days ago, bound 40)**; 114 rows across all horizons. Priority 0: `allow_auto_apply` **false**. Top
open roadmap item: **0.13, The Factor Lab** (age **28 days**); also open 0.10 (33d), 0.14 (26d).
**Tests:** before **4,243 passed / 1 skipped** (226 s parallel, green baseline - `origin/main`
already carried this morning's work and no branch was held back by the gates); after **4,271
passed / 1 skipped**. Gates 1-4 pass; gate 5 (`rehearse_publish.py`) **PASSED**, all five
published files restored byte for byte.
**Owner queue / rotation:** the morning brief listed **one open proposal**, which the prompt and
`OWNER_FOCUS.md` item 1 both make the first job, and I did not write it - so that was the session.
`write_brief.open_proposals()` returns `[]` now. **Deferred: 0.13, 0.10 and 0.14** - 0.13 for the
fourth consecutive session, and this is the deferral-streak rule's own exception rather than a
fourth defensible deferral: the rule binds on "the next session carrying **no owner item**", and
deciding an open proposal is an owner item. **It is the next session's first job if the queue is
still clear.**

### Did

**Decided the open proposal** - `research/2026-11-09-share-counts-and-corporate-actions.md`, which
asked which of three things to do for a stock whose implied share count has moved while the
earnings behind its multiples have not (FOXA, rank 34 -> 60 on 2026-11-09, price unchanged).
**Option 2 accepted in substance and refused as specified; option 3 refused; option 1 left open
with two conditions it did not have.** The argument is the note's new section 8; the measurements
are `research/measurements/2026-11-09-share-count-detector-review.py`, committed.

**1. The finding: the detector the note asks for already existed, and had been comparing
nothing.** `sec_crosscheck.market_value_check` has computed `marketCap / price` - the note's own
`implied_shares`, the same expression - against each company's **filed** weighted-average share
count every run since 2026-10-14, printed in `check_run_health`'s own note with an ATTENTION line
above 60 stocks. The note names `sec_crosscheck` as the precedent to copy without noticing it is
the same check.

From the run logs: `446/485 within 5% (21 over 10%)` on 11-05 and 11-06 02:00, `444/485 (19)` on
11-06 18:34, and **`0/0 within 5% (0 over 10%)` on 11-09**. `financials` moved into
`dashboard_detail.js` on **2026-11-08**; the check reads
`stock_detail[t]["financials"]["market_cap"]`; `load_payload` read `dashboard_data.js` alone. So
from the first run after that commit it compared **0 stocks of 485**, and **an ATTENTION line that
fires above 60 cannot be reached from zero.** Measured on the 11-09 run's own artifacts: 0
compared from `dashboard_data.js`, **485 with the shards folded back, 20 over 10%** - FOXA among
them at ratio **1.3109**, the stock the whole note is about. Fixed: the gate folds every shard
back, derived from the generator's own constants.

**Why nothing saw it, which is the transferable half.** `tests/test_sec_crosscheck.py` builds its
payload by hand with `financials` inline, so it reports the same thing whatever the generator does
with the field - rule 10's lesson (assert against the generator's output, not an artifact) one
level down, in a gate rather than a claim. And the shard's membership rule is written in terms of
the **page's** read sites ("every surface that reads it is the open sheet"), which is true of
`financials` and silent about the payload's readers that are not the page. Both now recorded where
the next session moving a field will read them: the generator's `DETAIL_KEYS` comment and
CLAUDE.md's settled row "gates".

**2. Event or state - the measurement that refuses the proposed detector.** The note
pre-registered a *run-to-run* delta over 5%. Its own section 4 question is a **state** (the
numerator predates the denominator for about three months); a delta is an **event**. Measured on
KMB, whose count moved +84.5% on 2026-11-04: the delta named it on **1** run and went quiet; the
filings comparison named it on **all 5** runs from 11-04 to 11-09, and is still naming it. The
filings anchor is also the only one that can see a stock already stale before the project started
watching.

**3. Named by what is new, not by how far - built, after measuring that the obvious version
fails.** A count is not actionable. But the 8 stocks furthest from their filed count on 11-09 are
**MCD, SKYD, IBKR, TKO, CVNA, VMRK, WAT, MNST** - the standing structure recorded on 2026-10-14 -
and **FOXA is 13th of 485**, so a magnitude list of eight would have missed it. Against the
published page the set is exactly the news: **CHTR and FOXA new, 18 standing, CPT gone.** Shipped
as `check_share_count_moves`, verified end to end on the 11-09 run:

> share counts behind the market values: 20 over 10% from the filed count, 2 of them new since the
> published page, 1 no longer (CPT)
> ATTENTION - 2 stock(s) ... which the published page does not: CHTR +47.7%, FOXA +31.1%.

Never a failure; an `ATTENTION -` line, which is what the brief surfaces, exactly as option 2
asked. Base rate from the 20 runs on file: entrants on **3 of 16 run dates**, at most 2 on any
one. The note's own void condition (>3 a run date) is in the code as `SHARE_COUNT_MOVE_VOID`, and
it uses the existing **10%** line (`sec_crosscheck.FAR`), **not** the note's 5% - 5% is S&P DJI's
threshold for a run-to-run move and is not the right number for a distance from a filed count.

**4. Two test modules, and 14 of 14 mutations refused.**
`tests/test_health_reads_every_shard.py` (11) asserts the fold two ways a hand-built fixture
cannot: a **round trip through the generator's own** `split_context`/`split_detail`, so a fourth
shard or a field moved out of one fails with nobody adding a case, and a derivation from
`data-run.ps1`'s own `$DataArtifacts` - the parse gate's precedent - with an empty, printed
`KNOWN_UNFOLDED`. `tests/test_share_count_moves.py` (17) pins the construct, including the one
finding that could silently regress (a nearer *new* gap is named over a further *standing* one)
and the void condition probed **at** its line rather than either side of it. Mutation-checked by
editing the production file and restoring: **6 of 6** and **8 of 8** refused, restored exactly
both times.

### Evidence / research

- **This project's own runs and logs.** The `0/0` line's first appearance, dated against the
  2026-11-08 commit; 0 compared vs 485 with the shards folded, on the same run's artifacts; KMB
  named by one detector on 1 run and the other on 5; FOXA 13th of 485 by magnitude; CHTR/FOXA new
  and CPT gone against the published page; entrants on 3 of 16 run dates.
- **One observation on the note's own open question.** CPT was over 10% on the published page and
  is not on the 11-09 run - its gap **closed**. A completed acquisition does not reverse, so that
  is evidence (not proof) it was the source refreshing rather than the company, which the note
  suspected and marked unestablished. A reversion is invisible to an event detector; only a state
  detector reports it.
- **No new literature.** The three papers the note marks `- not read` (Loughran & Vijh 1997,
  Rau & Vermaelen 1998, Shleifer & Vishny 2003) are still unread and still marked. The question
  they bear on - whether a mechanically re-priced acquirer is a signal as well as a measurement
  problem - is untouched and remains a Monday.

### Methodology changed

**None.** No weight, metric, threshold, score, scoring line or published sentence moved. The diff
is the health gate, two test modules, a comment in the generator, the note's section 8, the owner
queue and one clause in CLAUDE.md.

### Tried and rejected

- **Building the note's run-to-run detector.** Refused on the measurement above: it answers the
  event where the question is the state, and the state is already computed. Had I built it, the
  project would have carried two detectors of one quantity - the thing the 2026-10-31
  switching-margin finding is about.
- **Naming the stocks by distance from their filed count.** The obvious reporting design, measured
  before shipping: it would not have named the stock the note was written about. This is now a
  test, so it cannot quietly go back.
- **Option 3 (excluding the market-value metrics for the stub window).** Refused with a harder
  reason than the note's: the detector cannot tell a corporate action from a partnership-unit
  structure, 18 of the 20 over the line on an ordinary run are standing, so the exclusion would
  drop **IBKR, TKO, CVNA, MCD and WAT**'s largest Valuation inputs *permanently* rather than for a
  quarter.
- **Option 1 (a sentence on the page) today.** Not refused - deferred with the two conditions it
  needs written down (note section 8.5): the sentence must state the **state** rather than a
  "change" it cannot support, and it needs the discriminator section 8.4 shows is missing, or it
  prints a share-count sentence on five stocks every day for ever and teaches a reader to ignore
  it.
- **Adding the two new modules to the 02:00 publish gate.** Considered because
  `tests/test_input_gates.py` is in it by the same reasoning, and rejected: that module checks
  *payload fields* the gates read, which is a property of what is about to be published, while
  these check the gate's own plumbing - already covered by gate 1, fourteen times a week. They
  would add ~35 s to the publish gate and protect no published artifact.
- **Republishing the payload to make the site current.** Still refused, on settled row "bot", for
  the same reason this morning's session gave.

### Next

- **0.13, The Factor Lab, has now been deferred four sessions running** - three of them to work
  that outranked it (the stalled data loop, then this owner item). The next session with a clear
  proposal queue has it as its first job: experiment 19 is second-read, so the queue in
  `plan/factor-lab.md` wants the next experiment, and Friday 2026-11-13 is the lab's day.
- **Confirm the 02:00 run of 2026-11-10 publishes** (`logs/datarun-2026-11-10_*.log` ending "Data
  loop complete") **and read its two new lines**: `market value / price vs filed shares` should be
  back to ~440/485 rather than `0/0`, and `share counts behind the market values` should report
  **0 new** once FOXA and CHTR are on the published page. If either is wrong, the fold is wrong.
- **Option 1 of the share-count note is open** with its conditions pre-registered (section 8.5).
  It is a page change and needs the 8-K-or-newness discriminator first; do not build the sentence
  without it.

---

## 2026-11-10 - PRODUCT, THROUGH THE NUMBERS. Two refused loops instead: a company's name read as a recommendation, and a weekly cache refresh read as a changed result

**Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-09_130001-midday.log` (completed). Data loop published? **no** - `logs/datarun-2026-11-10_020001.log` (stopped deliberately; HEALTH: PASS). Evidence base at `1m`: 5 of 8 needed at the 1m horizon (45 rows, but overlapping windows are not independent; 114 rows across all horizons), newest 1m run date 2026-10-06 (35 days ago, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab** (age **29 days**); also open: 0.10 34d; 0.14 27d.

**Tests:** before **4,268 passed / 3 skipped / 1 FAILED** (the runner's baseline,
`logs/nightly-2026-11-10_060001.baseline.txt`: `test_ledger_is_level_with_the_code.py::
test_every_published_row_was_produced_by_the_committed_code[content_hash]`) -> after **4,291
passed / 3 skipped / 0 failed**. Gates 1-4 pass. Gate 5 (`scripts/rehearse_publish.py`)
**PASSED** - it failed twice first, and both failures were real; all five published files
restored byte for byte.

**Owner queue / rotation:** `write_brief.open_proposals()` returns `[]`, and `OWNER_FOCUS.md`'s
open items are the long-horizon universe expansion. The nominal focus was Tuesday's product day
and I did not reach it: **two loops were refused and `main` arrived red**, which the prompt and
CLAUDE.md both put ahead of the focus. **Deferred: the product focus, 0.13 (the Factor Lab, 29
days), 0.10 (34d) and 0.14 (27d).** 0.13 is now a **fifth** consecutive deferral; the
deferral-streak rule's exception does not cover a fifth, so I am recording plainly that a red
`main` and two non-publishing loops outranked it rather than claiming the rule was met - see
**Next**.

### Did

**1. The advice-language ban discarded a healthy data run, because Best Buy is a company.**

The 02:00 loop refused to publish on `tests/test_dashboard_navigation.py::
test_reporting_soon_is_a_calendar_not_a_leaderboard`, with `HEALTH: PASS` on every line above it:
499 scored, 499/499 priced, 497/499 with a target, Spearman 0.989 against the published page,
every category dispersion inside its band. Reproduced against the regenerated page
(`scripts/rehearse_publish`'s machinery, one test): the failing assertion is
`advice_terms_in(page.inner_text("#reporting-body")) == []`, and the term found is **`Buy`**, in
**Best Buy Co., Inc.** Best Buy's next report is 2026-11-24 - **exactly +14 days** from that run
date - so it entered the panel's longer window that morning for the first time.

The page was right: a calendar of forthcoming reports prints the company's legal name. The
*check's premise* was wrong - "no company reporting within fourteen days has a banned word in its
name" is a fact about S&P 500 membership and the earnings calendar, and it went false with no
commit touching anything. **Measured: Best Buy is the only name in the universe that trips the
ban** - 1 of 998 names and tickers, over all 499 rows of both the 11-06 and 11-10 payloads - which
is why a check written weeks ago fired on the fourth window it could have fired on.

- `stock_summary.without_data_names()` masks the data's own proper nouns before the ban runs:
  longest first, each replaced by a space so no two words fuse, case-insensitive like the ban. A
  genuine recommendation beside a masked name still matches.
- every advice check over text a *browser* rendered now masks first - Reporting Soon, the stale
  note, the option-quote panel, the track record.
- `tests/test_advice_ban_scope.py` (16 tests) holds the rule as an AST sweep over every test
  module. It follows a bare name argument back to its assignment **in the same scope**, so a local
  variable cannot defeat it; the first version joined lines by bracket depth and reported four
  sites that do not exist, and reading names module-wide produced a fifth. The sweep found a site
  my own hand search had missed. The one check that genuinely cannot mask - a node harness whose
  payload is three synthetic tickers - is named in `EXEMPT` with a reason printed every run, and a
  test fails if an exemption stops being needed.

**The part worth carrying:** my first version read `window.D`. The page declares
`const D = window.SCREENER_DATA || {}` at script scope, so `D` is **not** a property of `window` -
the same shape as the closure-scoped `CTX_LOADED` the dashboard notes already warn about.
`data_names` returned `[]`, the mask was a no-op, and gate 5 failed on exactly the test the data
run had failed on. The fix read as finished while doing nothing. `data_names` now **raises** on an
empty list, and a browser test asserts it reads 998 names for 499 rows off the real page.

**2. `main` was red because a gitignored cache the data loop refreshes weekly had moved.**

The inherited failure: all 32 newest ledger rows carried `content_hash=a7f3c9a52778` against a
live `3fc1fe78c56d`, while `code_hash`, `base_config_hash` and `method_version` were level and no
commit had touched any of the nine `RESULT_PATHS` files since the rows ran on 2026-11-07.

Measured, part by part: of the seven parts of the content fingerprint **exactly one had moved** -
`sec_facts`, `e86b780be015eb5d -> 1b21c3d2593fd1bb`. `month_files` identical
(`21d6dd1cf5f34531`), `month_file_count` 176, `base_run` identical, `v2_daily`, `v2_index`,
`splits` and `sec_facts_predecessor` identical. The mtime says why: the 02:00 run rewrote
`data/sec/pit/facts.parquet` at **02:16:52**, and `sec_fundamentals.FACTS_MAX_AGE_DAYS = 7` makes
that a **weekly** event. So this test turns `main` red roughly once a week, blocking every
session's merge, and prescribes a ~27-minute rerun.

The filings reach a row only **through** the monthly records, which `month_files` already hashes:
`backtest_v2._load_facts` reads them at build time, and the only `DATA_SOURCES` file read at
*experiment* time is `cache/v2_daily.parquet` (`v2_experiments` 215 and 913,
`backtest_v2_returns` 129). The months were not rebuilt, so no number could have changed - and the
rerun it prescribes re-stamps the rows while reading byte-identical inputs. It does not restore
reproducibility from the overwritten cache either, so **it cannot fix what the failure claims is
broken.**

- `content_hash` is now taken over `v2_experiments.DECIDING_PARTS`: the monthly records, their
  count, the base run, and `SOURCES_READ_AT_RUN`.
- `content_parts` still records **all five** sources, so every row still names the exact filings
  and prices its months were built from. The verdict is rescoped, nothing is dropped (rule 7).
- a build input that moved while the months did not is reported by `provenance_note` as
  information, in the same shape as the existing locator line, and says plainly that no number
  moved. It never says "rerun".
- `tests/test_ledger_is_level_with_the_code.py` holds all three halves: a run-time source still
  stales every row; a build-only source does not, but must still appear in `content_parts`; and a
  rebuilt **or added** month still stales everything - without that last one the split is a hole.

This is the argument `provenance_note` already won on 2026-10-18 for `code_version` against
`code_hash`, one layer down: **flag on the thing that can change a number, not on the thing that
merely moves.**

The rerun was needed either way (my edit moves `code_hash`), so it ran: all 32 rows, `--jobs 4`,
six at a time in foreground calls of about 5.5 minutes each, committed after every call. `STALE
REMAINING: 0`, report and `research.html` regenerated.

### Evidence / research

- **The Best Buy measurement.** 1 of 998 published names and tickers trips
  `stock_summary.BANNED_TERMS`, in both the 2026-11-06 and the regenerated 2026-11-10 payload; the
  window arithmetic is `2026-11-24 − 2026-11-10 = 14`, the panel's own bound. Reproduced and then
  re-verified green against the page the next data run will publish.
- **The fingerprint measurement.** `research/measurements/2026-11-10-filings-cache-rerun-moved-
  nothing.py`, reusing 2026-11-05's comparison rather than restating it: **32 ids compared, 27,225
  statistics, 0 differences.** The prediction - every statistic identical, only the provenance
  moves - was written in that script's docstring before the comparison ran. So the weekly filings
  rewrite staled every published row and changed no published number.
- Read rather than assumed: `sec_fundamentals.FACTS_MAX_AGE_DAYS = 7` for the cadence, and the
  three read sites for `v2_daily` that put it on the deciding side.
- No literature today. Neither change is methodology: no weight, metric, threshold or score moved,
  so **no `METHODOLOGY_CHANGELOG.md` entry** (rule 3 governs methodology changes; these are a test
  scope and a provenance verdict).

### Methodology changed

- None. Deliberately: both fixes are about what a check is allowed to conclude, and no number the
  page publishes moved. The ledger rerun reproduced all 27,225 statistics exactly.

### Tried and rejected

- **Accepting the weekly rerun instead of rescoping the verdict.** It is the prescribed remedy and
  it would have been green by 07:00. Rejected because it does not address what the failure
  detects: the months were not rebuilt, so the rerun reads byte-identical inputs and only
  re-stamps provenance - and the condition recurs every week, with a red `main` blocking every
  session's merge each time. The 27,225-statistic comparison is the evidence, and the same
  argument is already written into `provenance_note` for `code_version`.
- **Requiring every advice check to mask, with no exemptions.** The node-harness holdings test has
  no page to read names from, so masking there would pass an empty list - the exact no-op this
  work exists to prevent. A named `EXEMPT` entry with a printed reason is honest; a vacuous call
  is not.
- **Narrowing `content_hash` to `month_files` alone.** `v2_daily` is read at experiment time by
  three sites, so a change to it genuinely moves forward returns. It stays in the verdict, and
  `v2_index`/`splits` are flagged in the code as the two to move back the moment an experiment
  starts reading one.

### Next

1. **Tuesday's product day was not worked and 0.13 has now been deferred five sessions running.**
   The next session with a clear proposal queue, a green `main` and a published data loop should
   take **0.13, the Factor Lab** - experiment 19 also still has no second reader
   (`OWNER_FOCUS.md` 2026-10-20 item 2) - or demote it in "Current priorities" with the argument,
   per the deferral-streak rule. Do not defer it a sixth time without doing one of those two.
2. **Watch the 2026-11-11 02:00 run.** The live page has been 2026-11-06's data since Monday -
   three refused runs in a row (11-06's catch-up, 11-09, 11-10). Gate 5 passes now and the
   Reporting Soon failure is fixed, so it should publish; if it does not, read the new failure
   before anything else. I did not publish from this session on purpose: `data-run.ps1` is the
   only writer of `signals.json` and the only path with the seven data-loop gates in front of it
   (settled row "bot"), and a code session publishing data would reopen that asymmetry.
3. **`COMING_UP`'s 2026-11-09 line stays listed, correctly.** The context signals' first
   one-month reading arrives with a run that *publishes*, and the 11-09 and 11-10 runs were both
   discarded, so no reading has landed. Delete the line only once a published run's
   `context_eval` shows it.
4. The 2026-11-16 pre-registered filing cross-check table is six days out and unaffected by any
   of this.


## 2026-11-10 (owner-run, evening) - the site stayed stale because a refused run waits a day; it no longer does

**Health (rule 8, all five):** last code session ran? **no** - `logs/nightly-2026-11-10_130001-midday.log`:
"SESSION DID NOT RUN: ... Claude Code 2.1.229 does not support this model; version 2.1.290 or newer is
required" (the 06:00 session before it shipped). Data loop published? **no** - `logs/datarun-2026-11-10_020001.log`,
HEALTH: PASS, then the published-claims gate refused it (`test_reporting_soon_is_a_calendar_not_a_leaderboard`,
Best Buy's name read as advice; fixed by the 06:00 session). The live page is still the **2026-11-06 18:34**
run. Evidence base at `1m`: **45 rows, newest 2026-10-06 (35 days ago, bound 40)**. Priority 0:
`allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab** (age **29 days**).
**Owner direction (chat):** *"how did the run go this morning, why did the dashboard not update. make sure
this doesnt happen again"*.

### What happened, and the pattern

Three of the last five weekday data runs (11-06, 11-09, 11-10) were refused by the published-claims
gate, each for a fault in a test's premise (a rank read by position; a merger splitting Fox's share
counts; a company name read as advice), each fixed by the next 06:00 session - and each time the site
waited for the next 02:00 run anyway. By today it was three trading days stale. The gate did its job;
**the gap was that nothing re-ran the loop once the fix had landed.** Separately the CLI had not updated
itself, and the 13:00 session died on the version check - as tomorrow's 06:00 would have.

### Did

- **`Screener Data Retry`, 18:00 Mon-Fri, `data-run.ps1 -Retry`.** The data loop's success marker makes
  it a no-op on a day that published. On a refused day it reruns every gate on that day's close. It
  stands down on an owner session (before anything), and on a dirty tree or a branch off `main` (under
  the repo lock, before the 02:00 run's stash-and-checkout rescue, writing no brief) - so it cannot
  touch work in progress. No logon trigger, no late catch-up. `tests/test_data_retry.py`, 9 tests; every
  guard removed in turn fails one. The 02:00 run is unchanged.
- **The code runner updates the CLI and retries the session once** when the API says the CLI is too
  old for the model (`npm install -g @anthropic-ai/claude-code@latest`; one retry, only on a successful
  update).
- **Updated the CLI on this PC now:** 2.1.223 -> 2.1.322, and confirmed a `--print` session runs on the
  configured model. (npm skipped the package's postinstall script; the binary runs regardless.)
- CLAUDE.md settled row -1 and "The two loops" say both.

### Registered and exercised (with the owner's go-ahead, same evening)

- `scripts/register-tasks.ps1` run: six tasks Ready; `Screener Data Retry` runs
  `data-run.ps1 -Retry` at 18:00 Mon-Fri (`schtasks /query /v` checked).
- The retry task was started by hand at 19:37 and took the real path: "Same-day retry: today's 02:00
  run did not publish", a full run (`logs/datarun-2026-11-10_193759.log`): HEALTH: PASS, price dates
  **499/499 on 2026-11-10**, Spearman 0.985 vs the 11-06 page, PUBLISHED CLAIMS: PASS (2335), signals
  written for run `4108d8310b22`, **published 20:07**. The live page reads "PRICES AT THE NOV 10 CLOSE"
  at 1440 and 375, no errors, no sideways scroll.
- **Not exercised:** the stand-down branches (owner session, dirty tree, branch off main) - pinned by
  `tests/test_data_retry.py`, not yet seen in a real run.


## 2026-11-16 (owner-run, evening) - eleven code sessions lost to a CLI I installed in the wrong place; the two refusals fixed

**Health (rule 8, all five):** last code session ran? **no** - every one from 11-11 07:38 to 11-16 13:00
(eleven) aborted: "Required tool 'claude' is not on PATH". Data loop published? **yes** -
`logs/datarun-2026-11-16_180000.log` (the 18:00 retry, after the 02:00 run was refused). Evidence base
at `1m`: **51 rows, newest 2026-10-14 (33 days ago, bound 40)**. Priority 0: `allow_auto_apply`
**false**. Top open roadmap item: **0.13, The Factor Lab** (age **35 days**).

### What happened

1. **The CLI.** On 11-10 the owner-run session ran `npm install -g @anthropic-ai/claude-code@latest`
   from inside the Claude desktop app. That app is sandboxed: it keeps its own copy of `AppData\Roaming`
   (the files carry an `S-1-15-3-` app-container ACL). The update landed in the sandbox's copy; in the
   real folder the old version was removed and nothing replaced it - a scheduled task listed
   `%APPDATA%
pm` as `node_modules` alone. So every code session since aborted at its tool check.
   **My mistake, 11-10.** Reproduced with a one-off task; fixed by reinstalling from a one-off task
   (real context): 2.1.326, `Get-Command claude` and a `--print` session both work from a task.
   The diagnostic task is deleted.
2. **Smart App Control** blocked pyarrow's `parquet.dll` on 11-11 after KB5132061 (Code Integrity 3033/
   3077). It loads again by 11-16 with SAC still on - presumably reputation caught up. Not changed by us.
3. **Refusals:** 11-11 18:00 and 11-16 02:00 on the What-changed footnote's hand-worked example
   (it needed three movers; a window between two runs on one close had one); 11-12 02:00 and 18:00 on
   the phone side-by-side with four stocks (Expedia 4th: "Discretionary" was 72px in a 70px column,
   a 2px sideways scroll - a real page defect). 11-13 02:00 and 11-16 18:00 published.

### Did

- **Runner reinstalls a missing CLI once before its tool check aborts** (`tests/test_data_retry.py`).
- **Footnote test** builds its example from real entries padded to three.
- **Side-by-side sector label**: 10.5px at <= 420px so the longest sector word fits whole, plus
  `hyphens: auto` and `overflow-wrap: anywhere` as the guarantee (`break-word` does not lower min-
  content). Measured on both 11-12 runs: scroll width 307 -> 305 = client width. New test case
  `worst`: the four stocks with the longest sector/company words, whatever ranks top; it fails on the
  old CSS on today's data.
- Memory: `reference_desktop_app_sandboxed_appdata.md` - never install the loops' tools from a
  desktop-app session.

## 2026-11-17 (owner-run, after midnight) - an SEO playbook, triaged: the publish is now checked on the live site

**Health (rule 8, all five):** last code session ran? **no** - 2026-11-16 13:00 aborted on the missing
CLI, fixed that evening (the next real test is 06:00 today). Data loop published? **yes** -
`logs/datarun-2026-11-16_180000.log`, the 18:00 retry, live site confirmed serving
`2026-11-16T18:00:03` by the new check. Evidence base at `1m`: **51 rows, newest 2026-10-14 (34 days
ago, bound 40), 5 effective observations**. Priority 0: `allow_auto_apply` **false**. Top open roadmap
item: **0.13, The Factor Lab** (opened 2026-10-12, age **36 days**).

The owner pasted a long tiered SEO/discovery playbook and asked to take what applies. Audited the
live site first: most of the technical tier was already here (descriptions, sharing cards, 404,
cross-links, footers). The triage, with what does not fit and why, is `plan/site-discovery.md`.

### Shipped

1. **`scripts/check_live_site.py`, called by `data-run.ps1` right after the push.** The loop said
   "live dashboard refreshed" because `git push` exited 0; a failed Pages deploy afterwards was
   invisible to every log (the owner's GitHub failure emails in early November). It range-requests
   the live payload's first 400 bytes (cache-busted - Pages' CDN holds `max-age=600`), compares the
   `run_timestamp` with the one pushed, waits up to 20 min, prints one line and always exits 0.
   A stale site logs `ATTENTION - LIVE SITE NOT UPDATED` at WARN, which `write_brief.outcome` already
   surfaces. Verified against the real site: serving the 18:00 run. `tests/test_live_site_check.py`
   (9, fake clock, no network).
2. **Canonical URLs** on every page (`/` for both `index.html` and `dashboard.html`), equal to `og:url`;
   **`sitemap.xml`**; **`llms.txt`** (the page shows a JS-less reader ~8 KB of frame and no ticker -
   measured - so this points readers at the methodology and payload); one **`WebApplication` JSON-LD**
   block that repeats the meta description and claims nothing the page does not say; the research
   page's `<title>` now matches its `<h1>`. `tests/test_site_discovery.py` (8). Pages rebuilt with
   `scripts/rebuild_page.py` (payload byte-identical) and `python v2_experiments.py report`.
3. **Not done, on purpose:** `robots.txt` (a project site's is never read - the host root's returns
   404, so all is crawlable), outreach of any kind (publishing in the owner's name), pricing/trial/
   competitor pages (nothing is sold), trend-chasing (rule 4). Owner decisions queued in
   `OWNER_FOCUS.md`: Search Console, a byline, analytics.

### Next

- The 06:00 session today is the first since the CLI fix; its log is the proof.
- After the next real data run, read the `Live site:` line in `logs/datarun-*.log`.


## 2026-11-17 - PRODUCT, THROUGH THE NUMBERS. Sixteen stocks were scored on last year's statements and the page dated them last quarter

**Health (rule 8, all five):** last code session ran? **failed** - `logs/nightly-2026-11-16_130000-midday.log` (failed: Required tool 'claude' is not on PATH. Aborting.). Data loop published? yes - `logs/datarun-2026-11-17_020000.log` (completed; HEALTH: PASS). Evidence base at `1m`: 5 of 8 needed at the 1m horizon (53 rows, but overlapping windows are not independent; 130 rows across all horizons), newest 1m run date 2026-10-16 (32 days ago, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab** (age **36 days**); also open: 0.10 41d; 0.14 34d.

This is the first code session to run since 2026-11-10: the eleven between aborted on the missing
CLI (fixed by the owner-run session on 11-16). The runner started it normally.

**Tests:** before **4,319 passed / 2 skipped / 1 FAILED** (the runner's baseline,
`logs/nightly-2026-11-17_060001.baseline.txt`) -> after: **4,334 passed / 2 skipped / 0 failed** (15 new tests). Gate 2 (dry-run) exits normally; gate 3: the three payload files are byte-identical to the base commit and `index.html` is 943 KB (the tool guard refuses `node` in this session, so the parse itself is the runner's and `tests/test_payload_parse_gate.py`'s); gate 4 clean; gate 5 (`scripts/rehearse_publish.py --changed-since 77c95f8`) **PASSED**, published files restored byte for byte.

**Owner queue / rotation:** `write_brief.open_proposals()` returns `[]`. `OWNER_FOCUS.md`'s three
open items are the owner's own decisions (Search Console, a byline, analytics), the long-horizon
universe expansion, and the 2026-10-20 second-reader queue, whose sub-items are all clear - none
needed a session today. `git stash list` still shows `auto-rescue 2026-10-28_060001`, empty and
left in place as every entry since 10-28 records. **`main` arrived red**, which outranks the focus,
so that came first; then two dated commitments in the brief's "Coming up", both past due; and the
second of them turned into the product day's actual finding. **0.13 (the Factor Lab) was not
worked** - a sixth consecutive session. The deferral-streak rule excuses a session that inherits a
broken state, which this one did, but I am not going to lean on that: see Next.

### Did

**1. The red baseline: a web font that answered late.**

`tests/test_lab_reading_and_access.py::test_the_height_the_card_bound_reads_waits_for_the_page_and_sees_a_taller_card`
failed with `sum(settled) == 4549` against `sum(before) + 500 == 4459`. The 90 px between them is
the figure that module already records for Inter replacing the fallback face on the lab's cards.
`_settled` called a page settled once a reading had held for 1.3 s - which is a statement about
how fast Google Fonts answered. That morning it answered later, so `before` was read in the
fallback face (3,959 px) and everything after it in Inter (4,049).

Reproduced exactly rather than argued: with 1.5 s of latency on network requests (file:// is not
throttled, so only the font is late) the old helper returns **3,959 and the page reads 4,049 six
seconds later**; at 0 and 0.5 s it returns 4,049
(`research/measurements/2026-11-17-late-font-defeats-settled.py`). The helper now waits until the
font stylesheet is applied and no face it declares is still loading, or until the request is
known to have failed, and only then runs its stability loop. After: 4,049 at 0, 1.5, 3 and 6 s.

Two things worth carrying. My first reproduction slept inside a route handler, which blocks the
thread the helper runs on, so the helper waited with the font and the bug vanished - a
reproduction that cannot fail is not one. And my first version of the wait matched the page's
bare-host `preconnect` hint instead of the stylesheet, never came true, and ran to its 20 s
timeout on every call while returning the right number: it read as fixed and was a sleep. The new
test asserts the state the helper reports (`in`), not only the height, and fails on the old helper
(`MUTANT CAUGHT: ('in', 3959)`). This module is in the 02:00 publish gate, so the same flake could
have discarded a night of data; a blocked font host returns in under 5 s and measures the
fallback face, so it cannot jam the gate.

**2. The pre-registered filing cross-check re-run (0.14): the rule is NOT met.**

`research/2026-10-14-yahoo-vs-sec.md` section 5 committed, on 2026-11-02, to re-running its range
table on the first data run on or after 2026-11-16: total assets, equity and operating cash flow
become a gate at 25 only if each stayed at or below 5 stocks over 10% on every run. Over the 18
runs still on disk: total assets **0** on every run, equity **1-2**, operating cash flow **1, 3,
then 8 on 2026-11-03, 9 through 11-09, 10 on every run since**. Applied as written: nothing is
promoted. Section 6 of the note has the table.

**3. Why operating cash flow moved - which is the product day's finding.**

A live fetch of the ten (`research/measurements/2026-11-17-ocf-off-the-filing.py`): for nine, the
source's quarterly cash-flow statement carries 2026-09-30, 2026-03-31, 2025-12-31, 2025-09-30,
2025-06-30 - **no June 2026 quarter**. Four quarters cannot be summed; the fiscal year cannot be
rolled forward, because the stub after it is not consecutive; no three adjacent quarters exist to
scale by 4/3. So the fetch reads the annual statement, which is the right thing to do. **But it
dated the figures by the quarterly statement's newest column.**

Counted across the published run against the SEC cache
(`research/measurements/2026-11-17-annual-figures-under-a-quarterly-date.py`): **15 of 418 stocks
on operating cash flow and 17 of 427 on net income** carried the last fiscal year's figure under a
later date. Amazon: net income $77.7bn shown, $134.8bn filed for the twelve months to the date the
page gave. Boeing, Bank of America, Amphenol, Mondelez, Abbott likewise. On the public page, for
each of them:

- "Income statement as of Sep 30, 2026 - Cash flow as of Sep 30, 2026" over fiscal-2025 figures;
- "Filing Age: 48 days" for figures 321 days old;
- and no "A period behind the filings" note - the note built on 2026-10-15 **for Amazon, Bank of
  America and Boeing by name** reads the income statement's date, so the brief has said "0 stocks
  scored a period behind a filing over 30 days old" every morning while sixteen were scored three
  quarters behind.

That last one is why this sat for two weeks: every surface that should have said it read the same
wrong date, so they agreed with each other.

Fixed at the one place the date is made. `_stmt_date_financials` and `_stmt_date_cashflow` are now
the period end of the statement the figures were **read from**; the fetch records which statement
was replaced (`_annual_instead`, display-only, never read by `compute_metrics`); the Data
Provenance panel says "Last fiscal year used", names the statement and why; the "period behind"
note is worded for this case in both places it is written (the source *has* the newest quarter for
these stocks, so "had not published that period" would have been false); and the methodology page
states the fourth route to twelve months, which it had not. Live fetch after the change: BA and
AMZN date both statements 2025-12-31 with the balance sheet at 2026-09-30; MSFT and DXCM unchanged.

**No score moves** - the figures were already the annual statement's. The committed payload was
fetched with the old code, so the page changes with tomorrow's 02:00 run, not today. What I saw of
the page: the sentence rendered through the page's own `renderProvenance` in Chromium at 1280 px
(`tests/test_dead_badges.py`, two new tests) - both statements named with a plural verb, one with
a singular, the dates above it, and no "sum of the last four reported quarters" over it. I did not
look at it at 375 px on a real stock, because no published stock carries the flag yet; that is
Next 1.

**4. Tuesday's drilldown check.** `research/measurements/2026-11-17-drilldown-spot-check.py`, plain
arithmetic on the published payload, independent of `calc_trace`: HST (#1), AMZN (#479) and JPM
(#384) each rebuild their composite from the eight published category contributions to within
0.01, and earnings yield and asset growth from the published inputs to four decimals. JPM's ROE
"differs" from net income over total equity by design - banks are scored on net income to common
over common equity since 2026-10-20. Amazon's 70.8% asset growth looked wrong and is not: both
balance sheets match its filings to the dollar ($727.9bn to $1,243.4bn).

**5. Housekeeping the work implied.** `COMING_UP` in `scripts/write_brief.py`: the 2026-11-09 line
is deleted (the context signals' first one-month reading has landed - the brief reads "up to 1
effective observations of 8"), the 2026-11-16 line is deleted as done, and **2026-12-15** is added
for the re-registered cross-check question. `CLAUDE.md` 0.14 carries the result and the standing
rule; its 0.13 line said experiment 19 had no second reader, which stopped being true on
2026-11-07. `plan/dashboard-inventory.md` has the new panel sentence.

### Evidence / research

- **A documented failure, measured four ways**: the range table (18 runs), the live fetch of the
  ten stocks, the universe count against the SEC cache, and the live fetch after the change. All
  four scripts are committed under `research/measurements/2026-11-17-*`.
- **The flake**: reproduced at 1.5 s and 3 s of latency with the baseline's own two numbers
  (3,959 and 4,049), and absent at 0 and 0.5 s.
- No literature today, and none needed: nothing about how a stock is scored changed.

### Methodology changed

- **No score, weight, metric or threshold.** `METHODOLOGY_CHANGELOG.md` 2026-11-17 records the
  dating change as a display entry, because it changes what the public page says about sixteen
  stocks' inputs and turns their staleness flag on.
- Editing `factor_engine.py` staled all 32 research-ledger rows, as `CLAUDE.md` warns; they were
  rerun (`--jobs 4`) and the report regenerated. `STALE REMAINING: 0` after about 17 minutes; `RESULTS.md` differs only in its run count (680 -> 712) and provenance line, and `research.html` and `summary.json` did not change, so no statistic moved.

### Tried and rejected

- **Promoting total assets and equity alone.** Both met the bar on every run. The rule was written
  as a conjunction - "if total assets, equity and operating cash flow have each stayed at or below
  5 ... those three" - and splitting it after seeing which two passed is the re-fitting a
  pre-registration exists to prevent. Re-registered instead, before any post-fix count exists.
- **Scoring those sixteen stocks on something newer than the annual statement.** They are up to
  three quarters behind figures the source does carry. But the June quarter is the one that is
  missing, so no arithmetic on the quarterly statement recovers the twelve months, and scoring may
  not read a filing. `research/2026-10-29-ttm-fills-the-empty-quarter.md` section 5.5 holds the
  narrow pre-registered alternative; this is evidence that the population is not the one that
  note measured, and it is a Monday's question, not a Tuesday's patch.
- **A detector for DexCom's defect.** The tenth stock is a genuine source error: its 2026-09-30
  quarterly cash-flow column holds $1.120bn, the nine months to that date. Summed as a quarter the
  twelve months is $2.209bn; replacing it with the quarter it implies gives $1.414bn, the filed
  figure to the dollar. DexCom ranks 8th. A rule that spots a cumulative figure in a quarterly
  column would also fire on a genuinely seasonal company, and its false-positive rate across 500
  stocks is unmeasured - so it is recorded (note section 6, `CLAUDE.md` 0.14, and the brief's
  "Top 25, worth a look" already names it every morning) and not built.

### Next

1. **Read tomorrow's 02:00 run for the fix landing.** Expect, in `logs/datarun-2026-11-18_*.log`'s
   "inputs against SEC filings" line: operating cash flow over 10% back toward 1-3, and "stocks
   scored a period behind a filing over 30 days old" rising from 0 to roughly 16. If the second
   number stays 0 the date fix did not reach the payload - look at `_in__stmt_date_financials`.
   Then open Amazon's Data Provenance panel at 1440 and 375 px and read the "Last fiscal year
   used" sentence on the real page; today it could only be rendered from a synthetic stock.
2. **DexCom, as a Monday research question**: how often does the source publish a year-to-date
   figure in a quarterly cash-flow column, measured offline against the SEC cache across the
   universe, and can it be recognised from the statement alone? It is the eighth-ranked stock on
   the public page with operating cash flow 56% above its filing.
3. **0.13 has now gone six sessions.** It is the lab's standing Friday programme (19 experiments,
   all second-read), not a task with an end, and its age in the health line keeps growing while it
   ships on schedule. The next session with a green `main` and no dated commitment should either
   run the next experiment in `plan/factor-lab.md`'s queue or rewrite its entry under "Open" so the
   age measures something - Friday 2026-11-20 is a lab day (ISO week 47 is odd, so not a
   retrospective).
4. `tests/test_header_nav_fit.py` and `tests/test_sheet_round_three.py` each have their own
   `_settled` helper. Neither failed today and I did not change them, but both pages load the same
   web font: if either flakes, the lab module's `_wait_for_the_web_font` is the fix to copy.

## 2026-11-17 (13:00) - QUEUE. The eighth-ranked stock counts half a year of cash flow twice: measured, said on the page, and the correction left for a dated review

**Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-17_060001.log` (completed). Data loop published? yes - `logs/datarun-2026-11-17_020000.log` (completed; HEALTH: PASS). Evidence base at `1m`: 5 of 8 needed at the 1m horizon (53 rows, but overlapping windows are not independent; 130 rows across all horizons), newest 1m run date 2026-10-16 (32 days ago, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab** (age **36 days**; a standing programme, last advanced **11 days** ago per `plan/factor-lab.md` (bound 21)); also open: 0.10 41d; 0.14 34d.

(The roadmap half of that line is in the form this session gave it - see Did 1.)

**Tests:** before **4,334 passed / 2 skipped / 0 failed** (my baseline, 4:17 in parallel; the same
as the 06:00 session's "after", so `main` arrived green) -> after: **4,352 passed / 3 skipped /
0 failed** in 4:01 (18 new passing tests; the one new skip is
`tests/test_ytd_column_note.py::test_the_published_payload_agrees_with_the_filings`, which skips
loudly until a payload built with the note is published - tomorrow's). Gate 2 (dry-run) exits
normally before and after. Gate 3: the three payload files are byte-identical to the base commit
(`git diff --stat bdd5c70` on them is empty), `index.html` is 944 KB and is what the generator
builds (`tests/test_rebuild_page.py`); the tool guard refuses `node` in this session, so the
parse itself is `tests/test_payload_parse_gate.py`'s (22 passed with the rebuild module) and the
runner's. Gate 4: `git status --porcelain` empty. Gate 5, `scripts/rehearse_publish.py
--changed-since bdd5c70`: **REHEARSAL PASSED**, published files restored byte for byte. The
suite ran on the tree before this paragraph's numbers were filled in; nothing but this file
changed after it.

**Owner queue / rotation:** this is today's second session, so I carried on from the 06:00
entry's Next. Its item 1 needs tomorrow's 02:00 run and cannot be done today. Its **item 2** (the
DexCom question, as research) is what this session did, end to end. Its **item 3** (0.13 has gone
six sessions) is resolved in Did 1. `write_brief.open_proposals()` returned `[]` at the start and
returns `[]` now. `OWNER_FOCUS.md`: nothing moved - its three open items are the owner's own
decisions, the long-horizon universe expansion (not worked today; the morning's Next outranked it
because it was a live defect on the public page's eighth-ranked stock) and the second-reader
queue, which is clear. `git stash list` still shows the empty `auto-rescue 2026-10-28_060001`,
left in place as every entry since 10-28 records. Nothing the 06:00 session decided is reversed.

### Did

**1. 0.13's age now measures something (the deferral-streak rule, resolved rather than deferred a
seventh time).** The Factor Lab is one experiment every lab Friday with no end. Its "age 36 days"
grew daily while it shipped on schedule (experiment 19 on 2026-11-06, second-read 11-07), six
sessions in a row logged it "deferred", and the streak rule - written for a task nobody touches -
was about to make a Tuesday session "resolve" a programme that is not late. An open item whose
heading names its progress file (`progress in \`plan/factor-lab.md\``) now also reports the days
since the newest dated bullet under that file's Progress, derived, with a 21-day bound (lab
Fridays alternate with retrospectives: 14 days, plus one lost week). It reads **11 days**.
`scripts/health_numbers.py`, 4 new tests; `CLAUDE.md` 0.13 and `prompts/nightly.md` say that a
non-Friday session has not deferred it, and that past 21 days it is late and the streak rule
applies. This is the "demote with the argument" branch of the rule, and the argument is the
number: nothing about the lab's rules or its Friday changed. **Friday 2026-11-20 is a lab day.**

**2. The DexCom question, measured** (`research/2026-11-17-ytd-in-a-quarterly-column.md`). The
morning found the source's September-quarter cash-flow column carries the nine-month figure and
asked how often that happens and whether the statement alone can show it.

- **How often:** I fetched the quarterly, annual and trailing cash-flow statements for all 499
  stocks (498 returned; one sequential pass, about 11 minutes) and compared **2,300** quarterly
  operating-cash-flow columns with the discrete quarter and the year-to-date figure the SEC cache
  implies. **2,283 are the filed quarter within 2%; 3 are the filed year-to-date figure** -
  DexCom's 2026-09-30 and 2025-09-30 columns and GM's 2025-06-30 - and **one of the three is
  inside the twelve months scored.** One stock of 497.
- **When:** DexCom's figure matched its filing to the dollar on all 16 published pages to
  2026-10-30 and has been wrong on all 8 since 2026-11-03. It arrived with the new 10-Q.
- **What it is worth:** restated with the filed $1.414bn, free-cash-flow yield falls from the
  84th to the 31st percentile of Health Care and the rank from **8 to about 20** (an estimate
  that rebuilds both published category scores first and holds the other 498 stocks still). The
  sharper fact: the true twelve months *fell* (a weak third quarter, $0.326bn against $0.660bn)
  while the page showed them rising to $2.209bn and the rank improving from 7-10 to 5-8.
- **Can the statement alone show it? No, three ways.** "The column is at least the earlier
  quarters of its fiscal year" fires on **63 of 379** newest columns for the one true case. A
  completed fiscal year's four quarters match the annual statement for **61 of 61** - an identity,
  but blind to the current year, which is where a new 10-Q's error lands. And **the source's own
  trailing-twelve-month figure repeats the error** ($2.209bn), so the source does not disagree
  with itself. The morning's refusal to build a detector was right by a wide margin.
- **What can:** the run's figure minus the filed twelve months equals the filed year-to-date
  figure one quarter before one of the last four quarter ends ($2.2092bn - $1.4144bn = $0.7948bn
  = DexCom's filed six months to June). Run over every published page since 2026-10-08:
  **10,446 stock-days, 151 over 2% from a filing, the identity fires on 8 - DexCom's 8.**

**3. The page says it, where the number is.** Display only - no input, score or rank changes.
DexCom's free-cash-flow-yield and accruals rows carry, under "Against this company's filings":
*"The $2.2B of operating cash flow above counts part of the year twice. The data source's column
for the quarter to September 30, 2026 carries $1.1B, which is this company's filed figure for the
nine months to that date; the quarter alone was $326M as filed. With that quarter, the twelve
months to September 30, 2026 come to $1.4B in the company's filings, 36% below the figure used
here. This score is computed from the source's figure as delivered and has not been adjusted."*
Data Provenance says it in one sentence, and for this cause no longer follows "operating cash
flow (36%) differs" with "a difference usually means the two sources define the line differently
... rather than that either figure is wrong" - which on DexCom's page was false.

What I saw: the committed payload predates the note, so I built a throwaway copy of the page
(the generator's HTML, the published payload with the note attached from the SEC cache) and
opened DexCom's sheet. At **1440 px** the note sits under the inputs and the "Rebuilt from the
inputs above" line of the FCF-yield row, after "Worth knowing", five lines, nothing clipped. At
**375 px** it is the same block on the accruals row, eleven lines at 297 px wide, inside the
card, the document no wider than the viewport. The Provenance sentence I read from the rendered
panel's text and in the rendered test, **not in a screenshot** - my screenshot of it landed on
the collapsed section above, and I did not go back for it.

**4. The correction is not built, and it is not an open proposal either.** Whether a filing may
correct one field of one stock crosses the display-only rule that
`research/2026-10-14-yahoo-vs-sec.md` section 3c earned by measuring a per-stock filings fallback
and refusing it. This case is narrower (one field, where source and filings agree within 0.5%
for 402 of 418 stocks, and where the figure is provably not the company's) but it is **one
instance**. The note's section 7 gives three options and four conditions written before any
further page is read (two more confirmed firings; no rank moved by the substitution where the
figures already agree; no double jump when the source corrects itself; a `DECISIONS.md` entry and
the 32-row rerun if built). Its status line is **WAITING until 2026-12-15**, deliberately not
OPEN: the brief makes an OPEN proposal the next session's first job, and this one cannot be
decided for a month. `COMING_UP` in `scripts/write_brief.py` carries the date.

### Evidence / research

- **A documented failure, measured five ways**, every script committed under
  `research/measurements/2026-11-17-ytd-*`: the column-level table (with its JSON), the impact
  estimate, the page history, the identity over 24 pages, and the rehearsed payload's diff.
- **17 CFR 210.10-01(c)(3)** (Regulation S-X; read on Cornell LII today): "Interim statements of
  cash flows shall be provided for the period between the end of the preceding fiscal year and
  the end of the most recent fiscal quarter, and for the corresponding period of the preceding
  fiscal year." The income-statement paragraph names the quarter as well; this one does not. So
  every "quarterly" cash flow is a vendor's subtraction - which is why this error exists.
- **Documented practice:** WRDS's `QUARTERIZE` macro, "Quarterizes Compustat YTM Cash Flow
  Variables in FUNDQ Dataset" (header and code comments read today; the page has no prose).
  Deloitte's cash-flow roadmap, Appendix D, was found and **not read**.

### Methodology changed

- **No score, weight, metric, threshold or input.** `METHODOLOGY_CHANGELOG.md` 2026-11-17 (13:00)
  records the note as a display entry, with the evidence above and the waiting proposal.
- The rehearsed payload differs from the published one in `srcn` and `xwhy` for DexCom and the
  top-level `src_notes` (1); the only other difference is the 06:00 session's
  `flags.annual_instead` key. No `RESULT_PATHS` file was edited, so the research ledger is level.

### Tried and rejected

- **A statement-only detector** - measured, three ways, above. Not a matter of tuning: the one
  usable-looking rule has 62 false alarms to one hit, and the one exact rule cannot see the case.
- **Writing the note into `cavs`**, the existing per-metric slot. `tests/test_capex_lease_note.py`
  and `tests/test_special_item_note.py` each rebuild their `cavs` key from the filings and assert
  the *whole dict* equal, and both are in the 02:00 publish gate. A second writer to
  `cavs['fcf_yield']` would have failed the lease note's test on the first night DexCom's note
  published and **discarded the run**. I had it written that way for about ten minutes; reading
  the neighbouring test before running mine is what caught it. The note has its own key (`srcn`),
  and the inventory says why.
- **Marking the proposal OPEN.** See Did 4.
- **Correcting the score today**, or withholding the field. One instance; a settled rule; a
  second reader's decision.

### Next

1. **Tomorrow's 02:00 run carries two changes to read on the real page**: the 06:00 session's
   (its Next 1 stands unchanged - the cross-check line and Amazon's "Last fiscal year used"), and
   this one - `src_notes` should be **1**, DexCom's FCF-yield and accruals rows should carry
   "Against this company's filings", and its Data Provenance panel
   "Operating cash flow counts part of the year twice." Look at that panel sentence in a
   screenshot at 1440 and 375 px; I only read its text. If `src_notes` is 0 while DexCom's
   `xchk.ocf` is still about 0.36, the SEC cache moved or the source corrected the column -
   `python research/measurements/2026-11-17-ytd-signature-history.py` says which.
2. **Friday 2026-11-20 is the lab's day** (ISO week 47, not a retrospective): the next experiment
   in `plan/factor-lab.md`'s queue is `EarningsSurprise` on Open Source Asset Pricing, behind the
   Revisions category's surprise points - and it needs a dated Progress bullet when it ships,
   because that bullet is now what the health line reads.
3. The 14 columns that are neither the filed quarter nor the year-to-date figure (note section
   8; DTE's 2025-09-30 column is $0.133bn against $0.632bn filed and is inside its twelve months)
   are unexplained. A Monday could take them; the cross-check counts them every morning meanwhile.
4. The DexCom Monday question the 06:00 entry queued (its Next 2) is **done** - do not start it
   again.

## 2026-11-18 - SYNTHESIS. Six weeks of "the page says when an input is impaired", read as one layer: a sentence that was false for all 27 stocks it was printed for, and twenty companies the filing check cannot see

**Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-17_130001-midday.log` (completed). Data loop published? yes - `logs/datarun-2026-11-18_020001.log` (completed; HEALTH: PASS). Evidence base at `1m`: 5 of 8 needed at the 1m horizon (53 rows, but overlapping windows are not independent; 131 rows across all horizons), newest 1m run date 2026-10-16 (33 days ago, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab** (age **37 days**; a standing programme, last advanced **12 days** ago per `plan/factor-lab.md` (bound 21)); also open: 0.10 42d; 0.14 35d.

**Tests:** before **4,354 passed / 1 skipped / 0 failed** (my baseline, 4:05 in parallel; `main`
arrived green) -> after **4,363 passed / 1 skipped / 0 failed** in 3:58 (9 new tests). Gate 2
(dry-run) passes before and after. Gate 3: the three payload files are untouched (`rebuild_page.py`
reported "payload byte-identical"; `git status` clean after every gate) and `index.html` is what
the generator builds (`tests/test_rebuild_page.py`); the parse itself is
`tests/test_payload_parse_gate.py`'s and the runner's. Gate 4: `git status --porcelain` empty.
Gate 5, `scripts/rehearse_publish.py --changed-since 5f55b79`: **REHEARSAL PASSED** (381 page
tests, published claims 2,373 passed, files restored byte for byte). The suite ran on the tree
before this entry was written; nothing but this file changed after it.

**Owner queue / rotation:** I am today's first session. `write_brief.open_proposals()` returned
`[]`, and no branch was held back by the gates. `OWNER_FOCUS.md`: nothing moved - its three open
items are the owner's own decisions (Search Console, a byline, analytics), the long-horizon
universe expansion (not worked: the rotation's focus had a real subject and no owner item was
waiting on a session), and the second-reader queue, which is clear. **0.13 is not deferred** - it
is a standing programme 12 days from its last advance against a bound of 21, and Friday
2026-11-20 is its day. Nothing was deferred for a gate or a loop.

**Yesterday's Next 1, read first:** both of 2026-11-17's changes landed in the 02:00 run.
`src_notes` is **1** and DexCom carries its note on the FCF-yield and accruals rows. Operating
cash flow over 10% from the filing fell from 10 to **1** (DexCom), as predicted. **26** stocks
carry `annual_instead` - more than the "roughly 16" predicted; that count was of stocks the SEC
cache could confirm (15 of 418 and 17 of 427 checked), which is the likely reason and not one I
verified stock by stock. The brief's "scored a period behind a filing over 30 days
old" reads **7**, not 16: the other stocks' newer 10-Q is under 30 days old, so they will join it
over the next fortnight; the page's own note names 32. And looking at Amazon's sheet for the
"Last fiscal year used" sentence is what started Did 1.

### Did

**The synthesis subject.** Five changelog entries since the 2026-11-04 synthesis; none moved a
score; all five are the page learning to say an input is not what it appears to be - the same
kind of change as most of the six weeks before them. Nobody had looked at what they add up to.
`research/2026-11-18-synthesis-what-the-screener-knows-about-its-inputs.md`, with five committed
measurement scripts.

**1. Two correct changes met and made a false sentence - fixed.** The 2026-11-02 stale bar and
the 2026-11-17 statement dating were each sound and each verified. Together they printed, on
Amazon's page this morning, *"Its filings are flagged stale (322 days old)"* - beside a 10-Q
filed 19 days earlier that the same payload records. Measured: **27 stocks** carried the
sentence; **27 of 27 had a newer filing on record**, 22 of them filed inside the 165-day bar;
27 of 27 had a balance sheet inside the bar; 26 are the annual-statement fallback and the 27th
is DuPont's refused quarterly statement. The flagged count on every page under the bar was 8, 8,
5, 1, 1, 1, 1, then 27. The flag ages the income statement the score *read*; the sentence
blamed the company.

Now: *"Its income statement and cash-flow statement figures are the last fiscal year's (322 days
old): the data source's quarterly figures could not be summed to twelve months."* where the
fallback is the cause (the provenance panel's own condition, so the two cannot disagree), and
*"The income statement this score read is flagged stale (N days old)."* otherwise. The badge
"Filing Age" is **"Statement age"** - it was never a filing's age for any stock. "The case
against" clears with "stale statement". Claim `summary.confidence` restated, three new checks.
Display only; no score moves.

What I saw, on a rebuild of today's run with the new code: 26 stocks get the first wording, 1
the second, 0 the old. At **1440 px** Amazon's "Read with care" shows the sentence in three
lines under "Why it ranks here", nothing clipped. At **375 px** the badge row wraps to two lines
with "Statement age: 322 days" in the alert colour on its own, and the sentence is five lines at
343 px; the document is no wider than the viewport. I read both screenshots, not only the
measured boxes.

**2. The layer, counted.** Sixteen kinds of input caveat on the page. **247 of 499 stocks carry
at least one; 17 of the top 40; 11 of the top 25.** Every one is either a route the engine took
(the score reads its output) or a sentence (the score is untouched) - there is no caveat that
discounts a score a little, which is 0.14's rule holding everywhere. On today's page "statement
flagged stale" (27) is a strict subset of "annual fallback" (27) and of "a period behind the
filings" (32): redundant with two notes that say why. Kept as the coarse fallback it was built
to be - on 11-04 half the flagged stocks had no newer filing on record - but reworded so it does
not out-shout them. And one thing deliberately **not** designed: a per-stock "data quality"
roll-up. Half the universe would light it, or it would need weights between unlike things.

**3. What the fallback leaves out, by category - sized, not acted on.** 26 stocks have flows
from a fiscal year most of a year old beside a September balance sheet and yesterday's prices.
Momentum, Risk, Revisions, Size and Investment (43 points) read current data; **Valuation,
Quality and Growth (57 points) read the old flows.** Against the twelve months the companies
have filed: scored net income is below for **15 of 18** comparable (median gap -13.7%, median
absolute gap 32%; Amazon $77.7bn against $134.8bn), and those stocks' mean Quality score is 41.0
against 50.4 for everyone else (-2.9 se), with the market-fed categories level. That is a
measurement of inputs against filings on one morning, on a non-random 27 (five utilities, five
REITs); it sizes what is not being read and does not show the stocks are ranked too low. It is
**the same question as DexCom's waiting proposal at 26 times the size** - may a filing do more
than sit beside a scored input - so it should be researched before 2026-12-15 decides the
principle on its smallest instance. Queued in `COMING_UP` for the Monday of 2026-12-07 at the
latest.

**4. The filing check cannot see twenty companies, and said "0 days old".** Following up the
stocks section 3 could not compare: **20 of 499 scored companies have no 10-K or 10-Q in the SEC
cache filed in the last 165 days** - fourteen utilities, five REITs and Citigroup - on a morning
the cache was rebuilt. The check compares no field for 15 of them; one (Evergy) ranks 27th. I
asked the SEC's API directly for two: Regency Centers' record has no fact of any kind filed
after 2026-05-04, and Citigroup's nothing from a 10-Q in 2026. So it is the SEC's record, not
our refresh. **I did not establish why.** `sec_crosscheck.no_recent_filing` now puts the count on
the health line every morning and in the brief's row; the bar is the filing calendar's (two
periodic filings are never more than ~142 days apart, plus two weeks). A count, not an ATTENTION
line - it reads 20 on day one, and a line that fires daily is a line nobody reads.
`research/2026-10-14-yahoo-vs-sec.md` gains section 7 for whoever applies the 12-15 promotion
rule, **which I did not change**.

**5. The `auto-rescue` stash: answered and dropped.** The 2026-11-04 synthesis found it was not
empty (a `-u` stash; plain `git stash show -p` hides untracked files) and asked whether a refused
run's context log belongs in `data/context_log/`. Every session since logged it as "empty"
again. The answer has been in the data loop since the morning of that very run:
`Undo-RunArtifacts` runs `git clean -fd -- ... data/context_log` on a discard (commit `ac294b1`,
2026-10-28, "a discard that leaves no files"), and every refused day since - 11-02, 11-09,
11-11, 11-12 - is absent from the tracked series. The stash held exactly what that rule deletes,
plus two price-cache files long since regenerated. Dropped; it was commit
`7571fb42bc68ed79251886dc429cadcb471cdeab` (untracked parent `f2a4a402`), recoverable with
`git stash store` until git prunes it. `prompts/nightly.md` now gives the command that shows a
`-u` stash's files.

### Evidence / research

- **Documented failures, measured**, every script under `research/measurements/2026-11-18-*`:
  the sentence against the payload's own filing facts (and its history), the caveat census, the
  fallback against the filed twelve months, the cache's blind spot (with the two API queries),
  and the rendered page.
- **The 165-day bar** is the SEC's deadline arithmetic already cited in the 2026-11-02 changelog
  entry (10-Q within 40 days, 10-K within 60). Not re-read today.
- **No new literature.** The note cites MSCI's Fundamental Data Methodology section 1.1.5
  second-hand, from `research/2026-10-29-ttm-fills-the-empty-quarter.md`, and says so.
- Nothing here rests on a return, an IC or a backtest.

### Methodology changed

- **No score, weight, metric, threshold, input or rank.** `METHODOLOGY_CHANGELOG.md` 2026-11-18
  records the sentence, the badge and the health-line count as a display entry.
- No `RESULT_PATHS` file was edited, so the research ledger is level.

### Tried and rejected

- **Adding "Its filings are flagged stale" to `claims.FORBIDDEN`.** That list is checked against
  every published artifact, and the committed payload carries the old sentence until tomorrow's
  02:00 run - it would have turned `main` red today for a page that is about to be corrected.
  The three unit tests fail if the wording comes back; a later session can add the pattern once
  a corrected payload is published.
- **An ATTENTION line for the twenty.** See Did 4.
- **Scoring the 26 on something newer today.** Three settled things stand in the way and each
  was measured (note section 4). One thing I claimed less than I first wrote: that the 10-29
  note's narrow rule "admits none of today's 26" - I had not re-fetched them, so the note says
  what yesterday's fetch of nine showed and that the rest is unconfirmed.
- **Changing the pre-registered promotion rule** in light of the blind spot. It is
  pre-registered; the finding goes beside it, not into it.
- **Finding out why the SEC's record stopped for those companies.** Combined filings with
  subsidiaries or an operating partnership is the common feature and a guess. It would have
  taken the session; it is Next 3.

### Next

1. **Friday 2026-11-20 is the lab's day** (ISO week 47, not a retrospective): `EarningsSurprise`
   on Open Source Asset Pricing is next in `plan/factor-lab.md`'s queue. Add the dated Progress
   bullet when it ships - the health line reads it.
2. **Tomorrow's 02:00 run carries today's two changes.** Expect: the health line's "20 stocks
   with no 10-K or 10-Q in the SEC cache filed in the last 165 days" and the brief's "20 with no
   filing in the cache for 165 days"; 27 or so stocks with the new "Read with care" wording and
   none with "filings are flagged stale"
   (`python research/measurements/2026-11-18-stale-sentence-against-the-filings.py` prints "said
   without the flag" and the list). Once that payload is published, add the old sentence to
   `claims.FORBIDDEN`.
3. **Why has the SEC's `companyfacts` stopped carrying those twenty companies' filings?** Start
   from one filing: pull Regency Centers' June-quarter 10-Q from EDGAR and compare how its facts
   are tagged with its March one. If the facts are there under an entity axis, the free fix is to
   read the filing's own financial data instead of `companyfacts` for those companies; if EDGAR
   simply has not processed them, it is a limit to state on the methodology page. Either way the
   count on the health line is how to tell when it changes.
4. **The Monday on a hole in the quarterly series** (by 2026-12-07; note section 4) - and the
   Monday the 2026-11-04 synthesis queued on Investment's 5 points against Risk is still unwritten.
   The fallback goes first: it is measured on 26 stocks today, the other is +0.166 on one run.
5. From yesterday, still open: the 14 quarterly cash-flow columns that are neither the filed
   quarter nor the year-to-date figure (DTE's is inside its twelve months).

## 2026-11-18 (13:00) - QUEUE. Where twenty companies' filings went: the SEC now files a combined report under one co-registrant, and the weekly refresh follows it there

**Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-18_060000.log` (completed). Data loop published? yes - `logs/datarun-2026-11-18_020001.log` (completed; HEALTH: PASS). Evidence base at `1m`: 5 of 8 needed at the 1m horizon (53 rows, but overlapping windows are not independent; 131 rows across all horizons), newest 1m run date 2026-10-16 (33 days ago, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab** (age **37 days**; a standing programme, last advanced **12 days** ago per `plan/factor-lab.md` (bound 21)); also open: 0.10 42d; 0.14 35d.

**Tests:** the runner's baseline was GREEN (4,363 passed / 1 skipped). **My own baseline had one
failure**, `tests/test_lab_reading_and_access.py::test_a_web_font_that_lands_late_is_waited_for_and_one_that_cannot_land_is_not`
(card heights 208 against 183 at the moment it read them): 1 failed / 4,362 passed / 1 skipped in
4:39. It did not reproduce - the module alone, 35 passed - and it ran while I was making EDGAR
requests on the same machine. Recorded as a flake, not fixed; the saved output is
`logs/session-baseline-1300.txt`. After: **4,371 passed / 4 skipped / 0 failed** in 4:09 (11 new
cases). **The three extra skips are mine and are by design:** `test_capex_lease_note`,
`test_special_item_note` and `test_ytd_column_note` each skip, loudly, when the SEC cache on disk
is not the one the committed payload was built from (2,446,742 rows then, 2,449,990 now). They
skip the same way after every weekly refresh and run again once the 02:00 run publishes from the
new cache. Gate 2 (dry-run) passes. Gate 3: no payload or page file touched. Gate 5,
`scripts/rehearse_publish.py --changed-since origin/main`: **REHEARSAL PASSED**, files restored
byte for byte. Gate 4 clean. The suite was run twice with the same counts, the second time (4:27)
on the final tree with this entry already written; only this sentence changed after it.

**Owner queue / rotation:** second session today; I read the 06:00 entry first and took its
**Next 3** (Next 1 is Friday's, Next 2 needs tomorrow's run). `open_proposals()` is `[]`; no
branch was held back; no `auto-rescue` stash (`git stash list` is empty). `OWNER_FOCUS.md`: not
re-read item by item - the 06:00 session did four hours earlier and no commit has touched it
since; nothing moved. 0.13 is not deferred (12 days against a bound of 21; Friday is its day).
Nothing was deferred for a gate or a loop. No decision of this morning's was reversed; one of its
numbers is superseded (below).

### Did

**The question:** this morning found 20 of 499 scored companies with nothing in the SEC
`companyfacts` cache filed in 165 days, showed the SEC's record itself lacked them, and "did not
establish why". `research/2026-11-18-combined-filings-under-a-co-registrant.md`.

**1. The cause, for 19 of the 20.** Followed filing by filing through EDGAR's `submissions`
record and each filing's header (`research/measurements/2026-11-18-where-the-missing-filings-went.py`,
134 requests): the twenty have made **41** periodic filings since the newest in their own record,
all XBRL, none with a single fact there. **37 of them, for 19 companies, are whole in exactly one
co-registrant's record** - a subsidiary utility or an operating partnership, and a different one
from quarter to quarter (Sempra's June quarter is under Southern California Gas, September's
under San Diego Gas & Electric). Until mid-May the whole filing sat under the first registrant,
the listed company. I did not find the rule that now picks the registrant, and found no SEC
notice; the note says so.

**2. The moved facts are the parent's - measured, not assumed.** Every 10-Q repeats earlier
figures as comparatives, and the parent's own record already holds those. 7,004 of 7,326
overlapping facts are equal (**95.6%**); the same parents' last *un-moved* filing agrees with
their own history at **95.3%** (2,511 of 2,635). The 53% I first saw for American Electric Power
is AEP rounding to whole millions from 2026 - its own un-moved filing shows the same.
Independently, against the data source: total assets within 0.5% went from 440 of 440 to **459
of 459**.

**3. The fix, and four companies the age bar was hiding.**
`sec_fundamentals.refresh_companyfacts` now looks for a gap of over 150 days between a company's
periodic filings, and for a filing **the listed company leads** reads that one accession's facts
from whichever co-registrant holds them. A live refresh of all 503 companies (362 s) recovered
**41 filings for 23 companies, 3,248 facts** - 37 for the 19, plus one quarter each for Dow, Con
Edison, Edison International and United Rentals, whose next quarter was not moved, which a
165-day age bar cannot see. 0 of the old cache's 2,446,742 rows are absent from the new.

**4. Applied to the machine (rule 11).** The rebuilt cache is installed at
`data/sec/pit/facts.parquet`; the one it replaced is `logs/facts_before_1118_1300.parquet`. So
tomorrow's 02:00 run reads it whether or not this branch merges - and if it does not merge, the
old code's next weekly refresh simply rebuilds the old content. On the rebuilt cache
`no_recent_filing` reads **1** (Citigroup).

**5. What the next page shows.** From the rehearsal's regenerated payload against the published
one: `xchk` differs for 19 stocks (463 with a compared field, from 449; Evergy, ranked 27th, goes
from no check to five fields agreeing), `newer_filing` for 5 (Regency Centers' now names its
November 10-Q, not May's), and **no composite or rank differs**. `summary` differs for 27 - that
is this morning's sentence, not this change. **I did not open the page in a browser**: nothing in
how the panel renders changed, only how many stocks have rows in it.

### Evidence / research

- A documented failure, measured: the two scripts above (the second,
  `2026-11-18-whose-figures-the-moved-facts-are.py`, carries the baseline and the frames query).
- The 150-day gap is the SEC deadline arithmetic the 2026-11-02 entry cites (10-Q within 40 days,
  10-K within 60: at most ~142 days between consecutive filings). Not re-read today.
- No literature. Nothing rests on a return, an IC or a backtest.

### Methodology changed

- **No score, weight, metric, threshold or rank.** `METHODOLOGY_CHANGELOG.md` 2026-11-18 (13:00)
  records it as a data-source entry. The only engine reader of this cache is
  `earnings_variability` (weight 0, 10-K facts); every recovered fact is from a 10-Q.
- No `RESULT_PATHS` file edited; the ledger's verdict does not read this cache (`cache->red`).

### Superseded from this morning

- "20 ... with no filing in the cache for 165 days ... should fall only when the SEC's record
  catches up" (changelog, expected effect) and Next 2's "expect 20": **expect 1.** The SEC's
  record did not catch up; we now read the record the filing is in.
- The synthesis's section 4 ("15 of 18 comparable") was measured on the old cache. Not re-run.

### Tried and rejected

- **Keying the fix on the 165-day age bar** - it would have missed the four companies with one
  moved quarter.
- **Reading a co-registrant's whole record.** Only the facts carrying the missing accession
  number are taken; a subsidiary's own separate filings stay out (a test holds this).
- **Following a filing the listed company does not lead.** None of the 41 is one, so it would
  have been untested generality resting on a convention I could not check.
- **Explaining Citigroup.** It files alone; its three 10-Qs of 2026 are in no record and it is
  absent from the frames API for all three quarters. Same for one FirstEnergy filing. Cardinal
  Health and Clorox each have a gap I did not follow. All four are named in the refresh's log
  line every week.
- **Changing the pre-registered 12-15 rule.** `research/2026-10-14-yahoo-vs-sec.md` section 7
  now says its denominators step up on 2026-11-19; the rule is as written.

### Next

1. **Friday 2026-11-20 is the lab's day**: `EarningsSurprise` on Open Source Asset Pricing
   (`plan/factor-lab.md`); add the dated Progress bullet.
2. **Tomorrow's 02:00 run**: the health line should read "**1** stocks with no 10-K or 10-Q in the
   SEC cache" (not 20); total assets about 459 compared; the three note tests un-skip (suite back
   to 1 skipped). Then, from this morning's list: add "Its filings are flagged stale" to
   `claims.FORBIDDEN` once the corrected payload is published.
3. **The next weekly refresh (about 2026-11-25)** is the first by the data loop with this code:
   read `SEC companyfacts:` in `logs/datarun-*.screener.txt` - "N companies' filings were under a
   co-registrant's record" (41 filings / 23 companies today, more as later filings come) and the
   "still have a gap" names (C, CAH, CLX, FE today). `fetch.sec_roe_history` should rise by about
   100 s on that day only.
4. **February 2027: the first combined 10-Ks.** `earnings_variability` (weight 0) reads 10-Ks;
   the same follow-up covers them, and the gap line is the check. Worth a `COMING_UP` entry when
   a session is next in `write_brief.py`.
5. Unchanged from this morning: the Monday on a hole in the quarterly series (by 2026-12-07) -
   **re-run its section-4 measurement on the new cache first** - and the 14 quarterly cash-flow
   columns that are neither the filed quarter nor year-to-date.

## 2026-11-19 - BUILD. The universe block in `config.yaml` is read now - and the $2B floor it held had never excluded a stock

**Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-18_130000-midday.log` (completed). Data loop published? yes - `logs/datarun-2026-11-19_020001.log` (completed; HEALTH: PASS). Evidence base at `1m`: 5 of 8 needed at the 1m horizon (54 rows, but overlapping windows are not independent; 133 rows across all horizons), newest 1m run date 2026-10-19 (31 days ago, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab** (age **38 days**; a standing programme, last advanced **13 days** ago per `plan/factor-lab.md` (bound 21)); also open: 0.10 43d; 0.14 36d.

**Tests:** before **4,374 passed / 1 skipped / 0 failed** (4:03, my own baseline; yesterday's three
cache-dependent skips are running again, as predicted). After **4,393 passed / 1 skipped / 0
failed** (4:38) - 19 new cases, all in `tests/test_universe_config_block.py`. Gate 2 (dry-run)
passes. Gate 3: no payload or page file touched; the rehearsal's own parse of all three payload
files passed (a standalone `node --check` was refused by the tool guard this session, so I did not
run it separately). Gate 5, `scripts/rehearse_publish.py --changed-since origin/main`:
**REHEARSAL PASSED**, every published file unchanged byte for byte. Gate 4 clean.

**Owner queue / rotation:** first session today. `git stash list` empty; no branch held back;
`open_proposals()` is `[]`. `OWNER_FOCUS.md` read: the top open item (2026-11-17, three decisions)
is the owner's alone; the next is **universe expansion (2026-10-28)**, whose plan names the
`config.yaml` `universe:` block as next in phase 1 - that is what I took, and it is also what a
BUILD day is for (the week's research - Tuesday's dating fix, Wednesday's synthesis - shipped its
own builds and left dated reviews, not a pending change). Its Progress line is updated; the item
stays open (phases 1-4 continue). 0.13 is not deferred (13 days against 21; tomorrow is its day).
Nothing deferred for a gate or a loop.

**Yesterday's Next 2, checked first:** today's health line reads total assets **459/459**, "**1**
with no filing in the cache for 165 days" (was 20), and the three note tests run again (suite
back to 1 skipped). "Its filings are flagged stale" appears in **0** of the four published files.
**Not done:** adding that sentence to `claims.FORBIDDEN` - `claims.py` builds the page, and I kept
today's diff to one subject. It is Next 3.

### Did

**1. The finding.** The plan's resolution said `min_market_cap: 2e9 -> 0` is byte-identical
*today* because no member is under $2B. Checking whether that also held for the research
engine's history (S&P members were under $2B in 2012) I found it held for a different reason:
**the floor had never fired, anywhere.** `factor_engine.apply_universe_filters` acted only "if
`marketCap` in df.columns", and both callers (`run_screener.py`, `backtest_v2.rank_month`) hand
it `compute_metrics`'s frame, which carries `size_log_mcap` and no `marketCap`. Measured
(`research/measurements/2026-11-19-market-cap-floor-never-fires.py`): the shipped filter returns
**502 of 502 rows at floors of $2 billion, $1 trillion and $1 quadrillion**. `config.yaml` said
`# $2B minimum` since February; the Blueprint says it "eliminates micro-cap illiquidity".
`tests/test_edge_cases.py::TestUniverseFilters` passed throughout - it builds its frame by hand
with the column production does not have. Same shape as the 2026-11-09 `financials` shard miss
(settled row "gates"): the test and production disagreed about the input, and the test won.
Likewise `universe.index` was read by no code; its comment offered `R1000` and `R2000`, and
either silently ran the S&P 500.

**2. The build** (phase 1's `config.yaml` bullet):
- `apply_universe_filters` applies a positive floor from `marketCap` where present, else from the
  engine's own `size_log_mcap` (`exp(-x)`), and **raises** if a floor is set and neither exists.
  Unknown market caps are kept, as before.
- `config.yaml`: `min_market_cap: 0` (the 2026-11-03 decision), the dead options comment gone,
  the block says what each key does.
- `universe.py`: `UNIVERSES`, `from_name`, `from_config`; `CURRENT` is what `config.yaml` names
  (falls back to the S&P 500 only if the file cannot be *read*; raises if it names an undefined
  universe). `run_factor_engine` resolves the run's own config before the first request and
  passes it to `assign_size_bands`. `schemas.RunConfig` validates both keys.
- I kept the measured bounds (495-515, 490) in `universe.py` beside their measurements rather
  than moving them into YAML: the config names *which* universe; member counts are measurements,
  not tunables. Phase 2 adds a `Universe` and a key.

**3. Proof nothing moved.** (a) By construction: a no-op at 2e9, a no-op at 0. (b) The research
engine's 32 rows were rerun (three `RESULT_PATHS` files changed; `--jobs 4`, about 16 minutes):
**32 ids, 27,225 statistics, 0 differences** other than the two config hashes, which should move
(`research/measurements/2026-11-19-floor-to-zero-rerun-moved-nothing.py` + `.json`);
`summary.json` is byte-identical. (c) Rehearsal: all published files unchanged. (d) A five-ticker
live run (`--tickers AAPL,JPM,MOS,XOM,GOOGL --no-portfolio`, 21 s): prints "Loading S&P 500
universe...", every row banded "S&P 500", calc trace reproduced 40 category scores and 5
composites, `SCREENER_OVERVIEW.md` regenerated byte-identical. The run directory was deleted and
the three files it rewrote restored; `run_context.latest_run_dir()` is the published run again.
(e) Mutation: with the old filter restored, 4 of the 19 new tests fail.

**4. A second finding, recorded and not fixed** (`plan/backtest-v2.md`, new section). Had the
$2B floor worked it would have removed 672 of 65,495 name-months from the engine's history - and
about 655 of those are **wrong market caps, not small companies**: Berkshire Hathaway reads under
$2B in **176 of 176 months**, 16 tickers read under $100M somewhere (SPG 163 months, FOX and FOXA
90 each, BKR 67, SYK 34). The point-in-time record's share count is wrong for those names, which
reaches the engine's Size score and every market-cap ratio for them. I did not follow the causes
or size the effect on any ledger statistic.

### Evidence / research

- A documented failure, measured: the two scripts above. No literature read today.
- The reason 0 is the right floor is the 2026-11-03 decision
  (`research/2026-11-02-universe-expansion-phase0.md` sections 7, 9.3; `plan/universe-expansion.md`
  "Resolved for phase 1"). Not re-argued and the S&P methodology was not re-read.
- The engine rerun is a *Backtest observation* about provenance, not evidence of anything (rule 5).

### Methodology changed

- `METHODOLOGY_CHANGELOG.md` 2026-11-19: `universe.min_market_cap` 2e9 -> 0. **No score, weight,
  metric, rank or sentence moves.** `kpis.methodology_version` will change on the 2026-11-20 run
  because the parsed config changed.
- **Read this tomorrow:** that version change plus today's changelog date makes the bot-input
  gate's rank-continuity override *available* for the 2026-11-20 run
  (`check_run_health.continuity_override`). This change cannot cause a discontinuity, so if
  tomorrow's health output shows the override used, the discontinuity is unexplained - investigate
  it as if the override did not exist.

### Tried and rejected

- **Making the $2B floor work and leaving it at 2e9.** It would have been the first time the
  setting did anything, against a decision that says 0, and in the engine it would have deleted
  Berkshire from every month on a units error.
- **Carrying `marketCap` through `compute_metrics`.** A new column in every artifact from 01
  onward for something `size_log_mcap` already holds exactly.
- **Moving the member bounds into `config.yaml`.** They are measured facts with their provenance
  in `universe.py`; in YAML they read as knobs.
- **Fixing the engine's share counts today.** A data fix reruns every row and each of the 16
  tickers looks like a different cause; guessing a rule in the last hour is how the next defect
  gets written.
- **Adding the S&P 400/600 `Universe` objects now** to share the rerun. That is phase 2's first
  bullet and needs their bounds measured from their own lists.

### Next

1. **Friday 2026-11-20 is the lab's day**: `EarningsSurprise` on Open Source Asset Pricing
   (`plan/factor-lab.md`); add the dated Progress bullet.
2. **Tomorrow's 02:00 run**: confirm HEALTH: PASS without the continuity override being used, and
   that `kpis.methodology_version` changed (expected).
3. Add "Its filings are flagged stale" to `claims.FORBIDDEN` (the corrected payload is published;
   0 occurrences today) and a `COMING_UP` entry for February 2027's first combined 10-Ks - both
   carried from 2026-11-18.
4. **The engine's wrong market caps** (`plan/backtest-v2.md`, "An open data defect"): follow each
   of the 16 tickers' share count to its filing fact, then a plausibility bound where the record
   is built. Budget the full rerun.
5. Unchanged: the Monday on a hole in the quarterly series (by 2026-12-07); the weekly SEC
   refresh's first run with the co-registrant code (about 2026-11-25).

## 2026-11-19 (13:00) - QUEUE. The engine's wrong market caps traced to their filed facts: four causes, about 3% of name-months, and a candidate rule that failed one of its four criteria - designed, not built

**Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-19_060001.log` (completed). Data loop published? yes - `logs/datarun-2026-11-19_020001.log` (completed; HEALTH: PASS). Evidence base at `1m`: 5 of 8 needed at the 1m horizon (54 rows, but overlapping windows are not independent; 133 rows across all horizons), newest 1m run date 2026-10-19 (31 days ago, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab** (age **38 days**; a standing programme, last advanced **13 days** ago per `plan/factor-lab.md` (bound 21)); also open: 0.10 43d; 0.14 36d.

**Tests:** before **4,393 passed / 1 skipped / 0 failed** (3:58). After **4,394 passed / 1 skipped /
0 failed** (4:22) - one new parametrized case. Gate 2 (dry-run) completes. Gate 5,
`scripts/rehearse_publish.py --changed-since origin/main`: **REHEARSAL PASSED**, all published
files unchanged byte for byte (it parses the three payload files; I did not run a separate
`node --check`). Gate 4 clean.

**Owner queue / rotation:** second session today; this morning's entry read first. `git stash
list` empty, no branch held back, `open_proposals()` is `[]`. `OWNER_FOCUS.md`'s top item is the
owner's alone; the universe-expansion item advanced this morning. I carried on from the morning's
**Next**: items 1 and 2 are tomorrow's by their nature; I did **3** (small) and the diagnosis half
of **4**. No decision of the morning's was reversed. 0.13 is not deferred (13 days against 21;
tomorrow is its day). Nothing deferred for a gate or a loop.

### Did

**1. The engine's share counts, followed to the filed facts**
(`research/2026-11-19-engine-share-counts.md`). The morning found 16 tickers under $100M and
Berkshire under $2B and said to follow each count to its filing before choosing a rule. Four
causes, not the three it guessed at:

- **A stale tag read as "latest".** Simon Property: 8,000 shares, dated 2012, picked in every
  month to August 2026. Fox: 1 share, dated March 2019. Baker Hughes: 100.
- **A units slip downward.** Stryker's balance sheet says 383; its cover page, same date,
  382,672,989.
- **The same slip upward - which this morning's check could not see**, because it looked only
  for small values. Garmin reads $9.6 quadrillion in December 2016; 84 name-months across 15
  tickers are over $4 trillion and only 32 of them (Apple, Alphabet, NVIDIA, 2025-26) are real.
- **A logic error: the split adjustment is taken from the month-end, not from the day the count
  was filed.** NVIDIA reads $2,697bn, **$304bn, $288bn**, $2,928bn across May-August 2024. 194
  name-months, 77 tickers, at exact split ratios. The SEC's own rule (SAB Topic 4.C, fetched
  today) is why the filed date is the right basis.

**Size: 1,992 name-months (3.06%, 135 tickers) are more than 10% off, 473 by more than 10x, and
1,635 have no market value where one is available** - against the 494-655 recorded this morning.
Spread evenly, 2.6-3.5% of every year.

**2. A guard I expected to use and measured out.** The weighted-average share count looked like
the natural reference. It is itself filed in thousands or millions for **56 tickers**
(ConocoPhillips for 97 months; 598 ticker-months more than 10x off), so a guard trusting it
would have replaced correct counts.

**3. Three candidate rules, evaluated offline on all 65,458 name-months.** Version 1 was
committed before its evaluation ran (`280b983`); versions 2 and 3 were written after reading the
one before and are labelled so; version 3's four acceptance criteria were committed before it
ran (`8adeb3f`). **Version 3 met three and failed one, and I did not rewrite the one it failed:**
under $100M 494 -> 0; false values over $4T 52 -> 0; count jumps over 5x 97 -> 3; newest month
within 10% of the live fetch 454 -> 476 - but it turns **436** name-months from a value into
"unknown" against a bar of 150. 368 of those are Visa, Berkshire and Simon Property, whose
shipped values were wrong, which the criterion did not distinguish. Visa and Berkshire have no
usable count in the SEC's `companyfacts` at all (tagged per share class).

**4. Not built, and why.** `pit_record.py` is one of the nine files that stale the ledger, and a
data fix needs the 176 months rebuilt (~95 minutes at the measured rate) plus the rerun. That did
not fit after the diagnosis, and a rule passing three of four criteria is not one to build in
the last hour. **No production line changed for this item.** `plan/backtest-v2.md` and CLAUDE.md
priority 3 carry the build order.

**5. The morning's Next 3.** "Its filings are flagged stale" is in `claims.FORBIDDEN` with a
negative-control case (it was false for all 27 stocks it was printed for on 2026-11-18); and the
first combined 10-Ks have a date in the brief's "Coming up" - **2027-03-02**, the day after a
calendar-year large filer's 10-K is due - with the note committing to it.

### Evidence / research

- A documented failure, measured: `research/measurements/2026-11-19-engine-share-counts.py`
  (`--trace`, `--sweep`, `--evaluate --version 1|2|3`) and `...-readout.py`. The emulation of the
  shipped rule reproduces the engine's own price in 65,381 of 65,458 name-months (99.88%).
- SEC Staff Accounting Bulletin Topic 4.C, "Change in Capital Structure"
  (https://www.sec.gov/interps/account/sabcodet4.htm): the sentence "Such changes in the capital
  structure must be given retroactive effect in the balance sheet" read verbatim today; the
  surrounding Facts paragraph in fragments only.
- No literature on how data vendors clean XBRL share counts was read. That is a gap in the note.
- Nothing here is evidence about returns (rule 5).

### Methodology changed

- None. No `METHODOLOGY_CHANGELOG.md` entry: no score, weight, metric or published sentence
  moved, and the research engine is unchanged.

### Tried and rejected

- **Guarding with the weighted-average count** - item 2 above.
- **A market-value plausibility band as the main mechanism.** A 1,000x slip on a $150bn company
  lands at $150M and on a $4bn company at $4T, both inside any band wide enough for real
  members. Kept only as a $50M backstop.
- **Lowering criterion b to 436 so version 3 "passes".** The bar was written before the run.
- **Building version 3 today.** See Did 4.
- **A fourth version.** Three passes over the same data is already enough to worry about; the
  note says so.

### Not verified

- How far any ledger statistic moves: needs the rebuild.
- The 460-day staleness bound was reasoned, not measured from the distribution of count ages.
- EPS carries the same split-basis error (`trailingEps = eps / split`); not measured.
- `sharesBS` / `sharesBS_prior` read the same balance-sheet concept; not examined.
- The live-fetch comparison is one cross-section, August counts against November's.

### Next

1. **Friday 2026-11-20 is the lab's day**: `EarningsSurprise` on Open Source Asset Pricing
   (`plan/factor-lab.md`); add the dated Progress bullet. (Unchanged from this morning.)
2. **Tomorrow's 02:00 run**: confirm HEALTH: PASS without the continuity override being used, and
   that `kpis.methodology_version` changed. (Unchanged from this morning.)
3. **Build the share-count fix in a 06:00 session with nothing else in it** (about two and a half
   hours): note section 7. Decide criterion b in writing first - build version 3 and record that
   it fails b, or fetch class-level counts for Visa and Berkshire so it passes.
4. Unchanged: the Monday on a hole in the quarterly series (by 2026-12-07); the weekly SEC
   refresh's first run with the co-registrant code (about 2026-11-25).

## 2026-11-19 (owner-run, evening) - A fresh-eyes review: today's #1 is #1 because its price download failed, and three other things nobody was looking at

**Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-19_130000-midday.log` (completed). Data loop published? yes - `logs/datarun-2026-11-19_020001.log` (completed; HEALTH: PASS). Evidence base at `1m`: 5 of 8 needed at the 1m horizon (54 rows, but overlapping windows are not independent; 133 rows across all horizons), newest 1m run date 2026-10-19 (31 days ago, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab** (age **38 days**; a standing programme, last advanced **13 days** ago per `plan/factor-lab.md` (bound 21)); also open: 0.10 43d; 0.14 36d.
(Computed after this session's fix to `health_numbers.py`; before it, the same script said "Data loop
published? **no**", reading the 18:00 retry's no-op log as a failed run - see Did 4.)

**Owner request:** review the project as a reader who had not been following the sessions' notes;
use the product as its users would, then read the code. `logs/.owner-session` was created at the
start and removed at the end. `stash@{0}` not touched (the list was empty). No global tools touched.

**Tests:** baseline not re-run at session start; the 13:00 session's after-count on the same code
(4,394 passed / 1 skipped / 0 failed) is the reference. After: see the gates line below.

### Did

**The review** - `research/2026-11-19-fresh-eyes-review.md`, ranked, every finding marked SAW or
SUSPECT with evidence; four read-only reviewers in parallel (methodology, number tracing, runners and
gates, two weeks of process), each finding re-checked before it went in. The live page at 1440px,
the committed page plus `research.html` and `lab.html` at 375px.

**1. SanDisk is #1 on the live page and in `signals.json` because its price history did not
download.** Its Momentum and Risk are "—"; every price-history field in the run's raw fetch is empty
while its other fields are present. Yesterday it was #7 at 71.02 with Risk 1.03; with those two
scores back it is 71.02 again, #7. `compute_composite` renormalises over the categories present -
which fills a missing category with the stock's own average, not a neutral 50 - and the coverage
discount counts metrics, not weight (0.33% for 23% of the weight). An empty `history()` raises
nothing, so the run's retry pass never saw it. The methodology page said the discount "prevents
stocks with sparse data from ranking artificially high"; the changelog says renormalisation is
"neutral by construction". Both false here.

**2. Fixed (engineering, no score/weight/metric change):**
- `run_screener.missing_price_history` / `merge_recovered_history`: a successful fetch with no price
  history is re-fetched once (bounded at 5% of the universe; beyond that it is an outage the health
  check owns), and a re-fetch replaces a record only if it has a history. `tests/test_history_retry.py`.
- The coverage-discount sentence in `build_screener_overview` now says what the discount does not
  do, with this case; `SCREENER_OVERVIEW.md` regenerated from the generator, page rebuilt with
  `scripts/rebuild_page.py` (payload byte-identical); the old sentence is in `claims.FORBIDDEN` with
  a negative control.
- `check_published_claims.py`: a `TimeoutExpired` returned exit 3 ("pytest unavailable"), which the
  data loop reads as WARN-and-publish. It now refuses (exit 1).
- `health_numbers.data_loop` skips the 18:00 retry's "already completed / standing down" logs.

**3. Proposed, not done** (the owner's item in `OWNER_FOCUS.md`, opened today): score a missing
category at 50 (methodology - research note and changelog first); put the history test's "no
measurable edge" beside the home page's "+10.5%"; make "what is the ranking for, given IC ~0" the
research priority ahead of universe expansion; split the 02:00 claims gate into blocking and warning
checks (9 refusals in two weeks, at most 2 for a real page defect, four weekdays unpublished); cut
the process weight (log archive, ~20-line items in `OWNER_FOCUS.md`, one rule for the four "->red"
rows).

**4. For the sessions, verified and queued:** the EPS-mismatch caveat compares Yahoo's next-year
`forwardEps` with trailing EPS, so it fires on expected growth (SNDK, MU, AVGO flagged with the two
bases agreeing; WDC and CINF not flagged with them 50%+ apart) - 93 stock sheets print "reported and
normalised EPS disagree"; Berkshire's year-ago balances are NaN again in every run 11-13..11-19 (the
2026-10-19 fix regressed); the implied-volatility inversion ACCEPTED on 2026-11-04 has no code and has
left every Next list since 11-05 without a word; the 2026-11-13 retrospective aborted on the missing
CLI and nobody noted it; Jensen's alpha correlates +0.87 with 12-1 momentum and has no skip month.

### Evidence / research

- Measured: SNDK counterfactual and the 20-run history-missing scan (scratch scripts; figures in the
  note), category variance shares, alpha/momentum correlation, data-run log outcomes 11-02..11-19.
- No literature read tonight. The "missing factor = cross-sectional mean" convention in the proposal
  is stated as practice to be sourced by the research note, not as a citation.

### Methodology changed

- None. The overview sentence is a correction of a false description, not a change in method.

### Not verified

- The history retry has not run against the live source; tomorrow's 02:00 log will show a "History
  retry:" line only if a history comes back empty.
- The machinery reviewer's code-path findings (a failed push jamming both loops, the health check
  passing when it cannot read the published payload, Chromium's absence silently skipping ~20% of
  the suite) were read in the code, not observed failing.

### Next

1. **The owner's five decisions** (`OWNER_FOCUS.md`, 2026-11-19 item). Until then, nothing in them is
   to be built.
2. **Fix the EPS-mismatch flag** (S, plus a research-ledger rerun - `factor_engine.py` is a
   `RESULT_PATHS` file) and **find why BRK-B's year-ago balances went missing** (S).
3. **Run the missed retrospective** at the next Friday slot, whatever the week number says.
4. Either build the ACCEPTED implied-volatility inversion or write down why not.
5. Unchanged from 13:00: Friday is the lab's day; tomorrow's 02:00 run - confirm HEALTH: PASS without
   the continuity override, and look for SNDK scored on all eight categories.

## 2026-11-19 (owner-run, late evening) - The owner delegated the review's five decisions: a missing category counts at 50, the home page states the test result, layout-only failures no longer discard a night, and the process files lose two-thirds of their weight

**Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-19_130000-midday.log` (completed). Data loop published? yes - `logs/datarun-2026-11-19_020001.log` (completed; HEALTH: PASS). Evidence base at `1m`: 5 of 8 needed at the 1m horizon (54 rows, but overlapping windows are not independent; 133 rows across all horizons), newest 1m run date 2026-10-19 (31 days ago, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.15, What the ranking is for, given no measured edge** (age **0 days**); also open: 0.13 38d; 0.10 43d; 0.14 36d.

**Owner request:** after the evening's review, *"do whatever you think is best!"* - read as the go-ahead
on all five decisions it put to him. `logs/.owner-session` re-created at the start, removed at the end.

### Did

**1. A category a stock cannot be scored on counts at 50, the sector median** (methodology;
`research/2026-11-19-missing-category-at-neutral.md`; changelog 2026-11-19 late). Before, its weight
was shared across the stock's other categories - the same as filling it with the stock's own average
- which is what put SanDisk at #1. Sources: Chen & McCoy (arXiv 2207.13071: cross-sectional means
work because predictors are weakly correlated - and this screener's categories are built to be) and
Bryzgalova et al. (missingness is not random - stated as the condition that would make this wrong).
On the 11-19 payload **6 of 499 composites move**, the rest identical to the cent: SNDK 1 -> 3, ERIE
413 -> 395, HONA 431 -> 409, FDXF 328 -> 316, DPZ, BRK-B by a few places. One engine helper
(`factor_engine.category_terms`) feeds the composite, contributions and weight sensitivity. **Each run
records the rule it was scored with** (`effective_weights.json` -> `weights.missing_category_score`):
the first rehearsal regenerated today's run under the new generator, claimed the new rule for
composites built under the old one, and the build refused it - so the generator now passes the
record through and never adds it, `calc_trace` checks each payload by the rule it states, and the
page keeps the old wording for an old payload. Seen in a browser on a throwaway 5-stock run
(FedEx Freight: Momentum, Risk, Investment tiles "counted at 50.0", points adding up), the run
deleted and its four files restored afterwards.

**2. The EPS-mismatch caveat compares like with like** (display only; changelog). GAAP trailing EPS
against the consensus basis's own last four quarters, flagged at opposite signs or more than 1.5x
apart; the old forward/trailing ratio only where that figure is missing. 93 -> 104 flagged on 11-19,
72 in both; SNDK, MU, AVGO no longer flagged, WDC, CINF, BRK-B now are.

**3. The home page states the history test** beside the track record: "Tested on 176 months of
2012-2026 history, the ranking has shown no measurable edge at predicting the next month's returns.
See the test" (`generate_dashboard.history_verdict`, claim `page.hero_tested`, the same rule as the
methodology page's line). Seen at 375px: no sideways scroll.

**4. The 02:00 claims gate publishes on layout-only failures** (exit 6, `check_published_claims.
PRESENTATION`, 22 named layout tests, each with a reason; `data-run.ps1` logs `ATTENTION - PAGE
LAYOUT` and publishes). Any failure outside that list, a timeout, or no FAILED line to classify still
refuses. `data-run.ps1` parse-checked with PowerShell's own parser (0 errors). Tests pin that no
reproducibility, claims, advice, payload or evidence check can be listed.

**5. Process weight cut:** `NIGHTLY_LOG.md` entries to 2026-10-19 moved verbatim to `log-archive/`
(1.57 MB -> 0.73 MB live); `OWNER_FOCUS.md`'s Open section 1,194 -> ~120 lines (histories verbatim in
`log-archive/OWNER_FOCUS-item-histories-2026-11-19.md`), the finished 2026-10-20 item closed;
CLAUDE.md's four "->red" rows merged into one `premise->red` row (verbatim in `DECISIONS.md`), a
settled row 0.8m for the missing-category rule, and **0.15 - what the ranking is for, given no
measured edge - is the new top item, ahead of universe expansion phase 2** (decision 3; the universe
item says so).

**Research ledger rerun** (`factor_engine.py` changed; 32 rows, ~19 min with `--jobs 4`). *Backtest
observation, not decision-grade:* pooled baseline IC -0.00116 -> -0.00133, interval still spans zero,
0 alternatives better; `research.html`, `RESULTS.md`, `summary.json` rewritten by `report`.

### Gates

Gate 1: see the line below. Gate 2 (dry run) and gate 3 (`node --check` of all three payloads) ran
after it. Gate 5, `rehearse_publish.py --changed-since origin/main`: **REHEARSAL PASSED** (claims gate
2,390 passed / 1 skipped), published files restored byte for byte. Gate 4 clean.

### Not done, and why

- **A missed retrospective running at the next slot** - a runner change; the next scheduled one is
  2026-11-27 and should cover 11-13's.
- **Berkshire's year-ago balances, the implied-volatility inversion, the Jensen's-alpha note** - listed
  in the 2026-11-19 owner item for the sessions.
- **The bot** reads `signals.json`; tomorrow's run moves six stocks, SNDK out of #1. Its continuity
  gate cannot be breached by six moves, and `methodology_version` (a config hash) does not change.

### Next

1. **Tomorrow's 02:00 run:** HEALTH: PASS; `weights.missing_category_score` = 50 in the payload; any
   "History retry:" line in the log; whether SNDK has its price history back.
2. Friday 2026-11-20 is the lab's day (unchanged).
3. **0.15** - a Monday note: what the ranking is for, with what would count as an answer written first.

## 2026-11-19 (owner-run, night) - "keep finding improvements until the 2 am run": one false claim made true, four blind spots surfaced, a Drives column, and the top priority pre-registered

**Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-19_130000-midday.log` (completed). Data loop published? yes - `logs/datarun-2026-11-19_020001.log` (completed; HEALTH: PASS). Evidence base at `1m`: 5 of 8 needed at the 1m horizon (54 rows, but overlapping windows are not independent; 133 rows across all horizons), newest 1m run date 2026-10-19 (31 days ago, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.15, What the ranking is for, given no measured edge** (age **0 days**); also open: 0.13 38d; 0.10 43d; 0.14 36d.

**Owner request:** *"keep working and finding improvements in every aspect until the 2 am run please"*.
`logs/.owner-session` re-created at the start, removed at the end; merged well before 02:00.

### Did

1. **Berkshire's missing year-ago balance sheet: not a code regression.** Checked live: Yahoo's
   quarterly sheet for BRK-B now jumps from 2025-06-30 to 2025-12-31, so nothing lies within 35 days
   of a year before 2026-09-30 and the 2026-10-19 date rule correctly finds none. A filings fallback
   was refused on 2026-10-15; nothing changed. Recorded in the review note.
2. **Sustainable growth keeps the ROE its claim says it shares** (changelog 2026-11-19 night). The
   published claim `overview.bank_roe_roa` says bank ROE has no fallback to an ending balance and is
   "the same ROE as the one inside sustainable_growth"; sustainable growth fell back to ending equity
   for five stocks - STX and MTD (year-earlier equity negative) sat at the 100% clamp, BA read 29% on
   equity that had just turned positive. Now: both balances positive or no figure.
   `tests/test_bank_roe_roa.py`, `tests/test_metrics.py` (the old fallback test inverted).
3. **The brief names a top-40 stock that lost a category since the last published run**
   (`check_run_health.lost_categories`, an ATTENTION note, never a refusal) - today's SNDK would have
   read "SNDK (#1: momentum, risk)". Also ATTENTION lines now, not quiet notes: rank continuity
   unchecked because the published payload could not be read (the bot trades on that run), and
   shards not folded (how 0 of 485 market values were compared on 11-08 with every line green); and
   the claims gate says so when more than 10 checks were skipped (losing Chromium skips ~886 with
   pytest still exiting 0).
4. **Diagnostics shows what each category drives** beside its weight (`D.variance_share`, the
   standard w·cov/var decomposition, display only): Valuation 34.1%, Quality 29.1%, Momentum 15.9%,
   Revisions 7.2%, Risk 7.0%, Investment 3.1%, **Growth 1.9% on a 13% weight**, Size 1.7%. A caption
   says why without naming a category, so it cannot go stale. Phones keep Weight and Drives and drop
   the Jaccard bar column. Seen on the rehearsed page at desktop and 375px.
5. **A missed retrospective is owed at the next Friday** (`nightly-screener.ps1`; parse-checked; a
   test matches its pattern against the real log headings). The last ran 2026-10-30, so **tomorrow,
   Friday 2026-11-20, is a RETROSPECTIVE**, not the lab's day; the lab's "last advanced" will read 20
   days on 11-27 against its bound of 21 - the next Friday must be the lab's.
6. **0.15 pre-registered** (`research/2026-11-19-what-the-ranking-is-for.md`): four candidate answers
   (a screen; a slow forecast at 6-12 months; fewer factors; a teaching tool) and the criteria for
   each, fixed before any measurement, with the engine's positive control required in the same run.
   The brief's "Coming up" carries it for Monday 2026-11-23, and the dropped implied-volatility
   build for 2026-11-24 (it needs QuantLib installed by a scheduled session, not the desktop app).

**Research ledger rerun** for the `factor_engine.py` change: summary unchanged (pooled IC -0.00133,
interval -0.0177 to +0.015, 0 better). *Backtest observation, not decision-grade.*

### Gates

Gate 1 **4,438 passed / 1 skipped / 0 failed** (4:31; one failure on the first run - the Drives figure
skipped the page's near-zero percentage rule - fixed). Gate 2 dry run passes; gate 3 `node --check` on
all three payloads passes; gate 5 **REHEARSAL PASSED** (claims gate 2,391 passed / 1 skipped),
published files restored byte for byte; gate 4 clean.

### Next

1. **Tomorrow 06:00 is a retrospective** - it should read today's three owner-run entries and the
   review note, and judge the presentation-only exit and the archived files.
2. Tomorrow's 02:00 run: HEALTH: PASS, `missing_category_score` 50 in the payload, the new
   `variance_share`, any "History retry:" or "lost a category" line.
3. Monday 2026-11-23: 0.15 against its pre-registered criteria.
