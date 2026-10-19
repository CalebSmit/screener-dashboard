# Is the top of the ranking there on real figures? A data audit of the top 40 (2026-10-19)

**The question.** A public ranking, and a paper-trading bot that buys its top 25, are only as credible
as the inputs at the top. For each of the 2026-10-19 run's top 40 stocks (which include all 25 the bot
holds), is every statement input that a weighted metric reads the company's own figure for the period
the run scored? Is any of it wrong, a period stale, on the wrong scale, or a definitional outlier
that moves a stock 10 or more percentile points? And if the bad inputs were corrected, how many
names would leave the top 25?

**The answer.** **No input in the top 40 is wrong, and none is on the wrong scale.** Every
disagreement over 10% traces either to the SEC side (a filer's tagging) or to a definition the
source uses and the screener has already chosen to keep. Two stocks are scored a period behind
their filings (Micron, Eversource); correcting both moves **one name out of the top 25: Western
Digital, from #25 to #26**, displaced by Eversource (#28 to #25). Micron rises from #15 to #2.
Staleness held both stocks *down*, so it is not what holds anything up.

The audit did find two real defects, neither in the live inputs:

* **`pit_record` (backtest v2) read a mis-tagged 10-Q over the 10-K.** Comfort Systems' Q1 2026
  10-Q tags its first-quarter figures with a calendar-2025 context, and "latest filed wins" took
  them: FY2025 revenue $1.83bn against the 10-K's $9.10bn, trailing revenue $3.96bn against
  $11.23bn, and its gross profit, operating income, pretax income and D&A the same way. **Fixed**
  (section 4), with tests.
* **One company, two scores.** Fox Class B (FOX, #31) has no consensus estimates, so its forward
  EPS growth falls back to forward EPS over GAAP trailing EPS and reads **+49.6%**; Fox Class A
  (FOXA, #47) - the same company, the same $3.84 of diluted EPS per share - reads **+9.7%** on the
  consensus basis. On FOXA's figure FOX falls from #31 to #64. Alphabet's two classes carry FY1
  consensus of $13.14 (GOOG) and $20.59 (GOOGL) and score +31.5% and -18.7%. That is a methodology
  question - which figure stands for the company - so it is written up as an open proposal
  (section 6), not built.

Nothing here changes a weight or a live score.

## 1. Method

`research/measurements/2026-10-19-top-of-ranking-data-audit.py` (JSON beside it). Run 377ec40838b1
(2026-10-19 02:00), SEC companyfacts cache filed through 2026-10-15.

* **The same period as the run.** Each input is rebuilt from the filings for the run's own dates:
  trailing twelve months ending at `_stmt_date_financials` / `_stmt_date_cashflow`, balances at
  `_stmt_date_balance_sheet` (and a year earlier for asset growth and average equity), the two
  quarters revenue growth compared, and the fiscal years the 3-year revenue CAGR used.
* **Two comparisons per input.** *Nearest*: the closest figure the company filed under any tag or
  construction that can carry the line (revenue: every revenue tag; EBIT: operating income, the same
  plus restructuring and impairment charges, pretax income plus interest; debt: borrowings with and
  without leases). Over 10% from all of them, the run's number is not one the company reported, and
  it is tested for a neighbouring period (a quarter either side, a year earlier) and for a scale
  error (a power of ten or a split ratio). *Primary*: the standard tag. A figure near a company
  number but far from the standard one is a definition.
* **Stakes, by the engine.** One input at a time is replaced, `compute_metrics` recomputes that
  stock, the changed metric columns are written into the run's own `02_outliers_flagged` frame, and
  the engine's percentile, category, composite and rank steps re-rank all 501 stocks. With no
  substitution this reproduces every published composite exactly (the script asserts it).
* **Also checked:** market value against filed share counts; splits inside the momentum window
  (`cache/splits.parquet`); the price history (12-month and 6-month anchors against
  `cache/v2_daily.parquet`); statements a period behind a filing; dual share classes.

## 2. The top 40, input by input

818 inputs, 738 of them with a filed figure to compare: **689 agree within 2%**, 22 within 10%,
27 over 10% (25 "far", 2 annual figures closest to a neighbouring fiscal year); 80 had no filed
figure to compare with.
"Bot" = held by the bot on 2026-10-15.

| # | Stock | Bot | Composite | Agree | Near | Far | Not filed | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | EXPE | yes | 74.97 | 20/20 | 0 | 0 | 1 | Clean. Market value includes the Class B shares (implied 120.0M = 114.5M + 5.5M) |
| 2 | APA | yes | 71.37 | 12/13 | 1 | 0 | 6 | Clean where checkable; APA files no standard revenue or capex tag |
| 3 | INCY | yes | 71.08 | 5/5 | 0 | 0 | 15 | **Not checkable**: SEC's companyfacts ends at its Q1 10-Q (re-fetched 2026-10-19) |
| 4 | HST | yes | 70.82 | 18/19 | 0 | 1 | 2 | Debt: SEC-side (its notes are not under the standard debt tags). Known caveat: hotel-sale gains in net income |
| 5 | SNDK | yes | 68.99 | 19/20 | 0 | 1 | 0 | Clean (debt 12% under a $201M filed total; no score effect) |
| 6 | EG | yes | 68.22 | 13/15 | 1 | 1 | 3 | Clean; FY revenue sits between FY2024 and FY2025 filings (both within 1.6%) |
| 7 | CF | yes | 68.20 | 18/20 | 1 | 1 | 1 | Net income is the parent's share; CF tags only `ProfitLoss` (incl. the CHS minority interest) - source is right |
| 8 | VLO | yes | 67.95 | 19/20 | 0 | 1 | 1 | D&A: SEC-side (the standard tag holds a $50M component) |
| 9 | BMY | yes | 67.55 | 21/21 | 0 | 0 | 0 | Clean |
| 10 | ALL | yes | 67.48 | 15/16 | 0 | 1 | 4 | 3-year-ago revenue equals FY2021's filed figure and is 1.5% from FY2022's; too close to call, no effect |
| 11 | BBY | yes | 67.46 | 20/21 | 1 | 0 | 0 | Clean |
| 12 | EXPD | yes | 67.36 | 19/19 | 0 | 0 | 1 | Clean |
| 13 | DXCM | yes | 67.29 | 18/20 | 0 | 2 | 0 | Debt and short-term investments: SEC-side (convertible notes and marketable securities under other tags) |
| 14 | MPC | yes | 66.64 | 19/21 | 0 | 2 | 0 | Source operating income 18% *below* the filed figure (a definition; it is not the source's special-charges row); the filed figure would lift it to #6 |
| 15 | MU |  | 66.28 | 20/21 | 0 | 1 | 0 | **A period behind**: scored on May 2026; the FY2026 10-K was filed 2026-10-09 |
| 15 | CAH | yes | 66.28 | 19/20 | 1 | 0 | 1 | Operating income before special charges (settled definition); on filed OI #20 |
| 17 | TPR | yes | 66.22 | 20/21 | 0 | 1 | 0 | D&A 21% above filed; 2 points on EV/EBITDA, no rank change |
| 18 | ED | yes | 66.04 | 5/5 | 0 | 0 | 16 | **Not checkable**: companyfacts ends at its Q1 10-Q (re-fetched 2026-10-19) |
| 19 | AIZ | yes | 65.91 | 17/18 | 0 | 1 | 2 | Cash definition; insurers are not scored on it |
| 20 | NTAP | yes | 65.81 | 21/21 | 0 | 0 | 0 | Clean (revenue is the contract-revenue line, as filed) |
| 21 | TRV | yes | 65.12 | 16/17 | 0 | 1 | 2 | Cash definition; not scored for an insurer |
| 22 | SWK | yes | 65.07 | 20/21 | 1 | 0 | 0 | Clean |
| 23 | DLTR | yes | 64.94 | 18/18 | 0 | 0 | 2 | Clean |
| 24 | ACGL |  | 64.92 | 10/15 | 4 | 1 | 5 | Cash definition; not scored for an insurer |
| 25 | WDC | yes | 64.76 | 19/21 | 2 | 0 | 0 | Inputs clean. Known caveat: net income 2.56x after-tax operating income (SanDisk stake) |
| 26 | DECK | yes | 64.52 | 18/18 | 0 | 0 | 2 | Clean |
| 27 | DVA |  | 64.45 | 19/19 | 0 | 0 | 1 | Clean |
| 28 | ES | yes | 64.26 | 18/19 | 1 | 0 | 2 | **A period behind**: scored on March 2026; Q2 10-Q filed 2026-08-03 (77 days) |
| 29 | EOG |  | 64.06 | 15/18 | 2 | 1 | 3 | FY2022 revenue gross of derivative losses (13% above filed); on filed #26 |
| 30 | BKNG |  | 63.87 | 17/19 | 1 | 1 | 1 | 25:1 split 2026-04-06 handled (prices, EPS, share count all on one basis); D&A 20% above filed, no effect |
| 31 | FOX |  | 63.71 | 17/19 | 1 | 1 | 1 | **Forward EPS growth on the fallback basis** (section 6); EBIT definition (FOX files no operating-income line) |
| 32 | JBHT |  | 63.66 | 20/20 | 0 | 0 | 1 | Clean |
| 33 | HAS |  | 63.56 | 15/20 | 1 | 4 | 1 | Source capex 2.7x the PP&E tag (it understates FCF yield - conservative); gross profit and D&A definitions |
| 34 | FIX |  | 63.50 | 20/21 | 0 | 1 | 0 | Clean - **after** the `pit_record` fix (before it: 7 "far" inputs, all the SEC side's) |
| 35 | CNC |  | 63.46 | 20/20 | 0 | 0 | 0 | Operating income before a goodwill impairment (settled); on filed OI #78 |
| 36 | AVY |  | 63.37 | 19/21 | 1 | 1 | 0 | Capex 15% above filed (conservative) |
| 37 | CL |  | 63.34 | 19/20 | 0 | 1 | 1 | Operating income before the FY2025 $919M goodwill and intangible impairment (filed OI $3.13bn + $0.92bn = source's $4.06bn); on filed OI #51 |
| 38 | HIG |  | 63.28 | 14/17 | 2 | 1 | 2 | Cash definition; not scored for an insurer |
| 39 | NEM |  | 63.18 | 18/19 | 1 | 0 | 2 | Clean |
| 40 | SJM |  | 63.09 | 19/20 | 0 | 1 | 1 | Operating income before $962M of FY2026 impairments (filed $0.83bn + $0.96bn = source's $1.80bn); on filed OI #91 |

**Prices and splits.** Every top-40 price anchor matches the daily cache to within the few days'
difference in anchor date. The only split inside a momentum window is Booking's 25:1 (2026-04-06),
and its 12-month anchor ($196.30) is on the post-split basis. No mis-scaled price or share count.

**Market value against filed shares.** 37 of 40 within 1.5%. Expedia (+4.8%) is its Class B
shares, which belong in market value. Western Digital (+3.8%) understates its yields slightly,
which is conservative. Fox's two classes are discussed in section 6.

## 3. How many names would leave the top 25

Each class of finding corrected separately, then together, the whole universe re-ranked
(`scenarios` in the JSON). Only the top 40's records change.

| Correction | Top 25 loses | Top 25 gains | Largest moves |
|---|---|---|---|
| 1. Statements a period behind brought up to the newest filed quarter (MU, ES) | WDC (#25 -> #26) | ES (#28 -> #25) | MU #15 -> #2 |
| 2. FOX's forward EPS growth on FOXA's consensus figure | - | - | FOX #31 -> #64 |
| **1 + 2: the corrections that are not a methodology choice** | **WDC** | **ES** | |
| 3. EBIT = filed operating income (sensitivity on a settled choice) | - | - | SJM #40 -> #91, CNC #35 -> #78, CL #37 -> #51, CAH #15 -> #20, MPC #14 -> #6 |
| 4. Earnings yield on after-tax EBIT where net income > 1.25x it (sensitivity) | WDC (#25 -> #41) | DECK | HST #4 -> #4 |
| 1 + 2 + 3 + 4 | WDC | ES | |

**One name out of the top 25 on corrected inputs, and it is the 25th.** The bot already holds both
WDC and ES. Nothing in the top 24 moves out under any correction, including the two sensitivities on
choices the screener has already made.

How "a period behind" was scored: both quarters are rebuilt from the filings (`pit_record`) and the
*change* between them is added to the run's own metric value, so the source's definitions are kept
and only the newer quarter's news moves the stock. Micron's newer 10-K reports the memory cycle
(total assets $196bn against $134bn a quarter earlier; three-year revenue CAGR 6.7% -> 105% as
FY2023 leaves the window), Eversource's Q2 lifts its Piotroski score (7 -> 8) and lowers its asset growth and leverage, which outweighs a lower
earnings yield and revenue growth.

## 4. The defect fixed: a 10-Q cannot overwrite a fiscal year

`pit_record.Filed` keeps, of several versions of a context, the latest filed - so a restatement
filed before a date is what an investor read then. A 10-Q does not report a fiscal year, though,
and when one carries a fiscal-year context it is a comparative at best and a mis-tag at worst.
Measured on the cache (`tenq_fiscal_year_census`): **2,173 fiscal-year-length facts sit in 10-Qs;
135 disagree with the 10-K's value for the same context by over 2%; 29 of those (13 companies) were
never superseded by a later 10-K**, so "latest filed" read them. Most are old (AXP 2007, AEE 2011,
GE 2012-13); the recent ones are FIX (8 concepts, FY2025), CBOE and ZBRA (FY2025 impairment and
restructuring tags), HPE and MNST (FY2024 goodwill impairment).

Rule now: of several versions of a **fiscal-year** context the latest 10-K (or 10-K/A) wins; a
10-Q's value is used only where no annual report carries that year. Quarters and year-to-dates
still take the latest filing. Restatements still flow (a restated year arrives as a later 10-K's
comparative). `tests/test_pit_record.py`: `test_a_10q_cannot_overwrite_a_fiscal_year_its_10k_reported`,
`test_quarters_and_year_to_dates_still_take_the_latest_filing`.

**What it touches.** `pit_record` is backtest v2's record builder and is on
`v2_experiments.RESULT_PATHS`, so the ledger's next run will report the code hash as changed. The
v2 monthly records for FIX from May 2026, and the EBIT add-back for the four impairment cases, will
change on the next rebuild; the 2026-10-15 batch should be rebuilt and rerun before its numbers are
quoted again. Nothing in a published run reads `pit_record` (its tests enforce that), and the live
filing check (`sec_crosscheck`) already reads first-filed values, which is why it never saw FIX's
mis-tag. `sec_fundamentals.roe_history` reads 10-K facts only.

## 5. What is not a defect, and why

* **Operating income before special charges** (CAH, CNC, CL, SJM). The source's "Operating Income"
  is as-reported less its "Special Income Charges" - settled 2026-10-15
  (`research/2026-10-15-earnings-definition.md`): Rouen, So & Wang (2021) and MSCI support the
  pre-special-items figure for ROIC and EV/EBITDA. Measured here as a sensitivity only: it moves no
  name out of the top 25.
* **Net income above operating earnings** (HST, WDC). Already said on the page and in the morning
  line by `ranking_audit`; the proposal to score earnings yield on normalized income was refused
  2026-10-16. On after-tax operating income WDC would rank #41.
* **Staleness.** The engine is right not to read the source's empty column: Yahoo lists Eversource's
  2026-06-30 quarter with 4 of ~51 income rows filled (checked live 2026-10-19). Micron's 10-K is 10
  days old, inside any reasonable source lag. A rule that filled the quarter from the SEC would put
  scoring on the weekly SEC cache, which scoring deliberately never reads - and the measurement says
  staleness did not hold either stock *up*. No rule is proposed.
* **3-year revenue CAGR endpoints.** Across the universe the annual revenue the run used matched the
  latest fiscal year's filing for 472 of 494 stocks and the three-years-earlier figure for 466 of
  491; the few "other year" matches are coincidences of value (CLX, DE, MTB, BNY), not misaligned
  columns. No gap check is needed.
* **SEC-side gaps.** HST's and DXCM's debt, VLO's D&A, the insurers' cash: the filer uses a tag the
  check does not read. The source's figure is the plausible one in every case examined.

## 6. One company, two scores - for a later session

**Proposal status: OPEN** - score a share class whose consensus estimates are missing, or disagree
with its sibling class's, on one company-level consensus figure (the class with more analysts), and
say on the page when two rows are one company.

*The case.* Forward EPS growth is a per-share figure, and every share class of one company has the
same claim per share: FOX and FOXA report the same $3.84 of diluted EPS, NWS and NWSA carry
identical consensus. But:

| Pair | Class | Forward EPS growth | Basis | Growth score | Rank |
|---|---|---|---|---|---|
| Fox | FOX (B) | +49.6% | fallback: forward EPS / GAAP trailing EPS | 74.1 | 31 |
| Fox | FOXA (A) | +9.7% | MSCI 12-month consensus | 45.7 | 47 |
| Alphabet | GOOG (C) | +31.5% | MSCI, FY1 consensus $13.14 | 71.9 | 133 |
| Alphabet | GOOGL (A) | -18.7% | MSCI, FY1 consensus $20.59 | 41.1 | 226 |

FOX's fallback mixes an adjusted forward figure with GAAP trailing EPS, the defect the 2026-10-09
MSCI construction removed (`research/2026-10-09-forward-eps-growth.md`); it remains only where
consensus is missing - 7 stocks on this run, FOX the only one in the top 40 and the only one with a
sibling class that has the consensus. Alphabet's classes disagree on FY1 consensus itself, by 57%.

*What to decide.* (a) Whether a class may borrow its sibling's per-share consensus (exact for EPS;
**not** for price targets, which are in each class's own price); (b) which class's figure stands
when both exist and disagree (more analysts is the obvious rule; it needs checking against a third
source); (c) whether the ranking should mark - or the bot should hold at most one of - two rows that
are one company. Context: the S&P 500 holds 503 securities for 500 companies, so dual classes are a standing feature
of this universe; and a fundamental measured per share cannot differ between classes with the same
claim per share, so a difference between them is a data artefact by construction. Measured stakes on this run: FOX #31 -> #64 on FOXA's figure; no top-25 change.

## 7. Reproduce

`python research/measurements/2026-10-19-top-of-ranking-data-audit.py` (about a minute) reads the
newest run under `runs/`, `data/sec/pit/facts.parquet` and `cache/splits.parquet`, and writes
`2026-10-19-top-of-ranking-data-audit.json`: per stock, every input's run value, nearest and primary
filed figure, tag, relative difference, diagnosis and stakes; the scenarios of section 3; the 10-Q
census of section 4. INCY's and ED's missing quarters were confirmed by fetching their companyfacts
directly on 2026-10-19 (latest filings 2026-04-28 and 2026-05-07).
