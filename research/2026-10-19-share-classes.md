# One company, two rows: how the screener should treat share classes (2026-10-19)

*Proposal open (status line at the end): score each company once, on its designated listing, and
show the other class as an unranked line of the same company. For a later session to decide; no
scoring code was changed.*

**The question (from the top-of-ranking audit, section 6).** The S&P 500 holds 503 lines for 500
companies, so three companies appear in the ranking twice. Fox Class B (FOX) has no consensus
estimates and falls back to forward EPS over GAAP trailing EPS, scoring forward EPS growth +49.6%
against Fox Class A's (FOXA) +9.7%; on FOXA's figure FOX drops from #31 to #64. Alphabet's two
classes carry FY1 consensus of $13.14 (GOOG) and $20.59 (GOOGL) and score +31.5% and -18.7%. Which
figure stands for the company, should two lines of one company both be ranked, and what does the
paper-trading bot, which buys the top 25, need?

**The answer, in five lines.**

1. **The universe has three companies with two ranked lines (Alphabet, Fox, News Corp) and 21 more
   with a second class that is not ranked** (four listed outside the index - BRK-A, BF-A, LEN-B,
   MKC-V - and the rest unlisted). Every statement-driven metric is identical across a pair, as it
   must be. The rows differ for three reasons: **price** (earnings yield differs by exactly the ratio
   of the two prices, to four decimals, and enterprise value by exactly the gap in market value),
   **each line's own return series**
   (momentum and risk), and **the source's analyst record for the line**, which is the only
   difference with no economic content: FOX has none, NWS has no price target, and GOOG and GOOGL
   carry different consensus *and* different earnings histories.
2. **Index providers measure fundamentals once per company and use each line's own price.** MSCI
   computes per-share data "on a company level in which all classes (listed and unlisted) of equity
   are aggregated"; FTSE Russell gives every share class the style probabilities of the company's
   pricing vehicle; S&P DJI evaluates style "at the company level", decides index eligibility on the
   company's total market value, and for a list that should hold each company once uses a
   **Designated Listing** - the line with the highest liquidity and float-adjusted value.
3. **Scoring the company once, on its designated line, and dropping the other line from the ranking
   leaves the top 25 unchanged** on the 2026-10-19 run. Fox scores on FOXA (#47 -> #28), Alphabet on
   GOOGL (#226 -> #197), News on NWSA (#61 -> #52). It also removes a distortion nobody had measured:
   the three duplicated companies are 6 of the 22 rows in the Communication Services peer group, and
   removing the second lines raises the other 19 stocks' composites by **1.56 points on average**.
4. **The cheapest-looking fix is the worst.** Filling only a line's *missing* analyst fields from its
   sibling would put FOX at **#6**: it keeps its inflated +49.6% fallback growth and gains FOXA's
   revision record, and it would enter the top 25.
5. **The bot can hold two lines of one company today.** It buys the top 25 and holds to rank 40 with
   no company check; FOX and FOXA were both in the top 25 on 4 of 59 snapshots (2026-02-20 to 02-23)
   and both inside the hold band on 8. It holds none of the six lines now. Scoring once closes that
   in this repo, with no change to the bot.

A separate defect turned up and is **not** part of the proposal: Alphabet's two lines show that the
source's quarterly "actual EPS" for GOOGL is GAAP (it equals the 10-Q to the cent) while its estimates
are on the analysts' basis, so the MSCI construction's trailing base mixes accounting bases - the
defect the 2026-10-09 forward-EPS change was meant to remove. Amazon shows the same signature.
Section 7.

---

## 1. Method

`research/measurements/2026-10-19-share-classes.py` (offline, about 30 seconds; JSON beside it). Run
`377ec40838b1` (2026-10-19 02:00, 501 scored), the SEC ticker file `data/sec/company_tickers.json`, the
weekly SEC facts cache, and the 59 snapshots in `improvement/snapshots/`.

- **Inventory.** Lines in the universe that share an SEC CIK; lines whose company also lists another
  common class outside the universe (from the same file, checked by hand - it also lists preferreds,
  notes and warrants, which are not equity classes); lines whose own share count is under 90% of the
  company's balance-sheet count. For each, whether the source's market value is the *company's* (all
  classes at this line's price) and whether its per-share figures are on this line's unit.
- **Per pair.** Every weighted metric on both rows of the run's own metric table
  (`02_outliers_flagged`), which input class makes them differ, and the composite gap by category.
- **Stakes.** The run's metric table rescored with the run's own configuration and effective weights
  through the engine's own `compute_sector_percentiles` -> `apply_percentile_transform` ->
  `compute_category_scores` -> `compute_composite` -> `rank_stocks`. **The baseline reproduces all
  501 published ranks** (asserted). An option replaces metric values on the affected rows only.
- **The bot.** `CalebSmit/screener-bot` at commit `d06ff82` (`bot/strategy.py`, `config.toml`,
  `journal/sim/positions.json`), read with `gh api`; not changed.

## 2. Inventory: every multi-class company in the universe

**Two ranked lines (same CIK):**

| Company | Lines (votes) | Price | Designated (higher $ volume) | Volume ratio |
|---|---|---|---|---|
| Alphabet | GOOGL (Class A, 1 vote) / GOOG (Class C, none) | $340.59 / $336.91 | GOOGL | 1.5x |
| Fox | FOXA (Class A, none) / FOX (Class B, 1 vote) | $64.15 / $57.22 | FOXA | 4.8x |
| News Corp | NWSA (Class A, none) / NWS (Class B, 1 vote) | $30.03 / $32.88 | NWSA | 3.2x |

The price gaps are not one "voting premium": Fox's voting class trades 10.8% *below* its non-voting
class, News Corp's 9.5% above, Alphabet's 1.1% above. Liquidity and index demand move them as much as
votes do.

**One ranked line, another listed common class outside the universe:** BRK-B (BRK-A, 1,500 B shares
each), BF-B (BF-A), LEN (LEN-B), MKC (MKC-V).

**One ranked line whose own share count is under 90% of the company's** (an unlisted or unindexed
class): DELL, ECHO, RL, BF-B, EL, ABNB, HSY, RDDT, AOS, TSN, NKE, WDAY, COIN, LEN, META, UHS, HOOD, UPS,
ERIE (and PCG, which is not a share-class case: its balance-sheet count differs for another reason, not
investigated). For every one of them **the source's market value is the company's** - all classes at
this line's price, within 2.5% of price x balance-sheet shares - so size, earnings yield and the EV
ratios are company figures. Across the universe that holds for 457 of 502 lines within 2%.

**Per-share units.** Dividing equity by the source's `bookValue` and net income by `trailingEps` and
comparing with the company's share count finds **one per-share figure on another class's unit:
Berkshire's `bookValue` is per Class A share** ($522,226), while its price is per Class B share, so its
`priceToBook` reads 0.0010 instead of about 1.53 (equity / (Class A-equivalent shares x 1,500) = $348.26
per B share). Its `trailingEps` is per B share and correct. This feeds 60% of Berkshire's bank Valuation
and is written up in `research/2026-10-19-bank-price-to-book.md`. (The same check flagged SKYD's share
count, which is a separate source anomaly with no share-class content; SKYD ranks 500th.)

## 3. The three pairs, metric by metric

Of the 27 weighted metrics, **every company-statement metric is identical within each pair**
(ROIC, GP/A, net debt/EBITDA, Piotroski, accruals, Beneish, revenue growth, 3-year CAGR, sustainable
growth, asset growth). What differs, and why:

| Input class | Metrics | Fox | Alphabet | News | Economic? |
|---|---|---|---|---|---|
| Company figure over company value **at this line's price** | EV/EBITDA, FCF yield, earnings yield, EV/sales, size | differ | differ | differ | **Yes** - a buyer of the cheaper line pays less for the same claim. Earnings-yield ratio x price ratio = **1.0000** for all three pairs; EV gap = market-value gap to within $14m |
| **This line's own return series** | 12-1 and 6-month return, Jensen's alpha, volatility, beta, drawdown | differ | differ | differ | **Yes** - what a holder of that line earned |
| **This line's own market data** | short interest ratio | 9.72 vs 7.30 | 2.99 vs 2.96 | 2.41 vs 3.31 | Yes, line-level |
| **The source's analyst record for the line** | forward EPS growth, FY1 revision, surprise, beat streak, target upside | FOX has **none** | consensus **and** histories differ | NWS has no target | **No** - per-share forecasts of one company's earnings |

The first two classes are what MSCI calls security ratios: a company figure over the security's
price. The fourth is a data artefact by construction - every share of Fox's two classes has the same
claim to the same $3.84 of diluted EPS (both rows carry it), and News Corp's two lines carry identical
consensus to the cent.

**Where each composite gap comes from** (designated minus other line, composite points):

| Pair | Gap | Valuation | Growth | Momentum | Risk | Revisions | Size |
|---|---|---|---|---|---|---|---|
| FOXA - FOX | -1.50 | -1.74 | **-3.69** | +0.80 | -0.45 | **+4.05** | -0.45 |
| GOOGL - GOOG | -3.96 | -1.05 | **-4.00** | +0.59 | -0.65 | +1.38 | -0.23 |
| NWSA - NWS | +3.87 | +1.89 | 0 | +1.36 | +0.26 | +0.14 | +0.23 |

News Corp's gap is all price and returns. Fox's and Alphabet's are dominated by the analyst record:
FOX gains 3.7 points from the fallback growth figure and loses 4.1 because, with no analyst data, its
Revisions score rests on short interest alone (25.0). FOX's #31 is two errors in opposite directions.

## 4. What index providers do - read this session

**MSCI, *Fundamental Data Methodology* (March 2021).**
https://www.msci.com/eqb/methodology/meth_docs/MSCI_Fundamental_Data_Methodology_Mar2021.pdf

- Section 1 (p. 10): MSCI "calculates the data on a company level in which all classes (listed and
  unlisted) of equity are aggregated", and to compute per-share figures uses "the total number of
  shares at the company level at the date of the results". The rule covers both kinds of per-share data it
  publishes, historical and forecast (section 1.3, pp. 16-17: one consensus per company feeds EPS12F).
- Section 2.1 (p. 19): every valuation ratio is "Current security price" over the per-share figure.
  So two lines of one company share every fundamental and differ only through their prices - exactly
  the split section 3 measured.

**MSCI, *Global Investable Market Indexes Methodology* (October 2026).**
https://www.msci.com/index/methodology/latest/GIMI

- Section 2.2.2 (p. 16): "Some of the investability requirements are applied at the individual
  security level and some at the overall company level ... the inclusion or exclusion of one security
  does not imply the automatic inclusion or exclusion of other securities of the same company."
  Size is a company-level screen (2.2.3, p. 17); full company market value aggregates listed and
  unlisted lines, unlisted ones at the listed price times a conversion factor for "relative dividend
  rights or relative par value" (p. 50).

**MSCI, *Quality Indexes Methodology* (May 2025), sections 2.3-2.4 and Appendix IV (pp. 5-6, 14).**
https://www.msci.com/indexes/documents/methodology/2_MSCI_Quality_Indexes_Methodology_20250520.pdf
A factor index that selects *securities*: "If multiple securities have the same Quality Score, then
the security having a higher weight in the Parent Index is given a higher rank" (a tie rule two lines
of one company scored on company fundamentals would need), and "constituent weights are capped at
issuer level". MSCI allows both lines to be selected and controls the double exposure by issuer.

**S&P DJI, *Equity Indices Policies & Practices Methodology* (August 2026), "Multiple Share Classes"
and "Designated Listings" (pp. 17-18).** Read from the Internet Archive's copy of
https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-equity-indices-policies-practices.pdf
(spglobal.com refuses automated requests).

- Cap-weighted indices include every listed line separately, but "Index membership eligibility for a
  company with multiple share class lines is based on the total market capitalization of the
  company", and each line is weighted by its own float.
- "S&P DJI determines the share class with both the highest one-year trading liquidity ... and largest
  float-adjusted market capitalization as the Designated Listing", changed only when the other line
  leads on both by more than 20%.
- Index methodologies then pick one of three approaches: (A) all listed lines eligible; **(B) "Each
  company is represented once by the Designated Listing"**; (C) once, by the highest-yielding line.
- Berkshire is an exception: its share count is consolidated "under the B share class line".
- S&P DJI's January 2015 announcement introduced line-by-line treatment (Alphabet, then Google, already
  had two lines; Fox and News Corp gained theirs then):
  https://press.spglobal.com/2015-01-21-S-P-U-S-Indices-Methodology-Update

**S&P DJI, *S&P U.S. Style Indices Methodology* (June 2022), pp. 5-6** (read from a mirror,
https://cbonds.ru/download/indexdocs/indexdescrs/1306/). Lines are eligible under approach A, but the
scoring section is headed "Evaluating Growth and Value at the Company Level": "Raw values for each of
the above factors are calculated for each company", then standardised over the set. The book-to-price,
earnings-to-price and sales-to-price factors are the same family as the screener's Valuation.

**FTSE Russell, *Russell US Equity Indexes: Construction and Methodology* v7.2 (August 2026), rules
6.5.1-6.5.4 (p. 18).**
https://www.lseg.com/content/dam/ftse-russell/en_us/documents/ground-rules/russell-us-indexes-construction-and-methodology.pdf
Each class is reviewed separately for inclusion; for ranking, all classes are aggregated and
"multiplied by the primary exchange close price of the pricing vehicle"; the pricing vehicle is "the
share class with the highest two-year trading volume"; and **"Growth, value, defensive and dynamic
probabilities will be based on that of the pricing vehicle and assigned consistently across all
additional share classes."** The FTSE Global Factor Index Series ground rules (v8.8, July 2026, rule
4.2.1) make every eligible line of a company eligible for the factor indices.

**What it implies, taken together.**

- *Fundamentals and forecasts are company facts.* No provider lets two lines of one company carry
  different earnings, book value or consensus. A screener that does is reporting its data source, not
  the company.
- *Prices, returns and liquidity are line facts.* Every provider keeps them per line, and MSCI's
  valuation ratios divide company figures by the line's price.
- *The unit of selection depends on the product.* Cap-weighted indices hold every line, weighted by
  float, so a company's weight is split, not doubled. A list that selects names - which is what the
  screener's ranking and the bot's top 25 are - either represents each company once (S&P approach B,
  by designated listing) or caps exposure by issuer (MSCI). FTSE Russell scores the pricing vehicle and
  copies the result.
- *Peer statistics are computed over companies.* S&P computes and standardises its style factors "for
  each company". The screener's sector percentiles count rows, so a two-line company counts twice in
  its peers' distribution.

Not read, and not relied on: the S&P U.S. Indices Methodology itself (403 and a failed archive copy),
S&P DJI's 2017 exclusion and 2023 reinstatement of multiple-class companies for the S&P Composite 1500
(about which companies may join, not how lines are scored), and the voting-premium literature (Zingales
1995, *QJE* 110(4)), which explains why class prices differ but not how a screen should score them.

## 5. The options, rescored on the 2026-10-19 run

Everyone else unchanged; baseline = the published ranking (501 of 501 reproduced). "Analyst fields" =
forward EPS growth, FY1 revision (re-divided by the receiving line's own price, as the engine does),
surprise, beat streak and target upside.

| Option | FOX | FOXA | GOOG | GOOGL | NWS | NWSA | Others moved | Top 25 |
|---|---|---|---|---|---|---|---|---|
| As published | 31 | 47 | 133 | 226 | 98 | 61 | - | - |
| **FILL** - a line takes its sibling's analyst fields only where its own are missing | **6** | 49 | 135 | 224 | 112 | 61 | 86, max 5 | **FOX in, WDC out** |
| **CONSENSUS** - both lines take the designated line's analyst fields | 26 | 46 | 188 | 227 | 101 | 59 | 146, max 17 | unchanged |
| CONSENSUS, Alphabet on GOOG's fields instead | 26 | 49 | 123 | 163 | 110 | 61 | 113, max 12 | unchanged |
| **ONCE** - the other line takes every metric of the designated line | 43 | 43 | 207 | 207 | 69 | 69 | 206, max 15 | unchanged |
| **ONCE + DROP** - the company is ranked once, on the designated line | - | **28** | - | **197** | - | **52** | 381, max 59 | **unchanged** |

Reading it:

- **FILL is ruled out.** It keeps each line's own artefacts and adds the sibling's data: FOX keeps the
  +49.6% fallback and gains FOXA's revisions record, and jumps into the top 25. Any rule must replace a
  line's analyst record wholesale, never patch it field by field.
- **CONSENSUS fixes the data artefact and nothing else.** Two lines of one company still rank 20 places
  apart on price and returns (FOX 26, FOXA 46), both still count in the sector percentiles, and the bot
  can still buy both.
- **ONCE ties the two rows** (both FOXA's figures, both #43), which is honest but leaves a tie in a
  ranked list and both rows in the peer group, and the bot would buy both at the same rank.
- **ONCE + DROP is S&P DJI's approach B.** Every move outside the six lines is inside Communication
  Services (14 stocks move 10+ places, e.g. TKO 169 -> 112, LYV 391 -> 332), because the six rows were
  27% of a 22-row peer group: the second lines held the other 19 stocks' percentiles down by **1.56
  composite points on average**. That is a correction, not noise - each duplicated company was being
  counted twice in its peers' distributions. The rank moves are large because the middle of the
  ranking is dense (about 0.05 composite points per place between ranks 100 and 200).

## 6. What it means for the paper-trading bot

`screener-bot` (`bot/strategy.py`, commit `d06ff82`) buys the 25 best-ranked stocks, sells a holding
once its rank is worse than 40, caps a sector at 6 names, and has **no notion of a company**: two lines
of one company are two names to it, would count twice against the sector cap, and would double the
position in one issuer - exactly the concentration MSCI caps by issuer.

- **Today:** it holds none of the six lines (`journal/sim/positions.json`, 2026-10-19), and no option
  above changes the top 25 except FILL.
- **History:** across 59 snapshots (2026-02-20 to 2026-10-19), FOX and FOXA were **both in the top 25
  on 4 runs** (2026-02-20 to 02-23) and both inside the hold band with one in the top 25 on 8. Alphabet's
  and News Corp's lines never were (median rank gaps 31 and 39). The bot began trading on 2026-10-12, so
  it has not happened to it - but it has happened to the ranking it trades.
- **Fix location:** scoring once, in this repo, means the published `table_data` never contains two
  lines of one company, so the bot needs no change and its payload contract
  (`tests/test_bot_payload_contract.py`) is untouched. A company-aware rule in the bot would be a
  second line of defence, for its own sessions to weigh.

## 7. Found, not proposed: the source's EPS history mixes accounting bases

The MSCI construction (`factor_engine.py`, forward EPS growth) takes EPS12B as "the last four reported
quarters' actual EPS on the same (consensus) basis". Alphabet's two lines show the source does not
guarantee that:

| Quarter | Filed diluted EPS (10-Q) | GOOGL "actual" | GOOG "actual" | Estimate (GOOGL / GOOG) |
|---|---|---|---|---|
| 2025 Q3 | 2.87 | 2.87 | 2.87 | 2.26 / 2.26 |
| 2025 Q4 | 2.82 | 2.82 | 2.82 | 2.64 / 2.64 |
| 2026 Q1 | **5.11** | **5.11** | 2.76 | 2.67 / 2.63 |
| 2026 Q2 | **9.11** | **9.11** | 2.85 | 2.90 / 2.91 |

GOOGL's actuals are the GAAP figures, which include very large gains on equity investments; GOOG's 2026
actuals are on the analysts' basis. So GOOGL's trailing base is $19.91 against a forward figure on the
analysts' basis ($16.18), giving -18.7%, and its surprise score reads +92% and +214% for two quarters
that, on the analysts' basis, were in line. **Amazon has the same signature on a single line** (filed
2.78 and 5.75 against estimates of 1.65 and 1.83; surprises +68% and +215%), and 67 stocks have at
least one quarter with a surprise above 50%, most of them genuine. So this is not a share-class
problem and scoring Alphabet on GOOG would only hide it for one company. It needs its own note:
whether EPS12B should be checked against the filed EPS, or MSCI's own EPS12B (last fiscal year's EPS
blended with FY1 consensus, *Fundamental Data Methodology* p. 17) used instead. It bears on the
designated-listing rule below only in that **the rule picks which security represents the company; it
does not certify that line's data.**

## 8. How it fits the rest of the screener

- **What the screener is for.** It ranks companies for a club deciding what to look at and buy. Two
  rows for one company say the tool thinks Fox is two investment ideas; it is one idea with two
  tickets. A student who sees FOX at #31 and FOXA at #47 learns something false about Fox.
- **Valuation is not lost.** Because every valuation ratio differs between lines only through the
  price (section 3: earnings yield by exactly the price ratio, the EV ratios by exactly the gap in
  market value), the second line's valuation is fully stated by one sentence on the page: "FOX
  trades 10.8% below FOXA for the same claim on Fox's earnings." That is a fact about the two tickets,
  not a factor signal: the value premium is a cross-company finding. FTSE Russell copies the pricing
  vehicle's style to every class and S&P DJI scores style per company, so neither scores a class
  discount as value; MSCI's security-level ratios do differ by class price, and MSCI bounds the result
  by capping each issuer.
- **Momentum and risk are not lost either.** The paired lines' 12-month and 6-month returns differ by
  under 5 points; the designated line is the more liquid one, which is the line a club would trade.
- **Nothing else in the screener reads the second line.** It carries no statement data of its own; the
  context layer (options, insider trades) is per line and display-only; and the payload, holdings
  panel and search can name the second line as "scored once, as Fox Corporation (FOXA)" - the page
  already answers for a ticker that is not ranked (the 2026-10-19 club-readiness change).
- **Universe counts.** Scored rows fall from 501 to 498; the backtest's
  `universe_history.validate_membership` range (495-515) is about index membership and is not affected.

## 9. What to decide

**(a) The company's figure.** One analyst record per company, never field-by-field (FILL's #6 is the
argument). Under approach B this is automatic: the company is scored on its designated line's record.

**(b) Which line.** The designated listing - the higher average daily dollar volume in the run's own
fetch (S&P DJI's liquidity criterion, FTSE Russell's pricing vehicle). On this run it is also the line
with more analysts (54 vs 14; 18 vs none; 8 vs none), so the audit's "more analysts" rule and the
providers' rule agree. Today's volume ratios are 1.5x-4.8x, so S&P's 20% switching margin would not
bind; the run should note any pair that comes within 20%.

**(c) What the reader and the bot see.** The second line is not ranked. The payload carries it as an
unranked entry pointing to its company's ranked line, with its own price and the price gap; the bot
never sees it, so it cannot hold one company twice.

**What would make a reviewer refuse it, or change it:**

- A preference for MSCI's route (rank every line, cap by issuer) - but the bot has no issuer cap and
  the ranking is read as a list of ideas; that route needs a bot change and a page change and leaves
  the peer-group double count.
- Evidence that the designated line's analyst record is systematically worse than the other's. On this
  run the only such case is section 7's, which is a source-wide basis problem, not a class one.
- A view that a club should be able to rank the cheaper class above the dearer one. That would make the
  class discount a factor signal. MSCI's security ratios do this implicitly (section 8); S&P DJI's and
  FTSE Russell's style scores do not, and the value literature offers no evidence that a class
  discount earns the value premium.

**Before it can ship** (CLAUDE.md rows 0.8, 0.8c, 1.5 and the bot contract):

- `run_screener.py`, after `compute_metrics` and before `compute_sector_percentiles`: group rows by SEC
  CIK (`data/sec/company_tickers.json`, already cached weekly; fall back to identical `longName`), keep
  the line with the higher `avg_daily_dollar_volume`, and move the others to an unranked list. Log the
  grouping, and any pair within 20% on volume, in `run.log` and the health note.
- `generate_dashboard.py`: the unranked lines in the payload as `{ticker, company_line, price,
  price_gap}`; search and My Holdings resolve them to the company's line with one sentence; a claim in
  `claims.py` for that sentence and its test.
- `run_screener.build_screener_overview`: one methodology sentence ("A company with two listed share
  classes is scored once, on its more-traded class, as S&P DJI and FTSE Russell do"); regenerate.
- Tests: no two ranked rows share a CIK; the second lines are absent from `table_data` and from the
  sector peer groups; `tests/test_bot_payload_contract.py` unchanged and passing.
- A `METHODOLOGY_CHANGELOG.md` entry citing section 4 and the stakes in section 5.

## 10. Reproduce

`python research/measurements/2026-10-19-share-classes.py` reads the newest run under `runs/`,
`data/sec/company_tickers.json`, `data/sec/pit/facts.parquet` and `improvement/snapshots/`, and
writes `2026-10-19-share-classes.json`: the inventory (A), every weighted metric of each pair with its
input class and the price checks (B), the six options' ranks, peer-group effect and top-25 lists (C),
and the snapshot history (D). The bot's code was read at commit `d06ff82`.

**Proposal status: ACCEPTED 2026-10-21** by the 06:00 session (not this note's author), **with one
change the review measured and the note had not.** The proposal re-picked the designated line from
**each run's own** `avg_daily_dollar_volume`. That is not S&P DJI's rule - S&P changes a Designated
Listing only when the other line leads by more than 20% - and it is not stable: over 89 month-ends
(2019-06 to 2026-10, 63-day average daily dollar volume) **Alphabet's two classes sat inside that
margin in 31 of them (34.8%)**, ratio median 1.276 and minimum 1.003, and the higher-volume line
actually **changed four times** (2020-09, 2020-11, 2020-12, 2021-01). Fox and News Corp never came
close (2.077 and 2.193 at their closest). Rescored on the 2026-10-21 run, **Alphabet ranks 179 on
GOOGL and 101 on GOOG** - 78 places, and a different ticker in the ranking - because of the
accounting-basis defect in section 7. So the choice of line is now **declared** in `config.yaml`
(`share_classes.designated`), changed only with a changelog entry, and a run *reports* a crossing
of the margin instead of acting on it. Everything else shipped as proposed: ONCE + DROP, before the
sector percentiles, with the second lines published as unranked entries that name the company's
ranked line and the price gap. Independently re-measured on the 2026-10-21 run: CIK coverage **503
of 503** with the `longName` fallback giving identical groups, **top 25 unchanged**, FOXA 61 -> 35 /
GOOGL 199 -> 179 / NWSA 65 -> 54, the other 16 Communication Services stocks **+1.565** composite
points, and **0.0** change outside that sector. Sections 1-9 stand as written; the measurements are
in `research/measurements/2026-10-21-share-class-stability.py` and the record is
`METHODOLOGY_CHANGELOG.md` 2026-10-21. **Section 7 is still open** and needs its own note.

The proposal as it was put, for the record - score each company once: in `run_screener.py`, after `compute_metrics` and before `compute_sector_percentiles`, group universe rows by SEC CIK (falling back to identical `longName`), keep the line with the higher `avg_daily_dollar_volume` (S&P DJI's Designated Listing / FTSE Russell's pricing vehicle; on 2026-10-19 GOOGL, FOXA, NWSA) and remove the other lines from percentiles, scoring and ranking; publish them as unranked entries pointing to the company's ranked line with their own price and the price gap; shipped with a no-two-rows-per-CIK test, the page sentence and its claim, the generated methodology sentence and a changelog entry (section 9). Measured on the 2026-10-19 run: top 25 unchanged; FOXA 47 -> 28, GOOGL 226 -> 197, NWSA 61 -> 52; other Communication Services stocks +1.56 composite points on average
