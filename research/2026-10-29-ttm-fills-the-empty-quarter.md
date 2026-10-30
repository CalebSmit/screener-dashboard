# The quarter the source leaves empty is in its own trailing series - 2026-10-29

Owner-run session, evening. A finding with a measurement and a proposal; **no code changed**.
Measurement: `research/measurements/2026-10-29-ttm-carries-the-missing-quarter.py` (read-only,
about 45 requests, re-runnable).

**Proposal status: DECIDED 2026-10-30 (13:00 session) - REFUSED as written.** The substitution
itself is sound and the opportunity is larger than this note measured (13 of 15, not 9), but the
rule is specified against the wrong reference period and the prior-year half cannot be built from
this source at all. Section 5 has the review, the two measurements behind it, and the narrower
rule with its criteria written down.

In full: when a stock's newest quarterly income-statement column is empty, score its
income-statement inputs on the source's own trailing-twelve-month series where that series reaches
the newer quarter.

## 1. What is wrong

Every data run's health check prints "N stocks scored a period behind a filing over 30 days old".
On 2026-10-29 that was **15**, including Amazon and Bank of America. The cause was found on
2026-10-15 (`research/measurements/2026-10-15-statement-lag.py`, `sec_crosscheck.newer_filings`):
for these companies the source lists the newer quarter as a column with **4 or 5 cells filled and
the rest empty**, and the engine (correctly) never reads an empty column. So they are scored on the
quarter before. The page says so (`drilldown` "A period behind the filings", claim
`drilldown.filing_check`), and nothing more has been done about it.

It is not a short lag. Amazon filed its June quarter on 2026-07-31; on 2026-10-29 - **90 days
later** - the source's June column still has 4 of ~46 income-statement cells. It is still scored on
the four quarters to **March 2026**.

## 2. What the measurement found

For each of the 15, the newest filed period was compared with what three other places in the same
source hold: the trailing-twelve-month income series (`get_income_stmt(freq="trailing")`), the
quarterly balance sheet and the quarterly cash-flow statement. Where the trailing series reaches
the newer quarter, the quarter it implies (trailing net income less the three quarters before it)
was compared with the company's own 10-Q figure in the SEC companyfacts cache.

| | Count |
|---|---|
| Trailing series reaches the filed quarter | **9 of 15** |
| Quarterly balance sheet has it | 7 of 15 |
| Quarterly cash-flow statement has it | 2 of 15 |
| Implied quarter computable and an SEC figure to check against | 8 |
| ...agreeing with the 10-Q to the printed $0.01 bn | **6 of 8** (AEE and ES differ by $0.01 bn) |

| Ticker | Scored on | Filed quarter | Trailing has it | Implied quarter net income | 10-Q |
|---|---|---|---|---|---|
| AMZN | 2026-03-31 | 2026-06-30 | yes | **$62.65 bn** | $62.65 bn |
| APH | 2026-03-31 | 2026-06-30 | yes | $1.77 bn | $1.77 bn |
| CHD | 2026-03-31 | 2026-06-30 | yes | $0.20 bn | $0.20 bn |
| CHRW | 2026-03-31 | 2026-06-30 | yes | $0.19 bn | $0.19 bn |
| CPT | 2026-03-31 | 2026-06-30 | yes | $0.02 bn | $0.02 bn |
| DD | 2025-12-31 | 2026-06-30 | yes | $0.14 bn | $0.14 bn |
| AEE | 2026-03-31 | 2026-06-30 | yes | $0.31 bn | $0.32 bn |
| ES | 2026-03-31 | 2026-06-30 | yes | $0.05 bn | $0.06 bn |
| FDXF | 2025-11-30 | 2026-05-31 | yes | - | - |
| BAC, BNY, TFC, EXR, FLEX, ILMN | 2026-03-31 | 2026-06 | **no** | - | $9.07 / 1.76 / 1.55 / 0.26 / 0.28 / 0.21 bn |

Three banks (BAC, BNY, TFC) are among the six the trailing series does not reach, so this would not
close the gap for banks.

**DD is a different case, and the engine handles it.** Its quarterly statements are full, but the
source's December 2025 quarter shows revenue of **-$1.87 bn** (negative revenue, after the Qnity
spin-off), and the engine fell back to annual statements (`data_source: annual`). That is the
defensive behaviour working; it is why DD is "a period behind", and filling from the trailing
series would need the same sanity checks the quarterly path has.

## 3. How much it moves - Amazon

Amazon's scored net income is the four quarters to March, **$90.8 bn**. The four to June, per the
trailing series and the 10-Q, are **$135.3 bn** - 49% more, because the June quarter alone earned
$62.65 bn. Its earnings yield would go from **3.31% (22nd sector percentile)** to about **4.9%**.
Amazon ranks 419th of 498 today; this is one of its weighted inputs, and revenue, operating income
and every margin built from them move with it.

**The $62.65 bn very likely carries a one-off gain** (it is more than double any other recent
quarter). That does not argue against using it: the screener scores earnings **as reported**, a
decision settled on 2026-10-16 after the normalized-income proposal was refused
(`research/2026-10-15-earnings-definition.md` section 8.6), and the earnings-yield row already says
when a one-off is inside the figure (`drilldown.special_item_note`). A quarter left out is not a
neutral choice either - it is a quarter of as-reported earnings the reader's filing shows and the
score does not.

## 4. What has to be answered before building it

1. **Mixed periods.** Income would run to June while the balance sheet (7 of 15 have it) and cash
   flow (2 of 15) may stop at March. The engine already lets the three statements carry different
   dates (`stock_detail[t].asof` has one per statement; DD's is `bs 2026-06-30, is 2025-12-31`), so
   this is not new - but every ratio that divides an income figure by a balance-sheet figure (ROE,
   ROIC, asset turnover) should be checked for whether a one-quarter mismatch is worse than a
   one-quarter-stale numerator. Whether index and data vendors use each item's latest filed value
   or hold a company to one common period is the practitioner question to answer, with a quoted
   source, before building - this note has not checked it.
2. **Row names.** The engine reads `"Total Revenue"`, `"Operating Income"` and `"Net Income"` from
   the quarterly table; the trailing series names them `TotalRevenue`, `OperatingIncome`,
   `NetIncome`. Confirm they are the same rows - Operating Income especially, because the
   2026-10-15 decision keeps the source's "Operating Income" (pre-special-items) row on purpose.
3. **Which inputs.** Only the flow items the engine sums over four quarters (`_stmt_val_ltm`):
   revenue, operating income, net income, and anything else read that way. The prior-year
   comparisons (`*_prior`) would need the trailing series one year earlier, which the same call
   returns for some stocks and not others.
4. **Cost.** One extra request per affected stock, only when its newest quarterly column is empty -
   about 15-30 a run. The fetch is the slowest stage (646 s on 2026-10-29, 45 tickers
   rate-limited), so the request belongs in the retry or context pass, not the core fetch.
5. **Validation.** `sec_crosscheck` already compares the scored inputs with the filings every
   morning; after the change the 15 should leave the "period behind" count and their net income
   should agree with the filing, which is a measurement the health check would print the next
   morning.

This is a change to what the score reads, so it follows the route in CLAUDE.md: this note, then a
`METHODOLOGY_CHANGELOG.md` entry, one change at a time, with the scoring change and its
`metric_lineage` lines in the same commit (row 0.8c). It is **not** a change to a weight, and it
needs no backtest.

---

## 5. Review - 2026-10-30, 13:00 session. REFUSED as written

Reviewed by a session that did not write the note, as `OWNER_FOCUS.md` item 1 requires. Nothing
changed: no weight, metric, threshold, formula or published sentence. Measurements:
`research/measurements/2026-10-30-trailing-series-review.py` (+ `.json`) and
`research/measurements/2026-10-30-trailing-column-moves.py` (+ `.json`), read-only, about 75
requests between them, re-runnable.

Section 4 nominates five things to answer before building. Three are now measured. **Two come out
in the proposal's favour and decisively so; the third kills the rule as written.**

### 5.1 Item 2, row names: the trailing series is the same rows under different names

The note asked whether `TotalRevenue`, `OperatingIncome` and `NetIncome` in the trailing series
are the rows the engine reads from the quarterly table, flagging operating income in particular
because the 2026-10-15 decision keeps the source's pre-special-items `Operating Income` row on
purpose (`DECISIONS.md`, "0.14, the three input definitions settled against the filings").

Measured on the twelve highest-ranked stocks that are **not** flagged - so their quarterly
statement is the one `_stmt_val_ltm` actually sums - for all nine income-statement labels the
engine reads, the trailing series' value against the engine's own four-quarter sum:

| Label | n | median \|rel diff\| | max \|rel diff\| | within 0.5% |
|---|---|---|---|---|
| Total Revenue | 12 | 0.000000 | 0.000669 | 12/12 |
| Gross Profit | 11 | 0.000000 | 0.000212 | 11/11 |
| Operating Income | 12 | 0.000000 | 0.000614 | 12/12 |
| EBITDA | 11 | 0.000000 | 0.000000 | 11/11 |
| Net Income | 12 | 0.000000 | 0.000523 | 12/12 |
| Tax Provision | 12 | 0.000000 | 0.001458 | 12/12 |
| Pretax Income | 12 | 0.000000 | 0.000344 | 12/12 |
| Cost Of Revenue | 11 | 0.000000 | 0.000121 | 11/11 |
| Interest Expense | 11 | 0.000000 | 0.006803 | 10/11 |

Nine of twelve stocks agree **exactly** on every label; the three that do not (EG, ED, APA) differ
by at most 0.07% except for one figure, EG's interest expense at 0.68%. So substituting the
trailing series does **not** change a metric's definition, and in particular it preserves the
operating-income decision. That was the largest risk in the proposal and it is measured away.

**Coverage is also fine.** Of the eight flagged stocks whose trailing column sits on the newest
filed period, **seven carry all nine labels**; CHRW is missing Interest Expense alone. The
within-stock label mixture this review expected to find is one metric on one stock.

### 5.2 Item 1, mixed periods: practice pairs each item at its own latest period

The note asked, with a quoted source, whether index and data vendors hold a company to one common
period or use each item's latest filed value. **They use the latest value, explicitly, and for
exactly the ratio the note worried about.**

- **MSCI**, *Fundamental Data Methodology*, June 2024, §1.2.2 (p. 14): earnings per share is used
  for, among others, "**Return on Equity (ROE), which is calculated using the trailing 12-month
  earnings per share figure and latest book value per share**"; and §1.2.5 (p. 15) defines book
  value as "shareholders' equity available to shareholders **at the latest period end date**".
  https://www.msci.com/indexes/documents/methodology/0_MSCI_Fundamental_Data_Methodology_20240625.pdf
- **S&P Dow Jones Indices**, *S&P Quality Indices Methodology*, Appendix A, "Fundamental Ratios
  Calculation" (p. 16): "**Return on Equity (ROE). This is calculated as a company's trailing
  12-month earnings per share divided by its latest book value per share**", and the financial
  leverage ratio uses "a company's **latest total debt** divided by its book value".
  https://www.spice-indices.com/idpfiles/spice-assets/resources/public/documents/methodology-sp-quality-indices.pdf
  (S&P's own `spglobal.com/spdji` copy answers 403 to a scripted request; read from S&P's
  `spice-indices.com` copy, and `data/papers/invesco-sp-quality-indices-methodology.pdf` from an
  earlier session is a third copy of the same document.)

Two independent providers, same construction: a trailing-twelve-month income numerator over a
*latest*-period balance-sheet denominator. **A one-quarter mismatch between a June income
statement and a March balance sheet is the professional norm, not a defect** - so item 1's worry
about ROE, ROIC and asset turnover does not stand, and the engine already does this anyway
(FedEx is published today with `cf 2026-05-31` against `is 2025-11-30`, two quarters apart).

Two things read the other way and belong on the record:

- MSCI **normalizes** earnings - §1.2.2 excludes "extraordinary items or non-recurring items" and
  lists impairments, disposal gains, restructuring and litigation among them. This screener scores
  earnings **as reported** by its own settled decision (`research/2026-10-15-earnings-definition.md`
  §8.6, refused 2026-10-16). So practice supports the *period mixing* and does **not** support
  section 3's Amazon figure, whose $62.65 bn quarter the note itself calls "very likely a one-off
  gain" - MSCI would strip it. That is not an argument against the fill; it is an argument against
  quoting Amazon's 49% as the benefit.
- MSCI §1.1.5, *Sources and Updates*: MSCI reflects public information "**provided that the
  information is complete**", by "updating company information **only upon availability of complete
  information**, such as official filings with stock exchanges or regulators **or upon dual vendor
  validation**." That is the professional version of what the engine does today when it refuses a
  sparse column, and it is the source of the guard in §5.5 - *corroborate before you score*.

### 5.3 Item 3, the prior year: impossible from this source, and three weighted metrics break

The note guessed that the same call "returns [the year earlier] for some stocks and not others".
Measured: **the trailing series returns exactly one period end, for 15 of 15 flagged stocks and
26 of 26 tested.** For none.

That is not a gap in the proposal, it is a defect the proposal would ship.
`_stmt_val_ltm_year_ago` anchors the prior-year window on the quarterly statement's newest
**populated** period, which would still be March. Fill only the numerator and
`revenue_growth` - a weighted metric - plus the gross-profit and net-income prior-year
comparisons become **fifteen-month** growth rates for precisely the stocks being fixed. Trading a
one-quarter-stale input for a wrong one is not an improvement.

It is fixable: only the newest column is sparse, so the quarterly table can still supply the
twelve months to June **2025** by date. But the prior-year anchor has to move with the fill, and
the note does not say so.

### 5.4 The rule is specified against the wrong reference, and its own table is one morning's data

The proposal's rule is "fill where the trailing series reaches the newer quarter", with the newer
quarter taken from the newest filing in the weekly SEC companyfacts cache. Those two things update
on different clocks, and the three period ends that actually matter are not the two the rule
compares. Measured 2026-10-30, 24 hours after the note, on the fifteen stocks its table names
(`gap` = quarters between what the stock **is scored on** and the trailing column):

| gap | n | stocks |
|---|---|---|
| 0 quarters | 2 | DD, TFC |
| 1 quarter | 6 | AEE, APH, CHD, CHRW, CPT, ES |
| 2 quarters | 7 | AMZN, BAC, BNY, EXR, FDXF, FLEX, ILMN |

Three consequences, in order of how much they matter.

1. **The opportunity is bigger than the note claimed: 13 of 15, not 9.** The trailing column is
   newer than the column the engine scores for thirteen of the fifteen. The note's 9-of-15 /
   6-of-15 split came from comparing the trailing column against the newest *SEC filing* instead
   of against what the engine actually reads, and the six "no" rows are not stocks the trailing
   series cannot help - they are stocks where it had already moved **past** the period being
   compared.
2. **The note's section-2 table is a snapshot of one morning.** Amazon is recorded there as a
   match; on 2026-10-30 its trailing column is **2026-09-30** while the SEC cache still holds June,
   so under the rule as written today's build would fill **nothing for Amazon** - the case the
   proposal is written around. TFC, listed as a "no", is not flagged at all today: it is published
   scored to 2026-09-30, resolved by itself in a day. This is the class of defect this project's
   own retrospective named *this morning* - a premise that is a property of one morning's data
   (`prompts/nightly.md` section 2, process change 2 of `NIGHTLY_LOG.md` 2026-10-30).
3. **Half of the thirteen cannot be checked at all.** At a one-quarter gap the implied newest
   quarter is computable from the quarterly table - TTM less the three populated quarters before
   it - which is exactly how section 2 checked eight of them against the companies' own 10-Qs and
   found six exact. At a two-quarter gap there is nothing to check it against: for Amazon,
   `TTM(Sep) - (Jun + Mar + Dec)` is not computable *because June is the empty column*. FedEx is
   the extreme case - its trailing column is 2026-05-31 against a newest quarterly column of
   2025-11-30, two quarters past anything the statement carries, and section 2's own table could
   only print "-" for it.

So the uncheckable half is not a tail case; **it contains every stock whose fill would move a
score materially**, Amazon and Bank of America included.

### 5.5 What to build instead, with the criteria written down first

Refusing the rule is not refusing the finding. The narrower version, pre-registered here so that
whoever builds it cannot widen it to taste:

- **Reference period.** Compare the trailing column against the newest quarterly column that
  **carries the label being filled**, never against the SEC cache or the newest filing. The SEC
  cache may not gate a scoring input at all (CLAUDE.md 0.14: a test fails if `sec_crosscheck`
  appears in a scoring module), so the corroboration has to come from the source's own data.
- **Gap of exactly one quarter.** Fill only when the trailing column is one quarter past that
  reference, so the implied newest quarter is `TTM - (three populated quarters)` and can be
  checked. A gap of two or more is refused and the stock stays where it is. Measured today this
  admits **6 of 14** flagged stocks (AEE, APH, CHD, CHRW, CPT, ES - ranks 52, 85, 116, 139, 204,
  352) and refuses 7. It also happens to coincide, 6 for 6, with the stocks whose filing the
  weekly cache actually holds, which is why §5.2's completeness rule and this guard point the
  same way.
- **The implied quarter must survive the checks the quarterly path already applies**, including
  `quarterly_statement_defect`'s negative-revenue signature - DuPont is in this list precisely
  because that guard fired (§2), and a fill that bypasses it re-opens a defect already found.
- **The prior-year anchor moves with the fill or the fill does not happen.** `revenue_growth` and
  the two other `*_prior` comparisons must be re-anchored on the filled period from the quarterly
  table; if the twelve months one year before the filled period is not available from four
  adjacent populated quarters, that stock is not filled.
- **One change at a time** (CLAUDE.md route): the scoring change, its `metric_lineage` `LINEAGE` /
  `RECOMPUTE` / `EQUATIONS` lines and a `claims.py` registration for any sentence saying a figure
  came from the trailing series, in the same commit (row 0.8c), with a `METHODOLOGY_CHANGELOG.md`
  entry.
- **The honest statement of the benefit**: six of 498 stocks, one quarter fresher, no top-25 name,
  and the two cases with the largest effect excluded by the guard. Not "Amazon's earnings yield
  goes 3.31% to 4.9%".

**Whether that is worth building at all is a real question**, and this review deliberately does
not answer it by building it. The reconciliation step is the part with lasting value - it is the
only thing that distinguishes a fresher input from an unverifiable one, and it is also what makes
the change small.

### 5.6 Three things settled so nobody re-measures them

- **The trailing series carries one period and only one.** Any design that needs two periods from
  it - a growth rate, a prior-year comparison, a history - is impossible without a second source.
  15 of 15, 26 of 26.
- **Having a column is not being scored on it.** Section 2 reports the quarterly balance sheet
  reaching the filed quarter for 7 of 15; all six one-quarter stocks are nonetheless published
  today with `asof.bs = 2026-03-31` (checked in the payload: AEE, APH, CHD, CHRW, CPT, ES). A
  column the source lists and a column the engine can read are different facts, which is the whole
  shape of this problem one level up - and it means the fill really would create the one-quarter
  income/balance-sheet mismatch §5.2 licenses, for all six.
- **Mixed statement periods are already published and are correct practice.** FedEx carries
  `cf 2026-05-31` against `is 2025-11-30`; DuPont `bs 2026-06-30` against `is 2025-12-31`. §5.2's
  two sources say to pair the latest of each. Do not "fix" this by holding a company to one
  period.
