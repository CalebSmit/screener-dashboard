# The quarter the source leaves empty is in its own trailing series - 2026-10-29

Owner-run session, evening. A finding with a measurement and a proposal; **no code changed**.
Measurement: `research/measurements/2026-10-29-ttm-carries-the-missing-quarter.py` (read-only,
about 45 requests, re-runnable).

**Proposal status: OPEN** - fill an empty newest quarter from the source's own trailing series (9 of 15 lagging stocks, Amazon among them); answer section 4 first.

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
