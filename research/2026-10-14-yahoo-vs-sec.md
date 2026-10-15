# Do the numbers the screener scores agree with the companies' own filings? (2026-10-14, owner-run)

**Question.** Every fundamental the screener scores comes from Yahoo Finance through `yfinance` -
a scraper of a website, which this week silently stopped serving one field (`dividendRate`, 402 of
501 stocks to 6; changelog 2026-10-14). A college investment club, or anyone else, is entitled to
ask how often the inputs are simply right. This note answers it for the main inputs by rebuilding
each one from the company's own XBRL filings and comparing. No returns are read (rule 4).

**Finding.** **The inputs are overwhelmingly right where they are comparable**: total assets agree
with the filing to within 0.5% for 99.8% of companies, operating cash flow 98.8%, equity 97.9%,
net income 94.3%, revenue 88.5% (92.7% within 2%). Nearly every larger disagreement examined has a
definitional cause - bank and broker revenue, profit attributable to minority interests, revenue
net of excise taxes - not a wrong number. **The one input that is materially different by
construction is capital expenditure**, and it feeds free-cash-flow yield, the heaviest metric in
the composite (45% of Valuation, 9.9 points): Yahoo's "Capital Expenditure" adds purchases of
intangibles and, for companies with a finance arm, the vehicles and equipment bought to lease out.
For the published top 25 that changes nothing that matters (at most 0.5 composite points); for a
handful of companies it changes a great deal (GM's free-cash-flow yield 0.3% against 8.0%). That is
a methodology question, queued with a precise statement below. **Nothing was changed tonight.**

---

## 1. Method

`research/measurements/2026-10-14-yahoo-vs-sec.py`, on the run published 2026-10-14 02:00 and the
SEC `companyfacts` cache of 2026-10-09 (`data/sec/pit/facts.parquet`, 502 companies, refreshed
weekly by the run - `sec_fundamentals.refresh_companyfacts`).

- **The Yahoo side** is exactly what the run scored with: `stock_detail[t].inp` in the published
  payload, with `asof` giving the quarter each statement ends.
- **The SEC side** is rebuilt for that same quarter from the company's own tags: a trailing-twelve-
  month flow is the fiscal year if the quarter ends one, otherwise year-to-date + prior fiscal year
  - prior year-to-date (how a 10-Q's cumulative columns combine); a balance is the instant value.
- **Each figure is taken as first filed.** The first version took the latest filing and was wrong
  for FIX: its Q1 2026 10-Q tags the quarter's revenue ($1.83B) with fiscal 2025's dates, which made
  the rebuilt trailing revenue $4.0B against Yahoo's (correct) $11.2B.
- **Revenue takes the largest revenue concept a company reports at that date**, because the others
  are components (Camden's `RevenueFromContractWithCustomer...` is its $12.6M of non-rental income).
- **Not compared** (about 15-20% of stocks per field, more for gross profit): the quarter is newer
  than the weekly cache, a prior-year column is missing, or the company uses a tag outside the
  cached set. These are gaps in the check, not findings about the data.

## 2. Results

| Input | Compared | Within 0.5% | Within 2% | Within 10% | Over 10% |
|---|---|---|---|---|---|
| Total assets | 426 | 99.8% | 99.8% | 100.0% | 0 |
| Operating cash flow | 411 | 98.8% | 99.3% | 99.8% | 1 |
| Equity | 426 | 97.9% | 98.6% | 99.5% | 2 |
| Net income | 421 | 94.3% | 95.7% | 98.3% | 7 |
| Gross profit | 156 | 94.2% | 95.5% | 98.7% | 2 |
| Revenue | 410 | 88.5% | 92.7% | 96.8% | 13 |
| Capital expenditure | 343 | 79.9% | 83.4% | 86.6% | 46 |

**What the disagreements over 10% are** (excluding capex; from the filings' own tags, not a
separate reading of each 10-Q):

- **Financials' revenue (MTB, IBKR, BX, RJF; NDAQ's and CBOE's gross profit).** Banks, brokers and
  exchanges report several "revenue" lines - gross of interest expense, net of it, or net of
  routing fees - and Yahoo's choice differs from the tag. The screener does not use revenue for
  banks' valuation (they score on P/B), but revenue growth applies to them.
- **Minority interests (IBKR, FCX, CF, SPG, IVZ net income; AT&T equity).** The filing's
  `ProfitLoss` includes profit belonging to minority holders; Yahoo reports what belongs to common
  shareholders, which is the right figure for per-share valuation. Yahoo is right; the rebuild
  falls back to the wider tag where the narrower one is missing.
- **Excise taxes (MO, BF-B).** Revenue net of excise taxes (Yahoo) against gross (the tag).
- **Three-quarter annualisation (BRK-B revenue).** The screener itself annualises three quarters by
  4/3 when Yahoo is missing one (`_stmt_val_ltm`, `partial_labels`) - the $428,529,333,333.33 gives
  it away. Documented behaviour, flagged as approximate in the run.
- **Not yet explained:** RSG, HAS, APTV, WMB and DD revenue (11-21%), VMRK and BRK-B net income,
  VMRK equity, COIN operating cash flow. Each needs a look at the filing itself; until then they
  are disagreements, not errors on either side.

**So: no evidence tonight of Yahoo serving a wrong number for any comparable input**, which is the
reassuring result - with the honest qualifier that the check covers the inputs that have a clean
XBRL counterpart, and the cache is up to a week behind.

## 3. Capital expenditure: one definition, large effects for a few companies

The screener's free cash flow is operating cash flow minus Yahoo's `Capital Expenditure` row
(`factor_engine.py` around line 1201), and `fcf_yield` = FCF / enterprise value. A probe of
Yahoo's quarterly cash-flow lines (2026-10-14) shows the row is a **sum**:

| Company | Yahoo capex | of which PP&E | of which the rest | The rest is |
|---|---|---|---|---|
| GM | $22.6B | $8.8B ("reported") | $13.8B | vehicles bought for GM Financial's leases |
| BSX | $2.7B | $0.9B | $1.8B | purchases of intangibles |
| DE | $4.0B | $1.2B (filed) | $2.8B | equipment on operating leases (John Deere Financial) |
| MA, NTRS, FICO | $1.46B, $0.75B, $41M | $0.7B, $84M, $6M | about half or more | capitalised software |

`research/measurements/2026-10-14-capex-definition.py` rebuilds FCF yield with capex = the PP&E
purchases each company files, for the same period, on 337 stocks with both:

- Yahoo's capex exceeds filed PP&E by more than 10% for **45** stocks, by more than a third for **25**.
- FCF yield moves more than 0.5 points for **17** stocks, more than 1 point for 10, more than 2 for 4.
- The sector percentile of FCF yield moves more than 10 points for **14** stocks and more than 25
  for 3: **GM 10th -> 90th** (yield 0.3% -> 8.0%), **DVN 23rd -> 69th**, **BSX 15th -> 59th**. At
  FCF yield's weight (0.099 of the composite) the largest composite move is 7.9 points.
- **Of the published top 25, 17 were compared; none would move by more than 0.5 composite points.**

**It cuts both ways, so neither source is "the right one".** For EXR (a REIT) the filed
`PaymentsToAcquireProductiveAssets` includes buying properties ($438M) where Yahoo's capex is $20M:
there the filed tag is the one inflating capex.

**The question this leaves, for a Monday:** what should capital expenditure in a free-cash-flow
yield include? Three candidate rules, each with a practice behind it that needs reading rather
than recalling: (a) PP&E only - the narrow reading of the cash-flow line; (b) PP&E plus capitalised
software - the economic reading, since companies that capitalise development would otherwise look
better than ones that expense it; (c) consolidated, including a finance arm's lease assets - which
matches the consolidated enterprise value in the denominator. The screener today is (c) with
intangible purchases on top, by accident of which Yahoo row it reads. Whatever the answer, it
should be a written choice with a source, not the row that happened to be there. Queued as
`plan/factor-lab.md` question 6.

## 4b. Two checks that came back clean, and one small gap

- **Revenue growth compares like with like, including across spin-offs.** The worry was that after
  a spin-off the latest quarter would be set against last year's larger, pre-spin quarter.
  `research/measurements/2026-10-14-revenue-growth-same-basis.py` rebuilt the year-ago quarter as
  re-presented in the latest filing for 394 stocks: for all 8 whose year-ago quarter was restated
  by more than 5% (DD -46%, APTV -39%, IFF -31%, OXY, BDX, MLM, HON, IP), the scored growth equals
  the same-basis growth exactly - Yahoo serves the re-presented quarter. The 5 gaps over 10 points
  with no restatement (CPT, PEG, VST, NTRS, EQT) are revenue-definition cases like those above.
- **Three-quarter annualisation is rare and mostly harmless**: 4 stocks (BRK-B, V, COST, GIS). For
  BRK-B it is material - revenue 10% high, net income 15% low against the filed twelve months -
  and the filed quarter that Yahoo is missing is in the SEC cache. Filling it from the filing
  instead of multiplying by 4/3 is a small, well-defined data fix for a later session (it changes
  inputs, so it needs a changelog entry).

## 4. What this suggests building

**Built the same night (2026-10-14):** the standing check below runs in the data loop's health
check, and each stock's Data Provenance panel now says which of its inputs agree with its own
filing - "Checked against the company's own SEC filing for the same quarters: total assets,
equity ... agree within 2%; revenue (83%) differs from it", with the usual reasons for a
difference. 427 of 501 stocks carry the check on the run of 2026-10-14. The panel used to end
"it cannot check that they are true"; for those stocks that is no longer the whole story.

**A standing check of the inputs against the filings, in the data loop.** The dividend outage was a
field vanishing; the next one could be a field changing scale or meaning, which the
availability check added on 2026-10-14 cannot see. Measured tonight, a healthy run has **0** total-
assets disagreements over 10% and **1** operating-cash-flow one. A loop note when either count
jumps (say, above 10 stocks) would catch a mis-scaled field the morning it appears. It should be a
**note, not a gate**, until a few weeks of counts show its normal range - the precedent the health
gate's weight-0 rule follows - and it needs no network: the cache is already refreshed weekly.

## Sources

- SEC EDGAR `companyfacts` for each company, cached 2026-10-09 (`sec_fundamentals.py`).
- Yahoo Finance via `yfinance`: the run's own inputs (published payload), and a quarterly cash-flow
  probe of BSX, GM, FICO, DE, MA and NTRS on 2026-10-14 (`Capital Expenditure`, `Purchase Of PPE`,
  `Purchase Of Intangibles`, `Capital Expenditure Reported`).
