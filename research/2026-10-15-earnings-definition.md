# Which earnings should the screener score? Operating income stays as it is; earnings yield should match it (2026-10-15)

**Proposal status: OPEN** - earnings yield on the source's "Normalized Income" (net income without its unusual items) instead of net income as reported; operating income unchanged. For a later session to review (section 6); nothing changed tonight.

**Owner-run.** Two findings of the same day asked the same question. The top-five audit
(`research/2026-10-15-top-five-audit.md`) found Host Hotels' earnings yield lifted by ~$0.2bn of
hotel-sale gains, and 46 stocks with net income above 1.25x their after-tax operating income. The
13:00 session (`research/2026-10-15-operating-income-source-gap.md`) found the run's operating income
agreeing with the companies' filings for only ~43% of stocks, and left the question *"should an
operating profit used for ROIC and EV/EBITDA include impairments and restructuring charges?"* for a
Monday. This note answers both, from what the source serves, the literature and documented practice.

## 1. What the source serves - measured, not assumed

The quarterly income statement the fetch already downloads carries two operating-income rows and
two net-income rows (`research/measurements/2026-10-15-earnings-rows-scan.py`, all 503
constituents, 0 failed; `...-earnings-definition-stakes.py`):

| Row | What it is | Agrees with the filed `OperatingIncomeLoss`, fiscal year, within 0.5% |
|---|---|---|
| **"Operating Income"** (the screener reads this) | as reported, less "Special Income Charges" | **161 of 380 (42%)** |
| **"Total Operating Income As Reported"** | the company's own figure | **374 of 374 (100%)** |

Spot-checked to the dollar: General Mills $0.886bn as reported against $2.807bn adjusted (a $1.9bn
charge), Kraft Heinz -$4.669bn against +$4.637bn ($9.3bn of write-downs), Boeing +$4.281bn against
-$5.416bn (a $9.7bn **gain** on selling a business - the adjustment removes gains as well as
charges). So the 13:00 session's gap is not a data fault: **the screener has been scoring the
source's pre-special-items operating profit**, and the filing is the as-reported one. The two rows
differ by more than 10% for 74 of 365 companies.

"Normalized Income" is net income without the source's "Total Unusual Items" (tax-effected): Host
Hotels $0.879bn against $1.027bn reported, Western Digital $3.02bn against $9.30bn. The two differ by
more than 10% for 138 of 496 companies.

## 2. The evidence

**Read in full this session:**

- **Rouen, So & Wang (2021), "Core earnings: New data and evidence", *Journal of Financial Economics*
  142(3), 1068-1091** (Harvard DASH manuscript). Russell 3000, 1998-2017, ~54,000 firm-years. Table 3,
  Panel A: next year's earnings regressed on this year's - persistence **0.637 for net income, 0.810
  for core earnings** (net income without non-core items), and **0.178 for the excluded items
  themselves** ("about 30% higher than Net Income and 4.5 times the persistence of Total
  Adjustments"). Restructuring charges are the most common non-core item and among the least
  persistent. Returns: a portfolio long the stocks whose core earnings most exceed their GAAP
  earnings and short the reverse earned a four-factor alpha of about 0.66% a month (Table 9) -
  markets treat the transitory part as if it persisted, most clearly where non-recurring *gains*
  inflate reported earnings. **A caveat from the same paper:** a data provider's classification of
  "special" items can change retroactively (Compustat's three-year rule), which is a look-ahead
  problem for any *backtest* built on adjusted fields; a live screener reading today's figures does
  not inherit it.
- **MSCI, Fundamental Data Methodology (March 2021), pp. 13-14.** EPS is net income "excluding
  extraordinary items or non-recurring items", which are "excluded from earnings on an after-tax
  basis": profit or loss on sale of discontinued operations, restructuring charges, impairment
  losses, acquisition and merger costs, litigation charges, gains or losses on disposals. One stated
  exception: **impairments of oil and gas properties are treated as normal operating expenses** for
  oil and gas companies. MSCI's trailing P/E and ROE use this EPS.
- **Nareit, "Funds From Operations" white paper (2018 restatement).** FFO excludes gains and losses
  on sales of real estate and impairments of depreciable real estate - targeted exclusions, not a
  general "remove the one-offs".
- **Kenneth R. French's data library definitions.** The academic E/P uses earnings *before
  extraordinary items* (special items stay in); operating profitability is revenue less cost of
  goods, SG&A and interest.

**Seen only as abstracts or a digest - direction, not numbers** (the full texts were behind
access controls this session; a later session should read them before any entry cites a figure):
Burgstahler, Jiambalvo & Shevlin (2002, *JAR*) - special items are largely transitory and negative
ones partly pull future costs forward; Dechow & Ge (2006, *RAST*) - low persistence of low-accrual
firms is driven by special items, which investors misread; Doyle, Lundholm & Soliman (2003, *RAST*)
- expenses *managers* exclude from pro-forma earnings predict lower future cash flows; McVay (2006,
*TAR*) - managers shift core expenses into special items. Fairfield, Sweeney & Yohn (1996) could not
be opened at all and is not relied on.

**Where they disagree, and why it matters here.** The persistence and pricing evidence says remove
special items; the pro-forma and classification-shifting evidence says *managers'* exclusions are
biased. The reconciliation, and MSCI's practice, is **who classifies**: a consistent third-party
definition, never the company's own "adjusted" figure. The source's rows are a third-party
classification applied to every company the same way.

## 3. Operating income: keep the adjusted row - the 13:00 session's question, answered

The screener's `ebit` is the pre-special-items figure. That is the construction the persistence
evidence supports and the one practice uses for operating returns (MSCI's non-recurring
exclusions; NOPAT as valuation practitioners build it). Switching to the as-reported row would make
the input agree with the filings and make ROIC and EV/EBITDA *worse* measures of the recurring
business: Gilead would fall from 50th to 235th on a $14bn in-process R&D charge. **No change.**
The disagreement with the filing is the definition, and it is the defensible one; the filing check
should say so rather than count it as a disagreement if operating income is ever added to it.

Two refinements for later, neither urgent: MSCI's oil-and-gas exception (property impairments
recur in that industry and are operating there), and restructuring that recurs year after year
(the persistence literature's main caveat).

## 4. Earnings yield: the one input out of line

Earnings yield is net income over market value, and net income **includes** the special items that
operating income leaves out. So within one category, EV/EBITDA measures the recurring business and
earnings yield measures the recurring business plus this year's gains and charges - the
inconsistency that put Host Hotels' hotel sales and Western Digital's $6.4bn of gains into their
valuation scores. MSCI's EPS, the persistence evidence, and coherence with the screener's own
operating income all point the same way: **earnings yield on earnings without unusual items**.

## 5. The stakes, measured

Today's run rescored by the engine with each choice, everyone else's record unchanged
(`...-earnings-definition-stakes.py`):

| Change | Rank correlation with today | Median move | Moved 25+ places | Top 25 |
|---|---|---|---|---|
| Earnings yield on Normalized Income (**proposed**) | **0.996** | 3 places | 24 | **unchanged** |
| Operating income as reported (not proposed) | 0.989 | 4 places | 47 | 1 name in, 1 out |

The top five do not move under either. Host Hotels stays near the top on normalized earnings - its
rank never rested on the gain alone. The movers are the companies with large one-offs in either
direction: Western Digital (gains) falls, Kraft Heinz and General Mills (charges) rise.

## 6. The proposal, for a later session to decide

**Change:** `earnings_yield = normalizedIncome / marketCap`, where `normalizedIncome` is the source's
"Normalized Income" summed over the same four quarters as `netIncome` (`_stmt_val_ltm`, the annual
statement as the same fallback), falling back to `netIncome` when the row is absent (1 company of
497 on 2026-10-15). Nothing else moves: Piotroski, accruals and ROE keep net income as reported,
which is how their papers define them.

**Before it can ship** (CLAUDE.md rows 0.8 and 0.8c - a scoring change ships with its frontend):
the new raw field in the fetch and its class in `pit_record.FIELD_CLASSES`; `metric_lineage`
`LINEAGE` / `RECOMPUTE` / `EQUATIONS` for earnings yield (the page prints "$1.0B net income /
$15.9B market cap" today); the methodology page's Valuation text; `scoring_schema` bumped so the
scored cache is not reused; a `METHODOLOGY_CHANGELOG.md` entry citing section 2 (the verified sources
only); `ranking_audit`'s earnings-yield note, which would no longer apply once the yield excludes
unusual items; and `tests/test_metric_lineage.py`'s equation bar met, not loosened.

**What would make the reviewer refuse it:** evidence that the source's "unusual items" are
inconsistent across companies or misclassify recurring costs at scale (a sample of 20 against the
companies' own special-item disclosures would test it), or a reading of Doyle, Lundholm & Soliman
showing third-party exclusions carry the same cash-flow cost as managers' exclusions.

## 7. Reproduce

```
python research/measurements/2026-10-15-earnings-rows-scan.py        # network: one request per company
python research/measurements/2026-10-15-earnings-definition-stakes.py
```
