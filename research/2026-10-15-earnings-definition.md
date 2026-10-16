# Which earnings should the screener score? Operating income stays as it is; earnings yield should match it (2026-10-15)

**Proposal status: REFUSED 2026-10-16** (13:00 session) - earnings yield stays on net income as
reported. The research case in section 2 is accepted and is not what the refusal rests on: the
*input the proposal reads* cannot be put on one basis for the whole universe at the period the
engine scores, and that is the defect Phase 13 fixed in this exact metric. Section 8 has the
measurements and the conditions under which it returns. The sound half shipped instead: the
earnings-yield row now says, from each company's own filings, how large a one-off charge is inside
the figure (section 9).

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

---

## 8. The review: REFUSED, 2026-10-16 (13:00 session)

**What I am not disputing.** Section 2's evidence is good and I checked the part that carries the
argument. Rouen, So & Wang (2021) measure next-year persistence of **0.810** for core earnings
against **0.637** for net income and **0.178** for the excluded items themselves; MSCI's Fundamental
Data Methodology excludes non-recurring items from EPS on an after-tax basis and uses that EPS for
trailing P/E. Section 4's diagnosis is also right: EV/EBITDA measures the recurring business while
earnings yield measures it plus this year's gains and charges, and that is an incoherence inside one
category. Section 3's conclusion - keep the pre-special-items operating income - stands.

**What the refusal rests on.** Not the research. The *input the proposal reads*. Three measurements,
all offline-reproducible from committed files:

| # | Script |
|---|---|
| 1 | `research/measurements/2026-10-16-normalized-income-same-column.py` (network scan) |
| 2 | `research/measurements/2026-10-16-normalized-income-reconstructible.py` |
| 3 | `research/measurements/2026-10-16-missing-row-means-what.py` |
|   | `research/measurements/2026-10-16-unusual-items-vs-filings.py` (the sample-of-20 test, run on all 503) |

### 8.1 A correction to section 1's method, which changes some of its figures

The 2026-10-15 scan records, per row, that row's **latest populated fiscal year** - and that is not
the same year for every row. UPS's "Normalized Income" is blank for fiscal 2025 while its "Net
Income" is not; BAC's "Total Unusual Items" is blank for 2025. So any cross-row comparison off that
file can put one year's net income beside another year's normalisation. Measured on the 2026-10-15
file, **18 companies** have the three rows spread across different years. Section 1's within-row
figures (the 42% / 100% operating-income agreement, the "differ by more than 10% for 138 of 496")
are unaffected, because each is one row against one filing. Re-measured from a same-column scan
below.

### 8.2 The adjustment cannot be shown on the page for a quarter of the stocks it changes

The source's own construction is `Normalized Income = Net Income - (Total Unusual Items - Tax Effect
Of Unusual Items)`, and all three right-hand rows are served, so the equation *looks* printable. At
the latest fiscal year it is: **346 of 385** adjusted companies rebuild exactly (89.9%), and the
source's "Tax Rate For Calcs" equals the company's own Tax Provision / Pretax Income within 1
percentage point for **328 of 371** (median gap 0.02pp).

At the **trailing-twelve-month window the engine actually sums**, it is not:

| At the TTM window | Companies |
|---|---|
| Adjusted, and the served rows rebuild it exactly | **265** |
| Adjusted, but the unusual-items row cannot be summed over those four quarters | **51** |
| Adjusted, and the served rows do not rebuild it (median gap $44.5M) | **30** |
| No "Normalized Income" for the window at all - the proposal falls back to net income | **23** |
| No adjustment: normalized equals net | 125 |

So **81 of 346 adjusted companies (23.4%)** would carry a scored earnings yield whose adjustment
nothing on the page can reproduce - the largest being **NVDA**, where the source takes $6.49bn off
$192.88bn of trailing net income with the unusual-items row populated in **1 of 4 quarters**.
"Total Unusual Items" is populated in **0 of 4 quarters for 119 companies**: for them the row is
annual-only.

This does not strictly breach row 0.8c, because `normalizedIncome ÷ marketCap` would reproduce the
scored value trivially once the vendor's figure is published as an input. It breaches the thing
0.8c exists to protect. Today's numerator is net income as reported, which item 0.14 measured
against the companies' own filings and found agreeing within 0.5% for **94.3%** of them, and which
each stock's Data Provenance panel shows. The proposed numerator appears in no filing, and for
nearly a quarter of the stocks it moves, the page could not take it apart either.

### 8.3 The decisive one: it cannot be put on one basis for the whole universe

`factor_engine.py:2094` records why this matters, in this exact metric:

> Phase 13 (F19): SINGLE definition across the universe. Previously this fell back to
> trailingEps/price (a different, GAAP-EPS-based definition) when LTM NI or MC was missing, so the
> fallback cohort was ranked on an incomparable basis inside one percentile rank. Now: if LTM NI or
> MC is unavailable, return NaN and let per-row weight redistribution handle the gap.

The proposal's fallback - "falling back to `netIncome` when the row is absent (1 company of 497 on
2026-10-15)" - re-creates that. The "1 of 497" is the **annual** row. At the four-quarter window the
engine sums it is **23 companies**, and they are not a tail:

> ABT, AMZN, APH, BA, BAC, BRK-B, CHD, CHRW, CINF, CPRT, CPT, DLR, DTE, ES, EXR, GIS, ILMN, LEN,
> MDLZ, REG, TFC, WELL, ZBH

Amazon, Bank of America, Berkshire Hathaway and Boeing would be ranked on earnings *including* their
one-off items, in the same percentile rank as 265 companies ranked on earnings excluding them.

**And the fallback cohort is where the one-off items are largest.** Of the 18 whose annual figures
are both known, **8 report annual unusual items worth 10% or more of net income**: General Mills
2,193% (a $1.92bn charge against -$88M of net income), **Boeing 434%** ($9.70bn - the gain on
selling a business), Camden 65%, Digital Realty 56%, Zimmer Biomet 37%, Amazon 20%, Welltower 16%,
Eversource 11%. So the proposal would **remove the small distortions and keep the two largest ones
in the index** - which is worse than leaving the metric alone, because the exceptions are
concentrated exactly where the problem is.

Could "absent" simply be read as "there were none", giving one rule for everyone? **No - measured,
that is false for 70 companies.** Where the quarterly row is short, the company's own fiscal year
reports non-zero unusual items anyway for 70 of them, **33 of those worth 5% or more of net income**
(CVS: -$5.24bn against $1.77bn of net income; Colgate -$929M against $2.13bn; PepsiCo -$1.99bn
against $8.24bn). Only **15** companies have a short row that the fiscal year confirms as zero. The
row's absence is the source's, not the company's.

### 8.4 The refusal condition the note set itself, run on 503 companies instead of 20

Section 6 asked for "a sample of 20 against the companies' own special-item disclosures". Run from
the weekly `companyfacts` cache on every constituent, using only tags the companies apply
themselves (restructuring, goodwill, intangible and asset impairment):

- **Of 493 companies with four fiscal years filed, 120 (24.3%) tagged a restructuring or impairment
  charge in every one of the four years**, and 181 (36.7%) in three or four of them. At 5% or more
  of pre-tax income: 23 in all four years, 65 (13.2%) in three or four.
- Against the **filed** effective tax rate rather than the source's, the implied rate is within 5
  percentage points for **303 of 365** (median absolute gap 0.02pp), but off by more than 20
  percentage points for 54 and more than 50 for 29.

The first is the note's "misclassify recurring costs at scale" condition, and it is met at the scale
the condition names: for a quarter of the index, the item being removed as unusual happens every
year. This is the caveat the note itself parked as a "refinement for later" (section 3) and the
reason MSCI carves out oil-and-gas property impairments - and it is not a corner case.

### 8.5 What also weighed, and what did not

- **The motivating example is not fixed by the proposal.** Section 5 says so: "Host Hotels stays
  near the top on normalized earnings - its rank never rested on the gain alone."
- **The page already tells the reader, where it bites hardest.** `ranking_audit.stock_notes` puts a
  sentence on the earnings-yield row when trailing net income exceeds 1.25x after-tax operating
  income, saying in plain words that the yield uses net income as reported and reads higher than the
  operations alone would support. Explaining rather than silently adjusting is this screener's
  stated design.
- **The low blast radius (0.996 rank correlation, top 25 unchanged) argued neither way.** A change
  that is hard to show and splits the universe's basis is not made acceptable by being small; by the
  same token it is not urgent.

### 8.6 What would carry it

Not "more evidence" - the evidence is already there. A **definition that applies to every stock on
the same basis, built from figures the page can print.** Concretely, any one of:

1. **Quarterly coverage of "Total Unusual Items" and "Tax Effect Of Unusual Items" reaching every
   company the screener scores**, so the adjustment is computed here from served rows (the
   arithmetic that already rebuilds 346 of 385 at the fiscal year) rather than read from the
   vendor's derived row, with no fallback cohort. Re-measure with script 3: the condition is "row
   short, but the fiscal year reports unusual items" at **0**, and "no Normalized Income at the
   trailing window" at **0**.
2. **A free source of per-company one-off items at the quarterly grain**, which the SEC's own tags
   are not (they are display-only here by design, row 0.14 - a weighted metric may not depend on
   the weekly SEC cache, and the tags overlap in ways that need per-company judgement to sum).
3. **A construction that does not need a one-off-items figure at all** - e.g. earnings yield on a
   multi-year average of net income, which dilutes one-offs arithmetically and needs only the rows
   the screener already has. That is a different proposal with its own literature (Graham and Dodd's
   averaging; Shiller's CAPE) and should be raised as one, on a Monday.

Until one of those holds, the honest construction is the one shipped below: score the figure the
companies actually report, and **say what is inside it**.

## 9. What shipped instead (2026-10-16)

The same remedy the refused capex proposal got, for the same reason - if the published figure cannot
honestly be changed, the page says what it contains. The earnings-yield row now carries, for each
stock whose own filings report a material one-off charge in the same twelve months the run scored, a
sentence naming the amount, the share of net income and **the tag the company applied to it**, so a
reader can find it in the filing.

Why this is the complementary half rather than a duplicate of the existing note: the
`ranking_audit` sentence fires only when net income runs *above* after-tax operating income, which
catches the **gain** direction - and it is correctly one-sided, because interest expense routinely
puts net income below after-tax operating income for sound reasons, so the same ratio cannot be
read in reverse. The **charge** direction - General Mills and Kraft Heinz in section 5, whose
as-reported earnings yield reads too *low* - had nothing on the page at all. It does now, from the
filings rather than from a vendor's classification.

Display-only: `cavs['earnings_yield']` is prose, enters no `raw`, `pct`, category score or
composite, and a test fails if a scoring module imports `sec_crosscheck`.
