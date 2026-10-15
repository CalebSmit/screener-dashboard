# What should capital expenditure include in free-cash-flow yield? (2026-10-14, owner-run)

**Question** (`plan/factor-lab.md` question 6, raised by `research/2026-10-14-yahoo-vs-sec.md`
section 3). FCF yield is the heaviest single metric in the composite (45% of Valuation, 9.9
points). Its capex is Yahoo's `Capital Expenditure` row, which is a sum: purchases of PP&E, plus
purchases of intangibles, plus - for companies with a captive finance arm - the vehicles and
equipment bought to lease to customers. Which of those belongs in a free-cash-flow yield?
Research-led (rule 4): no return of the screener's ranking is used.

**Finding.** The sources split on intangibles and agree on finance-arm leases.

- **Captive-finance lease assets: every source that addresses them keeps them out of industrial
  capex.** GM, Ford and Caterpillar each define their own free cash flow without them; S&P Global
  Ratings classes them as captive-finance assets. No source counts them as capex.
- **Intangibles and capitalised software: the sources disagree.** MSCI's data definition is
  physical assets only; S&P Global Ratings explicitly includes intangible and capitalised
  development spending; Damodaran would widen capex further still.

**Proposed, not built:** remove finance-arm lease purchases from capex for the five S&P 500
companies with captive finance arms; leave everything else as it is. Section 4.

**Proposal status: REFUSED 2026-10-15** (section 5) - the rule removes the finance arm's capex
while keeping the finance arm's operating cash flow, and GM's own 10-K shows that gap is $8.2B a
year. What shipped instead: the page now says what the capex figure contains.

---

## 1. Documented practice, read from the documents

Spot-checked against the files on 2026-10-14 (a second reader fetched them; the three
definitions quoted below were re-read from the PDFs).

| Source | What capex includes | (a) PP&E | (b) intangibles | (c) finance-arm leases |
|---|---|---|---|---|
| **MSCI Fundamental Data Methodology** (March 2021, pp. 34, 43-44) | investment "to acquire or upgrade fixed, physical, non-consumable assets" | yes | not mentioned; "physical" leans against | not addressed |
| **S&P Global Ratings, Corporate Methodology: Ratios and Adjustments** (1 April 2019, para. 33, p. 6) | "funds spent to acquire or develop tangible and intangible assets", adjusted for capitalised development costs | yes | **yes** | treats leased assets of a captive acting as lessor as **captive-finance assets** (paras. 121-122) |
| **Damodaran**, NYU Stern valuation slides (Spring 2022, slide 146) | adds R&D once capitalised and normalised acquisitions | yes | **yes, and wider** | not addressed |
| **S&P DJI** (FCF Aristocrats, via State Street's 2025 overview) and **Solactive / Pacer US Cash Cows 100** (guideline v1.0, 2024, p. 14) | "operating cash flow less capital expenditures" - capex not defined; Financials excluded | - | - | - (the Financials filter does not catch captives inside autos and machinery) |
| **GM** (Q3 2025 8-K, Ex. 99) | automotive FCF deducts automotive "expenditures for property"; "purchases of leased vehicles" sit only in GM Financial's column | yes | - | **excluded** |
| **Ford** (8-K, 5 Feb 2025, Ex. 99) | company adjusted FCF deducts capital spending excluding Ford Credit; Ford Credit's operating-lease vehicle purchases are not deducted | yes | - | **excluded** |
| **Caterpillar** (Q3 2025 8-K, Ex. 99.1) | reports "capital expenditures - excluding equipment leased to others" separately from "expenditures for equipment leased to others" (mostly Financial Products) | yes | - | **separated** |

*Not obtained:* S&P DJI's own methodology PDF (behind a bot check, not worked around), S&P
Ratings' separate captive-finance criteria (where the restated-FCF rule itself sits), Moody's,
FactSet's and Yahoo's field definitions, PACCAR's and Deere's own FCF definitions read first-hand
(Deere's financial-services-only "cost of equipment on operating leases" is from a search snippet).

## 2. What the screener does today, measured

From `research/measurements/2026-10-14-capex-definition.py` (run of 2026-10-14) and the Yahoo
cash-flow probe of that day:

| Company | Yahoo capex | Filed PP&E | FCF yield today -> PP&E only | Sector percentile |
|---|---|---|---|---|
| GM | $22.6B | $8.8B | 0.3% -> 8.0% | 10 -> 90 |
| DE | $4.0B | $1.2B | 1.4% -> 2.6% | 10 -> 25 |
| PCAR | $1.4B | $0.6B | 4.7% -> 5.9% | 75 -> 92 |
| BSX | $2.7B | $0.9B | 2.4% -> 4.8% | 15 -> 59 (intangibles, not leases) |

Across the 337 stocks with both figures, the top 25's composites move by at most 0.5 points under
a PP&E-only rule. The large moves are a handful of companies, and the biggest are the captives.

## 3. Reading it

**Finance-arm leases (c).** The case is one-sided. A captive's lease fleet is the asset side of a
lending business, funded by the captive's own debt; the companies themselves report free cash
flow without it, and so does the rating agency's treatment. Counting it as capex sets GM's free
cash flow at $0.6B on $23.2B of operating cash flow - a number GM, Ford-style reporting and S&P
Ratings would all say describes a bank's loan growth, not the industrial business's investment.

**The honest complication: the denominator.** FCF yield divides by enterprise value, and Yahoo's
EV for GM includes GM Financial's debt. Removing the finance arm's lease spending from the
numerator while keeping its debt in the denominator is not fully consistent either - the clean
version is GM's own: automotive cash flow over automotive enterprise value, which needs segment
data the screener does not have. The proposal fixes the larger, one-directional distortion (a
lending book's growth treated as industrial investment) and leaves the smaller one, and says so.

**Intangibles (b).** The sources disagree - MSCI's "physical" against S&P Ratings' "tangible and
intangible" - and the economic argument (a company that capitalises software development should
not look better than one that expenses it) favours keeping them. BSX's $1.8B includes acquired
intellectual property, which is closer to an acquisition; Damodaran would keep even acquisitions,
normalised. With practice split, the rule written beforehand applies: the screener keeps what it
does today, and the note records the disagreement.

## 4. The proposal, for a second session to review

**Rule:** for companies whose filings report a captive finance arm's purchases of equipment or
vehicles for operating leases, FCF capex = Yahoo's `Capital Expenditure` minus those purchases.
In the S&P 500 today: **GM, DE, CAT, PCAR** - a named, published list, because "is this a
financing arm or the business itself?" is a judgement (United Rentals' rental fleet *is* its
business and must stay in), and a list someone can read is more teachable than a heuristic.

**The tags, read from each company's `companyfacts` the same night** (fiscal 2025, 10-K):

| Company | Lease-purchase tag | FY2025 | Filed PP&E | Yahoo capex = PP&E + lease line? |
|---|---|---|---|---|
| GM | `PaymentsToAcquireLeasesHeldForInvestment` | $15.8B | $9.3B | yes (TTM $8.8B + $13.8B = $22.6B) |
| DE | `PaymentsToAcquireEquipmentOnLease` | $2.9B | $1.4B | yes |
| CAT | `PaymentsToAcquireEquipmentOnLease` | $1.5B | $2.8B | yes (TTM $2.87B + ~$1.7B = $4.58B) |
| PCAR | `PaymentsToAcquireEquipmentOnLease` | $0.6B | $0.7B | yes |
| F | `PaymentsToAcquireLoansAndLeasesHeldForInvestment` (loans too) | $55.7B | - | **no** - Yahoo's Ford capex ($9.7B) is already capital spending without Ford Credit's leases |

**Ford is therefore out of the list**: Yahoo already reports it the way Ford does. The two tags were
added to `sec_fundamentals.FACT_CONCEPTS` that night, so the next weekly refresh of the cache
carries them and the rule can be built on filed figures, for the same trailing twelve months as
the cash flow.

**Why not built tonight:** it changes the heaviest metric's input; the evidence for the lease part
is strong but the S&P Ratings captive criteria themselves were not obtained; and the cache will
only carry the lease tags after its next weekly refresh.

**Measured effect to expect:** GM's FCF-yield percentile from about 10 to about 90 within its
sector (7.9 composite points); today DE sits at the 9th percentile, CAT the 21st and PCAR the 71st,
and each would rise; no top-25 change from these four on the 2026-10-14 run.

**What would be a good reason to refuse it:** the denominator argument in section 3, if a reviewer
judges a half-consolidated ratio worse than a fully consolidated one; or if the S&P Ratings
captive criteria, once read, restate parent FCF differently.

**If built:** a `METHODOLOGY_CHANGELOG.md` entry citing this note; `metric_lineage` must show the
subtraction on the FCF row for these five (row 0.8c: the page shows the engine's arithmetic);
`scoring_schema` bumped so the scored cache refetches; rewrite the status line above.

## 5. The review, 2026-10-15 (06:00 nightly session): REFUSED, and what shipped instead

Reviewed by the next session, as the routine requires - the author of a proposal is not its only
reviewer. The measurements below are new; they are the half of the ratio section 4 did not look
at. Reproduce with `research/measurements/2026-10-15-capex-proposal-review.py` (filed figures,
one `companyfacts` request per company), `2026-10-15-captive-concepts.py` (the tags these filers
actually use) and `2026-10-15-gm-automotive-ocf.py` / `2026-10-15-captive-own-fcf.py` (the
companies' own definitions, read from sec.gov).

**First, section 4's figures reproduce.** GM's trailing-twelve-month capex of $22.60B is
$8.80B of filed PP&E purchases plus $13.79B of `PaymentsToAcquireLeasesHeldForInvestment`, to the
cent; DE, CAT and PCAR likewise. Nothing in section 2 or 4 is wrong.

### 5.1 The proposal takes half of its own sources' definition

Every source in section 1 excludes a captive's lease purchases from capex. **Each of them also
starts from the industrial segment's own operating cash flow**, which the proposal does not - it
keeps Yahoo's consolidated figure. Read from the primary documents today:

* **GM**, FY2025 Form 10-K (filed 2026-01-27), "Adjusted Automotive Free Cash Flow": *"We measure
  adjusted automotive free cash flow as automotive operating cash flow from operations less
  capital expenditures adjusted for management actions. For the year ended December 31, 2025,
  **net automotive cash provided by operating activities was $18.7 billion**, capital
  expenditures were $9.2 billion, and adjustments for management actions were $1.1 billion."*
  (2024: $23.9B and $10.7B; 2023: $20.8B and $10.7B.)
* **Caterpillar**, 8-K filed 2026-06-11, Ex. 99.1: free cash flow is *"Machinery, Power & Energy
  (MP&E) free cash flow"* - the industrial segments, excluding Financial Products, on both sides.
* **Ford**, per section 1: company adjusted FCF excludes Ford Credit on both sides.

GM's **consolidated** operating cash flow for the same fiscal year is **$26.87B**
(`NetCashProvidedByUsedInOperatingActivities`, filed). The difference, **$8.17B**, is GM Financial
and the eliminations - the lease rentals and finance-charge income the proposal would leave in the
numerator after removing the asset purchases that earn them.

So, for fiscal 2025, over the same twelve months:

| Free cash flow for GM | FY2025 | vs GM's own figure |
|---|---|---|
| **a. today** - consolidated OCF - Yahoo capex (PP&E + lease purchases) | **$1.78B** | **-81%** |
| **b. the proposal** - consolidated OCF - filed PP&E | **$17.57B** | **+85%** |
| c. the lease book netted - b less (purchases $15.79B - proceeds $10.10B) | $11.88B | +25% |
| **GM's own**: net automotive OCF $18.7B - capex $9.2B | **$9.5B** (adjusted: $10.6B) | - |

The proposal's error is **exactly** the consolidated-minus-automotive operating cash flow gap
($17.57 - $9.5 = $8.07B against $8.17B, the rest being $9.3B of filed PP&E against GM's $9.2B of
automotive capex). It is not a rounding difference or a judgement at the margin: the construction
publishes a lending book's receipts as the industrial business's free cash flow.

### 5.2 A lease fleet is a three-legged cycle and the ratio would count one and a half legs

From GM's own filings, fiscal 2025:

| Leg | Tag | FY2025 | Where it lands in `OCF - capex` |
|---|---|---|---|
| Buy the vehicle | `PaymentsToAcquireLeasesHeldForInvestment` | $15.79B out | the line the proposal removes |
| Collect the rentals | `LeaseIncome` | $7.80B in | **inside operating cash flow - kept** |
| Sell it at lease end | `ProceedsFromLeasesHeldForInvestment` | $10.10B in | investing: **never counted, either way** |

GM's **net** investment in the lease book was $5.69B, not $15.79B. Today's figure overstates the
outflow 2.8x by ignoring leg 3; the proposal removes the whole of leg 1 while keeping leg 2. Row c
above - netting legs 1 and 3 - is the construction nearest the company's own number, and it is
**not buildable**: of the four companies, only GM files the proceeds leg distinctly (CAT tags a
general `ProceedsFromSaleOfPropertyPlantAndEquipment`; DE and PCAR tag nothing lease-specific).
A rule that netted where the tag exists and did not where it does not would be a fourth
inconsistency, not a fix.

### 5.3 The denominator, which section 4 already flagged

GM's enterprise value in the run is **$178.6B** and includes GM Financial's debt. The proposal
removes the finance arm's capex from the numerator, keeps its operating cash flow in the
numerator, and keeps its debt in the denominator. Section 4 named this as a legitimate reason to
refuse; with 5.1 measured, it is no longer the smaller of the two problems but part of the same
one.

### 5.4 It would put the heaviest metric on the SEC cache's critical path

FCF yield is 9.9 of 100 composite points. Scoring reads no SEC data today, deliberately - the
filing check is display-only and `tests/test_filing_check_in_the_drilldown.py` fails if
`sec_crosscheck` is imported by a scoring module. The proposal's rule needs filed figures *inside*
the metric, so a weekly cache whose own staleness alarm was added on 2026-10-15 would decide the
largest metric for four stocks: a failed refresh would silently restate their capex. The proposal
does not weigh this, and it is the kind of coupling the project has had to undo before.

### 5.5 Direction of the error

The proposal moves GM from roughly the 10th to the 90th percentile of its sector on this metric -
7.9 composite points - on a number no source publishes. Today's figure is wrong in the other
direction and costs GM points it should have. For a long-only screener the two are not equivalent:
a stock wrongly promoted can be bought, a stock wrongly held back cannot lose money. With both
constructions measurably wrong, the one that promotes is the worse default.

### 5.6 What shipped instead

Refusing leaves a -81% figure in the heaviest metric, so the refusal is not the end of it. **The
clean ratio is not computable from free data**: GM's automotive operating cash flow is a non-GAAP
measure in the MD&A text, not an XBRL fact (segment reporting does not require cash flow by
segment), and the same is true of CAT's MP&E figure. What *is* available is every number needed to
tell the reader what the published figure contains.

So the drilldown's FCF-yield row now carries a per-stock note, from the filings, wherever a
company's own filings report purchases of assets to lease out worth 10% or more of the capex the
run used: how much of the capex that is, the proceeds leg where it is filed, and that companies
with a finance arm report free cash flow for the industrial business only. Display-only; it
enters no score (claim `drilldown.capex_lease_note`, `tests/test_capex_lease_note.py`). It is a
rule on a filed tag rather than the named list of section 4, which also answers section 4's own
worry about judgement: United Rentals' rental fleet is its business, and the note describes what
the figure contains rather than deciding whether it should.

### 5.7 What would reopen it

1. **A free source of segment-level operating cash flow** for a captive-finance parent. The SEC's
   Financial Statement Data Sets carry dimensional facts; whether any of these filers tags a
   segment cash flow at all is the question to settle first, and the answer may be no.
2. **S&P Global Ratings' captive-finance criteria** (not obtained, section 1), which is where the
   restated-parent rule actually sits. If it restates parent free cash flow in a way computable
   from filed figures, that is a sourced construction rather than one of ours.
3. Either way, the rule must come with its own answer to 5.4 - a published, versioned figure
   rather than a live cache read - and the construction must be the same on both sides of the
   ratio.

**Not a reason to reopen it:** that today's number looks too low. It does, and it is said so on
the page now.

## Sources

* MSCI Inc. (2021). *MSCI Fundamental Data Methodology*, March 2021, pp. 15, 34, 42-44.
* S&P Global Ratings (2019). *Corporate Methodology: Ratios And Adjustments*, 1 April 2019, paras. 33, 121-122 (read in the copy reposted by Maalot).
* Damodaran, A. (2022). Valuation lecture slides, NYU Stern, Spring 2022, session 10, slides 145-146.
* Solactive (2024). *Pacer US Cash Cows 100 Index guideline*, v1.0, 21 February 2024, pp. 6, 14; Pacer COWZ fact sheet (data as of 30 September 2026).
* State Street Global Advisors (2025). Overview of the S&P 500 Quality FCF Aristocrats Index.
* General Motors, Form 8-K, Q3 2025, Ex. 99; Ford Motor, Form 8-K, 5 February 2025, Ex. 99; Caterpillar, Form 8-K, Q3 2025, Ex. 99.1 (read on sec.gov).
* Measurements: `research/measurements/2026-10-14-capex-definition.py`; `research/2026-10-14-yahoo-vs-sec.md`.
* Review (section 5): General Motors, Form 10-K for fiscal 2025, filed 27 January 2026,
  MD&A "Automotive Cash Flow" and "Adjusted Automotive Free Cash Flow" (read on sec.gov);
  Caterpillar, Form 8-K filed 11 June 2026, Ex. 99.1; GM Form 8-K filed 29 November 2023, Ex. 99.1
  (the same definition stated as guidance); filed XBRL facts per company via `companyfacts`.
  Measurements: `research/measurements/2026-10-15-capex-proposal-review.py`,
  `2026-10-15-captive-concepts.py`, `2026-10-15-captive-own-fcf.py`,
  `2026-10-15-gm-automotive-ocf.py`.
