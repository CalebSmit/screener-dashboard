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

**Proposal status: OPEN** - FCF capex excludes captive-finance lease assets for GM, F, DE, CAT, PCAR (section 4 of this note)

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
In the S&P 500 today: **GM, F, DE, CAT, PCAR** - a named, published list, because "is this a
financing arm or the business itself?" is a judgement (United Rentals' rental fleet *is* its
business and must stay in), and a list someone can read is more teachable than a heuristic.

**Why not built tonight:** it changes the heaviest metric's input; the evidence for the lease part
is strong but the S&P Ratings captive criteria themselves were not obtained; and the implementing
data - each company's tagged lease-purchase line, for the same trailing twelve months - is not in
the SEC cache yet (`sec_fundamentals.FACT_CONCEPTS` would need the companies' own tags, and the
cache refreshes weekly). A reviewer should check each of the five filings' tag first.

**Measured effect to expect:** GM's FCF-yield percentile from about 10 to about 90 within its
sector (7.9 composite points), DE and PCAR smaller; no top-25 change from these five on the
2026-10-14 run.

**What would be a good reason to refuse it:** the denominator argument in section 3, if a reviewer
judges a half-consolidated ratio worse than a fully consolidated one; or if the S&P Ratings
captive criteria, once read, restate parent FCF differently.

**If built:** a `METHODOLOGY_CHANGELOG.md` entry citing this note; `metric_lineage` must show the
subtraction on the FCF row for these five (row 0.8c: the page shows the engine's arithmetic);
`scoring_schema` bumped so the scored cache refetches; rewrite the status line above.

## Sources

* MSCI Inc. (2021). *MSCI Fundamental Data Methodology*, March 2021, pp. 15, 34, 42-44.
* S&P Global Ratings (2019). *Corporate Methodology: Ratios And Adjustments*, 1 April 2019, paras. 33, 121-122 (read in the copy reposted by Maalot).
* Damodaran, A. (2022). Valuation lecture slides, NYU Stern, Spring 2022, session 10, slides 145-146.
* Solactive (2024). *Pacer US Cash Cows 100 Index guideline*, v1.0, 21 February 2024, pp. 6, 14; Pacer COWZ fact sheet (data as of 30 September 2026).
* State Street Global Advisors (2025). Overview of the S&P 500 Quality FCF Aristocrats Index.
* General Motors, Form 8-K, Q3 2025, Ex. 99; Ford Motor, Form 8-K, 5 February 2025, Ex. 99; Caterpillar, Form 8-K, Q3 2025, Ex. 99.1 (read on sec.gov).
* Measurements: `research/measurements/2026-10-14-capex-definition.py`; `research/2026-10-14-yahoo-vs-sec.md`.
