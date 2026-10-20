# Bank price-to-book: what the source's ratio is, and what the screener should score (2026-10-19)

*Proposal open (status line at the end): score bank-like P/B as market value over common equity
from the run's own balance sheet. For a later session to decide; no scoring code was changed.*

**The question.** P/B is 60% of bank Valuation - at Valuation's 22%, **13.2 composite points for each
of the 46 bank-like stocks** - and it is read from the source's `.info` `priceToBook`
(`factor_engine.py`, "Bank Valuation: Price-to-Book"). The bank ROE/ROA note
(`research/2026-10-19-bank-roe-roa.md`) found that for 18 of the 46 the source's `.info` ratios
reflect a newer quarter than the statements the run scores, and left P/B for its own look. Does
`priceToBook` have the same mismatch, is it right, and what should be scored?

**The answer, in five lines.**

1. **The ratio is defined and mostly right.** `priceToBook` is exactly `currentPrice / bookValue` (502
   of 502 stocks), and `bookValue` is **common** equity per share - the basis MSCI and bank reporting
   use: 22 of the 28 stocks whose source quarter matches the run's statements agree within 0.5%.
2. **It has the same period mismatch as ROE, but a small one.** For the same 18 stocks `bookValue` is
   the September book while everything else scored is June (or March); the difference from the
   statement-date book is a median **0.3%**, 7 of 18 beyond 2% (Schwab +17%, Travelers -11%).
3. **It has one large error, and it is Berkshire's.** `bookValue` for BRK-B is per **Class A** share
   ($522,226) against a Class B price ($532), so its P/B reads **0.0010** instead of about **1.52**. It
   is the cheapest bank-like stock by a factor of nearly 1,000, its P/B percentile is 98.9, and it has been above
   91 in **all 52 snapshots** since March. Corrected alone, BRK-B goes **60th -> 114th**.
4. **Market value over common equity from the run's own balance sheet fixes both**, with no per-share
   figure that can be on the wrong class's unit: it agrees with the source within 2% for 33 of 45
   stocks (Berkshire aside), the run's equity agrees with the filings for 41 of 41 (and its common
   equity with filed equity less filed preferred for 32 of 41, every miss a company whose preferred
   the check could not find), and rescored on the
   2026-10-19 run the **top 25 is unchanged**, the bank-like median move is 2.5 places, and BRK-B
   goes 60th -> 121st.
5. **Two refinements are not proposed.** Stockholders' equity *including* preferred stock (no new
   field) changes the top 25 and makes preferred-heavy banks look cheaper for capital common holders
   do not own. Tangible book (P/TBV), the bank analysts' measure, moves the bank set a median 20 places
   and the source's tangible-book row agrees with the filings for only 22 of 41.

One exception to (4) is recorded, not fixed: **Interactive Brokers' market value counts its founders'
operating-company units** (market value / (price x balance-sheet shares) = 3.78), so any
market-value ratio overstates its P/B and understates its earnings yield about four times - the same
error its scored earnings yield (0.8%) already carries. That is a market-value question for Up-C
companies, not a P/B one (section 4).

---

## 1. Method

`research/measurements/2026-10-19-bank-price-to-book.py` (JSON beside it).

- **The run:** the newest full run, `377ec40838b1` (2026-10-19 02:00, 501 scored), its raw fetch and
  metric table.
- **The source's balance sheets and `.info`:** the bank ROE/ROA probe of the same morning
  (`2026-10-19-bank-roe-roa-probe.json`, 08:31). Its `bookValue` equals the run's for **46 of 46**, so
  it describes what the run scored. It carries the quarterly "Stockholders Equity", "Common Stock
  Equity", "Total Equity Gross Minority Interest" and "Tangible Book Value" rows.
- **The filings:** the weekly SEC facts cache (`StockholdersEquity`, `CommonStockSharesOutstanding` at
  the run's balance-sheet date, via `sec_crosscheck.sec_value`), plus a `sec` step that fetched SEC
  `companyconcept` series for `Goodwill`, `IntangibleAssetsNetExcludingGoodwill` and
  `PreferredStockValue` (the items between stockholders' equity and tangible common equity, which the
  cache does not carry), saved as `2026-10-19-bank-price-to-book-sec.json` so the rest re-runs offline.
- **The stakes:** the run's metric table rescored with the run's own configuration and effective
  weights - **the baseline reproduces all 501 published ranks** (asserted) - with only the bank-like
  stocks' `pb_ratio` replaced.

## 2. What the source's ratio is

`priceToBook` equals `currentPrice / bookValue` to within 1e-5 for **502 of 502** stocks. Unlike ROE and
ROA (section 3 of the ROE note), there is no hidden construction: the only questions are what
`bookValue` is and when.

`bookValue` against equity / balance-sheet shares at the run's statement date, three equity bases, for
the 28 bank-like stocks whose source quarter (`mostRecentQuarter`) is the run's statement quarter:

| Equity basis | Within 0.5% | Within 2% |
|---|---|---|
| Stockholders' equity (incl. preferred) | 12 | 15 |
| **Common equity (excl. preferred)** | **22** | **25** |
| Total equity incl. minority interests | 6 | 11 |

So `bookValue` is common book per share, the definition MSCI uses (section 5). The misses are AXP
(1.1%), CFG (1.0%), L (0.8%), WRB (4.5%), COF (9.5%) and Berkshire (section 4). Measured on the filings
the other way round, the source's common-equity row agrees with filed stockholders' equity less filed
preferred within 2% for 32 of 41; all nine misses are companies whose preferred stock the check could
not find in the filings at that date (JPM 5.6%, BAC 8.3%, STT 12.6% - roughly each one's preferred).

## 3. Which quarter: the same 18 stocks, a smaller effect

For the 18 stocks whose `mostRecentQuarter` is 2026-09-30 while the run's statements end in June (or
March) - BAC, BNY, C, CINF, GS, IBKR, JPM, MS, MTB, PGR, PNC, RF, SCHW, STT, TFC, TRV, USB, WFC, the same
list as the ROE note's - `bookValue` matches the statement-date common book for only 2 within 0.5%. The
ratio of the source's book to the statement-date book:

- median **0.997**; 11 of 18 within 2%;
- largest: SCHW **1.17**, TRV **0.89**, CINF 1.07, RF 0.95, IBKR 1.05.

Book value moves slowly - one quarter's retained earnings, buybacks and bond-portfolio marks - so the
period question costs far less here than for ROE, where a quarter of earnings enters a trailing sum.
But it is the same defect of record: P/B for these 18 rests on a balance sheet that neither the run's
statements nor the SEC cache can yet confirm (the September 10-Qs are not filed), while the earnings
yield that is the other 40% of bank Valuation is on June earnings.

## 4. Two share-count failures: Berkshire and Interactive Brokers

A per-share check (equity / `bookValue` against the company's share count implied by market value)
finds two stocks off by more than 25%:

**Berkshire Hathaway (BRK-B), 0.00067.** `bookValue` is $522,225.90 - book per Class A share. The price
is per Class B share, worth 1/1,500 of an A. Equity of $747.9bn over 1,431,693 Class A-equivalent shares
x 1,500 is **$348.26 per B share**, a P/B of **1.53**. The source's `trailingEps` ($39.76) and market
value ($1,139bn) are both on the B unit and right; only `bookValue`, and so `priceToBook`, is wrong. In
the bank P/B ranking BRK-B sits at the 98.9th percentile (cheapest), and its P/B percentile has been
**at least 91.4 in all 52 snapshots** that carry it (2026-03-09 to 2026-10-19, median 98.3). Today it
lifts Berkshire's Valuation score from 60.8 to 81.7 and its rank from 114th to 60th. S&P DJI treats
Berkshire as the exception to its multiple-class rule for the same reason - its share count is
"consolidate[d] ... under the B share class line" (*Equity Indices Policies & Practices*, August 2026,
p. 17; see `research/2026-10-19-share-classes.md`).

**Interactive Brokers (IBKR), 0.25.** Here `bookValue` is right ($13.74, IBKR Inc.'s own equity per
Class A share) and the *market value* is the outlier: it is price x about 3.8 times the balance-sheet
share count, i.e. it counts the founders' units in the operating company, which IBKR Inc. consolidates
as a noncontrolling interest. The source's P/B (6.02) is the right one; market value over IBKR's own
equity reads 23.9. The same overstated market value already sits in IBKR's scored earnings yield
(net income to IBKR Inc. / that market value = **0.8%**, 5.9th percentile), its size and its EV ratios.
The bank ROE/ROA note found the same Up-C structure behind IBKR's ROA. It is a market-value question
for Up-C companies, outside this note.

**The lesson for construction.** A per-share ratio needs the price and the book on the same share
unit; Berkshire's breaks that. A market-value ratio needs the market value to be the company's; IBKR's
breaks that. Market value is verified as the company's for 457 of 502 stocks within 2% (share-class note,
section 2) and for **45 of the 46 bank-like stocks** within 5% (IBKR the exception), while per-share
figures have no such check in the run. So the market-value construction fails once, visibly, in a way the
screener already carries elsewhere; the per-share construction fails silently.

## 5. What practice says - read this session

**MSCI, *Fundamental Data Methodology* (March 2021).**
https://www.msci.com/eqb/methodology/meth_docs/MSCI_Fundamental_Data_Methodology_Mar2021.pdf

- Section 1.1.1 (p. 10): per-share figures use "the total number of shares at the company level at the
  date of the results", all classes aggregated.
- Section 1.2.5 (p. 16): book value is "shareholders' equity available to shareholders at the latest
  period end date excluding minority interest, treasury shares and preferred shares that do not exhibit
  equity like characteristics", and "Price/Book Value ... is calculated using the latest book value per
  share". Section 2.1 (p. 19): P/BV = "Current security price" / book per share.
- So MSCI's P/B is current price over **common** book (excluding preferred and minority interests) per
  company share - which is current market value over common equity whenever market value is price x
  company shares. On the period, MSCI uses the *latest* book; its rule that book may not be newer than
  earnings is stated for ROE (section 2.2.2, p. 21), not for P/B.

**Damodaran (2009), "Valuing Financial Service Firms", NYU Stern working paper.**
https://pages.stern.nyu.edu/~adamodar/pdfiles/papers/finfirm09.pdf

- pp. 30-31: P/B for a financial firm is "the ratio of the price per share to the book value of
  equity per share"; return on equity is its "companion variable", and the link "should be stronger
  for financial service firms than for other firms" because book equity tracks the market value of
  financial assets. For U.S. banks in February 2009 the P/B-ROE correlation exceeded 0.70.
- p. 23: book equity "can be affected by stock buybacks and extraordinary or one-time charges" - which
  is what moves a bank's book between quarters (Travelers' source book, 11% below its June figure,
  shows how large one quarter's move can be; its cause was not checked).

**JPMorgan Chase, Form 10-K for 2022, MD&A, "TCE, ROTCE and TBVPS" (p. 59).**
https://www.sec.gov/Archives/edgar/data/19617/000001961723000231/jpm-20221231.htm
"TCE represents the Firm's common stockholders' equity (i.e., total stockholders' equity less preferred
stock) less goodwill and identifiable intangible assets (other than MSRs), net of related deferred tax
liabilities ... TBVPS represents the Firm's TCE at period-end divided by common shares at period-end.
TCE, ROTCE and TBVPS are utilized by the Firm, as well as investors and analysts, in assessing the
Firm's use of equity." Bank reporting starts from **common** equity and, for the tangible measures,
adds back the deferred tax on deductible goodwill - a detail no free data source carries.

**S&P DJI, *S&P U.S. Style Indices Methodology* (June 2022), p. 6.** "Book Value to Price Ratio" is one
of three value factors, computed "for each company" - P/B at the company level, as here.

**Academic convention, for context (not read this session; quoted in
`research/2026-10-09-bank-like-financials.md`):** Fama & French (1992, p. 429) exclude financial firms
from their book-to-market tests because their leverage "probably does not have the same meaning". The
published value premium is therefore not evidence about bank P/B; the screener's bank P/B rests on the
valuation practice above (Damodaran's P/B-ROE pairing), which is how `bank_metric_weights` was argued.

**Where practice divides, and why.** *Common or total equity:* everyone above uses common book; preferred
stock is a senior claim the common price does not buy. *Book or tangible book:* bank and broker analysts
quote P/TBV (JPMorgan's TBVPS; the GS/MS comparisons cited in the bank-like note), because goodwill from
acquisitions is not loss-absorbing capital; index providers and Damodaran use book; insurers are
conventionally valued on book (or book excluding unrealised bond marks), not tangible book. *Latest or
statement-date book:* MSCI uses the latest; the screener's rule since the ROE note is one period per
stock's record, checkable against the filings.

## 6. The options, rescored on the 2026-10-19 run

Everyone else unchanged; baseline = the published ranking (501 of 501 reproduced).

| Bank P/B on | Agrees with source (excl. BRK-B), within 2% | Composite rank corr. | Bank-like median move | Moved 10+ | Largest move | BRK-B | Top 25 |
|---|---|---|---|---|---|---|---|
| Source, Berkshire's unit corrected only | 45 of 45 | 0.99982 | 0 | 2 | BRK-B 60 -> 114 | 60 -> 114 | unchanged |
| Market value / stockholders' equity (incl. preferred; no new field) | 16 of 45 | 0.99945 | 7 | 17 | BRK-B 60 -> 138 | 60 -> 138 | **ACGL out, DECK in** |
| **Market value / common equity, statement date (proposed)** | **33 of 45** | **0.99972** | **2.5** | **6** | BRK-B 60 -> 121 | **60 -> 121** | **unchanged** |
| Market value / tangible book (source row) | - | 0.99836 | 20.5 | 34 | AFL 258 -> 197 | 60 -> 94 | AIZ out, CINF in |

Under the proposal: **BRK-B 60th -> 121st** (Valuation 81.7 -> 59.5), COF 175th -> 153rd (P/B 1.20 ->
1.09: its source book per share is 9.5% below the statement-date common book per share), RF 112th -> 95th (1.42 -> 1.34:
September book against June), WFC 370th -> 353rd, IBKR 401st -> 413th (section 4's market value), JPM
361st -> 370th. The best-ranked bank-like stocks stay the same five (EG 6th, ALL 10th, TRV 21st -> 17th,
AIZ 19th -> 20th, ACGL 24th -> 22nd).

Reading the rejected rows:

- **Stockholders' equity including preferred** agrees with the source for only 16 of 45, because 22 of
  the 46 carry preferred stock worth more than 5% of equity (ROE note, section 5). It makes those banks
  look cheaper by the amount of preferred they have issued (the P/B of KEY, STT, WFC, GS, USB and BNY
  falls 11-13%), and the reshuffle pushes ACGL from 24th to 27th, out of the top 25.
- **Tangible book** is the better bank measure in principle, but here it is the worst-measured: the
  source's row agrees with filed equity less preferred, goodwill and intangibles within 2% for 22 of 41
  (filers tag intangibles inconsistently - 13 of 46 under the standard concept - and JPMorgan-style
  deferred-tax add-backs are not tagged at all), and it reorders the bank set (median 20.5 places,
  P/B percentile moves of 10+ for 22 stocks). Insurers - 17 of the 46 - are valued on book, not
  tangible book. A tangible-book metric for banks proper is a research question of its own.

## 7. How it fits the rest of the screener

- **The bank Valuation category becomes one market value and one statement date.** Earnings yield is
  net income / market value; the proposed P/B is market value / common equity at the same balance
  sheet. Both are company-level ratios with no per-share figure, and both inherit any market-value
  error together (IBKR), so a fix to market value fixes the category at once.
- **With the ROE/ROA proposal.** That note proposes ROE on net income over *average stockholders'*
  equity (total-equity basis, chosen there because the source's common-income rows are unreliable).
  P/B on *common* equity is not inconsistent with it: each ratio keeps its own numerator and denominator
  on one claim (common price over common book; all-equity income over all equity). The P/B-ROE identity
  stops being exact by the preferred share of equity - under 5% for 24 of the 46 and at most 12.6%.
  If the reviewer wants exactness, the alternative is to move ROE to the common basis, which that note
  measured and set aside.
- **What it costs.** One new raw field, "Common Stock Equity" from the quarterly balance sheet the fetch
  already downloads (present for 46 of 46 at the statement date in the probe). `priceToBook` and
  `bookValue` stop being read for scoring; they can stay in the drilldown as the source's figures.
- **Backtest v2.** `pit_record.market_fields` builds `priceToBook` as market value / stockholders'
  equity (including preferred) - a third construction. It would follow the scored definition once
  common equity has a point-in-time source (the SEC facts cache carries no common-equity
  concept; stockholders' equity less `PreferredStockValue` is the filings' route).

## 8. The proposal, for a later session to decide

**Change.** For bank-like stocks only (`_is_bank_like`), in `factor_engine.compute_metrics`:

```
pb_ratio = marketCap / commonEquity    # both > 0 required, else NaN
```

where `commonEquity` is a new raw field, the quarterly balance sheet's "Common Stock Equity" at the
run's balance-sheet date (`_stmt_date_balance_sheet`, the same column as `totalEquity`), with no
fallback to `priceToBook`, to `bookValue` or to stockholders' equity, so every bank-like stock is ranked
on one construction (the Phase 13 F19 rule).

**Before it can ship** (CLAUDE.md rows 0.8 and 0.8c):

- `factor_engine._fetch_single_ticker_inner` (the balance-sheet block beside `totalEquity`): `rec["commonEquity"] =
  _stmt_val(_bs_src, "Common Stock Equity")`; `pit_record.FIELD_CLASSES` gains it as a statement field
  and loses `priceToBook` / `bookValue` from the scored set once `compute_metrics` no longer reads them
  (`tests/test_pit_record.py` fails otherwise).
- `metric_lineage`: `LINEAGE["pb_ratio"]` inputs become market value and common equity, "how" says
  "market value / common shareholders' equity at the balance-sheet date"; `RECOMPUTE["pb_ratio"]` the
  same arithmetic; `EQUATIONS["pb_ratio"]` one exact template,
  `{marketCap|market value} ÷ {commonEquity|common equity}`. The equation bar in
  `tests/test_metric_lineage.py` met, not loosened.
- `run_context` `scoring_schema` bumped, so the scored cache is not reused.
- The bank Valuation paragraph in `run_screener.build_screener_overview` says how P/B is computed;
  regenerate, never hand-edit.
- A `METHODOLOGY_CHANGELOG.md` entry citing section 5 (MSCI, Damodaran, JPMorgan) and sections 2-4's
  measurements; expected effect as in section 6.

**What would make the reviewer refuse it, or change it:**

- A judgement that the newest book is better than a consistent one (MSCI's latest-book rule). The
  measured cost of consistency is small (median 0.3%, section 3); the benefit is that every bank input
  is on one period and checkable against the filings.
- A judgement that IBKR's market value should be fixed first. The proposal does not make IBKR worse in
  rank terms that matter (401st -> 413th, among the most expensive either way), and the market-value fix
  would correct its earnings yield, size and EV ratios too.
- If only Berkshire is to be fixed now, the minimum is the first row of section 6 - but that is a
  ticker-specific patch on a figure the run cannot check, which is how the error survived since March.

**What it does not do.** It does not touch bank ROE/ROA (their own open proposal), tangible book, the
bank weights, which stocks are bank-like, or Up-C market values.

## 9. Reproduce

```
python research/measurements/2026-10-19-bank-price-to-book.py sec   # network: refreshes the SEC JSON
python research/measurements/2026-10-19-bank-price-to-book.py       # offline: newest run + probe + caches
```

It needs `2026-10-19-bank-roe-roa-probe.json` (the source's balance sheets and `.info` on the morning of
2026-10-19); re-probing later will move the 18 newer-quarter stocks onto September statements and
change section 3's counts.

**Proposal status: ACCEPTED 2026-10-20** - built as proposed by the 06:00 session, which did not
write this note. It re-measured the two things the decision turned on, on the **2026-10-20** run and
a **fresh** probe rather than this note's (`research/measurements/2026-10-20-bank-pb-review.py`):
**availability** - "Common Stock Equity" present and positive for **46 of 46** bank-like stocks, at
the same balance-sheet date as the run for **46 of 46**, the run's `totalEquity` equal to the probe's
stockholders' equity within 0.5% for 46 of 46 - which is what separates this from the
earnings-definition proposal refused on 2026-10-16 (23 companies with no figure at all); and **the
stakes** - top 25 unchanged both ways, bank-like median move **1.5** places, **0** non-bank
composites changed, universe composite Spearman **0.9996**, BRK-B 65th -> 139th. Berkshire's
corrected ratio was checked against its own filing rather than another of the source's rows: filed
stockholders' equity of **$747.91bn** (equal to the probe's common equity and the run's
`totalEquity`) over a market value of $1,139.35bn is **1.5234**, against the published **0.001019** -
**1,494.8x**, the Class A-to-B ratio. `METHODOLOGY_CHANGELOG.md` 2026-10-20;
`tests/test_bank_price_to_book.py`; claim `overview.bank_price_to_book`. The note's two refusal
candidates were weighed and rejected: MSCI's latest-book rule against a measured 0.3% median cost of
consistency, and fixing IBKR's market value first against a rank move (379 -> 392) that changes
nothing a reader acts on. Sections 4's IBKR finding and 5's tangible-book question stay open.

**Original proposal, as raised 2026-10-19** - for bank-like stocks, `pb_ratio = marketCap / commonEquity` (new raw field `commonEquity` = the quarterly balance sheet's "Common Stock Equity" at `_stmt_date_balance_sheet`; NaN unless both are positive; no fallback to `priceToBook`, `bookValue` or stockholders' equity), shipped with `metric_lineage` LINEAGE / RECOMPUTE / EQUATIONS, `pit_record.FIELD_CLASSES`, a `scoring_schema` bump, the generated methodology text and a changelog entry (section 8). Measured on the 2026-10-19 run: top 25 unchanged, bank-like median move 2.5 places, BRK-B 60th -> 121st (its source P/B of 0.0010 is book per Class A share against a Class B price)
