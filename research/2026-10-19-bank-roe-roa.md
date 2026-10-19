# Bank ROE and ROA: what the source's ratios are, and what the screener should score instead (2026-10-19)

*Proposal open (status line at the end): score bank-like ROE and ROA on the run's own net income over
average equity and average total assets, section 7. For a later session to decide; no scoring code
was changed.*

**Question (from an audit).** For the 46 bank-like stocks, 60% of the Quality category - ROE 35 and
ROA 25 of `config.yaml`'s `bank_metric_weights.quality` - is read straight from Yahoo's
`returnOnEquity` and `returnOnAssets` (`.info`; `factor_engine.py` around line 2908). At Quality's 22%
that is **13.2 composite points per bank-like stock** resting on two ratios nobody here has a
definition for. The engine dropped `earningsGrowth` for being exactly that ("a black-box input with no
verifiable definition", `factor_engine.py` near line 2427). This note works out what the two ratios
are, what the literature and practice say a bank ROE and ROA should be, and what scoring a defined
version would move.

**Finding, in four lines.**

1. **The ROE is reconstructible, the ROA mostly is not, and a third of both are on a different
   quarter from everything else the run scores.** Where Yahoo's ratio and its statements are on the
   same quarter (28 of 46), its ROE is *consolidated* net income (including the minority holders'
   share) over the average of *total* equity (including minority interests) at the quarter end and a
   year earlier - 25 of 28 to within 0.5%. Its ROA follows the same construction for 11 of the 12
   non-insurers and for **none of the 16 insurers** (2-28% off). For the other 18 (15 banks and
   brokers, among them JPM, BAC, C, WFC, GS and MS, and 3 insurers) Yahoo's ratios already reflect
   the September quarter while the statements the run scores end in June or March.
2. **Regulators, banks and index providers all define these ratios; none defines them as Yahoo does.**
   The FDIC and FFIEC use net income over *average* assets and *average* equity; JPMorgan reports net
   income to common over average common equity; MSCI and AQR use earnings over *latest* book equity
   that excludes minority interests and preferred stock. Yahoo's minority-interest-inclusive basis is
   the outlier, and it is what moves the three names the audit flagged (KKR, APO, IBKR).
3. **The fix needs no new data:** the run already fetches net income, equity and total assets at the
   quarter end and a year earlier, and they agree with the companies' filings: rebuilt from SEC XBRL,
   the proposed ROE and ROA agree within 2% for **38 of 39** comparable stocks (Yahoo's: 23 and 15 of
   39).
4. **The effect is small and concentrated where the source's definition is odd.** Rescored on the
   2026-10-19 run: composite rank correlation 0.9997, **top 25 unchanged**, bank-like median move 5
   places; the largest are IBKR 401st -> 435th, STT 139th -> 168th, PRU 170th -> 142nd.

---

## 1. Method

`research/measurements/2026-10-19-bank-roe-roa.py` (JSON beside it).

- **The run:** the newest full run, `377ec40838b1` (2026-10-19 02:00, 501 scored), its raw fetch
  (`00_raw_fetch.parquet`) and its own metric table (`02_outliers_flagged.parquet`).
- **The probe** (`... probe`, network, saved as `2026-10-19-bank-roe-roa-probe.json` so everything
  else re-runs offline): for each of the 46 bank-like stocks at 08:31 the same morning, Yahoo's
  `.info` ratios and `mostRecentQuarter`, and every quarterly and annual statement row an ROE or ROA
  could be built from - net income attributable to the company, to common shareholders and including
  minority interests; stockholders', common and total equity; preferred stock; total assets. The
  `.info` ratios at 08:31 equal the run's 02:00 values in **90 of 90** comparisons (45 stocks x 2;
  Morgan Stanley has neither), so the probe describes what the run scored.
- **The filings:** the weekly SEC `companyfacts` cache (`data/sec/pit/facts.parquet`), rebuilt for the
  quarter each stock's balance sheet ends with `sec_crosscheck.sec_value` - trailing twelve months of
  `NetIncomeLoss`, `StockholdersEquity` and `Assets` at that date and a year earlier.
- **The stakes:** the run's metric table rescored with the run's own `config.yaml` and
  `effective_weights.json` - **the baseline reproduces all 501 published ranks** - with only the
  bank-like stocks' `roe` / `roa` replaced. (The 2026-10-15 stakes scripts re-ran `compute_metrics`
  without the market series, which loses beta and Jensen's alpha and reproduces 20 of 501 ranks; this
  one starts after `compute_metrics` to avoid that.)

## 2. The audit's comparison, reproduced on the newest run

Against net income over the average of the two balance sheets the run already fetches
(`totalEquity`, `totalEquity_prior`, `totalAssets`, `totalAssets_prior`, a year apart):

| | Pairs | Rank correlation with Yahoo | Within 0.5% | Within 2% | Sector-percentile moves > 10 points |
|---|---|---|---|---|---|
| ROE | 44 | 0.978 | 18 | 24 | 7 - KKR 20.1, C 12.9, STT 12.3, PNC 12.2, PRU 11.8, CINF 11.5, APO 10.5 |
| ROA | 44 | 0.936 | 8 | 15 | 5 - **IBKR 52.0**, **KKR 29.0**, APO 13.0, MS 11.9, CFG 11.4 |

The audit's figures (0.982 and 0.94; KKR 19.6, IBKR 52.2, KKR 30.4) are reproduced to within a
run's drift. **One correction to the audit:** Allstate's 46.1% is not a mystery number. It is net
income over *average* equity, to four decimal places (0.4611 both ways); the 39.5% it was compared
with is over *ending* equity, and Allstate's equity grew by about 40% over the year. That is the
first clue to what the source computes.

Two stocks are already scored on a second definition: Yahoo serves **no ROE or ROA for Morgan
Stanley**, so the engine's fallback scores it on net income over *ending* equity and assets inside
the same percentile rank as 45 stocks on Yahoo's construction - the mixed-basis problem Phase 13
(F19) removed from earnings yield. And `pit_record.market_fields` (backtest v2) rebuilds
`returnOnEquity` as net income over ending equity, a third construction.

## 3. What Yahoo's ratios are - measured

No published definition exists: neither Yahoo nor `yfinance` documents the `.info` fields, and a
search for Yahoo's own definition of "Return on Equity (ttm)" found only other vendors' glossaries
(one of which averages *five* quarters of *common* equity - not what Yahoo's numbers do, below). So
the definition was found by trying every candidate - three numerators (net income attributable to the
company, to common shareholders, including minority interests) x three equity bases (stockholders',
common, total including minority interests) and total assets x four denominators (ending, beginning,
two-point average a year apart, five-quarter average) and the latest fiscal year - against each
stock's ratio.

**18 of 46 match no construction consistently (the best candidate fits 3 of 17), for a reason that
matters on its own:** Yahoo's `mostRecentQuarter` for them is **2026-09-30** while the statements
Yahoo serves - and the run scores - end in June (or March for BAC, BNY, TFC, CINF). They are 15 banks
and brokers (BAC, BNY, C, GS, IBKR, JPM, MS, MTB, PNC, RF, SCHW, STT, TFC, USB, WFC) and three
insurers (CINF, PGR, TRV). Their ROE and ROA already reflect a quarter the run's net income, equity,
equity ratio and earnings yield do not contain, and that neither the statements nor the SEC cache
can yet confirm. **For over a third of the bank set, 13.2 composite points come from a different
period than the rest of the stock's record.**

For the 28 where the quarters agree:

| Ratio | Construction that fits | Insurers (16, incl. AMP, PFG) | Everyone else (12) |
|---|---|---|---|
| ROE | **net income incl. minority interests / average of total equity incl. minority interests, now and a year earlier** | 14 within 0.5% (misses BRK-B 21%, HIG 6%) | 11 within 0.5% (miss COF 3.5%) |
| ROA | **net income incl. minority interests / average total assets, now and a year earlier** | **0** within 0.5%, 1 within 2% (AIG); the rest 7-28% off | 11 within 0.5% (miss COF 3.5%) |

Where the fit is exact it is exact to the fourth decimal (KKR, APO, ALL, AFL, HOOD all 0.0000), which
is the evidence it is the formula and not a coincidence. For insurers the ROA tracks the *latest
fiscal year's* net income over its average assets more closely than anything trailing (AFL 0.5%, ALL
2.4%, PFG 2.7%) but not exactly; whatever it is, it is not the same period as the insurer's own ROE.

**What that does to the audit's three outliers.** KKR, Apollo and Interactive Brokers consolidate
entities they only partly own (KKR's and Apollo's funds; IBKR's Up-C structure, under which about
three quarters of the operating company belongs to its founders' holding company). Including the
minority holders' profit and equity makes Yahoo's ROE a return on *everyone's* capital in the
consolidated group rather than the shareholder's: KKR 7.3% against 10.6% to its own shareholders.
For ROA it is the other way round: IBKR's 2.28% is consolidated profit over consolidated assets; the
profit belonging to IBKR's shareholders is 0.53% of those assets.

## 4. What the literature and practice define - read this session

**Regulators: net income over *average* assets and equity.**

- **FDIC, *Quarterly Banking Profile*, Notes to Users (second quarter 2026), p. 6 (definitions) and
  p. 2 (method).** Return on assets is "bank net income (including gains or losses on securities and
  extraordinary items) as a percentage of average total (consolidated) assets"; return on equity the
  same net income "as a percentage of average total equity capital". The averaging: asset and
  liability figures "represent average amounts for the period (beginning-of-period amount plus
  end-of-period amount plus any interim periods, divided by the total number of periods)" - for a
  twelve-month ratio on quarterly reports, a five-point average.
  https://www.fdic.gov/quarterly-banking-profile/qbp-notes-users-second-quarter-2026.pdf
- **FFIEC, *UBPR User's Guide*, Summary Ratios - Page 1 (updated 2026-08-20), item 16 (UBPRE013).**
  "Net income after securities gains or losses, extraordinary gains or losses, and applicable taxes
  divided by average assets", annualised (`PCTOFANN(RIAD4340, UBPRD659)`). The same page has a
  separate line for "net income attributed to minority interests divided by average assets" (item
  14), i.e. the headline ratio's numerator is the bank's own net income, not the consolidated group's.
  https://www.cdr.ffiec.gov/CDRDownload/CDR/UserGuide/v186/FFIEC%20UBPR%20User%20Guide%20Summary%20Ratios--Page%201_2026-08-20.PDF

**Banks themselves: average equity, common basis.**

- **JPMorgan Chase, Form 10-K for 2022, MD&A, "Explanation and Reconciliation of the Firm's Use of
  Non-GAAP Financial Measures" (p. 59 of the filing).** "ROE: Net income* / Average common
  stockholders' equity", "ROA: Reported net income / Total average assets", with * "net income
  applicable to common equity"; ROTCE divides the same numerator by average tangible common equity.
  https://www.sec.gov/Archives/edgar/data/19617/000001961723000231/jpm-20221231.htm

**Index providers and factor research: latest book equity, excluding minority interests and preferred.**

- **MSCI, *Quality Indexes Methodology* (May 2025), Appendix I.** ROE "is calculated using the trailing
  12 month earnings per share figure and latest book value per share", applied to every stock, with
  no carve-out for financials (since May 2025 ROE is mandatory for a stock to enter the index).
  https://www.msci.com/indexes/documents/methodology/2_MSCI_Quality_Indexes_Methodology_20250520.pdf
- **MSCI, *Fundamental Data Methodology* (March 2021), sections 1.2.5 and 2.2.2 (pp. 16, 21).** Book
  value is "shareholders' equity available to shareholders at the latest period end date excluding
  minority interest, treasury shares and preferred shares that do not exhibit equity like
  characteristics"; ROE is computed only if book value is positive and "the book value date is equal
  to or older than the earnings date" - i.e. never on a newer balance sheet than the earnings, the
  mismatch section 3 found in 18 Yahoo ratios.
  https://www.msci.com/eqb/methodology/meth_docs/MSCI_Fundamental_Data_Methodology_Mar2021.pdf
- **Asness, Frazzini & Pedersen (2019), "Quality minus junk", *Review of Accounting Studies* 24(1),
  34-112, Appendix (profitability).** "ROE is net income divided by book-equity" (IB/BE) and "ROA is net
  income divided by total assets" (IB/AT), with book equity defined as "shareholders' equity minus
  preferred stock"; no averaging is described. Their sample is "all available common stocks"; no
  exclusion of financials is described.
  https://research-api.cbs.dk/ws/portalfiles/portal/60211462/lasse_heje_pedersen_et_al_quality_minus_junk_publishersversion.pdf

**Bank valuation: ROE is P/B's companion.**

- **Damodaran (2009), "Valuing Financial Service Firms", NYU Stern working paper, pp. 30-31.** Of the
  determinants of price-to-book, "the return on equity has the biggest impact", and the relationship
  "should be stronger for financial service firms than for other firms, because the book value of
  equity is much more likely to track the market value"; for U.S. banks in February 2009 the
  correlation between P/B and ROE on trailing twelve-month earnings was above 0.70. His worked
  valuations multiply ROE by the *beginning* book value of equity (p. 24-25). He also warns (p. 6)
  that book-based ratios of financial firms measure a partly marked-to-market equity, not the capital
  originally invested. https://pages.stern.nyu.edu/~adamodar/pdfiles/papers/finfirm09.pdf

**Where they disagree, and why.** Two axes:

- **Average or latest equity.** Regulators and the banks' own reporting average, because a bank's
  income is earned on the balance sheet it carried *through* the year; index providers use the latest
  book because they compute ROE from per-share data at one date for thousands of stocks and pair it
  with P/B on the same book. Both are coherent for their purpose. A screener showing a club the ROE
  that JPMorgan's own earnings release prints is closer to the first.
- **Whose equity.** Every source above excludes minority interests from the equity a return is
  measured on (MSCI says so in terms; the FDIC/UBPR use the bank's own net income, with minority
  interests on a separate line; JPMorgan and AQR go further and exclude preferred too). **None uses
  Yahoo's consolidated, minority-inclusive basis.**

Not read this session, and not relied on: Wilcox (1984) on the P/B-ROE model, Nissim's work on
insurer valuation (which would bear on whether insurers' equity should exclude unrealised bond gains
and losses), and the Federal Reserve's BHCPR user's guide.

## 5. The trade-offs, and how it fits the rest of the screener

**Average or ending.** Measured on today's run, the choice barely moves the ranking (section 6), but
it matters for single names whose equity moved: Allstate's ROE is 46.1% on average equity and 39.5% on
ending. The average is the regulators' and the banks' convention, it is what Yahoo itself uses where it
can be reproduced (so it changes least), and it is already the screener's own ROE in sustainable
growth (`factor_engine.py`, "ROE uses average equity (current + prior / 2)"), so **the screener would
have one ROE, not two** - the lineage registry currently has to warn readers that "this ROE is not the
Yahoo ROE shown for banks". The five-quarter average the FDIC uses moves almost nothing further (it
ranks the bank set's ROE at 0.997 and ROA at 0.999 against the two-point average, and leaves the top
25 unchanged) and needs three more balance sheets per stock, so two points a year apart is the
teachable choice.

**Common or total equity (preferred stock).** 22 of the 46 carry preferred stock worth more than 5% of
equity (up to 12.6%: KEY, STT, SCHW). Net income to common over average common equity is what banks
and analysts quote and what MSCI's book value is. But measured on the probe it ranks the bank set
almost identically (rank correlation 0.987 with the total-equity basis; median ratio 1.008), needs two
fields the run does not fetch, and the source's common rows are the less reliable ones - Apollo's
"Net Income Common Stockholders" is 23% of its net income, which collapses its ROE to 2.3% and is a
data artefact, not economics. The total-equity basis - the company's net income, before preferred
dividends, over its stockholders' equity including preferred - is the FDIC's "total equity capital"
basis, uses inputs the run already fetches and that item 0.14 checks against the filings (equity
97.9% within 0.5%, net income 94.3%), and keeps numerator and denominator on the same claim.

**Minority interests.** The proposal measures both ratios on the company's own net income
(`netIncome`, attributable to its shareholders) and its own stockholders' equity, as every source in
section 4 does. For ROA the denominator is consolidated assets, as the FDIC's "average total
(consolidated) assets" and JPMorgan's "total average assets" are. **This is the one judgement call the
reviewer should look at:** for an Up-C company like IBKR, shareholders' net income over consolidated
assets (0.53%) understates how productively the consolidated balance sheet is used (2.28% on group
profit). The regulators' convention is the proposal; the alternative (net income including minority
interests over assets) needs one new fetch field and changes ROA for the three structurally unusual
names only. Whether KKR, Apollo and IBKR belong on the bank metric set at all was settled separately
(`research/2026-10-09-bank-like-financials.md`).

**One-off items.** Net income here includes gains and charges, like the FDIC's and UBPR's ("including
gains or losses on securities and extraordinary items"). That is also the decision for earnings yield
(`research/2026-10-15-earnings-definition.md`, refused 2026-10-16): the bank set's ROE, ROA and
earnings yield would then share one numerator, and the earnings-yield row's one-off-item note already
says what is inside it.

**How it sits beside the bank valuation metrics.** Banks score on earnings yield (net income / market
value) and P/B in Valuation, and on ROE in Quality. With ending equity these are tied by an identity -
ROE = earnings yield x P/B, on the same book value - so the composite rewards a bank for high ROE at a
low P/B, which is the P/B-ROE screen Damodaran describes. Average equity loosens the identity slightly;
it does not change what the bank set is implicitly betting on. Whether 35 points of bank Quality on a
ratio partly determined by two Valuation inputs is a double count is a synthesis question for a
Wednesday, not part of this proposal.

**Generic Quality, for contrast.** ROIC uses NOPAT over *ending* invested capital and Piotroski's ROA
*beginning*-of-year assets (as its paper does). The bank set is not made inconsistent with them by
averaging: they are different metrics with their own papers' definitions, and bank ROE follows its
own sector's.

## 6. The stakes - rescored on the 2026-10-19 run

Everyone else unchanged; baseline = the published ranking (501 of 501 reproduced).

| Bank ROE / ROA on | Composite rank correlation | Bank-like median move | Moved 10+ | Moved 25+ | Largest composite move | Top 25 |
|---|---|---|---|---|---|---|
| **Average equity / assets, a year apart (proposed)** | **0.99973** | **5 places** | 9 | 3 | IBKR -3.2 pts | **unchanged** |
| Ending equity / assets | 0.99963 | 8 | 19 | 3 | IBKR -4.3 | unchanged |
| Five-quarter average | 0.99974 | 6.5 | 11 | 2 | IBKR -3.2 | unchanged |
| Common basis (net income to common / avg common equity) | 0.99956 | 7.5 | 15 | 3 | BRK-B -4.8 (a data artefact) | unchanged |

Under the proposal: **IBKR 401st -> 435th** (ROA 2.28% -> 0.53%, section 3), **STT 139th -> 168th**,
**PRU 170th -> 142nd**, TFC 239th -> 220th, C 195th -> 214th, PNC 156th -> 175th, COF 175th -> 158th,
MS 443rd -> 431st (onto the same definition as everyone else). The best-ranked bank-like stocks are the
same five (EG 6th, ALL 10th, AIZ 19th -> 18th, TRV 21st -> 22nd, ACGL 24th -> 21st). KKR's ROE rises
(7.3% -> 10.6%) and its ROA falls (1.39% -> 0.79%), leaving it 477th.

**Berkshire Hathaway loses both ratios** under any averaged construction, because the run has no
balance sheet a year earlier for it: Yahoo's quarterly sheet omits September 2025, so the fetch's
"column four" is March 2025 and its 330-400-day check correctly rejects it (`totalEquity_prior` and
`totalAssets_prior` are NaN - which already leaves its asset growth unscored today).
Its two weights redistribute within bank Quality and it moves 60th -> 56th. The fix belongs in the
fetch - pick the year-ago column by date, not by position - and is a separate, general change.
*(Made later on 2026-10-19: `factor_engine._year_ago_position`; changelog "the year-ago period was
taken by column position". From the next run BRK-B has its June 2025 balance sheet, so this
paragraph's "loses both ratios" no longer applies to it.)*

## 7. The proposal, for a later session to decide

**Change.** For bank-like stocks only (`_is_bank_like`), in `factor_engine.compute_metrics`:

```
roe = netIncome / ((totalEquity + totalEquity_prior) / 2)    # both > 0 required, else NaN
roa = netIncome / ((totalAssets + totalAssets_prior) / 2)    # both > 0 required, else NaN
```

with no fallback to Yahoo's `returnOnEquity` / `returnOnAssets` and no fallback to an ending balance,
so every bank-like stock is ranked on one construction (the Phase 13 F19 rule). `netIncome` is the
trailing four quarters the run already sums for earnings yield.

**Before it can ship** (CLAUDE.md rows 0.8 and 0.8c - a scoring change ships with its frontend):

- `metric_lineage`: `LINEAGE["roe"/"roa"]` inputs become net income, equity (or assets) now and a year
  earlier, and the "how" text says "net income / average of ... now and a year earlier"; `RECOMPUTE`
  `_roe` / `_roa` the same arithmetic; `EQUATIONS` one exact template each, e.g.
  `{netIncome|net income} ÷ (({totalEquity|equity} + {totalEquity_prior|equity a year earlier}) ÷ 2)`;
  the sustainable-growth caveat "This ROE is not the Yahoo ROE shown for banks" removed (it becomes the
  same ROE). `tests/test_metric_lineage.py`'s equation bar met, not loosened, and its sparse-input
  allowlist updated.
- `pit_record`: `returnOnEquity` / `returnOnAssets` leave `FIELD_CLASSES["market"]` once
  `compute_metrics` no longer reads them, and `market_fields` stops building them
  (`tests/test_pit_record.py` line ~100 asserts the old construction).
- `run_context` `scoring_schema` 11 -> 12, so the scored cache is not reused.
- The methodology text in `run_screener.build_screener_overview` (the bank Quality paragraph) says how
  ROE and ROA are computed and that they are on average balances; regenerate, never hand-edit.
- A `METHODOLOGY_CHANGELOG.md` entry citing section 4's sources (FDIC, UBPR, JPMorgan, MSCI, AQR,
  Damodaran) and section 3's measurement; expected effect as in section 6.

**What would make the reviewer refuse it, or change it:**

- The minority-interest choice for ROA (section 5): if the reviewer judges that consolidated-group
  profit over consolidated assets is the better efficiency measure for Up-C and fund-consolidating
  names, the change becomes "ROA on net income including noncontrolling interests", which needs one
  new raw field (`Net Income Including Noncontrolling Interests`, its own `FIELD_CLASSES` entry).
- Evidence that the run's `totalEquity_prior` / `totalAssets_prior` are wrong for bank-like stocks at
  scale (re-measure: section 2 of the JSON, `C_filings`, currently 38 of 39 within 2% of the filings).
- A preference for the common-equity basis that is worth two new fetch fields and the source's less
  reliable common rows (section 5).

**What it does not do.** It does not touch P/B (also an `.info` field, and also on Yahoo's newest
quarter for the 18 stocks in section 3 - a definition that is at least simple, price over book per
share, but the same period question; worth its own look), the bank metric weights, or which stocks
are bank-like.

## 8. Reproduce

```
python research/measurements/2026-10-19-bank-roe-roa.py probe   # network: refreshes the probe JSON
python research/measurements/2026-10-19-bank-roe-roa.py         # offline: newest full run + probe + SEC cache
```

The probe reflects Yahoo on the morning of 2026-10-19; re-running it later will move the 18 "newer
quarter" stocks onto September statements and change section 3's counts, which is the point of
keeping the snapshot.

**Proposal status: OPEN** - for bank-like stocks, `roe = netIncome / ((totalEquity + totalEquity_prior) / 2)` and `roa = netIncome / ((totalAssets + totalAssets_prior) / 2)` (NaN unless both balances are positive; no fallback to Yahoo's `returnOnEquity` / `returnOnAssets` or to an ending balance), shipped with `metric_lineage` LINEAGE / RECOMPUTE / EQUATIONS, `pit_record.FIELD_CLASSES`, `scoring_schema` 12, the generated methodology text and a changelog entry (section 7)
