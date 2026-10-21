# Bank Quality against bank Valuation: ROE is earnings yield times P/B, and that is not a double count (2026-10-20)

*One proposal is OPEN (section 8): **no weight or metric change** - state the bank P/B-ROE pairing
as the design on the methodology page and guard it with a test. Nothing in this note was built.*

**Question** (the Wednesday synthesis question the 2026-10-20 13:00 session left in `NIGHTLY_LOG.md`).
For the 46 bank-like stocks, `config.yaml`'s `bank_metric_weights` score:

| Metric | Category (22% each) | Within-category weight | Composite points | Direction |
|---|---|---|---|---|
| `earnings_yield` (net income / market value) | Valuation | 40% | **8.8** | higher is better |
| `pb_ratio` (market value / common equity) | Valuation | 60% | **13.2** | **lower** is better |
| `roe` (net income / average equity) | Quality | 35% | **7.7** | higher is better |
| `roa` (net income / average assets) | Quality | 25% | **5.5** | higher is better |
| `equity_ratio` (equity / assets) | Quality | 15% | **3.3** | higher is better |
| `piotroski_f_score`, `accruals` | Quality | 15%, 10% | 3.3, 2.2 | higher is better |

Since the two changes shipped on 2026-10-20, ROE = net income / equity = (net income / market value)
x (market value / equity) = **earnings yield x P/B**, inexact only by average-against-ending balances
and by the preferred stock in ROE's denominator. Is 35 points of bank Quality a double count of two
Valuation inputs?

**Answer, in one paragraph.** No - and the loose version of the complaint gets the sign wrong. ROE
duplicates earnings yield (both rise with earnings) but it **opposes** price-to-book (a high ROE comes
with a high P/B, and P/B is scored lower-is-better). Measured on the published run, bank Quality and
bank Valuation correlate **-0.18**, ROE's percentile and the Valuation score **-0.50**; the Valuation
inputs explain **51%** of bank Quality's variance *jointly* but **2.5%** through the Valuation score,
because the two inputs enter Quality with opposite signs to the ones they carry in Valuation. Put on
three independent log coordinates (E/P, P/B, B/A), the weights in the table add up to **22 composite
points on earnings yield, 0.0 on P/B and 8.8 on the equity ratio** to first order - equivalently, 22
points on ROE and 22 against P/B: **an equal-weighted "high ROE at a low P/B" screen**, which in logs
*is* earnings yield. Measured on the scores the engine actually produced, the 44 points rank the bank
set **-0.890** (Spearman) against the residual of the P/B-ROE line - at least as close to that
classic combined signal as any of the six alternatives rescored here, including the two that build
the residual explicitly (the nearest, -0.883). That is the construction practice documents: Damodaran, Wilcox, Nissim's study of insurer valuation and sell-side bank desks all read P/B *conditional on* ROE, and MSCI's Diversified Multiple-Factor indexes add book-to-price, earnings yield and profitability into one score with no carve-out for financials. **So no weight or metric changes.** What is wrong is the
explanation: the methodology page calls P/B "THE key bank valuation metric" and ROE "the key bank
profitability metric" as if they were two facts, when the 44 points net them into one, and the zero
net P/B loading is an arithmetic coincidence of four separately chosen weights that the next
reweight of any of them would break without anyone noticing. That is the proposal (section 8).

Two findings that are not the question, recorded for a later session (section 7): earnings yield is
ranked against all **76** Financials while the other four bank metrics are ranked against the **46**;
and the third coordinate, equity per dollar of assets, is mostly *which kind of financial* a company
is - insurers carry a median 24.8% against banks' 10.4%, and **9 of the 10** best-ranked bank-like
stocks are insurers.

Everything is measured on the run the live site publishes - **`7c12b43a07b6`, started 2026-10-20
13:21**, the first carrying both of the day's changes (`dashboard_data.js`'s `run_timestamp` is that
run's `start_time`) - and is reproducible from two committed scripts (section 10). No IC, return,
backtest or research-engine number is used anywhere (CLAUDE.md rules 4 and 5).

---

## 1. Method

- **`research/measurements/2026-10-20-bank-overlap-measure.py`** (JSON beside it) reads the run's raw
  fetch, its metric table and its scored table for the 46 bank-like stocks: the identities (section
  2), the correlations and the variance of Quality the Valuation inputs explain (section 3), the net
  exposure (section 4), the contrast with non-bank stocks, and the sub-type breakdown (section 7). It
  asserts that each bank's Quality score rebuilds from its five weighted percentiles (46 of 46).
- **`research/measurements/2026-10-20-bank-overlap-rescore.py`** (JSON beside it) rescores the same
  metric table with the engine's own functions - `factor_engine.compute_sector_percentiles`,
  `apply_percentile_transform`, `compute_category_scores`, `compute_composite`, `rank_stocks` - and
  the run's own `config.yaml` and `effective_weights.json`, changing only the bank weight table (or,
  for two candidates, the value in the `pb_ratio` column). **The baseline reproduces 501 of 501
  published ranks** or the script stops. Every candidate leaves **0 of 455** non-bank composites
  changed, as it must.
- All 46 bank-like stocks carry all seven bank metrics on this run, so the bank table's weights are
  the effective ones for every one of them (no renormalisation for a missing metric).

## 2. The identities, on the engine's own constructions

| Identity | Exact on | How it holds on the scored figures |
|---|---|---|
| ROE = earnings yield x P/B | ending **common** equity | Spearman(ROE, EY x P/B) **0.973**; ratio median **1.013**, 10th-90th percentile 0.897-1.074; within 10% for 37 of 46 |
| ROA = ROE x equity ratio | ending total equity and assets | Spearman **0.996**; ratio median 1.007; within 5% for 35 of 46 |

The gap in the first line is ROE's denominator (*average total* equity, including preferred) against
P/B's (*ending common* equity): the ratio is ending common over average total. Preferred is a median
2.6% of equity (at most 12.6%). The largest gaps are banks whose equity moved over the year (FITB
1.16, HBAN 1.11) or that carry most preferred (KEY 0.88, SCHW 0.88, GS 0.89). Regressing log ROE on
log EY and log P/B gives coefficients **0.994 and 1.025** with **R-squared 0.978** - so to within two
percent of its cross-sectional variance, a bank's ROE is fixed once its earnings yield and its P/B
are.

## 3. Correlations, and how much of Quality the Valuation inputs explain

Raw values, Spearman, across the 46:

| | ROE | ROA | EY | P/B | Equity ratio | Valuation score | Quality score |
|---|---|---|---|---|---|---|---|
| **ROE** | 1 | 0.60 | 0.14 | **0.73** | 0.15 | **-0.54** | 0.79 |
| **ROA** | | 1 | 0.38 | 0.18 | 0.84 | -0.04 | 0.90 |
| **EY** | | | 1 | -0.46 | 0.41 | 0.69 | 0.34 |
| **P/B** | | | | 1 | -0.24 | **-0.95** | 0.35 |
| **Valuation score** | | | | | | 1 | **-0.18** |

(ROE against P/B, 0.73, is comparable to the correlation above 0.70 Damodaran reports for U.S. banks in February 2009.)

- **Jointly, the Valuation inputs explain 51.1% of bank Quality's variance** (Quality score on the EY
  and P/B percentiles), and **82.8% of ROE's percentile**. ROA's percentile, by contrast, only 34.6%:
  ROA carries the third coordinate (section 4).
- **Through the Valuation score, 2.5%.** The reason is in the coefficients: Quality loads **+0.80**
  on the EY percentile and **-1.04** on the cheap-P/B percentile. Valuation loads both positively. So
  a bank that is cheap on book scores high on Valuation and low on Quality *for the same reason* - its
  equity earns little - and the two categories mostly offset rather than add.
- **Where bank Quality's variance comes from** (each metric's weighted percentile against the score):
  ROE 42.8%, ROA 33.6%, equity ratio 14.5%, accruals 6.1%, Piotroski 3.0%.
- **For contrast**, Valuation against Quality across the 454 non-bank stocks is **+0.09** (by sector,
  -0.21 in Consumer Staples to +0.44 in Utilities). Banks are the one group where the two categories
  are tied by an accounting identity; Consumer Staples' -0.21 has no identity behind it.

Two stocks make it concrete. **Citigroup** has the cheapest book of the four largest U.S. banks (P/B 1.07, the
90th percentile) and an ROE of 8.4% (the 5th): Valuation 82.3, Quality 14.0, rank 219. **Progressive**
is the reverse - P/B 3.74 (the 10th percentile), ROE 34.9% (the 95th): Valuation 35.6, Quality 88.3,
rank 153. Each category on its own would tell a student something misleading about either company.
Read together they say what the P/B-ROE literature says: Citigroup's book is cheap because it earns
little on it, and Progressive's is dear because it earns a great deal.

## 4. What the 44 points net to

Every ratio in the bank table is built from four company-level amounts - earnings E, market value P,
book equity B, assets A. Dividing out scale leaves three independent log coordinates:

> x1 = log E/P (earnings yield), x2 = log P/B, x3 = log B/A (equity ratio),
> with log ROE = x1 + x2 and log ROA = x1 + x2 + x3 (exact on ending balances).

Multiplying each metric's composite points by its direction and adding them up coordinate by
coordinate gives the first-order position of the bank Valuation + Quality block:

| Coordinate | EY (+8.8) | P/B (-13.2) | ROE (+7.7) | ROA (+5.5) | Equity ratio (+3.3) | **Net** |
|---|---|---|---|---|---|---|
| x1 = log E/P | +8.8 | | +7.7 | +5.5 | | **+22.0** |
| x2 = log P/B | | -13.2 | +7.7 | +5.5 | | **0.0** |
| x3 = log B/A | | | | +5.5 | +3.3 | **+8.8** |

The same position read in the bank analyst's coordinates (log ROE, log P/B, log B/A): **+22.0 on
ROE, -22.0 on P/B, +8.8 on the equity ratio.** Both rows describe one function: rewarding ROE and
penalising P/B with equal weight in logs *is* rewarding earnings yield, because ROE / (P/B) = E/P.

This is first-order: the engine adds percentile ranks, not logs, and the three coordinates have
different spreads (log standard deviations 0.50, 0.63, 0.71) and are correlated (E/P with P/B
-0.70). So it was measured on the scores the engine actually produced:

| Block regressed on the percentile ranks of x1, x2, x3 | E/P | P/B | B/A | R-squared |
|---|---|---|---|---|
| Valuation + Quality block (status quo) | **+0.284** | **-0.067** | **+0.220** | 0.906 |
| *on standardised logs, per sd* | *+8.65* | *+1.00* | *+6.12* | *0.741* |
| Valuation score alone, per sd | +4.05 | **-17.92** | +0.16 | 0.800 |
| Quality score alone, per sd | +13.25 | **+19.92** | +12.08 | 0.882 |

Valuation alone is a P/B score (-0.95 with P/B); Quality alone is long P/B through ROE (+19.9 per
sd); added, they leave P/B a small fraction of the loading E/P carries - and the residual sign is
not even stable between the rank and the log measurement. **The 60-point P/B weight does not survive
into the composite as a bet on cheap book.** What survives is earnings yield, and - with almost as
much weight - equity per dollar of assets.

**The coincidence.** -13.2 + 7.7 + 5.5 is exactly zero because four weights chosen separately
(`pb_ratio` "THE key bank valuation metric", `roe` "THE key bank profitability metric", `roa` "Key
bank efficiency metric", and Valuation and Quality at 22 each) happen to sum that way. No comment,
note or changelog entry records it as a design. Moving ROE from 35 to 25, or Valuation from 22 to 25,
would move the bank block onto a P/B position nobody chose - the same kind of residue the
revenue-growth / asset-growth note found in Growth (`research/2026-10-19-revenue-growth-vs-asset-growth.md`).

## 5. What the literature and practice say

Sources were read this session from free copies (PDFs with `pypdf`); each is marked **read**,
**secondary** (the original was not obtained and a named secondary source was read instead) or
**abstract only**. Page numbers are the documents' own unless marked "PDF p". Paraphrased unless
quoted.

### 5.1 The theory: P/B and ROE are one valuation, and the identity is known

- **The justified price-to-book** - Damodaran, *Investment Valuation* (2nd ed.), ch. 19, "Book value
  multiples", PDF pp. 6-7 (**read**): from the dividend discount model, P/BV = ROE x payout / (r - g),
  which with g = (1 - payout) x ROE is **P/BV = (ROE - g) / (r - g)**. Inverted for the return the
  price implies, with no growth, r = ROE / (P/B) = **E/P**: in this framework, "ROE relative to P/B"
  and earnings yield are one quantity, and growth is what separates them.
  https://pages.stern.nyu.edu/~adamodar/pdfiles/valn2ed/ch19.pdf
- **Wilcox (1984), "The P/B-ROE valuation model", *Financial Analysts Journal* 40(1)** (**secondary**:
  the CFA Institute abstract, and Damodaran ch. 19, PDF p. 23, which reproduces the fit). On 949
  Value Line stocks in 1981, log(P/B) = -1.00 + 7.51 x ROE; the abstract reports the model beats
  price-earnings models on mean squared error. Wilcox & Philips (2005) was **not obtained**.
  https://rpc.cfainstitute.org/en/research/financial-analysts-journal/1984/the-pb-roe-valuation-model
- **Mercer Capital (Jeff K. Davis, CFA), *Shareholder Value Drivers* (2020), slides 21-22** (**read**,
  a bank-valuation firm's own deck): states the identity as the profession uses it - P/TBV equals
  ROTE times P/E - alongside ROE = ROA x leverage, and reads a persistently low P/TBV as a sign of
  depressed profitability rather than of cheapness.
  https://mercercapital.com/content/uploads/Shareholder-Value-Drivers.pdf
- **Penman (1996), *Journal of Accounting Research* 34(2)** (**secondary**: CFA Digest, Spring 1997,
  p. 11): P/B reflects expected future profitability; P/E reflects current ROE and is a poor guide to
  growth. Not relied on beyond that.

### 5.2 Practice: P/B is read *conditional on* ROE - deliberately

- **Damodaran, "Valuing Financial Service Firms" (2009)**, pp. 30-33 (**read**): ROE is the
  determinant with the biggest effect on a bank's P/B, the two correlate above 0.70 across U.S. banks
  (February 2009), and his worked example values a bank by regressing P/B on ROE (and price
  volatility) across 18 peers and reading the residual as over- or under-valuation.
  https://pages.stern.nyu.edu/~adamodar/pdfiles/papers/finfirm09.pdf
- **Damodaran, ch. 19, pp. 18-19 and 30-31** (**read**): the "mismatch" screen - low P/BV with high ROE
  - as a matrix or a sector regression, and a test on NYSE stocks 1981-90 in which the bottom-quartile
  P/BV *and* top-quartile ROE portfolio beat the opposite one by about 15 points a year. That is a
  **conjunctive** screen on all stocks, with no risk adjustment - it shows the signal is meant to be
  joint, not that it pays among large banks today.
- **Nissim (2013), "Relative valuation of U.S. insurance companies", *Review of Accounting Studies*
  18, 324-359** (**read**, published version; 372 insurers, 1990-2011): price-to-book conditioned on
  ROE is the standard in insurance and banking (p. 326), done by regressing P/B on ROE across peers
  and using the fitted value (p. 338, which also quotes a 2010 Citigroup report on Lincoln National
  doing exactly that); conditioning on ROE significantly improves valuation accuracy. Accuracy here
  means closeness to the current price - it documents practice, not a return premium.
  https://business.columbia.edu/sites/default/files-efs/imce-uploads/dnissim/publications/Doron%20Nissim%20-%20Relative%20Valuation%20of%20U.S.%20Insurance%20Companies%20published%20version.pdf
- **Sterne Agee (Todd Hagerman), "Bank Weekly", republished 2014-03-03** (**secondary**
  republication): banks below the P/TBV-on-ROTCE regression line are the buying list, provided ROTCE
  has a reasonable basis to improve; it flags banks more than one standard deviation below each
  regional line. https://hedgefundalpha.com/stocks/banks-correlation-rotce-ptbv/
- **ECB, *Financial Stability Review*, May 2019, Box 5, pp. 2-3** (**read**, a regulator's panel of 70
  euro-area and U.S. banks, 2000-2018): profitability raises P/B and capital lowers it; the model
  explains about half the variation.
  https://ecb.eu/pub/financial-stability/fsr/focus/2019/pdf/ecb~a18359e8d9.fsrbox201905_05.pdf

**So "high ROE at a low P/B" is a deliberate, combined signal** wherever banks are valued for a living.
It is built as a *residual* (P/B against the P/B-ROE line) or a *conjunction*, never as two separately
rewarded virtues.

### 5.3 Factor construction: value and profitability are *added*, and the point is that they offset

- **Novy-Marx (2013), "The other side of value", *Journal of Financial Economics* 108** (**read**,
  accepted draft, https://www.johnhcochrane.com/s/Novy_marx_OSoV.pdf): gross profitability and B/M
  correlate -18% (PDF p. 10), so each signal is polluted by the other; within profitability quintiles
  the value spread is 0.68% a month against 0.41% unconditionally (PDF pp. 16-17). His large-cap
  strategy **adds** each stock's rank on profitability to its rank on B/M across the 500 largest
  stocks (PDF p. 18). Financials are excluded from the main sample, though he reports that keeping
  them changes little (PDF p. 7), and his profitability is gross profit over assets, not ROE.
- **Fama & French (2006), *JFE* 82** - **not obtained**; the argument is restated in the authors'
  five-factor paper (2013 draft, PDF p. 3, **read**): from the valuation equation, holding book and
  market value fixed, higher expected earnings imply a higher expected return. Whether the 2006
  sample excludes financials could not be checked, so nothing here rests on it.
- **MSCI, *Diversified Multiple-Factor Indexes Methodology* (May 2022)**, pp. 4 and 11 (**read**):
  one score that adds Value (one third book-to-price, two thirds earnings yield) and Quality (which
  includes profitability) at 0.25 each, both sector-relative, with **no rule specific to
  financials**. This is the closest institutional analogue to the screener: E/P, B/P and a
  profitability measure added in one composite, deliberately.
  https://www.msci.com/eqb/methodology/meth_docs/2_MSCI_Diversified_Multi-Factor_Indexes_Methodology_20220524.pdf
- **MSCI, *Enhanced Value Indexes Methodology* (June 2017)**, p. 4 (**read**): for Financials the
  value score is forward P/E and P/B only (no EV-based ratio) - the screener's bank Valuation pair.
- **MSCI, *Quality Indexes Methodology* (May 2025)** (**read**, a copy saved by an earlier session):
  ROE, debt-to-equity and earnings variability, with no financials-specific rule.
- **S&P DJI's quality methodology could not be read** (every spglobal.com URL refused the download);
  nothing here relies on it.

### 5.4 The one reading that cuts the other way, and the thin bank-return evidence

- **Penman & Reggiani (2013), "Returns to buying earnings and book value", *Review of Accounting
  Studies*** (**read**, SSRN 1536618 working version; 1963-2006, financials *included*, excluded only
  as a robustness check): this is the identity tested on returns. Holding E/P fixed, a higher
  book-to-price - equivalently a *lower* ROE - came with *higher* subsequent returns, which they
  attribute to the risk of expected growth, not to mispricing. On that reading, the screener's ~zero
  net P/B loading at a given E/P (section 4) leaves out a priced dimension. It is an all-stock,
  risk-based result; it does not say the dimension pays among large banks, and it says that if it
  pays, it pays for bearing risk.
- **Bank-specific cross-sections, abstracts only** (no free full texts): Barber & Lyon (1997, *JF*
  52(2)) find the size and B/M relation similar for financial and non-financial firms; Cooper, Jackson
  & Patterson (2003, *JBF* 27) find neither B/M nor size important for banks, while earnings, leverage
  and loan-loss reserves predict returns; Viale, Kolari & Fraser (2009, *JBF* 33) find no evidence
  that size or B/M is priced in bank stocks; Bessler, Drobetz, Erdmann & Zimmermann (2007, Basel WWZ
  discussion paper; 235 European banks) find book equity over total assets loads *negatively* on later
  returns. **No paper found tests ROE itself as a return predictor among banks.** The bank evidence
  on either leg is thin and mixed, and none of it was read in full.

### 5.5 Where they agree and disagree

**Agree:** P/B and ROE are one valuation; practitioners deliberately read P/B *against* ROE; factor
builders deliberately *add* value and profitability because they lean against each other. Both say
the overlap is the design, not an accident to be removed.

**Disagree:** on what the combination should leave over. Practice's residual, on this run, is
mostly earnings yield (section 6). Penman & Reggiani say that, at a given earnings yield, cheap book
earns more for risk reasons - a reason to *keep* some net P/B tilt. The bank-only evidence cannot
settle it. That is a research question about the size of one loading (section 8, "Next"), not a
defect in the construction.

## 6. The bank ranking with the overlap removed

Each candidate rescored on the published run's metric table with the engine's own functions (section
1). "Bank-set rho" is the Spearman correlation of the 46 bank-like stocks' order with the status quo;
"vs P/B-ROE residual" is the block's Spearman correlation with the residual of log P/B on log ROE
across the 46 (the Wilcox line, fitted on this run: slope 0.84, R-squared 0.40) - negative means the
block rewards banks that are cheap *for their ROE*.

| Candidate | Bank-set rho | Median move (places) | Moved 25+ | Top 25 | Bank-like in top 25 / 100 | Net loading on ranks (E/P, P/B, B/A) | vs P/B-ROE residual |
|---|---|---|---|---|---|---|---|
| **S0 status quo** | 1 | 0 | 0 | - | 5 / 12 | **+0.284, -0.067, +0.220** | **-0.89** |
| C1 drop ROE (35 renormalised over the rest) | 0.968 | 24.5 | 23 | unchanged | 5 / 18 | +0.213, **-0.210**, +0.287 | -0.85 |
| C2 drop P/B (Valuation = EY alone) | 0.928 | 30.5 | 24 | **-ES -NEM -SWK, +CINF +GL +HIG** | 8 / 16 | +0.502, **+0.218**, +0.226 | -0.72 |
| C3 ROE's 35 moved to ROA | 0.971 | 18.0 | 14 | unchanged | 5 / 15 | +0.236, -0.156, +0.337 | -0.84 |
| C4 drop ROE and ROA | 0.928 | 41.5 | 31 | **-ACGL -ALL -TRV, +DECK +DLTR +WDC** | 2 / 17 | +0.172, **-0.308**, +0.198 | -0.79 |
| C5 P/B replaced by P/B relative to ROE; ROE kept | 0.954 | 21.5 | 21 | -ES -NEM -SWK, +CINF +GL +HIG | 8 / 13 | **+0.508**, +0.156, +0.266 | -0.84 |
| C6 C5 with ROE dropped | 0.978 | 20.0 | 18 | -ES -NEM -SWK, +CINF +GL +HIG | 8 / 14 | **+0.437**, +0.013, +0.333 | -0.88 |

What each candidate actually does:

- **Removing ROE (C1) or both profitability ratios (C4) does not remove a duplicate - it removes the
  conditioning.** P/B's net loading triples (C1) or quintuples (C4), turning the bank block into an
  unconditional cheap-book bet, the construction the literature in section 5 says is improved by
  controlling for profitability. C4 also drops three insurers out of the top 25. The stock it rewards
  most is the textbook case of a book that is cheap for a reason: **AIG** (P/B 1.05, the 95th
  percentile; ROE 7.2%, the 1st) rises **205th -> 98th** under C1 and **91st** under C4; Citigroup
  219th -> 189th and 171st.
- **Removing P/B (C2) flips the sign**: the block becomes long *expensive* book (+0.218), because ROE
  and ROA still rise with P/B and nothing now leans against them. Three insurers enter the top 25.
- **Building the Wilcox residual explicitly (C5, C6)** gets no closer to the combined signal than the
  status quo (-0.84 and -0.88 against -0.89), because on this run the residual *is* mostly earnings
  yield (Spearman -0.93 between them): making P/B "relative to ROE" turns bank Valuation into a second
  earnings-yield score and roughly doubles E/P's loading. It also changes the top 25. And it puts a
  regression fitted on each run's 46 stocks inside a score, which a student cannot check from the
  page and which moves when any one bank's figures move.
- **Moving ROE's weight to ROA (C3)** is the mildest change (top 25 unchanged, median 18 places), but
  it buys a P/B tilt (-0.156) and more of the third coordinate (+0.337), which section 7 shows is
  largely the insurer/bank split. Nothing in the literature read asks for either.

## 7. How it fits the rest of the screener

### 7.1 Intended or accidental? Accidental in how it was written, intended in what it does

Nothing in `config.yaml`, the methodology page or the changelog describes the bank metric set as a
P/B-ROE screen. The comments call P/B "THE key bank valuation metric" and ROE "THE key bank
profitability metric" - two virtues, weighted separately. The zero net P/B loading of section 4 is
the arithmetic of four independently chosen weights.

But what the weights *do* is the construction both bodies of practice in section 5 build on purpose:

- the valuation desks' P/B-against-ROE reading - the 44 points rank the bank set **-0.89** against the
  residual of the P/B-ROE line, as close as either candidate that builds that residual explicitly (-0.84, -0.88);
- the index builders' additive value-plus-profitability score - MSCI's Diversified Multiple-Factor
  index adds book-to-price, earnings yield and profitability in one composite, with no carve-out for
  financials.

So the double weight is not a duplicate to be removed. It is the mechanism that keeps the bank set
from being an unconditional cheap-book bet (C1, C4) or an expensive-book one (C2). The defect is that
it is undocumented and unguarded.

### 7.2 Against the rest of the screener

- **Banks are the one place Valuation and Quality are tied by construction** (-0.18, against +0.09
  for the 454 other stocks), because only banks carry a price-over-capital ratio (P/B) in Valuation.
  For other stocks, Valuation is a set of yields on cash flow, operating profit, earnings and sales,
  and Quality is a set of returns on capital and assets that contain no price. No identity ties the
  two categories together, so nothing cancels.
- **A bank's composite rests on one number far more than any other stock's does.** Net income is
  the numerator of earnings yield, ROE and ROA alike: 8.8 + 7.7 + 5.5 = **22.0 composite points of
  first-order exposure to log net income**. For a non-bank, earnings yield carries 4.4 points, beside
  three other yields built on cash flow, EBITDA and sales.
  - `research/2026-10-15-earnings-definition.md` kept net income as reported, one-off charges
    included, and the refusal stands on its own reasons.
  - This note adds one consequence: through the metrics built on net income, a one-off charge moves a
    bank's composite roughly five times as much as the same proportional charge moves a non-bank's
    (first order; Piotroski and accruals, which both stock types carry, aside).
  - The earnings-yield row's one-off-charge note (claim `drilldown.special_item_note`) is therefore
    worth more on a bank's drilldown than anywhere else. A page-side question, not a weight one.
- **Every category outside Valuation and Quality is unaffected.** Growth's `sustainable_growth` uses
  the same ROE, but only as an input to growth, at 15% of a 13-point category.

### 7.3 The third coordinate is mostly *which kind* of financial a company is

Section 4's second-largest loading is log B/A, equity per dollar of assets. It carries 8.8 points
(ROA 5.5 + equity ratio 3.3), and measured on ranks (+0.220) it is nearly as large as earnings
yield's (+0.284). Across the 46, though, that coordinate is largely the business model:

| Sub-type | n | Median equity / assets | Median ROA | Mean Quality score | In the 10 best-ranked bank-like |
|---|---|---|---|---|---|
| Insurers | 17 | **24.8%** | 4.12% | **61.9** | **9** |
| Banks | 14 | 10.4% | 1.07% | 33.9 | 1 (MTB) |
| Brokers, asset managers, credit | 15 | 7.5% | 1.33% | 45.8 | 0 |

The sub-type dummies alone explain:

- **41%** of log equity ratio;
- **36%** of bank Quality;
- **38%** of the Valuation + Quality block.

An insurer funds its investments with float and equity, while a bank funds its loans with deposits, so
ranking their equity ratios and ROAs in one percentile pool largely rewards *being an insurer*. The
one bank-specific study found on capital, Bessler et al. (2007), points the other way: equity over
assets loads negatively on later returns. Only its abstract was read.

This is a **peer-group** question - should balance-sheet ratios be ranked within sub-type? - not part
of the overlap question, and it touches `research/2026-10-09-bank-like-financials.md`. Recorded under
Next.

### 7.4 Earnings yield is ranked against a different pool from the other four bank metrics

A bank's P/B, ROE, ROA and equity ratio are ranked against the 46 bank-like stocks, because no other
stock carries them. Its earnings yield is ranked against all **76** Financials, 30 of which are
payment networks, exchanges, data vendors and asset managers with half the banks' median earnings
yield (4.1% against 8.1%).

So the banks' EY percentiles bunch toward the top: standard deviation **23.0** against about 29 for a
uniform ranking of 46. That is part of why bank Valuation behaves as a P/B score (+4.05 per sd on E/P
against -17.92 on P/B, section 4).

The methodology page's summary says each metric is ranked "within its sector, so a bank is compared
with banks". For earnings yield that is true of the sector and loose about the peers. This is a
claims-audit item for the next pass over the page (`research/2026-10-19-methodology-claims-audit.md`),
not a scoring proposal. It is a different question from this note's, and ranking EY within the 46
would also change what the 30 non-bank Financials are compared with.

### 7.5 Why a guard and not just a sentence

The 2026-10-20 changes are the reason this question exists. They put P/B and ROE on nearly one book
on the same day, and that tightened the identity (R-squared 0.978) without anyone choosing the
consequence.

The next change to any of the four weights would move the bank block's net P/B position, and nothing
would say so. The revenue-growth / asset-growth note recorded the same pattern in Growth: a net
position that is "luck, not design" and had already moved once when a correctness fix landed.

`tests/test_growth_category.py` is the precedent for the remedy. It holds the ranking to the
growth-neutral position the page describes and fails when a weight moves without the paragraph
being rewritten.

## 8. Conclusion and the proposal

**No change to any weight, metric or threshold.** Thirty-five points of bank Quality on ROE are not a
double count of two Valuation inputs: they duplicate earnings yield and cancel price-to-book. The
44 points net to an equal-weighted "high ROE at a low P/B" screen, which is the construction bank
analysts, insurer valuation research and MSCI's multi-factor indexes all use on purpose. Every way of
"removing the overlap" rescored here makes the bank set either an unconditional cheap-book bet, an
expensive-book bet, or a second copy of earnings yield. It changes the top 25 in four of the six
candidates and moves a median 18-42 bank-like places in all of them. None gets closer to the
documented combined signal than the status quo does.

What should change is that the design is written down and held in place. The proposal, for a later
session to decide (do not build it in the session that reads this note first):

**Proposal status: ACCEPTED 2026-10-21** (13:00 session, which did not write this note) - built as
specified in (1), (2) and (4), with **one change to the guard in (3) that the review measured**: the
failing condition is the *measured* loading at criterion (b)'s bar, not the first-order figure at
+-1.5 points. See section 11 for the decision and what it added. Original proposal, as written:

*As proposed (status line rewritten above; the brief's detector is anchored to the line start, so
this copy is prose):* no weight or metric change; state the bank P/B-ROE pairing as the design and guard it: (1) a paragraph in `run_screener.build_screener_overview()`'s "Bank-specific scoring" section saying a bank's ROE is its earnings yield times its P/B, so its Valuation and Quality scores are read together - the 44 points rank it on ROE relative to P/B (to first order, its earnings yield) and on equity per dollar of assets, with no net weight on P/B by itself - registered in `claims.py` as `overview.bank_pb_roe_identity`; (2) the two table cells calling P/B and ROE "THE key" bank metrics reworded to say each is read against the other; (3) `tests/test_bank_pb_roe_identity.py`, in that claim's `checked_by`, failing if the first-order net loading on log P/B computed from `config.yaml` (factor weights x bank weights: -P/B + ROE + ROA, in composite points; 0.0 today) leaves +-1.5 points, or if on the published payload log ROE regressed on log earnings yield and log P/B has R-squared below 0.90 across the bank-like stocks (0.978 today); (4) a `config.yaml` comment above `bank_metric_weights` naming that constraint and this note.

**Pre-flight** (measured on run `7c12b43a07b6`; the deciding session should re-measure on its own run):

- **Rank movement: none.** No weight, input or formula changes, so **0 of 501** composites move, the
  top 25 and every bank rank are unchanged, `scoring_schema` stays at 14, and the payload's numbers
  stay byte-identical. Only the methodology text changes, in `SCREENER_OVERVIEW.md` and the
  methodology dialog the generator embeds.
- **Tests it would trip, and what each needs:**
  - `tests/test_overview_is_generated.py`: the committed `SCREENER_OVERVIEW.md` must be regenerated
    from the generator, never hand-edited (rule 10).
  - `tests/test_claims_register.py`: the new claim must name a `made_true_by` that exists, a `detect`
    pattern that matches the generated text (the typography pass turns " - " into an en dash, as the
    two bank claims beside it allow for), and `checked_by` tests that exist.
  - `scripts/check_published_claims.py`: its membership is derived from `claims.py`, so the new module
    joins the data loop's publish gate automatically. It must run offline in seconds and read only the
    committed payload and `config.yaml`.
  - `tests/test_large_company_evidence.py`: the paragraph must not quote an all-stock premium. That
    rules out Damodaran's 1981-90 portfolio spread and Penman & Reggiani's return table.
  - Run `python scripts/rehearse_publish.py`, because the diff touches the generator.
- **Refutation criteria, written before anyone builds it:**
  - **(a) The identity sentence is false.** If, on the run it would ship with, the R-squared of log
    ROE on log earnings yield and log P/B is below 0.90, the paragraph must not ship. A change to how
    ROE, earnings yield or P/B is built (an earnings basis, a tangible-book P/B) is the likely cause.
  - **(b) "No net weight on P/B" is false.** If the measured loading on P/B's rank (section 4; this
    note's measure script) exceeds half of earnings yield's in absolute value, the sentence must give
    the measured figure instead. Today it is -0.067 against +0.284.
  - **(c) The no-change conclusion is refuted.** Reopen the weights if a research note, having read
    Penman & Reggiani and the bank-specific cross-sections (Cooper et al. 2003, Viale et al. 2009,
    Baek & Bilson 2015, Barber & Lyon 1997) in full, finds that at a given earnings yield cheap book
    earns more among large U.S. banks and insurers. A deliberate net P/B tilt would then be a
    candidate, and C3 (ROE's 35 points to ROA) is the mildest route measured: top 25 unchanged,
    median move 18 places, P/B loading -0.156. That is a change of *position*, made through the
    lab-to-weights route, one change at a time - not a fix for a double count.

**Next - questions this note raises and does not answer:**

1. **Peer group:** should the bank-like set's balance-sheet ratios be ranked within sub-type? Today,
   9 of the 10 best-ranked bank-like stocks are insurers, and 41% of the equity-ratio coordinate is
   sub-type (section 7.3). A Monday note, starting from `research/2026-10-09-bank-like-financials.md`
   and Bessler et al. (2007).
2. **Claims audit:** "so a bank is compared with banks" is loose for earnings yield, which is ranked
   against all 76 Financials (section 7.4).
3. **Page:** the earnings-yield row's one-off-charge note matters about five times more on a bank
   than on a non-bank (section 7.2). Check that it shows on the bank drilldowns where a one-off
   exists.

## 9. What this note does not establish

- **Nothing here is evidence that the combined bank signal earns a return.** It establishes what the
  weights *are*, and that what they are matches the construction documented in section 5. Whether
  that construction has paid among large U.S. banks since publication is a question for a lab
  experiment, not for this note, and no backtest or IC figure was consulted (rules 4 and 5).
- **One run.** The identities are accounting, so they hold on any run; the measured loadings and the
  -0.89 depend on this run's cross-section. The proposal's guard (section 8) is what keeps them true.
- **Tangible book.** The bank analyst's own pairing is P/TBV with ROTCE. The 2026-10-20 P/B review
  rejected tangible book on measurement (the source's row agrees with the filings for 22 of 41), and
  the identity is the same on either basis, so nothing here depends on that choice.

## 10. Reproduce

```
python research/measurements/2026-10-20-bank-overlap-measure.py
python research/measurements/2026-10-20-bank-overlap-rescore.py
```

Both read the newest full run under the main checkout's `runs/` (gitignored, so a worktree reads the
main checkout's) with `run_context.latest_run_dir`, and write the JSON beside themselves. The rescore
script stops unless its baseline reproduces every published rank.

---

## 11. The decision (2026-10-21, 13:00 session - not this note's author)

**ACCEPTED.** The no-change conclusion is confirmed, the two refutation criteria hold on a
different run, and items (1), (2) and (4) shipped as specified. **The guard in item (3) was
changed**, because the review measured it and found it could not see the thing it was for.

### 11.1 Re-measured on a different run

Everything above was measured on run `7c12b43a07b6` (2026-10-20 13:21). The deciding session
re-ran both scripts on run **`4dc4acf1d780`** (2026-10-21 02:00 - the one the live site was
serving), whose baseline also reproduces **501 of 501** published ranks:

| | Note's run | This run |
|---|---|---|
| **(a)** R-squared, log ROE on log E/P and log P/B | 0.978 | **0.978** |
| **(b)** measured loading on P/B's rank | -0.067 | **-0.069** |
| measured loading on E/P's rank (criterion (b)'s bar is half of it) | +0.284 | **+0.283** |
| S0 against the P/B-ROE (Wilcox) residual | -0.890 | **-0.892** |
| C1 / C2 / C3 / C4 / C5 / C6 against it | -0.85 / -0.72 / -0.84 / -0.79 / -0.84 / -0.88 | **-0.848 / -0.724 / -0.839 / -0.796 / -0.838 / -0.882** |

Every figure reproduces to the third decimal, and the status quo is still the closest of the seven
constructions to the combined signal the literature documents. Nothing in sections 1-10 needed
correcting; the committed JSONs are refreshed to this run.

### 11.2 What the review added: the proposed guard was blind to C3

The proposal specified **one** failing condition - the first-order net loading on log P/B computed
from `config.yaml` (`-pb_ratio + roe + roa`, in composite points; 0.0 today), outside +-1.5 points.
That is exact algebra on the three log coordinates, but **the engine ranks percentiles**, so the
review measured both quantities for every candidate
(`research/measurements/2026-10-21-bank-pb-roe-guard-review.py`, on run `4dc4acf1d780`):

| Candidate | First-order net on log P/B | Proposal's guard | Measured loading on P/B's rank | Criterion (b) |
|---|---|---|---|---|
| **S0 status quo** | 0.0 | passes | **-0.069** | holds |
| C1 drop ROE | -4.7 | FAILS | -0.211 | REFUTED |
| C2 drop P/B | +13.2 | FAILS | +0.216 | holds |
| **C3 ROE's 35 -> ROA** | **0.0** | **passes** | **-0.156** | **REFUTED** |
| C4 drop ROE and ROA | -13.2 | FAILS | -0.308 | REFUTED |
| P1 ROE 35 -> 25, equity ratio 15 -> 25 | -2.2 | FAILS | -0.119 | holds |

**C3 is the hole, and it is not a corner case - it is the change section 8's own refutation
criterion (c) names as "the mildest route measured" to a deliberate P/B tilt.** It reads exactly
0.00 first-order, identically to the status quo, and it must: ROE and ROA both enter log P/B with
coefficient +1, so moving weight between them cannot move the first-order figure *at all*. On the
scores the engine produces, its loading on P/B's rank is **-0.156 against the status quo's -0.069 -
2.3 times as large, and past criterion (b)'s own bar of 0.141.** The mechanism is that ROA also
carries log B/A, which correlates **-0.34** with log P/B across the bank set. So the proposal's
guard would have let a reweight through while the page went on saying "no net weight on P/B by
itself" and the measurement in section 4 said otherwise.

**P1 shows the same guard erring the other way:** a reweight nobody proposed trips the +-1.5 band
at -2.2 points while leaving the page's sentence measurably true. Fitted across the candidates,
+-1.5 points is about **+-0.03** on the measured loading - roughly a fifth of criterion (b)'s bar.
So the band was both too loose where it mattered and too tight where it did not.

### 11.3 What shipped instead

`tests/test_bank_pb_roe_identity.py` (11 tests, in the data loop's publish gate through
`claims.py`'s `overview.bank_pb_roe_identity`):

1. **The identity**, criterion (a): R-squared of log ROE on log E/P and log P/B across the
   bank-like set of the committed payload, floor **0.90**, plus both coefficients within 0.2 of 1.
2. **No net P/B position, criterion (b) - the failing condition**: the Valuation + Quality block's
   loading on P/B's percentile rank must stay under **half** of its loading on earnings yield's.
   This is the test C3 trips.
3. **The first-order check is kept as well**, at a quarter of the E/P net (5.5 points today)
   rather than +-1.5 - a band wide enough not to fire on a reweight that leaves the claim true,
   and narrow enough that the words "no net weight" cannot survive it.
4. **The page's printed arithmetic is held to `config.yaml`.** The whole section is generated from
   the weights, so the three net-points figures cannot go stale - and if a reweight ever moves the
   block off neutral, the generated sentence **states the measured tilt instead** of claiming
   neutrality. A false page is the failure mode a sentence-with-a-constant has; this one cannot
   have it.
5. The two "THE key" cells, the `config.yaml` CONSTRAINT comment, the note's own status line and
   the review measurement's committed JSON are each held in place by a test.

The general lesson, which is CLAUDE.md 0.11's: **a claim whose check tests a proxy instead of the
assertion is an unchecked claim.** The first-order figure is the page's *explanation*; the measured
loading is the page's *claim*. Guard the claim.

### 11.4 What was not done, and why

- **No weight, metric, threshold or input changed**, so **0 of 501** composites move and the
  payload's numbers are byte-identical. Only the methodology text changes.
- **The three questions under section 8's "Next" stay open** - the bank-like peer group
  (9 of the 10 best-ranked bank-like stocks are insurers), "so a bank is compared with banks"
  being loose for earnings yield, and the one-off-charge note's extra weight on a bank drilldown.
  None is part of the overlap question and each needs its own session.
- **Criterion (c) is untouched.** Reopening the weights still requires a note that has read
  Penman & Reggiani and the bank-specific cross-sections in full. This decision makes the current
  position explicit; it does not endorse it as the only defensible one.
