# Growth: what stands behind 13% of the score (2026-10-12, owner-run)

**Question.** The Factor Lab's scorecard ranked this first of five (`plan/factor-lab.md`, "From the
lab to the weights"): Growth is the largest weight with nothing rerun behind it. What is the
documented case - in research and in how professionals build screens - for a Growth category, and
does the answer support 13%? Research-led per CLAUDE.md rule 4: no IC series, no backtest number,
no return of the screener's own ranking is used below.

**Finding.** There is no published return premium for growth as this category measures it. Read
one metric at a time, the evidence is flat to negative; index providers and factor managers treat
growth as a *style* (a description of a company) and leave it out of the factors they build
products to harvest. What the category does in this composite is something else, and nobody had
written it down: its scores run against Valuation, Investment, Size and Risk, so its 13 points
work as a counterweight - **the published ranking, taken whole, has a correlation of -0.01 with
the Growth score.** The screener spends 13% of its weight on Growth and ends up not preferring it.

**Shipped with this note:** the methodology page now says both things (generator). **Proposed, not
built:** Growth 13 -> 10, Quality 22 -> 25, for a second session to review - section 7 says why it
was not made tonight and what would be a good reason to refuse it.

---

## 1. What the category is, as built

| Metric | Share of Growth | Points of the composite |
|---|---|---|
| `forward_eps_growth` - expected EPS over the next 12 months against the last four quarters | 45% | 5.85 |
| `revenue_growth` - one year | 25% | 3.25 |
| `revenue_cagr_3yr` | 15% | 1.95 |
| `sustainable_growth` - ROE x retention | 15% | 1.95 |

The methodology's stated reason for the category is one sentence: forward EPS growth "gets the
most weight because it's forward-looking (the market prices in the future, not the past)". That
is an argument that growth matters to *value*, not that it predicts *returns*; a trait the market
already prices earns nothing for holding it. No study is cited for the category (Mohanram 2005 is
cited for the growth-trap flag, which is a different thing - section 3).

It is worth noticing what the four metrics are, taken together. MSCI's Value and Growth index
methodology (May 2023, sections 1.2.1 and 2.1) classifies a stock's growth style with five
variables: long-term forward EPS growth, short-term forward EPS growth, the current internal
growth rate (ROE x (1 - payout)), the long-term historical EPS growth trend and the long-term
historical sales-per-share growth trend. **The screener's Growth category is, nearly one for one,
MSCI's definition of the growth style** - a classifier of what kind of stock this is, used here as
a predictor of how it will do.

## 2. What the literature says, metric by metric

Everything in this section was read from the papers (PDFs fetched 2026-10-12; a second reader's
figures spot-checked against the files) except where marked second-hand.

**Does growth persist?** Chan, Karceski & Lakonishok, "The level and persistence of growth rates"
(*Journal of Finance* 2003; read as NBER w8282), all US stocks on Compustat, 1951-1997. Of firms
whose **sales** growth beat the median, 6.3% did so five years running against 3.1% by chance;
for **income before extraordinary items**, 3.0% against 3.1% (Table 3). Among large firms (top
two NYSE size deciles) the five-year earnings figure is 2.2% - *below* chance (Table 4, Panel D).
Analysts' long-term growth forecasts do rank next year's growth, but the top quintile was
forecast to grow 22.4% a year and realised 9.5% over five years (Table 8, 1982-1998). The paper
contains no return test. What it establishes is that ranking companies on earnings growth does
not find the companies whose earnings will grow.

**Past sales growth and returns.** Lakonishok, Shleifer & Vishny, "Contrarian investment,
extrapolation, and risk" (*Journal of Finance* 1994), NYSE and AMEX, portfolios formed each April
1968-1989, equal-weighted, held five years. Sorted on the rank of sales growth over the prior five
years, the lowest-growth decile averaged 19.5% a year and the highest 12.7% (Table I, Panel D);
size-adjusted, +2.2% against -2.4%. **Among the largest half of stocks, holding cash-flow-to-price
fixed, the high-growth third earned 2.9 to 5.8 points a year less than the low-growth third,
size-adjusted** (Table III). In the regression with book-to-market and cash-flow-to-price, sales
growth keeps a coefficient of -0.058, t -2.83 (Table IV). Conditions that differ from the screener:
the sort is on five-year growth across the market; the screener scores one- and three-year growth
within sectors; the sample ends in 1990 and is equal-weighted.

**Analysts' long-term growth forecasts and returns.** Bordalo, Gennaioli, La Porta & Shleifer,
"Diagnostic expectations and stock returns" (*Journal of Finance* 2019; read in the July 2018
draft), IBES long-term growth, deciles formed each December 1981-2015, equal-weighted. The
lowest-forecast decile earned 15% in the following year and the highest 3%; the spread is 13.6%
(t 2.22). Value-weighted it is 6.4% (t 1.2). Among big stocks (top three NYSE size deciles),
equal-weighted, 12.6% against 3.1%, with the shortfall concentrated in the three highest-forecast
deciles (Table B.3). The spread's monthly alpha is 1.16% against the market alone and **0.03%
(standard error 0.18) once value, momentum, betting-against-beta, profitability and investment are
in the regression** (Table B.2, Panel C). *Second-hand, from that paper's footnote 5:* La Porta
(1996) found a 20% spread over 1982-1991 and Dechow & Sloan (1997) 15%; neither paper could be
obtained tonight (both sat behind bot checks, which were not worked around).

**Analysts' one-year growth forecasts and returns - the nearest thing to the screener's 45%
metric.** Da & Warachka, "The disparity between long-term and short-term forecasted earnings
growth" (*Journal of Financial Economics* 2011), about 2,500 US firms a month, 1983-2006, sorted
within industries. Their short-term measure is this fiscal year's consensus EPS over last year's
actual, less one. In monthly cross-sectional regressions its coefficient is 0.0003 (t 1.46) with
book-to-market, size, past returns and long-term growth, and -0.0009 (t -0.81) once revisions and
earnings surprise are added (Table 3, Panel B); in the authors' words, neither forecast "predicts
returns". **In growth's favour:** in raw returns the high short-term-growth third beat the low
third by 0.30 to 0.48% a month within every long-term-growth group (Table 2) - but those stocks
also had far higher prior-year returns, and the difference is gone once momentum, revisions and
surprise are controlled. The screener already gives those three 23 points.

**Growth in profitability.** Asness, Frazzini & Pedersen, "Quality minus junk" (*Review of
Accounting Studies* 2019). Their quality has three parts, and one is growth - defined as the
five-year change in residual profits per share over lagged assets or book, not growth in sales or
forecast earnings. US, July 1957 to December 2016: excess return 0.17% a month (t 2.46), CAPM
alpha 0.16% (t 2.28), four-factor alpha 0.46% (t 8.29) with an HML loading of -0.49 (Table 4).
**This is the one positive published result with "growth" in its name**, and it is the weakest of
the paper's three components on raw return (profitability 0.25%, safety 0.23%). None of the
screener's four metrics is this measure; `sustainable_growth` is nearest in spirit and is a level,
not a change.

**Within growth stocks.** Mohanram, "Separating winners from losers among low book-to-market
stocks" (*Review of Accounting Studies* 2005; read in the April 2004 draft, so table numbers may
differ), lowest book-to-market quintile, 1979-1999. Low book-to-market firms grew sales 30.8% a
year against 17.3% for all firms and earned -6.0% size-adjusted in the following year. His eight
signals are profitability and cash flow (three), *stability* of earnings and of sales growth
(two), and R&D, capital spending and advertising (three). **The level of growth is not one of
them**; the paper motivates the stability signals by the market's habit of extrapolating growth.
High minus low score: 21.2% size-adjusted (t 11.07), mostly from identifying losers.

**From the lab** (`research/2026-10-12-factor-lab-findings.md`). The two published factors that
are about growth both run against it: the investment factor (companies growing assets slowly
minus quickly; 0.15% a month among big companies, t 1.46, 1963-2026) and value, whose short side
is the growth style (0.19%, t 1.56). The library has no sort on sales or earnings growth, so the
lab has nothing on the category's own metrics.

### Read together

| Metric | Nearest published evidence | Direction |
|---|---|---|
| `forward_eps_growth` (5.85 pts) | Da & Warachka: one-year forecast growth | None of its own; its raw effect is momentum and revisions, already scored |
| `revenue_growth`, `revenue_cagr_3yr` (5.2 pts) | Lakonishok, Shleifer & Vishny; Chan, Karceski & Lakonishok | Negative for five-year growth, including among large stocks with valuation held fixed; sales growth does persist somewhat |
| `sustainable_growth` (1.95 pts) | It is ROE x retention: Spearman 0.79 with ROIC in the published run | Positive, as profitability - which is Quality, counted again |

Controlling for the things the screener scores elsewhere, growth's own contribution is about
zero: Bordalo et al.'s alpha of 0.03% a month, Da & Warachka's insignificant coefficients, and
(section 3) MSCI's pure growth factor at under 1% a year.

## 3. What practitioners do

Read from the documents; file names and pages are in the session's source list (section 9).

- **MSCI treats growth as a style and a risk-model factor, not a rewarded one.** "Foundations of
  Factor Investing" (December 2013, p. 7): Growth and Liquidity "have not outperformed the market
  over long periods" and are not good candidates for long-term factor investing. Its six
  risk-premia factors are value, low size, low volatility, high yield, quality and momentum, and
  its factor index family and Diversified Multiple-Factor indexes (May 2022) are built on those;
  growth appears in the latter only as an exposure to be *constrained*.
- **In growth's favour:** MSCI FaCS Methodology (January 2018, Exhibit 2) reports the pure Growth
  factor's return in its global model's cross-sectional regressions, 1995-2016, as **0.89% a
  year, information ratio 0.80** - positive, and small beside momentum (4.38%, 1.24), earnings
  yield (3.64%, 2.09) and profitability (1.17%, 1.05). MSCI also publishes Growth Target indexes
  (August 2020) as a diversifier, with no claim of a premium.
- **S&P Dow Jones Indices.** The Quality, Value & Momentum multi-factor index (methodology December
  2025) averages those three scores; growth is not in it. The S&P 500 GARP index (September 2026)
  uses three-year EPS and sales growth as a *screen* (top 150), then selects on quality and value.
- **Dimensional** ("Dimensional at a Glance", 07/26): the drivers of stock returns it names are
  size, relative price and profitability. Growth is the low-expected-return end of relative price.
- **AQR:** growth in profitability inside quality, as above. *Not obtained:* any AQR or Research
  Affiliates statement on growth as a standalone factor.

Where literature and practice disagree: they do not, on the main point. The one soft
disagreement is sign at the margin - MSCI's regression gives a small positive pure-growth return,
Lakonishok, Shleifer & Vishny a negative one with valuation held fixed. Different decades,
different growth measures (MSCI's is 70% analysts' long-term forecast), different controls.
Neither is a premium worth 13 points.

## 4. What the category does inside this screener

`research/measurements/2026-10-12-growth-category-structure.py`, on the published run of
2026-10-12 02:00 (501 stocks). Its approximation of the composite reproduces the published
ranking exactly (Spearman 1.0000, 25 of the top 25).

| Growth's score against | Spearman |
|---|---|
| Valuation | -0.44 |
| Investment | -0.40 |
| Size | -0.35 |
| Risk | -0.27 |
| Momentum | +0.19 |
| Quality, Revisions | +0.16 each |

- The top fifth on Growth averages 34 on Valuation and 32 on Investment; the bottom fifth 66 and 67.
- `revenue_growth` has a Spearman of **+0.51 with `asset_growth`**, which the Investment category
  scores the other way. For most companies the two categories are arguing about the same fact.
- **The composite's correlation with the Growth score is -0.012.** With Valuation it is 0.58, with
  Quality 0.55, Momentum 0.35, Risk 0.34, Revisions 0.28, Investment 0.24, Size 0.13.
- The published top 25 averages 54.6 on Growth (universe 49.8), 75.6 on Valuation, 73.2 on Quality.

So the category is not selecting growth companies. It is cancelling part of the pull of four
other categories - 42 points of weight - toward companies that are not growing. That is a
defensible thing for a ranking to do, and it is what "growth at a reasonable price" products do
with a screen. It is not what the methodology page said the category was for.

## 5. Synthesis: how this fits the other seven

- **Valuation (22).** Growth is its counterweight. The documented complement to value is
  profitability, not growth (Asness, Frazzini & Pedersen's "quality at a reasonable price";
  Mohanram's and Piotroski's fundamentals-within-style tests), and the screener has Quality at 22
  and a value-trap flag doing that job already.
- **Investment (5).** Growth and Investment score the same fact in opposite directions about half
  the time. The lab's scorecard had asked whether Investment double-counts Valuation; the sharper
  question is that Growth *un*-counts Investment.
- **Momentum (13) and Revisions (10).** Whatever raw predictive power one-year forecast growth has
  is theirs (Da & Warachka). The 5.85 points on `forward_eps_growth` buy little that 23 points
  elsewhere have not.
- **Quality (22).** `sustainable_growth` is profitability a second time. The published research
  on what separates good growth companies from bad is entirely about quality and stability.

What the whole composite is implicitly betting on: value and quality, lightly momentum and low
risk, **and neutrality on growth**. If neutrality is the aim, the page should say so, and 13
points is the current price of it.

## 6. What changed tonight

The methodology's Growth section gains a paragraph, in the generator, saying what the research
says and what the category does (`tests/test_growth_category.py`). The lab's scorecard row for
Growth now says the research has been read. **No weight, metric or threshold changed**, so there
is no `METHODOLOGY_CHANGELOG.md` entry.

## 7. Proposed for review: Growth 13 -> 10, Quality 22 -> 25

**The argument.** Weight should sit where evidence is. Growth has none of its own; Quality is the
best-supported category among large companies in the lab (experiment 8) and in every rewarded-
factor set in section 3, and it is what the growth literature itself says to look at. The step is
3 points, the most the plan's reading rules allow at once.

**Measured effect on the 2026-10-12 run** (same script): rank correlation with today's ranking
0.997; 24 of the top 25 and 47 of the top 50 unchanged (WDC leaves, NTAP enters); the
composite's correlation with Growth goes from -0.01 to -0.06, with Valuation from 0.58 to 0.60,
with Quality from 0.55 to 0.58, with Momentum from 0.35 to 0.32.

**The case against, which a reviewer should weigh rather than skim.**

1. *It barely matters.* A change that moves one name in 25 is hard to tell from no change, and a
   methodology that changes for small reasons is harder to explain than one that does not.
2. *Neutrality may be the right design.* At 13 the ranking is growth-neutral almost exactly. At 10
   it leans slightly against growth, and at the same time slightly further toward value - whose
   own premium among big companies is the weakest-supported of the large categories (t 1.56 over
   1963-2026, about zero since 2014).
3. *A long-only holder keeps under half of profitability's premium* (experiment 8), so "move it
   to the best-supported factor" is less strong than it sounds, and what Quality is *for* is its
   own open question (question 5).
4. *The evidence against growth is mostly old and equal-weighted* (1968-1990; 1981-2015), and the
   one modern multi-factor estimate (MSCI, 1995-2016) is slightly positive.

**Why it was not made tonight.** Not for want of authority. Tonight's 02:00 run is the first
under the rebuilt momentum rule and several generator changes; a weight change in the same run
would make anything odd in it harder to read. And this note's author also ranked the question
and briefed its readers. A second session that reads the note cold is a cheap check on that.

**For the session that reviews it:** decide, and write the decision down either way. If you build
it: `config.yaml` `factor_weights`, `presets.py` (balanced), `schemas.py` defaults and the tests
that pin them; a `METHODOLOGY_CHANGELOG.md` entry citing this note; `python factor_lab.py --build`
(the lab prints the weights and a test fails until it is rebuilt); regenerate the overview;
`python scripts/rehearse_publish.py --full`; re-run the measurement script and quote the new
figures. Do **not** change the metrics inside Growth in the same change. If you refuse it, the
right outcome is a sentence on the methodology page saying the 13 points buy growth-neutrality
on purpose - which would itself be an improvement on today.

### Pre-flight, 2026-10-13 (so the reviewing session does not have to discover it)

The change was applied in a throwaway checkout - `config.yaml` `factor_weights`, the balanced
preset in `presets.py`, the `FactorWeights` defaults in `schemas.py`, nothing else - and the full
suite run before and after. **Eight tests trip, and each needs something different:**

| Test | What it needs |
|---|---|
| `test_config_validation.py::test_production_config_passes`, `::test_empty_config_uses_defaults` | The pinned 22 and 13 become 25 and 10. |
| `test_investor_profiles.py::test_the_regime_a_run_used_can_be_read_back_from_its_weights` | It hard-codes `22 + freed / 2`; read the weights from the config instead. |
| `test_factor_lab.py::test_the_scorecard_is_computed_and_carries_the_screeners_own_weights` | `python factor_lab.py --build` (the lab's front page prints the weights). |
| `test_overview_is_generated.py::...test_committed_overview_is_what_the_generator_produces` | Regenerate the overview from the generator. |
| `test_golden.py::test_golden_file` | `pytest tests/test_golden.py --regen`, after checking the composites moved the way the weights say. |
| `test_growth_category.py::test_no_weight_moved_with_the_note` | Skips itself once `METHODOLOGY_CHANGELOG.md` cites this note. Write the entry first. |
| `test_scoring.py::TestWeightSensitivity::test_small_perturbation_high_stability` | **Understand this one before touching it.** On its 50-stock fixture, a 1% weight perturbation leaves the top 20 only about half the same under the new weights. That is probably the fixture sitting on its threshold, but it is the suite's one direct test of how stable the ranking is, so find out which before changing the bar. Run `run_weight_sensitivity` on the real run at both weightings and put the result in the changelog entry. |

Three other tests fail in any clean checkout, before and after, because they read the gitignored
`runs/` directory; they pass in the working tree and are not part of this.

After the build: `python scripts/rehearse_publish.py --full`, then look at two or three
drilldowns whose rank moved.

## 8. Left open

- **Is revenue growth scored in the right direction at all?** The published evidence on sales
  growth is negative; the screener rewards it with 5.2 points and then penalises its close
  relative, asset growth, with 5. A within-category question for after the weight is settled.
- **Would growth in profitability belong in Quality?** It is the one growth measure with a
  published premium. The payload's `roe5` field suggests some of the history it needs is already
  fetched; that has not been checked.
- **`sustainable_growth`** duplicates Quality (0.79 with ROIC). It is also the only Growth metric
  pointing the way the evidence does, so removing it first would make the category worse.
- **La Porta (1996) and Dechow & Sloan (1997)** remain unread; the Wayback Machine has La Porta's
  PDF (`faculty.tuck.dartmouth.edu/images/uploads/faculty/rafael-laporta/Expectations.pdf`).

**What would change this reading:** a published result showing sector-relative one- to three-year
revenue growth, or 12-month forecast EPS growth, predicting returns among large companies after
value, momentum and profitability - none was found tonight, and the search was not exhaustive.

## 9. Sources

Read in full or in the relevant tables, 2026-10-12. Volume and page ranges of the journal versions
are from memory where the copy read was a working paper or draft.

* Chan, L., Karceski, J. & Lakonishok, J. (2003). The level and persistence of growth rates. *Journal of Finance* 58(2), 643-684. Read as NBER Working Paper 8282 (May 2001): Tables 3, 4, 8.
* Lakonishok, J., Shleifer, A. & Vishny, R. (1994). Contrarian investment, extrapolation, and risk. *Journal of Finance* 49(5), 1541-1578. Published version, Tables I-IV; one large-firm decile table from NBER Working Paper 4360.
* Bordalo, P., Gennaioli, N., La Porta, R. & Shleifer, A. (2019). Diagnostic expectations and stock returns. *Journal of Finance* 74(6), 2839-2874. Read in the draft of 10 July 2018: Table 1, Figure 1, Tables B.1-B.3.
* Da, Z. & Warachka, M. (2011). The disparity between long-term and short-term forecasted earnings growth. *Journal of Financial Economics* 100(2), 424-442. Tables 2, 3, 7.
* Asness, C., Frazzini, A. & Pedersen, L. (2019). Quality minus junk. *Review of Accounting Studies* 24(1), 34-112. Section 2 and Table 4.
* Mohanram, P. (2005). Separating winners from losers among low book-to-market stocks using financial statement analysis. *Review of Accounting Studies* 10, 133-170. Read in the draft of April 2004: sections 3-4, Tables 1, 3, 4.
* MSCI. Foundations of Factor Investing (Research Insight, December 2013), pp. 2, 7, Exhibit 3. MSCI FaCS Methodology (January 2018), p. 3 and Exhibit 2. MSCI Global Investable Market Value and Growth Index Methodology (May 2023), sections 1.2.1, 2.1. MSCI Diversified Multiple-Factor Indexes Methodology (May 2022), p. 3, Appendix IV. MSCI Growth Target Indexes Methodology (August 2020), p. 3, Appendix III.
* S&P Dow Jones Indices. S&P Quality, Value & Momentum Multi-factor Indices Methodology (December 2025). S&P GARP Indices Methodology (September 2026). S&P U.S. Style Indices Methodology (June 2026).
* Dimensional Fund Advisors. Dimensional at a Glance (07/26).

Second-hand only, not read: La Porta, R. (1996), Expectations and the cross-section of stock returns, *Journal of Finance* 51(5); Dechow, P. & Sloan, R. (1997), Returns to contrarian investment strategies, *Journal of Financial Economics* 43(1). Their spreads are quoted from Bordalo et al.'s footnote 5.
