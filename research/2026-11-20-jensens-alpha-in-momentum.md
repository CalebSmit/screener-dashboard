# Jensen's alpha in the Momentum category: what it adds, and what the literature would put there instead (2026-11-20)

**Question.** `jensens_alpha` carries 25 of the Momentum category's 100 points (3.25 of the composite's
100 at Momentum's 13%). The 2026-11-19 fresh-eyes review (`research/2026-11-19-fresh-eyes-review.md`
section 5) measured it at +0.87 against `return_12_1` and noted that, unlike the category's other two
metrics, it does not skip the latest month. Nothing in `DECISIONS.md` decided that design. This note
asks what the metric is, whether a published source supports it as a momentum signal, and what a
source-backed construction would be. The owner's queue carries it (`OWNER_FOCUS.md` 2026-11-19, item 3:
"a note on Jensen's alpha").

**Short answer.** As built, the metric is the 12-month total return with a small beta tilt: its
cross-sectional ranking is 0.98 the plain 12-month return. No source this note found uses a same-window
CAPM alpha, unskipped, as a stock-selection momentum signal. The construction the literature does
support for "momentum after removing market exposure" is **residual momentum** (Blitz, Huij & Martens
2011; Gutierrez & Pirinsky 2007), which is built differently on every axis that matters here - a
36-month fit, the fitted alpha **excluded**, the latest month **skipped**, scaled by residual volatility -
and which is far less of a duplicate (0.77 against 12-1, not 0.92). The note proposes replacing the
metric with residual momentum, left open for a session that did not write it (section 7).

This is a Monday-type note written by a 13:00 QUEUE session; no production code changed.

## 1. What the screener computes

`factor_engine.py` (the "Jensen's Alpha" block after the risk metrics):

    jensens_alpha = return_12m - [rf + beta x (m12 - rf)]

with `return_12m` the stock's total return over the full 12 months to the run (no skip), `beta` from the
same 12 months of daily log returns against `^SP500TR`, `m12` the S&P 500 total return over the same
12 months, `rf` the 13-week bill. It is scored as a sector percentile, higher is better
(`factor_engine` direction table), weight 25 inside Momentum beside `return_12_1` (40) and `return_6m`
(35, also skipping the latest month: `price_1m_ago / price_6m_ago - 1`).

**The algebra decides most of the question.** Across stocks in one run, `rf` and `m12` are constants,
so

    jensens_alpha_i = return_12m_i - beta_i x (m12 - rf) - rf

The ranking is the 12-month return moved by a beta tilt whose size is the market's excess return that
year. Measured on today's run (`runs/4793e4da8a1a`, 500 stocks): the spread of `return_12m` across
stocks is **0.455** (standard deviation); the spread of the beta term is **0.058** - an eighth of it.
And because beta and the "alpha" come from the same window, the quantity is, up to compounding, the
**fitted intercept** of a 12-month market-model regression, annualised. That is exactly the term Blitz,
Huij & Martens remove from their signal on purpose (section 3).

## 2. Measured: how much of it is the 12-month return (descriptive; no forward return computed)

`research/measurements/2026-11-20-jensens-alpha-overlap.py` and its `.json`. Rank correlations only -
nothing here is a return, an IC or a backtest (rules 4 and 5). The script first rebuilds the scored
figure from its own four terms (maximum error 0.0) so it is reading the metric the page shows.

**A. Today's run, sector percentiles built the engine's way (midpoint rank within GICS sector), 500
stocks:**

| Spearman | `jensens_alpha` | `alpha_12_1` | `return_12_1` | `return_6m` | `return_12m` | `return_1m` |
|---|---|---|---|---|---|---|
| `jensens_alpha` (scored) | 1 | 0.918 | **0.871** | 0.418 | **0.981** | **+0.181** |
| `alpha_12_1` (same CAPM line, 12-1 window) | | 1 | 0.940 | 0.435 | 0.906 | -0.108 |
| `return_12_1` | | | 1 | 0.374 | 0.913 | -0.207 |

The fresh-eyes review's +0.87 reproduces exactly. The S&P 500 fell 6.9% over the latest month of this
window, so this is one month's snapshot of how much the unskipped month moves the metric.

**B. Every month-end 2012-01 to 2026-09 (177 months; median 476 stocks), backtest v2's own adjusted
prices, whole cross-section (that file carries no sector):**

| Spearman, across months | median | 10th pct | 90th pct |
|---|---|---|---|
| `jensens_alpha` ~ `return_12m` | **0.969** | 0.896 | 0.998 |
| `jensens_alpha` ~ `return_12_1` | **0.918** | 0.837 | 0.961 |
| `jensens_alpha` ~ `return_1m` | **+0.279** | +0.004 | +0.496 |
| `alpha_12_1` ~ `return_12_1` | 0.966 | 0.902 | 0.999 |
| `alpha_12_1` ~ `return_1m` | +0.015 | -0.232 | +0.273 |
| residual momentum ~ `return_12_1` (155 months) | **0.774** | 0.713 | 0.830 |
| residual momentum ~ `return_1m` | -0.089 | -0.300 | +0.105 |

"Residual momentum" here is Blitz, Huij & Martens's construction with CAPM against the S&P 500 total
return in place of their Fama-French three-factor model: a 36-month regression of monthly excess
returns, the residuals of months t-11..t-1 summed and divided by their standard deviation. It needs 36
months of history, hence 155 months rather than 177.

**What the two tables say.**
- The scored metric is the 12-month return (0.97-0.98), and therefore close to 12-1 (0.87-0.92). With
  `return_12_1` at 40, about **65 of Momentum's 100 points rank stocks on one 12-month price move**.
- Skipping the month (`alpha_12_1`) removes the latest month's pull (+0.28 -> +0.02) but leaves a
  near-copy of `return_12_1` (0.94-0.97). Skipping alone fixes the timing and not the duplication.
- Residual momentum shares about 0.77 of its ranking with 12-1 - a related but distinct signal - and
  carries no positive load on the latest month.

A note on the first version of the script, kept because it is an easy mistake: fitting the residuals on
the same 12 months, *with* an intercept, makes them sum to zero over the window, so their 12-1 sum is
exactly minus the latest month's (it measured -0.92 against `return_1m`). Blitz et al. fit on 36 months
for this reason among others.

## 3. The literature

Read from the papers unless marked. PDFs fetched to the gitignored `data/papers/`; headline figures
below were checked against the files by this session, not only by the sub-agents that first read them.

- **Jensen, M. C. (1968).** "The performance of mutual funds in the period 1945-1964." *Journal of
  Finance* 23(2), 389-416. https://doi.org/10.1111/j.1540-6261.1968.tb00815.x - **not read**. The
  measure was built to judge whether a *manager* beat the CAPM over a past period, not to rank stocks
  on what comes next. Stated here only for what the measure is; nothing below depends on it.

- **Jegadeesh, N. & Titman, S. (1993).** "Returns to buying winners and selling losers." *JF* 48(1),
  65-91. https://doi.org/10.1111/j.1540-6261.1993.tb04702.x. NYSE/AMEX 1965-1989. Every strategy is
  run with and without a one-*week* gap: "By skipping a week, we avoid some of the bid-ask spread, price
  pressure, and lagged reaction effects that underlie the evidence documented in Jegadeesh (1990) and
  Lehmann (1990)" (p. 68). Table I: 12-month formation, 3-month holding, **1.31% a month without the
  gap, 1.49% with it**; the gap matters most at short horizons and almost not at all at 12/12 (0.68 vs
  0.69). Event time (Table VII): the 6/6 strategy without a gap *loses* in the first month after
  formation (-0.25%, t -0.59). Also, p. 85: in 1927-40 the strategy lost 40.81% over 36 months with a
  beta near -0.5 - "the relative strength strategy tends to select high- (low-) beta stocks following a
  market increase (decrease)" - the dynamic-beta problem section 3's residual papers address.

- **Jegadeesh, N. (1990).** "Evidence of predictable behavior of security returns." *JF* 45(3),
  881-898. https://doi.org/10.2307/2328797 - **not read** (abstract): significant negative first-order
  serial correlation in monthly returns. The one-month reversal every skip cites.

- **Asness, C., Moskowitz, T. & Pedersen, L. H. (2013).** "Value and momentum everywhere." *JF* 68(3),
  929-985. https://doi.org/10.1111/jofi.12021. Read (p. 936): "We skip the most recent month, which is
  standard in the momentum literature, to avoid the 1-month reversal in stock returns, which may be
  related to liquidity or microstructure issues." **The other side, same page:** for other asset classes
  the skip is "not necessary ... Momentum returns for these asset classes are in fact stronger when we
  don't skip the most recent month" - the argument is liquidity, and S&P 500 stocks are the most liquid
  equities there are. Their US stock universe is the names making up 90% of market value (about 350-680
  stocks), close to this screener's.

- **Blitz, D., Huij, J. & Martens, M. (2011).** "Residual momentum." *Journal of Empirical Finance*
  18(3), 506-521. https://doi.org/10.1016/j.jempfin.2011.01.003 (accepted manuscript:
  https://repub.eur.nl/pub/22252/ResidualMomentum-2011.pdf; page numbers below are the manuscript's).
  CRSP 1926-2009, stocks under $1 excluded. **Construction:** Fama-French three-factor residuals from a
  rolling 36-month regression; stocks ranked on the 12-1 residual return over the residuals' standard
  deviation in the same period; equal-weighted top-minus-bottom deciles. **The fitted alpha is left
  out on purpose** (p. 12): over two-thirds of the observations behind it lie outside the formation
  window, so including it would mix long-term reversal into the signal. **Table 2, one-month holding:**
  total-return momentum 10.26% a year at 22.70% volatility, alpha 7.98; residual momentum 11.20% at
  **12.49%** volatility (Sharpe 0.45 vs 0.90). Unstandardised residuals: 11.88% at 13.28%, Sharpe 0.89
  (footnote 2) - most of the gain is from removing factor exposure, a little from the scaling. **Largest
  10% by market cap (Table 8):** total-return momentum 8.87% at 24.55% (Sharpe **0.36**), residual
  9.18% at 15.27% (Sharpe **0.60**). **Without the skip (Table 7, J=12, K=3):** total-return 4.58%
  (Sharpe 0.21), residual 7.59% (Sharpe 0.66) - against 12-1 versions with the skip in Table 2; the
  authors attribute part of the gap to residual momentum having "smaller losses in the skip month"
  (p. 29). So the skip still helps the residual signal, but less than it helps total-return momentum.
  **Against:** the return gain is about one point a year - the case is volatility and crash exposure,
  not a higher premium; the 2000-2009 residual premium is not significant (t 1.18, Table 3, as read by
  the sub-agent - not re-checked here).

- **Gutierrez, R. & Pirinsky, C. (2007).** "Momentum, reversal, and the trading behaviors of
  institutions." *Journal of Financial Markets* 10(1), 48-75.
  https://doi.org/10.1016/j.finmar.2006.09.002. NYSE/AMEX/Nasdaq 1963-2000, price above $5. A
  single-index (CAPM) model fitted over the prior 60 months; winners and losers by standardised
  cumulative residual; one month skipped "to mitigate the negative serial correlation due to bid-ask
  bounce". Checked here: "Measuring expected returns with ai estimated over only the prior 12 months or
  with ai set to zero also generates qualitatively similar results" (p. 53). Their finding, as read by
  the sub-agent (Tables 1, 2, 6 - not re-checked): abnormal-return momentum earns about as much as
  relative-return momentum over the first year (1.14% a month, t 8.01, at J=6) **but does not reverse
  afterwards** (months 13-60: 0.03, t 0.65), where relative-return winners give back about 0.40% a
  month; above the NYSE median size the abnormal-return portfolio still earns 0.41% a month (t 3.28).
  It is the closest published relative of a CAPM residual, and it too skips the month.

- **Grundy, B. & Martin, J. S. (2001).** "Understanding the nature of the risks and the source of the
  rewards to momentum investing." *RFS* 14(1), 29-78. https://doi.org/10.1093/rfs/14.1.29 - **not read**
  (abstract only): strategies ranking on stock-specific return components are more profitable than those
  ranking on total returns; factor exposures explain most of momentum's variance but not its mean.

- **Novy-Marx, R. (2012).** "Is momentum really momentum?" *JFE* 103(3), 429-453.
  https://doi.org/10.1016/j.jfineco.2011.05.003 - **not obtained**; second-hand via AMP 2013 footnote 8:
  months 7-12 predict better than 2-6 in US stocks, but the benefit is "negligible".

- **Da, Z., Liu, Q. & Schaumburg, E. (2014).** "A closer look at the short-term return reversal."
  *Management Science* 60(3), 658-674 - already in this repo (`research/2026-10-09-short-term-reversal.md`):
  the plain one-month reversal has faded in large caps, but survives relative to the industry. The
  scored metric is a sector percentile, which is the industry-relative comparison.

**Literature, net.** Every source that builds a momentum signal skips the latest month for stocks
(a week in JT 1993, a month since), and the one source that argues against the skip does so for
liquid non-equity markets. The sources that remove market exposure do it with a residual from a fit
*longer* than the formation window, excluding the fitted alpha, and report the benefit as roughly half
the volatility and smaller crashes at a similar mean - including among the largest decile of stocks.
**No source found uses the construction the screener scores.**

## 4. Documented practice

Primary documents where obtained; MSCI and AQR quotes checked against the files by this session.

| Provider | Skips the latest month | What it ranks | Read |
|---|---|---|---|
| **MSCI Momentum Indexes** (methodology, Sep 2014 ed., section 2.2) | Yes - "computed excluding recent 1-month" | `(P_T-1 / P_T-13 - 1) - risk-free`, divided by 3-year weekly volatility; 6- and 12-month z-scores averaged | Yes (checked) |
| **S&P Momentum Indices** (via Invesco SPMO prospectus supplement, SEC 497K, 2018-02-28) | Yes - "excluding the most recent month" | 12-month price change, volatility-adjusted | Filing only; S&P's own PDF blocked (403) |
| **Barra (USE4 / CNE5) relative strength** | Reportedly a 21-day lag | Excess log return, exponentially weighted over ~504 days | **Not read** - secondary summaries only; the USE4 Methodology Notes (read) do not define descriptors |
| **AQR** - Asness, Frazzini, Israel & Moskowitz (2014), "Fact, fiction, and momentum investing", *JPM* 40(5) | Yes - "skipping the most recent month's return (to avoid microstructure and liquidity biases)" | Plain 12-1 total return; "simplest is best", residual momentum not discussed | Yes (checked) |
| **Robeco** - Lansdorp (2013), "Robeco's residual momentum: less risky and more sustainable", robeco.com insight, 2013-10-25 | Yes (Blitz et al., its researchers) | A "proprietary 'residualization' technique ... applied to the momentum factor in all of Robeco's quantitative stock selection models" | Read by the sub-agent; no construction detail given |

URLs: MSCI https://www.msci.com/eqb/methodology/meth_docs/MSCI_Momentum_Indexes_Methodology_Sep2014.pdf ;
SEC https://www.sec.gov/Archives/edgar/data/1378872/000119312518063261/d520329d497k.htm ;
AQR https://www.aqr.com/-/media/AQR/Documents/Journal-Articles/JPM-Fact-Fiction-and-Momentum-Investing.pdf ;
Robeco https://www.robeco.com/en-ch/insights/2013/10/robecos-residual-momentum-less-risky-and-more-sustainable .
MSCI's current edition was downloaded but its text could not be extracted; later editions are reported
to keep the formula - check before quoting rebalancing rules.

**Practice, net.** Every provider skips the latest month. None ranks stocks on a CAPM alpha. The
documented adjustments are of two kinds: **divide by volatility** (MSCI, S&P) or **remove market/factor
exposure** (Barra's excess-of-market return, reportedly; Robeco's residual). AQR uses neither and argues
for the simplest measure. **Where literature and practice disagree:** AQR's "simplest is best" against
Blitz et al.'s residual - both are large, serious shops, and the screener can follow either without
contradicting the literature. Neither supports the current metric.

## 5. How it fits the rest of the screener

- **Duplication.** Section 2: about 65 of Momentum's points rank one 12-month move. The 2026-10-09 metric
  audit's lesson about Quality (accruals counted three times, measured and kept) was that duplication is
  acceptable when deliberate and visible. This one was never decided; it is a by-product of putting a
  performance-evaluation measure in a signal category.
- **The Risk category already holds beta** (28.57 of Risk's 100, lower is better). The alpha's only
  departure from the 12-month return is `- beta x (m12 - rf)`, so in a rising market it *also* rewards
  low beta, inside Momentum, a second time; in a falling market it rewards high beta. Its tilt flips
  with the market's sign - the same dynamic exposure JT 1993 (p. 85) and Blitz et al. identify as the
  source of momentum's crashes, entering here through the risk adjustment meant to remove it.
- **The momentum panic rule** (`factor_engine.market_panic_state`, `research/2026-10-09-momentum-regime.md`)
  exists because of momentum crashes. Residual momentum addresses the same risk at the signal (Blitz et
  al.: drawdowns about half as severe after 2000). That is complementary, not a replacement: the rule's
  three numbers are fixed in advance and are not touched by anything here.
- **Explainability.** "Jensen's alpha" invites a student to read the number as risk-adjusted skill. What
  it measures in this screener is last year's price move. A residual-momentum row reads plainly: *how
  much the stock beat what its market exposure predicted, from a year ago to a month ago, per unit of
  its own noise* - and every term can be printed on the row, as row 0.8c requires.
- **Value-momentum.** The value-momentum correlation note (`research/2026-10-15-value-momentum-correlation.md`)
  is about the total-return category score; nothing here changes the Value side.

## 6. Options

| | What | For | Against |
|---|---|---|---|
| A | Keep as is | No change; the page already says it is unskipped | No source; 0.97 duplicate of the 12-month return; carries the latest month every other metric skips |
| B | Skip the month (`alpha_12_1`) | Small change, two price points already in the fetch; fixes the timing | Still a 0.94-0.97 copy of `return_12_1`; still a same-window intercept, the term Blitz et al. exclude |
| C | **Replace with residual momentum** (CAPM on `^SP500TR`, 36-month monthly fit, months t-11..t-1, over their s.d.) | The construction the literature tests, including in large caps; distinct from 12-1 (0.77); lower crash exposure; every term printable | Needs 37 months of prices (the fetch keeps 13); new lineage, equation and recompute; CAPM rather than FF3 is a simplification to state |
| D | Drop it; 25 points to `return_12_1`/`return_6m` | Simplest; ends the duplication | Discards the one thing the metric was presumably for - a market-adjusted view |

**Why CAPM and not FF3 for C.** Blitz et al. use FF3; Gutierrez & Pirinsky report CAPM with "qualitatively
similar results" for alternatives. CAPM uses only the index the run already fetches (`^SP500TR`), needs
no third-party factor file in the scoring path, and is one regression a student can follow. The
simplification is stated, not hidden.

## 7. Proposal

**Proposal status: ACCEPTED 2026-11-21 (06:00 weekend session), not yet built** - option C, as
specified below; build in a 06:00 session after `research/2026-11-19-what-the-ranking-is-for.md`
section 4 is committed, not before 2026-11-25 (Monday 11-23 is 0.15's, Tuesday 11-24 the IV inversion's). The decision, with what the review measured, is section 9.

*As proposed:* replace `jensens_alpha` in Momentum with CAPM residual momentum (option C:
36-month monthly regression on `^SP500TR` excess returns, residuals of months t-11..t-1 summed over
their standard deviation; same 25 points, no weight change), for a session that did not write this note.

What deciding it involves, so the decision is not a skim:

1. **Read section 4 and the Blitz manuscript's Tables 2 and 8** yourself; the case rests on them.
2. **Measure the fetch cost**: the core fetch's history call keeps 13 months; residual momentum needs 37
   month-ends. Whether that is a longer `period` on the same request (no extra call) or a second request
   (settled row "ctx": extra per-ticker calls inside the core fetch tripped the rate limiter) must be
   read from the fetch code, not assumed. Read so far: it is one `t.history(period="13mo")` call
   (`factor_engine.py`, the per-ticker fetch), so a longer `period` is plausibly the same request -
   **but `price_history_fields` derives volatility, the daily-return series behind beta and max
   drawdown, and dollar volume from the whole frame it is handed**, so a longer history must be cut
   back to 13 months for those, or every Risk metric silently changes window. Coverage: how many of ~500 have 36 months (recent IPOs and
   spin-offs will not) - and what the metric does for them (missing within the category, shared across
   the other two momentum metrics, which is the existing rule).
3. **One change at a time** (`plan/factor-lab.md`, "From the lab to the weights"): the weight stays 25;
   only the construction changes.
4. **Ship the frontend with it** (row 0.8c): `LINEAGE`, `RECOMPUTE` and `EQUATIONS` for the new metric,
   its terms published through `ENGINE_KEYS` the way `max_drawdown_1y` publishes its two closes; the
   old `jensens_alpha` line and claim retired, and the "removed" reason added to `NOT_USED_BECAUSE` if
   it stays listed. `factor_engine.py` is a `RESULT_PATHS` file - budget the research-engine rerun.
5. **Backtest v2 can compute it** (its daily prices begin 2010-11), so a deciding session may register
   one research-engine experiment before building - prediction written first: *correlation with the
   current momentum category score about 0.8; no measurable difference in IC* (the engine cannot tell a
   typical edge from zero, per `research/2026-10-15-research-engine-first-batch.md`). It is a Backtest
   observation either way, never the evidence (rule 5).
6. If C's fetch cost is unacceptable, **B is the fallback**, and the reason goes in the decision.

**Refuse if:** the extended history costs per-ticker requests inside the core fetch, or coverage falls
below the category's existing floor - then say so and take B.

## 8. What this note did not do

- No production code, weight or page sentence changed.
- Chaves (2016, *Journal of Investing* 25(2)) and Blitz, Huij, Lansdorp & Verbeek's short-term residual
  reversal were not obtained.
- No forward return was computed; section 2 is rank correlations only.

## 9. Decision (2026-11-21, 06:00 weekend session - a session that did not write this note)

**ACCEPTED: option C, built later.** Taken against the note's own checklist (section 7), each item
done rather than read.

**1. The sources, re-read.** Blitz, Huij & Martens, accepted manuscript (`data/papers/blitz2011.pdf`,
text extracted with pypdf by this session). Every figure section 3 quotes reproduces from the file:
Table 2 panel A 1M 10.26 / 22.70 / Sharpe 0.45 / alpha 7.98, panel B 1M 11.20 / 12.49 / 0.90 / 10.85
(manuscript p. 44); Table 8 (largest 10% by market value, p. 53) total-return 1M 8.87 / 24.55 / 0.36
(t 2.75), residual 9.18 / 15.27 / 0.60 (t 5.32); Table 7 (no skip, J=12, K=3, p. 51) 4.58 / Sharpe
0.21 against 7.59 / 0.66; the alpha exclusion, p. 12: "over two-thirds of the observations behind the
estimated alpha are outside the 11-month formation period". The sub-agent's Table 3 reading, which the
note had not re-checked, also reproduces (p. 45): 2000-2009 residual momentum 4.65% a year, **t 1.18**;
total-return momentum **-8.54%**, t -1.19. So in the one decade both fail significance, the residual
version lost nothing where the total-return version lost 8.5 points a year.

**2. The fetch cost, measured** (`research/measurements/2026-11-21-residual-momentum-fetch-cost.py`
and `.json`):
- **Same request.** The core fetch's price history is one `t.history(period="13mo")` call
  (`factor_engine.py`, the per-ticker fetch). `period="37mo"` on the same call, timed on 30 random
  universe tickers with the order alternated: median **0.126 s -> 0.163 s** (+0.04 s), 274 -> 774 bars,
  frame 17.5 KB -> 49.5 KB. No extra request, so the refusal condition ("costs per-ticker requests
  inside the core fetch") is not met. About +20 s on a ~450 s fetch if it were serial; less in practice.
- **Coverage: 495 of 503 (98.4%)** have the close at 2023-10-31 that a run on 2026-11-21 needs (494 from
  backtest v2's daily cache, the other 9 probed live). Short: FDXF, GEV, HONA, Q, RDDT, **SNDK**, SOLV,
  VYLR - spin-offs and listings since 2024. For them the metric is missing and its 25 points are shared
  across `return_12_1` and `return_6m`, the existing within-category rule (settled row 0.8m: a missing
  *metric* shares inside its category; only a missing *category* counts at 50). Well above
  `MIN_CATEGORY_COVERAGE` (0.90), so the second refusal condition is not met either. SNDK is in today's
  top five; the build session should say so in the changelog's expected effect.
- **The market series.** `fetch_market_closes` already pulls `^SP500TR` for the panic rule, but at
  `period="3y"`, which on 2026-11-21 starts 2023-11-21 - three weeks short of the 2023-10-31 close the
  regression needs. Lengthen that one request (to `4y`) rather than add a second; the panic rule reads
  only its last 24 months and 126 days, so it is unaffected - assert that in a test.

**3. What a second reader adds - the case against, weighed.**
- **The evidence is long-short; the screener is long-only.** Blitz et al. report top-minus-bottom
  deciles. Han, Zhou & Zhu's stop-loss finding (`research/2026-10-22-stop-loss-on-the-bots-rule.md`)
  already taught this project that momentum's crash risk sits mostly in the *losers* leg, which a
  long-only book does not hold. So the crash-protection half of the case is weaker here than the paper's
  Sharpe ratios suggest, and **the changelog must not quote 0.36 -> 0.60 as what a reader of this
  screener should expect.** The decision does not rest on it.
- **Why C and not D (drop it).** D is AQR's "simplest is best" and is defensible. But D moves 25
  category points (3.25 composite) onto two metrics that are already 0.87-0.92 alike, deepening the
  duplication this note exists to remove, and it is a weight change, which the lab-to-weights route
  caps at 3 points a change. C moves no weight: it changes one metric's construction to the one the
  literature tests (including in the largest decile, Table 8, t 5.32) and replaces a 0.97 copy of
  the 12-month return with a signal 0.77 related to 12-1. And it removes, rather than adds, a second
  low-beta reward on top of Risk's beta (section 5).
- **Why not B (skip the month only).** Fixes the timing, keeps a 0.94-0.97 duplicate and the
  same-window intercept the source excludes on purpose.
- **Explainability.** One regression a student can follow; every term can be printed on the row
  (row 0.8c). "Jensen's alpha" invites a reading as skill that the metric does not support.
- **CAPM rather than FF3.** A stated simplification (Gutierrez & Pirinsky: "qualitatively similar");
  it keeps a third-party factor file out of the scoring path.

**4. Why not build it today.** Three reasons, none of them doubt about the change:
- **It collides with 0.15.** The Monday session (2026-11-23) measures "the composite" against criteria
  pre-registered on 2026-11-19. Changing a Momentum metric before that measurement runs changes the
  object of a pre-registration two days old; doing it after keeps that measurement clean. The 13:00
  session that wrote this note said the same.
- **It is a full 06:00 session's build.** The fetch window (with `price_history_fields` handed the last
  13 months only, so volatility, beta's daily returns, drawdown and dollar volume keep their window -
  section 7 item 2's trap; the split-integrity check must still run over the whole 37 months the new
  metric reads); the market series; `LINEAGE`/`RECOMPUTE`/`EQUATIONS` with the terms published through
  `ENGINE_KEYS`; `jensens_alpha` retired with a `NOT_USED_BECAUSE` reason; backtest v2 computing the
  same metric through the same function (`pit_record.FIELD_CLASSES`); and the research-engine rerun
  that any `factor_engine.py` edit forces (15-30 min with `--jobs 4`).
- **Gate 5 cannot see it** (it rescores nothing), so the first full run under it is the 02:00 claims
  gate. Per `prompts/nightly.md` section 4, the build session greps the tests for `jensens_alpha`'s
  wording and numbers and makes a throwaway full run (memory: verify a pipeline change with one) -
  which also wants a 06:00 session, not a 13:00 one four hours before nothing, or a weekend with no
  02:00 run until Monday.

**What the build session does, in order:** (a) optionally register one research-engine experiment
before building, prediction as section 7 item 5 wrote it; (b) build per section 7 items 2-4 and the
bullets above, weight 25 unchanged; (c) changelog entry with the evidence above, the long-only caveat,
and the expected effect measured on a throwaway full run (top-25 churn; SNDK and the other seven lose
the metric); (d) open a drilldown at 1440 and 375 px and say what the new row shows.
