# Size: what stands behind 5% of the score (2026-10-20)

**Question.** Factor-lab question 4 (`plan/factor-lab.md`, "From the lab to the weights"): what is
the case for spending 5% of the composite on `size_log_mcap` - minus the log of market value,
ranked within sector, smaller scores higher - inside a universe that is the 500 largest US
companies? Research-led per CLAUDE.md rule 4: no IC series, no backtest number and no return of
the screener's own ranking is evidence below. Section 8 lists the research engine's reading as
colour only, labelled as such (rule 5).

**Read first, and not redone:** `research/2026-08-31-size-factor-in-a-large-cap-universe.md`
(August's note: the tilt is an independent bet, it buys junk on average, and the 22 Quality
points absorb most of that at the top; its Candidate 1 - compress the tilt toward MSCI's 1/ln
shape - was **closed 2026-09-03**: every compression is indistinguishable from turning the weight
down, so the only honest knob is the weight, which is what this note is about). Then
`research/2026-10-12-factor-lab-findings.md` section 5 and `research/2026-10-20-backtest-overfitting.md`
section 4.

**Finding.**

1. **There is no unconditional size premium among large companies.** Fama & French's own 1992
   table has almost no gradient at the top: 0.95% a month for the ninth size decile against 0.88%
   and 0.90% for the two halves of the tenth. Fama & French (2008) put the
   log-size slope among big stocks at -0.08 (t -1.96), "marginal". Alquist, Israel & Moskowitz
   (2018) drop the smallest 20-30% of firms and the decile spread falls to 0.4-0.7% a year
   (t 0.20-0.34), with a *negative* CAPM alpha. The lab agrees on public data: the fourth size
   quintile minus the fifth earned 0.14% a month over a century (t 2.26), but 0.08% (t 0.85)
   since 1981, -0.12% (t -0.76) since 2013, and was not distinguishable from zero even inside
   Fama & French's own 1963-1990 sample (t 1.70).
2. **The case that survives is conditional on quality** (Asness, Frazzini, Israel, Moskowitz &
   Pedersen 2018): size pays when junk is controlled, because small stocks are junky and high-beta.
   Their 25 size-by-quality portfolios have the fourth size quintile above the biggest in every
   quality row, but they publish no test among large stocks alone, and the lab cannot yet rerun it.
3. **Practice is split, and the larger half includes it.** MSCI's Diversified Multiple-Factor
   indexes (a quarter of the signal is Low Size, and a small-size tilt between -0.6 and -0.1
   standard deviations is *mandatory*), FTSE Russell's Russell 1000 Comprehensive Factor index
   (size one of five equal tilts, defined as -ln cap - the screener's own formula), STOXX/iShares
   LRGF and Dimensional's large-cap ETFs all tilt toward the smaller of the large. AQR, S&P's own
   QVM indexes, J.P. Morgan's diversified-factor indexes and Research Affiliates leave it out;
   risk models (Barra) neutralise it. S&P DJI attributes over half of the S&P 500 Equal Weight
   index's excess return to its smaller-company tilt, and **all** of it to weighting equally
   *within* sectors - the screener's construction.
4. **In this composite, 5 points is a small tilt.** Without Size the other seven lean very slightly
   toward the largest companies (-0.04); with it, toward the smaller (+0.13) - the second-smallest
   of the eight category exposures, stable across every weekly payload since 2026-08-20. Holding it
   neutral changes **3 of the top 25**. It is partly a second value bet (+0.36 with Valuation),
   and it is one of the categories holding the ranking growth-neutral: without it, net growth
   exposure moves to +0.061, outside the 0.05 band `tests/test_growth_category.py` pins.

**Decision: the weight stays at 5%.** Lab and raw literature say no large-company premium;
the conditional literature and most of documented large-cap multi-factor practice say a modest
tilt, run beside a quality control, is reasonable. The pre-registered reading rules
(`plan/factor-lab.md`) say that where they disagree, the note says so and the weight stays - and
a 5-point tilt that moves 3 of 25 names sits at the gentle end of what the providers that
include size do (MSCI requires a tilt; FTSE gives size a fifth of its tilts).
**What the 5 points are for is stated in section 7**, and it is not a return premium. No
proposal is open; section 9 gives what would reopen it.

---

## 1. What the category is, as built

One metric, `size_log_mcap` = -ln(market cap), 100% of the category, 5 of the composite's 100
points (`config.yaml`; the comment there calls it a "Fama-French SMB proxy"). Like every metric it
is converted to a **sector percentile rank**, so the log does nothing to any score (closed
2026-09-03; `tests/test_size_tilt_is_documented_truthfully.py`). Two consequences matter below:
the score is linear in rank, so the tilt is equal-weight-like rather than log-compressed; and it
is *within sector*, which is the construction S&P DJI credits for the Equal Weight index's record
and the opposite of how MSCI, FTSE and STOXX score size (section 4).

## 2. What the lab measures among large companies

Experiment 4 (`factor_lab.py`, `size-effect`; Fama & French 1992 on the library's ten size
groups) measured the ninth group against the tenth. This note extends it on files the lab already
holds - Ken French's library, CRSP vintage 2026-08 - in the four directions the literature in
section 3 says matter: by period, holding value and momentum fixed, the "junk" mechanism, and
SMB's alpha against the other factors. Script and output:
`research/measurements/2026-10-20-size-lab.py` / `.json`. Every figure is a long-short return
between portfolios of all listed US stocks, free of costs; it describes the trait among large
companies, never the screener's ranking.

**Where the S&P 500 sits.** In August 2026 the library's fourth size quintile (NYSE breakpoints)
held 372 firms averaging $14.5B and the fifth 325 averaging $191.8B; deciles 8, 9 and 10 averaged
$19.1B, $43.1B and $334.2B. The screener's universe runs $5.7B to $5,236B with a median of $45.0B
(section 5.1). So the S&P 500 is roughly quintiles 4-5, deciles 7-10, and **quintile 4 minus 5 is
the closest public analogue of "smaller among the large"**.

### 2.1 The large-company spread, by period (% a month, t in brackets)

| Period | Q4 - Q5, value-weighted | Q4 - Q5, equal-weighted | D9 - D10, value-weighted | D8 - D10, value-weighted |
|---|---|---|---|---|
| 1926-2026 | **0.142 (2.26)** | 0.110 (2.38) | 0.085 (1.57) | 0.148 (2.17) |
| 1963-1990 (FF1992's sample) | 0.183 (1.70) | 0.131 (1.68) | 0.080 (0.84) | 0.177 (1.49) |
| 1963-2012 (AFIMP's sample end) | 0.208 (2.43) | 0.135 (2.27) | 0.143 (1.94) | 0.213 (2.24) |
| 1981-2026 (after Banz 1981) | **0.076 (0.85)** | 0.047 (0.77) | 0.044 (0.56) | 0.082 (0.84) |
| 2000-2026 | 0.172 (1.38) | 0.098 (1.20) | 0.140 (1.24) | 0.206 (1.51) |
| 2013-2026 | **-0.123 (-0.76)** | -0.052 (-0.48) | -0.166 (-1.09) | -0.115 (-0.67) |

January versus the rest, Q4 - Q5 value-weighted: 0.858 (t 4.05) in January and 0.077 (t 1.18) in
the other eleven months over the century; since 1981, 0.113 and 0.072, neither distinguishable.
The sign is positive in every long window and the size is about 1.7% a year over the century;
after 1981 it is half that and inside noise for 45 years, and negative for the last 14.

Under the reading rules written before any question was researched (`plan/factor-lab.md`):
**"not confirmed" since 1981 - a 45-year window, so this is not the "under twenty years" case -
and not confirmed inside Fama & French's own sample either**. It is not weakness since publication
alone; among large companies it was never strong.

### 2.2 Holding value or momentum fixed

The 5x5 size-by-book-to-market and size-by-prior-return files allow the double sort Asness et al.
use with quality: quintile 4 minus quintile 5 of size *inside* each quintile of the other trait,
averaged over the five.

| Period | Within book-to-market quintiles | Within prior-return quintiles |
|---|---|---|
| 1926-2026 | 0.121 (2.15) | 0.150 (2.59) |
| 1981-2026 | 0.062 (0.78) | 0.059 (0.73) |
| 2013-2026 | -0.155 (-1.20) | -0.144 (-0.95) |

Holding value or momentum fixed changes nothing. One pattern inside it is worth recording: within
prior-return quintiles the large-company size spread is near zero among the losers (0.008, t 0.06,
1963-2026) and 0.15-0.19 (t 1.8-2.2) in the three groups above them - size pays, where it
pays at all, among the stocks that are not already falling, which is the Asness et al. story in
miniature.

### 2.3 Are the smaller of the large the junkier ones?

The same 5x5 file publishes each portfolio's value-weighted operating profitability and asset
growth for every formation year. Quintile 4 against quintile 5, across the five book-to-market
rows:

| | 1963-2026 | 2013-2026 |
|---|---|---|
| Operating profitability, Q4 / Q5 | 0.300 / 0.320 | 0.378 / 0.395 |
| Year-cells where Q4 is the less profitable | **70%** | 71% |
| Log asset growth, Q4 / Q5 | 0.116 / 0.107 | 0.104 / 0.079 |
| Year-cells where Q4 grows its assets faster | 61% | 66% |

Yes, mildly: among large companies the smaller ones are less profitable and invest more
aggressively, in about two years out of three. The junk mechanism the literature describes is
present at the top of the market, not only among microcaps - which is why an uncontrolled size
tilt is a short position in Quality and Investment (section 5.2 measures the same thing inside the
screener).

### 2.4 SMB's alpha against the other factors (all stocks, % a month)

| Period | SMB raw | CAPM alpha | FF3 + momentum alpha | Alpha on market, HML, RMW, CMA | ... plus momentum |
|---|---|---|---|---|---|
| 1963-2026 | 0.186 (1.70) | 0.076 (0.71) | 0.097 (0.88) | 0.190 (1.83) | 0.223 (2.10) |
| 1963-2012 | 0.275 (2.18) | 0.189 (1.55) | 0.206 (1.61) | **0.365 (3.01)** | 0.344 (2.78) |
| 1981-2026 | 0.048 (0.39) | -0.050 (-0.40) | -0.010 (-0.08) | 0.105 (0.89) | 0.138 (1.15) |
| 2013-2026 | -0.133 (-0.61) | -0.348 (-1.59) | -0.238 (-1.20) | -0.256 (-1.34) | -0.126 (-0.70) |

Over Asness et al.'s window the public data reproduce their direction: controlling for
profitability and investment roughly doubles SMB's alpha (0.365, t 3.01, against their 33 basis
points, t 2.82, for 1963-2012). SMB loads -0.42 on RMW, so the quality-control mechanism is real
on this data too. After 1981 the controlled alpha is 0.105 (t 0.89), and since 2013 negative.
SMB inside each profitability tercile (2x3 file) tells the same story: 0.228 (t 1.75) among the
robustly profitable over 1963-2026, 0.098 (t 0.66) since 1981. RMW is not QMJ - Asness et al.'s
quality factor adds safety, payout and growth - so this is an approximation of their control, not
a rerun of it. A second, independent construction, Jensen, Kelly & Pedersen's `market_equity`
factor (US, all stocks, capped value weights), reads 0.189 (t 1.81) over 1926-2025, -0.083
(t -0.68) since 1981 and -0.198 (t -0.85) since 2013.

## 3. What the literature says

Read from the papers in this session (PDFs fetched and read with pypdf; the source copies are
listed in section 10) except where marked **not read**.

**Fama & French (1992), *JF* 47(2), Table II Panel A** (equal-weighted, NYSE breakpoints,
1963-1990). Monthly returns by size: 1.64 (1A) and 1.16 (1B) at the small end; deciles 2-8 between
1.07 and 1.29; decile 9 **0.95**; 10A **0.88**, 10B **0.90**. Almost all of the spread is the
smallest half-decile; there is no gradient between the ninth decile and the two halves of the
tenth. The size slope weakens in the second half of their sample (Table VI: -0.16, t -1.62, for
1963-76; -0.07, t -1.16, for 1977-90).

**Fama & French (2008), "Dissecting anomalies", *JF* 63(4), Table IV.** Fama-MacBeth slope on log
market cap with the other anomaly variables, 1963-2005: all stocks -0.18 (t -4.36); microcaps
-0.46 (t -6.95); small -0.03 (t -0.37); **big -0.08 (t -1.96)**. Their summary: the size effect
owes much of its power to microcaps and is "marginal among small and big stocks" (p. 1675). "Big"
is above the NYSE median - a wider group than the S&P 500.

**Alquist, Israel & Moskowitz (2018), "Fact, fiction, and the size effect", *JPM* 45(1)** (AQR's
reprint of the published article). SMB 2.5% a year (t 2.13), 1926-2017, CAPM alpha 0.9% (t 0.86);
Sharpe 0.22, the lowest of the five factors they compare. Non-January SMB alpha about zero. No
country in the MSCI universe significant (13 of 24 t-statistics negative). **Exhibit 18 is the
figure that matters for a large-company screen**: dropping the smallest 0/1/5/10/20/30% of firms
before forming the decile spread gives 7.2/6.2/4.1/2.5/0.7/0.4% a year (t 2.51 to 0.20), and CAPM
alphas of 4.3/3.3/1.0/-0.3/-1.9/-2.3%. Their list of "facts" includes that the effect is stronger
after controlling for quality; their caveat is that this does not resurrect size on its own -
"we also do not believe that size on its own is a key factor" (p. 4).

**Asness, Frazzini, Israel, Moskowitz & Pedersen (2018), "Size matters, if you control your
junk", *JFE* 129(3)** (the published version, open access). SMB 0.23% a month (t 2.27),
1926-2012; 0.04% (t 0.41) outside January; -0.04% (t -0.23) over 1980-99. Alphas, 1957-2012:
CAPM 12bp (t 1.12); with HML and UMD 14bp (t 1.23); **with QMJ 49bp (t 4.89)**, SMB loading -0.74
on QMJ; with FF5's RMW and CMA 33bp (t 2.82, 1963-2012). With QMJ the post-1980 subperiods are
50bp (t 3.06, 1980-99) and 89bp (t 4.04, 2000-12). Against BAB alone, SMB loads -0.42: small
stocks are high-beta as well as junky. **Table 3** (25 value-weighted size-by-quality portfolios):
small-minus-big is 0.23 (t 1.21) in the junk quintile and 0.42-0.54 (t 2.82-3.44) in the other
four. The large end of that grid is the nearest thing in the paper to a large-company test: the
fourth size quintile earned more than the biggest in every quality row (0.40 vs 0.12 junk; 0.78
vs 0.53 high quality), but no t-statistic is given for that difference and the quintile
breakpoints are not stated. The "not concentrated in microcaps" claim rests on a figure of decile
alphas (Fig. 6) with no numbers: with QMJ the relation is close to linear across deciles and the
alphas turn negative for the largest stocks. **No table in either version tests size among large
stocks alone.** Implementation: half-spreads of 0.1% for the biggest quintile against 2.5-3.8% for
the smallest - the conditional premium is cheapest to hold exactly where it is least documented.

**Hou & van Dijk (2019), "Resurrecting the size effect", *RFS* 32(7) - not read** (the free copy
sits behind a bot challenge; publisher paywalled). Abstract only: the post-1980 disappearance of
the realised size effect is due to negative profitability shocks to small firms and positive ones
to big firms; adjusted for them, a size effect remains in *expected* returns. No effect size is
quoted here, and nothing in the abstract speaks to large firms.

**Horowitz, Loughran & Savin (2000), *Journal of Empirical Finance* 7(2) - not read** (paywalled).
Abstract: over 1980-96, three methods find no consistent relation between size and realised
return. That removing firms under $5M eliminates the premium is confirmed second-hand by both
Asness et al. (fn 2) and Alquist et al. (fn 16).

**van Dijk (2011), "Is size dead?", *JBF* 35(12) - not read** (bot challenge). Abstract: empirical
work finds the effect disappeared after the early 1980s; theory still derives one from systematic
risk.

**Israel & Moskowitz (2013), *JFE* 108(2) - not read** (only the abstract cover was served).
Abstract: almost all of the size premium is the long side; the value premium falls with size;
results over 86 years. The closest verified figure is Asness et al.'s SMB of 0.25% a month
(t 1.95) for 1963-2012, which they describe as consistent with Israel & Moskowitz.

**What the literature agrees on.** No study read tests size among the 500 largest US companies.
Every source that gets near it - the top deciles in Fama & French (1992), their "big" group in
2008, Alquist et al.'s exclusion table - finds the *unconditional* gradient at the large end small
to absent. The one positive case is **conditional**: size controlled for quality (and low beta),
where Asness et al.'s size-by-quality grid has the fourth quintile ahead of the biggest in every
quality row. The screener's methodology page already says the first half of this ("no published
study establishes a size premium within the largest two market-cap deciles"); that sentence
survives this reading.

## 4. What practice does

Read from the primary documents (index methodologies, ground rules, fund prospectuses on EDGAR)
except where marked.

**Include a smaller-company tilt inside a large/mid-cap multi-factor product:**

- **MSCI Diversified Multiple-Factor Indexes** (methodology, May 2022, sections 2.3.1 and 2.4,
  Appendix III): four target factors - Value, Momentum, Quality and **Low Size** - each a quarter
  of the alpha score, Low Size being the negative Barra size exposure; the optimiser *requires*
  an active size exposure between **-0.6 and -0.1** standard deviations against the large+mid
  parent, sector weights within 5% of the parent. Value and Quality are z-scored within sector;
  **Low Size is not** - it relies on the sector bands. Appendix V: the "MSCI USA Sector Diversified
  Multiple-Factor Capped" indexes run the same optimisation, Low Size included, inside each GICS
  sector of MSCI USA - the one documented within-sector small tilt inside a large/mid universe
  found.
- **MSCI's single-factor products**: Low Size (weights proportional to 1/ln cap, large+mid parent;
  the brochure warns that simply overweighting small caps "may also result in an unintended
  exposure to lower quality stocks") and Size Tilt (square root of cap weight). MSCI research
  (Oberoi et al. 2016, "One size does not fit all") finds smaller deciles of MSCI USA IMI ahead,
  with a "performance cliff" after decile 8 that it attributes to momentum and residual volatility.
- **FTSE Russell, Global Factor Index Series ground rules** (v8.8, July 2026, rule 5.5 and Table
  one): Size = -1 x ln(full market cap), z-scored across the eligible universe (not within
  industry) - **the screener's own formula** - and the **Russell 1000 Comprehensive Factor Index**
  tilts equally on value, quality, momentum, low volatility and size, with industry weights bounded
  around the parent's.
- **STOXX U.S. Equity Factor Index / iShares LRGF** (summary prospectus, Nov 2025): large+mid caps,
  optimised toward momentum, quality, value, low volatility and size; LRGF tracked MSCI's DMF index
  (with Low Size) until June 2022.
- **Dimensional** (ETF Trust 485BPOS, Feb 2026): the US Large Cap Vector and US High Profitability
  ETFs "may emphasize" smaller-capitalisation companies *within* the large-cap segment, alongside
  lower relative price and higher profitability - i.e. a size tilt run together with a
  profitability tilt, which is the Asness et al. construction in long-only form.
- **Goldman Sachs ActiveBeta** (methodology, April 2024): no size factor, but since April 2024 the
  low-volatility score includes the inverse of market cap. No reason is given.
- **S&P 500 Equal Weight** (S&P DJI FAQ, 2026; Edwards, Ganti & Preston, "Worth the Weight", July
  2024): S&P DJI attributes over half of the index's historical excess return to its weight in the
  smaller constituents; of its cumulative 92% excess return from March 2003 to June 2024, **all**
  came from weighting equally *within* sectors (cross-sector weighting alone would have cost about
  45%). A size-matched S&P 500/MidCap 400 blend tracks its long-run return (R-squared 0.58 on
  trailing 12-month excess returns). This is the index provider's own attribution, not
  peer-reviewed - but it is the most direct documented evidence of a within-sector small tilt
  inside the S&P 500, and it is the screener's construction.

**Leave it out or neutralise it:**

- **AQR Large Cap Multi-Style Fund** (prospectus, Jan 2025): value, momentum and quality across
  the Russell 1000 range; no size. AQR's published position is Alquist et al. above.
- **S&P Quality, Value & Momentum Multi-Factor Indices** (methodology, Dec 2025): quality, value and
  momentum only; weights are market cap times score.
- **J.P. Morgan Diversified Factor** (FTSE-administered ground rules, Aug 2026): value, momentum
  and quality, ranked within industry; no size.
- **Research Affiliates' RAFI Multi-Factor** (rulebook v5.3, April 2026): size is a separate sleeve
  of small companies, not a smaller-is-better tilt among large ones. RA's research (Kalesnik & Beck
  2014, *not read* - summaries only) argues the premium after 1981 is insignificant.
- **Risk models** (Barra USE4 methodology notes, 2011): Size and Non-Linear Size are risk factors,
  standardised so the cap-weighted market has zero exposure and estimated industry-neutral - to be
  measured and controlled, not harvested. (Axioma's handbook could not be obtained.)

**And one practice that matters for this universe specifically:** the S&P 500's own eligibility
rule (S&P U.S. Indices methodology, July 2026, "Financial Viability") requires positive GAAP net
income in the most recent quarter and over the trailing four. The smallest S&P 500 members have
passed a profitability screen that the smallest stocks in the library's portfolios have not.

**What practice says about the construction.** No provider found scores size *within sector* in a
large-cap multi-factor index; MSCI, FTSE and STOXX score it across the whole parent and then bound
sector weights. Under tight sector bands much of the realised tilt is within-sector anyway, and
the screener's within-sector score correlates 0.946 with universe-wide -ln(cap) (section 5.1), so
the difference is real but small. Every provider that includes size runs it **beside a quality or
profitability tilt** (MSCI, FTSE, STOXX, Dimensional) - none runs size alone.

## 5. What Size does in the published composite (measured)

Run `7c12b43a07b6` (2026-10-20 13:21, the run `dashboard_data.js` carries; 501 stocks). Every
composite below is recomputed with the engine's own `factor_engine.compute_composite` from the
run's scored frame and config; the unmodified recomputation reproduces the published composite to
0.00 for every stock and the published top 25 exactly. Script and output:
`research/measurements/2026-10-20-size-composite.py` / `.json`. Structure only - no return is read.

### 5.1 How it is built

`size_log_mcap` is ranked within sector, so the category asks "is this company small *for its
sector*?" In practice that is nearly the same as "is it small?": the Size score's Spearman with
-ln(market cap) across the whole universe is **0.946**. Average Size score by market value:

| Market value | < $25B | $25-50B | $50-100B | $100-500B | >= $500B |
|---|---|---|---|---|---|
| Mean Size score | 83.6 | 58.2 | 35.3 | 16.6 | 3.5 |

The universe runs from $5.7B to $5,236B (median $45.0B; tenth percentile $14.0B).

### 5.2 Against the other seven categories

| Spearman with Size | Valuation | Investment | Quality | Revisions | Risk | Momentum | Growth |
|---|---|---|---|---|---|---|---|
| Whole universe | **+0.362** | **+0.260** | -0.117 | -0.128 | -0.171 | -0.199 | **-0.325** |
| Within sector (average) | +0.358 | +0.260 | -0.127 | -0.138 | -0.178 | -0.207 | -0.310 |

The other seven span **27.4%** of the Size score (R-squared; 27.9% on 2026-08-31), Valuation and
Investment together 16.1%, Valuation alone 13.1%. Against the *mean* of the other seven, Size
correlates **-0.058** - it runs neither with nor against the rest of the composite taken whole.
That is the difference from Growth, which runs against the rest at -0.388 and whose 13 points
are therefore what keeps the ranking growth-neutral (`research/2026-10-12-growth-category.md`
section 10).

### 5.3 The composite's net exposure to size

As the Size weight moves from 0 to 15 (the other seven scaled pro rata):

| Size weight | 0 | 1 | 2 | 3 | 4 | **5** | 6 | 8 | 10 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|
| Composite vs Size score (Spearman) | -0.039 | -0.008 | +0.024 | +0.058 | +0.092 | **+0.126** | +0.160 | +0.227 | +0.294 | +0.460 |
| Top 25 kept (of 25) | 22 | 23 | 23 | 23 | 24 | **25** | 24 | 23 | 21 | 15 |
| Top-25 median market value | $41.3B | 41.3 | 41.3 | 41.3 | 36.2 | **34.4** | 32.4 | 25.8 | 23.6 | 17.8 |

The ranking is size-neutral at a Size weight of **1.2** (1.25 against the Size score, 1.17 against
-ln market cap). Without the category, the other seven lean very slightly toward the largest
companies (-0.039); the 5 points turn that into a small-company lean of **+0.126**.

That reading is not one day's. On the first published payload of every week since 2026-08-20
(`research/measurements/2026-10-20-size-exposure-over-time.py` / `.json`, ten payloads), the
composite's exposure with Size was +0.087 to +0.139 and without it -0.069 to -0.031, and Size
correlated +0.33 to +0.37 with Valuation, +0.27 to +0.28 with Investment and -0.28 to -0.35 with
Growth. (The two early-August payloads, built before the 2026-08-20 rebuild, read differently and
are excluded from those ranges; the JSON keeps them.)

For scale, the composite's exposure to each category on this run: Valuation 0.560, Quality 0.559,
Momentum 0.396, Risk 0.332, Revisions 0.286, Investment 0.219, **Size 0.126**, Growth 0.013.

### 5.4 The top 25 with Size held neutral

Every stock's Size score set to 50 - the category keeps its 5 points and says nothing (the same top 25 as deleting it and
renormalising; Spearman with the published ranking 0.9831 against 0.9832):

| | Published | Size neutral |
|---|---|---|
| Names changed | - | **3 of 25** (6 of 50) |
| Spearman with the published ranking | 1 | 0.983 |
| Median / 90th-percentile / largest rank move | - | 17 / 47 / 65 places |
| Top-25 median market value | $34.4B | $41.3B |
| Top-25 mean Quality / Valuation / Growth / Momentum / Risk / Investment | 74.8 / 75.0 / 52.4 / 80.6 / 52.5 / 53.1 | 76.0 / 73.0 / 57.7 / 78.3 / 51.7 / 53.2 |

Leave: **AIZ** (rank 18 -> 27; $14.0B, Size 92.8), **SWK** (23 -> 35; $14.2B, Size 90.9, Quality
47.8), **ES** (25 -> 31; $24.7B). Enter: **WDC** (28 -> 23; $136.6B), **CL** (32 -> 24; $75.0B,
Quality 89.6), **BKNG** (40 -> 25; $123.5B).

### 5.5 What it promotes

The 50 stocks Size moves up most against the 50 it moves down most (rank with Size neutral minus
published rank):

| | Promoted | Demoted |
|---|---|---|
| Median market value | $15.1B | $323.2B |
| Quality | **47.0** | 56.9 |
| Risk score | 47.5 | 63.6 |
| Valuation | **64.7** | 39.0 |
| Investment | 59.2 | 44.3 |
| Growth / Momentum | 39.5 / 41.9 | 54.5 / 59.4 |
| Volatility / beta | 0.337 / 0.621 | 0.289 / 0.608 |

The same pattern the 2026-08-31 note found: Size, taken alone, buys cheaper, lower-quality, more
volatile and slower-growing companies - and part of its 5 points is a second value bet (the
composite's Valuation exposure is 0.560 with Size and 0.502 without it). At the top of the list
the 22 Quality points mostly absorb the junk: the top 25's mean Quality falls by 1.2
points because of Size (0.2 points across the top 50 on 2026-08-31).

### 5.6 What any change to the weight would do (pre-flight, recorded for whoever reopens this)

The reading rules allow at most 3 points per step. Every step from 5 down to 2, with the freed
points sent pro rata or to one category (`candidate_grid` in the JSON):

| Size | Freed points to | Size exposure | Growth exposure | Top 25 kept | Leave -> enter |
|---|---|---|---|---|---|
| **5** | - (published) | **+0.126** | **+0.013** | 25 | - |
| 4 | pro rata | +0.092 | +0.023 | 24 | ES -> WDC |
| 3 | pro rata | +0.058 | +0.032 | 23 | SWK, ES -> WDC, CL |
| 3 | Quality (25) | +0.054 | +0.038 | 24 | SWK -> CL |
| 3 | Valuation (25) | +0.079 | +0.006 | 25 | none |
| 2 | pro rata | +0.024 | +0.042 | 23 | SWK, ES -> WDC, CL |
| 2 | Quality (25) | +0.017 | **+0.050** | 23 | SWK, ES -> CL, WDC |
| 2 | Valuation (25) | +0.056 | +0.003 | 24 | AIZ -> DLTR |
| 0 | pro rata | -0.039 | **+0.061** | 22 | AIZ, SWK, ES -> WDC, CL, BKNG |
| 0 | Quality (27) | -0.050 | **+0.073** | 22 | AIZ, SWK, ES -> WDC, CL, BKNG |

Three things a later session would need to know:

- **Size is one of the counterweights that keeps the ranking growth-neutral.** It correlates
  -0.325 with Growth, so every cut raises the composite's growth exposure; removing the category
  puts it at +0.061 to +0.073, outside the 0.05 band that
  `tests/test_growth_category.py::test_the_ranking_really_is_close_to_growth_neutral` holds the
  published run to, and Size 2 with the points to Quality sits exactly on the edge. A Size cut is
  therefore also a Growth decision, and the Growth review (2026-10-13) refused to take a growth
  position the evidence does not settle.
- **The freed points have no neutral home.** Sending them to Valuation is the only destination
  that keeps the ranking growth-neutral, and it does so by adding value exposure (0.560 to 0.588
  at Size 2) - a second change, to a category whose large-company evidence is itself mixed
  (`research/2026-10-12-value-and-momentum-among-large-companies.md`).
- **The tests that pin the number 5** (none of them a reason to keep it, all of them work for
  whoever changes it): `tests/test_config_validation.py::test_empty_config_uses_defaults` (the
  default is `schemas.py`'s `size: float = 5`); the hand-written "holding it to 5% of the
  composite" sentence in `run_screener.build_screener_overview` (the section title reads the
  config, the sentence does not); and
  `tests/test_factor_lab.py::test_the_scorecard_is_computed_and_carries_the_screeners_own_weights`,
  which fails until `lab_data.js` is rebuilt with the new weight.

## 6. Where the literature, the lab and practice disagree, and why

**The lab and the unconditional literature agree**: among large companies there is no size
premium you could find without knowing it was there - a century-long sign, about 1.7% a year
between quintiles 4 and 5, and nothing distinguishable since 1981. Alquist et al.'s exclusion
table, Fama & French's 1992 deciles and their 2008 "big" slope all point the same way.

**Practice mostly disagrees** - MSCI, FTSE Russell, STOXX and Dimensional all put a smaller-company
tilt into their large-cap multi-factor products - and the reasons it gives are not the
unconditional premium:

1. **It is the conditional premium they are buying.** Every provider that includes size runs it
   beside a quality or profitability tilt, and MSCI's own brochure names the junk problem as the
   reason it reweights within large and mid caps rather than buying small caps. That is Asness et
   al.'s result put into practice: size is worth holding *only* where junk is controlled. The
   literature supports exactly that, and the lab can see the mechanism (section 2.3: the smaller
   of the large are less profitable two years in three) but cannot yet test the payoff among
   large companies on public data - the library's size-by-profitability 5x5 file would (section 9).
2. **Diversification against the cap-weighted market.** A multi-factor index held against a
   cap-weighted parent tilts away from its largest names; MSCI caps how far, rather than whether.
   S&P DJI's own attribution says most of what the Equal Weight index has earned since 2003 came
   from that tilt run within sectors.
3. **Cost.** At the large end, size costs almost nothing to hold (Asness et al.'s half-spread of
   0.1% for the biggest quintile), so a modest tilt is cheap even if its premium is small.

**The managers who leave it out** (AQR, S&P's QVM, J.P. Morgan, Research Affiliates) give the
lab's reason: no unconditional premium among large stocks, and size "on its own" is not a factor.
Note that AQR's own JFE paper is the strongest case *for* conditional size - their large-cap fund
leaves it out, which says a careful practitioner can read the conditional result and still decline
to harvest it long-only among large companies. That is a defensible position too.

**Why the disagreement is genuine:** the conditional premium is measured long-short across the
whole market with a quality factor that shorts junk; a long-only large-cap ranking cannot short
anything, and nobody has published the conditional test among large stocks alone. So practice is
making a judgement the literature has not yet confirmed or refuted for this universe - and so is
the screener.

## 7. How it fits the composite, and what the 5 points are for

Stated plainly, so the methodology can say it and a student can check it:

- **It is not a return premium the screener can document.** The page's "known weakness" sentence
  stands and is the right one.
- **It is the composite's small-company tilt, and it is small.** +0.13 against the Size score; the
  second-smallest of the eight exposures; three names in the top 25. MSCI's DMF *requires* a tilt
  of at least 0.1 and up to 0.6 standard deviations, and FTSE Russell gives size a fifth of its
  tilt. The screener's 5 points sit at the gentle end of the practice that includes size.
- **It relies on Quality.** The conditional literature is the only case for it, and that case is
  "size, given quality". The composite holds 22 Quality points, and the S&P 500's own profitability
  rule screens the universe; at the top of the ranking Size costs 1.2 points of mean Quality.
  If Quality's weight were ever cut, this reasoning would have to be redone.
- **It overlaps Valuation and Investment, and offsets Growth.** It correlates +0.36 with Valuation
  and +0.26 with Investment, so part of the 5 points is a cheapness-and-conservative-investment
  bet in another form; and its -0.33 with Growth makes it one of the categories keeping the ranking
  growth-neutral. Both are measured, stable since August, and none is a reason the category
  exists - they are what it does in the whole.

A Wednesday synthesis question this leaves (not a proposal): whether the methodology page should
say what the 5 points are *for* in the terms of this section - a gentle within-sector small tilt,
run beside Quality, of the kind most large-cap multi-factor indexes carry - rather than "the
small-cap premium", which section 3 shows the evidence does not support among large companies. The
page's question line ("Does this stock benefit from the small-cap premium?") is the sentence that
overstates. That is a wording change to `run_screener.build_screener_overview`, with its test,
and belongs to a session that can rebuild and check the page.

## 8. Not decision-grade colour (CLAUDE.md rules 4 and 5)

The research engine's "Size alone" screen led the current weighting in 2012-2019 (+0.027, t 1.83)
and trailed it in the held-out 2023-07..2026-08 months (-0.015, t -0.53), and it won 68% of the
overfitting check's in-sample trials (`research/2026-10-20-backtest-overfitting.md`). Both readings
sit on backtest v2's survivorship-limited universe, in which the members that later left the index
- disproportionately small ones that did badly - are missing from the early years, which is the
bias that would flatter a within-S&P size screen most. Nothing in this note rests on them; they
are listed because a reader will have seen them, and they are consistent with the lab's sign flip
after 2013.

## 9. Decision, and what would reopen it

**The weight stays at 5%.** Under the reading rules written before this question was researched,
a change needs the lab and the literature or practice pointing the same way; here the lab and the
unconditional literature point against a premium, the conditional literature and most large-cap
multi-factor practice point toward a modest quality-controlled tilt, and the screener's 5 points
are that modest tilt. Cutting it would also move the ranking off growth-neutral (section 5.6) and
put its freed points on a category that needs its own case. **No proposal is open**, and none of
the steps in section 5.6 is recommended.

What would reopen it - any one of these, written now so the reading cannot be bent later:

1. **The conditional test among large companies fails on public data.** Ken French's 25
   size-by-operating-profitability portfolios (`25_Portfolios_ME_OP_5x5`, not yet in
   `data/lab/french/`) allow Asness et al.'s Table 3 double sort with profitability: size quintile
   4 minus 5 inside each profitability quintile. Queued as a lab experiment (`plan/factor-lab.md`,
   question 4). **Refutes the case for the weight** if, over 1963-2026, the average of the five is
   no larger than the unconditional Q4 - Q5 spread (0.136% a month, value-weighted, section 2.1),
   *or* its t-statistic is below 1.5 - i.e. if holding profitability fixed does not strengthen size
   among large companies the way it does across the market. Then the only case left is
   diversification, which does not need a scored category (and the weight should be researched
   toward zero with Growth's neutrality handled in the same note).
2. **Quality's weight is cut.** The conditional case depends on it (section 7).
3. **The survivorship rebuild** (`research_page.size_paragraph`, queued with its prediction) shows
   the within-S&P size relation in 2012-2019 was survivorship: that removes no evidence this note
   uses, but it is the second measured reason to expect nothing from size here.
4. **Hou & van Dijk (2019) or Israel & Moskowitz (2013), once read**, report a large-company size
   result in either direction. Neither was obtainable tonight.

## 10. Sources

Read in this session (PDFs via pypdf; filings and methodologies from the issuers' own sites or
EDGAR):

- Fama, E.F. & French, K.R. (1992). The cross-section of expected stock returns. *Journal of
  Finance* 47(2), 427-465. Table II Panel A, Table VI.
- Fama, E.F. & French, K.R. (2008). Dissecting anomalies. *Journal of Finance* 63(4), 1653-1678.
  Table IV.
- Asness, C., Frazzini, A., Israel, R., Moskowitz, T.J. & Pedersen, L.H. (2018). Size matters, if
  you control your junk. *Journal of Financial Economics* 129(3), 479-509. Published (open-access)
  version, Tables 1-3, 8-9, Fig. 6. (The 2026-08-31 note quoted the January 2015 working paper; the
  figures it used agree with the published Tables 1-2.)
- Alquist, R., Israel, R. & Moskowitz, T.J. (2018). Fact, fiction, and the size effect. *Journal of
  Portfolio Management* 45(1). AQR reprint; Exhibits 1-2, 10-14, 18-23.
- MSCI (2022). MSCI Diversified Multiple-Factor Indexes Methodology, May 2022 (sections 2.3-2.4,
  Appendices III, V, VI); MSCI Low Size Indexes Methodology (Sept 2018) and brochure; MSCI Size
  Tilt Indexes Methodology (Feb 2023); Oberoi, Rao, Mrig & Subramanian (2016), "One size does not
  fit all", MSCI Research Insight; Menchero, Orr & Wang (2011), Barra USE4 Methodology Notes.
- FTSE Russell (2026). FTSE Global Factor Index Series Ground Rules v8.8 (rules 5.2, 5.5, 6.1-6.3,
  Tables one and three); J.P. Morgan Global Diversified Factor Equity Index Series Ground Rules
  v2.8.
- S&P Dow Jones Indices: S&P Quality, Value & Momentum Multi-Factor Indices Methodology (Dec 2025);
  S&P U.S. Indices Methodology (July 2026, "Financial Viability"); S&P 500 Equal Weight Index FAQ
  (2026); Edwards, Ganti & Preston (2024), "Worth the Weight"; Ganti & Lazzara (2022),
  "Concentration within Sectors and Its Implications for Equal Weighting".
- AQR Large Cap Multi-Style Fund prospectus (485BPOS, Jan 2025); Dimensional ETF Trust 485BPOS
  (Feb 2026); iShares U.S. Equity Factor ETF summary prospectus (497K, Nov 2025) and its March
  2022 index-change supplement; Goldman Sachs ActiveBeta Equity Indexes Methodology (April 2024);
  RAFI Multi-Factor Index Series rulebook v5.3 (Solactive, April 2026).
- Data: Kenneth R. French Data Library (CRSP 2026-08 vintage) - size, 5x5 size-by-book-to-market
  and size-by-prior-return, 2x3 size-by-profitability, five-factor and momentum files; Jensen,
  Kelly & Pedersen (2023) factor returns (CC BY-NC 4.0), `market_equity`.

**Not read** (abstracts only, said so where used): Hou & van Dijk (2019), *RFS* 32(7), 2850-2889;
Horowitz, Loughran & Savin (2000), *JEF* 7(2), 143-153; van Dijk (2011), *JBF* 35(12),
3263-3274; Israel & Moskowitz (2013), *JFE* 108(2), 275-301; Kalesnik & Beck (2014), Research
Affiliates. The STOXX rulebook and Axioma's handbook could not be obtained.

Measurements: `research/measurements/2026-10-20-size-composite.py`,
`research/measurements/2026-10-20-size-exposure-over-time.py`,
`research/measurements/2026-10-20-size-lab.py`, each with its JSON output. No backtest number and
no figure from `live_ic_history.csv` or `performance_history.csv` is used as evidence.
