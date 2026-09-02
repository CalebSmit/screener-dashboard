# The size factor in a large-cap-only universe

**Date:** 2026-08-31
**Author:** nightly code session (research day)
**Status:** complete. No production code changed by this note.

**Wednesday's synthesis happened on 2026-09-02** —
`research/2026-09-02-category-independence-synthesis.md`. Disposition of the
three candidates below:

- **Candidate 3 (winsorise-before-rank, Finding C):** shipped 2026-09-01,
  verified live 2026-09-02. The six megacaps now publish six distinct market
  caps. Closed.
- **Candidate 2 (`size` and `investment` — one bet or two?):** answered
  **two**. They correlate +0.280 but are 73% and 84% unspanned by the other
  seven categories respectively. Not a disguised 10% bet; merging them would
  lose information. Closed, no change.
- **Candidate 1 (is the size tilt more aggressive than intended, Finding B?):**
  **still open.** Deferred deliberately, not forgotten — synthesis day found a
  larger overlap (momentum ~ risk at +0.516, ~25% of composite weight) and
  spent itself there. The refutation criterion below still stands and should be
  measured before any code is written.

---

## The question

The screener spends **5% of composite weight** on a single metric, `size_log_mcap`
= `-ln(market cap)`, ranked within sector, higher = smaller = better. The size
premium is the most heavily challenged of the classic factors, and this screener
applies it inside the S&P 500 — a universe with no small caps at all.

**Does a size tilt belong in an S&P 500 screen, and is this the right way to
build one?**

Answering it needs three things: what the literature actually found, how index
providers and quant shops actually implement size, and what this screener's tilt
measurably does to its own output.

---

## Part 1 — What this screener does today (measured)

All figures below are computed from `dashboard_data.js` as published on `main`
from the 2026-08-31 02:00 data run (502 stocks, 501 with a market cap).
Reproduce them by ranking on `raw.size_log_mcap` and recomputing the composite
from `cat_scores` and `weights.factor_weights`.

### The universe has real size dispersion, but no small caps

| Percentile | Market cap |
|---|---|
| min | $8.2B |
| 25th | $23.7B |
| median | $46.0B |
| 75th | $98.8B |
| 95th | $419B |
| max | $2,874B (winsorized — see finding C) |

The full universe is $66.7T. The **smallest 250 names are 9.0% of it**. So
"small" here means an $8–25B company, which in the size literature is a large-cap
stock — it sits in roughly the top two deciles of the US market by capitalization.
Nothing in this universe is a small cap, and nothing is remotely a microcap.

### The tilt is not cosmetic

Dropping the size category and renormalising the other seven weights:

| Measure | Value |
|---|---|
| Spearman (ranking with vs without size) | 0.9826 |
| Median absolute rank change | **16 places** |
| 90th percentile | 48 places |
| Maximum | 70 places |
| Names entering/leaving the top 50 | **5** |

The largest names are the ones it penalises: JNJ, MSFT, RTX, CAT, NEE and BRK-B
each rank ~60–68 places lower than they would without it. A 5% weight is doing
real work.

### It is a genuine, independent bet

Regressing the size score on the other seven category scores gives **R² = 0.279**
— **72% of the size score is not spanned by the rest of the screener**. Whatever
else is true, it is not a redundant restatement of value or quality.

Sector-relative ranking does not neutralise it either: the size score still
correlates **+0.948 (Spearman) with −market cap** universe-wide, because sector
cap ranges overlap heavily. It really is a size bet.

### And it is a junk bet, exactly as the literature predicts

Comparing the 50 stocks the tilt promotes most against the 50 it demotes most:

| | Promoted by the size tilt | Demoted |
|---|---|---|
| Median market cap | **$15.0B** | $264.1B |
| Mean quality score | **49.3** | 55.6 |
| Mean risk score | **44.9** | 60.1 |
| Mean volatility | **0.33** | 0.30 |
| Mean growth score | 39.9 | 57.2 |
| Mean momentum score | 42.6 | 62.5 |
| Mean valuation score | **63.6** | 35.7 |

The tilt buys cheaper, lower-quality, more volatile, slower-growing, weaker-momentum
stocks. That is the *precise* pattern the size literature identifies as the reason
the raw size premium fails (Part 2) and that MSCI warns about in its own product
literature (Part 3).

One genuine surprise: promoted names have **lower beta** (0.58 vs 0.76) despite
**higher volatility** (0.33 vs 0.30). Within today's S&P 500 the mega-caps are the
high-beta names; smaller index members carry more idiosyncratic and less market
risk. Any claim that this tilt "adds market risk" would be wrong — it adds
*specific* risk.

---

## Part 2 — What the literature says

### The original finding

**Banz, R. (1981), "The Relationship Between Return and Market Value of Common
Stocks," *Journal of Financial Economics* 9(1), 3–18.** Small US stocks earned
higher risk-adjusted returns than large ones, 1936–1975. Reported spread between
extreme size quintiles: **~7.19%/yr equal-weighted, ~2.73%/yr value-weighted**
(figures as summarised in the secondary literature cited below, not read from
Banz directly — treat the exact decimals as approximate, the ~2.6x gap between
weighting schemes as the substantive point).

That gap is itself the first warning: most of Banz's effect lives in the
smallest, least investable names, which equal-weighting overrepresents.

### The challenges, and what survived them

**Asness, C., Frazzini, A., Israel, R., Moskowitz, T., and Pedersen, L. (2018),
"Size matters, if you control your junk," *Journal of Financial Economics*
129(3), 479–509.** This is the decisive paper. Numbers below are from the January
2015 working paper (US sample, July 1926 – December 2012), which is the version
I was able to read in full; the published version's tables may differ in detail.

Raw SMB is weak:

| Specification | SMB alpha (bps/month) | t-stat |
|---|---|---|
| Raw excess return, full sample 1926–2012 | 23 | **2.27** |
| Over Banz's own original sample | — | **insignificant** |
| CAPM alpha | 12 | 1.12 |
| FF3 + UMD alpha | 14 | 1.23 |
| **FF3 + UMD + QMJ (quality) alpha** | **49** | **4.89** |

Controlling for quality **triples the alpha and takes t from 1.23 to 4.89.** The
mechanism is that SMB loads strongly and negatively on QMJ — small stocks are
junky, and the junk drags the premium to zero.

Corroborating detail from the same paper:

- **Seasonality.** Raw size effect is 2.09%/month in January (t = 5.59) and
  **−0.04% outside January (t = −0.32)** — i.e. no size effect for eleven months
  of the year. Adding QMJ produces **+38 bps outside January (t = 3.62)**.
- **Within quality quintiles.** Small-minus-big averaged across quality quintiles
  = **50 bps/month, t = 3.18**. But **among the junkiest quintile the relation is
  not even monotonic** and the small-large difference (39 bps) is insignificant.
- **FF5 factors.** Adding RMW (profitability) and CMA (investment) — SMB loads
  negatively on both — doubles SMB's alpha to **33 bps, t = 2.81**. So the
  investment and profitability factors capture part, but not all, of what QMJ does.
- **Not concentrated in microcaps** once quality is controlled; holds across 30
  industries and 24 international markets.

**The significance bar matters here.** The paper itself invokes **Harvey, Liu and
Zhu (2016), "...and the Cross-Section of Expected Returns," *Review of Financial
Studies* 29(1), 5–68**, which argues that given the volume of factor data-mining,
**t > 3.0** is the appropriate threshold. Raw SMB's full-sample t of 2.27 does not
clear it. Quality-controlled SMB (t = 4.89) does. This is not a marginal
distinction — it is the difference between a factor you should and should not use.

**Alquist, R., Israel, R., and Moskowitz, T. (2018), "Fact, Fiction, and the Size
Effect," *Journal of Portfolio Management* 45(1), 34–61** reaches the same
conclusion in practitioner framing: most standalone claims for the size effect are
"fiction"; what survives is size *conditioned on quality*.

### Conditions the effect held under — and where they fail here

The quality-controlled size premium is documented on the **full CRSP
cross-section**, sorted into size deciles or quintiles spanning micro to mega cap.
**No paper cited here establishes a size premium within the top two deciles of US
market cap**, which is the entire S&P 500. Asness et al.'s "not concentrated in
microcaps" claim means the effect is not *exclusively* microcap — it is not a
demonstration that it survives among the 500 largest US companies alone.

**This is the single most important limitation for this screener, and I could not
close it from the literature.** Searches for direct evidence on a size premium
*within* large caps returned commentary rather than primary research. The honest
statement is: the screener is extrapolating a cross-sectional effect into the
narrowest slice of the cross-section, and the literature does not confirm that
extrapolation.

---

## Part 3 — What practice does

### MSCI treats size as something to tilt gently, not to score aggressively

The **MSCI Low Size Indexes** brochure documents the construction: at each
rebalance, each stock is weighted **in proportion to the inverse natural logarithm
of its market cap**, applied to the **large and mid cap universe**, reweighting
rather than excluding. MSCI's own worked example:

| Stock | Market cap | ln(cap) | 1/ln(cap) | Low Size weight |
|---|---|---|---|---|
| ABC | $90B | 25.2 | 0.040 | **47.7%** |
| XYZ | $10B | 23.0 | 0.043 | **52.3%** |

**A 9x difference in market cap produces a 4.6 percentage point difference in
weight.** MSCI states the rationale explicitly: capitalisation is asymmetrically
distributed, and the log transform is chosen precisely because it "minimized the
impact of the largest values."

MSCI also names the junk problem in its own sales literature — small-cap indexes
capture the size premium but do so "at the expense of relatively poorer quality
and more volatile stocks than the broader market," and "low quality and high
volatility have historically detracted from returns over long horizons." An index
provider is conceding the point in a brochure for the product.

### Barra treats size as a risk factor to neutralise, not an alpha source

In the **Barra US Equity Model (USE4)** and its predecessors, Size is one of a
small set of **style risk factors**. The standard institutional use is to *measure
and control* size exposure so it does not accidentally dominate a portfolio's
risk — "style factor stripping," pure factor portfolios with zero exposure to
everything but the target. Size appears in the risk model of essentially every
quant shop; it appears as an intentional alpha bet in far fewer.

This is the sharpest academia/practice divergence, and it is discussed in Part 4.

### The S&P 500 itself already screens out junk

**S&P Dow Jones Indices' S&P 500 eligibility criteria require positive as-reported
GAAP earnings in the most recent quarter *and* a positive sum over the trailing
four quarters.** S&P has publicly declined to waive this even for mega-cap
candidates. This is a genuine profitability gate at the universe boundary, and it
means the "small = junk" mechanism is **structurally weaker here than in the CRSP
cross-section**, where the small deciles are full of loss-making microcaps. It is
a point in this screener's favour that the literature alone would not surface.

### The closest live analogue: the S&P 500 Equal Weight Index

This is the most directly relevant piece of documented practice, because it *is* a
within-S&P-500 size tilt with a multi-decade live track record.

- Since 1990, equal weight has beaten cap weight by roughly **63 bps/year**.
- S&P DJI and others attribute this to a **smaller size bias, a value orientation,
  an anti-momentum bias, and broader sector diversification** — note that three of
  those four match the tilt profile I measured in Part 1 almost exactly.
- **It has been brutal recently.** Over 2023–2025 cap weight beat equal weight by
  roughly **32%** — reported as among the largest three-year relative
  outperformances on record, exceeding the ~31% seen in the run-up to the tech
  bubble. The top 10 names reached ~41% of index weight by end-2025.

Two lessons. First, a within-large-cap size tilt is a *real, practised strategy*,
not an invention of this screener. Second, its payoff is regime-dependent and its
drawdowns run for years. A 63 bps/year edge that can lose 32% over three years is
not something a 5% composite weight should be expected to deliver reliably.

---

## Part 4 — Where academia and practice disagree, and why

**They disagree on whether size is an alpha factor at all.**

The academic literature, after Asness et al., says there is a real,
quality-conditioned size premium worth harvesting — 49 bps/month with t = 4.89 is
economically on par with value and momentum.

Practice is more equivocal. MSCI sells a Low Size index but designs it to be a
*gentle* reweighting with deliberately marginal exposure to other factors. Barra
— the same firm — models Size primarily as a risk to be controlled. Many quant
managers neutralise size rather than bet on it.

**Why the divergence is genuine and not just conservatism:**

1. **The academic premium is a long-short, quality-controlled, monthly-rebalanced
   construct.** Practitioners run long-only portfolios where you cannot short the
   junk that the paper's QMJ control implicitly removes. The clean academic
   premium is not directly investable.
2. **Capacity and cost.** The premium is strongest where liquidity is worst. A
   long-only large-cap manager cannot access the part of the cross-section where
   the effect is largest.
3. **Size is a huge risk exposure relative to its expected return.** As a risk
   factor it explains a large share of cross-sectional variance; as an alpha
   factor it pays ~50 bps/month conditional on a quality control most portfolios
   cannot fully implement. That asymmetry is exactly why it lives in the risk
   model.

**Both sides agree on one thing, and it is the operative point for this
screener:** an uncontrolled size tilt buys junk, and that is what kills it.
Academia proves it with QMJ regressions; MSCI states it in a brochure. There is no
serious dissent.

---

## Part 5 — Where the evidence contradicts what this screener does

### Finding A — The composite already controls for junk, and this is the screener's strongest defence

This is the good news, and it is the finding I least expected.

The tilt is junk-seeking *on average* (Part 1). But the screener does not act on
the size score in isolation — it sums it with quality at 22% weight. Comparing the
published top 50 with and without the size category:

| Category | Top-50 mean, with size | Without size | Change |
|---|---|---|---|
| Quality | 68.91 | 69.11 | **−0.20** |
| Risk | 60.86 | 61.33 | **−0.47** |
| Growth | 53.89 | 55.97 | −2.09 |
| Momentum | 77.71 | 79.62 | −1.92 |
| Valuation | 71.85 | 69.39 | +2.46 |
| Median market cap | $30.4B | $38.5B | −$8.1B |

**The size tilt shifts the top 50 toward smaller names without meaningfully
degrading its quality (−0.20) or risk (−0.47).** The 22% quality weight has
already removed the junk before size gets to promote anything. A linear composite
with a heavy quality term is a crude but functioning analogue of Asness et al.'s
double sort — and here it demonstrably works.

Combined with the S&P 500's own GAAP-profitability entry gate, this screener is
running something much closer to the *quality-controlled* size premium
(t = 4.89) than to the raw one (t = 1.23). **That is a defensible design, and it
was not obviously so before this note.**

The caveat is that this holds *at the top of the ranking*, which is what the
product surfaces. Mid-table, the junk-seeking behaviour in Part 1 is real.

### Finding B — The `log` in `size_log_mcap` does nothing. Provably.

The pipeline computes `-ln(mcap)`, then converts it to a **percentile rank**
(`compute_sector_percentiles`, `factor_engine.py:2305`). Percentile ranking is
invariant to any monotone transform, so ranking `-ln(mcap)` gives byte-identical
results to ranking `-mcap` or `-sqrt(mcap)`:

```
max |rank(-log mcap) - rank(-mcap)|      : 0.0000000000
max |rank(-log mcap) - rank(-sqrt mcap)| : 0.0000000000
```

This matters because **the log is the entire mechanism by which MSCI makes its
size tilt gentle**, and the screener discards it one line later. The consequence
is a far more aggressive tilt than the practitioner standard it superficially
resembles. For comparable ~9–10x market cap gaps within one sector:

| | Cap ratio | Resulting gap |
|---|---|---|
| MSCI Low Size (weights) | 9x | **4.6 pp** |
| Screener: CAT $368B vs UAL $36B | 10x | **57 points** (score 1 vs 59) |
| Screener: LLY $1,047B vs MCK $104B | 10x | **29 points** (score 2 vs 31) |
| Screener: AAPL $2,874B vs PANW $303B | 9.5x | **15 points** (score 3 vs 18) |

These are not the same strategy. The screener's `size` category is a linear-in-rank
tilt, roughly an equal-weight-style bet; MSCI's is a heavily compressed one. The
metric name asserts a compression the pipeline does not perform.

Note this is a **description, not yet a defect**. Rank-based scoring is a
legitimate, defensible design used consistently across all 44 metrics, and it is
what makes the categories commensurable. The problem is narrower: the tool's name
and documentation imply a log compression that does not survive to the score.

### Finding C — Winsorising before ranking destroys information and cannot add any

Found while chasing the size metric; it **generalises to all 44 metrics**, so it
is flagged here and belongs to a future session.

`winsorize_metrics(df, 0.01, 0.01)` runs at `factor_engine.py:3429`, immediately
before `compute_sector_percentiles` at :3433. Winsorising clips the extreme 1% to
a common value. Since the very next step is a rank, and ranks are already immune
to outliers, this step:

- **cannot change any rank ordering** — the standard justification for
  winsorising (outlier control) is already provided by the rank transform;
- **can only create ties**, collapsing the ordering of the most extreme names.

Measured on the published payload: **27 continuous metrics show the 1%/99% tie
signature, collapsing 282 (stock, metric) cells** into artificial ties. (Discrete
metrics like `piotroski_f_score` and `consecutive_beat_streak` have genuine ties
and are excluded from that count.)

For size specifically, the six largest companies — **AAPL, NVDA, GOOG, GOOGL,
MSFT, AMZN** — all receive an identical winsorized market cap of $2,873.8B and are
therefore treated as exactly the same size, despite spanning a wide true range.
The same six-way tie exists at the bottom (TTD, AOS, TAP, NCLH, BLDR, MOS).

There is a **second, user-facing consequence**: the winsorized values are what the
dashboard publishes as `raw`. The site currently shows six different companies
with the same market capitalisation. That is a display correctness issue, not just
an internal one.

I have **not** changed this today — it is outside the size question, it touches
every metric in the screener, and it deserves its own session and its own
changelog entry.

### Finding D — `proximity_52w_high` is not the only zero-weight metric worth revisiting

Not pursued today; noted so it is not lost. `size` and `investment` are both
single-metric categories carrying 5% each. Asness et al.'s result that RMW and CMA
absorb part of SMB's alpha (33 bps, t = 2.81) implies **size and investment are
not independent bets** — my measured correlation between the two category scores
is **+0.281**, the second-highest of any pair involving size. Wednesday's
synthesis should look at them together rather than separately.

---

## What would change my mind

Falsifiable, in order of how much they would move the conclusion:

1. **Direct evidence on a size premium within the top two US market-cap deciles.**
   If a credible study shows the quality-controlled size premium vanishes among
   the largest 500 names, the 5% weight should go to zero regardless of anything
   in Part 5. I could not find this evidence either way today.
2. **Evidence that the top-50 quality neutralisation in Finding A is regime-
   dependent.** It is measured on one day's payload. If in a different market the
   size tilt starts dragging top-50 quality down by several points rather than
   0.20, the "the composite controls its junk" defence weakens.
3. **A live IC series long enough to test it.** Currently **3 effective
   observations at the `1m` horizon** against a gate of 8, so this is roughly five
   more months away and is not evidence today.

---

## Recommendation

**Keep the size category at 5% weight. Do not change it today.** Three reasons:

1. It is a genuinely independent bet (72% unspanned by the other seven).
2. The junk problem that invalidates the raw size premium is **measurably
   neutralised where the product actually points** — top-50 quality moves by
   −0.20 — by the 22% quality weight and the S&P 500's own GAAP-profitability
   entry gate. The screener is closer to the t = 4.89 version of the factor than
   the t = 1.23 version.
3. A within-large-cap size tilt is documented practice with a live track record
   (S&P 500 Equal Weight, +63 bps/yr since 1990), not an invention of this tool.

**But three things should be recorded as known weaknesses**, and the tool should
say so rather than implying more confidence than the evidence supports:

- No cited source establishes the premium **within** a large-cap-only universe.
  This is an extrapolation and should be labelled as one.
- The tilt is **much more aggressive than the practitioner standard** it resembles
  (Finding B), and nothing has justified that aggressiveness.
- The strategy's payoff is **strongly regime-dependent** — the closest live
  analogue lost ~32% relative over 2023–2025.

**Explicitly rejected today:** raising the size weight on the strength of Asness
et al.'s 49 bps/t = 4.89 result. That premium is measured on a long-short,
quality-controlled, full-cross-section portfolio. This screener is long-only,
large-cap-only, and quality-controlled only incidentally. Importing the effect
size would be exactly the "pile of good ideas" failure `CLAUDE.md` warns against.

---

## Wednesday's design section (synthesis)

Three candidate changes, in priority order. **None is justified by this note
alone** — Wednesday's job is to decide whether they fit the screener as a whole.

**1. Make the aggressiveness of the size tilt a deliberate choice (Finding B).**
Hypothesis: the current linear-in-rank tilt is stronger than any cited source
supports, and was never chosen — it fell out of applying the standard rank
pipeline to a metric whose log was meant to compress it. Sketch: either apply the
existing `config.yaml` non-linear percentile transform to `size_log_mcap` to
compress the tails toward MSCI's shape, or state plainly in
`SCREENER_OVERVIEW.md` that this is an equal-weight-style tilt rather than a
log-compressed one and keep it. **Refuted if** the compressed version changes the
top 50 by fewer than ~2 names, in which case it is complexity for nothing.

**2. Look at `size` and `investment` as one design question (Finding D).**
They are 10% of composite between them, one metric each, and correlate +0.281.
Asness et al. show CMA absorbs part of SMB's alpha. Are these two 5% bets or one
10% bet?

**3. Winsorise-before-rank (Finding C) — its own session, not Wednesday's.**
Touches all 44 metrics and has a user-facing display consequence. Needs a
changelog entry, a test that the published `raw` market caps are distinct, and a
decision about whether winsorisation should survive at all given that every metric
is rank-transformed.

---

## Sources

Primary:

- Banz, R.W. (1981). "The Relationship Between Return and Market Value of Common
  Stocks." *Journal of Financial Economics* 9(1), 3–18.
- Fama, E.F. and French, K.R. (1992). "The Cross-Section of Expected Stock
  Returns." *Journal of Finance* 47(2), 427–465.
- Fama, E.F. and French, K.R. (2015). "A Five-Factor Asset Pricing Model."
  *Journal of Financial Economics* 116(1), 1–22.
- Asness, C., Frazzini, A., Israel, R., Moskowitz, T., and Pedersen, L.H. (2018).
  "Size Matters, If You Control Your Junk." *Journal of Financial Economics*
  129(3), 479–509. Working paper (Jan 2015) read in full at
  https://jacobslevycenter.wharton.upenn.edu/wp-content/uploads/2015/05/Size-Matters-if-You-Control-Your-Junk.pdf
  — **all quoted t-statistics and alphas are from that working-paper version.**
  Published version: https://www.sciencedirect.com/science/article/pii/S0304405X18301326
- Alquist, R., Israel, R., and Moskowitz, T. (2018). "Fact, Fiction, and the Size
  Effect." *Journal of Portfolio Management* 45(1), 34–61.
  https://jpm.pm-research.com/content/45/1/34
- Harvey, C.R., Liu, Y., and Zhu, H. (2016). "…and the Cross-Section of Expected
  Returns." *Review of Financial Studies* 29(1), 5–68. (Source of the t > 3.0
  bar; cited via Asness et al.)

Practice:

- MSCI. *MSCI Low Size Indexes* brochure — weighting formula, worked example,
  and the quality/volatility caveat. Read in full.
  https://www.msci.com/documents/1296102/8473352/MSCI-Low-Size-Index-Brochure.pdf
- MSCI. *The Barra US Equity Model (USE4) Methodology Notes* — Size as a style
  risk factor.
  https://www.top1000funds.com/wp-content/uploads/2011/09/USE4_Methodology_Notes_August_2011.pdf
- S&P Dow Jones Indices. S&P 500 eligibility criteria (positive GAAP earnings,
  most recent quarter and trailing four quarters).
- S&P Dow Jones Indices. *FAQ: S&P 500 Equal Weight Index.*
  https://www.spglobal.com/spdji/en/education/article/sp-500-equal-weight-index-faq/

**Secondary sources, used only where flagged as such in the text** (Banz's
quintile spreads; equal-weight relative performance figures). These are reported
numbers I could not verify against a primary source in this session and should
not be treated as decision-grade on their own.

Internal measurements: `dashboard_data.js` from the 2026-08-31 02:00 data run;
`factor_engine.py:2256` (`winsorize_metrics`), `:2305`
(`compute_sector_percentiles`), `:3429`–`:3433` (call order), `:2182`
(`size_log_mcap`), `config.yaml` `factor_weights`.

**No backtest number appears in this note** (`CLAUDE.md` rule 5) and **no figure
from `live_ic_history.csv` or `performance_history.csv` is used as evidence**
(rule 4).
