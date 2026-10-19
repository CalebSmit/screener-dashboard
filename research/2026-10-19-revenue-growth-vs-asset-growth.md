# Revenue growth against asset growth: the screener's position is on neither (2026-10-19)

**Question.** The first of the open Growth composition questions left by the 2026-10-13 refusal
(`research/2026-10-12-growth-category.md` section 11): *the sign of `revenue_growth` against
`asset_growth` - two categories, one fact, opposite directions.*

Stated precisely, because the loose version hides the decision. The screener holds:

| Metric | Category | Within-category weight | Composite points | Direction |
|---|---|---|---|---|
| `revenue_growth` | Growth (13) | 25% | **3.25** | higher is better |
| `revenue_cagr_3yr` | Growth (13) | 15% | **1.95** | higher is better |
| `asset_growth` | Investment (5) | 100% | **5.00** | **lower** is better |

Two categories, each independently weighted, scoring correlated measures of the same year's
expansion in opposite directions. Nobody chose the net position that results; it is the residue
of two separate decisions. The question is whether that residue is defensible.

**Answer, in one paragraph.** The strong form of the complaint is wrong: within sector - the only
frame the engine ever scores in - the two measures share **24.2%** of their variance, so 75.8% of
each is its own, and 30.0% of the universe sits in the two cells where a company grew one without
the other. But the weights are such that the pair is **not a position on either level**: counting
both legs, its composite points correlate **+0.045** with revenue growth and **-0.736** with asset
growth, while correlating **+0.892** with the *within-sector spread between them*. So the screener
pays 3.25 points for a position on revenue growth that it does not hold, and what it holds
instead is a bet on **growth in asset turnover** - revenue rising faster than the balance sheet
that produced it. That is a real and separately-documented signal, not a double count. **No weight
change is proposed.** What is wrong is that nothing - not the methodology page, not the changelog,
not either category's rationale - says this is the position, and the Investment section states a
premium for large companies that its own cited factor does not establish for them.

Everything below is measured on the **2026-10-19 02:00 run** (501 stocks; 497 carry both metrics)
and reproducible from two committed scripts:

- `research/measurements/2026-10-19-revenue-growth-vs-asset-growth.py` (and its `.json`)
- `research/measurements/2026-10-19-investment-factor-by-leg.py` (and its `.json`)

No forward return, no IC from this system's history and no backtest number is used as evidence
anywhere in this note (CLAUDE.md rules 4 and 5). The factor-portfolio figures in section 4 are
public-library replications of published research, the same class of evidence the methodology page
already cites for Size and Momentum.

---

## 1. Both legs are on a verified 12-month window, so the spread is like-for-like

Worth establishing first, because a spread between two differently-windowed measures would be
arithmetic on noise, and `revenue_growth` had exactly that defect until 2026-10-09 (its
"year-over-year" spanned **12 to 23 months** depending on the company's fiscal calendar -
`research/2026-10-09-revenue-growth-window.md`).

Both are now pinned:

- **`revenue_growth`** is the latest quarter against the same quarter a year earlier, with the
  latest fiscal year against the one before as the fallback. Both branches are exactly 12 months
  (changelog 2026-10-09).
- **`asset_growth`** is the most recent quarterly balance sheet against column 4 - the same quarter
  a year earlier - and `factor_engine.py:1349-1365` **refuses the comparison** unless the gap
  between the two populated columns is 330 to 400 days, in which case every `*_prior` field reads
  `NaN` rather than silently comparing three months. Fixed in the same 2026-10-09 pass.

One asymmetry remains and is benign: `revenue_growth` is a *flow* compared across 12 months,
`asset_growth` a difference between two *levels* 12 months apart. Both describe the same 12
months, which is what the spread needs.

## 2. How much the two actually overlap: 24%, not "the same fact"

The 10-12 note measured one number - Spearman **+0.51** between the raw values, universe-wide -
and read it as "for most companies two categories argue about the same fact". Re-measured
(**+0.5083**, so the figure stands), but the reading does not, for two reasons.

**First, the universe-wide number is the wrong frame.** The engine percentiles every metric inside
the stock's GICS sector; a correlation pooled across sectors partly measures the fact that some
sectors are capital-heavy and others are not, which the scoring never sees. Ranking inside each
sector first and then pooling:

| Measure | Spearman |
|---|---|
| Universe-wide, raw values | +0.5083 |
| Universe-wide, the **scored percentiles** | -0.4925 (negative: Investment inverts its metric) |
| **Within sector, raw values** (the frame that scores) | **+0.4924** |

**Second, +0.49 is not "the same fact".** It is **24.2%** shared variance. A sector-demeaned
regression of each scored percentile on the other gives the same number from both directions -
beta -0.490 / -0.493, R² 0.242 - so **75.8% of each metric is not explained by the other**. Two
measures with three-quarters of their variation unshared are correlated, not redundant.

And the population is genuinely spread. Splitting each metric at its own sector's median:

| | asset growth low | asset growth high |
|---|---|---|
| **revenue growth low** | 177 | 75 |
| **revenue growth high** | 74 | 171 |

**149 of 497 stocks - 30.0% - grew one without the other.** The off-diagonal is not a thin noisy
residual; it is nearly a third of the universe, and it is the economically interesting third.

## 3. The decisive measurement: the pair is a position on the spread, not on either level

This is the section that answers the question, and it is the one the 10-12 note did not run.

Take each stock's actual scored percentiles, convert them to composite points at the configured
weights (`pct x 3.25/100` for `revenue_growth`, `pct x 5.00/100` for `asset_growth`), and add the
two. That sum is exactly what the pair contributes to the composite - it ranges from 0.63 to 7.80
points across the 497 stocks, standard deviation 1.28. Then ask what it is a position *on*:

| The pair's composite points, against... | Spearman |
|---|---|
| raw revenue growth | **+0.0446** |
| raw asset growth | **-0.7363** |
| **the within-sector rank spread** (revenue rank − asset rank) | **+0.8920** |
| raw revenue growth minus raw asset growth | +0.7406 |
| *for comparison:* the Investment leg **alone**, against asset growth | -0.9227 |
| *for comparison:* the revenue leg **alone**, against revenue growth | +0.8857 |

Read the first and the last rows together. **On its own the revenue leg is a +0.89 position on
revenue growth. Once the asset-growth leg is added, that becomes +0.04.** The 3.25 points buy
essentially no net exposure to revenue growth, because the asset leg is both larger (5.00 points)
and +0.49 correlated with it, so it taxes fast revenue growers almost exactly as fast as the
revenue leg rewards them. The asset leg, by contrast, keeps most of its own position: -0.92 alone
becomes -0.74 in the pair.

But the pair is not simply "the Investment category with extra steps" either, and the +0.892 row
is why. **The pair tracks the spread better than it tracks either level** - better than it tracks
asset growth alone (-0.736), which is the leg that survives. The two legs are not cancelling to
nothing; they are cancelling to a third quantity:

> **revenue growing faster than the assets that produced it** - in accounting terms, growth in
> **asset turnover** (sales ÷ assets).

The four cells make the same point in units a reader can feel. Median values:

| Cell | n | pair's points | Growth score | Investment score | composite |
|---|---|---|---|---|---|
| revenue up, assets flat (**capital-light growth**) | 74 | **5.90** | 61.7 | 74.5 | 53.41 |
| assets up, revenue flat (**capital without demand**) | 75 | **2.57** | 39.3 | 35.3 | 48.58 |

A 3.3-point swing in composite points, and a 4.8-point swing in the finished composite, between
two cells of near-identical size. The construction *works*; it is simply not the construction
either category claims to be making.

**Counting the third leg does not change the conclusion, and corrects the arithmetic.** The
headline "3.25 against 5.00, so a net short of 1.75 on growth" is wrong, because
`revenue_cagr_3yr` is also a top-line measure scored higher-is-better. Top-line is **5.20 points
against asset growth's 5.00** - near-balanced by accident. With all three legs counted the points
correlate +0.172 with revenue growth, +0.305 with the 3-year CAGR and -0.564 with asset growth:
the same shape, slightly tilted toward the top line.

## 4. What the Investment category's 5 points actually stand on

The methodology page's Investment section (`SCREENER_OVERVIEW.md` §8) is five lines long. It says
CMA "captures the historical tendency for companies that invest conservatively to outperform those
that aggressively expand their asset base", gives **no figure, no sample and no large-company
caveat** - where the Size section carries a thorough one ("No published study establishes a size
premium *within* the largest two market-cap deciles") and the Momentum section carries another.
That asymmetry is not defensible, because the same standard applied to Investment is not kind to
it.

From the lab's own French-library data, 758 months (1963-07 to 2026-08),
`research/measurements/2026-10-19-investment-factor-by-leg.py`:

| Investment factor, leg | 1963-2026 | since 2000 | since 2014 |
|---|---|---|---|
| among **small** companies | +0.338%/mo (t **+4.61**) | +0.263 (t +2.15) | -0.038 (t -0.22) |
| among **big** companies | +0.149%/mo (t **+1.46**) | +0.181 (t +1.03) | -0.121 (t -0.45) |
| **big, long leg only** (conservative minus market) | **+0.132%/mo (t +2.22)** | +0.156 (t +1.52) | +0.045 (t +0.29) |
| big, short leg only (market minus aggressive) | +0.017 (t +0.32) | +0.025 (t +0.28) | -0.166 (t -1.23) |

Three things to take from this, and the third is the one that matters most and is the least
obvious:

1. **Among big companies the investment factor is not distinguishable from zero over 63 years**
   (t 1.46). Small companies earn **2.27x** the big-company figure. The large-company conditioning
   is real and the page does not mention it.
2. **Since 2014 it is negative** among big companies (-0.121%/mo). By the reading rules written
   before any of these questions were researched (`plan/factor-lab.md`), twelve years inside two
   standard errors is "not confirmed", not "gone" - so this is not a reason to cut the weight.
3. **But this screener is long-only, and the long leg is the half that clears the bar.**
   Conservative-investment big companies beat the market by **+0.132%/month at t 2.22**; the short
   leg contributes +0.017 at t 0.32. The weak long-short figure is the *short* leg adding variance
   without mean. For a book that shorts nothing, the relevant number is the one that passes.

   This is not special pleading for Investment, which is why the measurement prints all four
   factors. Over the full sample **all four long legs clear |t| >= 2** - value +2.12,
   profitability +2.19, investment +2.22, momentum +2.96 - while **only two of the four
   long-short factors do**. The long side is systematically the better-supported half among big
   companies, and a long-only large-company screener is entitled to say so.

So the honest statement about Investment's 5 points is narrower and better-founded than the page's
current one: *the half of the investment effect a long-only holder keeps is distinguishable from
zero among large companies over the full sample, at about 0.13% a month; the long-short factor is
not, and the effect is more than twice as large among small companies, which this screener does
not own.*

## 5. Does the finished ranking inherit the position?

The pair is 8.25 of 100 composite points, so the coherence question (CLAUDE.md, "Build a coherent
screener, not a pile of good ideas") is whether the other 91.75 wash it out.

| The published composite, against... | Spearman |
|---|---|
| raw revenue growth | **-0.0039** |
| raw asset growth | **-0.2030** |
| the within-sector rank spread | **+0.2017** |

The ranking is **exactly neutral on revenue growth** - which is the same answer the Growth weight
review reached from the other direction (the composite's correlation with the *Growth score* is
-0.012, `research/2026-10-12-growth-category.md`), reached here from the raw metric instead of the
category. And it carries a real, modest position on the spread.

The pair is the main channel but not the only one. Each category score against the spread:

| Category | Weight | vs the spread |
|---|---|---|
| Investment | 5 | **+0.4687** |
| Growth | 13 | **+0.3231** |
| Momentum | 13 | +0.2100 |
| Revisions | 10 | +0.1827 |
| Valuation | 22 | -0.0823 |
| Risk | 10 | -0.0307 |
| Size | 5 | -0.0303 |
| Quality | 22 | -0.0226 |

Momentum and Revisions lean the same way, which is unsurprising and worth stating: a company whose
revenue is outgrowing its balance sheet tends to have had rising estimates and a rising price. So
the screener's exposure to asset-turnover growth is reinforced by 23 further points, not isolated
in the 8.25 - and **Valuation and Quality, the two 22-point categories, are flat on it** (-0.08,
-0.02), which is the reassuring half: the position is not a disguised value or quality bet.

---

## 6. The literature on asset growth: which measure, and does it survive among large companies

Three papers were obtained and read from the PDFs; every figure below was spot-checked against
the extracted text by the author of this note, not taken on a reader's report. Page numbers are
the published journal pages except where marked.

- **Cooper, Gulen & Schill (2008)**, "Asset Growth and the Cross-Section of Stock Returns",
  *Journal of Finance* 63(4), 1609-1651. DOI [10.1111/j.1540-6261.2008.01370.x](https://doi.org/10.1111/j.1540-6261.2008.01370.x)
- **Fama & French (2008)**, "Dissecting Anomalies", *Journal of Finance* 63(4), 1653-1678.
  DOI [10.1111/j.1540-6261.2008.01371.x](https://doi.org/10.1111/j.1540-6261.2008.01371.x)
  *(read as the December 2006 working draft; the published text was not obtained - see 6.4)*
- **Lipson, Mortal & Schill (2011)**, "On the Scope and Drivers of the Asset Growth Effect",
  *Journal of Financial and Quantitative Analysis* 46(6), 1651-1682.
  DOI [10.1017/S0022109011000561](https://doi.org/10.1017/S0022109011000561)
- **Fama & French (2015)**, "A five-factor asset pricing model", *Journal of Financial Economics*
  116(1), 1-22. DOI [10.1016/j.jfineco.2014.10.010](https://doi.org/10.1016/j.jfineco.2014.10.010)
- **Hou, Xue & Zhang (2015)**, "Digesting Anomalies: An Investment Approach", *Review of Financial
  Studies* 28(3), 650-705. DOI [10.1093/rfs/hhu068](https://doi.org/10.1093/rfs/hhu068)
- **Fama & French (2016)**, "Dissecting Anomalies with a Five-Factor Model", *Review of Financial
  Studies* 29(1), 69-103. DOI [10.1093/rfs/hhv043](https://doi.org/10.1093/rfs/hhv043)

### 6.1 The screener computes the measure that has the large-company evidence

This turned out to be the most consequential single fact in the literature, and it is a point in
the screener's favour that nobody had written down.

CGS, p. 1613: *"our main variable of concern, the annual firm asset growth rate (ASSETG), is
calculated using the year-on-year percentage change in total assets (Compustat data item 6)"*.
That is **exactly** what `factor_engine.py:2969` computes: `(totalAssets - totalAssets_prior) /
totalAssets_prior`. Same quantity, same one-year window, same scaling by beginning assets.

It is **not** the measure Fama & French (2008) use. Their `dA/A` is the log change in *assets per
split-adjusted share*, which they state (draft p. 23) *"is equivalent to the natural log of the
ratio of gross assets at t-1 divided by gross assets at t-2 **minus net stock issues** from t-2 to
t-1"* - deliberately, so that their separate net-share-issues variable can carry the issuance
(draft p. 7). That distinction decides the whole large-cap dispute, and Lipson, Mortal & Schill
proved it by running both measures on one sample with the same breakpoints and weighting
(Table 3, p. 1662; equal-weighted quintile spreads, big stocks):

| Measure | Big-stock spread | t |
|---|---|---|
| FF's per-share `dA/A` | 0.25%/month | 1.71 |
| **CGS's total-asset growth** | **0.67%/month** | **4.04** |

LMS, p. 1663: *"The distinction between the 2 findings is based on the definition of the measures
and in particular the dampening effect caused by the per share normalization. This normalization
effectively ignores growth associated with share issuance, which we find impacts growth measures
for large firms more than small firms."* CGS's own decomposition says the same from the other end
(p. 1638): among large firms *"growth in stock financing exhibits the strongest effect"*, and with
stock financing in the regression asset growth itself goes to t = -0.56.

So the honest reading is **not** "FF found no asset-growth effect in big stocks, therefore the
screener's 5 points are unsupported." FF measured a different thing. The thing the screener
measures has three independent large-company confirmations:

| Source | Large-company result | Where |
|---|---|---|
| CGS (2008) | VW 3-factor alpha spread **-0.81%/month, t -2.91** (~-10%/year); EW -0.86%, t -3.12 | Table II Panel C, p. 1620 |
| CGS (2008) | Fama-MacBeth slope **-0.0594, t -3.60** in the large-firm panel | Table III Panel D, p. 1630 |
| LMS (2011) | Big-stock EW quintile spread **0.67%/month, t 4.04**; VW regression slope -0.008, t -3.54 | Table 3 Panel C p. 1662; Table 2 Panel B Reg 8 p. 1660 |

CGS state it plainly (p. 1620-1624): *"Thus, the asset growth effect is present across all size
groups"*, and in the introduction, *"the annualized three-factor alpha VW portfolio spread between
low and high growth rate firms for the large size group is 10%."*

### 6.2 But it is roughly half-size among large companies, and FF2015's own big-stock leg is weak

Two counterweights, both of which belong on the page.

**The effect declines monotonically in size on every measure in every paper.** CGS's
Fama-MacBeth slopes run -0.0944 (small) -> -0.0793 (medium) -> **-0.0594 (large)**. CGS's own
headline: *"firms with low asset growth rates earn subsequent annualized risk-adjusted returns of
9.1% ... while firms with high asset growth rates earn -10.4%. The large 19.5% per year spread is
highly significant. Weighting the firms by capitalization reduces the spread to a still large and
significant 8.4% per year"* (p. 1647). LMS: 1.37% -> 0.77% -> 0.67%.

**And FF2015's CMA - which *does* use total-asset growth, not the per-share measure - still finds
the big-stock leg weak.** This is a genuine disagreement with CGS and must not be smoothed over.
FF2015 defines `Inv` as *"the growth of total assets for the fiscal year ending in t-1 divided by
total assets at the end of t-2"* (p. 4) and reports the factor split by size (Table 4 Panel A,
p. 7, 606 months, July 1963 - December 2013):

| Construction | CMA | t | CMA small | t | **CMA big** | **t** |
|---|---|---|---|---|---|---|
| 2x3 | 0.33 | 4.07 | 0.45 | 5.49 | **0.22** | **2.00** |
| 2x2 | 0.22 | 3.72 | 0.33 | 5.37 | **0.11** | **1.50** |
| 2x2x2x2 (joint controls) | 0.14 | 2.71 | 0.23 | 4.64 | **0.07** | **1.03** |

And in their own words (p. 8, verified against the PDF): *"there is strong evidence that the
expected investment premium is larger for small stocks. The average value of CMA_S is 4.64-5.49
standard errors from zero, but the average value of CMA_B is only 1.03-2.00 standard errors from
zero ... and **almost all the average value of CMA is from small stocks.**"*

**This independently corroborates the lab measurement in section 4.** Section 4 measured the
big-company investment factor at t 1.46 on the French library's own portfolios; FF's published
range for the same object is t 1.03-2.00. Two routes, one answer. The lab is reproducing the
paper, which is what it is for.

### 6.3 The shape: an extreme-tail effect, which a linear percentile score is the wrong instrument for

This is the finding with the most direct consequence for the screener, and it is derived from
CGS's table rather than stated by them - so here is the row, read off Table II Panel C.2, p. 1620
(value-weighted 3-factor monthly alphas, large size group, decimal in the paper):

| Decile | 1 (low) | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 (high) |
|---|---|---|---|---|---|---|---|---|---|---|
| alpha | **+0.52%** | +0.18 | +0.13 | +0.19 | +0.03 | +0.05 | +0.18 | +0.15 | +0.08 | **-0.28%** |

The end-to-end spread is 0.80pp. **Deciles 2 through 9 span 0.16pp** - a fifth of it - and are not
monotone (they go 0.18, 0.13, 0.19, 0.03, 0.05, 0.18, 0.15, 0.08). The whole large-company effect
is the two end deciles against a flat interior. Fama & French find no systematic gradient for big
stocks even in the extremes (draft p. 13): *"only tiny stocks produce a systematic negative
relation between asset growth and abnormal returns."*

The screener scores `asset_growth` as a **linear sector percentile** - every rank step worth the
same points. That spends almost all of its 5 points' resolution on the interior, where the
large-company evidence is flat, and compresses the two tails where it lives. It also explains part
of the CGS/FF2015 disagreement: FF's CMA uses 30/70 breakpoints, which average the tails into
thirds; CGS uses extreme deciles. An extreme-tail effect shows up in the second design and washes
out in the first.

**This is a candidate finding, not a change.** Moving `asset_growth` from a linear percentile to a
tail-weighted treatment is a methodology change that needs its own note, its own synthesis against
the other seven categories, and a changelog entry - and it would make the screener harder to
explain, which the mandate counts as a real cost. It is recorded here, with the row above, as the
strongest open question this note found.

### 6.4 The limit the published evidence does not reach: large *and* low-volatility

LMS's central finding is not about size at all, and it is the one that should temper any
confidence here. Their abstract (p. 1651): *"In general, there appears to be no asset growth
effect in firms with low IVOL."* Table 7 Panel C, p. 1672, equal-weighted spreads by
idiosyncratic-volatility quintile:

| IVOL quintile | 1 (low) | 2 | 3 | 4 | 5 (high) |
|---|---|---|---|---|---|
| low-minus-high asset growth | **0.1% (t 1.02)** | 0.6 (4.48) | 1.0 (6.67) | 1.2 (7.72) | 1.7 (7.47) |

Monotone across all five - stronger evidence than their size result, which is not monotone. And in
regression (Table 4 Reg 2, p. 1665): with an asset-growth x IVOL interaction included, *"the
coefficient on the asset growth rate becomes insignificant ... (-0.002 [t-statistic = -0.62]),
suggesting that the asset growth effect exists only when in conjunction with IVOL."* They conclude
(p. 1679) that *"the firm IVOL is a necessary condition for asset growth effects."*

**Large-company stocks are disproportionately low-IVOL, and the cell that matters - large *and*
low-IVOL - is reported by none of these papers.** LMS say the IVOL gradient survives a size control
but the test is unreported (footnote 19, p. 1673): *"In unreported tests, we use 3-way sorts ... we
still observe that the asset growth effect is increasing in the degree of IVOL."* That tells us the
gradient holds; it does not give the level in the cell an S&P 500 screener occupies.

Three further honest limits:

- **"Large" in all three papers is far broader than the top 500.** CGS: above the 70th NYSE
  percentile. FF and LMS: above the NYSE median, which FF say is *"more than 90% of total market
  cap"* (draft p. 9). The S&P 500 sits deep inside the upper tail of even CGS's large group, where
  every paper's gradient is weakest.
- **Reward-to-variability in the low-IVOL quintile is 0.04** (LMS Table 8, p. 1674) against 0.21,
  0.31, 0.35, 0.35 for the rest. Even where a spread exists in calm stocks, it is not a spread an
  investor is paid for bearing.
- **The published FF (2008) text was not obtained.** It was read as the December 2006 draft from
  ivey.uwo.ca. Its sample (1963-2005) and headline figures match the published abstract, and LMS
  quote the published pages 1653 and 1655 second-hand, but the draft pagination is marked as such
  above and no published page is cited from it in this note.

**Net reading for the Investment weight.** The measure the screener uses is the one with
large-company evidence; the effect there is about half the all-stock size, concentrated in the two
end deciles, and the published work does not establish it in the large-low-volatility corner this
screener actually occupies. By the reading rules fixed before these questions were researched
(`plan/factor-lab.md`) that is "not confirmed for this universe", which is **not** grounds to cut
the weight - and the long-only half measured in section 4 (+0.132%/month, t 2.22) is the half a
book like this keeps. **5 points stays.** What must change is what the page claims for it.

---

## 7. The literature on sales growth: a different dimension, and an unrewarded one

### 7.1 The modern factor models classify sales growth away from investment, explicitly

This is the cleanest answer to "two categories, one fact" that the literature gives, and it is
structural rather than a matter of interpretation.

**Hou, Xue & Zhang (2015)** sort roughly 80 anomalies into six categories (Table 2, p. 664). Their
**Panel C, "Investment"**, holds 14 variables including `I/A` - *"the annual change in total assets
... divided by 1-year-lagged total assets"* (p. 659), the same quantity the screener computes.
**Sales growth is not in it.** `SG` is filed in **Panel B, "Value-versus-growth"**, beside
book-to-market, cash-flow-to-price, dividend yield and analysts' long-term growth forecasts, and
attributed to Lakonishok, Shleifer & Vishny (1994). The appendix files it the same way
(A.1.2.10, p. 689, under "Value-versus-growth", not A.1.3 "Investment").

**Fama & French never mention sales growth at all.** Grepped over the full extracted text of
FF2015: **zero occurrences of "sale"** (the word "revenue" appears only inside the operating-
profitability definition). FF2016's anomaly list is fixed and short - market beta, net share
issues, total variance, residual variance, accruals, momentum (pp. 73-75) - and sales growth is
absent. So FF2016 cannot be cited for sales growth in either direction, and this note does not.

**The theory has no revenue term.** HXZ's q-theory (eq. 4, p. 655) prices expected return as
expected profitability divided by `1 + a(I/A)`: *"low discount rates imply high marginal q and
high investment, and high discount rates imply low marginal q and low investment"* (p. 655). The
right-hand side is cash committed to assets. FF's version is an accounting identity on book equity
(FF2015 p. 2): *"for fixed values of B, M, and expected earnings, higher expected growth in book
equity - investment - implies a lower expected return"*; sales appear in none of equations (1)-(3).
Revenue is not a subtraction from the cash flow available to shareholders, so on either derivation
it cannot be the investment term. HXZ extend the investment channel to net stock issues, composite
issuance, accruals, valuation ratios and long-term reversal (p. 655) - **not** to sales growth.

A detail worth having, because it cuts the other way: HXZ call asset growth *"the most
comprehensive measure of investment-to-assets ... the premier manifestation of the investment
channel"* (p. 656), citing CGS. The screener's Investment category is, by this literature's own
account, the right single metric for the dimension it names.

### 7.2 What sales growth earns: nothing measurable - but check which construction

**HXZ Table 3, "Insignificant anomalies in the broad cross section" (p. 667), value-weighted
high-minus-low decile: `SG` m = -0.27% per month, t = -1.34.** Sales growth is one of the 38 of
~80 anomalies they cannot distinguish from zero, which also means it never reaches their q-factor
model - there is nothing for the investment factor to explain.

**CGS put sales growth and asset growth in the same regression and asset growth wins outright**
(Table III, pp. 1628-1630). All firms, with sales growth included: asset growth t = **-7.41**,
sales growth t = **-0.27**. And the sentence that matters most for this note, verified verbatim
from the PDF (p. 1631):

> *"In fact, asset growth appears to subsume many of the other growth-related variables, especially
> within the large firms: In the regressions with asset growth, capital investment is not
> significant for large size firms; **5-year growth rate in sales is not significant within any of
> the size groups**; and the accrual effect is not significant within medium and large firms."*

Large-firm panel, sales-growth coefficient: -0.0132, t = -0.68 (Table III Panel D, p. 1630).

### 7.3 The construction caveat, stated rather than buried

**Both papers test the LSV five-year weighted rank, not a one-year growth rate.** CGS's `5YSALESG`
is *"a 5-year weighted average rank of growth rate in sales"* (Table III note, p. 1628), with years
-5, -4, -3, -2 weighted 10/20/30/40%. HXZ's `SG` is the same object: *"the weighted average of the
annual SG ranks for the prior 5 years"* (p. 689).

The screener's `revenue_growth` is **one year**, latest quarter against the same quarter a year
earlier. Its `revenue_cagr_3yr` is a **three-year CAGR**. Neither is the five-year weighted rank
these papers test.

This matters and is the sort of thing this project has got wrong before - the 2026-10-14 session
recorded the lesson as *"a figure from a paper printed beside a number the paper did not measure"*.
So, precisely:

- The null on **five-year** sales growth is well established and holds in every size group (CGS
  p. 1631; HXZ p. 667). That bears directly on `revenue_cagr_3yr` (1.95 points), which is the
  nearest of the screener's legs to the tested construction, and indirectly on `revenue_growth`.
- The null on **one-year** sales growth specifically is *not* what these two papers establish.
  What they establish is that a longer-horizon sales-growth rank adds nothing once asset growth is
  in the regression, and that the factor-model literature does not treat sales growth as an
  investment-dimension variable at all.
- Nothing in either paper supports scoring revenue growth **positively**. The sign of every
  sales-growth coefficient reported above is negative (CGS -0.0041 all firms, -0.0132 large; HXZ
  -0.27%/month) - small and insignificant, but never positive. The screener scores it the other
  way, which is the direction the evidence is least able to support even though it cannot reject
  it.
