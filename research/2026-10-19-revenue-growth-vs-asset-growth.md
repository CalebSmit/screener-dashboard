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
that produced it. **So it is not a double count.** Whether it is a *good* bet is the harder half and
the answer is uncomfortable: **documented practice rewards exactly that quantity** (FTSE Russell
scores change in asset turnover positively inside Quality and uses neither of the screener's raw
legs; S&P frames its quality index the same way), while **the published return evidence neither
supports nor refutes it** - the paper that would settle it is paywalled and was not obtained, the
major replication never tested the change, the related *level* does not price in any of eight
specifications, and the change is 97%-transitory within a year (§9.6). Academia and practice
disagree about the screener's own joint position, so by the reading rule written before any of
this was researched, **the weight stays. No weight change is proposed, and no proposal is left
open.** What is actually wrong is that nothing - not the methodology page, not the changelog, not
either category's rationale - says this is the position; that the Investment section carries no
large-company caveat where Size and Momentum both do; and that the screener's *positive* sign on
one-year revenue growth is against the weight of published evidence and survives only because the
asset leg cancels it to +0.045 - **which is luck, not design, and moved once already when a
correctness fix landed (§8.1).**

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

### 7.4 The two papers that study sales growth directly - and they *do* have large-company evidence

Both obtained and read from the published versions; the figures below were read off the printed
tables by the author of this note (LSV is an image-only scan, so its pages were rendered and read
visually).

- **Lakonishok, Shleifer & Vishny (1994)**, "Contrarian Investment, Extrapolation, and Risk",
  *Journal of Finance* 49(5), 1541-1578.
  DOI [10.1111/j.1540-6261.1994.tb04772.x](https://doi.org/10.1111/j.1540-6261.1994.tb04772.x)
  *(note: the DOI widely cited for this paper, `tb00079.x`, resolves to a different article -
  Rubinstein's "Implied Binomial Trees". Corrected here.)*
- **Chan, Karceski & Lakonishok (2003)**, "The Level and Persistence of Growth Rates", *Journal of
  Finance* 58(2), 643-684. DOI [10.1111/1540-6261.00540](https://doi.org/10.1111/1540-6261.00540)
- **Fairfield, Whisenant & Yohn (2003)**, *The Accounting Review* 78(1), 353-371 - **not obtained.**
  It is the paper that puts accruals and operating-asset growth in one specification, and it has no
  free full text (the only open record is an abstract-only SSRN page). Nothing in this note is
  attributed to it. That is a real gap in this note's coverage, named rather than papered over.

**LSV: high past sales growth predicted *lower* returns, and they tested large companies.** Their
`GS` is *"the weighted average rank, giving the weight of 5 to its growth rank in Year −1, the
weight of 4 to its growth rank in Year −2, etc."* (p. 1550) - so again a five-year rank, not a
one-year rate. Table I Panel D, p. 1549, verified off the page (note Panel D carries its **own
reversed header**: decile 1 is *lowest* GS / Value, decile 10 *highest* GS / Glamour):

| GS decile | 1 (lowest) | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 (highest) |
|---|---|---|---|---|---|---|---|---|---|---|
| avg annual return, 5 yrs | 0.195 | 0.195 | 0.188 | 0.183 | 0.171 | 0.171 | 0.165 | 0.164 | 0.155 | **0.127** |
| **size-adjusted (SAAR)** | +0.022 | +0.027 | +0.025 | +0.024 | +0.015 | +0.015 | +0.008 | +0.008 | 0.000 | **−0.024** |

A 4.6pp size-adjusted spread, in the direction *opposite* to the screener's sign. LSV's own verdict
(p. 1550): *"These magnitudes are not as dramatic as those for the B/M and C/P strategies,
nevertheless the spread in returns is sizeable"* - GS is the weakest of their four one-way sorts
(B/M 7.8pp, C/P 8.8pp, E/P 5.4pp, GS 4.6pp).

**And here is the large-company part, which matters because section 6 made so much of it for the
other leg.** Table III takes the largest 50% of NYSE/AMEX: GS still separates size-adjusted returns
by 2.5-5.8pp a year *within every valuation tercile*, and the extreme-corner (C/P, GS) spread is
**8.7pp - identical to the full sample**. Then, verified verbatim from p. 1557:

> *"We have also done the analysis for the largest 20 percent of the stocks, **which effectively
> mimics the S&P 500**, and get a very similar spread of returns between glamour and value stocks.
> The conclusion is clear: **our results apply to the largest stocks as well.**"*

**No numbers are published for that largest-20% test** - it is an assertion, and must be read as
one. But it is the only statement in any paper found for this note that speaks directly to this
screener's universe, and it says the sales-growth effect is there. In LSV's Fama-MacBeth
regressions (Table IV, p. 1558) `GS` also survives every multivariate specification
(−0.044, t −2.125 with B/M, size and C/P in) while **B/M collapses to zero** - so it is not a
book-to-market proxy.

**CKL: revenue growth is genuinely persistent; the profit it is supposed to represent is not.**
This is the finding that most changes how `revenue_growth` should be read. Percentage of firms
beating the cross-sectional median *every year* for five consecutive years, against 3.1% by chance
(Table III, p. 654; Table IV Panel D, p. 656 for large firms):

| | all firms | **large firms** | chance |
|---|---|---|---|
| **sales** growth | 6.3% | **7.5%** | 3.1% |
| operating income growth | 3.6% | 3.7% | 3.1% |
| income before extraordinary items | 3.0% | **2.2%** | 3.1% |

CKL, p. 655: *"**There is a great deal of persistence in sales growth**... roughly twice more than
expected achieve runs over five years"*, and *"**Any sign of persistence vanishes as we get closer
to the bottom line.**"* Among large firms the earnings figure is *below* chance (p. 657: *"While
sales growth tends to be more persistent for large firms, it does not translate into persistent
growth in income"*). In their forecasting regressions (Table X, pp. 677-678) past five-year sales
growth predicts next-year **sales** growth at t +3.7 and five-year **earnings** growth at
**t −4.2** - the right sign for revenue, the wrong sign for profit.

And CKL name the mechanism themselves, which ties the two halves of this note together (p. 655):

> *"Persistence in sales may also arise from managers' **'empire-building' efforts, such as
> expanding market share regardless of profitability**. In all these cases, however, **profit
> margins are likely to be shrinking as well**, so growth in profits may not show as much
> persistence as sales growth."*

That is the same empire-building story the asset-growth literature tells, reached from the revenue
side. The two metrics are not merely correlated by accident: the literature says persistent revenue
growth and aggressive asset growth are often *the same corporate behaviour*, and that the behaviour
does not reach the bottom line. The screener rewards the first and penalises the second.

### 7.5 Both legs are tail effects, and the screener scores both linearly

Worth putting beside section 6.3, because the two findings converge and neither paper makes the
point. LSV's size-adjusted returns across GS deciles 1-9 span **2.7pp** (+0.027 to 0.000); decile 10
alone drops to −0.024, contributing about **41% of the 4.6pp total spread while being 10% of the
universe**. CGS's large-company asset-growth alphas have the same shape, more sharply: deciles 2-9
span 0.16pp of an 0.80pp spread.

So on both legs, the published effect lives in the extremes and the interior is close to flat - and
the screener scores both as **linear sector percentiles**. LSV draw the implication for their own
variable (p. 1550, that GS *"alone is not an ideal contrarian strategy"*); the implication for a
linearly-scored screen is this note's, not theirs, and it is recorded in section 11 as the strongest
open question rather than acted on.

---

## 8. Is the spread position stable, or one run's artifact?

The Growth weight review left this open about its own headline number
(`research/2026-10-12-growth-category.md` section 11: *"What the ranking's growth exposure does
over time. -0.01 is one run."*). It is a fair challenge to this note too, so it was answered rather
than deferred: every committed `dashboard_data.js` from 2026-09-10 to 2026-10-19 was replayed - 40
payloads - each scored at the weights **its own commit's** `config.yaml` configured
(`research/measurements/2026-10-19-spread-exposure-over-time.py`).

| The claim | min | median | max |
|---|---|---|---|
| the pair's points vs the within-sector spread | **+0.892** | **+0.910** | **+0.911** |
| the pair's points vs raw revenue growth | +0.045 | +0.135 | +0.138 |
| the pair's points vs raw asset growth | -0.737 | -0.736 | -0.736 |
| the finished composite vs the spread | +0.193 | +0.200 | +0.215 |

Flat across all 40 runs. The central claim is a property of the construction, not of a day.

### 8.1 And it dates a step, which a correctness fix caused

Between the two payloads committed on 2026-10-09 the pair's residual exposure to revenue growth
**falls from +0.137 to +0.045**, and the two metrics' raw overlap **rises from +0.477 to +0.508**.

The attribution was checked rather than assumed, because the whole metric audit landed in that
same range. The step is in the correlation between the two **raw metric values**, which only a
change to one of the two computations can move. In that commit range:

- `asset_growth` is untouched in `factor_engine.py` (`git log -S` over the range returns nothing);
- exactly one commit touches `revenue_growth` - **`0a470e4`, "growth: revenue_growth compares
  periods exactly a year apart (latest quarter vs same quarter a year earlier)"** - and it is an
  ancestor of the later payload but not of the earlier one.

So the window fix is the cause, and the effect is the opposite of what anyone would have predicted
from the changelog entry: **putting `revenue_growth` on an honest 12-month window made it measure
the same twelve months as `asset_growth`, so the two correlate more, so the asset leg cancels more
of the revenue leg.** A fix made for correctness moved the pair roughly a third of the way further
toward being a pure spread position, and reduced the net exposure the screener holds to revenue
growth from a small positive to almost nothing.

Nobody intended that, and nobody would have seen it: it is invisible in the changelog, invisible in
the tests, and invisible in any single run. It is the clearest illustration in this note of why the
pair's net position needs writing down - two independently-chosen weights produce a joint position
that moves when either leg's *definition* changes, not only when a weight does.

---

## 9. Documented practice: eleven provider families, and the construction they converge on

Primary methodology documents were obtained and read. The four load-bearing quotations below were
spot-checked against the PDFs by the author of this note. Where a document could not be obtained
it is named as such in 9.5 rather than paraphrased from a summary.

### 9.1 The crux: does anyone put opposite signs on both? Yes - and it is MSCI

**"Datasheet - Barra Global Total Market Equity Trading Model", MSCI, September 2016, p. 4**
([PDF](https://cdn2.hubspot.net/hubfs/2174119/Return%20Downloads/Datasheet%20-%20Barra%20Global%20Total%20Market%20Equity%20Trading%20Model.pdf))
carries both, as two separate style factors in one model - verified verbatim:

| Style factor | MSCI's stated purpose | Descriptors |
|---|---|---|
| **Growth** | *"Measures company growth prospects using sales growth and earnings growth over trailing five years and predicted earnings growth"* | Analyst-Predicted Earnings Long-Term Growth; Earnings per Share Growth Rate; **Sales per Share Growth Rate** |
| **Investment Quality** | *"Combination of asset, investment, net issuance growth measures that captures common variation in stock returns of companies experiencing rapid growth or contraction of assets"* | **Total Assets Growth Rate**; Capital Expenditure Growth; Issuance Growth |

The same split appears in the two US models under the name *Management Quality*
(`barra-usslow-datasheet.pdf` pp. 2-3; `barra-usfast-datasheet.pdf` pp. 3-4), and the full
descriptor tree is drawn in one diagram in **"Anatomy of Active Portfolios", MSCI, July 2017,
p. 5** - verified: *"Investment Quality: Asset growth, Capex growth, Issuance growth."*

So the screener's construction has a commercial precedent. **But MSCI does not leave the two to
fight.** **"Growth Target Indexes Methodology", MSCI, November 2023, §2.4, p. 5** - verified
verbatim:

> *"Exposure to Book-to-Price, Earnings Yield, Long-Term reversal, Dividend Yield, Momentum,
> Profitability, Earnings Quality, **Investment Quality** and Mid-Capitalization factors will be
> constrained to be within the range of **0 to 0.25** standard deviations relative to the Parent
> Index."*

and, in the same list, Beta, Residual Volatility, Earnings Variability, Leverage and Size are
constrained to **-0.25 to 0**. Both bounds are deliberately **one-sided**. MSCI maximises a Growth
score that is 20% sales-per-share growth (Appendix III, p. 10: `Growth = 0.7 EGRLF + 0.1 EGRO +
0.2 SGRO`) while **forbidding the resulting portfolio from taking negative Investment Quality
exposure**. In plain terms: *the growth tilt is allowed, but it may not be financed by asset
growth.* That is a constraint on the joint position - precisely the thing this screener has never
stated and does not control.

### 9.2 Practice converges on the spread - the same quantity section 3 found

The most striking result in the practitioner documents is that where revenue growth survives at
all as a return signal, it survives **scaled by the assets that produced it**.

**FTSE Russell, "FTSE Global Factor Index Series Ground Rules", v8.8, July 2026, rule 5.4,
pp. 10-11** - verified verbatim: *"Quality is defined as a composite of profitability and
leverage"*, and the three profitability measures are Return on Assets, Accruals, and:

> **Δ Asset Turnover = Sales_t / Total Assets_t − Sales_{t−1} / Total Assets_{t−1}**

Measured over all 43 pages of that document: **"asset growth" 0 occurrences, "sales growth" 0
occurrences.** FTSE uses neither of the screener's two metrics. It uses the *change in the ratio
between them*, scores it positively, and puts it inside Quality.

That is the same economic quantity the screener's pair expresses at +0.892, arrived at
independently, by the provider that carries neither raw leg. And the correspondence extends to the
awkward cases: ΔAsset Turnover is zero for a company growing sales and assets in step, and
positive for one that shrinks its balance sheet faster than its sales - exactly how the screener's
spread behaves.

S&P frames its quality screen the same way. **Invesco Exchange-Traded Fund Trust, Form 485BPOS**
(SPHQ) describes the index as selecting companies that *"seek to generate higher revenue and cash
flow than their counterparts **through prudent use of assets and finances**"*, and the metric is
**Accruals Ratio = (NOA_t − NOA_{t−1}) / average total assets**, entered as an **inverted** z-score
("S&P Quality Indices Methodology", March 2025, Appendix A, p. 22: *"Accruals and Leverage are
represented by an inverted Z score... as lower values are preferred"*). Revenue appears as a
numerator over assets, never as a growth rate.

### 9.3 Nobody rewards sales growth as a return factor - and one shop has measured the screener's own construction

**This closes the construction gap section 7.3 left open**, and it is the single most directly
relevant practitioner finding in this note.

**David Blitz, "The many faces of growth", Robeco, April 2025, p. 3**
([PDF](https://www.robeco.com/files/docm/docu-20250424-the-many-faces-of-growth.pdf)) - measured on
the Jensen-Kelly-Pedersen factor library, US, capped value weighting, July 1963 to December 2023.
Verified verbatim:

> *"quality factors such as cash-based operating profitability and net operating assets, which can
> be interpreted as cumulated accruals and investments"* earn *"around **5 to 7% per annum**"*.
> *"By contrast, profitability growth... carries a return of just **2%**, while **sales growth even
> has a negative return of around −2%**. In other words, **companies with high sales growth over the
> previous one or three years tend to have below-average subsequent returns.**"*

**"over the previous one or three years"** - that is the screener's `revenue_growth` (one year) and
`revenue_cagr_3yr` (three years), not the five-year rank CGS and HXZ tested. Same library, same
period, same method, opposite signs on the two legs: the asset side **+5 to 7%**, the sales side
**−2%**. And Blitz's conclusion, p. 4: *"growth is not a solid factor in its own right on a par
with established quant factors... growth factors fail to make it into the top 15."*

Corroboration from elsewhere:

- **MSCI's own research excludes Growth from its rewarded set.** "Foundations of Factor Investing",
  December 2013, p. 7 - verified verbatim: *"there are factors that are significant in explaining
  returns but which do not earn a premium over reasonably long horizons. For instance... **factors
  such as Growth and Liquidity... have not outperformed the market over long periods. These types
  of factors are not considered good candidates for long-term factor investing.**"* MSCI's six
  equity risk premia are Value, Low Size, Low Volatility, High Yield, Quality and Momentum.
  (Reported honestly: MSCI nonetheless *sells* Growth Target Indexes, so its research and its index
  business do not fully agree. Both documents are primary.)
- **FTSE tested growth inside Quality and dropped it.** "Factor exposure indexes - Quality factor",
  FTSE Russell Research, August 2014, p. 14: *"combining growth with profitability has not
  historically resulted in any incremental improvement in the spread between the Sharpe Ratios of
  extreme quintiles."*
- **AQR scores growth, but deliberately not of sales or assets.** "Quality Minus Junk" (Asness,
  Frazzini & Pedersen), Appendix A2: growth is *"the change in the numerator (e.g. profits)
  divided by the **lagged** denominator (e.g. assets)"*. Holding the denominator lagged is exactly
  what stops asset growth contaminating the growth measure - a construction trick worth knowing.
  No sales-growth and no asset-growth variable appears in the paper.
- **MSCI's Value & Growth classification** scores `g = ROE x (1 − payout)` positively ("MSCI Global
  Investable Market Value and Growth Index Methodology", February 2021, Appendix I, p. 30) - the
  sustainable growth rate of book equity, which is the quantity the investment factor penalises.
  But its stated purpose (§1.2.3, p. 7) is to *"divide constituents... each targeting 50% of the
  free float-adjusted market capitalization"* - an exhaustive partition. **Classification, not a
  premium.** The screener's `sustainable_growth` metric is the same `ROE x retention`.

### 9.4 Asset growth: penalised almost everywhere, but housed in Quality, not as a standalone factor

| Provider | Asset growth | Where it lives | Sales growth |
|---|---|---|---|
| Barra / MSCI risk models | **Yes, penalised** | Investment Quality (Quality group, +0.25 of the Quality score) | Yes, in **Growth** |
| MSCI indices | Quality index: **no**; Growth Target: constrained >= 0 | - | +0.2 weight in Growth Target |
| S&P DJI | **Yes in substance** - ΔNOA / avg assets, inverted | Quality ("accruals ratio"), 1/3 | Yes, + - 3-yr sales-per-share growth, 1/3 of the Growth Score (classification) |
| FTSE Russell | **No - none at all** | - | **No** - uses Δ asset turnover instead |
| AQR (QMJ) | No | - | No |
| Dimensional | **Yes, negative** | "investment characteristics", small-cap exclusion | **No - zero occurrences** |
| Robeco | **Yes** | Inside Quality | **No** - measured at **−2%/yr** |
| Research Affiliates | **Yes, subtracted** | Half of Quality: *"Quality = average of Profitability minus the average of Investment"* | No in core RAFI; yes in a **separate** growth index |
| Northern Trust | **Yes** - capex and financing penalised | Quality ("Management Efficiency") | No; growth actively **minimised** as a style exposure |
| Scientific Beta | **Yes** - standalone **Low Investment** factor | Its own factor, 1 of 6 | **No - absent entirely** |
| Invesco | Via licensed FTSE / S&P scores | Inside Quality | No |

Two things follow for this screener's design.

**First, almost nobody runs asset growth as a standalone 5%-weight category.** Scientific Beta is
the one clear exception ("Low Investment", 1 of 6 factors). Everyone else who uses it puts it
**inside Quality**, alongside profitability - Research Affiliates most explicitly: *"Quality measure
is the average of Profitability minus the average of Investment... Investment is the average of the
z-scores of asset growth and book growth"* (RAFI Multi-Factor rulebook v5.3, §2.8.1, p. 10). And
Hsu, Kalesnik & Kose, "What Is Quality?", *Financial Analysts Journal* 75(2), 2019, p. 57
([DOI 10.1080/0015198X.2019.1567194](https://doi.org/10.1080/0015198X.2019.1567194)): *"The
combination of the profitability and investment signals is the most parsimonious quality definition
and is associated with the strongest academic evidence."* The screener's separate Investment
category is a minority construction - defensible, but it should be known to be one.

**Second, practice confirms the large-company caveat from the other side.** Dimensional, "What Does
High Asset Growth Say About a Company's Expected Return?", 21 July 2020: *"the return spreads are
mainly driven by the strong and reliable underperformance of small cap stocks with high investment.
**The return spreads in large cap stocks, while directionally consistent with valuation theory, are
generally not reliable.**"* Their implementation matches: the exclusion is small-cap-only and capped
at 5% of the eligible small-cap universe, applied that way even inside all-cap indices, and they
have built no large-cap high-asset-growth product. Two further decay measurements, both from
Research Affiliates: Low Asset Growth monthly long-short 0.30% (t 3.97) pre-publication against
**0.07% (t 0.48) post-publication** ("What Is Quality?" Table 6, p. 56, with their own
noise caveat); and the investment factor's annualised return 5.4% for 1963-2003 against **0.3%**
after ("Alice's Adventures in Factorland", *JPM* 45(4), 2019, p. 32).

Against that, Scientific Beta publishes Low Investment long-leg Carhart alpha **+1.66%** and
short-leg **−1.40%** on the **largest 500 US stocks**, 1975-2015 ("The Dimensions of Quality
Investing", May 2017, Exhibit 5, p. 17) - the closest published figure to this screener's universe,
and it supports the long leg, consistent with section 4's +0.132%/month at t 2.22.

### 9.5 Where academia and practice disagree, and where two practitioners disagree with each other

The mandate asks for this explicitly, and there are three disagreements worth recording rather than
resolving.

1. **Academia vs practice on housing.** The literature treats investment as a *standalone* factor
   (Fama-French's CMA, Hou-Xue-Zhang's I/A, both of which stand in the model beside profitability).
   Practice overwhelmingly folds it **into Quality**. The reason practice gives is overlap: CMA's
   correlation with HML is 0.70 (FF2015 Table 4 Panel C), and Hsu, Kalesnik & Kose find
   *"investment has a weaker multifactor alpha because it is correlated with the value factor"*
   (p. 54). The screener keeps it standalone, which is the academic construction - and this project
   has already measured that overlap at 0.70 across all stocks and 0.52 among big companies
   (`plan/factor-lab.md` question 3). **That question is now answered for the asset-growth half:
   the overlap is real, it is why practitioners merge the two, and merging is a live option the
   screener has never considered.**
2. **Two top-tier quant shops, both publishing, genuinely disagreeing about sales growth.**
   Research Affiliates, "Outgrowing Glamour: A Fundamental Approach to Growth Investing", April
   2026, p. 1: *"**Income statement growth rewards shareholders through increasing sales and
   profits, while balance sheet growth often reflects capital accumulation that does not
   necessarily translate into shareholder value**"*, and p. 2, on book-value and dividend growth,
   *"these types of growth do not add value... Balance sheet growth, in particular, can be
   backward-looking or capital-intensive, while income-statement growth is economically scalable."*
   Against Robeco's measured **−2% a year** for sales growth at one and three years. **RA is the
   best available written defence of the screener's pair of signs; Robeco is the best available
   refutation of its positive sales-growth leg.** Both are primary; neither is decisive; the
   disagreement is the finding.
3. **MSCI's research against MSCI's index business**, noted in 9.3. Its research says Growth has
   not earned a premium and is *"not considered a good candidate for long-term factor investing"*;
   its index arm sells Growth Target Indexes described as providing *"diversification to a factor
   portfolio or a standalone single factor strategy."* A reader should know that a provider's
   product line is not evidence about a premium.

### 9.6 But is asset turnover actually a *return* signal? The evidence is thinner than §9.2 suggests

This subsection exists because the answer qualifies §9.2 sharply, and finding it out changed this
note's conclusion. §9.2 establishes that practice **rewards** change in asset turnover. It does not
establish that doing so is right.

- **Soliman (2008)**, "The Use of DuPont Analysis by Market Participants", *The Accounting Review*
  83(3), 823-853, [DOI 10.2308/accr.2008.83.3.823](https://doi.org/10.2308/accr.2008.83.3.823) -
  the paper that would answer this - **was not obtained.** Paywalled at the AAA, the only recorded
  open location is an SSRN page that serves a bot check, and the author's faculty page hosts no
  PDF. **No number is quoted from it anywhere in this note.** Treat any ΔATO return figure
  attributed to it elsewhere as unverified.
- **Fairfield & Yohn (2001)**, *Review of Accounting Studies* 6(4), 371-385, is closed access and
  was also not obtained.

What *was* obtained and read changes the picture:

**1. The annual asset-turnover level does not price - in Soliman's own variable.** Hou, Xue & Zhang
(2020), "Replicating Anomalies", *Review of Financial Studies* 33(5), 2019-2133,
[DOI 10.1093/rfs/hhy131](https://doi.org/10.1093/rfs/hhy131), replicate it by name. Verified off
the PDF, high-minus-low decile monthly returns and |t| (row *"Ato: Assets turnover, Soliman
(2008)"*):

| | NYSE-VW | NYSE-EW | All-VW | All-EW | FM-WLS | FM-OLS |
|---|---|---|---|---|---|---|
| `Ato` | 0.33 / **1.89** | 0.25 / 1.60 | 0.33 / 1.92 | 0.25 / 1.57 | 0.05 / 1.03 | 0.01 / 0.44 |
| `Atoq1` (quarterly) | 0.62 / **3.44** | 1.02 / 6.53 | 0.81 / 4.06 | 1.05 / 6.58 | 0.18 / 3.73 | 0.21 / 6.18 |

**Not one specification clears |t| >= 1.96**, and it fails equally on equal-weighted all-stock sorts
(1.57) - so this is not a microcap artefact in either direction. Only the *quarterly* turnover level
prices, and strongly. Their construction, verified: *"Ato is sales for the fiscal year ending in
calendar year t−1 divided by Noa for the fiscal year ending in t−2"* - note **net operating assets**,
so it is not literally the screener's sales-over-total-assets spread.

**2. ΔATO has never been tested by the major replication.** Grepped over the full 115-page extracted
text of HXZ (2020): **`dAto` 0 occurrences, "change in asset turnover" 0 occurrences.** The only
change-in-turnover variable in their library is a binary indicator inside the Piotroski composites.
So there is **no replication evidence for ΔATO's return predictability, for or against** - a gap, not
a negative result, and it is the gap that matters most here.

**3. ΔATO is rapidly transitory.** Nissim & Penman (2001), "Ratio Analysis and Equity Valuation",
*Review of Accounting Studies* 6(1), 109-154,
[DOI 10.1023/A:1011338221623](https://doi.org/10.1023/A:1011338221623) - read in full from the
authors' own copy. Spearman rank autocorrelation with the t-lagged value, 1963-1999 (Table 3,
p. 146), verified off the PDF:

| | t+1 | t+2 | t+3 | t+4 | t+5 |
|---|---|---|---|---|---|
| **ATO (level)** | **0.947** | 0.894 | 0.857 | 0.830 | 0.808 |
| **ΔATO** | **0.166** | **−0.100** | −0.082 | −0.039 | −0.009 |
| ΔCore Sales PM | −0.039 | −0.083 | −0.059 | −0.043 | −0.012 |

with variance ratios for ΔATO of 0.031, 0.024, 0.010 - **97% of its cross-sectional dispersion is
gone after one year.** Their words, p. 147: *"changes in ATO are mean reverting, and rapidly so"*,
and footnote 21 adds that even the 0.166 is partly an artefact of averaging the denominator. The
**level** of ATO is the most persistent ratio in their whole hierarchy; the **change** is close to
noise. For a screener on a monthly-to-quarterly review cadence that is a serious caution.

**4. The decomposition gives a specific reason to think the spread adds nothing new.** Read the
spread as *(sales growth − asset growth)*. In HXZ (2020), the **asset-growth leg is a robust
large-cap-surviving predictor** while the **turnover level is not a predictor at all**
(NYSE-VW, high-minus-low, |t|): `dNoa` −0.55 / **4.14**; `Ig` −0.46 / **3.76**; `Noa` −0.44 /
**3.25**; `I/A` −0.44 / **2.89**; `dLno` (Fairfield, Whisenant & Yohn 2003) −0.39 / **2.99**;
against `Ato` 0.33 / 1.89. So the prior has to be that whatever a ΔATO sort would show is carried by
its growth leg - **which `asset_growth` already is**.

**5. What ΔATO *is* well supported for is future profitability, not returns.** Fairfield & Yohn's
headline - ΔATO predicting next year's change in return on net operating assets at **0.1799,
t 8.08**, with ΔPM insignificant - is reported here **second-hand**, from Wu & Jiang (2026),
"Earnings Predictability of DuPont Factors", *Journal of Risk and Financial Management* 19(6), 408,
[DOI 10.3390/jrfm19060408](https://doi.org/10.3390/jrfm19060408) (open access), which replicates
their Table 2 and quotes the original coefficients from p. 378. Wu & Jiang's own 1981-2020
re-estimate: ΔATO **0.157, t 9.02**, with balance-sheet growth (ΔNOA) in the regression at
−0.070, t −28.37 - so it is not merely repackaged asset growth *for forecasting profitability*.
Their p. 28 names the return test as still undone: future work *"could examine... whether trading
strategies based on the model generate abnormal returns."*

**One overlap to carry forward.** Li, Nissim & Penman (2014) note that *"asset turnover... becomes a
proxy for unobservable operating leverage"*. The screener holds `operating_leverage` at weight 0 (a
candidate since 2026-10-09, `research/2026-10-09-operating-leverage.md`). If the spread is partly
operating leverage, then re-weighting either one needs the other considered.

**Net reading, and it is the one that governs §11.1:** practice rewards the quantity the screener's
pair expresses; the published return evidence neither supports nor refutes it, shows the related
level does not price, and shows the change is transitory. **That is a genuine academia-versus-practice
disagreement about the screener's own joint position** - which, by the reading rule written before
any of these questions were researched, means the weight stays.

**Not obtained, stated rather than paraphrased:** MSCI "Introducing Investment Quality" (Ting Fang,
2016 - the originating paper; full PDF behind a contact form, only the public abstract read);
S&P Global blocks automated access, so the S&P Quality methodology was read from a Wayback snapshot
and the S&P Style methodology from a mirror, both genuine S&P texts, while S&P GARP's methodology
was **not** obtained; Scientific Beta's Strategy and Index Calculation Rules are gated, so its
definitions here come from its white papers rather than its rulebooks; Dimensional's Rizova & Saito
and O'Reilly & Rizova white papers are behind a login, so **no effect size is quoted from them**;
and two Robeco SSRN papers returned 403. No bot check was worked around.

---

## 10. The pre-registered test, and a prediction that came out wrong

`plan/research-engine.md` queue item 6 says a Monday note that ends in a testable claim registers
one experiment. This note's claim - that the one-year revenue-growth leg earns nothing, so removing
it should not make the ranking worse - was registered as **`drop-revenue-growth`** and **committed
before it was run** (rule 1; commits `1383a38` and `687229e` precede the run). No outcome was known
at registration: nothing in sections 1-9 computes an IC, a forward return or a backtest figure.

**Rule 5 governs what follows. This is a *Backtest observation*, not Evidence.** It may not justify
a weight change and is not quoted in `METHODOLOGY_CHANGELOG.md` under Evidence. It is here because a
pre-registered prediction that fails should be reported by whoever made it.

| | discovery | held out | vs baseline (held out) | 2012-2019 vs baseline |
|---|---|---|---|---|
| `drop-revenue-growth` | IC -0.002 | IC -0.003 | **-0.007 (t -2.5; sees 0.013)** | -0.001 (t -0.69; sees 0.006) |

The engine's own verdict line: **"different only in the held-out months - not confirmed"**, and
"not distinguishable in 2012-2019".

**My prediction said the difference would be "zero or positive, never materially negative". The sign
came out negative.** Removing the leg made the held-out ranking *worse* by 0.007 of IC at t -2.5.
Read honestly, that is:

- **Not confirmed, by the rule written before the run.** The bar is p < 0.05/28 and |t| >= 3; t -2.5
  does not clear it, and the pre-sample era shows nothing (t -0.69) - the registry's `presample`
  block requires the same direction *and* the bar in both, which this fails.
- **But the direction is against me, and that is the informative half.** I registered the direction
  precisely so that this could not be waved away afterwards. The honest statement is that section
  7's reading - that the leg is unrewarded - is **not refuted and not supported**: the one test run
  on this screener's own point-in-time history leans the other way, below the bar, in one era only.
- **It does not reverse the recommendation, and it was never going to be allowed to.** Rule 5 forbids
  citing it either way; CLAUDE.md puts the ban explicitly in both directions ("do not cut a weight
  because the backtest shows no edge: that is rule 5's ban read backwards"). Keeping the leg
  *because* of t -2.5 would be the same error with the sign flipped.

Registering it also exposed a structural cost worth recording: **`registry.yaml` is in
`v2_experiments.RESULT_PATHS`, so adding an experiment changes the engine's `code_hash` and rule 3
requires every experiment to be rerun** so the rows still compare. That is 30 runs - the 29
registry entries and the trading-rule evaluation. *(Corrected the same day by an owner-run session:
this line first read "All 29 were", but the 06:00 session's time ran out after 15 of them, and
its ship gates failed on exactly that mixture. The other 15 were run from the committed code, with
the same `code_hash`; `drop-revenue-growth`'s own row was not rerun and its figures above are
unchanged.)* That is correct -
the multiple-testing bar moved for every row anyway (p < 0.05/27 to 0.05/28, |t| >= 3.35 to 3.37) -
but it means **every Monday note that registers an experiment triggers a full rerun of the engine**,
which queue item 6 asks for one of per week. Noted in `plan/research-engine.md` so the next session
budgets for it rather than discovering it.

---

## 11. Conclusions, and what should change

### 11.1 No weight change - and the reason, after two earlier reasons were measured or read away

**The recommendation is to change neither weight.** `revenue_growth` stays at 25% of Growth,
`asset_growth` at 100% of Investment. But the argument matters more than the verdict, because the
obvious argument turns out to be wrong.

The obvious argument - *"removing the revenue leg would leave the asset-growth leg unopposed and
swing the ranking from neutral on revenue growth to short it"*, the metric-level version of the
2026-10-13 Growth refusal - **was measured and does not hold**
(`research/measurements/2026-10-19-what-removing-the-revenue-leg-does.py`; the rebuilt composite
reproduces the published one at Spearman 1.0000, 25 of 25 top names):

| The published composite, against... | now | without `revenue_growth` | change |
|---|---|---|---|
| raw revenue growth | -0.0040 | **-0.0629** | -0.0589 |
| raw asset growth | -0.2030 | -0.2270 | -0.0240 |
| **the within-sector spread** | **+0.2015** | **+0.1625** | **-0.0390** |
| the Growth score vs revenue growth | +0.6643 | +0.4289 | -0.2354 |

-0.063 is a mild tilt, not a position. So the note does **not** argue that removal would take a
short, and a future session should not resurrect that claim.

*(The "now" column reads +0.2015 against §5's +0.2017 for the same quantity: this script requires
every stock to carry all five Growth percentiles so the two columns are computed on one sample,
where §5 requires only the two metrics under examination. Two decimal places in the fourth place,
not a disagreement - but worth saying, since an unexplained difference in a published number is
exactly what a sceptical reader should stop on.)*

**What removal would actually cost is the exposure to the spread.** `revenue_growth` is what
converts an asset-growth screen into an **asset-turnover** screen - and whether that is worth having
is exactly where academia and practice part company, which is what decides this note.

- **Practice rewards it.** FTSE Russell scores **Δ Asset Turnover** positively inside Quality and
  uses *neither* of the screener's raw legs (§9.2); S&P frames its quality index as companies
  generating revenue *"through prudent use of assets"*; Research Affiliates' written defence is
  that income-statement growth is *"economically scalable"* where balance-sheet growth is *"empire
  building"* (§9.5).
- **The published return evidence does not.** Soliman (2008), the paper that would settle it, could
  not be obtained. Hou, Xue & Zhang's replication shows the annual asset-turnover **level** does not
  price in any of eight specifications (|t| 0.44-1.92, and no better equal-weighted), **never tested
  the change at all**, and shows the spread's asset-growth leg is a robust predictor while its
  turnover leg is not - so the predictability a ΔATO sort would show is probably already in
  `asset_growth`. Nissim & Penman measure ΔATO as near-transitory: autocorrelation 0.166 then
  −0.100, **97% of its dispersion gone within a year**. What ΔATO is well supported for is
  forecasting next year's *profitability*, not returns (§9.6).

**So the reading rule written before any of these questions were researched decides it**
(`plan/factor-lab.md`: *"A change needs the lab and the literature or practice pointing the same way.
Where they disagree the note says so and the weight stays."*). They disagree, about the screener's
own joint position. **The weight stays.**

That is a weaker and more honest basis than the one this note first drafted, which was that the
spread exposure is plainly worth keeping. It is not plainly worth keeping. What can be said is that
removing the revenue leg would trade a position practice endorses and academia has not tested for a
slightly larger dose of a position (asset growth) whose large-company support is tail-concentrated,
halved against small caps, absent in FF2015's big-stock leg and unestablished in the low-volatility
corner this universe occupies (§6.2, §6.4, §9.4). Neither side of that trade is well enough
established to make it on.

**And a guard, because §3 invites the opposite mistake:** *do not build a ΔATO metric on the strength
of this note.* The practitioner precedent is real, but the return evidence is absent rather than
favourable, the quantity is transitory at the horizon this screener reviews on, and the screener's
spread is sales-over-**total**-assets where the literature's ATO is sales-over-**net operating
assets**. Adding such a metric would also duplicate `operating_leverage` to an unmeasured degree
(§9.6). That would need its own Monday, starting by obtaining Soliman.

**The honest residue, stated plainly:** the screener's *positive* sign on one-year revenue growth
is against the weight of published evidence - Robeco measures it at **-2% a year at exactly one and
three years** (§9.3), LSV at **-4.6pp size-adjusted** and asserted to hold in an S&P-500-like
universe (§7.4), HXZ and CGS at insignificant-but-negative on the five-year rank (§7.2). The
screener survives that only because the asset leg cancels the exposure to **+0.045**. **It holds
almost no position on the thing it has the sign wrong on - and that is luck, not design.** It was
+0.137 until a correctness fix on 2026-10-09 moved it (§8.1), and it would move again if either
leg's definition or weight changed. That is the finding this note most wants on the record.

### 11.2 What the methodology page should say, and why that is a build day's work not today's

`SCREENER_OVERVIEW.md` §8 (Investment) is five lines, gives no figure, no sample and **no
large-company caveat** - where the Size section carries a thorough one and the Momentum section
another. The same standard applied unevenly is the defect here.

**Nothing on the page is false**, which is why this is a recommendation and not a fix. *"The
Fama-French CMA factor captures the historical tendency for companies that invest conservatively to
outperform"* is true of CMA in aggregate (0.33%/month, t 4.07). *"The screener rewards companies
that grow efficiently rather than just growing big"* is, remarkably, **supported** by §3 and §5 at
the system level (+0.892 for the pair, +0.202 for the composite) - it is accidentally right about
the screener while describing a category that measures no such thing.

What a build session should add, in the generator (`build_screener_overview()` in
`run_screener.py`, never the markdown - rule 10), with claims registered in `claims.py`:

1. **A large-company caveat on Investment**, matching Size's. The figures are measured and
   committed: among big companies the factor is +0.149%/month at **t 1.46** over 63 years, the
   long-only half +0.132% at **t 2.22**, small companies **2.27x** the big-company figure
   (§4). FF2015's own CMA_B is t 1.03-2.00 and *"almost all the average value of CMA is from small
   stocks"*.
2. **That the screener uses the measure with the large-company evidence** - CGS's year-on-year
   percentage change in total assets, not FF2008's per-share version (§6.1). This is a point in the
   screener's favour and is currently unsaid.
3. **What the pair does jointly** - that Growth's revenue leg and Investment's asset leg together
   express growth in asset turnover, with the measured +0.892, and that FTSE scores that same
   quantity. A reader following the page today cannot discover this, and it is the single most
   interesting true thing about these 8.25 points.
4. **The honest line on revenue growth's sign**, in the Growth section, which already carries an
   unusually candid "What the research says" paragraph: that the published work on past sales
   growth points the other way at the screener's own horizons, and that the category's net exposure
   is near zero.

Today is Monday - research, **no production code** - so none of this was built. It is specified
above to the point of being mechanical.

### 11.3 The open questions this note leaves, ranked

1. **Both legs are tail effects scored linearly** (§6.3, §7.5). CGS's large-company asset-growth
   alphas are flat across deciles 2-9 (0.16pp of an 0.80pp spread); LSV's decile 10 carries 41% of
   its spread. The screener scores both as linear sector percentiles, spending its resolution where
   the evidence is flat. **This is the strongest open question in the note** - and it is not a small
   change: it would affect every metric scored by percentile, make the tool harder to explain, and
   needs its own Monday, its own synthesis and a changelog entry. Do not treat §6.3's table as
   authorising it.
2. **Should Investment be folded into Quality?** Practice says yes almost unanimously (§9.4) - nine
   of eleven provider families house asset growth inside Quality, Research Affiliates as literally
   *"Profitability minus Investment"*, and Hsu, Kalesnik & Kose call that pairing *"the most
   parsimonious quality definition... associated with the strongest academic evidence"*. The
   screener keeps it standalone, which is the academic construction. **This answers the
   asset-growth half of `plan/factor-lab.md` question 3** ("Is Investment Valuation counted
   twice?"): the overlap with value is real (CMA-HML 0.70; 0.52 among big companies) and is exactly
   why practitioners merge them.
3. **Three papers could not be obtained, and two of them are load-bearing.** **Soliman (2008)** is
   the one that would say whether change in asset turnover predicts *returns* - the quantity §3
   shows this screener's pair actually expresses - and without it §9.6 can only report that the
   major replication never tested it. **Fairfield, Whisenant & Yohn (2003)** is the one that puts
   accruals and operating-asset growth in a single specification; **Fairfield & Yohn (2001)** is
   quoted here only second-hand through an open-access replication. All three are paywalled with no
   free full text, and no bot check was worked around to try. **Getting Soliman needs a library
   login, not a different URL** - a session with institutional access, or willing to email the
   author, should close that before anyone reopens question 2 or builds anything on the spread.
4. **Does `revenue_cagr_3yr` duplicate `revenue_growth`?** Measured today at **+0.316** within
   sector (§6 of the measurement script) - lower than expected, so they are not redundant - but it
   carries 1.95 points on the horizon Robeco measures at -2%, and was not the subject of this note.

### 11.4 Status

**No proposal is left open by this note.** The weight question it was asked to settle is settled -
no change, for the reason in §11.1 - so there is deliberately no `**Proposal status: OPEN**` line
for a later session to decide. What is left is the page work in §11.2, which is a build task with no
judgement call in it, and the four ranked questions in §11.3.
