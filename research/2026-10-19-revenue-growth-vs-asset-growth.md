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

<!-- LITERATURE AND PRACTICE SECTIONS 6-9 TO FOLLOW -->
