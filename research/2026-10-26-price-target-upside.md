# `price_target_upside`: the level is 71% price, and the literature scores the change

**Monday research note, 2026-10-26.** One metric, learned properly: the consensus analyst price
target as a scored factor.

Measurements: `research/measurements/2026-10-26-price-target-upside.py` and its `.json`.
Figures belong to the published run `bc1fe4648fd6` (2026-10-26T02:00:04) and to the 56 committed
versions of `dashboard_data.js`.

**Proposal status: REFUSED 2026-10-26 (13:00 session)** - the substitution in §6.2 was measured
and fails the §8.4 spanning test the proposal itself made step 1. See **§8**, written by a
session that did not write §1-§7. The metric is **not** a duplicate of `fy1_revision_3m`
(+0.3475 within sector, so the proposal survived its own first check), but it is **94% momentum
in its spanned variance** and would take `revisions ~ momentum` from **+0.2223 to +0.3778**,
through the ~0.35 bar pre-registered in
`research/2026-09-07-revisions-category-has-no-revisions.md` §8.4. No weight, metric or
threshold changed. The three **[VERIFY FIRST]** items in §3.1 are now verified from PDF bytes,
with **three corrections** (§8.5), and §6.1's reading of the *incumbent* is weaker than it
claimed as a result - §8.7 states the next question with its criteria written in advance.

---

## The question, and why this metric

`price_target_upside` carries **13.5 of the Revisions category's 100 points**, and Revisions is
**10% of the composite** — so **1.35 composite points**. It is the last weighted metric in the
screener with no research note of its own. Everything around it has one: the category was rebuilt
on `research/2026-09-07-revisions-category-has-no-revisions.md` (1,022 lines),
`earnings_acceleration` was zeroed on `research/2026-10-09-metric-audit.md`, and
`short_interest_ratio`'s data history was probed on 2026-10-23. In all of that,
`price_target_upside` is mentioned four times and never examined; the 09-07 note's own table says
only "unchanged in role; sell-side optimism bias already noted in config" and "−0.053 with the new
metric; no overlap; role unchanged."

That is a thin record for a scored input, and there is a specific reason to look. The metric is a
**level, not a change**, and it has price in its denominator:

```python
# factor_engine.py:2896-2904
_target    = d.get("targetMeanPrice")
_cur_price = _coalesce(d, "currentPrice", "price_latest")
if _n_analysts >= 3:
    rec["price_target_upside"] = float(np.clip((_target - _cur_price) / _cur_price, -0.50, 1.00))
```

Every other metric in the category differences something over time (`fy1_revision_3m` is a 90-day
change; `analyst_surprise` and `consecutive_beat_streak` compare actuals with estimates). This one
divides a slow-moving number by a fast-moving one. If consensus targets move more slowly than
prices do, the cross-sectional spread in this metric is mostly the spread in **recent price, with
the sign reversed** — which would make it a negative-momentum signal sitting inside a category
whose own `config.yaml` comment warns:

> The revisions category has now SPENT its independence budget: it goes from ~91% unspanned by the
> other seven categories to 85.2% ... Adding any further momentum-adjacent metric here requires
> re-running the §8.4 spanning regression first, not just a citation.

Whether that is what is happening is a measurable fact about data already on disk. It needs no
forward return, no IC and no backtest, so rules 4 and 5 do not bind.

---

## 1. What this screener's own data says

### 1.1 Coverage and distribution

| | |
|---|---|
| Coverage | 496 of 501 (99.0%) — the `>= 3 analysts` floor costs almost nothing |
| Median implied upside | **+13.2%** |
| Mean | +17.7% |
| 10th / 90th percentile | −0.8% / +43.0% |
| Share with a *negative* implied upside | **11.7%** |
| At the clamps (−50%, +100%) | 0 low, **1** high |

The clamp does almost nothing, so the metric is effectively unclamped in practice. The
distribution is one-sided: the consensus sees almost every company in the S&P 500 as underpriced,
by a median of 13%. That is the sell-side optimism the config comment alludes to, now measured
rather than asserted — and because the screener scores **sector percentiles**, a uniform optimism
bias is harmless. What is not harmless is anything that makes the within-sector ranking reflect
something other than what the metric claims to measure.

### 1.2 The decisive measurement: targets are stale, so the metric is a price signal

From the 56 committed versions of `dashboard_data.js` (one per published run date, each carrying
`pt_mean` and `price` for ~497 stocks). Implied upside in logs is exactly
`log(target) − log(price)`, so its change decomposes additively and the shares below are a real
decomposition rather than an attribution.

**Between consecutive published runs** (one to three days apart):

| From → to | median \|Δlog price\| | median \|Δlog target\| | target *exactly* unchanged | corr(Δupside, Δprice) |
|---|---|---|---|---|
| 10-16 → 10-19 | 0.0126 | **0.0000** | 76.7% | **−0.990** |
| 10-19 → 10-20 | 0.0110 | **0.0000** | 60.0% | −0.955 |
| 10-20 → 10-21 | 0.0026 | **0.0000** | 76.9% | −0.823 |
| 10-21 → 10-22 | 0.0094 | **0.0000** | 72.2% | −0.965 |
| 10-22 → 10-23 | 0.0269 | **0.0000** | 65.5% | −0.971 |
| 10-23 → 10-26 | 0.0124 | **0.0000** | 75.0% | −0.936 |

For the median stock, between one published run and the next, **the consensus target does not move
at all** while the price moves up to 2.7%, and 60–77% of targets are byte-identical. Day to day the
metric is the negative of the price change and essentially nothing else.

Daily is not the horizon that matters — the review cadence is monthly and the bot rebalances
monthly — so here is the **term structure**, pooling all date pairs near each gap. Beta is the
slope of Δlog target on Δlog price: **0 means the target never moves** (upside is pure inverted
price), **1 means the target fully tracks the price** (upside carries no price information at all).

| Gap | date pairs | stock obs | beta | corr(Δupside, Δprice) | price explains | target unchanged |
|---|---|---|---|---|---|---|
| ~7 days | 52 | 25,833 | +0.246 | −0.849 | **72.1%** | 46.8% |
| **~30 days** | **39** | **19,255** | **+0.245** | **−0.845** | **71.4%** | 8.4% |
| ~91 days | 7 | 3,410 | +0.540 | −0.656 | 43.1% | 0.5% |
| ~182 days | 2 | 968 | +0.823 | −0.356 | 12.7% | 0.0% |

**At the monthly horizon the screener re-reads this metric over, 71.4% of the variation in implied
upside is the price moving and 28.6% is analysts changing their minds.** By six months the targets
have caught up (beta +0.82, close to the 1.0 of full tracking) and the price share has collapsed to
12.7% — so the metric's own information does arrive, it just arrives late, and by the time it has
arrived it no longer distinguishes anybody.

**Three honest caveats on this table.**

1. The 7-day and 30-day rows rest on 52 and 39 date pairs inside a dense 90-day window
   (2026-07-28 to 2026-10-26) and are solid. The 91-day row has 7 pairs and the 182-day row has
   **2**, both anchored on the two isolated pre-outage snapshots (2026-03-16, 2026-04-14) that
   survive in the payload history. 968 stock observations from two start dates are not two
   independent observations; read +0.823 as indicative of direction only. The claim I will stand
   behind is the 7-to-30-day one, which is also the decision-relevant one.
2. `pt_mean` is Yahoo's `targetMeanPrice`, a vendor consensus on a vendor refresh schedule. Some
   of the 60–77% "exactly unchanged" between consecutive runs could be the vendor not refreshing
   rather than analysts not revising. That objection bites hardest at the 1–3 day gap and barely at
   all at 30 days, which is why the 30-day row carries the argument.
3. The window is three months of one year. It says what the metric *is*; it is not a claim about
   what it earns, and nothing here reads a return.

**An independent cross-check that the mechanism is real, not a vendor artifact.** Brav & Lehavy
(2003) estimate how fast a target/price gap closes: 9% of the deviation per week, by the analyst
moving (§2.2 below). Compounding that gives the share of a gap closed by 4, 13 and 26 weeks of
**31%, 71% and 91%**. My betas at the matching gaps are **24.5%, 54.0% and 82.3%** — the same
shape, slightly slower, from different data, a different vendor and a sample 27 years earlier. Two
independent estimates of the same slow adjustment agreeing to within about ten points is the best
evidence available that §1.2 is measuring analyst behaviour rather than a Yahoo refresh cycle.

### 1.3 So what is it correlated with? Not its own category

Spearman against `price_target_upside` in the published run, pooled and within sector (the screener
scores sector-relative percentiles, so the within-sector column is the operative one):

| Against | pooled | **within sector** |
|---|---|---|
| `jensens_alpha` | −0.383 | **−0.403** |
| `risk_score` (low risk is better) | −0.409 | **−0.473** |
| `return_6m` | −0.345 | −0.364 |
| `momentum_score` | −0.279 | −0.317 |
| `return_12m` | −0.271 | −0.306 |
| `short_interest_ratio` | −0.253 | −0.238 |
| `investment_score` | −0.185 | −0.224 |
| `quality_score` | −0.164 | −0.171 |
| `growth_score` | +0.304 | **+0.318** |
| `revisions_score` (its own category) | +0.222 | +0.223 |
| `valuation_score` | −0.037 | −0.038 |
| `size_score` | −0.007 | +0.005 |
| **`fy1_revision_3m`** (the only real revision metric) | +0.029 | **−0.007** |
| Composite | −0.211 | −0.242 |

Read the top and the bottom together. Within sector the metric is **−0.40** with risk-adjusted
momentum, **−0.47** with the Risk score, and **−0.007** with `fy1_revision_3m` — the one metric in
its own category that measures an analyst revision. A high implied upside identifies a **volatile
stock that has recently fallen**. It does not identify a stock analysts are getting more optimistic
about.

**This is the exact shape of a defect this project has removed twice before**, which is why it
reads as a defect rather than a surprise:

- `sharpe_ratio` / `sortino_ratio`, removed from Risk 2026-09-02: "measured on the published
  payload, `sharpe~sortino` = +0.993 and `sharpe~return_12_1` = +0.944, while `sharpe~volatility`
  = +0.025 — they carried momentum, not risk."
- `earnings_acceleration`, zeroed 2026-10-09: "−0.54 with the prior surprise while surprises
  autocorrelate +0.38, so it penalised a stock that beat last quarter."

The test applied in both cases was: *does the metric correlate with what its category is for, or
with something else?* `price_target_upside` fails it more cleanly than either, at −0.007 with
revisions and −0.40 with momentum.

### 1.4 But what does removing it actually do? Very little

Redistributing its 13.5 points across the category's other weighted metrics in proportion
(`fy1_revision_3m` 48 → 55.5, `analyst_surprise` 15 → 17.3, `consecutive_beat_streak` 10 → 11.6,
`short_interest_ratio` 13.5 → 15.6):

| | |
|---|---|
| Rank Spearman, with vs without | **0.99845** |
| Median absolute rank move | **4** |
| 90th percentile / max rank move | 13 / 48 |
| **Top 25 names changed** | **0** |
| Composite's momentum exposure | +0.3749 → +0.3896 |
| Composite's valuation exposure | +0.5630 → +0.5503 |

The reconstruction is **exact** before any of that is read: the rebuilt Revisions score is Spearman
**1.0** against the published one and differs by **0.0 points** on average (Revisions uses the
generic metric weights for banks too, so there is no bank branch to get wrong), and the rebuilt
composite is Spearman 0.9999 against the published one. The counterfactual is this screener's
arithmetic, not an approximation of it.

**The metric is incoherent but it is not loud.** It suppresses the composite's momentum exposure by
0.015 and changes no name in the top 25. Any argument about it has to be made on construction and
explainability, not on materiality — the same place `research/2026-09-07-...` §8.5 ended up, and
the same test it set: *argue this on explainability.*

---

## 2. What the literature says

Both papers below were downloaded and read from the PDF bytes; every figure is quoted with the page
of the PDF it was read from. A third paper is reported second-hand and flagged as such.

### 2.1 Da & Schaumburg (2011) — the only paper that tests the *level*, and it splits exactly where this matters

Da, Z. & Schaumburg, E. (2011), "Relative valuation and analyst target price forecasts",
*Journal of Financial Markets* 14(1), 161–192. Read from the author's copy:
<https://academicweb.nd.edu/~zda/TargetPrice.pdf>. (No DOI is printed on the copies I could reach;
RePEc handle `RePEc:eee:finmar:v:14:y:2011:i:1:p:161-192`.)

Their signal is this screener's metric, exactly: "**TPER = TP/P − 1**" (PDF p. 2). Sample: First
Call, December 1996 – December 2004, stocks above $5 with a target in the first 25 calendar days of
the month (PDF pp. 4–5). Portfolios are formed monthly on the **standing level**, not on an
announcement.

**The result that matters most here is the comparison between two ways of ranking the same signal**
(PDF pp. 13–14):

| Sort | Spread, portfolio 1−9 | t |
|---|---|---|
| **Absolute** TPER, ranked across all stocks | **79 bp/month** | **0.86 — not significant** |
| **Within-sector** TPER | **177 bp/month** | **3.55** |

> "The spread Portfolio 1–9 returns only 79 bp/month (t-value of 0.86), much smaller than the
> spread of 177 bp/month when the long-short position is constructed within sector. **This is an
> indication of the fact that it is relative and not absolute valuations that convey the most
> information to investors.**" (PDF p. 14)

And from the abstract: "within industry relative valuations implicit in analyst target prices do
provide investors with valuable information although **the implied absolute valuations themselves
are much less informative** ... our findings are not merely a small stock phenomenon but apply to
the sample of S&P 500 stocks and do not rely on trading at the exact time of announcement"
(PDF p. 1).

**This is good news for the screener's construction and must be said plainly.** The screener does
not score the absolute level — it scores a **sector percentile** of it. That is the side of Da &
Schaumburg's distinction the evidence supports, on an S&P 500 sample, and it was arrived at by this
project's general sector-relative design rather than by anybody reasoning about price targets. The
risk-adjusted spread survives costs — "risk-adjusted profit net of transaction costs of 148 bp/month
(203 − 28.4 − 26.6) per month (t-value 3.67), or 17.8% per year" (PDF p. 18) — and survives
controls: "If we use all 15 stock characteristics, TPER remains significant" (PDF p. 21), the 15
including momentum, size and book-to-price.

**Four findings in the same paper cut the other way, and three of them are about this screener's
own universe and cadence:**

1. **The horizon is one month, and two-thirds of it is two weeks.** "approximately two-thirds of
   the first month profit (110 bp) accrues during the first two weeks" (PDF p. 13). A monthly
   review cadence captures the tail of that at best.
2. **It is weakest exactly where a large-company screener lives.** In the full-sample size/value
   breakdown (PDF p. 28), **Large growth** shows 0.65% raw and **0.23% risk-adjusted, t = 0.37** —
   indistinguishable from zero.
3. **It decays after 2000.** The paper attributes lower post-2000 profits to "a market-wide
   improvement in liquidity due to decimalization in early 2001 and consistent with target prices
   being less informative after regulation FD (Fair Disclosure) which took effect in October 2000"
   (PDF p. 3). The screener's sample is 2026, a quarter-century past that.
4. **The staleness channel is in the paper.** "**High (low) TPER stocks tend to have a lower
   (higher) price and a lower (higher) past return at the one-to-six month horizon**" (PDF p. 7) —
   their data says the same thing §1.3 says about ours.

### 2.2 Brav & Lehavy (2003) — why the level is mostly firm identity, and who closes the gap

Brav, A. & Lehavy, R. (2003), "An Empirical Analysis of Analysts' Target Prices: Short-Term
Informativeness and Long-Term Dynamics", *Journal of Finance* 58(5), 1933–1967.
**DOI 10.1111/1540-6261.00593.** Read from the author's copy:
<https://webuser.bus.umich.edu/rlehavy/BL.pdf>. Sample: First Call 1997–1999, **223,016**
target-price observations on **6,544** firms (PDF p. 5).

Two findings bear directly on scoring the level.

**The level is a persistent firm-level baseline, not a signal.** Estimating the long-run
cointegrating ratio of target to market price for 900 firms with continuous coverage: "the mean
(median) of the 900 estimates equals **1.28 (1.26)**. That is, conditional on at least two years of
continuous consensus coverage, the average firm in this sample is expected to earn 28 percent
annually. Furthermore, with the 25th and 75th percentiles equal to **1.20 and 1.33** ..." (PDF
p. 27). Each firm has its own standing implied upside, and most of the cross-sectional spread in
the level is that baseline. Rank it across all stocks and you are largely ranking firm identity —
which is why Da & Schaumburg's absolute sort returns t = 0.86, and why sector-demeaning rescues it.

**When the gap is wide, it closes because the analyst cuts the target — not because the price
rises.** "**Market prices, in contrast, barely contribute to this correction phase**" (PDF p. 3);
"any correction to the long-term relation between target and market prices is **predominantly made
by analysts**" (PDF p. 28). This is the mechanism behind §1.2's term structure, and it is the
single most damaging fact for scoring the level: a wide implied upside is an expectation that
*analysts will revise down*, not that the stock will go up.

**And the predictable part of a revision is not paid.** Regressing abnormal returns on target
revisions, the unexpected component carries a slope of "2.671 with t-statistic = 27.0" while "we
also find no reliable relationship between abnormal returns and the expected revision in target
price" (PDF p. 29). The information is in the *surprise in the change*, not in the standing level
that makes the change predictable.

### 2.3 Asquith, Mikhail & Au — second-hand, and flagged

Asquith, P., Mikhail, M.B. & Au, A.S. (2005), "Information content of equity analyst reports",
*Journal of Financial Economics* 75(2), 245–282, DOI 10.1016/j.jfineco.2004.01.002.

**I did not obtain this paper.** A sub-agent read the NBER working-paper version (w9246, October
2002, <https://www.nber.org/system/files/working_papers/w9246/w9246.pdf>) through a text-extraction
proxy, not from PDF bytes, so the following is **second-hand and unverified by me** and nothing in
§6 rests on it. Reported: hand-coded reports from *Institutional Investor* All-America analysts,
1997–1999; a target revision maps to a five-day CAR with a slope of ~0.22–0.32, larger and more
significant than the comparable earnings-forecast revision; targets are reached at some point
within 12 months in **54.28%** of cases; and achievement falls steeply with the size of the implied
upside — ~74% for targets implying 0–10%, ~60% for 10–20%, and **under 25% for targets implying 70%
or more**. If that last figure is right it is directly on point, which is a reason to verify it
properly rather than to lean on it now.

---

## 3. What practice does

I could not finish an independent sweep of provider methodology documents inside this session (see
§7). What follows is this project's **own previously sourced** reading of two primary documents,
re-used because it is already documented practice and it answers this question squarely —
`research/2026-09-07-revisions-category-has-no-revisions.md` §2.1–2.2.

**MSCI Barra, "Datasheet — Barra US Total Market Equity Trading Model" (USFAST), March 2015.** Its
`Sentiment` factor — "Explains the return differences between stocks based on sell-side analyst
revisions and news sentiment" — has these analyst descriptors:

> • **Revision ratio** • **Change in analyst-predicted earnings-to-price** • **Change in
> analyst-predicted earnings per share**

All three are **changes**. Across 24 style factors and their full descriptor lists there is **no
price-target descriptor of any kind**, and no level-of-upside descriptor.

**Zacks Rank**, the most widely distributed revisions-based ranking sold to retail and advisors,
has four components: **Agreement** (share of analysts revising in the same direction),
**Magnitude** (size of the recent change in consensus estimate), **Upside** (Zacks' proprietary
"most accurate estimate" versus consensus) and **Surprise** (recent EPS surprises). Note that
Zacks' "Upside" is *estimate* versus *consensus estimate* — not target versus price. **None of the
four is a price target.**

So the two production systems this project has documented both build their sell-side-analyst factor
out of **changes in earnings estimates**, and neither scores the level of a price target. That is a
real finding and it points the same way as §2.2: the informative object is the revision.

### 3.0 The wider sweep, returned late — the verified half

Two further reads returned **after** §1–§2 were committed. Where they handed over a PDF, I checked
the figure against the bytes myself; those are below and are load-bearing. Where they did not,
the material is in §3.1 and is explicitly not load-bearing.

**Green, Hand & Sikochi (2024) measure my §1.2 finding, independently, in a peer-reviewed open-access
paper — and it is the single most important citation in this note.** "The Asymmetric Mispricing
Information in Analysts' Target Prices", *Review of Accounting Studies* 29(1), 889–915,
DOI [10.1007/s11142-022-09730-z](https://doi.org/10.1007/s11142-022-09730-z). Analyst-report text
from Thomson ONE Investext, 2001–2017; 9,781 US and 64,285 international analyst-report
observations. From their Table 7 quantile regressions (verified, PDF p. 18):

> "The coefficient of **−0.92 on MOM− in the 90th quantile IRET regression** indicates that a 1%
> more negative MOM− is associated with an 0.92% higher IRET, **almost an inverse one-to-one
> relation**. In comparison, the coefficient of 0.03 on MOM+ is just 1/30th as large."

and their own reading of it:

> "This table suggests that **an important determinant of analysts' claimed undervaluation is
> transitory declines in market prices.**"

At the top of the implied-upside distribution — which is where this screener's percentile puts its
best scores — a 1% fall in the stock produces a 0.92% rise in implied upside. That is §1.2's
71.4% stated as a slope, from different data, a different vendor, a peer-reviewed sample, and a
window ending in 2017. My measurement is not a Yahoo artifact.

Their headline result also bounds how much is left once that is accounted for: "target prices only
predict future returns (at **16 cents to 18 cents on the dollar**) when analysts claim
undervaluation, not when they claim overvaluation" (PDF p. 1) — and only one to two quarters out.

**Bradshaw, Brown & Huang: the size of the implied upside is the strongest known predictor that the
target will not be met.** "Do sell-side analysts exhibit differential target price forecasting
ability?", *Review of Accounting Studies* 18(4), 930–955,
DOI [10.1007/s11142-012-9216-5](https://doi.org/10.1007/s11142-012-9216-5). *Read from the 2005
working-paper version, which is a different sample (First Call, 1997–2002, N = 95,852) from the
published 2013 paper — flagged.* Verified from the bytes: across TP/P quintiles the target is met
at 12 months "**41 percent of the time**" in the least optimistic quintile and "met just **7 percent
of the time** at 12 months" in the most optimistic (PDF pp. 18–19); and in the logit, "the control
variable with the **greatest explanatory power is TP/P**, which is negatively related to target
prices being met" (PDF p. 21). Note also that the published abstract reports targets met at 12
months only 38% of the time on a 2000–2009 sample (second-hand, abstract only).

**Bilinski, Lyssimachou & Walker: targets carry thin but real skill, and revisions are not what
makes them accurate.** "Target Price Accuracy: International Evidence", *The Accounting Review*
88(3), 825–851, DOI [10.2308/accr-50378](https://doi.org/10.2308/accr-50378). I/B/E/S, 16
countries, 2002–2009, N = 552,165. Verified: "**59.1%** of TP forecasts are met at some point
during the 12-month forecast period" (PDF p. 21); "analyst TPs meet or exceed the accuracy of naïve
price forecasts in **74.5%** of cases, and the analysts' absolute TP forecast error is **9.8%
lower**" than a naïve trailing-return forecast (PDF p. 3); mean absolute error 44.7% overall and
"**in the US ... among the highest in the sample at 49.5%**" (PDF p. 22).

**The strongest counter-evidence to my own §1.2, which must be stated.** Da & Schaumburg's Table 2
Panel B (verified, PDF p. 9) says that in their S&P 500 sample a high TPER is *not* a stale target:

> "The S&P 500 stocks receive frequent target price coverage: **less than 1.5% of these stocks have
> a target price during the current month and none during the previous two months.** The percentage
> of target price upgrades increases monotonically with TPER. In Portfolio 1, which has the highest
> TPER, the recent target price changes are dominated by **upgrades** (percentage upgrades and
> downgrades are 55% and 22%, respectively)."

Bilinski points the same way: adjusting for revisions moves the met-target rate only from 59.1% to
43.4%, and the revision-adjusted and raw accuracy measures correlate 0.726–0.744, "which suggests
that TP revisions have little effect on the accuracy measures."

**How I reconcile that with §1.2, and the limit of my own claim.** These are different statements
and both can hold. Mine is a *time-series* fact — between two runs a month apart, the target moves
a quarter as much as the price. Theirs is a *cross-sectional* fact — the stocks that end up with the
highest implied upside are covered, and recently upgraded. A target can be revised often and still
move less than the price; Brav & Lehavy's 9%-a-week correction is exactly that. But it does mean
**"the metric is stale data" is too strong a claim, and I am not making it.** The claim I make is
the measured one: at a one-month horizon 71.4% of the variation in this metric is price, and
Green, Hand & Sikochi get −0.92 for the same thing at the top of the distribution.

### 3.1 The rest of the sweep, NOT VERIFIED BY ME

The provider sweep §7 originally recorded as "did not return" **did return, after this note's
measurement and literature sections were committed.** Its material is substantial and it changes
the balance of §4, so it is recorded here rather than dropped — but **every figure in this
subsection is second-hand**: it comes from a sub-agent that read primary methodology PDFs and
papers, and I have **not** re-checked any of it against the documents in the way I checked §2.1 and
§2.2. Nothing in §6 is permitted to rest on it, and the first job of the session that decides the
proposal is to verify the three items marked **[VERIFY FIRST]**.

**The sharpest reported fact, and it is about a deliberate omission.** MSCI's *Analyst Sentiment
Indexes Methodology* (April 2025) **does** use price-target data — as two descriptors in a
`PRICE TGT` group: a price-target **revisions ratio** (`PTG_RR_SLOW`, scaled by analyst count) and a
**change** in the analyst-predicted target (`PTGF_C`, scaled by the mean absolute level of the
target itself). And MSCI builds an *estimate-to-price* descriptor for three other estimate types —
`ETOPF_C`, `CPSTOPF_C`, `SALTOPF_C` — but reportedly **defines no target-to-price descriptor at
all.** If that is right, the provider best placed to build this screener's metric considered the
family, built the price-scaled version for earnings, cash flow and sales, and declined to build it
for price targets. That is a stronger statement than "nobody uses it".

**Practice is unanimous, and the sweep now covers five more primary rulebooks.** Read by a
sub-agent with page-pinned quotes, not re-checked by me: the **FTSE Global Factor Index Series
Ground Rules** v8.8 (July 2026, 43pp — which governs the Russell 1000/2000 Comprehensive Factor
indexes too) scores eight factors, every one trailing-reported or price-derived, with **zero**
occurrences of analyst/forecast/estimate/I/B/E/S; **S&P Quality Indices** (March 2025, 33pp) and
**S&P Quality, Value & Momentum Multi-factor** (August 2022, 26pp) likewise have **zero** hits, and
S&P's "risk-adjusted momentum" turns out to be a realised 12-1 price return over its own realised
daily volatility (**S&P Momentum Indices**, Nov 2014, Appendix A p. 17) — no analyst component at
all. Across all five documents the strings "price target" and "target price" appear **zero times**.
The one analyst input either index house scores is **Russell style's I/B/E/S forecast medium-term
(two-year) earnings growth** (*Russell US Equity Indexes* v7.2, §8.2.1, p. 24) — and even that is
handled with more care than this screener gives implied upside: §8.10.1 (p. 28) **shrinks it toward
the sector mean when coverage is thin** (1 analyst → ⅓ own score, 2 → ⅔, 3+ → full), and it is
banded to damp turnover. That shrinkage rule is worth copying regardless of what happens to this
metric.

The distinction the two fully-transparent index houses draw is the one this note keeps arriving at:
a growth forecast is an analyst's estimate of a **fundamental**; implied upside is an analyst's
**valuation opinion** divided by the market's. They score the first and not the second.

**Three papers that, if they hold, flip §4 rather than refine it:**

- **[VERIFY FIRST]** Palley, Steffen & Zhang (2024), "The Effect of Dispersion on the
  Informativeness of Consensus Analyst Target Prices", *Management Science* 71(3), 2264–2288.
  465,797 firm-months, Jul 1999 – Jun 2018. Reported: deciles 1–8 on implied return are flat near
  12%, then **fall** to 10.84% and 7.28%, so long-top/short-bottom is **−4.60% (t −3.08)** — the
  highest implied upside has the *worst* subsequent returns. Dispersion is the moderator:
  **+2.96% (t 2.54)** in the lowest-dispersion decile against **−11.44% (t −5.34)** in the highest.
- **[VERIFY FIRST]** Han, Kang & Kim (2022), "Betting against analyst target price", *Journal of
  Financial Markets* 59, 100677. Reported: prices initially move with targets then drift back;
  betting against earns **0.98%/month (t 3.80)**, 1999–2015, and — the part that matters most here
  — **stronger among large and liquid stocks.**
- Da, Hong & Lee (2016), *JBFA* 43(3–4), 487–510, DOI 10.1111/jbfa.12176 — **the same lead author
  as §2.1**, reported as finding the *absolute* valuation implied by target prices uninformative,
  consistent with the 2011 paper's own sector-relative/absolute split.

**And the one explicit practitioner statement, which is almost exactly this note's proposal:**
S&P Global Market Intelligence Quantamental Research (Tortoriello, Sept 2018) reportedly states
that "consensus target prices and recommendations lack predictive power themselves" while
"short-term **changes** in target prices ... have been historically predictive", with the simple
**1-month change in consensus target price** (`TP_t / TP_{t−1m}`) giving long-only active 6.48% and
long-short 11.39%, IR 1.42, Russell 3000, 2004–2017 — **at 86% monthly turnover**, which is the
catch and is new information for §6.2.

**[VERIFY FIRST]** Finally, the item most specific to this screener: AQR's *Quality Minus Junk*
(Asness, Frazzini & Pedersen, *Review of Accounting Studies* 24(1), 2019,
DOI 10.1007/s11142-018-9470-2) reportedly computes **exactly this metric** — "the ratio of the
target price to the current price minus 1" — and uses it as evidence that analysts are wrong:
Table X, US quality deciles, I/B/E/S 1999–2013, implied expected return runs **0.27 (junk) → 0.16
(quality), H−L −0.11, t −9.86**, while realized 12-month return runs **0.08 → 0.13, H−L +0.05,
t +6.62**. If that table reads as reported, then **implied upside falls monotonically in quality
while realized return rises monotonically in it** — and a composite carrying 22 points of Quality
alongside 1.35 points of implied upside is, in that one respect, scoring against itself. My own
§1.3 measurement is the same sign and much weaker: −0.171 within sector against `quality_score`.

---

## 4. Where academia and practice agree, and where they disagree

**They agree on the object.** Score the *change* in analyst opinion, not the *level* of a target.
Brav & Lehavy: the predictable part of a revision earns nothing and the gap closes via the analyst.
Barra and Zacks: every analyst descriptor either provider ships is a difference over time.

**They disagree about whether the level is worth anything at all.** Practice says no — nobody
documented here scores it. Academia says *the sector-relative level is worth something, for about a
month, outside large growth, and less so since 2001*. Da & Schaumburg is one paper, on 1996–2004,
and its own large-growth corner is insignificant; but it is a direct test of this exact construct
on a sample including the S&P 500, with 15 controls and net of costs, and it should not be waved
away.

**The balance shifted once §3.0 landed, and the honest statement is now narrower than "they
disagree".** Da & Schaumburg's within-sector result is the one direct test that supports the
construct, on 1996–2004. Against it, on later and larger samples: Green, Hand & Sikochi (2001–2017)
find the level's information worth 16–18¢ on the dollar for one to two quarters and **only** on
claimed undervaluation, with −0.92 of it manufactured by the price falling; Bradshaw, Brown & Huang
find the size of the implied upside is the strongest known predictor that the target will **not** be
met; and §3.1 reports two further papers (unverified) finding the top implied-upside decile has the
*worst* subsequent returns and that betting against analyst optimism pays **more in large caps**.
Practice is unanimous and now rests on seven primary documents: everyone scores *changes*, nobody
scores the level, and MSCI — which built a price-scaled descriptor for earnings, cash flow and
sales — reportedly declined to build one for price targets.

**So the pre-written reading rule still lands on "no weight change", but for a different reason
than I expected when drafting §1.** It is not that the evidence is balanced; it is that
`plan/factor-lab.md` requires *"the lab and the literature or practice pointing the same way"* and
**one change at a time, of at most 3 points** — and the change the evidence now points to is not a
cut at all but a **substitution** (§6.2), which is a construction change needing its own
measurement first. Cutting the weight to zero and stopping there would discard the one thing both
camps agree is informative — the revision — and §1.4 shows it would move no top-25 name anyway. The
weight stays **today**; the proposal below is the live question, and its priority is higher than it
was before §3.0 returned.

---

## 5. Where this contradicts what we currently do

1. **The screener is on the right side of the one distinction that was tested.** It scores a sector
   percentile, not an absolute level. Da & Schaumburg's absolute sort is t = 0.86; their
   within-sector sort is t = 3.55. Nobody chose that on purpose for this metric, and it deserves to
   be written down before anyone "simplifies" the sector-relative percentile machinery.
2. **But what we score is still 71% price at the horizon we score it over.** Nothing in the
   literature claims that a signal three-quarters composed of inverted one-month price change is a
   revisions signal. It is read off the price, and the price is already scored by Momentum (13
   points) and Risk (10 points), in the opposite direction.
3. **It shares nothing with its own category.** −0.007 within sector against `fy1_revision_3m`. A
   category is a claim that its members measure one thing; this member does not.
4. **The momentum-adjacency rule was never applied to it.** `config.yaml` requires re-running the
   §8.4 spanning regression before any further momentum-adjacent metric enters Revisions. This
   metric is −0.317 with the Momentum score within sector and predates the rule, so it has never
   been put through it. That is a gap in our own process, not in the research.
5. **Even the supported version of the signal is faster than this tool.** Two-thirds of Da &
   Schaumburg's first-month profit lands in two weeks, and there is nothing in months 2–6. The
   review cadence is monthly and the bot rebalances monthly.
6. **The Revisions category cannot be tested by the research engine, ever, as currently built.**
   `v2_config` is a filings-and-prices screener with no analyst inputs
   (`research/2026-10-15-pit-record-validation.md`). So no pre-registered experiment can be
   registered for this metric, and no engine number will ever arrive to settle it. For this
   category, research and documented practice are not the *preferred* evidence — they are the
   *only* evidence. Worth stating plainly, because it is easy to assume a test is coming.

---

## 6. Recommendation

### 6.1 No weight change on the strength of this note

The reading rules in §4 say the weight stays when literature and practice disagree, and they do.
Materiality says the same thing from the other direction: removing the metric moves the median
stock 4 ranks and changes the top 25 by zero names, so a cut buys almost nothing and would spend
this project's one-change-at-a-time budget on a rounding error.

### 6.2 The proposal, left OPEN for a second reader

**Replace the scored level with a change in the consensus target**, taking
`price_target_upside`'s 13.5 points, with the level retained as a display-only field.

**On the scaling, which §3.0 changed my mind about.** My first draft divided the change by the
lagged price. Two sourced constructions say not to: MSCI's `PTGF_C` scales the change by **the mean
absolute level of the target itself** (a sign-robust symmetric percentage change, summed over four
63-day buckets with weights 9/7/5/3), and S&P Global MI's documented signal is simply
**`TP_t / TP_{t−1m}`**. Scaling by price would smuggle the price back into the numerator's
denominator and partially re-create the defect being fixed. **Prefer the target's own level as the
scale**, and follow MSCI's bucketed form if the payload history supports it. Also from §3.1, and
new to me: S&P Global MI reports **86% monthly turnover** for the first decile of a 1-month target
change — "analysts make frequent target price changes, particularly with stocks that are trending" —
so a shorter window is not free, and the 90-day version is likely the right trade for a monthly
screener. Novy-Marx & Velikov's 50% turnover bound is the relevant check, as it was for the hold
band.

Why this and not a cut:

- It is the object both the literature and practice point at: Barra's three analyst descriptors are
  all changes; Brav & Lehavy find the information in the unexpected part of the revision and
  **none** in the level once the long-run ratio is accounted for.
- It is the same move this project already made in this category once, and that move was right:
  the 2026-09-10 reweight did not delete the broken thing, it built `fy1_revision_3m`, the
  category's first real revision metric, which is now its heaviest.
- It fixes the staleness defect by construction. A change in the target has the analyst in the
  numerator and no current price at all, so it cannot be 71% price.
- The data is already on disk. §1.2 computed exactly this quantity across 56 payload versions; the
  screener has been publishing `pt_mean` per stock per run since July.

**What the reviewing session must do before building it, and must not skip:**

1. **Re-run the §8.4 spanning regression** (`research/2026-09-07-...` §8.4). A target revision is
   plausibly momentum-adjacent too — targets may follow price — and the category has 85.2% of its
   independence left and a written rule requiring the test. §1.2's beta of +0.245 at 30 days is a
   reason to expect the overlap to be *small*, not an excuse to skip measuring it.
2. **Check it is not `fy1_revision_3m` again.** That metric is the 90-day change in FY1 consensus
   EPS over price. If analysts move targets and EPS estimates together, the new metric is a
   duplicate and the right answer is to do nothing at all. This is the first thing to measure, and
   it is cheap: both series are in the committed payload history.
3. **Decide the fallback cohort question.** 56 run dates exist, so a 90-day lookback is available
   for every stock today, but it was not before late July and will not be after a data outage.
   `factor_engine.py:2094` records Phase 13 fixing a fallback-cohort defect of exactly this kind,
   and `research/2026-10-15-earnings-definition.md` §8.6 refused a better-supported change for it.
   A metric that silently scores "no history" as "no revision" is worse than the metric it replaces.
4. **Ship the frontend in the same commit** (CLAUDE.md 0.8c): `LINEAGE`, `RECOMPUTE` and
   `EQUATIONS` entries, and a `claims.py` registration for any sentence stating how it is computed.

If (2) finds the duplication, the honest outcome is **no change**, and this note should be closed
that way.

### 6.3 What would change my mind about §1.2

The measurement is the load-bearing part of this note, so here is what refutes it, written now:

- If a longer payload history (say 150+ run dates, available around mid-2027) puts the **30-day**
  beta of Δlog target on Δlog price **above 0.60**, then targets are not stale at the horizon that
  matters and the staleness argument fails.
- If `pt_mean`'s refresh cadence turns out to be a Yahoo artifact — testable by comparing against
  any second target source — the 1–3 day rows are void. The 30-day row and the Brav & Lehavy
  cross-check would both have to fail before the conclusion moves.
- If the within-sector correlation with `fy1_revision_3m` rises above +0.25 on a later run, the
  "shares nothing with its own category" claim weakens and should be restated.

---

## 7. What this note does not cover

- **Re-verification of §3.0 and §3.1.** The provider sweep and the second literature read returned
  *after* §1 and §2 were committed. I verified the four figures §3.0 leans on against PDF bytes
  myself (Green/Hand/Sikochi's −0.92 and 16–18¢; Bradshaw/Brown/Huang's 41%→7% and the TP/P logit
  sentence; Bilinski's 59.1% / 74.5% / 49.5%; Da & Schaumburg's Table 2 Panel B upgrade
  percentages). **Everything in §3.1 is unverified by me**, including the three items marked
  **[VERIFY FIRST]** and the five FTSE Russell / S&P DJI rulebooks — those were page-pinned by a
  sub-agent from primary documents, which is better than an abstract and is not the same as
  checked. The deciding session verifies Palley/Steffen/Zhang, Han/Kang/Kim and AQR's Table X
  before any of them is used for anything.
- **Version gaps in the practice documents**, reported and not resolved: the S&P QVM methodology
  read was August 2022 against a current December 2025, S&P Momentum November 2014 against August
  2026, S&P Quality March 2025 against September 2025. The S&P **Enhanced Value** methodology — the
  document defining QVM's value leg — was never opened. None of this bears on the price-target
  question, which is a zero-hit finding in every version read.
- **Asquith, Mikhail & Au (2005)**, read only second-hand (§2.3). Also not obtained: Bonini et al.
  (2010), Bradshaw/Huang/Tan (2019) and **Dechow & You (2020)** — the last being the paper whose
  title is closest to this note's question; its reported finding, that errors in forecasting the
  expected return to risk proxies are the largest component of implied returns, is consistent with
  everything above but rests on abstracts.
- **Brav & Lehavy's beta-by-size breakdown.** Reported second-hand as 1.37 / 1.29 / 1.23 across
  small / medium / large terciles. I could not locate it in the PDF in the time available, so it is
  **not** asserted anywhere above; only the full-sample 1.28 / 1.26 / 1.20 / 1.33 figures, which I
  read directly, are used.
- **`short_interest_ratio`**, the category's other non-revision metric at 13.5 points, −0.238
  within sector against this one. It has a documented free data history
  (`research/2026-10-22-free-data-that-adds-evidence.md`, re-probed 2026-10-23) and no research
  note. It is the obvious next Monday in this category.

## Sources

**Read from the PDF bytes this session:**

1. **Da, Z. & Schaumburg, E. (2011)**, "Relative valuation and analyst target price forecasts",
   *Journal of Financial Markets* 14(1), 161–192.
   <https://academicweb.nd.edu/~zda/TargetPrice.pdf>;
   RePEc `RePEc:eee:finmar:v:14:y:2011:i:1:p:161-192`.
   Figures used, by PDF page: TPER definition p. 2; abstract claims p. 1; sample pp. 4–5; 177 bp
   / t 3.55 p. 13; 79 bp / t 0.86 and the "relative and not absolute" sentence p. 14; two-thirds in
   two weeks (110 bp) p. 13; past-return tilt p. 7; net-of-cost 148 bp / t 3.67 p. 18; 15-control
   robustness p. 21; decimalization and Reg FD p. 3; Large growth 0.65% / 0.23% [0.84] [0.37]
   p. 28.
2. **Brav, A. & Lehavy, R. (2003)**, "An Empirical Analysis of Analysts' Target Prices: Short-Term
   Informativeness and Long-Term Dynamics", *Journal of Finance* 58(5), 1933–1967.
   DOI [10.1111/1540-6261.00593](https://doi.org/10.1111/1540-6261.00593);
   <https://webuser.bus.umich.edu/rlehavy/BL.pdf>.
   Figures used, by PDF page: 223,016 observations / 6,544 firms p. 5; long-run ratio 1.28 / 1.26 /
   1.20 / 1.33 and the 28%-a-year reading p. 27; "market prices barely contribute" p. 3;
   "predominantly made by analysts" p. 28; unexpected revision 2.671 [27.0] and no relationship for
   the expected revision p. 29.

**Second-hand, flagged in the text, and not load-bearing:**

3. Asquith, P., Mikhail, M.B. & Au, A.S. (2005), "Information content of equity analyst reports",
   *Journal of Financial Economics* 75(2), 245–282,
   DOI [10.1016/j.jfineco.2004.01.002](https://doi.org/10.1016/j.jfineco.2004.01.002). Read via a
   text proxy from NBER working paper w9246 (Oct 2002),
   <https://www.nber.org/system/files/working_papers/w9246/w9246.pdf>.

**Documented practice, quoted from this project's own prior sourced reading
(`research/2026-09-07-revisions-category-has-no-revisions.md` §2.1–2.2):**

4. MSCI, "Datasheet — Barra US Total Market Equity Trading Model" (USFAST), March 2015 — the
   `Sentiment` factor and its descriptors.
5. Zacks Investment Research, the four published components of the Zacks Rank.

**This project's own record, cited above:**

6. `research/2026-09-07-revisions-category-has-no-revisions.md` (the category rebuild, §2.1–2.2,
   §8.4, §8.5).
7. `research/2026-10-09-metric-audit.md` (`earnings_acceleration` zeroed).
8. `METHODOLOGY_CHANGELOG.md` 2026-09-02 (`sharpe_ratio` / `sortino_ratio` removed from Risk).
9. `research/2026-10-15-pit-record-validation.md` (`v2_config` carries no analyst inputs).
10. `research/measurements/2026-10-26-price-target-upside.py` + `.json` (every figure in §1).

---

## 8. The review, 2026-10-26 (13:00 session) — the proposal is REFUSED

Written by a session that did not write §1–§7, which is the point of leaving it open
(`prompts/nightly.md`: *"a proposal needs a reader who did not write it"*). Measurements:
`research/measurements/2026-10-26-target-change-candidate.py` and
`-target-change-span.py`, plus `-target-dispersion-feasibility.py`, each with its `.json`.
Figures belong to the published run `bc1fe4648fd6` (2026-10-26T02:00:04) and to the 57
committed versions of `dashboard_data.js`.

**The decision in one line:** the candidate is a genuine revision metric and not a duplicate —
the proposal passed its own first test — but it is a *momentum* signal where the incumbent was
an *anti*-momentum one, and the category rule written into `config.yaml` refuses it.

### 8.1 Step 2 first, as the note asked: it is not `fy1_revision_3m` again

§6.2's step 2 was "the first thing to measure", with "if (2) finds the duplication, the honest
outcome is **no change**". It does not. The candidate, built as MSCI `PTGF_C` scales it, against
`fy1_revision_3m` on the same end date, over every end date in the payload history that has a
start date in range:

| Window | end dates | pooled | **within sector** |
|---|---|---|---|
| ~30 days | 32 | +0.2885 | **+0.1978** |
| ~63 days | 30 | +0.3821 | **+0.3028** |
| **~90 days** (the proposal's own) | **16** | **+0.4340** | **+0.3475** |

At the proposal's window the two share about a third of their within-sector ranking. That is
substantial overlap between two metrics inside one 10-point category, but it is not duplication,
and the comparison that matters is with the incumbent: `price_target_upside` is **−0.007**
within sector against `fy1_revision_3m` (§1.3). On construct validity the candidate is exactly
the improvement §6.2 claimed — it belongs to the category it would sit in, and the incumbent
does not.

**So the escape hatch did not fire, and the proposal has to be judged on the test it deferred.**

### 8.2 Step 1, the §8.4 spanning regression: it fires

`config.yaml:128`, in the Revisions block, is not advice:

> Adding any further momentum-adjacent metric here requires re-running the §8.4 spanning
> regression first, not just a citation.

The candidate is momentum-adjacent. Within sector, over 18 end dates at the 90-day window, the
median Spearman against the Momentum **category score** is **+0.5331**, against `return_6m`
**+0.6260**, and against the change in log price over its own window **+0.5328**. For contrast
the incumbent is **−0.317** against Momentum (§1.3) — the opposite sign.

Re-running §8.4's construction on the published run, with the incumbent's 13.5 points moved to
the candidate and nothing else touched. The rebuild reproduces the published Revisions score at
Spearman **1.0** and a mean absolute difference of **0.0 points** (the composite at 0.99994), so
the counterfactual is this screener's arithmetic and not an approximation of it:

| Category pair | published | after the swap | Δ |
|---|---|---|---|
| **revisions ~ momentum** | **+0.2223** | **+0.3778** | **+0.1555** |
| revisions ~ risk | −0.1488 | −0.0362 | +0.1126 |
| revisions ~ growth | +0.2054 | +0.1553 | −0.0501 |
| revisions ~ quality | +0.0459 | +0.0804 | +0.0345 |
| revisions ~ investment | −0.0690 | −0.0399 | +0.0291 |
| revisions ~ valuation | −0.0123 | −0.0411 | −0.0288 |
| revisions ~ size | −0.1247 | −0.1426 | −0.0179 |
| revisions ~ composite | +0.2907 | +0.3694 | +0.0783 |

**+0.3778 is through the ~0.35 bar §8.4 set before seeing its own number.** That is the whole
purpose of a pre-registered threshold: §8.4 used it to *admit* `fy1_revision_3m` at +0.317, and
it is not available to be widened now that it refuses something. And the independence the 09-07
reweight deliberately spent does not merely shrink again, it is spent in the one direction the
rule protects:

| | published | after the swap |
|---|---|---|
| revisions unspanned by the other seven | 87.3% | **82.1%** |
| momentum unspanned by the other seven | 83.5% | **75.9%** |

Momentum's own independence falls 7.6 points, because Revisions starts carrying it.

**The sharpest single number is the decomposition of the candidate's own overlap:**

| Metric | R² on all eight category ranks | R² on **momentum alone** | share of the spanned variance that is momentum |
|---|---|---|---|
| `price_target_upside` (incumbent) | 0.3507 | 0.0776 | 22% |
| the 90-day target change (candidate) | 0.2598 | 0.2431 | **94%** |

The candidate is *more* independent overall — and almost all of what it does share is momentum.
The swap would not reduce the screener's duplication, it would **concentrate** it, and flip a
−0.32 offset against Momentum into a +0.53 alignment with it. A category whose job the 09-07
note describes as holding "a 10% slot uncorrelated with everything else" would be spending that
slot on the screener's second-largest category.

### 8.3 Why §6.2 expected the overlap to be small, and the error worth generalising

§6.2 wrote: *"§1.2's beta of +0.245 at 30 days is a reason to expect the overlap to be *small*,
not an excuse to skip measuring it."* Two things are wrong with that inference, and the second
is the useful one.

**The arithmetic one.** +0.245 is the **30-day** row of §1.2's own table. The proposal's metric
is a **90-day** change, and that table's ~91-day row reads **+0.540**. The note had the right
number four rows up from the sentence that used the wrong one.

**The one worth carrying forward: a level beta does not bound a rank correlation.** Measured on
the same series at the same gaps — the regression slope of Δlog target on Δlog price beside the
rank correlation a percentile screener actually scores:

| Gap | pairs | obs | beta (levels) | rank corr, pooled | rank corr, **within sector** |
|---|---|---|---|---|---|
| 7 days | 52 | 25,833 | +0.246 | +0.066 | +0.116 |
| 30 days | 39 | 19,255 | +0.245 | +0.220 | **+0.296** |
| 63 days | 20 | 9,798 | +0.374 | +0.499 | **+0.526** |
| 90 days | 7 | 3,410 | +0.540 | +0.602 | **+0.581** |

The beta is flat between 7 and 30 days while the within-sector rank correlation nearly triples.
A target that moves a quarter as far as the price still moves *in the price's direction*, and a
screener that ranks rather than levels keeps all of the direction and none of the attenuation.
**Any future note reasoning from a beta to an overlap in this screener should read this row.**

### 8.4 Three smaller things the review settled

**The scaling choice in §6.2 cannot matter.** §6.2 spends a paragraph preferring MSCI's
`PTGF_C` (change over the mean absolute level of the target) to the note's first draft (change
over price), citing S&P Global MI's `TP_t / TP_{t−1m}` as support. For positive targets both
MSCI's form and S&P's ratio are strictly monotone functions of `TP_t / TP_{t−W}`, so they are
**rank-identical** — verified on **all 98 end dates** across the three windows, equal to four
decimal places on every one. A percentile screener cannot distinguish them. The price-scaled
variant does differ, and trivially: +0.3546 against +0.3475 on the duplication check at 90 days.
The scaling was never the live question.

**Step 3, the fallback cohort, is a real defect and not a solvable one here.** No vendor supplies
a 90-day-ago consensus target, so the candidate can only be built from this project's own payload
history — which means the metric does not exist for any stock until 90 days of runs have
accumulated, and silently loses its window after any outage. The history already contains the
case: the payload committed on **2026-08-07** carries no usable target for any stock, so a window
anchored on it yields **0% coverage** where the next date yields 99.2%. `factor_engine.py:2094`
records Phase 13 fixing a fallback-cohort defect of exactly this shape *in this metric*, and
`research/2026-10-15-earnings-definition.md` §8.6 refused a better-supported change for it.

**The Russell shrinkage rule §3.1 calls "worth copying regardless" is a no-op here — measured,
so it should not be shipped.** Russell shrinks a thin-coverage estimate toward the sector mean
(1 analyst → ⅓ own score, 2 → ⅔, 3+ → full). On the published run, of 496 stocks with a target:
**1** has fewer than 3 analysts (and so already scores nothing, at the metric's own floor),
**1** has exactly 3, **1** has 4–5, and **461 (92.9%) have 11 or more**. Median analyst count
**20**, 10th percentile 12. The rule would touch no stock that is scored. It is good practice on
a Russell 2000 universe and it buys nothing on the S&P 500.

### 8.5 The three [VERIFY FIRST] items, now read from the PDF bytes — and three corrections

§3.1 marked three items unverified and §7 made verifying them the deciding session's first job.
All three were downloaded and read with `pypdf`; the figures below were read from the bytes.

**1. Palley, Steffen & Zhang — figures exact, sample wrong.** "The Effect of Dispersion on the
Informativeness of Consensus Analyst Target Prices", *Management Science*,
DOI [10.1287/mnsc.2021.03549](https://doi.org/10.1287/mnsc.2021.03549). Read from the authors'
copy, <https://insights.som.yale.edu/sites/default/files/2025-01/Consensus%20target%20prices.pdf>
(55pp, "Management Science, forthcoming"; SSRN 3467800). Their PRET is **this screener's metric**:
"the 12-month predicted stock return based on the IBES consensus analyst target price ... defined
as the average target price in month t minus stock price in month t, scaled by stock price in
month t" (PDF p. 13).

Verified exactly as §3.1 reported: deciles PRET1–PRET8 "remain fairly stable around 12% before
dropping to **10.84%** for PRET9 and **7.28%** for PRET10", and a long-PRET10 / short-PRET1 hedge
"yields a return of **−4.60% (t = −3.08)**" (PDF p. 16); the dispersion moderator is
**+2.96% (t = 2.54)** in DISP1 against **−11.44% (t = −5.34)** in DISP10 (PDF p. 17).

**CORRECTION 1 — the sample.** §3.1 reports "465,797 firm-months, Jul 1999 – Jun 2018". The paper
says **537,519 firm-month observations** over **July 1999 to December 2020**, **258 sample months**
(PDF pp. 13, 16–17). The second-hand figures understated the sample and ended it two and a half
years early. The correction strengthens the finding rather than weakening it.

**CORRECTION 2 — the nuance §3.1 dropped.** The positive relation is *not* a general property of
low dispersion: "this positive correlation only manifests when dispersion is very low (e.g., the
DISP1 and DISP2 deciles), consistent with the overall lack of target price informativeness found
in prior studies ... while conditioning on low dispersion can be useful, it is not a panacea"
(PDF p. 17).

And the mechanism, which is **this note's §1.2 in a peer-reviewed sample**: "the differing effect
of dispersion stems from incentive-driven staleness in price targets by some analysts after bad
news. As a stock performs poorly and some analysts are slow to update their target prices,
dispersion increases, and the consensus target price becomes too high" (abstract, PDF p. 1).

**2. Han & Kim — figures exact, but it is not a test of this metric.** The published paper is
Han, Kang & Kim (2022), "Betting against analyst target price", *Journal of Financial Markets*
59, 100677, DOI [10.1016/j.finmar.2021.100677](https://doi.org/10.1016/j.finmar.2021.100677).
The ScienceDirect copy serves a bot check, so per `prompts/nightly.md` it was not worked around;
what was read is the **EFMA 2019 conference version, Han & Kim, January 2019** —
<https://www.efmaefm.org/0EFMAMEETINGS/EFMA%20ANNUAL%20MEETINGS/2019-Azores/papers/EFMA2019_0282_fullpaper.pdf>
(48pp). **A different author list and a different draft from the published paper; flagged, and
nothing below should be attributed to the 2022 version.** Verified (PDF p. 3): US, **1999–2015**;
a long-short portfolio earns "**0.98% per month (t-statistic = 3.80)**"; "the return is more
prominent among **large, liquid, and high institutional ownership stocks**"; and a cumulative
one-year excess return of **15.10% (t = 8.61)**.

**CORRECTION 3 — the signal is not implied upside.** §3.1 reports this as "betting against earns
0.98%/month". The paper's signal is **NOA, the net number of optimistic analysts** — "the
difference between the number of optimistic analysts and the number of pessimistic analysts" —
and the paper explicitly prefers it *over* the construct this screener scores: NOA "is considered
to reflect the impact of target price releases better than other alternative measures such as the
mean or median of target price implied returns" (PDF p. 3). So it is strong evidence that
sell-side target optimism is a **contrarian** signal in large, liquid stocks, and it is **not** a
direct test of `price_target_upside`. Da & Schaumburg (§2.1) and Palley et al. remain the only
two direct tests on the record.

**3. AQR's *Quality Minus Junk* Table X — exact, and it is a direct test.** Asness, Frazzini &
Pedersen, published as *Review of Accounting Studies* 24(1), 2019,
DOI [10.1007/s11142-018-9470-2](https://doi.org/10.1007/s11142-018-9470-2). Read from the working
draft dated **June 19, 2014** (<https://www.nhh.no/globalassets/departments/business-and-management-science/seminars/2015-fall/020915.pdf>,
65pp) — *a pre-publication version, flagged.* The construct is this screener's, stated in the
table's own header: "The implied expected return is given by the ratio of target prices to current
prices minus 1" (PDF p. 45, sheet "Page T12"). Table X, Panel A: U.S., I/B/E/S **1999–2013**,
value-weighted deciles on quality, NYSE breakpoints:

| | P1 (junk) | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 | P10 (quality) | H−L | t |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mean implied expected return | 0.27 | 0.23 | 0.21 | 0.18 | 0.18 | 0.18 | 0.18 | 0.17 | 0.16 | 0.16 | **−0.11** | **−9.86** |
| Realized future 12-month return | 0.08 | 0.10 | 0.10 | 0.11 | 0.11 | 0.11 | 0.12 | 0.12 | 0.13 | 0.13 | **+0.05** | **+6.62** |

Every figure §3.1 reported is exact, and so is its reading: implied upside falls monotonically in
quality while realized return rises monotonically in it. The authors' own words: "analysts have
lower return expectations for higher-quality stocks than junk stocks. In other words, analyst
expectations are inconsistent with the high ex-post realized returns of high-quality stocks"
(PDF p. 21). §1.3's within-sector −0.171 of this metric against `quality_score` is the same sign
on this screener's own data, much weaker.

### 8.6 What this does to §6.1, stated honestly

§6.1 kept the incumbent's weight on the reading that "literature and practice disagree", where
the literature supporting it was **one** paper — Da & Schaumburg (2011), First Call 1996–2004,
whose own large-growth corner is t = 0.37. After verification the ledger on the *level* reads:

- **For**, one direct test: the within-sector sort, **+177 bp/month (t 3.55)**, 1996–2004.
- **Against**, one direct test that is later, larger and peer-reviewed: **−4.60% (t −3.08)** over
  537,519 firm-months and 258 months to December 2020 — and the screener's own sector-relative
  construction is no defence here, because Palley et al.'s moderator is dispersion, not sector.
- **Against**, a second direct test of the construct: AQR's **−0.11 (t −9.86)** of implied upside
  against **+0.05 (t +6.62)** of realized return across quality deciles, 1999–2013.
- **Against**, indirectly and in this screener's own universe: Han & Kim's contrarian NOA result,
  stronger in large and liquid stocks.
- **Practice: unanimous and unchanged.** Seven primary rulebooks, zero price-target level
  descriptors.

**So §6.1's "disagree" is now too generous to the incumbent, and I am recording that rather than
acting on it.** Nothing in §8 justifies a weight change today: the one-change-at-a-time rule
binds, a sign flip or a conditioning rule is a methodology change that needs its own Monday note
and synthesis, and §1.4 measured that removing the metric entirely moves the median stock 4 ranks
and **no top-25 name**. What has changed is that the question is now *live* and pointed, where
before it was open-ended.

### 8.7 The next question, with its criteria written now

**Is the Revisions category's 13.5 points on implied upside pointed the wrong way, and does
dispersion decide it?** Palley et al. is the first paper on this record to answer both parts, and
its moderator is partly measurable from what the fetch already pulls. Measured today
(`-target-dispersion-feasibility.py`):

- **Their DISP cannot be built here.** It is the standard deviation of the individual targets over
  price. The payload carries `pt_high` and `pt_low` (**495 of 498 stocks, 99.4%**) and an analyst
  count, and **no individual targets and no standard deviation anywhere**. Only a high–low range
  proxy is possible, `(pt_high − pt_low) / price`: median **0.474**, p10 0.251, p90 0.997.
- **The proxy has a defect their measure does not**, and it is measured rather than assumed: a
  range widens mechanically with the number of estimates, and **Spearman +0.3999** against the
  analyst count — median range 0.390 at 3–9 analysts against **0.626** at 25+. Any use of it has
  to neutralise coverage first.
- **The finding that makes this worth a Monday.** Within sector, implied upside and the dispersion
  proxy correlate **+0.3679**. The top implied-upside quintile sits at a median dispersion
  percentile of **79.2** against **40.0** for the bottom quintile; **49.1%** of top-upside names
  are also in the top dispersion quintile, and **5.7%** are in the bottom one. *The stocks this
  screener scores best on this metric are concentrated in the region where Palley et al. measure
  −11.44%, and almost absent from the region where they measure +2.96%.* That is not a return
  measurement on this screener's data — it reads no return — and it is exactly the overlap a
  Monday should resolve.

**Pre-registered, before anyone looks further** (so the next session cannot fit the rule to its
answer):

1. The question is researched from the literature **and** practice, including the papers §7 still
   lists as not obtained — **Dechow & You (2020)** first, as the paper closest to it, then
   Bradshaw, Brown & Huang (2013) in its *published* sample, and whether any index provider or
   quant shop conditions an analyst descriptor on dispersion at all. Practice has been unanimous
   on the level; if it is also silent on dispersion-conditioning, that is a disagreement and the
   pre-written rule keeps the weight where it is.
2. **A coverage-neutralised dispersion proxy is a precondition, not a result.** If the proxy's
   correlation with the analyst count cannot be brought under **+0.10** by ranking within
   coverage buckets or an equivalent rule, the moderator is not measurable here and the
   dispersion branch is closed — say so and stop.
3. **Three candidate actions, and the bar each must clear.** (a) *Zero the metric* — needs only
   the §8.6 ledger plus a coherent account of what the freed 13.5 points buy, and must state that
   §1.4 measured the effect as no top-25 change. (b) *Condition it on dispersion* — must pass
   §8.4's ~0.35 bar on `revisions ~ momentum` like any other change here, and must not reintroduce
   the fallback-cohort defect of §8.4 above. (c) *Flip its sign* — **refused in advance** unless a
   session can show the reversal in a long-only large-company book specifically; Palley et al.'s
   −4.60% is a long-short spread whose short leg this tool cannot hold, and three of its four
   adverse points come from the top two deciles rather than from a monotone reversal.
4. **No engine test will ever arbitrate this** (§5.6): `v2_config` carries no analyst inputs, so
   research and documented practice are the only evidence this category will ever have.

### 8.8 What shipped: the rule is now a tripwire

The §8.4 rule did real work today, and nothing enforced it — it was a comment in `config.yaml`
that a session had to remember to read. `tests/test_revisions_momentum_budget.py` holds the
published run's `revisions ~ momentum` to the **0.35** bar, and holds the comment stating the rule
in place.

**Calibrated before shipping, the way the bot's input gates were** (`DECISIONS.md`, settled row
0.5/0.7: *"Replayed against every published page ... all four known-bad runs fail, no ordinary day
does"*). Replayed across **all 57 committed payload versions**: minimum **−0.0835**, median
**+0.2498**, maximum **+0.3206** — that maximum on 2026-09-11..09-18, immediately after the 09-10
reweight, which is §8.4's own predicted +0.317 showing up in the published data. **No ordinary day
fires. The change refused today reads +0.3778 and does.**

No weight, metric or threshold changed.
