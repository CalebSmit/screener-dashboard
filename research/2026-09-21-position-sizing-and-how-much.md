# Position sizing: what "how much" can honestly mean

**Date:** 2026-09-21
**Type:** Research note (Monday). Literature + documented practice, complete in
one session. No production code.
**Measurements:** `research/measurements/2026-09-21-position-sizing-dispersion.py`

---

## 1. The question

Of the four questions `plan/dashboard-north-star.md` says the dashboard exists
to answer, **question 4 — "how much / does it fit?" — has no surface at all**,
and has had none since the Model Portfolio was removed on 2026-08-26. The
2026-09-18 session flagged exactly this to the owner.

Before building anything, two things need settling:

1. **What does the evidence say a position-sizing rule should be?** Is
   conviction-proportional sizing defensible, or is equal weight the honest
   default?
2. **What can a decision-support tool say about "how much" without becoming a
   recommendation engine?** The Model Portfolio was deleted because a fixed
   25-name list on a public site came too close to advice. Whatever replaces it
   must not re-create that problem.

This note answers both. It also documents a defect it found on the way: **the
sizing scheme this repo has been configured to use for its entire life does
essentially nothing**, and what it is *trying* to do is the single most
dangerous thing the estimation-error literature identifies.

---

## 2. What the screener does today

`config.yaml -> portfolio:`

```yaml
num_stocks: 25
weighting: 'score'      # 'equal' | 'inverse_vol' | 'score' | 'markowitz'
max_position_pct: 5.0
max_sector_concentration: 8
review_cadence: 'quarterly'
```

`portfolio_constructor.py` computes all four weight columns and writes them to
the Excel sheet; `weighting: 'score'` is the configured default. Score weighting
is composite-proportional (`portfolio_constructor.py:479-481`):

```python
composites = port["Composite"].fillna(...).clip(lower=1.0)
port["Score_Weight_Pct"] = round(composites / composites.sum() * 100, 2)
```

followed by an iterative cap-and-redistribute at `max_position_pct`.

### 2.1 Measured: score weighting is equal weight with noise

Run `research/measurements/2026-09-21-position-sizing-dispersion.py`. It applies
the selection and weighting rules to every snapshot in
`improvement/snapshots/`, one per run date, excluding two degraded February
files that hold 3 rows rather than 502.

**This is a property of the construction arithmetic, not a backtest and not a
return measurement.** It asks only "what weights does this rule emit given these
scores". Nothing here is gated by rules 4 or 5.

Across **39 run dates, 2026-02-20 .. 2026-09-21**:

| | Value |
|---|---|
| Equal weight, 25 names | 4.00% |
| Score weight, full span | **3.77% .. 4.56%** |
| Max deviation from equal weight | **0.57 pp** (median 0.42 pp) |
| Active share vs equal weight, same names | median **1.31%**, max 1.98% |
| Heaviest / lightest weight ratio | **1.21x** |
| Positions ever hitting the 5% cap | **0** |

Restricted to the current composite scale (2026-08-01 onward, 26 run dates) the
picture is identical: span 3.77–4.56%, median active share 1.58%, zero cap hits.

Two consequences:

- **`weighting: 'score'` is not a conviction-weighting scheme.** It is equal
  weight perturbed by less than half a percentage point. The cause is structural,
  not incidental: composite scores are bounded roughly 0–100 and the top 25 of
  502 occupy a narrow band near the top (today 64.75–73.48). Weights are
  proportional to *levels*, so a 13% spread in level becomes a 13% spread in
  weight around 4%. It cannot produce a meaningful tilt and never will.
- **`max_position_pct: 5.0` is inert.** For the cap to bind, one name would need
  a composite 25% above the mean of the selected 25. That has not happened in
  39 run dates and cannot happen while scores are level-bounded and the book
  holds 25 names.

So the screener currently advertises a sizing methodology it does not deliver.
That is a documentation and explainability problem regardless of which scheme is
correct — the tool must be teachable, and "we weight by score" is not true.

---

## 3. What the literature says

### 3.1 Optimised weights lose to 1/N out of sample

**DeMiguel, Garlappi & Uppal (2009), "Optimal Versus Naive Diversification: How
Inefficient is the 1/N Portfolio Strategy?", *Review of Financial Studies*
22(5), 1915–1953.**

Fourteen optimisation models — sample mean-variance, minimum-variance, Bayesian
shrinkage, Bayes-Stein, and others — evaluated across seven empirical datasets.
**None consistently beat 1/N** on Sharpe ratio, certainty-equivalent return, or
turnover. Out of sample, the gain from optimal diversification is more than
offset by estimation error.

The effect size that matters is their calibration result: for a sample-based
mean-variance strategy to reliably beat 1/N, the estimation window needs to be
roughly **3,000 months for 25 assets and 6,000 months for 50 assets** — 250 and
500 years respectively.

**Conditions:** US equity calibration, monthly rebalancing, no short-sale
constraints in the base case, and the comparison is against 1/N *over the same
asset set*. It is a statement about weighting given a universe, not about
selection. That is exactly the question here: the screener has already selected
25 names; the question is only how to weight them.

### 3.2 Why optimisation fails: means are the thing you cannot estimate

**Chopra & Ziemba (1993)**, as set out in **Ziemba & MacLean (2011), "Using the
Kelly Criterion for Investing", ch. 1 in *Stochastic Optimization Methods in
Finance and Energy*, Springer (ISOR 163)** — read from the primary chapter:

> "Chopra and Ziemba (1993) show that in typical investment modeling, errors in
> the means average about **20 times** in importance in objective value than
> errors in co-variances with errors in variances about **double** the
> co-variance errors."

And, crucially, the ratio worsens as risk aversion falls:

> "...for the extreme log investors with essentially zero risk aversion the
> errors are worth about **100:3:1**. So log investors must estimate means well
> if they are to survive."

This is the central result for this note. **Score-proportional weighting is a
mean-forecast-proportional rule.** The composite score is this screener's
estimate of expected relative return. Sizing in proportion to it puts the full
weight of the sizing decision on the one input the literature says is ~20x more
error-sensitive than any other — and the screener's own IC evidence base has
**3 effective observations at the `1m` horizon**, which is to say the accuracy
of that estimate is presently unmeasured.

The combination is the worst case: size by the least reliable quantity, using an
estimate whose reliability is unknown.

### 3.3 Kelly bounds the conviction-sizing intuition

The natural objection is "surely a higher-conviction name deserves more money".
Kelly theory says yes *in principle* and warns severely about it *in practice*.

From the same Ziemba & MacLean chapter:

- **Never bet more than full Kelly.** "Since the growth rate and the security are
  both decreasing for f > f*, it follows that it is never advisable to wager more
  than f*." Overbetting is dominated — worse growth *and* worse security.
- **Betting exactly 2x Kelly drives the growth rate to zero** (plus the risk-free
  rate). The penalty for overbetting is not gradual.
- **Fractional Kelly**: f = 1/(1-α) = 1/RRA, exact for lognormal assets,
  approximate otherwise (MacLean, Ziemba & Li 2005); **Thorp (2008) shows the
  approximation can be very poor**.
- Half Kelly is described as "a toned down version of full Kelly that provides a
  lot more security to compensate for its loss in long-term growth", with
  growth-security dominance formalised in **MacLean, Ziemba & Blazenko (1992),
  *Management Science***.

**A note on a figure I could not verify.** The widely-quoted claim that half
Kelly retains "~75% of the growth rate with ~50% of the volatility" appears in
numerous secondary sources. I could not confirm it in the primary chapter text
and have not used it as evidence. Per `research/README.md`, a blog summarising a
paper is a pointer, not a citation.

**What this licenses for us: nothing.** Kelly sizing requires a calibrated
probability distribution over outcomes. This screener produces a cross-sectional
rank score with no probability attached and no calibrated mapping from score to
expected return. Applying Kelly here would mean inventing the input it needs.

### 3.4 How many names — 25 is at the low end

- **Statman (1987), "How Many Stocks Make a Diversified Portfolio?", *JFQA***:
  at least **30** stocks for a borrowing investor, **40** for a lending investor.
- **Campbell, Lettau, Malkiel & Xu (2001), "Have Individual Stocks Become More
  Volatile? An Empirical Exploration of Idiosyncratic Risk", *Journal of
  Finance***: idiosyncratic volatility rose over 1962–1997, so the number of
  randomly-selected stocks needed for full diversification rises to about **50**.
- **Domian, Louton & Racine (2007), "Diversification in Portfolios of Individual
  Stocks: 100 Stocks Are Not Enough", *Financial Review* 42(4), 557–570**:
  judged by **shortfall risk** over a 20-year horizon against a Treasury-bond
  target rather than by variance reduction, **63 stocks give a 10% shortfall
  risk, 93 give 5%, and 164 give 1%**. Shortfall-risk reduction continues well
  past 100 names. They also find that diversifying across industries helps a
  small portfolio somewhat, but simply adding names helps more.

**Conditions:** all three use randomly-selected portfolios from broad US
universes. A deliberately-selected 25 from the S&P 500 is not a random 25 — it
is 25 large caps pre-screened on quality and liquidity, so its residual
idiosyncratic risk is lower than the random case. The direction still stands:
`num_stocks: 25` sits below every one of these thresholds, and the Domian
criterion (shortfall risk, which is what a student actually cares about) is the
harshest.

This is not an argument to change `num_stocks` — 25 is a defensible
concentration choice for an *active* screen, and the breadth argument in §3.6
cuts the other way. It *is* an argument that the tool should tell a reader where
25 sits relative to the literature rather than presenting it as settled.

### 3.5 Equal weight's advantage is rebalancing, not the weights

**Plyakha, Uppal & Vilkov, "Why Does an Equal-Weighted Portfolio Outperform
Value- and Price-Weighted Portfolios?"** (SSRN 2724535; and "Equal or Value
Weighting? Implications for Asset-Pricing Tests", SSRN 1787045).

With monthly rebalancing, the equal-weighted portfolio beats the value-weighted
portfolio on total mean return, four-factor alpha, and Sharpe ratio. Of the
**2.71% per annum** excess mean return, **58% comes from higher systematic
exposure** (market, size, value) and **42% from alpha**.

The finding that matters here: **the alpha "arises from the monthly rebalancing
required to maintain equal weights, which is a contrarian strategy that exploits
reversal and idiosyncratic volatility... alpha depends only on the monthly
rebalancing and not on the choice of initial weights."**

So the benefit is a property of the *rebalancing discipline*, not of equal
weighting per se. That is directly relevant to a screener whose cadence is
`quarterly` and which has no rebalancing mechanism at all — and it is a reason
the cadence statement shipped on 2026-09-17 is doing more work than it looks.

### 3.6 Breadth, and why constraints cost more than they appear to

**Grinold (1989), "The Fundamental Law of Active Management", *JPM***:
IR ≈ IC x sqrt(Breadth), where breadth is the number of *independent* bets per
year.

**Clarke, de Silva & Thorley (2002), "Portfolio Constraints and the Fundamental
Law of Active Management", *Financial Analysts Journal***: adds the **transfer
coefficient** (TC), the correlation between ideal unconstrained positions and the
positions actually held, so IR ≈ TC x IC x sqrt(Breadth). Constraints —
long-only, sector caps, position caps — reduce TC and therefore scale down
realised IR even when forecasting skill is unchanged.

Two implications for this screener, pulling in opposite directions:

- Cutting from 502 names to 25 cuts breadth hard. sqrt(25/502) ≈ 0.22, so on the
  law's own arithmetic a 25-name book keeps roughly a fifth of the information
  ratio available in the full universe, before any TC penalty.
- But the law assumes *independent* bets, and
  `research/2026-09-02-category-independence-synthesis.md` already establishes
  that this screener's categories are not independent. Nominal breadth
  overstates real breadth here, exactly as raw IC row counts overstate effective
  observations — the same trap, for the third time (`research/README.md`
  Standards).

Neither of these is a reason to change `num_stocks` today. Both are reasons the
tool should not imply that 25 is optimised for anything.

### 3.7 Volatility scaling: the one sizing input with genuine support — and a contested one

**Moreira & Muir (2017), "Volatility-Managed Portfolios", *Journal of Finance*
72(4), 1611–1644.** Scaling a factor's exposure by the inverse of its previous
month's realised variance produces large alphas and higher Sharpe ratios across
the market, value, momentum, profitability, ROE, investment and betting-against-
beta factors, plus currency carry. The mechanism is that changes in volatility
are not offset by proportional changes in expected return. Volatility is
forecastable at short horizons; expected returns are not.

**The replication is genuinely contested, and the disagreement is not minor.**

**Cederburg, O'Doherty, Wang & Yan (2020), "On the performance of
volatility-managed portfolios", *Journal of Financial Economics* 138(1), 95–117.**
Across **103 equity strategies**, volatility-managed portfolios **do not
systematically outperform** their unmanaged counterparts in direct comparison.
The spanning-regression trading strategies are not implementable in real time,
and reasonable out-of-sample versions **generally earn lower certainty-equivalent
returns and Sharpe ratios than the unmanaged originals** — attributed primarily
to structural instability in the spanning regressions. Volatility management
does help **momentum in particular, and profitability and BAB**; it adds nothing
for the other six common factors.

**DeMiguel, Martin-Utrera et al. (2024), "A Multifactor Perspective on
Volatility-Managed Portfolios", *Journal of Finance*** — revisits the question in
a multifactor setting.

**Where this leaves inverse-volatility sizing:** it is the *least* estimation-
sensitive of the sizing inputs (§3.2: variance errors matter ~2x covariance
errors and ~1/10th as much as mean errors), and volatility is the one moment
that is genuinely forecastable. But the out-of-sample evidence for *timing* on it
is weak enough that a note claiming it improves returns would be overclaiming.

The defensible statement is narrower and survives both papers: **inverse-vol
weighting equalises risk contribution rather than dollar contribution.** That is
a statement about what the portfolio *is*, not a forecast — and it does not
depend on the Moreira-Muir result holding.

### 3.8 The audience makes this worse, not better

**Goetzmann & Kumar (2008), "Equity Portfolio Diversification", *Review of
Finance* 12(3), 433–463.** US individual investors hold substantially
under-diversified portfolios. Under-diversification is greater among **younger,
lower-income, less-educated and less-sophisticated** investors, and correlates
with overconfidence, trend-following and local bias. Investors who overweight
high-volatility and high-skewness stocks are less diversified. Under-
diversification is costly to most investors; only a small subset appear to do it
on superior information.

A college investment club is precisely the demographic in that finding. This is
a direct argument that the "how much" surface should push toward **breadth and
concentration awareness**, not toward conviction sizing — the failure mode of
this audience is already too few, too correlated, too volatile.

---

## 4. What practitioners actually do

Documented practice is first-class evidence here, and it is strikingly
consistent: **professionals do not size by conviction. They size by rule, and
then they cap.**

### 4.1 Regulation sets a floor on diversification

- **US RICs, IRC Subchapter M — the 25/5/50 rule.** No single issuer above **25%**
  of total assets, and the sum of all positions exceeding **5%** may not exceed
  **50%** of assets. Essentially every US mutual fund and ETF operates inside
  this. (SEC staff report to Congress on threshold limits for diversified funds,
  2022-02-22; Morningstar, "How the 25/5/50 Rule for Regulated Investment
  Companies Affects ETFs".)
- **UCITS, the 5/10/40 rule.** No more than **5%** of NAV in one issuer,
  extendable to **10%**, provided all holdings above 5% together stay under
  **40%** of NAV. Index-tracking UCITS get a 20%/35% relaxation. (ESMA, UCITS
  Directive Art. 52.)

Note what these are: **caps, not targets.** Regulation constrains the top end of
position size and says nothing about how to distribute weight below the cap.

### 4.2 Index providers cap mechanically, and revise the mechanism

**S&P Dow Jones Indices, Select Sector indices** — rebalance thresholds: weights
are modified if any company exceeds **24%**, or if the sum of companies weighing
more than **4.8%** exceeds **50%** of index weight. Effective before the open on
**2024-09-23**, S&P changed *how* the cap is applied: the legacy method clipped
the smallest company in the breaching group to 4.5% and iterated; the new method
reduces **all** stocks above the 4.8% threshold proportionately to market cap.
The Technology Select Sector index was most affected. (S&P DJI Indexology,
"Explaining Changes to Select Sector Indices", 2024-09-10; S&P DJI Select Sector
capping impact analysis, 2024-08-08.)

**S&P 500 Equal Weight Index** — every constituent is reset to a fixed **0.2%**
at each **quarterly** rebalance. Higher turnover than cap weighting by
construction; it systematically sells winners and buys laggards. Historically it
outperformed the cap-weighted S&P 500 by roughly **1.05% annually** through 2023
and has underperformed since, on the concentration of mega-cap returns. Over half
the historical excess return is attributed to greater weight in smaller
constituents. (S&P DJI index methodology and FAQ; Chincarini, Dash, Dellapa &
Blitzer, "The S&P Equal Weight Index: Uses, Properties and Historical
Simulations", SSRN 4359253.)

That last clause is a warning about the direction of the evidence in §3.5: part
of equal weight's historical edge is a size tilt, which this screener already has
an explicit `size` category for. Adopting equal weight for its return history
would be double-counting a bet the screener already places. Adopting it for its
*estimation-error* properties would not.

### 4.3 Quant shops: constrained optimisation against a risk model

The standard institutional construction is a risk model plus a constrained
optimiser — **Barra** (the US equity model decomposes risk into ~54 industry and
~13 style factors) or **Axioma**, minimising tracking error subject to mandate
constraints on active weight, sector and factor exposure, turnover and leverage.
(MSCI research, "Active Portfolio Construction When Risk and Alpha Factors Are
Misaligned"; SimCorp/Axioma Portfolio Optimizer documentation.)

The load-bearing observation: **the optimiser's job there is mostly to control
risk and constraints, not to express conviction in position size.** Alpha enters
as a forecast that the optimiser trades off against a *measured covariance
structure*. A shop with a Barra model has the covariance estimate that
§3.1's critique assumes is missing; this screener does not — it has no risk model,
and `portfolio_constructor.py:210-211` already records that the ~250x50 sample
covariance panel is "noisy/near-singular and yields unstable corner weights",
which is why `markowitz` is marked experimental. **That comment is an independent
rediscovery of DeMiguel et al. (2009) inside this repo.**

The repo's response was to apply **Ledoit-Wolf shrinkage** to the covariance
matrix (`portfolio_constructor.py:212-213`, via `portfolio_risk._ledoit_wolf_shrink`).
That is the right instinct and it is *also* not enough: shrinkage estimators of
exactly this family are among the fourteen models DeMiguel et al. tested, and
none of them consistently beat 1/N. Shrinking the covariance does nothing about
the mean estimate, which §3.2 says is where ~20x of the damage lives.

### 4.4 The honest summary of practice

Nobody in the documented record sizes long-only equity positions in proportion to
a bounded composite score. The observed schemes are: equal weight; cap weight;
inverse-volatility or risk-parity weight; and optimiser weight against a
commercial risk model. In all four, **the alpha signal drives *selection*, and
weighting is a separate, risk-driven decision.**

---

## 5. Where academia and practice disagree

| | Academia | Practice | Why |
|---|---|---|---|
| **Optimisation** | 1/N beats optimised weights out of sample (DeMiguel 2009) | Large managers optimise anyway, against Barra/Axioma | Practitioners have commercial covariance models with far more structure than a sample covariance matrix, plus mandates that *require* explicit risk control. DeMiguel's critique targets sample-based estimation; it bites hardest on exactly the setup this repo has. |
| **Position caps** | Largely silent — caps reduce the transfer coefficient and cost IR (Clarke et al. 2002) | Universal: 25/5/50, 5/10/40, 4.8%/24% | Caps exist for regulatory, liquidity and career-risk reasons that do not appear in a utility function. Practice accepts a known IR cost to bound tail outcomes. |
| **Number of names** | 30 → 50 → 63-164 depending on criterion (Statman; Campbell et al.; Domian et al.) | Concentrated active books of 25-50 are common | The academic numbers assume *random* selection; active managers argue selection reduces the residual risk being diversified away. Both are right about different portfolios. |
| **Volatility timing** | Large gains (Moreira & Muir 2017) vs. no reliable out-of-sample gain (Cederburg et al. 2020) | Risk-parity and vol-target products are widely sold | An unresolved live dispute in the literature, with product built on the optimistic side of it. |

**The disagreement that matters for us is the first one, and it resolves cleanly
in academia's favour** — because the reason practitioners can defend optimisation
is a risk model this project does not have and is not going to build.

---

## 6. Where the evidence contradicts what we do

**1. `weighting: 'score'` is the wrong scheme, on the strongest evidence in this
note.** It sizes in proportion to an expected-return estimate, which Chopra &
Ziemba show is ~20x more error-sensitive than covariance (and ~100:3:1 for
low-risk-aversion investors), using a score whose predictive accuracy this system
has 3 effective observations on.

**2. It is simultaneously not doing what it claims** (§2.1): 3.77–4.56% against
an equal weight of 4.00%, median active share 1.31%. So the fix costs nothing
in behaviour and buys honesty. This is the rare case where the defensible
change and the inert change are the same change.

**3. `max_position_pct: 5.0` has never bound** in 39 run dates and cannot bind
under the current construction. A parameter that cannot fire is the same failure
shape as an alarm that always fires (fixed 2026-09-01) — it reads as a safety
control and is not one.

**4. `num_stocks: 25` sits below every diversification threshold in the
literature** (30/40, 50, 63–164) and the tool says so nowhere.

**5. The dashboard cannot answer "how much" at all,** and the engine's sizing
work is only in the Excel sheet.

---

## 7. What would change my mind

- **On score weighting:** a calibrated mapping from composite score to expected
  return, with enough independent observations to estimate its slope — i.e. the
  `1m` IC series reaching the 8-effective-observation gate and showing a stable,
  significantly non-zero IC. Then conviction sizing has an input. That is
  roughly 2027-04 at one observation a month, per the existing pre-registered
  hold-band rule.
- **On equal weight:** evidence that the composite's cross-sectional spread
  widens enough that score weighting produces a materially different portfolio.
  The measurement script re-runs and would show it.
- **On inverse-vol:** resolution of the Moreira-Muir vs Cederburg dispute in
  favour of vol timing on broad long-only equity, not just momentum/profitability/BAB.
- **On position caps:** any run where the cap binds. The script reports it.

---

## 8. Recommendation

### 8.1 Methodology: change the default to `equal`, and say why

**Change `portfolio.weighting` from `'score'` to `'equal'`.**

Evidence: DeMiguel, Garlappi & Uppal (2009) — no optimised scheme beats 1/N out
of sample, and the estimation window needed for 25 assets is ~3,000 months;
Chopra & Ziemba (1993) via Ziemba & MacLean (2011) — mean errors dominate at
~20:2:1, worsening to ~100:3:1 for low risk aversion; documented practice —
no institutional scheme sizes long-only equity by a bounded composite score.

Expected effect: **near zero on the portfolio, by measurement** (max 0.57 pp per
position, median active share 1.31%, zero cap breaches). The gain is
explainability: the tool would do what it says. This is a change that makes the
screener *more* teachable, which is the trade `CLAUDE.md` asks for explicitly.

This needs a `METHODOLOGY_CHANGELOG.md` entry. **It is Wednesday's or Thursday's
work, not today's** — today is research and the rotation says no production code.

> **Shipped 2026-09-23** (synthesis day). `portfolio.weighting` is `'equal'`.
> The measurement was re-run first and holds over **41** run dates: span
> 3.771–4.569%, max deviation 0.569 pp, median active share 1.30%, zero cap
> breaches. Changelog 2026-09-23; `tests/test_weighting_disclosure.py`, 28
> tests, 5 of which fail against the pre-change config and artifacts.
>
> **The same session found a defect this note did not look for, in the same
> three lines of config.** `SCREENER_OVERVIEW.md` — embedded verbatim into
> `index.html` — described the weighting with a **two-branch ternary over a
> four-option setting**: `'equal'`, else "Risk-parity (inverse-volatility
> weighting — lower-volatility stocks get more weight)". With `'score'`
> configured it took the `else` on **every run the tool has ever made**, so the
> public site stated the portfolio was inverse-volatility weighted while it was
> score weighted, which tilts the *opposite* way. Limitation 7 misdescribed the
> same thing, listing `score` as using "single-name volatility only". §2.1 said
> score weighting "is not doing what it claims" and measured the weights; the
> claim itself was wronger than that — it was published as a different scheme
> entirely. Both sites are now a per-scheme mapping pinned to the live config.

Keep `score` and `inverse_vol` as selectable options; the Excel sheet already
shows all four columns side by side, which is good teaching material.

### 8.2 Do not adopt inverse-vol as the default

It is the more sophisticated-looking choice and the evidence does not support
promoting it over equal weight. Moreira & Muir is contested by Cederburg et al.
across 103 strategies, and equal weight is the benchmark that keeps winning.
Leave it available and labelled for what it is: equalising risk contribution, not
improving expected return.

### 8.3 Do not implement Kelly or fractional Kelly

No calibrated probability distribution exists to feed it (§3.3), and the penalty
for overbetting on a bad mean estimate is severe and asymmetric. Building it
would mean inventing the input.

### 8.4 Product: answer "how much" with inputs, not a weight

The reason question 4 has no surface is that the obvious implementation — print
a recommended weight per stock — is precisely what got the Model Portfolio
deleted, and rightly. But "how much" decomposes into questions the tool can
answer without advising:

1. **How concentrated is what I already hold?** My Holdings knows the tickers. It
   can show sector concentration, the count of names against the literature's
   thresholds (§3.4), and how correlated the holdings' factor exposures are —
   all facts about the reader's own list, not advice about it.
2. **How volatile is this name relative to its peers?** `volatility`, `beta` and
   their percentiles are already in `stock_detail`. A position twice as volatile
   as another contributes twice the risk at equal dollars. That is arithmetic, and
   it is the single most decision-relevant sizing fact available.

   > **Corrected 2026-09-22, on implementation: use the RAW volatility, not the
   > percentile.** This item originally said "volatility percentile", and the
   > percentile cannot carry the claim. Percentiles on this site are
   > *sector-relative* (`compute_sector_percentiles` groups by `Sector`) **and**
   > direction-inverted (`METRIC_DIR['volatility']` is `False`), so a high value
   > means "calm *for its sector*" - which is not comparable across a
   > mixed-sector list, and the "twice as volatile" arithmetic above is
   > explicitly a cross-holding comparison. Measured over all **111,417**
   > cross-sector pairs of the 501 names carrying both figures, the percentile
   > orders the pair **backwards 23.9%** of the time; worst case LITE reads as
   > the safer holding than ARE while carrying **2.00x** the volatility. Raw
   > annualised volatility spans **2.55x** from p10 to p90, so the comparison is
   > worth making - just not with that input.
   > `research/measurements/2026-09-22-holdings-risk-comparability.py`
   > reproduces every number; changelog 2026-09-22.
3. **What does an equal-weight slice look like?** For a reader holding N names,
   1/N is a statement, not a recommendation — and it is the benchmark the
   literature says is hard to beat.
4. **Where does my concentration sit against documented practice?** 5% (UCITS),
   25/5/50 (RIC), 4.8%/24% (S&P Select Sector) are published external anchors,
   not this tool's opinion.

**The line that keeps this defensible:** the tool describes *the reader's list*
and *the external standards*, and never emits a target weight for a specific
stock. "Your 6 holdings are 4 of 11 sectors, and Statman (1987) found 30-40
names for a diversified portfolio" is decision support. "Hold 4.2% of AVGO" is
not.

Zero payload cost — every field required is already in `stock_detail` and
`localStorage`, exactly as the holdings panel was.

---

## 9. Wednesday's design section

**Hypothesis.** Question 4 can be answered by describing the reader's existing
holdings against published external anchors, without emitting any target weight
— and doing so serves this audience better than conviction sizing, because
Goetzmann & Kumar (2008) show under-diversification, not mis-weighting, is the
demographic's actual failure mode.

**Implementation sketch** (for Thursday, not Wednesday):

- **Methodology:** flip `portfolio.weighting` to `'equal'` with a changelog entry
  citing §3.1, §3.2 and §4.4. Note in the entry that the measured portfolio
  effect is ~0.5 pp per position and that the change is made for estimation-error
  and explainability reasons, not performance.

  > **Shipped 2026-09-23.** See the block in §8.1 for what else the change
  > turned up. One addition to the coherence argument below, worth keeping:
  > score weighting was the one place where the **category-overlap problem
  > propagated into position size**. Whatever double-counting exists among the
  > eight categories entered the composite, and the composite then set the
  > weight — in the input Chopra & Ziemba identify as ~20x the most damaging.
  > Separating selection from sizing does not fix the overlap, but it stops it
  > compounding.
- **`max_position_pct`:** leave the value alone but state in the changelog that
  it is currently inert, so a later session does not mistake it for an active
  control. Do not delete it — it becomes live if `num_stocks` ever falls.

  > **Shipped 2026-09-23, and stated on the page rather than only in the
  > changelog.** A changelog entry is read by whoever goes looking; the reader
  > of the methodology page is the one being told a 5% cap protects them. Under
  > equal weighting of 25 names every position is exactly 4.00%, so the cap
  > binds only **below 20 holdings** — arithmetic, not an observation — and
  > `SCREENER_OVERVIEW.md` now says so next to the cap itself. `_max_pos_note()`
  > stays silent for the dispersed schemes, where the cap genuinely can fire,
  > and for the infeasible case `portfolio_constructor` already warns about.
- **Dashboard:** a **Concentration** block on the My Holdings panel — name count
  vs the §3.4 thresholds, sector spread, and the volatility percentile of each
  holding. Read entirely from `stock_detail` and the saved ticker list. No cost
  basis, no share count, no P&L (the 2026-09-14 constraints still bind), and no
  target weight.

  > **Shipped 2026-09-22**, with one change: the risk line reports the **raw**
  > annualised volatility of the widest-apart pair on the list, not a
  > percentile — see the correction in §8.4 item 2, which measures the
  > percentile ordering risk backwards for 23.9% of cross-sector pairs. Sector
  > spread was already on the 2026-09-15 fit line, so the block adds the name
  > count against the published thresholds, the equal-split slice against the
  > published caps, and the risk gap. Changelog 2026-09-22;
  > `tests/test_holdings_concentration.py`, 33 tests, all 33 failing against the
  > pre-change generator.

**How this fits the rest of the screener (the coherence question).** Sizing is
the one place where the eight categories do *not* apply, and that is the point:
selection uses the composite, weighting must not. Keeping them separate is what
§4.4 says every practitioner does, and it also insulates the sizing surface from
the category-independence problem in
`research/2026-09-02-category-independence-synthesis.md`. Note also that the size
tilt in equal weight's historical record (§4.2) overlaps the screener's existing
`size` category — a reason to justify equal weight on estimation-error grounds
rather than on its return history, or the screener would be betting the same way
twice and calling it two things.

**The measurement that would refute it.** Re-run
`research/measurements/2026-09-21-position-sizing-dispersion.py` after any change
to the composite scale or `num_stocks`. If the score-weight span ever exceeds
roughly 3.0–5.5% or the 5% cap binds, §2.1's "score weighting is inert" claim has
expired and the choice of weighting scheme becomes a live decision with real
portfolio consequences rather than a documentation fix.

---

## 10. Sources

- DeMiguel, V., Garlappi, L. & Uppal, R. (2009). "Optimal Versus Naive
  Diversification: How Inefficient is the 1/N Portfolio Strategy?" *Review of
  Financial Studies* 22(5), 1915–1953.
  https://academic.oup.com/rfs/article-abstract/22/5/1915/1592901
- Ziemba, W.T. & MacLean, L.C. (2011). "Using the Kelly Criterion for Investing",
  ch. 1 in *Stochastic Optimization Methods in Finance and Energy*, Springer
  ISOR 163. (Reports Chopra & Ziemba 1993 at 20:2:1 and 100:3:1; MacLean, Ziemba
  & Blazenko 1992; MacLean, Ziemba & Li 2005; Thorp 2008.)
  https://webhomes.maths.ed.ac.uk/mckinnon/blackouts/StochOptFinanceAndEnergySpringer/Chap1_KellyZiemba.pdf
- MacLean, L.C., Ziemba, W.T. & Blazenko, G. (1992). "Growth versus Security in
  Dynamic Investment Analysis." *Management Science* 38(11).
- Statman, M. (1987). "How Many Stocks Make a Diversified Portfolio?" *Journal of
  Financial and Quantitative Analysis* 22(3), 353–363.
- Campbell, J.Y., Lettau, M., Malkiel, B.G. & Xu, Y. (2001). "Have Individual
  Stocks Become More Volatile? An Empirical Exploration of Idiosyncratic Risk."
  *Journal of Finance* 56(1), 1–43.
- Domian, D.L., Louton, D.A. & Racine, M.D. (2007). "Diversification in Portfolios
  of Individual Stocks: 100 Stocks Are Not Enough." *Financial Review* 42(4),
  557–570. https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6288.2007.00183.x
- Plyakha, Y., Uppal, R. & Vilkov, G. "Why Does an Equal-Weighted Portfolio
  Outperform Value- and Price-Weighted Portfolios?"
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2724535 ; and "Equal or
  Value Weighting? Implications for Asset-Pricing Tests."
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1787045
- Moreira, A. & Muir, T. (2017). "Volatility-Managed Portfolios." *Journal of
  Finance* 72(4), 1611–1644.
  https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513
- Cederburg, S., O'Doherty, M.S., Wang, F. & Yan, X.S. (2020). "On the
  performance of volatility-managed portfolios." *Journal of Financial Economics*
  138(1), 95–117.
  https://www.sciencedirect.com/science/article/abs/pii/S0304405X2030132X
- DeMiguel, V., Martin-Utrera, A. et al. (2024). "A Multifactor Perspective on
  Volatility-Managed Portfolios." *Journal of Finance*.
  https://lbsresearch.london.edu/id/eprint/3716/
- Grinold, R.C. (1989). "The Fundamental Law of Active Management." *Journal of
  Portfolio Management* 15(3), 30–37.
- Clarke, R., de Silva, H. & Thorley, S. (2002). "Portfolio Constraints and the
  Fundamental Law of Active Management." *Financial Analysts Journal* 58(5),
  48–66. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=934440
- Goetzmann, W.N. & Kumar, A. (2008). "Equity Portfolio Diversification." *Review
  of Finance* 12(3), 433–463.
  https://academic.oup.com/rof/article-abstract/12/3/433/1598033
- SEC (2022). *Staff Report to Congress Regarding Threshold Limits for Diversified
  Funds.* https://www.sec.gov/files/staff-report-threshold-limits-diversified-funds.pdf
- Morningstar. "How the 25/5/50 Rule for Regulated Investment Companies Affects
  ETFs." https://www.morningstar.com/funds/does-your-index-fund-actually-represent-market
- ESMA. UCITS Directive Article 52 (the 5/10/40 rule).
  https://www.esma.europa.eu/publications-and-data/interactive-single-rulebook/ucits/article-52
- S&P Dow Jones Indices (2024-09-10). "Explaining Changes to Select Sector
  Indices." https://www.indexologyblog.com/2024/09/10/explaining-changes-to-select-sector-indices/
  ; capping impact analysis 2024-08-08,
  https://www.spglobal.com/spdji/en/documents/additional-material/select-sector-capping-impact-analysis-20240808.pdf
- S&P Dow Jones Indices. *S&P 500 Equal Weight Index* methodology and FAQ.
  https://www.spglobal.com/spdji/en/education/article/sp-500-equal-weight-index-faq/
- Chincarini, L.B., Dash, S., Dellapa, M. & Blitzer, D.M. "The S&P Equal Weight
  Index: Uses, Properties and Historical Simulations." SSRN 4359253.
- MSCI. "Active Portfolio Construction When Risk and Alpha Factors Are
  Misaligned." https://www.msci.com/documents/10199/c6e5e3f7-cd44-4322-aeb5-331e20e2afb7
