# A category a stock cannot be scored on: count it at the sector median, not at the stock's own average

2026-11-19, owner-run (the owner delegated the decision that evening: *"do whatever you think is
best!"*, on the five proposals in `research/2026-11-19-fresh-eyes-review.md`).

**Proposal status: DECIDED 2026-11-19** - built the same night; `METHODOLOGY_CHANGELOG.md` 2026-11-19
(late).

## 1. The question

When a stock has no score for a whole category - its price history failed to download (Momentum
and Risk), it has fewer than three analysts (Revisions), it has no year-ago balance sheet
(Investment) - what should that category count as in the composite?

Until today: nothing. `compute_composite` divided by the weight of the categories the stock had,
so the missing category's weight was shared across the others. Algebraically that is

    composite = sum_present(w_c * s_c) / sum_present(w_c)

which is exactly what you get by **filling the missing category with the stock's own weighted
average of the categories it has.** It is not neutral. It is *row-mean imputation*.

## 2. The failure that made it a question (measured)

On 2026-11-19 SanDisk's daily price history came back empty (the only one of 503 tickers; it had
been present in every one of the previous 19 runs). Its Momentum (13%) and Risk (10%) dropped out.
The day before it had Momentum 75.5 and **Risk 1.03** - the most volatile stock in its sector.
Its other six categories averaged ~79, so the two missing ones were in effect scored at ~79:

| | Composite | Rank |
|---|---|---|
| 2026-11-18, all eight scored | 71.02 | 7 |
| 2026-11-19, two missing, weight shared (shipped) | **79.09** | **1** |
| 2026-11-19 with the 11-18 Momentum and Risk put back | 71.02 | 7 |
| 2026-11-19 with the two missing counted at 50 (this rule) | 72.36 | 3 |

The +8 points were published on the live page and in `signals.json`, which the trading bot reads.
The 2026-09 changelog entry that introduced price-series withholding said the renormalisation was
"neutral by construction - the stock neither gains nor loses from it". That holds only if the
missing category would have scored near the stock's own average.

## 3. What the literature says

- **Chen & McCoy, "Missing Values Handling for Machine Learning Portfolios"** (arXiv 2207.13071,
  v6 revised 2024-01-11; read 2026-11-19, abstract). Across the cross-sectional return predictors
  they study, *simply imputing with cross-sectional means* performs well against rigorous
  expectation-maximization imputation. Their stated reason: cross-sectional correlations among the
  predictors are small, so the observed predictors carry little information about a missing one,
  and a model that tries to use them adds estimation noise.
- **Bryzgalova, Lerner, Lettau & Pelger, "Missing Financial Data"** (SSRN 4106794, 2022; read via
  Quantpedia's summary of 2023-02-25, not the paper itself). Missingness is widespread and **not at
  random** - it clusters across firms and over time, and returns differ by whether a
  characteristic is missing - so ad-hoc imputation can introduce selection bias. They propose a
  factor-model imputation using both time-series and cross-sectional information.

**How they apply here.** The two agree on the thing that decides this case: *the information a
stock's other characteristics carry about the missing one.* Chen & McCoy's argument for the
cross-sectional mean is that it is small. This screener's eight categories are built to be weakly
related - measured today, their correlation matrix has 6.1 effective dimensions out of 8 and
pairwise correlations between −0.45 and +0.31 (`research/2026-11-19-fresh-eyes-review.md`
finding 7). The stock's own average of its other categories is therefore a poor predictor of the
missing one, and for Risk specifically it points the wrong way: Risk correlates −0.29 with Growth
and −0.20 with Revisions. Row-mean imputation is the most information-hungry of the ad-hoc rules
and the one least supported by low cross-category correlation. Bryzgalova et al.'s caution is that
no simple rule is unbiased when missingness is systematic; their remedy (a fitted latent-factor
imputation) would be a second model nobody in an investment club could check, which this project's
mandate rules out. The cross-sectional mean is the simple rule they and Chen & McCoy both treat as
the baseline.

**Within a category the answer is different, and unchanged.** The metrics inside a category are
built to measure the same thing (the valuation ratios correlate +0.50 to +0.67), so the stock's
other metrics *are* informative about a missing one, and sharing a missing metric's weight across
its category - which the screener does and keeps doing - is the defensible rule there.

**Practice.** A missing exposure in a z-scored factor model is conventionally zero, the
cross-sectional mean; on this screener's 0-100 sector-percentile scale the equivalent is 50. *Not
sourced to a named vendor document tonight* - MSCI's factor-index methodologies were searched and
the snippets did not show their missing-descriptor rule. The equivalence (z = 0 is the mean) is
arithmetic, and the argument above does not depend on any vendor doing it.

## 4. The rule built

- A category with a positive weight that a stock has no score for counts at
  `factor_engine.MISSING_CATEGORY_SCORE = 50` at its full weight, in the composite, the category
  contributions, the weight-sensitivity analysis and the investor profiles (all through
  `factor_engine.category_terms`).
- A category the run switched off for everyone (auto-disable under 30% coverage, a 0 weight in a
  profile) is not a term at all, as before - that is a universe-wide decision and redistributing its
  weight is right.
- Unchanged: the coverage filter (under 60% of weighted metrics, not ranked) and the coverage
  discount. The discount still bites on sparse metric coverage; it is simply no longer the only
  thing between a data gap and a higher rank.
- `calc_trace.composite_trace` applies the same rule from its own constant, so the build still
  refuses to publish a composite it cannot reproduce from the payload.

**Why 50 and not the sector median of that category.** Category scores are weighted averages of
sector percentiles, so their sector medians sit near 50 but not on it (measured on 11-19: 44.1 to
57.4). Using each sector's own median would make the filled value depend on the day's cross-section
and would need publishing per sector and category to be checkable. 50 is one number a student can
hold in their head, and the page states it.

## 5. What it changes (measured on the 2026-11-19 payload)

Recomputed from the published payload (category scores, weights and coverage discounts as
published): **6 of 499 composites change; every other one is identical to the cent**, because the
rule touches only a stock with a weighted category missing.

| Stock | Missing | Composite shipped -> rule | Rank shipped -> rule |
|---|---|---|---|
| SNDK | Momentum, Risk | 79.09 -> 72.36 | 1 -> 3 |
| DPZ | Investment | 56.37 -> 56.05 | 121 -> 126 |
| BRK-B | Investment | 52.81 -> 52.67 | 183 -> 185 |
| FDXF | Momentum, Risk, Investment | 46.68 -> 47.25 | 328 -> 316 |
| ERIE | Quality | 41.68 -> 43.34 | 413 -> 395 |
| HONA | Momentum, Risk, Investment | 39.89 -> 42.29 | 431 -> 409 |

The rule is symmetric, which is the point: a strong stock no longer gains from a gap, and a weak
one no longer loses from one. On the next run SNDK should be scored on all eight again if its
history downloads (the history retry shipped the same evening makes that more likely).

## 6. What would make this wrong

- If missingness in a category were itself predictive of that category's score (for example, if
  stocks with no analyst coverage systematically had low Revisions-like outcomes), 50 would be
  biased too. Bryzgalova et al. find exactly this kind of pattern for characteristics in general.
  The counts here are small (6 of 499 today; the health line could track it), so the effect on the
  ranking is small either way - but if the count grows, a per-category check of where missing
  stocks land when their data returns would be the evidence to revisit this.
- Rule 5: no backtest number is offered or used. The research engine is rerun because
  `factor_engine.py` changed, and any difference is a *Backtest observation*, not evidence.
