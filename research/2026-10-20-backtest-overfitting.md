# If you had kept the best backtest - the probability of backtest overfitting on the research engine

**2026-10-20, owner-run.** Owner: *"should we be running more backtests, like all the backtests we
possibly can to decide what is best moving forward for the methodology, weightings, etc?"*

**Answer: no, and the engine's own ledger now shows why.** Picking the best of the weightings already
tested would have chosen a screen whose lead is most plausibly a gap in the data, and with that one
screen set aside, or on the recent era alone, the best-looking backtest ranked in the bottom half out
of sample more often than not. More tests on the same fifteen years would add more candidates for luck
to favour, not more evidence. No weight changes (rules 4 and 5).

## 1. The method

Bailey, D. H., Borwein, J., Lopez de Prado, M. & Zhu, Q. J. (2017), "The probability of backtest
overfitting", *Journal of Computational Finance* 20(4), 39-69 - combinatorially symmetric
cross-validation (CSCV):

1. Split the months into S contiguous blocks.
2. For every way of choosing S/2 blocks as the in-sample history, select the configuration with the
   best in-sample mean, then find its rank among all configurations on the remaining blocks.
3. PBO is the share of splits in which that winner ranks in the bottom half out of sample
   (logit of its relative rank <= 0).

Under the null of no real differences the winner's out-of-sample rank is uniform, so PBO is about
0.5; a configuration that is genuinely and consistently better drives it toward 0. The paper's
practical point is the one the owner's question turns on: with enough trials, the best in-sample
result is guaranteed to look good whether or not anything is there. The same concern is behind
Harvey, Liu & Zhu (2016, *RFS* 29(1)) and their t > 3 bar, which the engine already applies to each
single comparison. PBO asks it of the whole set at once.

## 2. How it is applied here (`overfitting.py`)

- **Candidates:** the current weighting and every comparable, non-void registered alternative - 28
  in all (`sector-relative-composite` is void). Comparable means the engine's own rule: same monthly
  records, same method version, same provenance as the baseline row.
- **Score per month:** each alternative's IC minus the current weighting's IC on the stocks both
  ranked that month (`ic_by_month - base_ic_by_month`, method version 2); the current weighting
  scores 0. This is the engine's own currency - a difference in ranking, not in which companies were
  ranked.
- **Months:** 176, 2012-01 to 2026-08; S = 16 blocks of 11 months, C(16, 8) = 12,870 splits. The
  recent era (2020-2026, 80 months) uses S = 8, 70 splits.
- **Reads the committed ledger only.** It reruns nothing and is not in `v2_experiments.RESULT_PATHS`,
  so editing it cannot stale a ledger row.

Sanity checks (`tests/test_overfitting.py`): pure noise, 28 configurations, 176 months, five seeds -
mean PBO within 0.35-0.65; one configuration given a real edge of half a monthly standard deviation -
PBO < 0.1 and it wins > 90% of splits.

## 3. Result (ledger as of the 2026-10-20 evening rerun; identical before it)

| Reading | PBO | In-sample winner beat the current weighting out of sample | Most frequent winner |
|---|---|---|---|
| 2012-2026, all 28 | **0.21** | 79% of splits | Size alone, 68% of splits |
| 2020-2026 only | **0.73** | 34% | Valuation alone, 21% |
| 2012-2026, Size alone left out | **0.63** | 38% | Growth alone, 27% |

Size alone's lead over the current weighting was +0.027 in 2012-2019 (t 1.83) and -0.015 over the
held-out 2023-07..2026-08 months (t -0.53).

## 4. Reading it

**A low PBO measures consistency, not cause.** The whole-history 0.21 is almost entirely one screen,
and that screen's lead sits in 2012-2019, the era with the largest survivorship gap: backtest v2 holds
58% of 2012's members against 78% of 2019's (`cache/v2_census_presample.json`), and the members it is
missing are the ones that later left the index - disproportionately the small ones that did badly.
Ranking on size within the S&P 500 then rewards exactly the small members that survived. The research
page has carried that as a hypothesis since 2026-10-16 (`research_page.size_paragraph`); this check
does not prove it, but the two readings that remove the survivorship-heavy era or the screen itself
both land above 0.5. A bias that is the same every month looks, to CSCV, like skill.

So: **"run everything and keep the winner" would have selected the configuration most exposed to the
backtest's known defect, and otherwise selected noise.** That is the measured case for the rule the
project already follows - weights from published research and practice, one change at a time, every
test registered before it runs - and against widening the search.

## 5. What would actually add evidence

Not more configurations on the same months, which only raises the multiple-testing bar (p < 0.05 / n)
and gives luck more to choose from. What adds evidence is more *independent* data per question:

1. **Longer and broader history for the factor questions** - the Factor Lab already reruns papers on
   Ken French's data back to 1963. Two free, published libraries extend that (see
   `plan/factor-lab.md`, "Free data to evaluate"): Jensen, Kelly & Pedersen (2023, *JF* 78(5)) - 153
   factors, the US and 92 other countries, CC BY-NC 4.0 - and Chen & Zimmermann (2022, *Critical
   Finance Review* 11(2)) - portfolio returns for 200+ published predictors.
2. **Closing the survivorship gap** - the queued rebuild of departed members from free filings and
   prices (the Size test written down in advance), which is what would settle section 4.
3. **The live record** - about one independent monthly observation a month, the only data no one has
   fitted to.

## 6. Decision

No methodology change. `overfitting.py` and the page section "If you had kept the best backtest"
(`research.html`) report the three readings, each sentence generated from the numbers, and they
regenerate with every `python v2_experiments.py report`.
