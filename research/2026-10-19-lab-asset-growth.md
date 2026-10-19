# Factor Lab experiment 14: asset growth (Cooper, Gulen & Schill 2008)

**Date:** 2026-10-19. **Kind:** a lab rerun and what it says about a standing research question.
**Not a proposal.** Nothing here changes a weight or a metric; the lab never justifies a screener
parameter by itself (`plan/factor-lab.md`, rule 1, and "From the lab to the weights").

## 1. Why this experiment

Asset growth is the whole of the screener's Investment category (5% of the composite), scored
lower-is-better. Until today the lab put only factor-level evidence behind it (experiments 3 and
8, the CMA factor). It is also the measure at the centre of the open Growth question:
`research/2026-10-12-growth-category.md` sections 10-11 found that `revenue_growth` (25% of
Growth) has a Spearman of **+0.51** with `asset_growth` across the screener's companies, so for
many companies Growth and Investment score one fact in opposite directions. The lab now has a
direct rerun of the study that established the measure.

## 2. The paper, and where each figure came from

Michael J. Cooper, Huseyin Gulen and Michael J. Schill, "Asset growth and the cross-section of
stock returns", *Journal of Finance* 63(4), 1609-1651 (2008), doi:10.1111/j.1540-6261.2008.01370.x.
Read from the text layer of a copy of the publisher's PDF (43 pages, pp. 1609-1651) with `pypdf`;
the text layer carries every cell of Table II, so nothing was read off a rendered image.

- **Measure** (p. 1613, equation 1): ASSETG(t) = total assets at the fiscal year ending in calendar
  t-1 over total assets at t-2, less one (Compustat item 6). Formed each June, held July to June.
- **Sample** (pp. 1612-1613): NYSE, Amex and NASDAQ **non-financial** firms (SIC 6000-6999 excluded),
  two years on Compustat required; returns July 1968 to June 2003. The decile breakpoints are
  drawn on the paper's whole sample (NYSE breakpoints are mentioned only for its size sorts,
  footnote 7, p. 1614).
- **The table reproduced:** Table II, Panel B.1 (p. 1618, equal-weighted) and Panel B.2 (p. 1619,
  value-weighted), the "YEAR 1" row: average monthly raw returns of the ten deciles in the year
  after formation, the 10-1 spread and its t-statistic. The text on p. 1622 restates the ends and
  spreads: equal-weighted 1.99% and 0.26%, spread -1.73% (t -8.45); value-weighted 1.48% and
  0.43%, spread -1.05% (t -5.04).
- **Not reproduced:** the Fama-French alphas (Panel C, p. 1620), including the large-firm
  value-weighted alpha spread of -0.81% a month (t -2.91) that supports the abstract's "even on
  large capitalization stocks"; the subperiod alphas (Panel D, p. 1621); the regressions (Table III).
  The public file has no size-by-asset-growth deciles.

What the paper does not say: how the decile breakpoints treat NASDAQ firms beyond "deciles
based on annual asset growth rates" (p. 1614); whether delisting returns are included. The
second matters (see section 5).

## 3. Tolerances, written and committed first

Commit `8f7ec9f` (2026-10-19) holds the paper figures and tolerances, committed before the
library's `Portfolios_Formed_on_INV` file had been compared with anything. Only the file's
header and its first rows (1963-64, before the paper's sample) had been looked at, to find the
columns. The library sorts on the paper's own variable, so this is not a different-measure rerun;
three construction differences remain - it keeps financials and utilities, draws its decile lines
on NYSE stocks only, and does not require two years on Compustat.

| Figure | Tolerance | Why |
|---|---|---|
| Each decile's average month (20 figures, both weightings) | 0.10% | Experiment 10's scale for a construction difference (same-file reruns used 0.05) |
| Spread, slowest minus fastest (2) | 0.20% | About one standard error of the paper's own estimate (1.73/8.45 = 0.205; 1.05/5.04 = 0.208) |
| t-statistic of the spread (2) | 1.5 | The spread's tolerance is one unit of t, plus half a unit for a different volatility |

**Plan rule 8**, checked from the record before committing: a result of no effect matches **6 of
the 24** (three middle deciles in each weighting) and misses every end, both spreads and both
t-statistics. Expected beforehand: the shape matches; the equal-weighted ends and spread land
below the paper's; the value-weighted ones closer.

## 4. The comparison: 13 of 24 within

| | Paper | Here | Verdict |
|---|---|---|---|
| Equal-weighted, slowest decile | 1.99 | 1.67 | outside |
| Equal-weighted, fastest decile | 0.26 | 0.45 | outside |
| **Equal-weighted spread** | **1.73** | **1.22** | **outside** |
| Its t-statistic | 8.45 | 8.03 | within |
| Value-weighted, slowest decile | 1.48 | 1.23 | outside |
| Value-weighted, fastest decile | 0.43 | 0.56 | outside |
| **Value-weighted spread** | **1.05** | **0.67** | **outside** |
| Its t-statistic | 5.04 | 4.05 | within |

All figures in % a month except t. The middle of the table matches: four of the eight middle
equal-weighted deciles and seven of the eight middle value-weighted ones (all but decile 9, the
second-fastest growers) are within 0.10. In both
weightings the slowest growers earn the most and the fastest the least, as in the paper.

**Reading it.** The direction reproduces strongly - t 8.0 and 4.1 on the public sort - and both
spreads fall short of the paper's, by 30% equally weighted and 37% by market value. The
equal-weighted shortfall was predicted from the construction (NYSE lines put more, and milder,
companies in each end group; financials kept). **The value-weighted shortfall was not predicted**,
and it is the figure that matters for a large-company screen. Two candidate reasons, neither
tested here: the paper's non-financial universe (large banks and insurers sit in the library's
value-weighted end groups, where balance sheets grow for reasons unlike an industrial company's
capex), and the treatment of delisting returns, which the paper does not describe. A later study
that came up while searching for this paper argues the low-growth premium is partly a
delisting-return effect; it was not read for this note, so it is recorded as a lead, not relied
on. The tolerances stay as written; the two misses are published as misses.

## 5. After the paper

The same file, value-weighted unless stated, % a month (t-statistic):

| Period | Months | Equal-weighted spread | Value-weighted spread |
|---|---|---|---|
| The paper, 1968-07 to 2003-06 | 420 | 1.22 (8.03) | 0.67 (4.05) |
| After the paper, 2003-07 to 2026-08 | 278 | 0.32 (1.73) | **-0.07 (-0.29)** |
| Since 2014 | 152 | 0.12 (0.44) | -0.17 (-0.45) |
| Everything, 1963-07 to 2026-08 | 758 | 0.87 (7.81) | 0.32 (2.50) |

And the library's investment factor among big companies (experiment 8's series, conservative
minus aggressive): 0.37 (t 2.63) in the paper's years, **-0.06 (t -0.35)** since. Its held side
(conservative big companies minus the market) went from 0.25 (t 2.93) to 0.03 (t 0.36).

So the asset-growth premium **weighted by market value has been about zero for the 23 years since
the paper's sample ended**, by the sort and by the factor. Equally weighted it survived at a
quarter of its size, not distinguishable from zero. By the reading rules in `plan/factor-lab.md`
this is "not confirmed", not "gone": 278 months with a t-statistic inside 2. It is consistent with
the post-publication decay McLean & Pontiff (2016) measure across anomalies, and with experiment
3's finding that none of the four non-market factors has been distinguishable from zero since
2014. One thing the sort does keep: the slow-growth end has been the calmer one in every period
(value-weighted volatility 5.10% against 5.65% a month since 2003), the same pattern experiment 10
found for profitability.

## 6. What it says about the Growth / Investment composition question

The open Growth question (growth note, section 11) is composition: the sign of `revenue_growth`
against `asset_growth` - two categories, one fact, opposite directions. This experiment adds
evidence to one side of that question and changes nothing on the other.

1. **Investment's measure has a published premium, and it reproduces in the paper's years** - in
   direction and in significance, at a smaller size. It is the one growth-like measure in the
   screener with a rerunnable premium behind it, and the screener scores it the way that premium
   points (slow growth preferred).
2. **Among large companies since 2003 it has earned about nothing.** The screener ranks only the
   S&P 500, and a long-only, large-company book is the value-weighted one
   (`project_real_money_is_the_goal`). On the lab's own evidence, the 5 points on asset growth are
   today a bet on a premium that has not shown up where the screener operates for two decades -
   the same position as Value, Investment and Size in experiments 3 and 8, and no worse than them.
3. **That does not make Growth's revenue growth "right".** No study the growth note found gives
   revenue growth a premium of its own (Lakonishok, Shleifer & Vishny 1994 find five-year sales
   growth *negative*). The +0.51 correlation means that where the two categories disagree they
   largely cancel; this experiment says the asset-growth side of that cancellation was well
   founded in its paper and has been inert since. It does not say the revenue-growth side earns
   anything.
4. **The composition question therefore becomes narrower, not settled:** if the screener keeps a
   growth exposure near neutral (the growth note's section 10 finding, at 13 points), is it
   better to hold that neutrality with a measure that partly *duplicates* Investment in reverse
   (`revenue_growth`), or with one that does not? That is a design question about overlap and
   explainability, and the evidence here bears on it only by showing that neither side of the
   overlap is earning a premium among large companies today. It needs the Monday note the growth
   note already asked for - the literature and practice on sales growth versus asset growth, and
   what index providers do with each - before anything moves.

**What would change this reading:** a value-weighted, non-financial rebuild of the sort (needs
company-level data the lab does not have); the paper's large-firm alpha rerun with the library's
three factors (the 2x3 size-by-investment file could approximate it - a candidate queue item);
or the screener's own `1m` record, which accrues about one independent observation a month and
decides nothing yet (CLAUDE.md rule 4).

## 7. Route to a weight

None proposed. Per "From the lab to the weights": the lab raised the question (the scorecard's
Growth row now cites this sort), a Monday note must answer it from literature and practice, a
Wednesday synthesis must place it among the other seven categories, and only then a changelog
entry, one change at a time, at most 3 points. Lab and literature would also have to point the
same way; today the literature (the paper, and the CMA factor's place in Fama & French 2015 and
Hou, Xue & Zhang 2015) supports asset growth and the lab says it has not paid among large
companies since 2003. Where they disagree, the weight stays.

## 8. Reproduce

    python factor_lab.py --build     # needs data/lab/french/Portfolios_Formed_on_INV_CSV.zip (--fetch)
    python -m pytest tests/test_factor_lab.py -q

`tests/test_factor_lab.py::test_the_asset_growth_tolerances_are_the_ones_committed_before_the_comparison`
pins the paper figures and tolerances; `::test_the_asset_growth_sort_is_published_as_it_landed`
pins 13 of 24, the two misses and the shape; takeaway check 14 holds the front-page sentence to
the data.
