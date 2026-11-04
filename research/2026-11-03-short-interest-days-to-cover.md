# Days to cover (`short_interest_ratio`): the Revisions category's last unresearched leg

**2026-11-03, owner-run night.** Lab question 8 (`plan/factor-lab.md`) asked what Revisions' 10% is
actually on. Its heaviest leg was researched in September and October (`fy1_revision_3m`, lab
experiment 18), `price_target_upside` on 2026-10-26; **`short_interest_ratio` - 13.5 of the
category's 100 points, 1.35 composite points - never had a note.** This is that note.

**The metric:** Yahoo's `shortRatio`, i.e. days to cover (DTC) = shares sold short / average daily
volume, scored as a sector percentile, **lower is better** (`factor_engine.py`; `METRIC_DIR`
False). Yahoo refreshes it twice a month from exchange short-interest reports, 1-2 weeks late.

**Verdict.** The direction is right in every source. The *strength* the literature documents is an
all-stock, equal-weighted result; among large companies and value-weighted it is a third to a half
as large and rests on few tests, none on the S&P 500. Practice keeps short interest as its **own**
factor, never inside analyst sentiment. By this project's rule - a weight changes only where the
lab and the literature or practice agree, and the lab has nothing here - **the weight stays**. Two
things change in the record: the metric is now researched, and its placement inside Revisions is
named as a structural question for a synthesis day. One proposal, below, makes it testable.

Read from the papers (two readers; figures marked *(checked)* re-found in the PDF text by the
second). Measurement: `research/measurements/2026-11-03-short-interest-ratio-overlap.py`.

## 1. The literature

- **Hong, Li, Ni, Scheinkman & Yan, "Days to Cover and Stock Returns"** (NBER WP 21166, 2015; no
  journal version found). NYSE/Amex/Nasdaq common stocks, 1988-2012, decile sorts held a month.
  DTC as defined is exactly `shortRatio` (footnote 3). Long low-DTC / short high-DTC: **1.19% a
  month equal-weighted (t 6.67)** and **0.67% value-weighted (t 2.24)** *(checked)*; five-factor
  alphas 1.31% (t 8.04) and 0.79% (t 2.60) (Table 3). DTC beats the plain short ratio in a horse race
  and survives beta, size, book-to-market, reversal and momentum (Table 5). Large stocks: with the
  bottom 20% of NYSE size removed, DTC still earns 0.79% equal-weighted (t 6.71) - but that still
  holds mid and small caps; **there is no S&P 500 or top-size test**, and the authors expect DTC's
  edge to shrink among big stocks, whose trading costs vary less.
- **Asquith, Pathak & Ritter (2005, *JFE* 78)**, 1988-2002: the heavily shorted underperform
  equal-weighted, but the 99th-percentile portfolio's value-weighted four-factor alpha is -0.38%
  (t -1.09), and they conclude **"larger stocks with high short interest do not reliably
  underperform"** *(checked)*.
- **Boehmer, Huszar & Jordan (2010, *JFE* 96)**: the *lightly* shorted earn positive abnormal
  returns - the side a long-only screener holds (abstract; their size breakdown is unverified,
  full text not obtained).
- **After publication:** Open Source Asset Pricing's `ShortInterest` (short interest / shares
  outstanding) has **not** decayed equal-weighted (0.93% a month, t 3.70, after publication) - but its
  legs hold ~790 stocks, so small caps dominate; OSAP carries no DTC signal. A practitioner test on a
  large universe (IHS Markit, North America, 95% of float cap, 2007-2012) found a 0.27% monthly spread
  with no t-statistic.

## 2. Practice

MSCI Barra's US models carry **Short Interest and Sentiment as separate factors**, both in the
medium-term trading tier rather than the fundamental set; IHS Markit / S&P Global Research Signals
group "Short sentiment" apart from "Earnings momentum", with days to cover one of its inputs, and
note that exchange short interest arrives twice a month with a lag. Short sellers and analysts are
treated as different, often opposing, sources of information (Drake, Rees & Swanson 2011).

## 3. What it overlaps with here, and what rides on it (measured on the 2026-11-03 payload)

- Coverage 498 of 499; median 3.3 days to cover, 1.7-5.9 between the 10th and 90th percentiles.
- Within sector its percentile (100 = fewest days) correlates **+0.34 with `price_target_upside`**
  and +0.32 with the Revisions score it sits in; **-0.24 with the Size score and -0.24 with Risk** -
  the screener's low-DTC end leans to larger and more volatile companies. Nothing above |0.34|.
- **Removing it** (Revisions renormalised over the rest, composites recomputed and re-ranked the
  engine's way, the published ranks reproduced first): the median stock moves **4 ranks** (90th
  percentile 12, largest 42) and **one top-25 name** changes (HIG out, HAS in).

## 4. Reading it together

Every source agrees on the sign, so lower-is-better stays. The documented size of the effect where
this screener lives - large, value-weighted, long-only, within sector - is marginal, and the
paper's own theory predicts it shrinks there. That argues for caution, not removal: the long-only
side is where Boehmer et al. find the effect, the metric is explainable and free, and it overlaps
little with anything else. The weight question cannot be settled from the literature alone, and
unlike the rest of Revisions it does not have to be: **days to cover is not analyst data**, so the
research engine *can* test it - FINRA's free biweekly files carry `daysToCoverQuantity` for
exchange-listed names back to 2017-12-29 (`research/measurements/2026-10-23-finra-short-interest-probe.py`).

**Placement.** It is not a revision and practice never files it with revisions; the category is
named for something this leg is not, which costs explainability (the 2026-09-07 note's complaint,
one leg later). Moving it is a category-structure change - a Wednesday synthesis, with the
missing-data and profile consequences the Investment-inside-Quality note listed for its own fold.

## 5. Proposal

**Proposal status: OPEN** - register in `research/experiments/registry.yaml`, before running it, a research-engine experiment that drops `short_interest_ratio` from Revisions (and a second that scores FINRA days to cover from 2018 in its place of Yahoo's field, if the v2 record can carry it), with the prediction written first: per the literature, a small positive contribution among large companies, not distinguishable from zero over ~95 months. Decide the weight - and whether it stays inside Revisions - only after reading that result beside this note.
