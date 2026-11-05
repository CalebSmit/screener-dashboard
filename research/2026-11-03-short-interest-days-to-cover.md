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

**Proposal status: DECIDED 2026-11-05 (06:00 session, which did not write this note) - REFUSED as
written. The first arm cannot measure anything: the research engine has never scored this metric,
so "drop it" is the baseline.** The research the note is built on stands, and so does its verdict
that the weight stays; what is refused is the registration. Section 6 is the review.

## 6. Review of the proposal (2026-11-05, 06:00 session)

Measured: `research/measurements/2026-11-05-short-interest-registration.py`, output committed
beside it. Every figure below is from the 176 monthly records on file (2012-01 .. 2026-08).

### 6.1 The first arm is the baseline, not an experiment

`v2_experiments.run` starts every registered transform from `base_config()`, which is
`backtest_v2.v2_config(config.yaml)` - the live weighting with every metric in
`UNMEASURABLE_METRICS` set to 0 and any category left with no weight set to 0 itself.
`short_interest_ratio` is in that tuple. So:

- **Revisions is the only category v2 zeroes.** The seven it scores are Valuation 22, Quality 22,
  Growth 13, Momentum 13, Risk 10, Size 5, Investment 5; Revisions is 0, and all eight of its
  metrics are 0 - including the 13.5 within-category points (**1.35 composite points**) the live
  screener gives days to cover.
- **The column is empty in every month on file**: present in 176 of 176 records and **100% NaN in
  every one** (68,197 stock-months). `pit_record` classes `shortRatio` and `shortPercentOfFloat` as
  `no_free_history`.
- So the arm, written the way the registry's own metric-level precedent writes it
  (`drop-revenue-growth`, `kind: config`, `set: {metric_weights.revisions.short_interest_ratio: 0}`),
  produces a configuration **dict-identical to the baseline**, and the ranking to match:
  **0 of 67,113 stock-months with a different composite, 0 with a different rank, maximum absolute
  difference 0.0.** Read as "the category without its one non-analyst leg"
  (`kind: drop_category`, `category: revisions`) it is identical too, on the same three counts.

A row like that would report "no difference, not distinguishable from zero" - which is the note's
own prediction, and which a reader would take as *the leg is not earning its 1.35 points*. It is
instead an identity. **The null would be a property of the missing panel, not of the weighting** -
which is exactly the ground the 2026-10-23 session refused the same metric on from the other
direction, when it was asked to put weight *on* it.

The third way the registry could express it - `kind: metrics` naming every other metric at its live
weight - does change the ranking (60,125 stock-months), but only because that kind rewrites the
metric tables from scratch and so **puts weight back on five all-NaN analyst columns**
(`forward_eps_growth`, `fy1_revision_3m`, `analyst_surprise`, `consecutive_beat_streak`,
`price_target_upside`). It also changes *who is ranked* in 6,943 stock-months. That is not the
proposal.

**The registry already knew this, which is the part worth noticing.** It carries `only-X` and
`drop-X` for each of the seven categories v2 scores and **none for Revisions** - 0 entries name the
category. The fact was in `backtest_v2.py`'s comment too, in the project's own words: the metric
"stays at weight 0 until that panel is built and verified to be scored - a registered test of it
before then would run on an all-NaN column". Three sessions have now had to rediscover it by hand.

### 6.2 The second arm is blocked by the engine's own confirmation rule, not only by the build

The note's second half - FINRA days to cover from 2018 in the Yahoo field's place - is written
"if the v2 record can carry it", so this is the check it asked for. Two things, the second of which
the note did not have:

1. **The panel has to be built first**, and the build order is already written
   (`research/2026-10-22-free-data-that-adds-evidence.md` section 7, licence check first). Until
   then `pit_record.FIELD_CLASSES` classes the field `no_free_history` and the record builder NaNs
   it, so there is nothing to weight.
2. **Even built, the experiment could not be confirmed as the rules require.** FINRA's biweekly
   exchange-listed files begin **2017-12-29**
   (`research/measurements/2026-10-23-finra-short-interest-probe.py`). Against the engine's months
   that is **80 of 80** in 2020-2026 and **25 of 96** in the presample - nothing before 2017-12.
   The registry's confirmation rule is that an experiment is CONFIRMED only if its **2012-2019**
   difference clears the bar *and* points the same way as its 2020-2026 difference. A presample arm
   covering a quarter of its own period, all of it after 2017, cannot satisfy that: the 71
   uncovered months would be scored on an empty column while the baseline scores them on a full
   one, so the difference measured there would be the coverage, not the metric. The honest version
   is a test judged on 2018-2026 alone, with its presample reading declared unavailable - a change
   to how an experiment is judged, which is a `METHOD_VERSION` question and reruns everything.

So: real, buildable, and not registrable today. The note's prediction stays on file for whoever
builds it.

### 6.3 What changed, and what did not

**No weight, metric, threshold, published sentence or production line moved.** The note's research
is not in question - section 1's reading of Hong et al. and Asquith et al. was not re-opened, and
its verdict (the weight stays; the placement inside Revisions is a synthesis question) is
untouched.

What shipped is the guard, because a refusal derived by hand three times is a rule nothing enforces:
`tests/test_registry_can_measure_what_it_registers.py` (11 tests) fails the suite if any registered
transform leaves v2's configuration unchanged, puts weight on a column the record leaves NaN, or
leaves nothing measurable carrying weight - with the three refused entries as fixtures, so a guard
that could pass on what it was built to catch is impossible.

**Measured: before today nothing in the suite checked this.** With the entry appended to the
registry and the new module excluded, the full suite reports **2 failed, 3,906 passed, 1 skipped**
- and neither failure is a check of the arm. Both read `summary.json`'s **count** of registered
alternatives (`tests/test_research_page.py::test_the_committed_page_is_what_the_generator_makes_from_the_ledger`,
a drift tripwire, and `tests/test_v2_experiments.py::test_the_methodology_page_quotes_the_engine_s_own_summary`),
so both go green the moment the registering session runs the engine and regenerates the page -
which the rules require it to do. The same distinction the 2026-10-31 pass drew: a mutation can
read as caught and still ship.

A runtime refusal in `v2_experiments.run` would be better than a test, and is deliberately not
built today: `v2_experiments.py` is in `RESULT_PATHS`, so editing it stales all 32 ledger rows and
forces a ~20-minute rerun whose numbers provably cannot change. Queued in
`plan/research-engine.md` to fold in the next time something forces that rerun anyway.
