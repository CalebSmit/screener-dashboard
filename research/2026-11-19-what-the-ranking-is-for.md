# What is the ranking for, given no measured edge? - the question and its pre-registration

2026-11-19 (owner-run, night). **CLAUDE.md priority 0.15**, opened the same evening on the owner's
delegation. This note does **not** answer the question. It writes down, before anything is
measured, what the candidate answers are and what result would count for or against each - so the
Monday session that measures cannot choose its reading after seeing the numbers.

**Proposal status: DECIDED 2026-11-21** - measured against section 3 unchanged (method fixed in
section 4a and committed before the run): the positive control is seen; A fails, B is undecidable at
6 and 12 months, C does not run, so **the answer is D - a teaching tool with no return claim**.
Section 5 lists the two page changes that follow. (Originally due the Monday of 2026-11-23.)

## 1. The fact that makes this the question

The research engine rebuilt 176 months of 2012-2026 from the filings as they stood each month and
asked whether the screener's ranking orders the **next month's** returns. Pooled monthly rank IC
-0.001 (95% interval -0.018 to +0.015 before tonight's rerun; -0.00133 after); none of 27
pre-registered alternative weightings did better; the harness fed leaked data finds t 4.7, so it
can see a signal when there is one (`research/2026-10-15-research-engine-first-batch.md`,
`research.html`). The bot's own rule lagged an equal-weighted S&P 500 in every period tested.

Two things that result does **not** say, and which this note exists to separate:

- It tests **one horizon** (one month) and **one use** (ordering all ~500 names). The screener's
  heaviest categories - Valuation and Quality, 63% of what moves the composite (finding 7 of the
  review) - are the ones the literature finds slow: value's premium is earned over years, not
  weeks (Asness, Moskowitz & Pedersen 2013, *Value and Momentum Everywhere*), and gross
  profitability predicts returns at horizons well beyond a month (Novy-Marx 2013, *The Other Side
  of Value*). A one-month rank IC is the test most likely to miss a slow signal.
- It is measured without the analyst inputs (Revisions, 10%) and without the companies that left
  the index (survivorship) - both stated on `research.html`.

And one thing it does say: in practice an IC of 0.05 is considered good and 0.02-0.03 ordinary for
a monthly cross-sectional forecast (Grinold & Kahn, *Active Portfolio Management*, 2nd ed. 2000,
on the "fundamental law"); an interval spanning -0.018 to +0.015 rules out the good case and
cannot rule out the ordinary one. "No measured edge" is not "measured no edge".

## 2. The candidate answers

| | What the ranking is for | What the page would then say | What it is judged on |
|---|---|---|---|
| A | **A screen**: narrow 500 to ~40 worth a student's reading time | "start your research here", never "these will do better" | Whether the top 40 avoid the worst outcomes (not whether they beat the rest on average) |
| B | **A slow forecast**: the composite orders returns over 6-12 months, not 1 | a horizon on every claim of performance; the review cadence becomes the horizon | Rank IC at 3, 6 and 12 months, non-overlapping |
| C | **Fewer, better factors**: some categories carry the signal and the rest add noise | a smaller, more teachable composite | Category-by-category IC at the horizon B finds, against the composite |
| D | **A teaching tool only**: no return claim at all | the track-record tile becomes a lesson in why short records mean nothing | Nothing to measure: the default if A-C all fail |

These are not exclusive; A and B can both hold.

## 3. Pre-registered criteria (fixed 2026-11-19, before any measurement)

All on the research engine's existing monthly records (`backtest_v2`, 2012-2026, both eras
reported separately and pooled), with its existing look-ahead positive control run alongside -
**a result only counts if the positive control clears its own bar in the same run.**

- **A (screen) holds** if, over non-overlapping months, the top 40 by composite fall in the
  **bottom decile** of next-month return less often than 10% of the time with a one-sided 95% interval
  whose upper end is below 10% - in **both** eras. (A screen that steers readers away from the worst
  names is useful even with no average edge.) Fails otherwise.
- **B (slow forecast) holds** if the pooled rank IC at **6 or 12 months**, measured on
  non-overlapping windows only, has a 95% interval entirely above zero **and** the same sign in both
  eras. With ~29 non-overlapping 6-month windows and ~14 12-month ones, the engine's own power
  calculation must be stated first; if it cannot detect 0.05 at those counts, B is recorded as
  **undecidable**, not as failed.
- **C (fewer factors) holds** only if B holds, and then if a category's own IC at B's horizon is
  above the composite's with a paired difference whose interval excludes zero in both eras. One
  category at a time, Bonferroni across eight.
- **D** is the answer if A, B and C all fail or are undecidable.

**What may not happen:** no new weighting is fitted to these results (rule 5 holds to 2027-02-11;
the research engine reports, it does not set weights); no horizon is added after the first run;
no criterion is relaxed to let an answer through. A finding that the engine cannot decide is a
finding.

## 4. Measurement (to be completed by the Monday session)

Done by the 2026-11-21 06:00 weekend session (Saturday, two days early; the Monday it was dated for
is then free for the residual-momentum build this note's answer was blocking).

### 4a. Method, committed before the first run

Section 3 fixes the criteria. It leaves a handful of operational choices open; they are fixed here,
in the commit that also adds the script, **before the script has been run once** - so, like the
criteria, they cannot have been chosen after seeing a number. Script:
`research/measurements/2026-11-21-what-the-ranking-is-for.py` (writes the `.json` beside it).

**Where it is registered, and why not in `registry.yaml`.** Section 4 said to register each test in
the engine's registry. Not done, deliberately: `registry.yaml` is one of `v2_experiments.RESULT_PATHS`,
so any edit to it stales all 32 ledger rows and forces a ~25-minute rerun that changes no number; and
its `experiments:` section holds weighting transforms that every loader reads `["transform"]` from
(the 2026-10-23 session refused three registrations for exactly that). These are not weightings and
must not join the multiple-testing denominator. The pre-registration's actual purpose - proof that
nothing was chosen after seeing a result - is served identically by this commit, which carries the
method and the script and precedes any output file. The script writes nothing to the ledger.

**Data and ranking.** The research engine's own monthly records (`cache/v2_metrics/`, 176 month-ends
2012-01 .. 2026-08), ranked with the baseline configuration by `v2_experiments.rank_all` - the same
composite every engine row is paired against. Returns from `backtest_v2_returns.month_end_prices`
(adjusted close on the last trading day on or before each month-end; the engine's own function,
including its forward-fill of a price that stopped). Its caveats carry over unchanged: survivors
only, no analyst inputs (Revisions and forward EPS growth carry no weight), not decision-grade before
2027-02-11.

**Eras.** 2012-01 .. 2019-12 (the engine's pre-sample, 96 months) and 2020-01 .. 2026-08 (its
sample, 80 months). "Both eras" means these two; "pooled" means both together.

**Positive control.** The registered `lookahead-filings` control, run in the same process through the
engine's own `per_month` / `summarise` / `control_verdict` with N = the registry's experiment count.
It exists for 2020-2026 only (as registered). **If it is not seen, nothing below counts.**

**A (screen).** Each month: the stocks ranked that month with a next-month return; deciles of that
return by `rank(pct=True, method="first")`, as the engine cuts deciles; the 40 with the highest
composite (ties by ticker); the share of those 40 in the bottom decile. One observation per month
(one-month windows do not overlap). Per era: the mean share and a one-sided 95% upper bound,
mean + t(0.95, n-1) x SE, where **SE is the larger of the plain and the Newey-West (3-lag) standard
error** - the 40 names in one month are not independent of each other, which is why the month is the
unit, and the larger SE guards against month-to-month dependence too. Holds if the upper bound is
below 0.10 in **both** eras.

**B (slow forecast).** For h = 6 and 12: within each era, formation months are the era's first month
and every h-th month after it, kept while a price exists h months later (prices end 2026-10-15, so
the last usable month-end is 2026-09-30); a window's IC is the Spearman correlation of the composite
with the h-month return over the stocks ranked that month with both prices. Pooled 95% interval on
the t distribution from the windows' plain SE (the windows do not overlap). **Power first:** the
smallest IC detected 80% of the time at two-sided 5%, (t(0.975, n-1) + t(0.80, n-1)) x SE, from the
windows' own dispersion. Holds at a horizon if the pooled interval is entirely above zero and both
eras' means have the same sign. A horizon that does not hold is **undecidable** if its detectable IC
exceeds 0.05, **fails** otherwise. B holds if either horizon holds; no correction across the two
(section 3 did not set one, and adding one would change a criterion).

**C (fewer factors).** Only if B holds, at the horizon(s) where it holds: per window, each category
score's IC minus the composite's, on the stocks with that category score; per era a two-sided
interval at 0.05/8; holds for a category if the interval lies entirely above zero in both eras.

**D.** The answer if A, B and C all fail or are undecidable.

### 4b. Results (first and only run: commit `129ee13`, inputs clean, 176 months ranked)

Output: `research/measurements/2026-11-21-what-the-ranking-is-for.json`. A backtest observation
(rule 5); read only against section 3.

**The positive control is seen**, so the results below count: `lookahead-filings` +0.0516 IC over
the baseline, t 5.64 over 80 months, against N = 29 and inside its registered band (+0.030 to
+0.075). The design can see a signal of that size at one month.

**A (screen): FAILS.** The top 40 landed in the next month's bottom decile:

| Era | Months | Mean share | SE used | One-sided 95% upper | Holds (< 10%) | For comparison: share in the top decile |
|---|---|---|---|---|---|---|
| 2012-2019 | 96 | 9.79% | 0.49% (NW) | 10.61% | no | 9.38% |
| 2020-2026 | 80 | 9.78% | 0.56% (plain) | 10.71% | no | 9.63% |

Every month had 40 names (at least 278 ranked a month). The interval is tight - this is not a test
that ran out of data - and the answer is that the top 40 are **as likely as any stock to have one of
the worst months, and no more likely to have one of the best**. As measured, the ranking does not
steer a reader away from next month's worst outcomes.

**B (slow forecast): UNDECIDABLE at both horizons.** Power first, as section 3 requires:

| Horizon | Windows | Smallest IC detected 80% of the time | Pooled mean (95% interval) | 2012-2019 | 2020-2026 | Verdict |
|---|---|---|---|---|---|---|
| 6 months | 29 | **0.057** | -0.007 (-0.047 to +0.033) | -0.018 (16) | +0.006 (13) | undecidable |
| 12 months | 14 | **0.065** | +0.004 (-0.042 to +0.051) | -0.021 (8) | +0.038 (6) | undecidable |

Neither horizon holds (no interval above zero; the eras disagree in sign at both), and at neither can
fourteen years of non-overlapping windows see an IC of 0.05, so both are recorded as undecidable, not
failed. The 2020-2026 twelve-month figure (+0.038, t 2.08 on 6 windows) is the kind of number that
tempts a reading; section 3 does not allow one, and the earlier era points the other way.

**C: not run** (it is tested only if B holds).

**So the pre-registered answer is D: a teaching tool, with no return claim.**

### 4c. What the result does not say

- **Survivorship bites A hardest.** The engine has only stocks still in the index, so the worst
  outcomes of all - bankruptcies, delistings, distressed takeovers - are missing from every month's
  bottom decile. A screen that kept readers out of *those* would not show it here. This is a limit of
  the test, written down now so nobody later reads it as a reason to reopen A: section 3 fixed the
  test, and the test failed.
- **It is the filings-and-prices ranking**, without Revisions (10% of the published weight) or
  forward EPS growth - the same caveat every engine row carries.
- **B undecidable is not "no slow edge".** It means 2012-2026 cannot tell; the data that could
  (decades more of point-in-time filings) is not free.
- **Nothing is fitted to it** (section 3): no weight moves, no horizon is added.

## 5. What it changes on the page (completed 2026-11-21)

Section 2 wrote D's page consequence before any measurement: *no return claim at all; the
track-record tile becomes a lesson in why short records mean nothing.* Against the live page:

- **Most of D is already true.** The page's own description calls it "a research and teaching tool,
  not investment advice"; the hero has said since 2026-11-19 that the 15-year test found no measurable
  edge at one month (`page.hero_tested`); and the Track record tile has printed "Too short to tell skill
  from luck: N standard errors" since 2026-10-27. Nothing on the page claims an edge.
- **What D adds, and is to be built:** (1) the hero's tested sentence should also state today's two
  results in plain words - the top 40 were as likely as any stock to have one of the worst months,
  and at six and twelve months the history is too short to tell - from this file, as a registered
  claim, so the sentence cannot outlive the measurement; (2) the Track record tile leads with the
  lesson rather than the return: the return stays (hiding a number would be worse), but under a
  heading that says what it is - a short record that cannot separate skill from luck.
  **Both shipped 2026-11-21:** (1) by the 06:00 weekend session (claim `page.hero_screen`); (2) by
  the 13:00 session - while the lead is under two standard errors the tile's heading reads "Track
  record: too short to judge" and "It cannot yet tell skill from luck: N standard errors, where two is
  the usual bar" sits above the return (claim `context.track_luck`; the order is pinned by
  `tests/test_page_fourth_pass.py`). Above two, the tile reverts to "Track record" with no caveat.
- **What D does not change:** the ranking, its weights and the screen itself stay - a structured,
  checkable way to read 500 companies through published factors is what a teaching tool is. Residual
  momentum (ACCEPTED the same morning) is justified on construction and explainability, not on a
  return, so it is unaffected. **Universe expansion phase 2** was paused for this answer
  (`OWNER_FOCUS.md` 2026-10-28): D does not forbid it - more companies to study is a teaching gain -
  but phase 3's publication must carry the same no-return-claim framing, and the S&P 1500's ranking
  inherits this result rather than being presumed better. The trading bot trades this ranking: that is
  the bot repo's question (its `STRATEGY_LOG.md`), not this one's, and nothing here touches it.

## 5. What it would change on the page (to be completed after section 4)

*Empty on purpose.* Whatever the answer, the home page already states the one-month result
(`page.hero_tested`, 2026-11-19). If A holds, the hero's framing moves from "ranked" toward
"screened". If B holds, every performance figure carries its horizon. If D, the track-record tile is
reframed as a lesson.

## Sources

- Asness, C., Moskowitz, T. & Pedersen, L. (2013). Value and Momentum Everywhere. *Journal of
  Finance* 68(3).
- Novy-Marx, R. (2013). The Other Side of Value: The Gross Profitability Premium. *Journal of
  Financial Economics* 108(1).
- Grinold, R. & Kahn, R. (2000). *Active Portfolio Management*, 2nd ed. McGraw-Hill.
- The project's own: `research/2026-10-15-research-engine-first-batch.md`, `research.html`,
  `research/2026-11-19-fresh-eyes-review.md`.

*Not re-read tonight:* the three published works are cited for their well-known headline findings
only; the Monday session should check each claim against the paper before relying on it, per
`prompts/nightly.md`'s reading rule.
