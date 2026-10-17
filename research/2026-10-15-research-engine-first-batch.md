# The research engine's first two batches: 27 pre-registered ideas, 176 months, nothing beats the current weighting - and nothing could be shown to

**2026-10-15, owner-run (evening). Backtest observation throughout - not decision-grade** (CLAUDE.md
rules 4 and 5: nothing here moves a weight before 2027-02-11, and a backtest number is never the
evidence for a methodology change). Registry: `research/experiments/registry.yaml`. Every run:
`research/experiments/ledger.jsonl`. Generated table: `research/experiments/RESULTS.md`. Analysis:
`research/measurements/2026-10-15-research-engine-analysis.py`, `...-size-survivorship.py`.

## 1. What was asked, and what was built

The owner asked for the methodology to be tested the way automated research loops are being built -
many ideas from the literature run against each other - so that the result can be shown to other
people and trusted, first for a trading bot and eventually as the method behind a fund.

The danger of such a loop is documented and large: test enough ideas on one history and some look
excellent by chance (Harvey, Liu & Zhu 2016 put the bar for a new factor at t > 3; McLean & Pontiff
2016 find published anomalies' returns 26% lower out of sample and 58% lower after publication). The Factor Lab has measured the same thing on its own reruns: 9 of
the 10 directional results it reproduces were weaker after their papers' samples. So the engine
(`v2_experiments.py`) is built around rules that make a result from it worth showing, not around a
search for the best number:

1. **Pre-registration.** Each experiment's hypothesis, source and predicted outcome is committed
   before it runs. The commits are the proof: batch 1 at **17:13:30** (`ab983cc`), the 2012-2019
   test at **18:03:17** (`fb6f7c2`), batch 2 at **18:22:56** (`b64db0e`).
2. **Held-out months.** Discovery 2020-01..2023-06, held out 2023-07..2026-08, fixed in code.
3. **Every test counts.** The ledger keeps every run, including the 25 superseded by a data fix; the
   significance bar is p < 0.05 / (experiments registered) = **0.0019** for 27, and |t| >= 3.
4. **Paired with the baseline**, month by month, so the comparison is on the
   difference, not on two noisy levels. *(Corrected 2026-10-16: this said "on the same stocks". The
   pairing is by month only. Each configuration ranks the stocks it has enough weighted data to
   score, so a narrow screen's IC is computed on a smaller set than the baseline's;
   `research/measurements/2026-10-16-stocks-ranked-per-config.json` counts them. Pairing on each
   month's common stocks, with a full rerun, is queued in `plan/research-engine.md` item 8a.)*
   *(Superseded 2026-10-17: item 8a is built. Since **method version 2** both sides are recomputed
   on each month's intersection before the difference is taken, every ledger row carries
   `method_version` and `n_common_by_month`, and every experiment and evaluation in this note was
   rerun on it. **Every figure in this note below is a method-version-1 reading** and the engine's
   report now refuses to mix the two; `research/experiments/RESULTS.md` carries the version-2
   figures. What changed and what did not is in section 11.)*

> **Disclosure, 2026-10-16 (audit).** Seven of the batch-1 entries, each category on its own
> (`only-valuation` .. `only-investment`), had their 2020-2026 outcome computed and committed before
> they were registered: backtest v2's returns measurements publish each category score's IC
> (`research/measurements/2026-10-15-backtest-v2-returns.json`, commit `b6f7fd9` at 11:31, and
> `-rebuild2.json`, `710f610` at 16:55), and the registry's first commit is `ab983cc` at 17:13. The
> ledger's first runs of those seven match the rebuild2 figures to about four decimals. For them only
> the 2012-2019 test is blind. The registry now says so (`outcome_known_at_registration`), the
> report and the public page mark them, and `v2_experiments.run` refuses an undisclosed case.

**The data.** Backtest v2 (`backtest_v2.py`): each month's S&P 500 members as they stood (Wikipedia
revision history), each company's statements as filed with the SEC *by that month-end*
(`pit_record.py`), prices up to that day, scored by the engine's own `compute_metrics` - the
screener as it would have spoken then, without the look-ahead that makes v1 unusable (72.5% of v1's
name-months sit in a different decile from what the screener would have said at the time;
`research/2026-10-15-pit-record-validation.md`). The ranking is the **filings-and-prices screener**:
Revisions and forward EPS growth have no free history and carry no weight here.

## 2. Three defects the engine found before it found anything else

- **A configuration switch that does nothing to the ranking.** `sector_relative_composite` was
  registered as an experiment ("normalise the composite within each sector", as `config.yaml`
  described it). It produced *identical* rankings in all 80 months. Since Phase 13 it changes only
  the display column `Composite_Pct`; the ranking key is the cardinal composite. The experiment is
  marked **void** in the registry (its prediction untouched), and the config comment is corrected.
  A switch whose description is false is an explainability defect even when it is off.
- **Renamed companies counted as missing.** v2's inputs are keyed by today's symbols, so a member
  listed under an old one - FB (META), ANTM (ELV), BK (BNY), ABC (COR), BLL (BALL) - was counted as
  having left the index. The SEC registrant number (CIK) survives a rename: **635 member-months**
  recovered in 2020-2026, **2,332** in 2012-2019. Meta's 2022 fall is now in the record.
- **A reused symbol scored as the wrong company.** A symbol can have belonged to another company:
  IR was Ingersoll-Rand plc until 2020 (today TT), JCI's filings today are Tyco's, CB's are ACE's.
  Each month's revision lists the registrant (CIK) from mid-2014 on, and a mismatch with today's is
  resolved three ways: the then-company is found under its current symbol (IR -> TT, 70 months; Q ->
  IQV); or, where today's filer under the symbol was *itself another member that month* (Tyco as
  TYC, ACE as ACE), the symbol is counted and never scored with that other member's data (JCI 28
  months, CB 20); otherwise the symbol stands and its own filings decide - a holding-company
  reorganisation (Exxon, Disney, Cigna, BlackRock) has no filings before its new registrant existed,
  and Wikipedia's CIK column was simply stale for Fox, Xerox and others in early 2020. *(A first,
  stricter version excluded every mismatch, and the first 2012-2019 results were read on it. It was
  replaced the same night - the stale-CIK months excluded companies whose filings were valid - and
  every experiment rerun on the corrected records; both sets of runs are in the ledger, and the
  figures below are from the rerun.)*

And one in the tests themselves: two `build_month` tests wrote a synthetic one-stock "month"
(2024-06-28) into the real metrics cache. It never reached a result - a month under 50 stocks is
skipped - but the engine now reads month-ends only and the tests write to a temporary directory.

## 3. Results on 2020-2026 (80 months, corrected records)

The baseline's IC (rank correlation of the composite with next month's return) is **-0.00003**
(+/- 0.027, t 0.00) - the same as before the rename and reuse fixes (-0.001), so that reading is robust. Of 26
alternatives, **none differs from the baseline** in both discovery and held-out months; discovery
and held-out differences agree in sign for **12 of 26** (chance: 13); one held-out comparison has p
< 0.05 against 1.3 expected by chance (orthogonalised categories, t 2.39 held out, 0.74 in
discovery - "different only in the held-out months - not confirmed").

The best held-out screens were the momentum ones (12-1 return alone +0.026; value-and-momentum
+0.028), each about +0.02 above the baseline with t under 1.6, and each *below* the baseline in
discovery. The low-risk screens were the worst in both halves (held-out ICs -0.017 to -0.021, 0.021 to
0.025 below the baseline), never significant.

## 4. The untouched test: 2012-2019 (96 months)

Registered at 18:03 - after the 2020-2026 results had been read and before a single 2012-2019
month existed (the membership was fetched and the records built afterwards, and no return was
computed on them until every experiment was registered). Testing on the history *before* a sample
is the design of Linnainmaa & Roberts (2018), who found most anomalies weaker or absent there. The
rule, fixed in advance: an experiment is confirmed only if its 2012-2019 difference clears p <
0.05/27 and |t| >= 3, in the same direction as 2020-2026.

| | Pre-registered prediction | Result |
|---|---|---|
| Baseline IC, 2012-2019 | positive, below 0.03, not distinguishable from zero | **-0.003 (t -0.28)** - not distinguishable; the sign was wrong |
| Experiments confirmed | none | **none** |
| Momentum screens, best held out in 2020-2026 | not better in 2012-2019 | the two momentum-only screens +0.004 and +0.007 above the baseline (t <= 0.50); value-and-momentum -0.003 - not better |
| p < 0.05 by chance | ~1.3 expected | **1** (dropping Size, t -2.71) |

Sign agreement between the two eras' differences is 17 of 26, against 13 by chance - but the
experiments overlap heavily (dropping a category and keeping only it are two views of one thing), so
the 26 are far from independent and this is not evidence of a consistent effect.

**The one number that looks like something is survivorship.** Size alone (favouring the smaller
members) has the highest 2012-2019 IC: +0.026, t 3.04 by itself, +0.028 over the baseline (t 1.95,
not significant - the rule judges the difference from the baseline, not the level). v2 scores only the members still in the index today; in 2012 that is 58% of them.
The small 2012 members that are still there are disproportionately the ones that did well - a
small company that shrank fell out of the index and out of the data. If that is what Size is
picking up, its IC should be largest where coverage is thinnest. Stated before measuring, then
measured (`2026-10-15-size-survivorship.py`): Size's IC is **+0.033** in the lowest-coverage third
of months (61% present), **+0.019** in the middle, **-0.004** in the highest (90%); correlation with
coverage -0.17 (t -2.27). The baseline's IC shows no such relation (-0.02, t -0.26). Coverage also
rises with time, so the two are entangled, but the direction is the one survivorship predicts and
the Size number should be read as an artefact until a survivorship-free source says otherwise.

> **Correction, 2026-10-16 (audit).** The paragraph above claims more than this data can show.
> Coverage correlates **0.998** with the month index (57% of members in 2012, rising steadily to 97%
> in 2026), so "the lowest-coverage third of months" is the earliest third, and the across-era
> relation cannot tell survivorship from the passage of time. The part of the data that could
> separate them says nothing either way: within 2012-2019 Size's monthly IC correlates with coverage
> **-0.126** and the baseline's **-0.122**, and the Size-minus-baseline difference against coverage
> is **r +0.012** within 2012-2019 and **r -0.08 (t -1.06)** over all 176 months. Survivorship
> remains a plausible hypothesis for Size's lead, stated as such, and the test that could settle it
> is a rebuild that adds the departed members from free filings and prices, queued with its
> prediction in `plan/research-engine.md` (item 8d): if survivorship drives the lead, Size's
> 2012-2019 difference from the baseline falls from +0.028 to below +0.014, and the hypothesis is
> falsified if it stays at +0.020 or more. The public page now states the hypothesis and generates
> its sentence from `research_page.size_coverage_relation`
> (`research/measurements/2026-10-16-coverage-by-month.json` with the ledger's `ic_by_month`).

## 5. What the design can and cannot see

A pre-registered, paired, held-out test is only worth its power. The paired standard error of an IC
difference is about **0.017** over the 38 held-out months and **0.010** over 2012-2019's 96. At the
engine's bar (z 3.1, 80% power) the smallest difference it can reliably detect is **0.066** IC held
out and **0.039** in 2012-2019. A practitioner rule of thumb is that an IC of 0.05 is good and 0.10
excellent (Grinold & Kahn 2000), and factor premia among large companies are about half their
all-stock size (the Lab's experiments 8 and 9) - so a real improvement could easily sit below what
this can see. So
"not distinguishable" here is not "no difference"; it is "no difference large enough to see".

> **Correction, 2026-10-16 (audit).** The bar above came from the normal curve. On 38 held-out
> months the exact two-sided critical value for p < 0.05/27 is **3.35** (Student's t, 37 degrees of
> freedom), and **3.20** on 96 months; the engine now uses the t-distribution throughout
> (`v2_experiments.t_critical`). The smallest reliably detectable difference becomes **0.070** held
> out and **0.040** in 2012-2019. No verdict changed: the only comparison near p = 0.05, Size alone
> in 2012-2019 (t 1.95), moves from p 0.051 to 0.054.

**How long a record would it take?** The baseline's IC moves about **0.12** from month to month
(0.123 in 2020-2026, 0.100 in 2012-2019). With 80% power and the 2020-2026 spread, a true monthly IC
of **0.05** - "good" by Grinold & Kahn's rule of thumb - needs about **47 months** to show at the
conventional 5% level and **89 months (7½ years)** at t >= 3; an IC of **0.03** needs **132 months
(11 years)** and **248 (about 20 years)**. (These are the figures `research.html` computes from the
ledger; an earlier draft rounded the spread to 0.12 and quoted 45, 85, 125 and 236.) That arithmetic applies to the live track record too: no monthly large-company screen,
this one included, can prove a realistic edge from its own record in less than several years. Its
credibility in the meantime has to rest on the published long-run evidence for each factor (the
Lab), on process (this engine), and on stating the uncertainty - which is also what institutional
investors are trained to ask for.

**The harness can see signal when there is signal.** The same pipeline fed *today's* filings for
every month - v1's look-ahead, rebuilt on purpose as a diagnostic - reports IC **+0.050 (t 4.71)**
and a top-tenth excess of 0.85% a month (t 5.0). That is the positive control: information the
screener would not have had makes the test light up; what the screener actually had at the time
does not.

## 6. What it means for the screener

**No weight changes, and none proposed.** The rules forbid it (4 and 5), and the evidence would not
support it if they did not: nothing passed, and the one standout is explained by the data's gap.

What the batches do say, as research questions rather than findings:

1. **Over 176 months of large-company history, the filings-and-prices ranking did not rank next
   month's returns.** That is consistent with the Lab (value, profitability, investment and size all
   weaker after their samples and about half-size among large companies) and with experiment 11
   (cash-flow yield by market value: about zero since 1990). It is the honest headline for anyone
   asked to trust the screener as a return predictor: **on public data it has not been shown to be
   one.** Its defensible claim today is decision support - every number checkable, every factor
   sourced - not a forecast.
2. **The categories v2 cannot test are the ones left to test.** Revisions and forward EPS growth are
   absent here (about a sixth of the published composite: Revisions 10%, forward EPS growth 5.9%).
   The revision effect is documented (Chan, Jegadeesh & Lakonishok 1996), and the only route to
   measuring it here is the live forward record, which accrues about one independent `1m` observation a
   month.
3. **Low risk was the most consistent loser** (worse than the baseline in both eras, every risk
   screen, never significant) in a period that was mostly a rising market - the regime in which
   long-only low-volatility portfolios are documented to lag (Frazzini & Pedersen 2014's premium is
   a beta-*neutral* construction; a long-only low-risk screen carries the market-beta bet). **The
   page already claims the defensible thing** - that higher volatility "has not been paid for", not
   that low volatility earns more - and the ledger shows Risk doing that job: the top tenth's monthly
   volatility is **3.31% with Risk and 3.50% without it** (2012-2019) and **4.79% against 5.19%**
   (2020-2026), while its return cost is small, never significant, and of opposite sign in the two
   eras (+0.05 and -0.22 points a month). A smoother ride at no measurable cost is what the page
   says Risk is for. No change.
4. **Survivorship is now the binding limit**, not look-ahead. At 58% coverage in 2012 the period
   before the sample is less reliable than the one it tests. A paid delisted-price feed would close
   it; the owner decided on 2026-10-15 to use no paid data, so the gap is stated beside every result
   it touches and free ways to narrow it are the queue's job.

## 7. What the owner can show people

Not "this screener beats the market" - the evidence does not say that, and a claim the data does not
support is what destroys a fund's credibility. What can be shown, and checked:

- a **pre-registered research log**: 27 hypotheses from the literature, each with its prediction,
  committed before it ran, with every run (including the ones a data fix superseded) in the ledger;
- **a test it can fail**, demonstrated by the look-ahead control (t 4.7 with leaked data, ~0 without);
- **three defects found and fixed** because the tests were run honestly, one of them a false
  description of a live configuration switch;
- the **power** of each test stated next to its verdict, so "nothing found" is read correctly.

That is what institutional research process looks like (pre-registration and multiple-testing
adjustment are what Harvey, Liu & Zhu ask of the field), and it is the part of a fund's methodology
that can be built honestly on free data.

## 8. Next

- **Nightly sessions keep the engine honest, not busy:** a new experiment needs a registry entry with
  a source and a prediction, committed before its run; a data fix means rerunning *all* registered
  experiments (the report flags mixed data versions). Do not register variations of a near-miss
  (size, momentum) - that is the forking path the rules exist to prevent.
- **Monday candidate:** whether the momentum screens' 2020-2026 lead and 2012-2019 flatness match
  the published large-company momentum record (the Lab, experiment 8). Risk's role is answered (point 3).
- **Data:** the survivorship decision above. (The 2020-2026 CIKs were fetched and both eras rebuilt
  the same night - section 2.)

## 9. The trading bot's own rule, on the same history (evaluation, registered before it ran)

Added the same night (`plan/research-engine.md` item 4). The question is the owner's: what should
he expect from the bot? The bot's rule as coded in `screener-bot` (`bot/strategy.py`, v1.1 - hold
25, sell past rank 40, skip trap flags, at most 6 names a sector, 2% cash, no new buys 25% below
peak, 5 bps a side) was registered with a prediction at **19:25:41** (`abef7ec`) and then run
monthly on the baseline's rankings (`python v2_experiments.py evaluate`). Differences from the live
bot are listed with the result in `RESULTS.md` - chiefly monthly instead of weekly, and the
filings-and-prices ranking without the analyst inputs.

| Period | Return a month | Excess over the equal-weighted universe (t) | vs S&P 500 total return (t) | Turnover a year | Max drawdown (universe) |
|---|---|---|---|---|---|
| 2020-01..2023-06 | +0.51% | **-0.73% (-2.72)** | -0.64% (-1.83) | 174% | -20.3% (-19.6%) |
| 2023-07..2026-08 | +0.91% | -0.22% (-0.55) | -0.62% (-1.23) | 134% | -7.6% (-8.4%) |
| 2012-2019 | +1.18% | -0.14% (-1.02) | +0.02% (0.12) | 142% | -14.6% (-12.2%) |

**Against the prediction:** wrong on the first period - the shortfall there is outside the +-0.3% a
month predicted and past |t| 2 - and on turnover (174% against a predicted ceiling of 150%); right
on the index comparison, the drawdowns (within 2.5 points) and the halt (one month, 2020).

**Is it the ranking or the rule?** Checked afterwards, as a diagnostic that changes no verdict
(`research/measurements/2026-10-15-bot-rule-checks.py`): a plain equal-weighted top 25, fully
rebalanced each month, trailed the universe by -0.49% a month (t -1.72) in 2020-2023, -0.12% held
out, +0.01% in 2012-2019. So most of the shortfall is the ranking's top - the same weak top tenth as
section 3 - in the growth-led years (calendar gaps of -15.7 points in 2020, -10.1 in 2021, -12.7 in
2016, -11.4 in 2019). The rule's mechanics cost a further 0.10-0.23 points a month in each period;
that comparison was not registered, so it is a question, not a finding. Selling a name that went
missing from a ranking (the one simplification that could have caused it) happened 3 times in 176
months.

**What it means.** On the evidence available tonight the bot, as built, should be expected to
roughly match or lag an equal-weighted S&P 500 fund, not beat it - and in a growth-led year it can
lag by 10-15 points. That is not a reason to change the bot (its own rules forbid tuning to a
backtest: strategy changes come from research, are logged in its `STRATEGY_LOG.md` first, one at a
time), and the live ranking carries the analyst inputs v2 cannot test. It is the expectation to set
before anyone is shown the bot's paper record, and the reason that record should always be shown
next to RSP.

## 10. Are the months independent? (queue item 3, same night)

Every t above assumes each month is a fresh observation. The engine now records, beside each plain t,
the lag-1 autocorrelation of the series it tests and a Newey-West t (3 lags, Bartlett kernel), and
keeps each run's monthly series in the ledger so the check never needs a rerun. Over the 81 series
compared: median lag-1 autocorrelation **+0.002** (range -0.28 to +0.36). Where it is material it is
mostly *negative*, which makes the honest t slightly *larger*: dropping Size, held out, goes from t
1.30 to 2.11 (its discovery difference has the other sign, so it is still "not confirmed"). **No
Newey-West t reaches 3**, so no verdict changes. Figures in sections 3, 4 and 9 are from the rerun on
the corrected records (data version `d8cb038fff2b`); the earlier runs stay in the ledger.

