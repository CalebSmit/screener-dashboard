# The positive control: can the research engine see a signal at all? (2026-10-25)

`plan/research-engine.md` item 4(c); `OWNER_FOCUS.md` 2026-10-20 item 4. Session: 06:00 weekend.

## 1. The question, and why a null result cannot answer it

The research engine has run 28 pre-registered weightings against the current one on backtest v2's
point-in-time history. **Not one of them is distinguishable from the baseline**, and the baseline
itself ranks next month's returns at an IC of about zero in both eras
(`research/2026-10-15-research-engine-first-batch.md`). The public page says so.

A sceptical reader should ask the obvious question: *is that a fact about the weightings, or a fact
about the test?* A row of flat results is exactly what a broken measurement produces. And a null
result cannot settle it, because the thing that would settle it - a case where the right answer is
known in advance - is by definition not in the set of things being tested.

This is the standard argument for a **positive control**, and it is sharpened here by the engine's
own multiple-testing bar. Harvey, Liu & Zhu (2016), "...and the Cross-Section of Expected Returns",
*Review of Financial Studies* 29(1), 5-68, DOI [10.1093/rfs/hhv059](https://doi.org/10.1093/rfs/hhv059),
argue that with hundreds of published factors a new one needs a t-statistic nearer 3 than 2; this
engine adopts that bar and a Bonferroni correction on top of it. Raising the threshold for a finding
necessarily lowers the power to detect one, so **the power has to be demonstrated separately** or
every "not distinguishable" verdict is uninterpretable.

Two halves were specified. The synthetic half shipped 2026-10-20 (`tests/test_engine_calibration.py`):
differences of 0.03, 0.05 and 0.10 are planted in a copy of the months and come back within four
standard errors in every period. That tests the arithmetic. **This note is the other half:** a real
arm, built by the real pipeline, carried in the ledger and rerun with everything else.

## 2. The arm: v1's look-ahead, built on purpose

The control holds the weighting fixed and changes the **data**. Every month is scored on the filings
as they stand at the build date rather than on the filings that had actually been published by that
month-end - which is what `backtest.py` (v1) does by construction, and which is why CLAUDE.md rule 5
benches v1 until 2027-02-11.

The size of what this reintroduces is measured, not assumed:

| Half of the look-ahead | Size, on the name-month panel | Source |
|---|---|---|
| Prices | >= 63.2% of name-months move; 30.3% by two deciles or more | `research/2026-10-01-lookahead-bias-size.md` |
| Fundamentals | 72.5% of name-months | `research/2026-10-15-pit-record-validation.md` s.3 |
| Together | 75.8% | same |

The arm keeps the point-in-time **universe** and the point-in-time **prices**: the only difference
from the baseline is which filings were visible. That matters for reading the result - the difference
is attributable to the filings and to nothing else, and the 2026-10-15 check confirmed the mechanism
the right way round, with momentum and risk (which read only prices) agreeing to the third decimal
between the two arms.

**Rebuilt today**, because the arm on disk was from 2026-10-15 and the monthly records have been
rebuilt twice since (operating income on 2026-10-15; pre-2014 renames and predecessor CIKs on
2026-10-24). Pairing a current baseline against a stale arm would have put two data vintages inside
one comparison, which is the thing the engine refuses everywhere else.

    python backtest_v2.py --diagnostic-filings-as-of 2026-10-25 --shard k/4   (then --merge 4)

81 months in 44 minutes across four shards. **35,470 stock-months, against the 2026-10-15 arm's
34,741** - the 10-24 fix reached this arm too, and it carries slightly more stock-months than the
point-in-time arm's 35,305, which is what you would expect: today's filings exist for companies
whose filings as of 2020 did not.

## 3. What was registered, before the number existed

`research/experiments/registry.yaml`, new `controls:` section, committed in `e55d36a` **before** the
arm was rebuilt and before the comparison was run.

- **Prediction.** Positive and large: the mean monthly IC advantage over the baseline, paired month
  by month on the stocks both rankings carry, over the 80 months 2020-01 .. 2026-08, lands between
  **+0.030 and +0.075** and clears the same bar every experiment is judged against (p < 0.05/N and
  |t| >= 3).
- **Failure condition, written before the run.** If the difference does not clear that bar, it is
  **not** a finding about the screener. It would mean this design cannot see a signal of the size
  look-ahead manufactures, and every "not distinguishable" verdict in the ledger - and the research
  page's sentence about the test seeing a signal - would have to be re-read as uninformative until
  the cause was found. A defect to investigate, not a result to publish.
- **Disclosure.** A control's outcome is known by construction, so the entry carries
  `outcome_known_at_registration: ["2020-2026"]` and the report marks it with the same dagger the
  seven `only-*` experiments carry. `refuse_undisclosed_controls` makes an undisclosed control
  impossible, because a control that did not say so would be a false pre-registration wearing a
  real one's clothes.

**What was genuinely new and what was not**, stated because this is a known-outcome row. Already on
file: each arm's own full-universe IC over these months (+0.050 with t 4.71 against -0.002),
committed 2026-10-15, from which a reader could have guessed a difference near +0.05. **Not** on
file: the paired difference itself. Method version 2 recomputes both sides on each month's common
stocks before differencing and has never been applied to this arm; the records have been rebuilt
twice since; and the arm itself is rebuilt here.

## 4. The result: the design sees it, and inside the band registered beforehand

Ledger row `lookahead-filings`, run 2026-10-25, method version 2, 80 months, arm SHA-256
`b1a941f2765ecf24`, 35,470 stock-months.

| Period | Months | The arm's own IC | The baseline, same stocks | Difference (95% interval) | t | Stocks in common |
|---|---|---|---|---|---|---|
| Discovery, 2020-01..2023-06 | 42 | +0.0425 | -0.0043 | **+0.0467** (+0.0135 to +0.0799) | 2.84 | 420 |
| **Held out, 2023-07..2026-08** | 38 | +0.0602 | +0.0041 | **+0.0561** (+0.0425 to +0.0698) | **8.34** | 464 |
| **All, 2020-01..2026-08** | 80 | +0.0509 | -0.0003 | **+0.0512** (+0.0330 to +0.0694) | **5.59** | 441 |

**Verdict: the design sees it.** On the registered period (all 80 months) the difference is
**+0.0512** with **t 5.59** and p below 0.0001, against a bar of p < 0.05/28 = 0.00179 and |t| >= 3.
It lands inside the **+0.030 to +0.075** band registered before the arm was rebuilt. The engine's own
`detectable` for this comparison is **0.0373** - the smallest difference it would catch eight times
in ten at that bar - so the signal is about 1.4x the floor, which is the honest reading of "it can
see this".

Three things worth noting in the table:

1. **The baseline on the same stocks is -0.0003 over 80 months.** The two arms differ in the filings
   they read and nothing else, so this is the cleanest statement of the screener's own position: with
   the filings as they stood, approximately nothing; with the filings as they stand today,
   +0.05 a month. The gap is the look-ahead, measured.
2. **The held-out half is the stronger one** (+0.0561, t 8.34, against +0.0467, t 2.84 in discovery).
   Nothing was tuned to either, and the discovery half contains 2020-2022, when prices moved far more
   than filings did.
3. **It reproduces the 2026-10-15 figures** it replaces (arm IC +0.0509 against the published
   +0.050; baseline -0.0003 on the common stocks against the published -0.002 on its own universe),
   on monthly records rebuilt twice since and with the pairing applied. The earlier measurement was
   right; it just could not be checked by anyone, because it was not a row.

**And the rerun moved nothing**, which is the other half of a result nobody should take on trust.
Registering the control staled all 31 existing rows, so they were rerun, and each newest row was then
compared with its predecessor at full precision:
**31 ids, 24,796 stored statistics, 0 differences, largest 0.0**
(`research/measurements/2026-10-25-control-rerun-effect.py` + JSON). That was the expected answer -
the diff adds a parallel path and does not touch the experiment path - but the public page quotes
these figures, and "it should be identical" is how a silent change gets published.

## 5. What this does and does not license

**Does.** The 28 flat verdicts and the baseline's near-zero IC can now be read as findings about the
rankings rather than as an artefact of a blind test, **for differences of about this size**. That is
the sentence the research page has been making since 2026-10-15; from today it is a number from the
ledger, rerun with everything else, rather than an assertion.

**Does not.** A control establishes power at the size it was run at, and nothing more. The engine's
own `detectable` figure is the honest statement of the rest: this design sees a difference of roughly
0.04 eight times in ten, and a *typical* published multi-factor rank IC is 0.02-0.05. So a large
effect is ruled out and a typical one is not - which is precisely what
`research/2026-10-15-research-engine-first-batch.md` already says, and the control does not improve
it. **Nothing here is decision-grade** (CLAUDE.md rule 5), and a control is further from
decision-grade than anything else in the ledger: it is deliberately cheating.

Three further limits, all registered:

1. **2020-2026 only.** 2012-2019 would need a second diagnostic build of 96 months.
2. **The hindsight shrinks across the window**, from about six years at the start to about two
   months at the end, so the difference is a **lower bound** on what a uniform look-ahead would do.
3. **The universe's own limits** - survivorship, no analyst inputs - apply to both arms equally and
   so cancel in the difference. They still bound what the *baseline* row means.

## 6. A cost this exposed, worth writing down

`tests/test_v2_experiments.py::test_every_ledger_row_the_report_reads_names_its_code_and_its_baseline`
requires every newest ledger row to share one `code_hash`. Adding the control therefore forced a
rerun of all 31 existing rows - **whose numbers provably cannot move**, because nothing on the
experiment path was touched (the diff adds a parallel path and a report section).

The test is right: rows that do not compare must not sit in one table, and the alternative - a
footnote - is what the project has repeatedly found gets ignored. The rerun is right for the same
reason. But the consequence should be budgeted rather than discovered: **a session that adds a
control, an evaluation or an experiment is spending about 45 minutes on a rerun**, and it must not
start one it cannot finish. The ledger is append-only (engine rule 3), so a half-done rerun leaves
the report mixed and gate 1 red and **cannot be undone by deleting rows** - only by finishing. That
is how the 2026-10-18 and 2026-10-19 sessions failed their gates, and today the first rerun attempt
was abandoned mid-flight for an unrelated reason (it was starving the arm's build of CPU), which left
one baseline row at the new hash and made finishing the rerun mandatory rather than optional.
