# Factor Lab - the research behind the screener, rerun

**Created 2026-10-12 (owner-run).** Owner, that evening, after reading about a set of agents that
turned a reading list of papers into a browser lab where readers rerun the experiments: *"this is
something i read today and thought it was a really good idea ... i think it would be really cool"*,
then: *"do whatever you think would improve these tools please! just keep looking for improvements,
then have also make a plan for the nightly sessions to dig even deeper!"*

This file is that plan. The first experiment shipped the same night; the queue below is the
nightly sessions' work.

## What it is

`lab.html` (public, linked from the methodology) takes a paper the screener cites, reproduces its
headline result from data anyone can download, publishes how far the reproduction lands from the
paper, and lets a reader change the conditions in the browser.

Why it belongs here: the screener's methodology rests on published research and its backtest is
benched until 2027-02-11 (CLAUDE.md rules 4-5). Until now a student had to take each citation on
trust. The lab turns "we cite it" into "rerun it" - which is the credibility the tool sells and
the teaching the investment-club audience needs.

## The four parts every experiment shows

Taken from the idea as the owner read it, and the same discipline as `claims.py`:

1. **The claim being tested**, with its source.
2. **The conditions used** - data, sample, estimator, settings.
3. **The difference from the reference result**, figure by figure, against tolerances written in
   `factor_lab.py`, with every raw difference printed beside its verdict.
4. **The details still needed** to call it verified.

A result that does not match stays on the page marked "outside". It is never hidden and a
tolerance is never widened to absorb it (`test_the_tolerances_are_the_recorded_ones`).

## How it is built

| Piece | Job |
|---|---|
| `factor_lab.py` | **The reference.** Parses the source files, defines every calculation, holds each experiment's record (`EXPERIMENTS`), and builds `lab_data.js`. |
| `lab_engine.js` | The same arithmetic for the browser. Must reproduce the Python on the published data at the paper's settings and away from them. |
| `lab_data.js` | **Generated** (`python factor_lab.py --build`). Derived monthly series, the paper's figures, the computed comparison. Never hand-edit. |
| `lab.html` | Hand-written source (not generated). Controls, charts, the four parts, the code it ran. |
| `data/lab/french/` | The library's own files. **Gitignored**, re-downloadable (`--fetch`), never committed. |
| `tests/test_factor_lab.py` | 45 tests: the arithmetic, the parity, the computed verdicts, the four parts, no advice language, no scoring import, a real-browser run at 1440 and 375 px. |

The dashboard may not compute its own numbers (row 0.8). The lab has to - rerunning is the
point - so it carries that rule in a different form: **Python is the reference, the browser must
match it, and the page warns if its own calculation leaves the build's.**

## From the lab to the weights

Added 2026-10-12, in answer to the owner: *"based off everything you are learning and seeing in
these factor labs, will that be used to improve the methodology in the screener? ... down the
road i want this to be the methodology behind my hedge fund/real trading bot."*

**Yes - by a written route, never by a code path.** Rule 1 below stands: nothing in the lab sets
a parameter, and no pipeline module may import it. What the lab produces is *questions with
evidence attached*, and they reach a weight in four steps.

1. **The lab raises the question.** `factor_lab.scorecard` builds one row per category for the
   front page: the weight (read from `config.yaml`), what the lab has measured that bears on it
   among large companies, and what that leaves open. A row is not a verdict.
2. **A Monday research note answers it** from the literature and from documented practice
   (CLAUDE.md, "Your mandate"). The lab's figures go in as measured evidence about the
   *published research*; the note must also say what the literature and practitioners say about
   the same question, and may conclude "no change".
3. **A Wednesday synthesis** says how the change fits the other seven categories: what it
   overlaps, what it makes redundant, what the whole composite is then betting on.
4. **A `METHODOLOGY_CHANGELOG.md` entry, one change at a time**, then the build. From there the
   live record takes over: the published ranking's forward returns, the track record, and the
   paper accounts of `CalebSmit/screener-bot`, which accrue about one independent observation a
   month and confirm or contradict a change over time (rule 4: they do not drive it yet).

**What a lab result is evidence for:** that a cited study says what the methodology says it
says; whether its result held after publication; whether it exists among large companies; how
much two factors overlap. **What it is never evidence for:** a particular weight, a setting
found by trying settings, or the return of the screener's own ranking. The lab's portfolios are
value-weighted, go long one group and short another, pay no costs, and sort on the academic
definition of each trait; the screener ranks 500 large companies on its own metrics within
sectors and shorts nothing.

**How to read the evidence - written before any of the questions below is researched**, so the
reading cannot be bent to the answer:

- A period of under twenty years with a t-statistic inside 2 is "not confirmed", not "gone". A
  weight is not cut on weakness since publication alone.
- For this screener, evidence among large companies outranks evidence across all stocks.
- A change needs the lab and the literature or practice pointing the same way. Where they
  disagree the note says so and the weight stays.
- One category at a time, in steps of at most 3 points (the improvement engine's own
  `max_change_per_cycle`), so each change can be read in the live record before the next.

### The questions, in the order to research them

Ranked by weight at stake times how little stands behind it. Each is a Monday.

| # | Question | Weight | What the lab has | Read first |
|---|---|---|---|---|
| 1 | **What is the case for Growth?** **RESEARCHED 2026-10-12 - `research/2026-10-12-growth-category.md`.** No published premium for growth as the category measures it; index providers treat it as a style; in the composite it is a counterweight (the published ranking's correlation with the Growth score is -0.01). **A proposal waits for a second session's review: Growth 13 -> 10, Quality 22 -> 25** - section 7 of the note has the argument, the case against, and the build list. Decide and record it either way. | 13% | Nothing of its own; experiments 3 and 8 for asset growth and value | The note |
| 2 | **Momentum and value among large companies.** **RESEARCHED 2026-10-12 - `research/2026-10-12-value-and-momentum-among-large-companies.md`.** Two different stories. Value on book-to-market was never reliably there among the largest stocks (Israel & Moskowitz 2013; AQR 2015), the premium is larger on the earnings and cash-flow measures the screener weights, and about 90% of it is the long side. Momentum is strong in every published sample and flat on the public data since 2000 - the lowest 320-month window on record; outside the panic state +0.33% a month. **Lab and literature disagree, so by the reading rules the weight stays.** Recorded as the composite's least well-supported large weight. | 13% + 22% | Experiment 8; the measurement script | The note; then rerun Israel & Moskowitz's largest quintile (queue) |
| 3 | **Is Investment Valuation counted twice?** Correlation 0.70 across all stocks, 0.52 among big companies; the paper's own regression leaves value nothing once investment and profitability are in. | 5% (+22%) | Experiments 3 and 8 | Fama & French (2015) section 7; Hou, Xue & Zhang (2015) (from memory; verify) |
| 4 | **Is Size worth 5%?** 0.10% a month between the two largest groups, distinguishable from zero only over a century. | 5% | Experiment 4 | Asness et al. (2018), "Size matters, if you control your junk" (from memory; verify) |
| 5 | **What is Quality for?** Profitability is the one factor as strong among big companies as small, and the only one above the bar there since 2000 - but under half of it is the held side; most is unprofitable companies lagging. That reads as a screen that keeps weak names out, not one that lifts strong ones. Only worth asking about its weight once 1-4 free any. | 22% | Experiments 3 and 8 | Novy-Marx (2013); Asness, Frazzini & Pedersen (2019) |

**What the lab cannot give a real-money methodology, and what does** (so nobody mistakes eight
experiments for a validated strategy):

- *The composite's own record, point in time.* `plan/backtest-v2.md`; look-ahead bias touches at
  least 63% of today's backtest and nothing decides on it until 2027-02-11.
- *A live record.* The bot's paper accounts began 2026-10-12. A year is twelve observations.
- *What a long-only investor keeps.* Every headline lab figure is long minus short. Experiment 8
  now splits each big-company factor at the market's return (`split[k]["long"]`, `["short"]`):
  over 1963-2026 the held side is 0.20% a month for value, 0.13% for investment, 0.20% for
  momentum and **0.09% for profitability, under half of that factor** - research note section 8.
  This is still a value-weighted third of big companies, not a top-25 list.
- *Costs, turnover and capacity.* None is in any lab figure.

## Rules that keep it honest

1. **The lab never sets a screener parameter.** A slider that finds a better-looking setting is
   how backtests lie. Methodology stays research-led. `test_no_scoring_module_reads_the_lab`.
2. **It reruns the paper's portfolios, not the screener.** Long-short factor portfolios with no
   costs. It is not a backtest of the published ranking and must never be described as one.
3. **Fix the tolerances before the comparison is run.** Experiment 1's were written after its
   first comparison was seen, and the page says so. Every later experiment writes them first.
4. **One experiment per session, finished.** Reader, builder, reviewer and interface work are the
   four steps of one session (below), not four sessions. A half-built experiment does not ship.
5. **A failed replication is a finding, not a bug to tune away.** If a cited result cannot be
   reproduced, or vanishes after its sample ends, write it up in `research/` and weigh what it
   means for the weight that cites it - as research, under rule 4, with a changelog entry if
   anything changes.
6. **Every experiment has an "after the paper" preset**, so the reader sees how the result held
   once its sample ended. Experiment 1's Sharpe ratios go from 0.54 / 1.00 in-sample to
   0.20 / 0.37 after 2011: weaker, with the risk-managed version still ahead.
7. **No advice language** (`advice_terms_in`), and "Not investment advice" on the page.

## The four steps of a lab session

The idea used separate agents per role. A nightly session is one agent and one scarce session, so
the roles are steps, each with something to hand to the next. Independent reading or reviewing
may be delegated to a subagent when it genuinely runs in parallel.

1. **Read.** Get the paper itself (an open copy; read PDFs with `pypdf`). Extract the method's
   equations, the sample, and the one table to reproduce, **with page numbers**. List what the
   paper does not say. Write the tolerances now.
2. **Build.** The Python reference in `factor_lab.py`, on data fetched to `data/lab/`. Publish
   only the derived series the page needs.
3. **Review.** Compare with the paper's table. Then try to break it: does the forecast use only
   earlier data (a test injects a wild value and checks it is not seen early)? Do the legs add
   up? Are the months aligned? Does it agree with any series the data's own publisher prints for
   the same thing? Record every gap in `open_details`.
4. **Interface.** The controls a student would reach for, the same four sections, and a look at
   1440 and 375 px before shipping. Say what you saw.

## Queue

Ordered. Each line says which of the screener's eight categories it puts a rerunnable result
behind, because the aim is that **every weight on the methodology page has either an experiment
or an honest "cannot be rerun on free data, and here is what it would need"**.

| # | Experiment | Category | Data | State |
|---|---|---|---|---|
| 1 | Barroso & Santa-Clara (2015), risk-managed momentum | Momentum | French momentum deciles, monthly + daily | **Shipped 2026-10-12.** 13 of 14 figures within tolerance; the risk-managed worst month is outside (-24.16% here, -28.40% in the paper). |
| 2 | Daniel & Moskowitz (2016), momentum crashes | Momentum, and the screener's own regime rule | French momentum deciles + market factors | **Shipped 2026-10-12.** 9 of 15 figures within tolerances written beforehand. Every figure about what follows a market decline is within (183 bear months against 183; the beta change -1.11 against -1.13). The six outside all describe the portfolio itself: the paper builds its own deciles, which earn about three points a year more than the library's. |
| 2b | Experiment 2 again, on the authors' own portfolios | Momentum | Kent Daniel's website publishes the paper's decile returns | **Check its terms first.** If usable, the six portfolio-level misses should close; if they do not, that is a finding. |
| 3 | Fama & French (2015), five factors | Valuation, Quality (profitability), Investment, Size | French 5-factor file (2x3) | **Shipped 2026-10-12.** 20 of 21 figures within tolerances written beforehand; profitability's volatility is outside by 0.01. After the paper's sample (2014 on) none of the four factors is distinguishable from zero - see `research/2026-10-12-factor-lab-findings.md` before reading anything into that. |
| 4 | The size effect, and whether it reaches large stocks (Fama & French 1992) | Size | French size deciles | **Shipped 2026-10-12. The lab's first replication that mostly fails: 3 of 11 figures within** tolerances written beforehand. The paper's table uses only non-financial firms with accounting data; the library's groups hold every stock. It stays on the page as a miss. **The measurement that matters here:** the ninth size group out-earned the tenth by 0.07-0.14% a month in every period and weighting tried, distinguishable from zero only over the full century. `research/2026-10-12-factor-lab-findings.md` section 5. |
| 5 | **Novy-Marx (2013), profitability** | Quality | French operating-profitability portfolios | **Next.** The library sorts on operating, not gross, profitability: that difference goes in "still needed". |
| 6 | Do riskier stocks earn more? Ang, Hodrick, Xing & Zhang (2006) | Risk | `Portfolios_Formed_on_VAR` | **Shipped 2026-10-12 as experiment 5. 5 of 11 within** tolerances written beforehand; the most volatile fifth was expected to miss and does (0.81% a month here, 0.09% in the paper). It is the first user of the **generic sorted-portfolio layout** (`kind: "sort"`, `factor_lab.SORTS`, `sort_run`, `sortRun`): a new sort now needs a `SORTS` entry, a record with its `sort` block and reference figures, and nothing on the page. The beta sort shipped the same night as **experiment 7** (Fama & French 1992, Table II Panel B: 7 of 11 within, and the flat relation between beta and return reproduces). Still open for Risk: the paper's own headline sort (`_RESVAR`). |
| 6b | Does a high earnings yield pay? Fama & French (1992), Table IV | Valuation | `Portfolios_Formed_on_E-P` | **Shipped 2026-10-12 as experiment 6, with no change to the page** (the generic layout). **7 of 11 within**; the misses are small and mostly at the two ends. Still open: the cash-flow-yield sort (`_CF-P`), which is nearer the screener's heaviest valuation measure, and the book-to-market panel of the same table. |
| 6c | Accruals (Sloan 1996) | Quality | `Portfolios_Formed_on_AC` (confirmed present) | A partial stand-in for the accounting-quality studies that cannot be rerun. |
| 7 | Asness, Moskowitz & Pedersen (2013), value and momentum together | Valuation + Momentum | French HML and momentum factors | The correlation the 2026-10-12 changelog entry leans on for sending freed weight to Valuation. Verify it. |
| 8 | Results in their papers' samples and since (the question McLean & Pontiff 2016 study) | All | The experiments' own series | **Shipped 2026-10-12** as a table on the front page (`factor_lab.after_the_paper`): eleven headline results, each with its months. 9 of the 10 that claim a direction were weaker after their sample. A new experiment adds its row there. Still open: reproducing McLean & Pontiff's own figures, which needs their list of anomalies. |
| - | **Cannot be rerun on free data:** Piotroski (2000), Beneish (1999), analyst revisions (Chan, Jegadeesh & Lakonishok 1996), post-earnings drift | Quality, Revisions, Growth | Needs stock-level fundamentals and estimates history | Give each a card saying exactly that and what data it would take. Check first whether the French library's accruals and net-issuance portfolios can stand in for Sloan (1996). |

## Dig deeper: replication as an audit of the screener's own citations

The lab is also a lens for the work the nightly sessions already do. Three standing tasks:

1. **Every effect size the methodology quotes gets a tag.** Go through `SCREENER_OVERVIEW.md`'s
   generator and `METHODOLOGY_CHANGELOG.md` for quoted numbers from papers. Tag each: *rerun in
   the lab* (link it), *cannot be rerun on free data*, or *not yet tried*. Keep the table in this
   file. A number that turns out not to be in its source is a correction - the 2026-10-09 pass
   over the context panel found three.
2. **When Monday's research note cites a result the lab could rerun, add it to the queue** rather
   than quoting it on trust.
3. **Read each new experiment against the weight it supports.** Experiment 1 already says
   something: the paper's scaling needs exposures up to 2.09, which is why the screener takes
   only the defensive half. Write that kind of sentence into "Where this sits in the screener".

## Cadence

- **Friday (harden and teach) is the lab's day**: one experiment, finished, or a refresh.
- **Monday** adds to the queue when its research note cites something rerunnable.
- **Tuesday (product)** owns the interface backlog below.
- The context-layer item keeps its place on the other days; this does not displace it.
- **First Friday of each month:** `python factor_lab.py --fetch --build`, look at the page, commit
  `lab_data.js`. The verdict count can move when the library is rebuilt; say so in the log if it does.

## Interface backlog

- ~~A pointer readout on the charts~~ - done 2026-10-12: point at (or touch) a chart and the
  month is read out in a sentence under it.
- ~~An index once there is a second experiment~~ - done 2026-10-12 (the switch above the
  experiment). Keeping one scene while switching method is still open.
- ~~A link that opens a given scene~~ - done 2026-10-12: the address carries the experiment and
  any settings that differ from its opening ones (`lab.html#momentum-crashes?threshold=0.3`),
  with a "Copy a link to this scene" button. A setting a link cannot mean is ignored.
- ~~A way into the lab from the dashboard~~ - done 2026-10-12: the footer and the search palette
  (Ctrl/Cmd+K), plus the link in the methodology. The header's design is closed; leave it.
- A light look at whether the first paint needs the fonts it waits for.

## Open, for the owner

- **Data terms.** The French library's pages carry a copyright line and no stated terms for
  reuse. `lab_data.js` publishes a *derived* monthly series with attribution and a link, not the
  library's files. If that is more than you want public, delete `lab_data.js` and `lab.html` and
  the rest of the site is untouched.
- **More agents at once.** The idea as written used twelve. One experiment a week needs none.
  If you want the whole queue quickly, say "use a workflow" in a session and it can run several
  experiments in parallel - at several times the usage.

## Progress (update each session)

- **2026-10-12 (owner-run, past 23:00) - experiment 8 shipped: do the factors work among large
  companies?** Fama & French (2015) Table 4A's second block, the factors within each half of the
  market: **12 of 18 within** tolerances written first (17 or so were expected; the misses are all
  value and profitability among big companies, and stay marked). A new kind, `split`
  (`split_data`, `split_run`, `E.splitRun`, renderer `bigAndSmall`), with momentum shown beside
  the paper's three. **And the front page now has a scorecard** (`factor_lab.scorecard`): each
  category's weight from `config.yaml`, what the lab measured among large companies, what that
  leaves open. **"From the lab to the weights" above is the process and the ranked questions** -
  start a Monday there. Eight experiments, 76 of 112 figures within tolerance.
  **Before the 02:00 run: experiment 9 shipped - Israel & Moskowitz (2013), the largest fifth.**
  5 of 12 within tolerances written first; all four largest-fifth figures match and every miss
  involves the smallest fifth. It reuses experiment 8's layout: **`kind: "split"` now takes an
  optional `split` record** (`data`, `keys`, group names, presets), so another two-group
  comparison is a record and a data function, no page work. Momentum in the largest fifth since
  2000: 2.7% a year, t 0.54 - the flat stretch is there too. Nine experiments, 81 of 124
  figures. **Friday 2026-10-16 is a retrospective, so the lab's next day is 2026-10-23**: the
  paper's risk-adjusted figures, or Novy-Marx (2013).
  **After midnight: question 2 (value and momentum among large companies) was researched** from
  six more papers - `research/2026-10-12-value-and-momentum-among-large-companies.md`. No weight
  proposed to change; the methodology's Valuation and Momentum sections now state the
  large-company evidence, and momentum's record since 2000, with the figures held to the lab's
  data by `tests/test_large_company_evidence.py`. **Next for the lab: rerun Israel & Moskowitz
  (2013)** - the largest size quintile, long and short sides, by period, to 2026.
  **Near midnight: question 1 (Growth) was researched** from six papers and the index providers'
  own methodology documents - `research/2026-10-12-growth-category.md`. The methodology page and
  the scorecard's Growth row say what it found. No weight moved; a 3-point proposal waits for
  review (the table above).
  **Later the same night: experiment 8 also splits each big-company factor into its two sides**
  (held group minus the market, market minus shorted group; they add up to the factor, and a
  test checks it month by month). Not in the paper's table, so not in the comparison.

- **2026-10-12 (owner-run, the last of the night) - experiment 7 shipped: market beta**, and
  **the methodology page now links each category to its experiments** (generator; a test keeps
  the links in step with `CATEGORY_MAP`, because the pipeline may not import the lab). Seven
  experiments, six of eight categories covered; Growth and Analyst revisions have nothing and
  the front page says why. **When you add an experiment, three things must move together or
  the suite fails:** `TAKEAWAYS`, `CATEGORY_MAP`, and the "Rerun the research" paragraph in
  `run_screener.build_screener_overview`.

- **2026-10-12 (owner-run, the same night) - experiment 6 shipped: earnings yield.** Fama &
  French (1992) Table IV Panel B on the library's earnings-yield deciles: 7 of 11 within, the
  shape and the end-to-end gap both matching. Six experiments; Valuation now has a sort on a
  yield, not only book-to-market. What it shows since 2007 is in
  `research/2026-10-12-factor-lab-findings.md` section 7 and is the most important thing the lab
  has found for the screener's largest category.

- **2026-10-12 (owner-run, the same night) - experiment 5 shipped on a reusable layout.** Ang,
  Hodrick, Xing & Zhang (2006) on the library's variance-sorted fifths: 5 of 11 within. The
  page layout, the Python (`sort_run`) and the browser arithmetic (`sortRun`) are generic, so
  queue items 5, 6b and 6c (profitability, earnings and cash-flow yield, accruals) are each a
  record and a paper's table away - **read the paper's table and write the tolerances first.**
  The experiment switcher is one swipeable row on a phone.

- **2026-10-12 (owner-run, the same night) - the lab has a front page.** `lab.html` now opens
  on an overview: a card per experiment with its verdict and a one-sentence takeaway, and a
  table of what the lab can and cannot put behind each of the screener's eight categories
  (`factor_lab.CATEGORY_MAP`, `TAKEAWAYS`). Every takeaway is checked against the published
  data by a test; the map must list the config's eight categories exactly. **Keep both true when
  adding an experiment** - the tests will insist. The same session added limitation 11 to the
  methodology page (generator): what the four factors have earned since 2014, as a registered
  claim that a monthly refresh of the lab can falsify.

- **2026-10-12 (owner-run, the same night) - experiment 4 shipped, and it mostly fails.** Fama &
  French (1992) Table II on the library's size deciles: 3 of 11 within tolerances written
  before the first run. Rule 5 in action - the failure is published with its likely cause (a
  different set of companies), not tuned away, and a test fails if the tolerances move. The page
  adds a bar chart by size group with the two largest in the accent colour, and the difference
  between the ninth and tenth groups for any period, equal- or value-weighted.

- **2026-10-12 (owner-run, the same night) - experiment 3 shipped.** Fama & French (2015) on the
  library's five factors: 20 of 21 within tolerances written before the first run. The page
  follows one factor in the accent colour against the other four, with each factor's average
  month, t-statistic and growth, and what the other four leave unexplained of the one followed.
  **What the three experiments say for the screener is in
  `research/2026-10-12-factor-lab-findings.md`** - read it before the next methodology session:
  the results weakened after their samples, and value and investment are largely one bet.

- **2026-10-12 (owner-run, later the same night) - experiment 2 shipped**, at the owner's
  "keep improving ... be creative". Daniel & Moskowitz (2016) on the library's momentum deciles
  and market factors. The page sorts every month into four market states known beforehand and
  shades the panic state on the chart; at the screener's own thresholds over the paper's sample
  that state covers 94 of 1,035 months (9.1%) and momentum averaged **-1.68% a month in it
  against +1.52% otherwise**. The page now switches between experiments (`#momentum-crashes`),
  and its two script files carry a version so a cached copy cannot run against a newer page
  (`factor_lab.ASSET_VERSION`). **Tolerances were written before the first run**; rule 4's "one
  experiment per session" was set aside by the owner's instruction, not by the session.
  **A finding for the screener's rule, recorded in `research/2026-10-09-momentum-regime.md`:**
  measured on all US stocks the state also held going into April 2020 and January 2023, where
  the S&P 500 total-return index the screener reads did not show it.

- **2026-10-12 (owner-run) - experiment 1 shipped.** `factor_lab.py`, `lab_engine.js`,
  `lab_data.js`, `lab.html`, 15 tests. Reference figures from the authors' April 2012 working
  paper, read through a text copy; the published tables are not yet read. Looked at 800 and
  375 px: no sideways scroll, comparison table fits a phone with its verdict column visible,
  both charts redraw on a change, no console errors.
