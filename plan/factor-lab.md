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
| `tests/test_factor_lab.py` | 21 tests: the arithmetic, the parity, the computed verdicts, the four parts, no advice language, no scoring import, a real-browser run at 1440 and 375 px. |

The dashboard may not compute its own numbers (row 0.8). The lab has to - rerunning is the
point - so it carries that rule in a different form: **Python is the reference, the browser must
match it, and the page warns if its own calculation leaves the build's.**

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
| 3 | **Fama & French (2015), five factors** | Valuation, Quality (profitability), Investment, Size | French 5-factor file | **Next.** Reproduce the factors' average returns and t-statistics over the paper's sample; the "after the paper" preset shows value's weak decade. |
| 4 | **Is there a size premium among large stocks?** | Size | French size deciles | The methodology page admits "No published study establishes a size premium within the largest two market-cap deciles." Measure it on the size deciles and say what is found. This one can end in a research note about the Size weight. |
| 5 | Novy-Marx (2013), profitability | Quality | French operating-profitability portfolios | The library sorts on operating, not gross, profitability: that difference goes in "still needed". |
| 6 | Low-risk: Frazzini & Pedersen (2014) or Baker, Bradley & Wurgler (2011) | Risk | French beta / variance portfolios, or AQR's library | **Check AQR's terms of use before publishing anything derived from it.** |
| 7 | Asness, Moskowitz & Pedersen (2013), value and momentum together | Valuation + Momentum | French HML and momentum factors | The correlation the 2026-10-12 changelog entry leans on for sending freed weight to Valuation. Verify it. |
| 8 | Post-publication decay across all of the above (McLean & Pontiff 2016) | All | The experiments' own series | One table: each result in its paper's sample and after it. |
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

- A pointer readout on the charts (date, both values).
- An index once there is a second experiment, and a way to keep one scene while switching method.
- A link that opens a given scene (`#period=...`), like the dashboard's `#stock=`, so a club leader
  can send "look at this".
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
