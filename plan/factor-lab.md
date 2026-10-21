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
| `lab_engine.js` | The same arithmetic for the browser. Must reproduce the Python on the published data at the paper's settings and away from them - including each experiment's comparison with its paper (`compare`, `compareCrashes`, `compareFactors`, `compareSize`, `compareSort`, `compareSplit`, `compareCombo`), from which the page fills "Against the paper". |
| `lab_data.js` | **Generated** (`python factor_lab.py --build`). Derived monthly series, the paper's figures, the computed comparison. Never hand-edit. |
| `lab.html` | Hand-written source (not generated). Controls, charts, the four parts, the code it ran. |
| `data/lab/french/` | The library's own files. **Gitignored**, re-downloadable (`--fetch`), never committed. |
| `data/lab/osap/` | Open Source Asset Pricing's files for the sorts whose `SORTS` entry says `source: "osap"` (2026-10-20). Same rules; `--fetch` downloads them too, and Google Drive's "Quota exceeded" means retry later. |
| `tests/test_factor_lab.py` | 61 tests (counted 2026-10-19; recount rather than trust it): the arithmetic, the parity (including every row of every comparison with the paper, `E.compare*`, to 1e-9), the computed verdicts, the four parts, when each experiment's tolerances were written, what a null result would match, every printed difference against its verdict, no advice language, no scoring import, and real-browser runs at 1200/1440 and 375 px (in-page links, the year pickers, the charts by keyboard, no table sliding sideways on a phone). |

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
| 1 | **What is the case for Growth?** **RESEARCHED 2026-10-12, WEIGHT DECIDED 2026-10-13 - `research/2026-10-12-growth-category.md`, sections 7 and 10.** No published premium for growth as the category measures it; index providers treat it as a style; in the composite it is a counterweight. The proposal (Growth 13 -> 10, Quality 22 -> 25) was **refused** by the reviewing session: net growth exposure is monotonic in the weight, crosses zero at **13.5** and stays within 0.05 of neutral only over **11-16**, because Growth is the only category whose score runs against the rest (-0.388 against the mean of the other seven). If growth earns nothing, the supported exposure is zero, and 13 points is what delivers it - the cut would take a position rather than remove one, in a direction the evidence does not settle. **The open Growth question is now composition, not weight** (note section 11): `revenue_growth` against `asset_growth`, and whether growth in profitability belongs in Quality. | 13% | Nothing of its own; experiments 3 and 8 for asset growth and value | The note, and `research/measurements/2026-10-13-growth-weight-sweep.py` |
| 2 | **Momentum and value among large companies.** **RESEARCHED 2026-10-12 - `research/2026-10-12-value-and-momentum-among-large-companies.md`.** Two different stories. Value on book-to-market was never reliably there among the largest stocks (Israel & Moskowitz 2013; AQR 2015), the premium is larger on the earnings and cash-flow measures the screener weights, and about 90% of it is the long side. Momentum is strong in every published sample and flat on the public data since 2000 - the lowest 320-month window on record; outside the panic state +0.33% a month. **Lab and literature disagree, so by the reading rules the weight stays.** Recorded as the composite's least well-supported large weight. | 13% + 22% | Experiment 8; the measurement script | The note; then rerun Israel & Moskowitz's largest quintile (queue) |
| 3 | **Is Investment Valuation counted twice? PARTLY ANSWERED 2026-10-19 - `research/2026-10-19-revenue-growth-vs-asset-growth.md` section 11.3 item 2.** The overlap is real and is precisely why practitioners merge the two: nine of eleven provider families house asset growth **inside Quality** rather than as a standalone category (Research Affiliates literally "Quality = Profitability minus Investment"), and Hsu, Kalesnik & Kose (2019) call that pairing "the most parsimonious quality definition... associated with the strongest academic evidence", noting investment's "weaker multifactor alpha because it is correlated with the value factor". The screener keeps Investment standalone, which is the *academic* construction (FF2015's CMA, HXZ's I/A) - a minority choice, defensible, and now known to be one. **What is still open is whether to fold it into Quality**, which is a category-structure change needing its own Monday. The same note also measured the *other* overlap this row names - `revenue_growth` against `asset_growth` - at 24.2% shared variance within sector, and found the pair is a position on the spread between them, not on either level. Original framing kept below. Correlation 0.70 across all stocks, 0.52 among big companies; the paper's own regression leaves value nothing once investment and profitability are in. **Read first, so this is not researched twice:** `research/2026-09-02-category-independence-synthesis.md` already found Size and Investment to be two bets, not one (correlation +0.28, each mostly unspanned by the other seven). And the Growth note found the sharper overlap is the other way round: `revenue_growth` and `asset_growth` correlate +0.51, so **Growth un-counts Investment**. | 5% (+22%) | Experiments 3 and 8 | Fama & French (2015) section 7; Hou, Xue & Zhang (2015) (from memory; verify) |
| 4 | **Is Size worth 5%?** 0.10% a month between the two largest groups, distinguishable from zero only over a century. **Read first:** `research/2026-08-31-size-factor-in-a-large-cap-universe.md` researched this in August; its Candidate 1 (is the size tilt more aggressive than intended?) is still open with a refutation criterion already written. Start there, not from scratch. | 5% | Experiment 4 | Asness et al. (2018), "Size matters, if you control your junk" (from memory; verify) |
| 5 | **What is Quality for?** Profitability is the one factor as strong among big companies as small, and the only one above the bar there since 2000 - but under half of it is the held side; most is unprofitable companies lagging. That reads as a screen that keeps weak names out, not one that lifts strong ones. Only worth asking about its weight once 1-4 free any. | 22% | Experiments 3 and 8 | Novy-Marx (2013); Asness, Frazzini & Pedersen (2019) |
| 6 | **What should capex include in free-cash-flow yield?** Added 2026-10-14 (`research/2026-10-14-yahoo-vs-sec.md` section 3). The screener's FCF subtracts Yahoo's `Capital Expenditure` row, which is PP&E **plus** intangible purchases and, for companies with a finance arm, lease assets (GM $22.6B of which $13.8B leased vehicles). Top 25 unaffected (at most 0.5 composite points); GM's FCF yield 0.3% vs 8.0% on PP&E only, BSX and DVN roughly double. Candidate rules: PP&E only; PP&E plus capitalised software; consolidated including lease assets (matches the consolidated EV). Read how index providers and practitioners define it before choosing. **RESEARCHED 2026-10-14 - `research/2026-10-14-capex-definition.md`.** Finance-arm lease assets: every source that addresses them keeps them out (GM, Ford, CAT, S&P Ratings). Intangibles: sources split (MSCI physical only; S&P Ratings includes them) - unchanged. **Proposal REFUSED 2026-10-15** (section 5 of that note): the rule drops the finance arm's capex but keeps its operating cash flow - GM's own 10-K starts adjusted automotive free cash flow from **net automotive** operating cash flow of $18.7B against the consolidated $26.87B, so the proposal publishes $17.57B against GM's own $9.5B (+85%, where today's figure is -81%). The clean ratio needs a segment-level operating cash flow, which is MD&A prose rather than an XBRL fact. **Shipped instead:** the FCF-yield row decomposes the capex figure from the company's own filings (claim `drilldown.capex_lease_note`). Question 6 is closed on the lease half; the intangibles half stays as the sources left it. | 9.9 (FCF yield) | None - a data-definition question | The note; then MSCI / S&P / Damodaran on FCF |

| 7 | **Should the operating profit behind ROIC and EV/EBITDA include impairments and restructuring charges?** Added 2026-10-15 (`research/2026-10-15-operating-income-source-gap.md`). Measured, not a hunch: the run's `ebit` agrees with the companies' own filed `OperatingIncomeLoss` within 0.5% for only **42.1% of 375 comparable stocks**, against 90.5% for revenue and 98.8% for operating cash flow, and **24% disagree by over 10%** - with the trailing rebuild removed, so it is the two sources' figures for the same fiscal year. The source's figure is the larger for **71%** of those, concentrated in companies that took a write-down (GIS +68%, CVS +55%, KHC and CNC sign flips), which flatters exactly the kind of company a valuation screen surfaces. A minority, mostly real estate, goes the other way, and 38 companies stopped tagging the concept - so "use the filing" is **not** the answer either (and would put three weighted metrics on the weekly SEC cache, which the capex refusal already ruled out). This is a definitional question with an answer in the sources. **SETTLED 2026-10-15 (owner-run): keep the source's "Operating Income" row** - it is as-reported less the source's "Special Income Charges", and the "Total Operating Income As Reported" row equals the filings 374 of 374. Rouen, So & Wang (2021, *JFE* 142(3) 1068-1091) on persistence and MSCI's practice both support the pre-special-items figure for ROIC and EV/EBITDA, so the disagreement with the filing is the definition, not a defect; do **not** "fix" it by switching rows or to the filings. See CLAUDE.md 0.14 and `METHODOLOGY_CHANGELOG.md`. *(This row said "open" until 2026-10-19, four days after it was decided - a stale plan line, corrected under rule 9.)* | 3 metrics: `roic`, `ev_ebitda`, `net_debt_ebitda` (plus `ebitda` via `ebit + da_cf`) | Nothing - a data-definition question, like 6 | The note; then Greenblatt (2006) and Koller et al. *Valuation* on operating earnings, MSCI and S&P quality methodologies on reported vs adjusted |

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
   first comparison was seen, and the page says so. Every later experiment writes them first -
   and where one did not, its record says so: each reference carries `tolerances_written`
   (`before`, `after_first_comparison`, `after_figures_seen`) and, where it is not `before`, a
   `tolerances_note` from the record's own comment, which the page prints under the comparison
   (one function, `tolNote`). Experiments 1 and 12 are `after_first_comparison`, experiment 13
   `after_figures_seen` (2026-10-16 audit: the page had printed "written before" under both).
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
8. **A result of no effect must fail.** Added 2026-10-16 (audit). A new experiment's tolerances
   must make a null result miss its headline figure(s) and most of its figures. The null is built
   from the record alone (`factor_lab.null_rows`): every group earns the same, every spread,
   t-statistic, correlation and premium is zero, and volatilities, the market's own figures and
   counts stay at the paper's values; experiment 1's null is the risk-managed strategy matching
   the plain one. The build publishes `null_within` and the page prints it beside every verdict
   and on the overview card ("A result of no effect at all would match 6 of these 11"). Where the
   paper's own claim is that there is no effect (experiment 7, market beta: 9 of 11), the null
   passing *is* the replication. **The existing experiments are disclosed, not retuned** - no
   tolerance moves to meet this rule (rule 3); low volatility and cash-flow yield (6 of 11 each)
   are the ones where the null matches most, and in both it misses the spread.

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
| 5 | **Novy-Marx (2013), profitability** | Quality | French operating-profitability portfolios | **Shipped 2026-10-15 (owner-run) as experiment 10. 6 of 6 within** tolerances written first (0.10, wider than the 0.05 of a same-variable rerun, because the library sorts on operating profits over book equity where the paper uses gross profits over assets, and keeps the financial firms it drops). The paper reports **excess** returns, so `SORTS` takes `"excess": True` and the sort's data publishes the T-bill rate for its own months (`rf`); the comparison subtracts its average. Spread 0.215% a month against the paper's 0.31% - at the edge, as predicted. Since 2011: the same 0.215%, t 0.79, and the profitable end the calmest (volatility 4.1% against 5.9% a month). Read from the June 2012 accepted draft, Table 2 Panel A. |
| 6 | Do riskier stocks earn more? Ang, Hodrick, Xing & Zhang (2006) | Risk | `Portfolios_Formed_on_VAR` | **Shipped 2026-10-12 as experiment 5. 5 of 11 within** tolerances written beforehand; the most volatile fifth was expected to miss and does (0.81% a month here, 0.09% in the paper). It is the first user of the **generic sorted-portfolio layout** (`kind: "sort"`, `factor_lab.SORTS`, `sort_run`, `sortRun`): a new sort now needs a `SORTS` entry, a record with its `sort` block and reference figures, and nothing on the page. The beta sort shipped the same night as **experiment 7** (Fama & French 1992, Table II Panel B: 7 of 11 within, and the flat relation between beta and return reproduces). Still open for Risk: the paper's own headline sort (`_RESVAR`). |
| 6b | Does a high earnings yield pay? Fama & French (1992), Table IV | Valuation | `Portfolios_Formed_on_E-P` | **Shipped 2026-10-12 as experiment 6, with no change to the page** (the generic layout). **7 of 11 within**; the misses are small and mostly at the two ends. The cash-flow-yield sort (`_CF-P`) **shipped 2026-10-15 as experiment 11** (Lakonishok, Shleifer & Vishny 1994; 11 of 11 within). Still open: the book-to-market panel of the same table. |
| 6c | Accruals (Sloan 1996) | Quality | `Portfolios_Formed_on_AC` | **Shipped 2026-10-15 (owner-run, night) as experiment 12. 6 of 11 within**; the direction reproduces (t 4.7 in the paper's years) but the library's gradient is flatter - half the paper's spread - because its measure has no depreciation term and divides by book equity. The paper's PDF text layer carries only the end rows: the middle rows were read from the rendered page, *after* the ends and spread had been compared (the record's comment says so). The paper has no DOI; the lab now takes a stable `url` in its place. |
| 7 | Asness, Moskowitz & Pedersen (2013), value and momentum together | Valuation + Momentum | French 2x3 value and momentum portfolios | **Premise MEASURED 2026-10-15 (owner-run) - `research/2026-10-15-value-momentum-correlation.md`.** Among big companies over the paper's 1972-2011 window each premium matches the paper (Sharpe 0.27 vs 0.29; 0.35 vs 0.33) but the correlation is milder (-0.20 vs -0.53, likely the library's lagged book-to-market price); the hedge grew stronger after 2011 (-0.39) while both premia went to about zero. The rule's weight transfer is supported as a diversifier, not as a return source. **Shipped the same night as experiment 13** with a new layout (kind `combo`: `factor_lab.combo_run`, `lab_engine.comboRun`, `together()` in `lab.html`): 7 of 10 within. The tolerances follow experiment 10's scale because the Sharpe ratios and correlation had been seen in the measurement first - the record says so. |
| 7b | **Cooper, Gulen & Schill (2008), asset growth** | Investment (and the Growth composition question) | `Portfolios_Formed_on_INV` | **Shipped 2026-10-19 as experiment 14. 13 of 24 within** tolerances committed in a separate commit before the run (`8f7ec9f`). First sort compared under **both** weightings its paper reports (`reference["also"] = "vw"`, figures `vw_mean1`..., `vw_spread`; the null gives each weighting its own centre). The shape reproduces (t 8.0 equal-weighted, 4.1 by market value); both spreads land below the paper's and outside (1.22 vs 1.73; 0.67 vs 1.05). **Since the paper, by market value, about zero** (-0.07% a month, t -0.29, 278 months). `research/2026-10-19-lab-asset-growth.md`. Still open: the paper's large-firm alpha (Table II Panel C) on the 2x3 size-by-investment file. |
| 8 | Results in their papers' samples and since (the question McLean & Pontiff 2016 study) | All | The experiments' own series | **Shipped 2026-10-12** as a table on the front page (`factor_lab.after_the_paper`): eleven headline results, each with its months. 9 of the 10 that claim a direction were weaker after their sample. A new experiment adds its row there. Still open: reproducing McLean & Pontiff's own figures, which needs their list of anomalies. |
| 7c | **Piotroski (2000), the F-score among high book-to-market companies** | Quality | Open Source Asset Pricing (Chen & Zimmermann 2022), signal `PS`: the authors' own cut (value-weighted deciles) and their equal-weighted deciles | **Shipped 2026-10-20 as experiment 15. 1 of 4 within** tolerances committed before any OSAP return had been read (`ce78f7d`). The highest-score decile matches (30.4% a year vs 31.3%); the lowest does not (26.1% vs 7.8%), so the spread is 4.3% a year (t 1.5) against 23.5%. Measured reason: OSAP's end deciles of a 0-9 score average 2.7 and 7.5, not the paper's 0-1 and 8-9. By market value - the authors' own cut - 12.6% a year (t 3.6) in the paper's years, 7.5% (t 1.4) since 1997. A rerun of the authors' published portfolio returns, not of the score. `research/2026-10-20-lab-osap.md`. |
| - | **Cannot be rerun on free data:** Beneish (1999); Bernard & Thomas's own drift construction. **Now reachable at the portfolio level through Open Source Asset Pricing (checked 2026-10-20):** analyst revisions (Chan, Jegadeesh & Lakonishok 1996, signal `REV6`), earnings-surprise drift (`EarningsSurprise`, the Foster-Olsen-Shevlin SUE), Mohanram (2005) (`MS`). Piotroski (2000) is done (7c) | Quality, Revisions, Growth | Beneish needs stock-level accounts for decades and is not in OSAP | Next: `REV6` gives Analyst Revisions its first experiment; `MS` gives Growth one on its own measures. Read each paper's table and commit tolerances first. Check whether the French library's accruals and net-issuance portfolios can stand in for Sloan (1996). |

### Free data to evaluate (added 2026-10-20, owner-run)

Owner, 2026-10-20: *"is there any other apis that would be helpful for bringing in lots of useful
data?"* Two free, published research libraries could widen what the lab can rerun - and the second
may close the "cannot be rerun on free data" row above. **Check each one's terms and confirm the
download works before building anything**, and keep the lab's rule: data anyone can download.

| Library | What it adds | Terms, as found 2026-10-20 |
|---|---|---|
| **Jensen, Kelly & Pedersen (2023), "Is there a replication crisis in finance?", *JF* 78(5)** - `jkpfactors.com`; maintained code `github.com/bkelly-lab/jkp-data` | 153 factors in 13 themes, the US and 92 other countries, with size-group versions - the large-company question (`research/2026-10-12-value-and-momentum-among-large-companies.md`) on a second, independent construction, and out of the US entirely | CC BY-NC 4.0 (non-commercial, attribution). Files served from a public bucket; the newest vintage is also on WRDS |
| **Chen & Zimmermann (2022), "Open source cross-sectional asset pricing", *Critical Finance Review* 11(2)** - `openassetpricing.com` | Portfolio returns for 200+ published predictors, built from CRSP/Compustat by the authors - **including signals the lab cannot build itself** (Piotroski's score, analyst revisions, earnings surprise drift). A portfolio-level rerun of those papers needs only the returns | **Checked 2026-10-20 (`research/2026-10-20-lab-osap.md` section 2).** Release 2.0.0 (October 2025), returns to December 2024. **No account, payment or bot check for the portfolio returns**: they are public files in a Google Drive folder linked from the data page; the Python package `openassetpricing` (0.0.2, GPLv2) reads the same folder. Only the CRSP-derived *firm-level* signals (Price, Size, STreversal) need WRDS. **Terms:** the code is **GPL-2.0** (the GitHub repo's licence); the data page and FAQ state **no separate licence for the data**; the README asks *"If you use data or code based on our work, please cite the paper"*. So, as with the French library (which carries a copyright line and no reuse terms), the lab publishes a derived monthly series with the citation and a link, not the files. **Access caveat:** on 2026-10-20 from 20:56 CT Google Drive answered every file in the folder (and in the 2024 release's) with "Quota exceeded - too many users have viewed or downloaded this file recently", which Google says can last up to 24 hours; the package fails at start-up for the same reason (`empty CSV`). It is a download-rate limit, not a sign-in or bot check, and was retried, never worked around. **Status:** see Progress, 2026-10-20 |

Not useful here: the **api.data.gov key** (owner has one). It is a gateway key for EPA, FDA, GPO,
NREL and similar; the federal sources this screener uses - SEC EDGAR, FRED's CSV endpoint - need no
key. If a key is ever needed it goes in a gitignored file beside `data/sec/user_agent.txt`, never in
a tracked one.

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

- **2026-10-20 (evening) - experiment 15: Piotroski (2000), on Open Source Asset Pricing.** The
  "Free data to evaluate" row for OSAP was checked (terms in the table above; full account in
  `research/2026-10-20-lab-osap.md` section 2): free, no account, code GPL-2.0, no stated data
  licence, the authors ask to be cited. Google Drive refused every file for about ninety minutes
  ("Quota exceeded"); the tolerances were committed in that window, before any return was read
  (`ce78f7d`), and the files arrived on retry. **1 of 4 within**: the high-score end matches, the
  low-score end does not, and the decile cut of a ten-value score is the measured reason. The
  generic sort now reads a second source (`SORTS[...]["source"] = "osap"`, `read_osap`,
  `osap_sort_data`, `fetch_osap`; files in gitignored `data/lab/osap/`, found beside
  `data/lab/french/`), and the page's footer carries a second attribution
  (`attribution["others"]`). Nothing on the page changed otherwise. Fifteen experiments, 125 of
  190 within. **Do not `pip install openassetpricing` into the project's Python** - it pulls
  `wrds`, which downgraded pandas below the pinned 2.3.2 (restored); `fetch_osap` needs neither.

- **2026-10-19 - experiment 14: Cooper, Gulen & Schill (2008), asset growth.** Read from the published
  article (Table II Panels B.1/B.2, pp. 1618-1619); tolerances committed on their own before the
  library's investment file was compared (rule 3), and checked against rule 8 from the record (a null
  matches 6 of 24, missing every end, spread and t-statistic). **13 of 24 within**; both spreads below
  the paper's and outside. The generic sort now takes `also` (a second weighting compared in the same
  table) and reports each spread's t-statistic (`t_spread`). After the paper, value-weighted: -0.07% a
  month (t -0.29). What it says for the Growth/Investment composition question is in
  `research/2026-10-19-lab-asset-growth.md` section 6 - research only, no weight proposed. The menu has
  an Investment group; Growth's map row and scorecard now point at this sort. Fourteen experiments,
  124 of 186 within.

- **2026-10-15 (owner-run, 22:30) - experiment 13: Asness, Moskowitz & Pedersen (2013), value and momentum
  together, and a new layout (`combo`).** 7 of 10 within: each premium and the mix's average match
  the paper; the hedge is milder (-0.22 vs -0.53) so the mix's Sharpe is lower (0.49 vs 0.63). After
  2011 among big companies the hedge is stronger (-0.37) and every Sharpe ratio about zero. A new
  combination experiment is a record with `kind: "combo"`; it reads `split` and `factors`, no new data.
  Thirteen experiments, 111 of 162 within.

- **2026-10-15 (owner-run, night) - experiment 12 shipped: Sloan (1996), accruals.** 6 of 11 within;
  spread 4.8% a year against the paper's 10.4% (Table 6, size-adjusted, year t+1). Since the paper:
  equal-weighted 0.30% a month (t 2.9), by market value 0.22% (t 1.45), and since 2007 by market value
  0.09% (t 0.47). Accruals are 5% of Quality; nothing to change - the screener's measure (net income
  less operating cash flow, over assets) is closer to Sloan's than the library's is. Twelve
  experiments, 104 of 152 figures within.

- **2026-10-15 (owner-run, evening) - experiment 11 shipped: Lakonishok, Shleifer & Vishny (1994),
  cash-flow yield.** The nearest public sort to the screener's heaviest single metric (FCF yield,
  ~10% of the composite). The paper's Table 2 (NBER w4360) reports size-adjusted annual returns, so
  `reproduce_sort` gained `relative` (each group against the average of the groups) and `scale`
  (12, percent a year). Tolerances (2.0% a year a decile, 4.0 the spread) written before the run:
  **11 of 11 within**, spread 11.05% a year against the paper's 8.4. **After the paper, the
  weighting decides it** - high-minus-low cash-flow yield, % a month: equally weighted 0.47 (t 2.7)
  1990-2026 and 0.15 (t 0.6) since 2007; **by market value 0.02 (t 0.1) since 1990 and -0.32
  (t -1.0) since 2007.** A large-company book is the value-weighted one. **What it means for the
  weights: nothing yet** - the public sort differs from the screener's metric in three ways
  (capex not subtracted, shares not enterprise value, whole market not within sector), so it is a
  question for a research note, not a reason. Eleven experiments, 98 of 141 figures within.

- **2026-10-15 (owner-run) - experiment 10 shipped: Novy-Marx (2013), profitability.** The owner
  asked for more in the lab and for whether the lab says the methodology should change. 6 of 6
  within tolerances written before the comparison; the spread lands below the paper's for the two
  reasons written down beforehand. **What it means for the weights: nothing to change.** Gross
  profit against assets is 22% of Quality, and Quality 22% of the composite; the rerun supports
  that measure at its strength in the paper's sample and shows it, like every other result here,
  weaker and not yet distinguishable from zero since - while delivering lower volatility at the
  profitable end, which is the Risk category's argument made by Quality's measure. Ten
  experiments, 87 of 130 figures within tolerance.

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
