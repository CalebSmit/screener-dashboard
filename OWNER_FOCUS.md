# Owner focus queue

**This is how the owner tells the nightly session what to work on.**

Write what you want in plain English and save the file. That is the whole
process — no formatting rules, no ticket numbers, no need to say where in the
codebase it lives. The next code session (6:00 AM, Mon–Fri) reads this file
**before** it reads the weekly rotation, and open items here outrank the day's
nominal focus.

You do not need to be precise about *how*. "The sector chart is useless" is a
perfectly good item; working out what to do about it is the session's job. If
an item turns out to be a bad idea, the session will say so in
`NIGHTLY_LOG.md` and explain why rather than silently skipping it.

Two things still outrank this file, and neither is negotiable:

1. **A broken data loop or a failing ship gate.** A stalled pipeline means the
   tool stops improving at all, so it gets fixed first. The session will say in
   the log that it deferred your item and why.
2. **The four ship gates.** Nothing here can authorise a push that fails them.

---

## Open

- **2026-10-15 - A research engine: test the methodology the way automated research loops do, so it
  can be shown to people and trusted.** Owner: *"we're trying to come up with the best methodology
  to screen stocks by, and it has to be backed up, it has to be proven, so ... I can show this to
  people, and they can trust me that this screener works ... first, it's going to be a good trading
  bot, but then eventually ... the methodology behind a hedge fund type of thing that I want to
  start."* **First version built and run the same night (owner-run):** `v2_experiments.py`, 27
  experiments from the literature, each pre-registered with a prediction before it ran, judged on
  held-out months and on 2012-2019 (built only after the rule for it was committed), against a
  multiple-testing bar. **What it found, plainly: nothing beats the current weighting, and the
  current weighting does not measurably rank next month's returns on public data** (IC ~0 in both
  eras; the same harness fed leaked data finds t 4.7, so it can see a signal when there is one).
  Proving a realistic edge from any monthly large-company record takes years (an IC of 0.05 needs
  ~7 years at t >= 3). So what can honestly be shown today is the process and the evidence for
  each factor, not a proven return - `research/2026-10-15-research-engine-first-batch.md`. The
  nightly queue is `plan/research-engine.md`.
  **One decision only the owner can make:** whether to buy the $199/yr delisted-price feed deferred
  on 2026-09-30. It now has a concrete use - 42% of 2012's index members are missing from the test,
  nine in ten of them because they have since left the index, and one result (Size) visibly tracks
  that gap.

- **2026-10-14 - More sessions, faster development.** Owner: *"i want you to be doing more,
  scheduling more runs. I want to speed up the development of these tools please."*
  **Done the same day (owner-run):** two extra scheduled code sessions - 13:00 every day and
  06:00 on Saturday and Sunday - so fourteen a week instead of five, each through the same four
  ship gates. They stand down rather than collide: when an owner session is marked in progress,
  when the tree has someone's work in it, and for a day after a usage limit. Kept open so a
  later session can report whether the extra sessions are producing work of the same standard
  (the next retrospective should look), and so the owner can change the cadence here.

Add items below. Anything under this heading is unclaimed work.

<!-- Add items here, newest at the top. Free text, one item per bullet. -->

- **2026-10-12 - The Factor Lab: turn the papers the screener cites into experiments a reader
  can rerun. First experiment built by an owner-run session; the nightly sessions grow it.**
  Owner, verbatim, after reading about agents that turned a reading list into a browser lab:
  *"this is something i read today and thought it was a really good idea, dont know how it would
  integrate with these tools (the screener and the trading bot), but it think it would be really
  cool"*, and then: *"do whatever you think would improve these tools please! just keep looking
  for improvements, then have also make a plan for the nightly sessions to dig even deeper!"*

  **The plan is `plan/factor-lab.md` - read it first.** In short: `lab.html` reproduces a cited
  paper's headline result from public data, shows the claim, the conditions, the difference from
  the paper and what is still unverified, and lets the reader change the conditions. The queue
  there is ordered so that every weight on the methodology page ends up with either an
  experiment or an honest "cannot be rerun on free data". **Friday is the lab's day** (one
  experiment, finished); Monday adds to its queue; the context-layer item below keeps the other
  days. The lab never sets a screener parameter and no scoring module may import it.

  The same plan carries the "dig deeper" half: use replication as an audit of the screener's own
  citations - every effect size the methodology quotes gets tagged *rerun in the lab*, *cannot
  be rerun on free data* or *not yet tried*.

  This item is about the screener. The trading bot is a separate repo that nightly sessions do
  not work on (CLAUDE.md "Downstream consumer"); its own next steps are in its `ROADMAP.md`.

  **Progress (update this line each session):**
  - **2026-10-14 (06:00 nightly) - deferred for a red ship gate, nothing lost.** The day's work was a
    data defect published by the 02:00 run (see the context-layer item below and `NIGHTLY_LOG.md`).
    The lab's next day is Friday **2026-10-23** either way - 2026-10-16 is a retrospective - and the
    next experiment is still profitability (Novy-Marx 2013). No proposal is waiting for a decision:
    `write_brief.open_proposals()` returns `[]` since the Growth refusal.
  - **2026-10-13 (06:00 nightly) - the Growth proposal was reviewed and REFUSED, and the page now
    says why the weight is where it is.** The waiting proposal (Growth 13% -> 10%, Quality 22% ->
    25%) measured one point; the review measured the curve, which is what separates the two
    readings of the same evidence. Net growth exposure moves steadily with the weight, crosses zero
    at **13.5**, and stays within 0.05 of neutral only between **11 and 16** - because Growth is the
    one category of eight whose score runs *against* the others. So its weight is the dial that
    sets whether the finished ranking leans toward growing companies or away from them, and at 13%
    it leans neither way. The proposal's own argument is that growth earns nothing; if that is
    right the supported position is none, and 13 points is what delivers none. **Cutting to 10
    would not remove a position on growth - it would take one**, in a direction the four studies do
    not agree on (two flat, one negative but on 1968-1990 five-year market-wide growth, and MSCI's
    modern estimate slightly positive). The methodology page gained **"Why the weight stays at
    13%"**, with six tests behind it - including one that fails if a published run ever stops being
    growth-neutral, so the sentence cannot quietly go stale. The sound half of the proposal
    survives and moved: the category's *composition* (one metric duplicating momentum, one arguing
    with Investment about the same fact, one duplicating Quality) is now the open Growth question.
    **Found while wiring that up:** the publish gate's tripwire only noticed tests that read the
    committed methodology file, not ones that read the generator - which is the better way - so
    **three further modules holding public page claims were never run before publishing**. All four
    are in the gate now (454 claims-gate tests, up from 371).
  - **2026-10-13 (owner-run, before the 02:00 run) - the nightly sessions are set up to carry
    this on, and a ninth experiment answers the question the review left open.**
    `prompts/nightly.md` now tells each session about the lab, the route from a finding to a
    weight, the proposal waiting for review, and how to read a paper so a note can be trusted.
    The Growth proposal was **pre-flighted** in a throwaway checkout: eight tests trip, and the
    note lists what each needs. A full rehearsal of the 02:00 publish with tonight's code passed
    the whole suite. Experiment 9 reruns Israel & Moskowitz (2013) in the largest fifth of
    companies: the paper's figures there reproduce, and momentum since 2000 is 2.7% a year
    (t 0.54) - the flat stretch is in the largest fifth as well. **Your morning brief now has a
    "Waiting for a decision" section** listing any proposal a session left for review, with how
    many days it has been open - today, the Growth one.
  - **2026-10-13 (owner-run, just after midnight) - the second question is researched too: value
    and momentum among large companies.**
    `research/2026-10-12-value-and-momentum-among-large-companies.md`. Value on book value was
    never reliably there among the largest stocks, but the screener weights earnings and cash-flow
    measures, where the published large-company evidence is better, and nearly all of that premium
    is on the side a long-only book keeps. Momentum is strong in every published study and has
    been flat among big companies on the public data since 2000 - a longer, lower stretch than
    any on record, which no published study yet covers. **No weight change is proposed:** the lab
    and the literature disagree, and the rule written beforehand says the weight then stays.
    The methodology page now tells a reader all of this instead of quoting all-stock figures.
  - **2026-10-12 (owner-run, near midnight) - the first of the five questions is researched:
    Growth's 13%.** `research/2026-10-12-growth-category.md`, from six papers and MSCI's and S&P's
    methodology documents. No published return premium stands behind growth as the category
    measures it; index providers treat it as a style, not a rewarded factor; and in this
    screener it works as a counterweight - the published ranking does not lean toward growth at
    all (correlation -0.01). The methodology page now says so. **FOR THE NEXT SESSION, ahead of
    the rotation: review section 7 of that note - a proposal to move 3 points from Growth to
    Quality - and decide it.** Build it with a changelog entry, or refuse it and write why; the
    note lists both the build steps and the case against. It was not made the same night on
    purpose: the 02:00 run was the first under the rebuilt momentum rule, and the note's author
    should not be its only reviewer.
  - **2026-10-12 (owner-run, past 23:00) - your question, "will the lab be used to improve the
    methodology?", has a written answer: `plan/factor-lab.md`, "From the lab to the weights".**
    In short: yes, through research notes and the changelog, never through code; five questions
    about the weights are ranked there (Growth's 13% first - nothing stands behind it that can
    be rerun). Experiment 8 shipped with it - the factors **among large companies**, which are
    all the screener ranks: value, investment and momentum are about half-size there, and
    profitability is the one that holds. The lab's front page now shows where each category's
    weight stands on that evidence. **No weight changed.** The same experiment also shows what
    a holder who shorts nothing keeps of each factor among big companies: nearly all of value
    and investment, about half of momentum, **under half of profitability** (the rest is weak
    companies lagging, which a long-only portfolio benefits from only by not owning them).
  - **2026-10-12 (owner-run, the end of the night) - the front page compares every headline
    result inside its paper's sample with the years since** (9 of the 10 that claim a direction
    were weaker afterwards), and **two methodology sentences were corrected to what the lab
    measured** (Risk: extra risk has not been paid for, rather than "one of the most robust
    findings in finance"; Size: the measured ninth-minus-tenth difference beside the caveat).
    Also new and not about the lab: `scripts/rehearse_publish.py` - see CLAUDE.md "Ship gates" -
    after a rehearsal showed tonight's page would have made the 06:00 baseline red.
  - **2026-10-12 (owner-run, the last of the night) - experiment 7 shipped: does market risk
    pay?** Fama & French (1992) on the library's beta deciles: **7 of 11 within**, and the paper's
    point - returns flat from the lowest-beta group to the highest - reproduces, in its sample
    and since. The Risk category now has two of its three measures behind it. The methodology
    page links every category to its experiments. **Seven experiments, 64 of 94 figures within
    tolerance, six of eight categories covered.** Next for a Friday: profitability (Novy-Marx
    2013) or the cash-flow-yield sort; next for a Monday: the research notes that
    `research/2026-10-12-factor-lab-findings.md` asks for (value since 2007 first).
  - **2026-10-12 (owner-run, the same night) - experiment 6 shipped: does a high earnings yield
    pay?** Fama & French (1992) on the library's earnings-yield deciles, the nearest public sort
    to the screener's own valuation measures. **7 of 11 figures within** tolerances written
    beforehand; returns rise step by step with earnings yield in the paper's sample and for
    decades after. **Since 2007 the gap has all but closed, and among large companies it
    reversed** - the finding that matters most for the screener's biggest category, written up
    in the research note (section 7) as a question for a Monday, not acted on. Six experiments;
    57 of 83 figures within tolerance across them.
  - **2026-10-12 (owner-run, the same night) - experiment 5 shipped: do riskier stocks earn
    more?** Ang, Hodrick, Xing & Zhang (2006), the study behind the Risk category, on the
    library's volatility-sorted fifths. **5 of 11 figures within** tolerances written
    beforehand. The calmest fifth earned about as much as the rest with half the volatility; the
    paper's large penalty for the most volatile stocks does not appear on the public version of
    the sort. Built on a generic layout, so the next sorted-portfolio experiments (profitability,
    earnings and cash-flow yield, accruals) need only a record and a paper's table. Five
    experiments now; six of the eight categories have something behind them.
  - **2026-10-12 (owner-run, the same night) - a front page, and an honest line on the
    methodology page.** The lab opens on an overview: four cards (verdict and one tested sentence
    each; 45 of 61 figures within tolerance across them) and a table of what stands behind each
    of the eight categories - including the two that have nothing yet (Growth, Risk) and the one
    that probably never will (Analyst revisions: its studies need estimate history that is sold,
    not published). The methodology page gained limitation 11: the value, investment and size
    factors have averaged close to zero since 2014. The plan's queue now names the library files
    for Risk, for value measured the screener's way, and for accruals.
  - **2026-10-12 (owner-run, the same night) - experiment 4 shipped, and it is the lab's first
    replication that mostly fails:** Fama & French (1992) on size, **3 of 11 figures within**
    tolerances written beforehand, because the paper's table covers only non-financial firms
    with accounting data and the public file covers every stock. It is published as a miss, with
    the reason. The page also answers the screener's own question - is there a size effect among
    large companies? - for any period: the ninth size group has out-earned the tenth by
    0.07-0.14% a month every way it was cut, but only the full century clears the usual bar.
    The lab also gained a readout on every chart and links that reopen an exact scene.
    **Next: experiment 5, profitability (Novy-Marx 2013)**, then 2b.
  - **2026-10-12 (owner-run, the same night) - experiment 3 shipped:** Fama & French (2015), the
    five factors behind the Valuation, Quality, Investment and Size categories. **20 of 21
    figures within tolerances written beforehand.** After the paper's sample (2014 on) none of
    the four factors is distinguishable from zero, and value is almost fully explained by the
    other four in both periods. `research/2026-10-12-factor-lab-findings.md` sets out what the
    three experiments mean for the screener and what they do not - it is the "dig deeper"
    starting point, and it asks for a Monday research note on what the value and size premia
    are now worth. **Next: experiment 4, a size premium among large stocks**, then 2b.
  - **2026-10-12 (owner-run, later that night) - experiment 2 shipped**, after the owner's "keep
    improving, whatever you think would benefit these tools, be creative and start making this
    great!": Daniel & Moskowitz (2016), "Momentum crashes" - the study the momentum regime rule
    is built on. **9 of 15 figures within tolerances written before the comparison was run**;
    every figure about what follows a market decline matches, and the six misses all describe
    the portfolio itself (the paper builds its own deciles). The page sorts each month into four
    market states and shades the panic state on the chart: at the screener's own thresholds,
    1927 to 2013, momentum averaged -1.68% a month in that state and +1.52% outside it. The lab
    now switches between experiments. **Next: experiment 3, the Fama-French five factors**, and
    2b in the plan (rerun experiment 2 on the authors' own portfolios, terms permitting).
  - **2026-10-12 (owner-run) - experiment 1 shipped:** Barroso & Santa-Clara (2015),
    risk-managed momentum, on the Kenneth French library's momentum deciles. **13 of 14 figures
    within tolerance** (Sharpe 0.54 -> 1.00 here against the paper's 0.53 -> 0.97); the
    risk-managed worst month is outside (-24.16% against -28.40%) and stays marked. After the
    paper's sample (2012 on) the Sharpe ratios are 0.20 and 0.37. `factor_lab.py`,
    `lab_engine.js`, `lab_data.js`, `lab.html`, `tests/test_factor_lab.py` (15).
    **Next: experiment 2, Daniel & Moskowitz (2016), momentum crashes** - the study the momentum
    regime rule is built on. Write its tolerances before running the comparison.

- **2026-10-08 - The context layer: technicals, options, insiders, macro and a track record.
  First draft built by an owner-run session; the nightly sessions now own making it right.**
  Owner, verbatim: *"This is where I want people to come for all of their investing needs ... maybe we should take like technicals into account for timing of investments? Or macro data? Or options data? ... the goal is still long term ... if free somehow ... you would just build the first rough draft of it, then the nightly sessions would really drill into it, and make sure that it is perfect moving forward."* He chose all four (track record, a per-stock "Before you
  decide" panel, a market backdrop, insider buying) and chose **"context only, track them"**: shown
  beside the score, never in it, recorded every run so each signal builds an out-of-sample record.

  **The brief is `plan/context-layer.md` - read it first.** Its "Known limits of the draft" list is
  the queue, in order: (1) measure that the context pass costs the core fetch nothing; (2) the share
  of stocks with usable option quotes at 2 AM - **measured and fixed in code 2026-10-09**;
  (3) insider data from the SEC's own filings - **done
  2026-10-08 (late)**: the owner gave a contact email for the SEC's User-Agent, which lives outside the
  repo in `data/sec/user_agent.txt` (gitignored); never copy it into a tracked file; (4) one research note per signal;
  (5) the evaluation harness over `data/context_log/`; (6) track-record hardening; (7) macro data
  vintages. This is product and data work on the new layer - it does **not** reopen the closed
  redesign, and it never moves a score (CLAUDE.md row "ctx").

  **Progress (update this line each session):**
  - **2026-10-15 (13:00 slot) - deferred, and why, with nothing lost.** The session carried on from
    the owner-run midday entry's Next instead: a measured input defect left open in
    `research/2026-10-14-yahoo-vs-sec.md` section 4b, where the screener invented a quarter the
    source had never published (Berkshire's net income was scored 14.5% below its own filing). It is
    fixed and exact against the filings, and wiring its sentence onto the page turned up a live
    defect - the drilldown function those badges live in is **never called**, so this morning's
    quarterly-rejection badge could never have appeared. **Next for this item is unchanged:**
    item 4's two remaining per-signal notes (macro backdrop, track record), then item 6,
    track-record hardening. Nothing here was started, so nothing is half-done.
  - **2026-10-14 (13:00 session) - queue item 4's three open page-sentence findings are DONE, and
    reading the paper turned up a correction the next session has to carry.** All three were the
    same failure - a figure from a paper printed beside a number the paper did not measure.
    **(a)** The put skew is now Xing, Zhang & Zhao's own construction (the out-of-the-money put
    nearest 95% of the price minus the **at-the-money call**, not the mean of call and put at 90%),
    with both implied volatilities printed beside it so the subtraction is checkable on the row.
    The old measure was their skew *minus half of a different published signal*; measured on 25
    names from the same option chains, the two **disagreed on the sign for 7 of 22 (32%)**, and the
    old one produced +46.6 IV points for one stock where the paper's own 95th percentile is +19.92.
    **(b)** The sentence now quotes the paper at the horizon a monthly reader can act on - 10.9% a
    year held one week **and 6.5% held four** - and gives McLean & Pontiff's 58% post-publication
    shrinkage as an upper bound. **(c)** Put/call open interest is off the page: it was a bare
    *level* with no sentence, where the evidence is about flow and is spent within a week. It is
    still logged, so its record keeps accruing.
    **The correction:** the 10-12 note concluded that *every* option-implied signal decays faster
    than this screener acts, listing XZZ as "weekly rebalanced" - it had the paper second-hand. The
    paper's Table 4 measures the same spread at a **four-week** holding period (6.52% a year,
    t 2.70) and out to 28 weeks with *rising* t-statistics. Withdrawn for XZZ, still standing for
    the other three studies. It does **not** make the skew a candidate for the score, but the first
    `1m` record from `context_eval.py` (window closes 2026-11-07) must be read as a test of a claim
    the paper makes at that horizon - and the five log dates before today hold the old
    construction, so they are excluded rather than spliced.
    **No score moved** (rehearsal: `dashboard_data.js` byte-unchanged; context payload changed 0,
    added 0, removed 941 - the withheld old skews and the open-interest row).
    **Next: item 4's two remaining per-signal notes** - the macro backdrop and the track record -
    then item 6, track-record hardening.
  - **2026-10-14 - deferred, and why.** The 06:00 session's test baseline was red: the source had
    stopped serving one `.info` field and the dividend yield had emptied for 495 of 501 stocks in the
    run published at 02:00, which also put "Dividend Yield: None" on the page for Coca-Cola, J&J and
    Procter & Gamble. A failing gate outranks this queue, so the session spent itself on that, on the
    three further defects it exposed, and on giving the 02:00 loop a gate that would have caught it.
    **One thing landed for this item anyway:** Xing, Zhang & Zhao (2010) was obtained and read from
    the paper, so the put-skew correction in item 4 now has its definition from the primary source
    (`SKEW` = OTM put at 0.95 moneyness minus **ATM call**, 10-60 day options) rather than
    second-hand. **Next is unchanged:** item 4's remaining per-signal notes (macro backdrop, track
    record) and its three open options-panel sentence corrections, then item 6.
  - **2026-10-12 - queue item 2 is CLOSED, and closing it caught a false statement on the live site.**
    Today's 02:00 run was the first scheduled one after a 20:00 refresh: options `ok` **397 of 503**,
    `partial` 59, `quotes-closed` 47, against **0 of 503** before the fix. The after-close cache works.
    But the quotes were stamped **Saturday 2026-10-10** and the page told readers of **455 of 501**
    stocks "Quotes are the close of Oct 10" - the refresh runs at 21:00 ET, already the next UTC day,
    and it stamped the UTC date, so every refresh had named the session *after* its quotes. Fixed
    (`market_session_date`, plus one `is_session_date` predicate shared by the cache and the
    generator), the cache and the run's copy repaired, and the site republished the same session:
    `dashboard_data.js` **byte-identical**, `dashboard_context.js` 455 changes, **all of them the
    date**, and no panel lost. **Queue item 4 advanced:** the options panel's per-signal note is done
    (`research/2026-10-12-option-implied-signals.md`) and concludes the option signals are **not**
    candidates for the score - every supported one decays faster than a monthly review cadence - with
    three page-sentence corrections left open in `plan/context-layer.md` item 4.
    **Next: item 4's remaining per-signal notes** (the macro backdrop and the track record have none),
    then item 6, track-record hardening.
  - **2026-10-09 (owner-run, afternoon) - a new signal, and the evaluation's sector input repaired.**
    "Against its own five years" (each stock's earnings and FCF yield against its past 60 month-ends,
    from SEC filings; `valuation_history.py`) joined the panel and the context log as two signals
    (`_ctx_vh_ey_pct`, `_ctx_vh_fy_pct`) that `context_eval.py` evaluates like the rest. A review found
    the logs had never carried the GICS sector, so the sector-relative signal was always empty; both
    committed logs were repaired. A `--tickers` run no longer rewrites the log or the track record.
  - **2026-10-09 (owner-run, daytime) - the outstanding step is done, and items 4-6 moved.** The
    `Screener Option Quotes` task is registered and verified (`Get-ScheduledTask`: one weekly 20:00
    trigger, no logon trigger; loops still PT3M / PT20M) and a live smoke test wrote the cache at
    08:46 CT. Item 5 (the evaluation harness) is built - `context_eval.py`, a line in the morning
    brief, first window closes 2026-11-07. Item 4 step one: every research sentence on the panel was
    checked against its source, three corrected (`research/2026-10-09-context-panel-claims.md`).
    Item 6 progressed: thin cached price columns are repaired, and picks the price source no longer
    serves are named on the page. **Next: confirm the `ok` share from the first 02:00 log after a
    20:00 refresh, then item 4's per-signal research notes.**
  - **2026-10-09 - item 2 measured and fixed in code; one machine step outstanding.** The options
    panel was empty for **every stock on every scheduled run**: `ok` for **0 of 503** at the 02:00
    loop's hour against **394 of 503** at 21:27 ET the evening before, because the source serves the
    chain overnight with every bid, ask and implied volatility at zero. Fixed by collecting quotes
    after the close (`options_cache.py`, a new weekday-20:00 task defined in `register-tasks.ps1`)
    and having the 02:00 pass **probe** rather than guess the hour - it now spends 3 option requests
    instead of ~1,000, reads the cache, and the page names the session the quotes came from.
    **Outstanding:** the task was not registered on the machine - every PowerShell call, including a
    read-only `Get-ScheduledTask`, is auto-denied in a non-interactive session, so this was out of
    reach rather than skipped (rule 11). Next session: run `scripts/register-tasks.ps1`, confirm all
    three tasks with `Get-ScheduledTask`, then re-measure the `ok` share from the next 02:00 log.
    **Then item 2 closes and item 4 (one research note per signal) is next.** The exact verification
    steps are in `plan/context-layer.md` item 2.

*(No open owner items. Calculation transparency and the premium redesign were finished on
2026-10-07 in owner-run sessions; their text is archived under **Done**. Owner direction that
night: *"I actually decided that I want you to finish up all UI/UX work in here tonight. And make
the nightly sessions just focus on what it was previously focusing on, making sure the methodology
is sound. And improving this tool overall every single night."* Nightly sessions return to the
weekly rotation: research-led methodology and overall improvement. The residuals that are data or
methodology, not design, are in CLAUDE.md "Current priorities": 0.9 (four methodology questions)
and the inputs the 11 history-based metrics need kept at fetch.)*

---

## Done

Completed items, newest first.

- **2026-10-15 - the capex proposal: REVIEWED and REFUSED, and the page now says what the capex
  figure contains.** Raised 2026-10-14 (owner-run) and decided by the next session, as the routine
  requires. The proposal would have taken a captive finance arm's lease purchases out of
  free-cash-flow capex for GM, Deere, Caterpillar and PACCAR - GM's FCF yield 0.3% -> 8.0%, its
  sector percentile 10 -> 90, its composite up 7.9 points on the heaviest metric in the screener.
  **It was refused because its own sources do not construct the ratio that way.** GM's fiscal 2025
  10-K defines adjusted automotive free cash flow from **net automotive** operating cash flow of
  $18.7B; the consolidated figure the screener uses is $26.87B. The proposal removes the finance
  arm's capex and keeps its $8.17B of operating cash flow, which would publish $17.57B of free
  cash flow against GM's own $9.5B - **85% too high**, where today's figure is 81% too low. The
  whole lease cycle, from GM's own tags: $15.79B of vehicles bought, $7.80B of rentals collected
  (inside operating cash flow, so the proposal keeps them), $10.10B received when the vehicles are
  sold at lease end (counted by neither construction). **The clean ratio cannot be computed from
  free data** - a segment-level operating cash flow is MD&A prose, not an XBRL fact - so what
  shipped is the honest alternative: every affected stock's FCF-yield row now says, from its own
  filings, how much of its capex is assets bought to lease out and that its own free-cash-flow
  figure is the industrial business only. Section 5 of `research/2026-10-14-capex-definition.md`
  has the measurements and the two things that would reopen it.

- **2026-10-07 - Premium redesign and calculation transparency: finished.** Owner-run, five
  passes in one evening, verified in a real browser at 1440, 375 and 320px and live on the
  public site. Shipped, beyond the first version: search palette (Ctrl/Cmd+K), stock links,
  J/K stepping, Compare with the composite gap decomposed, workings CSV, Add to Holdings from the
  sheet, analytics that filter the table, every metric's calculation one click away with its
  equation checked against the engine for every stock, reasons for unused metrics, a real rank
  history chart with the ordinary-variation band, brand mark / favicon / page title / meaningful
  stat strip / footer, focus trapped in every dialog, AA contrast on every surface, keyboard
  navigation of the rankings, the shell painting before the data lands (LCP 4.8 s -> ~0.3-1.1 s
  at 10 Mbit/s), and fixes found by looking: the sheet lifting its own header 164px, analytics
  cards running 44px off phone screens, the drilldown's nav order, Peers' green/red verdicts,
  a history rank 1-3 off for 284 stocks. Nightly sessions now return to methodology and overall
  improvement (owner direction, same night).

  <details><summary>Archived text of the two finished items, as they stood when closed</summary>

- **2026-10-06 — Show the numbers that go into every score, and make the
  arithmetic provably right. This builds trust; spend real sessions on it, in
  step with the premium item below.** Owner, verbatim: *"I want it to have more
  data integrity, or calculation integrity - we can see how things score, but we
  don't actually see the numbers going into any calculations in the breakout
  details for each company. This will add trust to the screener."*

  **The full plan is `plan/calculation-transparency.md` - read it first.** It holds
  the goal, the design principles, seven stages (T0a, T0b, T1-T6), the tests and
  the sources. The short version: a reader should be able to follow any number from
  the composite, down through category score, metric percentile and raw value, to the
  company's own reported inputs - see every weight and peer count on the way - and
  recompute it in a spreadsheet and get the same answer.

  **Why this is urgent and not just a feature - three defects, measured on the live
  site 2026-10-06 while writing the plan** (reproduce with the two scripts in
  `research/measurements/2026-10-06-*.py`; do not trust these, re-run them):

  1. **The per-metric weights on the drilldown are not the weights used, for 276 of
     502 stocks (55%).** The page prints the generic configured weight; the engine uses
     bank weights, Piotroski-conditional weights and per-stock renormalisation.
     334 of 4,012 stock-category pairs cannot be reproduced from what is on screen.
     JPM's Valuation panel shows three heavily-weighted metrics as N/A, calls P/B
     (the bank's main metric) "Inactive", and prints a score no arithmetic on the
     page produces. Same class of bug as 2026-08-28.
  2. **"Composite = sum of the points" is false for 2 stocks** (FDXF, L): a coverage
     discount is applied after the weighted average and never shown.
  3. **The first sentence of every drilldown is false.** *"Its composite of 73.7 is a
     percentile: it scores above 74% of the universe"* - for the stock ranked **1st**.
     Off by a median of 19.6 points, by more than 10 points for 75% of stocks. The
     composite has been cardinal since Phase 13; the README, the generated methodology
     page's Step 5 and one plan doc still say percentile, while the same methodology
     page's Limitation 8 says the opposite.

  **Order, and why it outranks the design stages:** these are false statements on a
  public site, so **T0a (fix the false sentences, build the claims register) then T0b
  (true weights, reproducibility test, build-time refusal to publish a page whose
  arithmetic does not add up) come before any presentation stage.** Then T1 (lineage
  registry), then T2/T3 (inputs and percentile context in the payload), then T4 built
  *together with* the drilldown redesign (design stage D3 - do not style the old
  category tables twice), then T5 (an independent checker and a gate) and T6
  (per-input provenance).

  **Constraints that do not move:**
  - **No scoring change of any kind.** This is explanation, not methodology; nothing
    here may be justified by the backtest (rule 5) or the thin IC series (rule 4).
  - **The page shows the engine's numbers, never its own re-derivation.** Refactor
    `compute_category_scores` so weight resolution is one function used by scoring and
    export - a second copy in the generator or in JS is how defect 1 happened.
  - **Say what cannot be shown** ("provider-supplied", "computed from 252 daily
    closes") rather than print a formula that does not reproduce. A false equation is
    worse than none.
  - **Payload budget:** <= +150 KB gzipped inline, else per-ticker shards; measure
    after gzip and log it. Decision support, not advice: `BANNED_TERMS` applies to
    every new sentence. Display-only: inputs enter no metric, weight or rank.
  - **Any discrepancy the new checks find is reported, not smoothed over** - in the
    log and `METHODOLOGY_CHANGELOG.md`. Finding them is the point.

  **Progress (update this line each session):**
  - **2026-10-06 - plan written, nothing built yet.** Next: **T0a**.
  - **2026-10-07 - T0a shipped.** Defect 3 is fixed everywhere and can no longer
    come back. The drilldown's first sentence now reads *"Ranks 1st of 502 -
    ahead of 100% of the other 501 stocks. Its composite of 73.8 is a 0-100
    score computed from its 8 category scores and their weights, not a
    percentile"* - the share is `(N-rank)/(N-1)`, exact for all 502. The plan
    named four places saying the composite was a percentile; there were **six**
    (also `FORENSIC_AUDIT_REPORT.md`, and **a test that asserted the false
    claim**, which is why it survived). `claims.py` + 24 tests now make an
    unregistered or unchecked claim fail the build, and the data loop's publish
    gate runs them. Every number, rank and other sentence byte-identical:
    181,446 payload leaves, **502 changed (all this sentence), 0 added, 0
    removed**; **+309 bytes gzipped** against a 150 KB budget. Dashboard and
    methodology page regenerated, so it is live now rather than at 02:00.
    **A fourth defect was found and recorded, not fixed:** "rests on N of 18
    metrics" and the provenance badge's 60/80% colours use a hard-coded
    18-metric list, not the applicable-metric coverage the discount reads (35
    for a bank, 41 otherwise) - 62 stocks read under 80%, 3 were discounted.
    **Next: T0b** (true per-stock metric weights from the engine, the
    reproducibility test, the build-time refusal to publish) - which now also
    owns that fourth defect. `scripts/diff_payload.py`, which the plan assigned
    to T0b, is already built.
  - **2026-10-07 (owner-run session) - T0b, T1, T2, T3, T4, T5 and most of T6 are built and
    shipped, at the owner's request ("make all improvements right now ... then the nightly
    sessions do an even deeper pass").** The page now prints the weights the engine used
    (bank / Piotroski-conditional / rescaled), every metric opens to its formula, the
    figures behind it, whether they rebuild the value, and who it was ranked against; the
    nine Piotroski signals and eight Beneish indices are listed; the composite chain shows
    the coverage discount. **4,010/4,010 category scores and 502/502 composites rebuild from
    the payload; the build refuses to publish otherwise; `scripts/audit_stock.py --all`
    independently reproduces 502/502.** 24 metric equations rebuild at 99.5-100%. Payload
    gzipped is flat (1,268,733 B vs 1,278,885). **The deep pass - read
    `plan/calculation-transparency.md` "Status, 2026-10-07 evening" for the table - is, in
    order:** (1) the "download this stock's workings" CSV; (2) add the missing inputs at
    fetch (analyst per-quarter EPS, the risk-free rate and market return) so the last 12
    metrics can show an equation; (3) **research the four findings the build surfaced**
    (open item 0.9 in `CLAUDE.md`) - negative `operating_leverage` ranking best is the one
    with a measured effect; (4) put `audit_stock.py --sample 25` in the morning brief.
    **(1) shipped in pass 2 the same night** (Download as CSV).
  - **2026-10-07 (late, owner-run pass 3) - the numbers are on every row, not behind a
    click.** Owner, with a screenshot of the Valuation workings: *"there is nothing like
    showing the actual numbers going into any scores ... make sure the nightly sessions know
    that if something is changed with scoring, that it also updates on there, the frontend
    side of things to match the backend."* Each metric now carries a second line, e.g.
    "$4.5B free cash flow ÷ $31.0B enterprise value = 14.4% · 1st of 46 in sector, median
    4.0% · 100.0 x 45%"; a two-line card on a phone. The equations are templates
    (`metric_lineage.EQUATIONS`) the suite **evaluates against every stock's scored value** -
    19 exact ones reproduce 100% - so a formula changed in the engine without the page fails
    the build, and the data loop's publish gate runs the same check. Standing rule:
    CLAUDE.md 0.8c, `prompts/nightly.md` section 3, `DECISIONS.md` 0.8c.
  - **Commit after every stage.**

  **The test:** a student picks any stock, opens it, and can answer *"where did this
  63 come from?"* three levels down without leaving the panel; downloads the workings;
  a spreadsheet agrees with the page to the displayed precision; and the 02:00 data
  loop would refuse to publish a day on which that stopped being true.

- **2026-10-05 — Make the dashboard look and feel premium. Spend a lot of the
  next two weeks on it.** **The surface-by-surface plan is
  `plan/dashboard-redesign-master.md` (18 surfaces, stages D2-D8, measured budgets,
  order of work) - read it first; the brief below is the owner's words and the
  original measurements.** The calculation-transparency item above takes the first
  code session (T0a, T0b) because those are correctness defects; after that the two
  items interleave as the master plan's order table sets out, meeting at the
  drilldown. Owner, verbatim: *"It looks like AI slop. Make it look
  premium and expensive. Right now it does not look good, and it doesn't feel
  good to move around in it either."* Both halves count: how it **looks** and how
  it **feels to use**. This outranks the day's rotation focus on every day it can
  be worked, Mon-Fri, until the test at the bottom is met; the retrospective
  decides when it is done.

  **What was measured on the live site, 2026-10-05** (the documented failure the
  evidence rule asks for; re-measure before and after, do not trust these):

  - **The palette is GitHub's dark theme, unchanged** - `#0d1117` / `#161b22` /
    `#21262d` / `#e6edf3` / `#7d8590` / `#58a6ff`. That is the default look of
    generated dashboards, which is what "AI slop" is picking up on.
  - **Colour is spent everywhere, so it means nothing.** A rainbow gradient bar
    across the header, blue-gradient headers on the Top 5 cards, eight
    saturated bars per card (one hue per category), coloured sector pills,
    green/red/orange/purple text. Premium reads as restraint.
  - **The type is small and mono-heavy.** 12px is the size of ~2,100 text
    elements and 13px ~770; JetBrains Mono sets **~2,560 elements** - nearly
    every label and number - alongside two other families. Three typefaces with
    no clear role each.
  - **The rankings table is the main thing people use and it is the weakest
    surface.** All 502 rows are in the DOM (~9,900 nodes on the page), the table
    scrolls *inside a box inside the page* (nested scrolling is the single
    worst "doesn't feel good" culprit), scores are bare numbers with no visual
    encoding, and the filters are unstyled native `<select>` controls.
  - **The drilldown has no hierarchy.** "Why it ranks here" is the point of the
    tool and renders as ~11 sentences of identical weight in one block.
  - **Mobile is an afterthought.** Three KPI cards in a two-column grid leave an
    orphan, the header card consumes the first screen, and Top 5 is a sideways
    carousel.
  - **Motion is everywhere and unconsidered** - ~660 elements carry transitions.

  **Direction, not a spec** - you are the designer; argue for what you change:

  1. **A real design system first, in the generator, once.** One neutral scale,
     **one** accent, and colour reserved for *meaning* (better/worse, trap flag,
     stale data) rather than decoration. A deliberate type scale (body 14-15px,
     tabular numerals for every figure, at most two families, each with a job),
     a spacing scale, hairline borders, one radius and one elevation system.
     Tokens as CSS variables so every later change is cheap.
  2. **Then the surface people live in: the rankings table.** Sticky header,
     comfortable row height, proper hover / selected / keyboard states, scores
     shown with restrained visual encoding rather than bare digits, designed
     filter controls, **no nested scroll box**, and rendering that does not put
     502 rows' worth of work in front of the first paint.
  3. **Then Top 5 / KPIs / My Holdings, then the drilldown** (give the summary a
     headline and a hierarchy; a side panel or a well-built sheet, not a dimmed
     wall of text), **then mobile as a first-class layout**, then a polish pass.
  4. **Feel is measurable.** Interactions respond in well under 100ms, motion is
     short and ease-out and honours `prefers-reduced-motion`, scrolling is
     smooth, nothing shifts under the cursor. Record DOM node count and a
     scroll/click responsiveness number before and after.
  5. **Look at it every session.** Open the live page in the browser pane at
     desktop *and* 375px, before and after each change, and say in the log what
     you saw. A design change judged only from the CSS is how this got here.
     References worth studying as documented practice, not copying: Linear,
     Stripe Dashboard, Vercel, Mercury, Apple's financial surfaces.

  **Rotation, so the week still works:** Monday's research day is a design note -
  what premium financial UI actually does and why (type, colour, density, data
  tables) - with sources. Wednesday's synthesis checks that the new look still
  *explains* the eight categories and the sector-relative percentiles correctly.
  Thursday builds. Tuesday and Friday are the product and teach days anyway.

  **Constraints that do not move:**
  - **No number, score, rank or sentence changes meaning.** This is presentation.
    Diff the rebuilt payload against the live one and say it is identical.
  - **Decision support, not advice** - nothing that reads as a recommendation,
    and `BANNED_TERMS` stays enforced. Defensibility features stay (rule 7).
  - **15 test modules pin dashboard markup, ids and JS function names**
    (`test_dashboard_surfaces`, `test_holdings_panel`, `test_stock_summary`,
    `test_dashboard_js` with its `node --check`, and more). Prefer CSS and layout
    changes. If a pinned hook has to move, update its test in the same commit and
    say why - never delete an assertion to get green.
  - **Stay a single static page built by `generate_dashboard.py`** - no
    framework migration, no build step, no new runtime dependency. Edit the
    generator, never the generated files (rule 10).
  - **Keep payload weight flat** (it is ~1.2 MB gzipped; measure after gzip) and
    keep working offline from the file once loaded. If you load webfonts, choose
    them deliberately, subset them, and make the fallback stack look right.
  - Update `plan/dashboard-inventory.md` in the same session (rule 9).

  **Progress (update this line each session):**
  - **2026-10-06 - stage 1 shipped:** design-system tokens in the generator, with the
    written argument in `plan/dashboard-design-system.md` - one neutral ramp and one
    accent, Inter with tabular figures, 14px base, rainbow/glow/entrance-animation
    removed, `color-scheme: dark`. Payload byte-identical. **Next: stage D2, the
    rankings table** (nested scroll box, 502-row DOM, native selects, score
    encoding) - after calculation-transparency T0a/T0b. Baseline to beat: 9,928 DOM
    nodes (8,050 of them in the table), 662 transitioned elements.
  - **2026-10-06 (evening) - full plan written:** every surface audited at 1440 and
    375px and listed with what is wrong and the target; stages D2-D8; budgets for DOM
    nodes, click-to-paint, LCP, CLS and contrast. One finding to act on early: the
    public **Refresh Data button is dead for every visitor** (it connects to
    `localhost:7720`).
  - **2026-10-07 (owner-run session) - a first version of D1 through D7 is built and
    shipped, verified in a real browser at 1440 and 375px.** Measured: DOM nodes **9,928 ->
    2,779**, transitioned elements **662 -> 99**, row-click to paint **70 ms**, sort to paint
    **21 ms**, layout shift **0.001**, payload gzipped flat. `tests/test_dashboard_browser.py`
    (25 tests) holds those budgets. **The deep pass is the work listed per stage in
    `plan/dashboard-redesign-master.md` "Status, 2026-10-07 evening"** - in this order: (1) the
    lower drilldown blocks (Rank History, Price Targets, Company Snapshot, Sector Peers) were
    re-chromed, **not redesigned**, and Peers still colours cells red/green against the stock;
    (2) **look at My Holdings populated** at both widths - never re-inspected; (3) the
    sector matrix has no phone treatment; (4) one shared chart module and alt text; (5) one
    "no data" / "not applicable" vocabulary and a formatting module; (6) D8: a full keyboard
    pass, a real focus trap in the sheet, contrast on **every pair in use**, LCP, 320 and 414px,
    and a side-by-side against Linear / Stripe / Vercel / Mercury. **Be critical:** a first
    version built in one sitting is not "perfect" and the owner asked for perfect - find what
    is still not premium and fix it, and say what you saw in the log.
  - **2026-10-07 (late, owner-run pass 2) - owner, verbatim: *"keep improving the
    UI/UX/Frontend please, it looks alot better now! But do as much as you can right now ...
    and for the fine tunups give that work to the nightly sessions to do and test. Be
    creative!"*** Shipped, presentation only (payload byte-identical): a **search palette**
    (Ctrl/Cmd+K, any stock or section from anywhere); **stock links** (`#stock=TICKER`
    opens it; the back gesture closes the sheet); **J / K stepping** through the table's
    current filter and sort; **Compare** (up to four, side by side, with the composite gap
    taken apart into category points - registered as claim `compare.composite_gap`);
    **Add to Holdings** from any drilldown; **Download as CSV** of a stock's whole workings;
    the drilldown **reordered to match its own jump links** (they skipped four sections);
    **Peers lost its green/red verdicts** (it judged "better" on rules the screener does
    not use, e.g. dividend yield) and gained a peer median; trap rates as HTML bars (the
    canvas clipped sector names); sector matrix and trap bars **filter the rankings** on
    click; section previews on collapsed headers; a dismissible first-visit guide;
    shortcuts sheet (`?`); price targets, snapshot and populated Holdings de-boxed; phone
    filter bar, score grid and contribution rows fixed (two lost the cascade). 18 new tests
    in `tests/test_dashboard_navigation.py`; 3 payload tests for the gap claim in
    `test_calculation_reproducibility.py` (in the data loop's publish gate).
    **The nightly fine-tuning list is now `plan/dashboard-redesign-master.md` "Status,
    pass 2"** - it supersedes the six items above where they overlap. Test what shipped
    tonight first (palette, links, J/K, compare, CSV) on the live site at 1440 and 375px.
  - **Commit after every stage** - the 2026-10-06 session was cut off by a usage
    limit with nothing committed (see `NIGHTLY_LOG.md` 2026-10-06).

  **The test:** the owner opens it and it *feels* like a product someone paid for -
  calm, confident, fast, and obvious where to look. If you would be
  embarrassed to screenshot it next to Linear or Stripe, it is not done.

  </details> The session moves them here with the date and a
pointer to what it did, so this file doubles as a record of what you asked for
and what actually happened.

- **2026-08-26 — Remove the model portfolio.** Removed from the dashboard: the
  Model Portfolio section, its payload key, the sector-allocation chart, and
  the S&P sector-weight table that only that chart used. Top 5 now reads the
  ranking directly and shows the same five names. The `portfolio_constructor`
  engine and its Excel sheet were **kept** — see `METHODOLOGY_CHANGELOG.md`
  2026-08-26 (evening) for why, and say the word if you want those gone too.

- **2026-08-26 — Add "about" sections to the stock drilldown.** Each stock's
  detail view now opens with a plain-English description of what the company
  does, plus its specific industry. Sourced from the same Yahoo Finance
  response the screener already downloads, so it costs no extra requests.
  Descriptive only: never scored, never ranked.

---

## Notes on what this file is for

Good items are about **what the tool should do for you** — a surface that
doesn't help, a question you can't answer, something you find yourself checking
elsewhere. Those are things only you can know.

You do not need to file methodology work here. Deciding what a factor is worth,
which metrics overlap, and how the eight categories fit together is what the
research rotation is for, and it is driven by published research and
professional practice rather than by requests. If you *want* a specific factor
researched, though, say so — that is a legitimate item.

- **2026-08-26 — Layout: Top 5 first, two sections collapsed.** "What Changed"
  moved below "Top 5 Stocks"; both it and "Factor Analytics" now start
  collapsed, so the landing view is the Top 5 plus the full table and
  everything else is one click away.

- **2026-08-26 — Full fresh run to load the descriptions.** Ran with a forced
  refetch; 501 of 502 stocks now carry a real business description on the live
  site.

- **2026-09-01 — General smoothness/reliability pass.** Audited scheduled
  tasks, repo hygiene, and stale documentation claims. Fixed a permanent false
  "High severity" alarm in the data-quality log (bank-only metrics scored
  against the wrong population), closed a synthetic-data-fabrication gap in
  `factor_engine.py`'s own entry point, and made stray git branches from
  interrupted runs self-clean instead of silently accumulating.
