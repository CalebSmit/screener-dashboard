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

Add items below. Anything under this heading is unclaimed work.

<!-- Add items here, newest at the top. Free text, one item per bullet. -->

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
