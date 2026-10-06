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

- **2026-10-05 — Make the dashboard look and feel premium. Spend a lot of the
  next two weeks on it.** Owner, verbatim: *"It looks like AI slop. Make it look
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

  **The test:** the owner opens it and it *feels* like a product someone paid for -
  calm, confident, fast, and obvious where to look. If you would be
  embarrassed to screenshot it next to Linear or Stripe, it is not done.

---

## Done

Completed items, newest first. The session moves them here with the date and a
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
