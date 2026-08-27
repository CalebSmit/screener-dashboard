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

- _(nothing open — the queue is empty)_

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
