# Morning Brief - Tuesday 08 September 2026, 06:30

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-09-08T02:00:04.268668 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | HST, EXPE, APA, CAH, VLO |
| Evidence for weight changes | 3 of 8 needed at the 1m horizon (8 rows, but overlapping windows are not independent; 34 rows across all horizons), newest 2026-09-01 |

## What changed in the repo

- `402708b docs: record the chat removal and keep the plan files true`
- `3d80a9b data: regenerate the published dashboard without the chat`
- `9680674 product: remove the "Screener AI" chat, render the summary in its place`
- `388c614 product: deterministic per-stock "Why it ranks here" summaries`
- `f200b41 brief: data run 2026-09-08`
- `c72d9c1 data: screener run 2026-09-08 - 502 scored, top: HST EXPE APA CAH VLO`
- `51cfa15 brief: code session 2026-09-07`
- `cde9089 log: 2026-09-07 research session - the Revisions category has no revisions`
- `7de239c docs: correct the false "revisions data requires FactSet/Refinitiv" claim`
- `d09e7bc research: the Revisions category contains no revisions`
- `2aa2102 brief: data run 2026-09-07`
- `e2afbe4 data: screener run 2026-09-07 - 502 scored, top: HST EXPE APA CAH VLO`

## The session's own account

> 2026-09-08 - PRODUCT. Open the live dashboard as a user would. Does it answer what should I look at / should I buy this / should I sell what I hold / how much? Read plan/dashboard-inventory.md before building anything - the most likely failure is rebuilding what exists. Ship a dashboard change, or write down precisely what it cannot answer and why.
> 
> ### Health numbers (rule 8, all five)
> 
> | Check | Reading |
> |---|---|
> | Last code session ran? | `logs/nightly-2026-09-07_060001.log` - "Run complete: shipped to main" |
> | Data loop published? | `logs/datarun-2026-09-08_020002.log` - "Data loop complete", HEALTH: PASS, 502 scored |
> | Evidence base | **34 rows, newest 2026-09-01, 3 effective observations at `1m`** (8 raw) against a gate of 8 |
> | Priority 0 | DONE 2026-08-24, not reopened |
> | Top open roadmap item | **Priority 4, per-stock summaries - owner directive 2026-08-10, open 29 days. Taken and shipped today.** Next up is priority 5, sell-side workflow (north-star gap 2, 2026-08-05, **34 days**) |
> 
> **Tests:** before 965/965, after **1117/1117** (no pre-existing failures; +152 tests)
> **Data loop:** healthy. Evidence base moved 33 -> 34 rows, newest 08-31 -> 09-01.
> **Owner queue / rotation:** `OWNER_FOCUS.md` **Open** is empty. Nothing deferred.
> ISO week 37, Tuesday - product day, taken as the focus. The top open roadmap
> item happens to *be* a product item, so for once the rotation and the queue
> pointed at the same work.
> 
> ### Did
> 
> **Shipped priority 4: the "Screener AI" chat is gone and every stock's
> drilldown now opens with a deterministic "Why it ranks here" block.** Owner
> directive 2026-08-10, open 29 days. `METHODOLOGY_CHANGELOG.md` 2026-09-08.
> 
> This is the first north-star item to ship since 2026-08-25. The 2026-09-04
> retrospective added the roadmap-age line to rule 8 precisely because nine
> consecutive sessions had produced real work and no north-star item; writing the
> age down is what made "29 days" visible at the moment the day's focus was being
> chosen.
> 
> **Removed** (891 lines: 80 HTML, 583 JS, 228 CSS; `generate_dashboard.py` is
> 921 lines shorter): the chat FAB and panel, the Chat Settings dialog with its
> API-key field and model picker, 27 JS functions, three keyframe blocks and the
> `AI CHAT PANEL` stylesheet. The `config_traps` payload key went with it - it
> carried the four trap thresholds solely so the chat could put them in its system
> prompt, nothing rendered them, and the Methodology section already publishes
> them from `config.yaml`. Same reasoning that retired `spx_weights` on
> 2026-08-26.
> 
> **Why it had to go** - four consequences, all readable off the shipped code
> rather than argued. It required each visitor to paste an Anthropic API key into
> `localStorage` and called `api.anthropic.com` from the browser, so: every
> student in a college investment club needed a paid API account most do not have;
> a public page carried a password field labelled "Anthropic API Key", which is
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

