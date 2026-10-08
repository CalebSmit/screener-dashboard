# Morning Brief - Thursday 08 October 2026, 02:15

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-10-08T02:00:03.589594 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | EXPE, HST, BBY, APA, DLTR |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (25 rows, but overlapping windows are not independent; 73 rows across all horizons), newest 2026-10-01 |

## What changed in the repo

- `d704cc6 data: screener run 2026-10-08 - 502 scored, top: EXPE HST BBY APA DLTR`
- `98afd4d final UI pass 6: the redesign is finished; nightly sessions return to methodology`
- `4e36763 final UI pass 5: price-target labels cannot collide, arrow/Home/End keys walk the windowed rankings table, tests for keyboard and focus trap`
- `cc05f00 final UI pass 4: gentler score tint, diagnostics alignment, compare gap grid and tray count, holdings concentration folded behind a summary`
- `3237695 final UI pass 3: focus trapped in every dialog, muted text passes AA on every surface, shell paints before the data lands (LCP 4.8s -> ~0.3-1.1s at 10 Mbit/s), loading placeholders`
- `1af0236 final UI pass 2: rank history as a real chart with the ordinary-variation band; history's last point is the published run (284 drilldowns showed a rank 1-3 off)`
- `0900e50 final UI pass 1: brand mark and favicon, page title block, meaningful stat strip, centred content width, footer; phone and 320px overflow fixes`
- `4466320 owner-run pass 4: calculations behind a clear click on each row, reasons for unused metrics, sheet no longer lifts its header`
- `dbd3258 owner-run pass 3: the numbers behind every score on the row itself, with equations checked against the engine`
- `783d432 owner-run UI pass 2: navigation layer, compare with gap decomposition, workings CSV, analytics filter links, Peers without verdicts, Chart.js removed, holdings de-boxed; docs and nightly fine-tuning list`
- `ee4ee07 wip: peers table loses its green/red verdicts and gains a peer median; lower drilldown blocks de-boxed; workings CSV download; toolbar label fixes`
- `5a09665 wip: navigation layer (search palette, stock links, J/K stepping, compare with gap decomposition), drilldown order matches its nav, HTML trap bars, guide, section previews, phone fixes`
- `2128a53 owner-run build: calculation transparency T0b-T6 and the premium redesign D1-D7, first version`
- `d670a65 wip: provenance dates, independent auditor and publish gate, claims register entries, methodology workings section`
- `df98b8c wip: T0b true weights and reproducibility, T1-T3 lineage and inputs, drilldown sheet, rankings table, shell and analytics redesign`

## The session's own account

> 2026-10-07 (late, pass 5) - OWNER-RUN: the redesign is finished; nightly sessions go back to methodology
> 
> Owner: *"I actually decided that I want you to finish up all UI/UX work in here tonight. And make
> the nightly sessions just focus on what it was previously focusing on, making sure the methodology
> is sound. And improving this tool overall every single night. So I want you to start working until
> you think this dashboard is something that someone would genuinly pay for."* Health numbers
> unchanged from pass 2 (same evening).
> 
> **For the next session, the short version:** there is no open owner item. The redesign is closed -
> do not start design passes. Work the rotation and CLAUDE.md "Current priorities" (0.9 first: the
> four methodology questions; 0.10: keep the inputs the history-based metrics need). The page's two
> correctness rules still bind: a scoring change ships with its frontend (0.8c), and a broken or
> false page is a defect.
> 
> **Audited at 1440, 375 and 320px before starting.** Found and fixed:
> - No brand, no page title, content edge to edge, a stat strip of four facts a visitor could not use
>   (502 / 120 / 121 / 94%). Now: brand mark + favicon + title + meta, a page title block, a stat
>   strip of what moved, trap flags and this run's weighting, centred content, a real footer.
> - **Analytics and diagnostics cards ran 44px off phone screens** (`min-width: 100%` plus padding);
>   the weight-sensitivity table and the correlation grid were cut off at 320px.
> - Rank History was a 90px sparkline: now a chart with the measured ordinary-variation band.
> - **284 of 502 drilldowns showed a history rank 1-3 places off the rank beside it** - the history
>   kept the 02:00 snapshot for today while the page came from an evening re-run. The published run
>   now defines its own date's point (`history.py`, test added). Payload regenerated: only `history`
>   and the 328 "what changed" sentences that read it differ; table_data and every score identical.
> - No focus trap in any dialog; muted text at 4.4:1 on raised surfaces (now >= 4.5:1 everywhere,
>   token `#8e8c86`); no keyboard movement in the table (now arrows/Home/End through the windowed rows).
> - **The page painted nothing for 4.8 s at 10 Mbit/s** - a blocking 1.3 MB data script in `<head>`.
>   Preloaded and moved to the end of `<body>`, with placeholder cards: largest paint ~0.3-1.1 s.
> - Price-target labels could collide; the score tint left the lowest scores as black holes; the
>   Holdings concentration note was three open paragraphs (now a summary that expands).
> 
> Final budgets: DOM 2,989 nodes, click to paint ~70-75 ms, sort ~22 ms, layout shift 0.002, LCP ~0.3-1.1 s at 10 Mbit/s. **Tests 1860 -> 1863** (+ history same-day point, arrow keys, focus trap), dry-run, `node --check` and the publish gate (359) pass.
> Docs: OWNER_FOCUS (both items to Done, archived), CLAUDE.md (0.8 closed, 0.10 added),
> `prompts/nightly.md` (design closed; Tuesday is product-through-the-numbers), the redesign plan
> (CLOSED, final budgets), inventory.

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

