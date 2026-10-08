# Morning Brief - Thursday 08 October 2026, 06:55

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-10-08T06:18:36.982674 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | EXPE, HST, APA, BBY, DLTR |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (25 rows, but overlapping windows are not independent; 73 rows across all horizons), newest 2026-10-01 |

## What changed in the repo

- `bbe5758 log: 2026-10-08 session entry - gate results`
- `576951b fix: factor_vol_history keeps one row per date; clearer drawdown input labels; docs`
- `1077e83 wip: rescore with the drawdown fix; index.html is the dashboard, not a redirect stub`
- `6fc0297 wip: changelog entry, scoring_schema bump to 3, log entry started`
- `1b8613b fix: one observation per run date in the evidence base (priority 0.6)`
- `d99c20d wip: max_drawdown_1y measures the price path, and publishes the two closes it measured between`
- `f9582f7 measure: 22.6% of composite has no shown arithmetic; max_drawdown_1y compounds log returns as simple`
- `8a8c195 brief: data run 2026-10-08`
- `d704cc6 data: screener run 2026-10-08 - 502 scored, top: EXPE HST BBY APA DLTR`
- `98afd4d final UI pass 6: the redesign is finished; nightly sessions return to methodology`
- `4e36763 final UI pass 5: price-target labels cannot collide, arrow/Home/End keys walk the windowed rankings table, tests for keyboard and focus trap`
- `cc05f00 final UI pass 4: gentler score tint, diagnostics alignment, compare gap grid and tray count, holdings concentration folded behind a summary`
- `3237695 final UI pass 3: focus trapped in every dialog, muted text passes AA on every surface, shell paints before the data lands (LCP 4.8s -> ~0.3-1.1s at 10 Mbit/s), loading placeholders`
- `1af0236 final UI pass 2: rank history as a real chart with the ordinary-variation band; history's last point is the published run (284 drilldowns showed a rank 1-3 off)`
- `0900e50 final UI pass 1: brand mark and favicon, page title block, meaningful stat strip, centred content width, footer; phone and 320px overflow fixes`

## The session's own account

> 2026-10-08 - BUILD. Implement what the week's research justified. Write tests alongside the code.
> 
> **Health (rule 8, all five):** last code session ran? **yes** - `logs/nightly-2026-10-07_060001.log`
> ends "Run complete: shipped to main", tagged `good/2026-10-07`. | Data loop published? **yes** -
> `logs/datarun-2026-10-08_020001.log` ends "Data loop complete", HEALTH: PASS, 502 scored. |
> Evidence base at horizon `1m`: **25 rows, newest `run_date` 2026-09-08 (30 days ago, bound 40),
> 4 effective observations** - inside the steady-state 30-33 day lag, nothing to investigate. |
> Priority 0: holding as designed - `allow_auto_apply` still `false`, 4 effective against a gate of
> 8, and the engine reported rather than applied. | Top open roadmap item: **0.9, the four
> methodology questions the transparency build surfaced - age 1 day** (opened 2026-10-07).
> **Tests:** before **1863 passed / 0 failed**; after **1894 passed / 0 failed** (+31: 11 drawdown, 11 one-observation-per-date, 5 index.html, 2 lineage guards, 2 series-equation guards). All four ship gates pass: suite clean, dry-run OK, index.html 461,071 B and the payload parses (node --check rc 0), tree clean. The data loop publish gate passes too (363).
> **Owner queue / rotation:** `OWNER_FOCUS.md` has **no open item** - the redesign and the
> transparency work were closed on 2026-10-07 and nightly sessions were told to return to
> methodology. So this was the rotation's Thursday: build. The week justified no *weighting* change
> (it was spent on the owner's UI/UX work, not on factor research), so per the prompt's instruction
> for that case I took the top open item in "Current priorities" that is a build task - priority
> 0.10, keeping the inputs the history-based metrics need - and writing the first of those
> equations exposed an arithmetic error in `max_drawdown_1y`, which became the session.
> 
> ### Did
> - **Fixed a measured arithmetic error in `max_drawdown_1y`, and put its arithmetic on the
>   page.** `compute_metrics` step 16d built the path it measured the fall on with
>   `cumprod(1 + log return)`. `_daily_returns` holds **log** returns, so that series is neither
>   the price path nor the log path: because `ln(1+r) <= r` it drifts below the real path, and
>   the drift compounds, so the peak-to-trough ratio taken on it was not the stock's largest
>   fall. It is now `exp(cumsum(log return))`. Two smaller corrections in the same block: the
>   series is read in **date order** (the drawdown is order-dependent and was relying on the
>   fetch's dict insertion order), and the engine now publishes the two closes the fall was
>   measured between. *I know this is an improvement because* the old expression did not compute
>   the quantity its own label, its own code comment and its own published formula all claimed -
>   and the error is one-directional and measurable.
>   **Measured on the full rescoring run:** smaller fall for **499 of 499** stocks, median
>   **+1.317pp**, max **+13.520pp** (SNPS **-52.68% -> -39.16%**); sector percentile Spearman
>   0.993 with **264 of 499** moving more than half a point and a largest move of **24.3**;
>   composite median |move| **0.070**, max **4.19**; **379 of 502** ranks move, max **22
>   places**. Same top ten, BBY/APA and CAH/INCY/BMY reordered within it.
>   `METHODOLOGY_CHANGELOG.md` 2026-10-08; `tests/test_max_drawdown_price_path.py` (11 tests,
>   written against paths whose drawdown is known by construction, including one that keeps the
>   old formula present as the thing that must not come back).
> 
> - **The page shows it, same commit (0.8c).** The metric's row opens to
>   `($367.70 at the trough - $604.37 at the prior peak) / $604.37 at the prior peak = -39.2%`
>   with both dates and the rebuild check. It is the engine's own pair, published through
>   `ENGINE_KEYS` from the one place it is computed - the page does not re-derive it.
>   `max_drawdown_1y` moves from `SOURCES` to `EQUATIONS` as **exact**, so the suite now
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

