# Morning Brief - Friday 09 October 2026, 06:37

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-10-09T02:00:03.566643 |
| Stocks scored | 501 |
| With a price | 501/501 |
| With an analyst target | 497/501 |
| Top 5 | EXPE, HST, APA, BBY, DLTR |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (26 rows, but overlapping windows are not independent; 75 rows across all horizons), newest 2026-10-02 |

## What changed in the repo

- `3a14c2a log: 2026-10-09 - the fix, the evidence, and the one step out of reach`
- `ef4b8ab options: say what actually happened, and record the outstanding step`
- `1df4d0f options: fetch quotes when they exist, read them when they do not`
- `07dcacf log: 2026-10-09 health, the red ship gate, and the 0.0% options measurement`
- `ed40cb5 tests: stop pinning the universe size, which blocked the ship gate`
- `8baad0f brief: data run 2026-10-09`
- `e8991bc data: screener run 2026-10-09 - 501 scored, top: EXPE HST APA BBY DLTR`
- `b8039f9 insiders: count only the company's own filings, keep 10%+ holders apart, one trade per filing-day`
- `48066b2 Reporting Soon, and insider trades from the SEC's own Form 4 filings`
- `88fa678 context layer: ship it in its own file, loaded after the page is usable; context pass measured`
- `db38cf3 context layer, first draft: technicals, options, insider trades, market backdrop and the ranking's track record - shown beside the score, never in it`
- `28237af brief: code session 2026-10-08`
- `bbe5758 log: 2026-10-08 session entry - gate results`
- `576951b fix: factor_vol_history keeps one row per date; clearer drawdown input labels; docs`
- `1077e83 wip: rescore with the drawdown fix; index.html is the dashboard, not a redirect stub`

## The session's own account

> 2026-10-09 - HARDEN AND TEACH. Tests, docs, error handling, and the investment-club experience. Would a finance student understand what they are looking at?
> 
> **Health (rule 8, all five):** last code session ran? **yes** - `logs/nightly-2026-10-08_060001.log`
> ends "shipped to main", tagged `good/2026-10-08`. Data loop published? **yes** -
> `logs/datarun-2026-10-09_020001.log` ends "Data loop complete", HEALTH: PASS, 501 scored.
> Evidence base at `1m` = **26 rows, newest 2026-09-09 (30 days ago, bound 40), 4 effective**
> (`_n_observations`; the whole file reads 75 rows / 2026-10-02 and is not the number to quote).
> Priority 0 - `allow_auto_apply` still `false`, 4 effective against a gate of 8, still top of the
> queue. Top open roadmap item: **0.9, the four methodology questions the transparency build
> surfaced - age 2 days** (opened 2026-10-07); 0.10 and 0.11 are the same age.
> **Tests:** before **1936 passed, 3 failed**; after - see the end of this entry.
> **Owner queue / rotation:** took the open owner item (`OWNER_FOCUS.md` 2026-10-08, the context
> layer, brief `plan/context-layer.md`) - but a **failing ship gate outranked it** and came first.
> 
> ### Did
> 
> **1. The ship gate was red at 06:00, and would have blocked tonight's merge whatever I shipped.**
> Baseline was 3 failed / 1936 passed, all three in `tests/test_dashboard_browser.py`. Cause: the
> 02:00 run scored **501** stocks rather than the 502 of the days before - an ordinary S&P 500
> membership change and a correct run - and three tests asserted the literals `502` / `503` against
> the published payload.
> 
> That is worse than three red tests. The runner's gate 1 is `pytest tests/ test_screener.py -q`
> and it merges only on **exit code 0**, with no baseline (`scripts/nightly-screener.ps1`, "Gate 1:
> tests") - deliberately, because a gate that tolerates yesterday's failures is not a gate. So a
> literal the *data* can move on its own does not fail one test, it **halts every merge until a
> human edits the number**. Index membership changes several times a year.
> 
> Fixed by asserting the invariant rather than the count: the table's `aria-rowcount`, its last
> rank and its "N stocks" text all agree with the payload's own universe, inside the **495-515**
> band `universe_history.validate_membership` already enforces. That is stronger than the literal -
> it catches an off-by-one or a truncated table at *any* universe size.
> `tests/test_universe_size_is_not_pinned.py` (4 tests) is the tripwire so it cannot come back
> anywhere else, including a test that the pattern still matches the exact line that failed today -
> rule 8's "a tripwire wired to a number that cannot stand still is decoration", applied to its
> opposite: one wired to a number that cannot move.
> 
> **2. Owner item, `plan/context-layer.md` queue item 2: the options panel is empty for every stock
> on every scheduled run.** Item 2 asked for the share of stocks with `_ctx_opt_status == "ok"` and
> whether that is good enough. Measured, and it is not:
> 
> | When | Hour (ET) | n | `ok` | usable |
> |---|---|---|---|---|
> | 2026-10-08 owner-run | 21:27 | 503 | 394 | **78.3%** |
> | 2026-10-09 **02:00 data loop** | 03:00 | 503 | **0** | **0.0%** |
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

