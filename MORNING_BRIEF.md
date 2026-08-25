# Morning Brief - Monday 24 August 2026, 22:06

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-08-24T02:00:03.959537 |
| Stocks scored | 501 |
| With a price | 501/501 |
| With an analyst target | 497/501 |
| Top 5 | HST, EXPE, APA, EIX, CF |
| Evidence for weight changes | 2 of 8 needed at the 1m horizon (6 rows, but overlapping windows are not independent; 23 rows across all horizons), newest 2026-08-14 |

## What changed in the repo

- `b99158b log: evening session - priority 1 closed, tomorrow's evidence path de-risked`
- `8a87b3e fix: the screener refuses to fabricate data when the network is down`
- `30ee3c5 brief: code session 2026-08-24`
- `a118727 docs: priority 0 is done; record the evidence and what has not changed`
- `5200fec fix: report effective observations, not the raw row count`
- `8cc18d9 data: repair the evidence base, and the 1-month backfill it unblocks`
- `9a03e96 fix: the improvement engine's observation count was meaningless`
- `cff2675 brief: data run 2026-08-24`
- `37ab0c9 data: screener run 2026-08-24 - 501 scored, top: HST EXPE APA EIX CF`

## The session's own account

> 2026-08-24 (evening) - Priority 1 closed: the screener no longer fabricates data
> 
> Owner-run session. Asked to make the routine work as well as it can going
> forward, on the evening after priority 0 landed.
> 
> **Health numbers:** last code session **ran and shipped** (06:21 today,
> `good/2026-08-24`); data loop **published** 02:11, HEALTH: PASS; evidence base
> **23 rows, newest 2026-08-14, 2 effective at the `1m` horizon**; priority 0
> **DONE**.
> 
> **Tests:** before 590/590, after **596/596**
> 
> ### Did
> 
> **1. De-risked tomorrow's 02:00 run before it happens.** This morning's session
> flagged that tonight is the first unattended exercise of the new evidence path,
> and named the exact log line to expect. Ran both reporting paths by hand:
> 
> - `data-run.ps1`'s inline snippet prints *"2 effective (6 rows) at the 1m
>   horizon; 23 rows across all horizons"* - exactly as predicted.
> - `write_brief.py` prints *"2 of 8 needed at the 1m horizon (6 rows, but
>   overlapping windows are not independent...)"*.
> - `compute_forward_returns()` with today's date: **0 new rows in 0.1s, no
>   fetch triggered.** This was the flagged risk - the bounded-fetch change could
>   have turned every revisited snapshot into a full-universe download. It does
>   not.
> 
> **2. Priority 1 - the synthetic-data fabrication defect, closed at source.**
> Detail in `METHODOLOGY_CHANGELOG.md` 2026-08-24 (evening). `run_factor_engine()`
> now exits 2 rather than generating fiction when the network probe fails;
> `--allow-synthetic` is the deliberate opt-in and labels its own output.
> 
> Open since 08-06 and gated only downstream, so the scheduled loop was protected
> and nothing else was.
> 
> ### Tried and rejected
> 
> **Adding the flag to `cli.py` alone.** That was the first attempt and it was
> **inert** - `run_screener.py` defines its own `parse_args()` and never imports
> `cli.py`. `args.allow_synthetic` would not have existed, the `getattr` default
> would have refused *every* run including the intended opt-in, and the 02:00
> loop would have failed tomorrow. Caught only because `--help` did not list the
> new flag. The regression test now asserts the flag on the **live** parser and
> parses an empty argv through it.
> 
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

