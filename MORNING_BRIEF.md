# Morning Brief - Tuesday 25 August 2026, 20:49

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-08-25T02:00:03.840225 |
| Stocks scored | 501 |
| With a price | 501/501 |
| With an analyst target | 497/501 |
| Top 5 | HST, EXPE, APA, CF, EIX |
| Evidence for weight changes | 2 of 8 needed at the 1m horizon (6 rows, but overlapping windows are not independent; 23 rows across all horizons), newest 2026-08-14 |

## What changed in the repo

- `f2f5c74 chore: remove the last things that needed a human`
- `3083968 brief: code session 2026-08-25`
- `d33dd04 docs: record the history gate, its evidence, and a defect it uncovered`
- `7e7d23c test: syntax-check the emitted dashboard JavaScript`
- `99fa64d feat(dashboard): What Changed - movers, rank deltas and per-stock history`
- `db294e5 feat: history.py - a quality-gated historical spine for the dashboard`
- `0232d40 brief: data run 2026-08-25`
- `4a0a90f data: screener run 2026-08-25 - 501 scored, top: HST EXPE APA CF EIX`
- `b149c65 brief: evening session 2026-08-24`
- `b99158b log: evening session - priority 1 closed, tomorrow's evidence path de-risked`
- `8a87b3e fix: the screener refuses to fabricate data when the network is down`

## The session's own account

> 2026-08-25 (evening) - Removing the last things that needed a human
> 
> Owner-run. Brief: make it run smoothly without being asked daily whether it
> ran, and without needing me to make updates.
> 
> **Health numbers:** last code session **ran and shipped** (06:25 today,
> `good/2026-08-25-0625`); data loop **published** 02:11, HEALTH: PASS; evidence
> base **23 rows, newest 2026-08-14, 2 effective at the `1m` horizon**; priority 0
> **DONE**.
> 
> **Tests:** before 639/639, after **647/647**
> 
> ### Did
> 
> **1. `plan/` moved out of `.claude/`.** Eight plan files sessions work from
> daily. `.claude/` is blocked as sensitive, so a session could read them and not
> correct them - which is exactly what happened this morning: the session shipped
> the time dimension and then could not mark it shipped in
> `plan/dashboard-inventory.md`. Second occurrence of this shape; `prompts/` was
> moved for the same reason on 08-21. All references updated across nine files.
> 
> **2. Refreshed `plan/dashboard-inventory.md`** from the live artifacts, which
> this morning's session was blocked from doing. It claimed "as of 2026-08-05",
> 252,191 chars and no time dimension; reality is 270,427 chars, a `history` key
> of 0.25 MB over 18 accepted run dates, and gap 1 closed. Also recorded the two
> things not to undo: the Spearman >= 0.50 comparability gate, and the ~1-month
> default comparison window.
> 
> **3. `CLAUDE.md` rule 9 - keep your own docs true.** Now that `prompts/` and
> `plan/` are editable there is no excuse for stale process docs, and the Tuesday
> focus tells the next session to *trust* the inventory. A wrong inventory sends
> it to rebuild something that exists.
> 
> **4. `scripts/prune_artifacts.py` + wired into the data loop.** `runs/` and
> `logs/` are gitignored working directories nothing ever removed. Measured
> today: **44 directories, 62 MB**, three weeks in, growing ~1.4 MB per run. That
> is a disk-space failure some months out whose first symptom would be a failed
> run. Keeps the newest 20 runs and 60 logs; **never touches `improvement/`,
> `cache/` or `validation/`** - the evidence base gets *more* valuable with age,
> and `cache/` freshness rules are load-bearing. 8 tests.
> 
> ### Tried and rejected
> 
> Nothing rejected - but the pruner took **three wrong diagnoses** before it
> worked, and all three are now pinned by tests:
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

