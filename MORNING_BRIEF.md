# Morning Brief - Monday 28 September 2026, 06:24

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-09-28T02:00:04.753730 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | EXPE, HST, BBY, EIX, MPC |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (18 rows, but overlapping windows are not independent; 58 rows across all horizons), newest 2026-09-21 |

## What changed in the repo

- `9a83f3b log: changelog, nightly log, and CLAUDE.md rule 10 now names the generated doc`
- `b4b24e1 docs: regenerate the methodology page and republish the corrected live site`
- `0d298f4 fix: the public methodology page is generated, so correct it in the generator`
- `6f3d25d brief: data run 2026-09-28`
- `2e08f62 data: screener run 2026-09-28 - 502 scored, top: EXPE HST BBY EIX MPC`

## The session's own account

> 2026-09-28 - RESEARCH. Take one specific thing - a factor, a metric, a threshold, a construction rule - and learn it properly, from the literature AND from documented practice, in this one session. Real citations, effect sizes, the conditions the effect held under, and how quant shops and institutional screens actually handle it. Where academia and practice disagree, say so and say why. A dated note in research/, complete today. No production code.
> 
> **Health (rule 8, all five):** last code session ran? **yes** -
> `logs/nightly-2026-09-25_060001.log` ends "Run complete: shipped to main",
> tagged `good/2026-09-25` | data loop published? **yes** -
> `logs/datarun-2026-09-28_020001.log` ends "Data loop complete", 502 scored, top
> EXPE HST BBY EIX MPC | evidence base at `1m` = **18 rows, newest 2026-08-28 (31
> days ago, bound 40), 4 effective** - steady-state lag, healthy | priority 0
> **fixed** (2026-08-24, untouched) | top open roadmap item: **priority 3,
> backtest v2, 34 days old** - **deferred again today**, see below
> **Tests:** before **1537/1549 (12 pre-existing failures)**, after
> **1561/1561** (+12 new tests, all 12 baseline failures fixed)
> **Owner queue / rotation:** `OWNER_FOCUS.md` **Open is empty**, so nothing to
> move to Done. **The nominal research focus was displaced by a failing ship
> gate**, which outranks both it and the owner queue: the full suite had **12
> failures at baseline** before I touched anything, meaning the runner would have
> merged nothing today. **No research note was produced and no production-code
> ban applied, because the session was a gate repair, not a research session.**
> Priority 3 deferred for the seventh time in nine sessions; its gating step is
> still a procurement decision (cost a price source for delisted tickers).
> 
> ### Did
> 
> **Found that the public methodology page is a generated file, that nobody knew
> it, and that the 02:00 data run had silently reverted the previous session's
> four corrections and republished them to the live site.**
> 
> **1. The baseline was red, which is the whole reason this became the session.**
> `python -m pytest tests/ test_screener.py -q` reported **12 failed, 1537
> passed** on an untouched tree. Every failure was in
> `tests/test_overview_claims.py`, the module the **2026-09-25** session shipped
> green three days earlier. Both halves failed - the ones reading
> `SCREENER_OVERVIEW.md` and the ones reading `index.html`.
> 
> **2. The cause is that `SCREENER_OVERVIEW.md` is generated, and `CLAUDE.md` did
> not say so.** `run_screener.py` step 11 calls `generate_screener_overview()`,
> which templates the whole document from `config.yaml` and **overwrites the file
> on every full run**. Rule 10 listed `dashboard.html`, `index.html` and
> `dashboard_data.js` as generated; it did not list this one, and "Where things
> live" filed it under hand-maintained public docs. So the 09-25 session corrected
> the markdown in good faith, its 14 tests read the markdown and passed, and the
> **2026-09-28 02:00 data run** (`2e08f62`) regenerated the file, reverted all
> four corrections, and carried the matching 26-line change into `index.html` and
> `dashboard.html`. `git log -- SCREENER_OVERVIEW.md` shows this is the **second**
> time a data-run commit has landed in that file's history (`88b4b46`, 09-11).
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

