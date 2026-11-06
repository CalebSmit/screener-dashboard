# Morning Brief - Friday 06 November 2026, 16:01

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **stopped deliberately** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-05 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | VLO, MPC, EXPE, MU, SNDK |
| Ranking's lean on growth | +0.060 (OUTSIDE the review's +-0.05 band; 2026-10-13 read -0.009) |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (44 rows, but overlapping windows are not independent; 112 rows across all horizons), newest 1m run date 2026-10-05 (32 days ago, bound 40) |
| Context signals | 19 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Data run timing | **2 min** total; slowest `snapshot` 49 s, `context.track_record` 31 s, `dashboard` 5 s |
| Code-session login | 2026-12-03 09:01 (26 days) |
| Context coverage | options usable 476/499 (23 quotes-closed) - insider from SEC 499/499 - earnings variability 439 |

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-09 (in 3 days) - The context signals' first one-month reading arrives with this run; read the brief's `context_eval` line (CLAUDE.md 0.11). The three option signals wait for 2026-12-04 (below). `CLAUDE.md`
- 2026-11-16 (in 10 days) - Re-run the filing cross-check's range table; promote total assets, equity and operating cash flow to a gate at 25 only if each stayed <= 5 on every run. `research/2026-10-14-yahoo-vs-sec.md`

## Things that needed attention

- Health check could not evaluate this run (exit 2).
- Run discarded - could not verify it was healthy.

## What changed in the repo

- `f450757 brief: code session 2026-11-06-midday`
- `840e8a5 log: the suite count after is 4046, the same as the baseline`
- `2e918cb nightly 2026-11-06 (13:00) - the Factor Lab, experiment 19`
- `04237b4 lab: experiment 19 written up, and the overview's phone budget measured where it settles`
- `3f3cb98 lab: experiment 19, Mohanram's growth score on OSAP's six MS groups - 4 of 7 within`
- `4a55141 lab: experiment 19 tolerances, written before the comparison (Mohanram 2005, on OSAP's MS groups)`
- `e18feaf Say how old the prices are, and make two tiles tell their story`
- `17b85a1 brief: code session 2026-11-06`
- `4473ac7 Both branches of the capped line are reached whichever payload is installed`
- `a9dd812 A capped metric row prints the figure its own arithmetic gives`
- `5acfa4c The holdings strip test reads each stand-in's rank, not its position`
- `f8af01f brief: data run 2026-11-06`
- `50b535f Read the flag-key scroll where it ends, not 700 ms into it`
- `9db2e4f Compare says what sets each stock apart, under its composite`
- `ffe51b5 Say in words what sets each stock apart: leaders, sheets and holdings`

## The session's own account

> 2026-11-06 (13:00 session) - QUEUE: the Factor Lab, experiment 19 (Growth's first rerun)
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-06_060000.log`
> (completed). Data loop published? **no** - `logs/datarun-2026-11-06_020000.log` (stopped
> deliberately by the publish gate; HEALTH: PASS; both causes were fixed and shipped by this
> morning's 06:00 session, and the owner-run session scheduled a one-time catch-up run for **16:00
> CST today**, after the close - reading it is the next session's first job). Evidence base at `1m`:
> **44 rows, newest 2026-10-05 (32 days ago, bound 40), 5 effective of the 8 needed**. Priority 0:
> `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab** (age **25 days**) -
> **advanced today**; also open: 0.10 30d, 0.14 23d.
> **Tests:** before 4046 passed / 2 skipped (green baseline, nothing inherited), after 4046 / 2 -
> the same count: today added assertions inside existing tests, not new test functions.
> **Owner queue / rotation:** no open proposal (`write_brief.open_proposals()` -> `[]`). No
> held-back branch. Today's 06:00 entry ended "the next Friday that is not a retrospective owes the
> lab one finished experiment" - today **is** that Friday (ISO week 45, odd), so this session
> carried on from its Next rather than starting a fresh owner item. `git stash list` holds one
> `auto-rescue 2026-10-28` entry: **it is empty** (`git stash show -p` prints nothing), so there is
> nothing in it to recover.
> 
> ### Did
> 
> **Experiment 19: Mohanram (2005), the growth score, on Open Source Asset Pricing's `MS` groups.
> 4 of 7 figures within tolerances committed before any return was read (`4a55141`).**
> `research/2026-11-06-lab-mohanram.md`.
> 
> **Growth was the last of the eight categories with no rerunnable experiment - and the lab's own
> category table said why, falsely:** "Mohanram (2005), which it cites, needs company accounts, and
> the library has no sort on sales or earnings growth." The second half is still true; the first is
> not. Chen & Zimmermann publish portfolio returns for exactly this score, and their construction
> code **pools the score's ends the way the paper does** (6-8 to 6, 0-1 to 1), so their six
> portfolios *are* the paper's six groups - which is why this comparison could be row by row where
> experiment 17's, on deciles of a discrete score, could only be end to end. That is the lab
> auditing the screener's own citations in the direction nobody plans for: the claim about what
> *cannot* be checked was the thing that needed checking.
> 
> **The result, and the part that matters to a screener that only buys.** The strong end reproduces
> (16.5% a year against the paper's 17.4%); the weak end does not (+3.2% against -4.0%), so the gap
> is 13.3% a year (t 4.96) against 21.4% and outside its tolerance. The whole shortfall is the
> losers the public construction does not find - and the paper's own size-adjusted panel says that
> is where its effect lives (+3.3% strong against -17.9% weak; its words, "more effective in
> identifying potential losers ... than winners"). So the half that reproduces is the half this
> screener could act on, and the half that does not is the half it never could. Score 5 sits below
> score 4, so the monotonic rise is not reproduced in the middle.
> 
> **It survived its own sample, which almost nothing in this lab has:** 5.0% a year equally weighted
> ...

---

If a run says **stopped deliberately** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

