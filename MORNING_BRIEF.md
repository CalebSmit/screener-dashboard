# Morning Brief - Monday 05 October 2026, 02:14

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## THE ROUTINE IS NOT RUNNING

- **Code session** has not run since 2 days ago

Nothing below is current. A loop that stops firing writes no log, so
the rest of this page describes the last run that *did* happen, not
today. Most likely cause: the PC rebooted and nobody logged back in -
the tasks only run while a user is signed in. See NIGHTLY_LOG.md
2026-08-20 and `scripts/register-tasks.ps1`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran 2 days ago |
| Dashboard data from | 2026-10-05T02:00:06.261856 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | EXPE, HST, BBY, VLO, CAH |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (23 rows, but overlapping windows are not independent; 68 rows across all horizons), newest 2026-09-28 |

## What changed in the repo

- `5a4bb9e data: screener run 2026-10-05 - 502 scored, top: EXPE HST BBY VLO CAH`

## The session's own account

> 2026-10-02 - RETROSPECTIVE. Evaluate whether this routine is actually producing value, and change the process where it is not.
> 
> **Health (rule 8, all five):** last code session ran? **yes** -
> `logs/nightly-2026-10-01_060001.log` ends "Run complete: shipped to main", tagged
> `good/2026-10-01` | data loop published? **yes** -
> `logs/datarun-2026-10-02_020001.log` ends "Data loop complete", HEALTH: PASS, 502
> scored, top EXPE HST BBY CAH VLO | evidence base at `1m` = **21 rows, newest
> 2026-09-02 (30 days ago, bound 40), 4 effective** - the middle of the 30-33-day
> steady state, healthy | priority 0 **fixed 2026-08-24, not weakened**
> (`_effective_observations()` still gates, `allow_auto_apply` still false, 4
> effective against a gate of 8) | top open roadmap item: **priority 3, backtest v2
> - 38 days old.** Not taken: a retrospective does not work on the screener.
> 
> **Tests:** before **1658/1658**, after **1680/1680** (+22,
> `tests/test_published_claims_gate.py`; **9 of them fail against the pre-change
> tree**)
> 
> **Owner queue / rotation:** `OWNER_FOCUS.md` **Open is empty**, so nothing to
> claim or move to Done. ISO week 40, Friday, even week - retrospective, per the
> rotation. Both loops healthy and the baseline green at 1658/1658, so nothing was
> deferred for either.
> 
> ### Retrospective findings
> 
> - **Sessions reviewed: 9 scheduled** (2026-09-21 to 2026-10-01). No owner-run
>   sessions - the second consecutive fortnight with none.
> - **Genuinely valuable: 9 | Churn: 0 | Failed gates at merge: 0** (one red
>   *baseline*, 09-28, caused by the data loop - finding 1).
> 
> **1. What fraction produced something genuinely valuable? All nine, with a merge
> commit each.** 09-21 researched position sizing and found `weighting: 'score'` is
> equal weight with noise (max deviation 0.57 pp over 39 run dates) and
> `max_position_pct` inert; 09-22 shipped the Concentration block at zero payload
> cost; 09-23 shipped the weighting change *and* found the public page had claimed
> inverse-volatility weighting on **every run the tool has ever made**; 09-24 sized
> survivorship bias at 4.3%/yr and built point-in-time membership; 09-25 closed the
> nine-session `currentPrice` item and found four false statements on the
> methodology page; 09-28 repaired a red gate and discovered
> `SCREENER_OVERVIEW.md` is generated; 09-29 shipped earnings dates, measuring
> 42.5% of them to be provider estimates before designing; 09-30 priced the
> delisted-price feed at $199/yr and decided against it; 10-01 sized look-ahead at
> **>= 63.2%** of the panel, 5.5x survivorship, and closed backtest-v2 step 1.
> 
> **The habit that makes the rest credible held again: six of the nine corrected
> this project's own published claims rather than defending them** - and two of
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

