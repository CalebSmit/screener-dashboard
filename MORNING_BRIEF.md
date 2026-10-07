# Morning Brief - Wednesday 07 October 2026, 06:38

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-10-07T02:00:04.100170 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | EXPE, HST, BBY, DLTR, BMY |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (24 rows, but overlapping windows are not independent; 71 rows across all horizons), newest 2026-09-30 |

## What changed in the repo

- `7fc45a7 T0a: record the change, the fourth defect, and what T0b inherits`
- `f847801 T0a: the claims register, and a payload differ the later stages need`
- `6986248 wip(T0a): stop calling the cardinal composite a percentile, in all six places`
- `9ca5bdc brief: data run 2026-10-07`
- `99c2431 data: screener run 2026-10-07 - 502 scored, top: EXPE HST BBY DLTR BMY`
- `770bb9e plan: full dashboard redesign plan and calculation-transparency plan, with three live defects found while writing them`
- `d5f67ff dashboard: ship design-system stage 1, and stop an interrupted session from stranding or leaking its work`
- `54bec97 wip: salvage the 2026-10-06 session's design-system pass (cut off by a 429 before it could commit)`
- `ca6f0ac brief: code session 2026-10-06 - SESSION DID NOT RUN`
- `df40d95 brief: data run 2026-10-06`
- `2913fac data: screener run 2026-10-06 - 502 scored, top: EXPE HST BBY DLTR CAH`
- `a46eb1a owner: make the dashboard look and feel premium - queue it, brief it, standing rule`

## The session's own account

> 2026-10-07 - SYNTHESIS, spent on the owner queue: T0a of calculation transparency. The composite is not a percentile, in all six places it said it was, and a claims register so an unchecked claim fails the build
> 
> **Health (rule 8, all five):** last code session ran? **No - 2026-10-06 was cut
> off by a 429 at 06:18 and its gate 4 failed on the dirty tree it left; recovered
> the same evening by an owner-run session, and the tree was clean and green at
> 1692 this morning.** | data loop published? **yes** -
> `logs/datarun-2026-10-07_020001.log` ends "Data loop complete", HEALTH: PASS, 502
> scored, top EXPE HST BBY DLTR BMY | evidence base at `1m` = **24 rows, newest
> 2026-09-07 (30 days ago, bound 40), 4 effective** - dead centre of the 30-33 day
> steady state | priority 0 **fixed 2026-08-24, not touched** (`allow_auto_apply`
> still false, 4 effective against a gate of 8) | top open roadmap item:
> **0.8, calculation transparency - 1 day old**, and it was today's work. Below it
> priority 3 (backtest v2) is **43 days** and was not taken: an owner item outranks
> the rotation and the roadmap both.
> 
> **Tests:** before **1692/1692**, after **1716/1716** (+24, no failures)
> **Owner queue / rotation:** took the top `OWNER_FOCUS.md` item (2026-10-06,
> calculation transparency), stage **T0a**, which `plan/calculation-transparency.md`
> names as next. Wednesday's nominal synthesis focus was **not** worked separately -
> the owner item outranks it, and T0a's output is a `METHODOLOGY_CHANGELOG.md`
> entry anyway, which is what synthesis day is meant to produce. Nothing deferred
> for a broken loop; both loops healthy.
> 
> ### Did
> 
> **Fixed defect 3 - the false composite sentence - everywhere, and built the
> machinery that stops its class of defect returning.** Stage T0a, committed in
> two parts.
> 
> **Re-measured before fixing, as the prompt requires. Two of the plan's numbers
> had moved:**
> 
> | Defect | Plan said (2026-10-06) | Measured today | Note |
> |---|---|---|---|
> | 1, metric weights | 334 of 4,012 pairs, 276 stocks | **333 of 4,010, 275 stocks** | T0b |
> | 2, composite line | **2** stocks (FDXF, L) | **3** - PSKY joined | T0b |
> | 3, rank sentence | median 19.6pt, 75.1% >10pt | **identical** | fixed today |
> 
> **The sentence.** `_sentence_rank` said *"Its composite of 73.8 is a percentile:
> it scores above 74% of the universe"* - for the stock ranked **1st of 502**. It
> now says *"Ranks 1st of 502 - ahead of 100% of the other 501 stocks. Its
> composite of 73.8 is a 0-100 score computed from its 8 category scores and their
> weights, not a percentile."* The share is `(N - rank) / (N - 1)`: exact
> arithmetic on two numbers printed in the same sentence, verified for all 502.
> 
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

