# Morning Brief - Monday 26 October 2026, 13:40

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-10-26T02:00:04.754086 |
| Stocks scored | 498 |
| With a price | 498/498 |
| With an analyst target | 496/498 |
| Top 5 | EXPE, MU, APA, HST, INCY |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (36 rows, but overlapping windows are not independent; 95 rows across all horizons), newest 2026-10-16 |
| Context signals | Context signals: 13 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 425/426 within 0.5%; operating cash flow 409/413 within 0.5%; net income 399/422 within 0.5%; market value 443/484 within 5%; filings cached 3 days ago |
| Top 25, worth a look | EXPE (#1: equity 4.2% of assets); HST (#4: net income 1.46x after-tax operating income); CF (#15: net income 16% off the filing) |
| Context coverage | options usable 477/498 (21 quotes-closed) - insider from SEC 498/498 - earnings variability 440 |

## What changed in the repo

- `8894e3d log: 2026-10-26 13:00 - the price-target-change proposal reviewed and refused`
- `fd142c7 decide: the price-target-change proposal is REFUSED, and the rule that refused it is now a test`
- `b31e971 measure: the proposed target-change metric is 94% momentum in its spanned variance`
- `39f6a30 brief: code session 2026-10-26`
- `6ed80a6 research: the delegated reads returned late and sharpened the note's reasoning`
- `6043575 research: price_target_upside - the level is 71% price, and the literature scores the change`
- `3dcc5a4 measure: price_target_upside is 71% price, not analyst opinion, at the monthly horizon`
- `6a618c8 brief: data run 2026-10-26`
- `18bb07d data: screener run 2026-10-26 - 498 scored, top: EXPE MU APA HST INCY`
- `30f7b28 CLAUDE.md: the bot reads signals.json since 2026-10-25 - keep write_signals.py's fields and order`
- `c637b84 brief: code session 2026-10-25-midday`
- `67c6ff8 log + OWNER_FOCUS: the 2026-10-25 midday session - the claims register's second mutation pass`
- `61890df plan: the 2026-10-25 mutation audit - 24 mutations, 11 claims, 4 real gaps and a false claim`
- `8564552 claims audit: the capex note's 5% reconciliation had no check that could see it`
- `905b2aa claims: register the two new summary.weakest boundary checks`

## The session's own account

> 2026-10-26 (13:00 midday session) - QUEUE. The morning's proposal reviewed and REFUSED: the candidate is a real revision metric, and 94% of its overlap with the rest of the screener is momentum
> 
> **Health (rule 8, all five):** last code session ran? **yes** - the 06:00 session this morning
> ended "Run complete: shipped to main" at 06:51, tag `good/2026-10-26`. Data loop published?
> **yes** - 2026-10-26 02:00, "Data loop complete", **HEALTH: PASS**, 498 scored, top EXPE MU APA
> HST INCY. Evidence base at `1m` = **36 rows, newest 2026-09-23 (33 days ago, bound 40), 4
> effective** (`_n_observations` 4 against a gate of 8; unchanged since 02:00, as it should be -
> no data run between the two sessions). Priority 0: `allow_auto_apply` **false**, unchanged. Top
> open roadmap item: **0.13, the Factor Lab** (opened 2026-10-12, **age 14 days**) - advanced,
> since the proposal decided today came off its "From the lab to the weights" route.
> **Tests:** before **3152 passed, 1 skipped** (367.4s) - identical to the 06:00 session's
> after-count, so no drift. After **3159 passed, 1 skipped** - the 7 new tests are the guard below;
> no production code changed.
> **Owner queue / rotation:** took `OWNER_FOCUS.md` **item 1, deciding the open proposals** - the
> one the 06:00 session left this morning, which the brief and `prompts/nightly.md` both put ahead
> of the day's focus. Carried on from the morning's **Next**, which named exactly this ("the open
> proposal above needs a second reader ... its first step is cheap and decisive") and also asked
> for §3.1's three **[VERIFY FIRST]** papers to be verified; both are done. **Nothing was
> reversed.** Deferred: `OWNER_FOCUS.md` sub-item 6's extended universe, now **four** consecutive
> sessions - see **Next** for why the deferral-streak rule has still not triggered and what would
> trigger it.
> 
> ### Did
> 
> **Decided the open proposal: REFUSED.** `research/2026-10-26-price-target-upside.md` §6.2 asked a
> later session to take `price_target_upside`'s 13.5 points off the scored *level* of analyst
> implied upside and put them on a 90-day *change* in the consensus target. The decision and every
> figure are **§8** of that note, written by a session that did not write §1-§7. Two measurement
> scripts: `research/measurements/2026-10-26-target-change-candidate.py` (the candidate built over
> three windows on every end date the committed payload history supports - 49, 31 and 18 of them)
> and `-target-change-span.py` (the §8.4 regression on the published run).
> 
> - **It passed its own first test.** The note made the duplication check step 2 and said "if (2)
>   finds the duplication, the honest outcome is **no change**". It is not a duplicate: **+0.3475**
>   within sector against `fy1_revision_3m` at the proposal's own 90-day window (16 end dates),
>   +0.3028 at 63 days, +0.1978 at 30. Against the incumbent's **-0.007**, the candidate genuinely
>   is the construct-validity improvement §6.2 claimed - it belongs to the category it would sit in
>   and the incumbent does not. So the escape hatch did not fire and the proposal had to be judged
>   on the test it deferred.
> - **It fails the §8.4 spanning test, which is the one `config.yaml` requires.** With the 13.5
>   points moved and nothing else touched, **`revisions ~ momentum` goes +0.2223 -> +0.3778**,
>   through the **~0.35** bar `research/2026-09-07-...` §8.4 set *before seeing its own number* -
>   the same bar that **admitted** `fy1_revision_3m` at +0.317, which is precisely why it is not
>   available to be widened now that it refuses something. The rebuild the counterfactual rests on
>   reproduces the published Revisions score at Spearman **1.0** and **0.0 points** mean absolute
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

