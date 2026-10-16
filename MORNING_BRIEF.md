# Morning Brief - Friday 16 October 2026, 06:42

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-10-16T02:00:03.815047 |
| Stocks scored | 501 |
| With a price | 501/501 |
| With an analyst target | 497/501 |
| Top 5 | EXPE, APA, HST, INCY, EG |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (31 rows, but overlapping windows are not independent; 85 rows across all horizons), newest 2026-10-09 |
| Context signals | Context signals: 7 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 425/426 within 0.5%; operating cash flow 407/411 within 0.5%; net income 398/421 within 0.5%; market value 447/487 within 5%; filings cached 0 days ago |
| Top 25, worth a look | EXPE (#1: equity 4.2% of assets); HST (#3: net income 1.46x after-tax operating income); CF (#8: net income 16% off the filing); MU (#18: a period behind its filings); ES (#19: a period behind its filings) |
| Context coverage | options usable 492/501 (9 quotes-closed) - insider from SEC 501/501 - earnings variability 443 |

## Waiting for a decision

A session researched these, proposed a change and left it for review. The next
session decides and records why; you can also answer in `OWNER_FOCUS.md`.

- **earnings yield on the source's "Normalized Income" (net income without its unusual items) instead of net income as reported; operating income unchanged. For a later session to review (section 6); nothing changed tonight.** - open 1 day - `research/2026-10-15-earnings-definition.md`

## What changed in the repo

- `aa2b114 log: flag the one exposure the widened publish gate adds`
- `8afcf10 log: the 2026-10-16 retrospective - 54 of 55 sessions produced something, and the fourth sibling-artifact hole`
- `ce1f35b nightly prompt: a research citation carries a DOI or a stable URL`
- `f0239b7 retrospective: the runner comparison becomes an enumeration, and proposals move to Orient`
- `be75c2c retrospective: CLAUDE.md's Open section is archaeology again - 1042 lines to 965`
- `62e8f00 retrospective: the focus string the session actually reads had drifted into contradiction`
- `d9a9c17 retrospective: the second payload file was published unguarded, and the claims gate's membership goes stale`
- `9e6fafb Company names in full: prefer the source's longName over its 31-character shortName`
- `2209f27 investor pass, phase 15: no sideways scroll at 320px; the pass recorded as done`
- `eb5bf78 investor pass, phase 14: one panel radius across the three pages, no monospace figures on the research page`
- `2bab769 investor pass, phase 13: one type scale; the night's log entry`
- `5d41c96 brief: data run 2026-10-16`
- `7d1b33c data: screener run 2026-10-16 - 501 scored, top: EXPE APA HST INCY EG`
- `676cdff investor pass, phase 12: no default-violet links, one segmented control, sentence-case chart titles`
- `73f14bc investor pass, phase 11: the research page and the lab share the front page's bar`

## The session's own account

> 2026-10-16 - RETROSPECTIVE. Evaluate whether this routine is actually producing value, and change the process where it is not.
> 
> **Health (rule 8, all five):** last code session ran? **yes** -
> `logs/nightly-2026-10-15_131743-midday.log` ends "Run complete: shipped to main". Data loop
> published? **yes** - `logs/datarun-2026-10-16_020001.log` ends "Data loop complete", HEALTH:
> PASS. Evidence base at `1m`: **31 rows, newest 2026-09-16 (30 days ago, bound 40), 4
> effective** - inside the bound, in steady state, and the data log's own line agrees ("4
> effective (31 rows) at the 1m horizon"). Priority 0: unchanged, `allow_auto_apply` false, 4 of
> 8 effective observations. Top open roadmap item: **0.11, the context layer** (opened
> 2026-10-08, **age 8 days**).
> **Tests:** before **2465 passed / 0 failed**, after **2488 passed / 0 failed** (+23: 12 in the
> new `tests/test_rotation_consistency.py`, 8 in the claims gate's module, 3 in the payload parse
> gate). Dry run exits 0.
> **Owner queue / rotation:** retrospective, so no rotation item. `OWNER_FOCUS.md`'s open items
> were read and left open - a retrospective does not work on the screener. One open proposal
> (`research/2026-10-15-earnings-definition.md`, earnings yield on Normalized Income, 1 day old)
> is **deliberately left for the next normal session**; a retrospective is the wrong session to
> decide a methodology question, and the prompt now puts that check in Orient so it will be the
> first thing that session does.
> 
> ### Retrospective findings
> 
> - **Sessions reviewed: 55** (2026-10-02 to 2026-10-16) - **11 scheduled, 44 owner-run.**
> - **Genuinely valuable: 54 | Churn: 0 | Lost: 1 | Failed gates: 1** (the same session:
>   2026-10-06, Tuesday, killed by an API 429 eighteen minutes in; gate 4 clean-tree failed, the
>   work was salvaged onto the branch and an owner-run session finished it that evening).
> 
> **1. What fraction of sessions produced something genuinely valuable? 54 of 55.** This is the
> first retrospective with nothing to put in the churn column. The scheduled eleven:
> 10-05 research (the premium-redesign brief, from measurements), 10-06 **lost to a 429**, 10-07
> the false composite-percentile sentence in all six places plus the claims register, 10-08
> `max_drawdown_1y`'s equation and the log-return compounding error it exposed, 10-09 the metric
> audit, 10-12 research, 10-13 the Growth proposal refused on a measurement the note had not made,
> 10-14 the input-churn/health-gate source-outage case, 10-14 13:00 the put-skew citation
> corrected, 10-15 the capex proposal refused and the capex figure decomposed, 10-15 13:00 the
> rolled-forward quarter. Every one is a defect fixed or a question answered. The owner-run 44 are
> the Factor Lab (13 experiments), the context layer, backtest v2's point-in-time record layer,
> the research engine, and the fifteen-phase investor presentation pass.
> 
> **2. Which rotation day earns its place?** All five, and I checked rather than assumed.
> I expected to find Wednesday (synthesis) was decorative and it is not: 09-09 answered the
> coherence question Monday deferred on all 502 names, 09-16 wrote the design section its Monday
> note was left open for, 09-23 shipped the weighting change the week justified, 09-30 answered the
> procurement decision gating backtest v2. Two Wednesdays went elsewhere (10-07 to the owner queue,
> 10-14 to a red gate-1 baseline), both correctly and both recorded.
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

