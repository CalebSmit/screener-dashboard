# Morning Brief - Friday 20 November 2026, 02:21

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **stopped deliberately** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-19 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | SNDK, EXPE, HST, APA, MPC |
| Ranking's lean on growth | +0.042 (within the review's +-0.05 band; 2026-10-13 read -0.009) |
| Shipped to readers | 13 commit(s) changed what a reader sees, 8 methodology change(s), in the last 7 days |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (54 rows, but overlapping windows are not independent; 133 rows across all horizons), newest 1m run date 2026-10-19 (32 days ago, bound 40) |
| Context signals | up to 1 effective observations of 8 needed. Not enough for a t-statistic yet. |
| Inputs vs SEC filings | total assets 458/458 within 0.5%; operating cash flow 443/444 within 0.5%; net income 432/456 within 0.5%; market value 453/487 within 5%; 2 scored a quarter behind a filing over 30 days old; 1 with no filing in the cache for 165 days; filings cached 2 days ago |
| Top 25, worth a look | EXPE (#1: equity 6.2% of assets); HST (#6: net income 1.29x after-tax operating income); PSX (#7: net income 1.28x after-tax operating income); DXCM (#8: operating cash flow 36% off the filing); TPR (#12: equity 5.2% of assets); CF (#21: net income 15% off the filing); ES (#23: a period behind its filings) |
| Data run timing | **18 min** total; slowest `fetch.tickers` 593 s, `fetch.retry` 152 s, `context.sec_insider` 106 s; **45 tickers rate-limited** |
| Code-session login | 2026-12-03 09:01 (13 days) |
| Context coverage | options usable 494/499 (5 quotes-closed) - insider from SEC 499/499 - earnings variability 440 |

## Waiting for a decision

A session researched these, proposed a change and left it for review. The next
session decides and records why; you can also answer in `OWNER_FOCUS.md`.

- **a Monday research session completes sections 4-5 against the criteria in section 3, which may not be changed after the first measurement runs** - open 1 day - `research/2026-11-19-what-the-ranking-is-for.md`

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-23 (in 3 days) - Monday: CLAUDE.md 0.15, what the ranking is for given no measured edge - measure against the criteria pre-registered in section 3 (fixed 2026-11-19; do not relax them), registering each test in the engine's registry before it runs. `research/2026-11-19-what-the-ranking-is-for.md`
- 2026-11-24 (in 4 days) - Build the implied-volatility inversion ACCEPTED on 2026-11-04 and dropped from every Next list since 11-05 (found 2026-11-19). Section 1d's build order: exDividendDate into the core .info fetch, a throwaway full run, QuantLib pinned and installed by a scheduled session (never from the desktop app: its AppData is sandboxed). A 06:00 session with nothing else in it. `research/2026-11-03-open-source-quant-stack.md`
- 2026-12-04 (in 14 days) - The option signals' (expected move, ATM volatility, skew) first one-month reading: readings before 2026-11-04 recorded no price and are excluded, not spliced. `METHODOLOGY_CHANGELOG.md`

## Things that needed attention

- The regenerated artifacts contradict claims this project asserts about them.
- Refusing to publish. The live site keeps the last good version.
- Run discarded: published-claims check failed - fix the generator, not the file.

## What changed in the repo

- `08b86c6 Shipped-to-readers number in the brief and health line; Set aside section and a five-line lane in the prompt`
- `c121314 Brief-skip test counts the lock-failure case; log text repaired`
- `447b820 No brief from a loop that never got the lock; CLAUDE.md 0.14 narrative to DECISIONS.md`
- `635d6de NIGHTLY_LOG: the night's gate results`
- `1386003 Drives column uses the page's percentage rule (pctPlain)`
- `ae78520 NIGHTLY_LOG 2026-11-19 (owner-run, night); the sustainable-growth fallback test inverted`
- `063486c Diagnostics shows what each category drives; a missed retrospective is owed next Friday; 0.15 pre-registered`
- `b328dbd Sustainable growth keeps its stated ROE; brief names top stocks that lost a category; gate blind spots surfaced`
- `e99fd9c Golden file regenerated for the missing-category rule; dated-item test reads a fixed future day`
- `653c02d NIGHTLY_LOG 2026-11-19 (owner-run, late evening): the review's five decisions`
- `138b176 wip: per-run rule record, EPS claim and test wording (gates follow)`
- `e234f7f Home page states the history test; layout-only gate failures publish with a warning; process weight cut`
- `be7455e A missing category counts at 50, the sector median; the EPS-mismatch flag compares like with like`
- `9468d15 Owner-run fresh-eyes review: retry an empty price history, correct the coverage-discount claim, refuse a timed-out claims gate`
- `69a7cab brief: code session 2026-11-19-midday`

## The session's own account

> 2026-11-20 (owner-run, 00:30) - Three ideas from an agent-team prompt the owner shared: one number for product progress, a "Set aside" section, and a five-line lane for each session
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-19_130000-midday.log` (completed). Data loop published? yes - `logs/datarun-2026-11-19_020001.log` (completed; HEALTH: PASS). Evidence base at `1m`: 5 of 8 needed at the 1m horizon (54 rows, but overlapping windows are not independent; 133 rows across all horizons), newest 1m run date 2026-10-19 (32 days ago, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.15, What the ranking is for, given no measured edge** (age **1 days**); also open: 0.13 39d; 0.10 44d; 0.14 37d. Shipped to readers: 13 commit(s) changed what a reader sees, 8 methodology change(s), in the last 7 days.
> 
> **Owner request:** asked whether a generic "design your first AI agent team" prompt would improve
> this system. Mostly no - the project is past that stage and needs less process, not more - but
> three of its ideas fit, and the owner said to keep going. `logs/.owner-session` held throughout.
> 
> ### Did
> 
> 1. **"One number that shows the team is actually helping"** -> `write_brief.shipped_to_readers`:
>    commits in the last 7 days that changed a reader-facing file (the page generator, the per-stock
>    sentences, the methodology generator, the research page, the lab, lineage, calc trace) and
>    methodology changelog entries in the same window; "nothing reached a reader this week" when both
>    are zero. In the brief's "At a glance" and as a sixth clause of the health line (not a gate).
>    Reads 13 and 8 today, because of tonight; 17 scheduled sessions from 11-05 to 11-19 would have
>    shown how close to zero the routine's own share was. `tests/test_shipped_to_readers.py`.
> 2. **"Explain why the others can wait"** -> a **Set aside** section in the log template
>    (`prompts/nightly.md`): every previous Next item and ACCEPTED proposal not done, one line each.
> 3. **"What it never does / when it must stop and ask"** -> a five-line lane at the top of the
>    nightly prompt: what a session owns, never does, stops for, is judged on, and how it ends.
>    The retrospective prompt now reads the "Shipped to readers" clause first.
>    CLAUDE.md rule 8 documents both.
> 
> ### Evidence / research
> 
> - None new; the reason is the fresh-eyes review's finding 4 (measured) and the owner's prompt.
> 
> ### Methodology changed
> 
> - None.
> 
> ### Tried and rejected
> 
> - **A role charter per agent as separate files.** Five lines in the prompt every session already
>   reads costs nothing; separate files would be one more thing to read and keep true.
> - **Making "Shipped to readers" a gate.** A quiet week can be right (a research week); the number
>   is for the retrospective and the owner to read, not for a runner to act on.
> 
> ### Set aside
> 
> - The CLAUDE.md "->red" sibling rows' test modules were not consolidated - only the rows were; the
>   tests still protect what they did.
> - A further CLAUDE.md trim (the data-loop gates section, ~50 lines of history) - not tonight, to
>   leave the 02:00 run a quiet tree.
> ...

---

If a run says **stopped deliberately** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

