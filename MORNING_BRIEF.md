# Morning Brief - Tuesday 13 October 2026, 02:19

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-10-13T02:00:02.873759 |
| Stocks scored | 501 |
| With a price | 501/501 |
| With an analyst target | 497/501 |
| Top 5 | EXPE, HST, INCY, APA, BMY |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (28 rows, but overlapping windows are not independent; 79 rows across all horizons), newest 2026-10-06 |
| Context signals | Context signals: 4 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07, so no signal has a record yet (gate: 8 effective observations). |
| Context coverage | options usable 484/501 (17 quotes-closed) - insider from SEC 501/501 - earnings variability 443 |

## Waiting for a decision

A session researched these, proposed a change and left it for review. The next
session decides and records why; you can also answer in `OWNER_FOCUS.md`.

- **Growth 13% -> 10%, Quality 22% -> 25% (section 7 of this note)** - open 1 day - `research/2026-10-12-growth-category.md`

## What changed in the repo

- `a5e84b8 data: screener run 2026-10-13 - 501 scored, top: EXPE HST INCY APA BMY`
- `0bbd9c5 brief: list proposals waiting for a decision, with their age`
- `97e7a29 Factor Lab experiment 9 (the largest fifth), and the nightly sessions set up to carry the work on`
- `97ab4a8 research: value and momentum among large companies - no weight change, and a page that says why`
- `5f39f6f research: what stands behind Growth's 13%, and what the category actually does`
- `55ccf6e Factor Lab: what a holder who shorts nothing keeps of each factor among big companies`
- `d73e26f Factor Lab: the factors among large companies, and how a lab finding reaches a weight`
- `447736d Factor Lab: every headline result inside its paper's sample and since, on the front page`
- `573fd30 methodology page: the Risk and Size sections say what the Factor Lab measured`
- `e1eee1a scripts/rehearse_publish.py: see the next data run's page before the data run does`
- `9280df1 dashboard: the methodology loads when it is opened - found by rehearsing tonight's 02:00 publish`
- `d00a72e Factor Lab experiment 7: does market risk pay? (Fama & French 1992, Table II B); the methodology links each category to its experiments`
- `cba877c Factor Lab experiment 6: does a high earnings yield pay? (Fama & French 1992, Table IV) - the nearest public sort to the Valuation category`
- `5f8e7be Factor Lab experiment 5: do riskier stocks earn more? (Ang, Hodrick, Xing & Zhang 2006), on a generic sorted-portfolio layout`
- `b985630 Factor Lab: a front page and a category map; the methodology page says what the factors have done since 2014`

## The session's own account

> 2026-10-13 (owner-run, last before the 02:00 run) - open proposals in the morning brief
> 
> **Health (rule 8):** unchanged since the last entry - both loops ran 2026-10-12, HEALTH: PASS;
> evidence base at `1m` **28 rows, newest 2026-09-11 (32 days, bound 40), 4 effective**; priority 0
> unchanged. Top open roadmap item: **0.13, the Factor Lab** (opened 2026-10-12, age 1 day).
> **Tests:** before 2222 passed / 0 failed; after 2227 passed / 0 failed.
> 
> ### Did
> - **`MORNING_BRIEF.md` gains "Waiting for a decision"** (`write_brief.open_proposals`). A
>   research note that proposes a change and leaves it for review carries one line,
>   `**Proposal status: OPEN** - <what>`; the brief lists each with its age in days until the
>   line is rewritten. About two scheduled sessions in three did not start at all in August; a
>   proposal that only a session can see would have waited on that. The Growth note carries the
>   line. The function never raises - the brief is written from `finally`.
>   `tests/test_morning_brief.py::TestOpenProposals`, 5 tests.
> - **The plan's questions 3 and 4 now point at the notes that already researched part of them**
>   (2026-08-31 on size, 2026-09-02 on category independence), so a Monday does not start from
>   scratch.
> - **The bot's next run was rehearsed** against the published rankings (`python -m bot.rehearse
>   --date 2026-10-13`): no orders due, a second equity row per account, exit 0.
> 
> ### Methodology changed
> - None.
> 
> ### Gates
> 1. **2227 passed / 0 failed** (baseline 2222 / 0). 2. `run_screener.py --dry-run`
> **DRY-RUN PASSED**. 3. payload and lab files pass `node --check`. 4. `git status --porcelain`
> clean after the commit. No generator was touched in this commit; the full publish rehearsal
> passed on the commit before it.
> 
> **Machine state, rule 11:** nothing changed; the three tasks were read back earlier tonight.

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

