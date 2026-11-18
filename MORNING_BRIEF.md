# Morning Brief - Wednesday 18 November 2026, 02:27

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-18 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | HST, EXPE, VLO, MU, APA |
| Ranking's lean on growth | +0.043 (within the review's +-0.05 band; 2026-10-13 read -0.009) |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (53 rows, but overlapping windows are not independent; 131 rows across all horizons), newest 1m run date 2026-10-16 (33 days ago, bound 40) |
| Context signals | up to 1 effective observations of 8 needed. Not enough for a t-statistic yet. |
| Inputs vs SEC filings | total assets 440/440 within 0.5%; operating cash flow 430/431 within 0.5%; net income 420/443 within 0.5%; market value 452/487 within 5%; 7 scored a quarter behind a filing over 30 days old; filings cached 0 days ago |
| Top 25, worth a look | HST (#1: net income 1.29x after-tax operating income); EXPE (#2: equity 6.2% of assets); DXCM (#8: operating cash flow 36% off the filing); PSX (#9: net income 1.28x after-tax operating income); TPR (#10: equity 5.2% of assets); CF (#17: net income 15% off the filing) |
| Data run timing | **22 min** total; slowest `fetch.tickers` 573 s, `fetch.sec_roe_history` 262 s, `fetch.retry` 168 s; **51 tickers rate-limited** |
| Code-session login | 2026-12-03 09:01 (15 days) |
| Context coverage | options usable 494/499 (5 quotes-closed) - insider from SEC 499/499 - earnings variability 440 |

## What changed in the repo

- `3aecb1a data: screener run 2026-11-18 - 499 scored, top: HST EXPE VLO MU APA`
- `fd62296 brief: code session 2026-11-17-midday`
- `e68a574 NIGHTLY_LOG 2026-11-17 (13:00): the gates' results`
- `7f6a8fb NIGHTLY_LOG 2026-11-17 (13:00): the year-to-date column, the health line, what is next`
- `260a77b Research note, changelog, inventory, CLAUDE.md 0.14: the year-to-date column, measured; the proposal waits until 2026-12-15`
- `c4e1e48 wip: the note seen on DexCom's sheet; nine mutations of its claim, nine caught`
- `14663f2 wip: the rows that read operating cash flow say when the source counted part of a year twice`
- `c64629a wip: measurement - year-to-date figures in the source's quarterly cash-flow columns`
- `fea7cca Health line: a standing programme reports days since it last advanced, not only its age`
- `bdd5c70 brief: code session 2026-11-17`
- `c5b5766 NIGHTLY_LOG 2026-11-17: the gates' results`
- `01d8ed8 Research engine: rerun all 32 rows on the new code hash (no statistic changed)`
- `65af5f6 NIGHTLY_LOG 2026-11-17, CLAUDE.md 0.14 and 0.13, the drilldown spot check`
- `fb7ed28 Filing cross-check: the pre-registered re-run (rule not met), changelog, inventory, rendered checks`
- `1b68dac Figures read from the annual statement are dated by it, and the stock's page says so`

## The session's own account

> 2026-11-17 (13:00) - QUEUE. The eighth-ranked stock counts half a year of cash flow twice: measured, said on the page, and the correction left for a dated review
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-17_060001.log` (completed). Data loop published? yes - `logs/datarun-2026-11-17_020000.log` (completed; HEALTH: PASS). Evidence base at `1m`: 5 of 8 needed at the 1m horizon (53 rows, but overlapping windows are not independent; 130 rows across all horizons), newest 1m run date 2026-10-16 (32 days ago, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab** (age **36 days**; a standing programme, last advanced **11 days** ago per `plan/factor-lab.md` (bound 21)); also open: 0.10 41d; 0.14 34d.
> 
> (The roadmap half of that line is in the form this session gave it - see Did 1.)
> 
> **Tests:** before **4,334 passed / 2 skipped / 0 failed** (my baseline, 4:17 in parallel; the same
> as the 06:00 session's "after", so `main` arrived green) -> after: **4,352 passed / 3 skipped /
> 0 failed** in 4:01 (18 new passing tests; the one new skip is
> `tests/test_ytd_column_note.py::test_the_published_payload_agrees_with_the_filings`, which skips
> loudly until a payload built with the note is published - tomorrow's). Gate 2 (dry-run) exits
> normally before and after. Gate 3: the three payload files are byte-identical to the base commit
> (`git diff --stat bdd5c70` on them is empty), `index.html` is 944 KB and is what the generator
> builds (`tests/test_rebuild_page.py`); the tool guard refuses `node` in this session, so the
> parse itself is `tests/test_payload_parse_gate.py`'s (22 passed with the rebuild module) and the
> runner's. Gate 4: `git status --porcelain` empty. Gate 5, `scripts/rehearse_publish.py
> --changed-since bdd5c70`: **REHEARSAL PASSED**, published files restored byte for byte. The
> suite ran on the tree before this paragraph's numbers were filled in; nothing but this file
> changed after it.
> 
> **Owner queue / rotation:** this is today's second session, so I carried on from the 06:00
> entry's Next. Its item 1 needs tomorrow's 02:00 run and cannot be done today. Its **item 2** (the
> DexCom question, as research) is what this session did, end to end. Its **item 3** (0.13 has gone
> six sessions) is resolved in Did 1. `write_brief.open_proposals()` returned `[]` at the start and
> returns `[]` now. `OWNER_FOCUS.md`: nothing moved - its three open items are the owner's own
> decisions, the long-horizon universe expansion (not worked today; the morning's Next outranked it
> because it was a live defect on the public page's eighth-ranked stock) and the second-reader
> queue, which is clear. `git stash list` still shows the empty `auto-rescue 2026-10-28_060001`,
> left in place as every entry since 10-28 records. Nothing the 06:00 session decided is reversed.
> 
> ### Did
> 
> **1. 0.13's age now measures something (the deferral-streak rule, resolved rather than deferred a
> seventh time).** The Factor Lab is one experiment every lab Friday with no end. Its "age 36 days"
> grew daily while it shipped on schedule (experiment 19 on 2026-11-06, second-read 11-07), six
> sessions in a row logged it "deferred", and the streak rule - written for a task nobody touches -
> was about to make a Tuesday session "resolve" a programme that is not late. An open item whose
> heading names its progress file (`progress in \`plan/factor-lab.md\``) now also reports the days
> since the newest dated bullet under that file's Progress, derived, with a 21-day bound (lab
> Fridays alternate with retrospectives: 14 days, plus one lost week). It reads **11 days**.
> `scripts/health_numbers.py`, 4 new tests; `CLAUDE.md` 0.13 and `prompts/nightly.md` say that a
> non-Friday session has not deferred it, and that past 21 days it is late and the streak rule
> applies. This is the "demote with the argument" branch of the rule, and the argument is the
> number: nothing about the lab's rules or its Friday changed. **Friday 2026-11-20 is a lab day.**
> 
> ...

---

If a run says **stopped deliberately** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

