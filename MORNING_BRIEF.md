# Morning Brief - Wednesday 18 November 2026, 06:52

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

- `0a4e4c0 NIGHTLY_LOG 2026-11-18: the synthesis, the gates' results, what is next`
- `1aecd56 The synthesis note commits to the date COMING_UP gives it`
- `e0f2c28 Synthesis note, changelog, CLAUDE.md 0.14: what the screener knows about its inputs`
- `cfeb7bc wip: the health line counts the stocks the filing check cannot see`
- `3751f4b wip: measurements - every input caveat on the page, counted; the reworded sentence seen at 1440 and 375`
- `743f929 wip: staleness is said of the statement the score read, never of the company's filings`
- `5f55b79 brief: data run 2026-11-18`
- `3aecb1a data: screener run 2026-11-18 - 499 scored, top: HST EXPE VLO MU APA`
- `fd62296 brief: code session 2026-11-17-midday`
- `e68a574 NIGHTLY_LOG 2026-11-17 (13:00): the gates' results`
- `7f6a8fb NIGHTLY_LOG 2026-11-17 (13:00): the year-to-date column, the health line, what is next`
- `260a77b Research note, changelog, inventory, CLAUDE.md 0.14: the year-to-date column, measured; the proposal waits until 2026-12-15`
- `c4e1e48 wip: the note seen on DexCom's sheet; nine mutations of its claim, nine caught`
- `14663f2 wip: the rows that read operating cash flow say when the source counted part of a year twice`
- `c64629a wip: measurement - year-to-date figures in the source's quarterly cash-flow columns`

## The session's own account

> 2026-11-18 - SYNTHESIS. Six weeks of "the page says when an input is impaired", read as one layer: a sentence that was false for all 27 stocks it was printed for, and twenty companies the filing check cannot see
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-17_130001-midday.log` (completed). Data loop published? yes - `logs/datarun-2026-11-18_020001.log` (completed; HEALTH: PASS). Evidence base at `1m`: 5 of 8 needed at the 1m horizon (53 rows, but overlapping windows are not independent; 131 rows across all horizons), newest 1m run date 2026-10-16 (33 days ago, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab** (age **37 days**; a standing programme, last advanced **12 days** ago per `plan/factor-lab.md` (bound 21)); also open: 0.10 42d; 0.14 35d.
> 
> **Tests:** before **4,354 passed / 1 skipped / 0 failed** (my baseline, 4:05 in parallel; `main`
> arrived green) -> after **4,363 passed / 1 skipped / 0 failed** in 3:58 (9 new tests). Gate 2
> (dry-run) passes before and after. Gate 3: the three payload files are untouched (`rebuild_page.py`
> reported "payload byte-identical"; `git status` clean after every gate) and `index.html` is what
> the generator builds (`tests/test_rebuild_page.py`); the parse itself is
> `tests/test_payload_parse_gate.py`'s and the runner's. Gate 4: `git status --porcelain` empty.
> Gate 5, `scripts/rehearse_publish.py --changed-since 5f55b79`: **REHEARSAL PASSED** (381 page
> tests, published claims 2,373 passed, files restored byte for byte). The suite ran on the tree
> before this entry was written; nothing but this file changed after it.
> 
> **Owner queue / rotation:** I am today's first session. `write_brief.open_proposals()` returned
> `[]`, and no branch was held back by the gates. `OWNER_FOCUS.md`: nothing moved - its three open
> items are the owner's own decisions (Search Console, a byline, analytics), the long-horizon
> universe expansion (not worked: the rotation's focus had a real subject and no owner item was
> waiting on a session), and the second-reader queue, which is clear. **0.13 is not deferred** - it
> is a standing programme 12 days from its last advance against a bound of 21, and Friday
> 2026-11-20 is its day. Nothing was deferred for a gate or a loop.
> 
> **Yesterday's Next 1, read first:** both of 2026-11-17's changes landed in the 02:00 run.
> `src_notes` is **1** and DexCom carries its note on the FCF-yield and accruals rows. Operating
> cash flow over 10% from the filing fell from 10 to **1** (DexCom), as predicted. **26** stocks
> carry `annual_instead` - more than the "roughly 16" predicted; that count was of stocks the SEC
> cache could confirm (15 of 418 and 17 of 427 checked), which is the likely reason and not one I
> verified stock by stock. The brief's "scored a period behind a filing over 30 days
> old" reads **7**, not 16: the other stocks' newer 10-Q is under 30 days old, so they will join it
> over the next fortnight; the page's own note names 32. And looking at Amazon's sheet for the
> "Last fiscal year used" sentence is what started Did 1.
> 
> ### Did
> 
> **The synthesis subject.** Five changelog entries since the 2026-11-04 synthesis; none moved a
> score; all five are the page learning to say an input is not what it appears to be - the same
> kind of change as most of the six weeks before them. Nobody had looked at what they add up to.
> `research/2026-11-18-synthesis-what-the-screener-knows-about-its-inputs.md`, with five committed
> measurement scripts.
> 
> **1. Two correct changes met and made a false sentence - fixed.** The 2026-11-02 stale bar and
> the 2026-11-17 statement dating were each sound and each verified. Together they printed, on
> Amazon's page this morning, *"Its filings are flagged stale (322 days old)"* - beside a 10-Q
> filed 19 days earlier that the same payload records. Measured: **27 stocks** carried the
> sentence; **27 of 27 had a newer filing on record**, 22 of them filed inside the 165-day bar;
> ...

---

If a run says **stopped deliberately** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

