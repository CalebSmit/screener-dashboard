# Morning Brief - Tuesday 17 November 2026, 06:58

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-17 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | HST, EXPE, VLO, MU, SNDK |
| Ranking's lean on growth | +0.039 (within the review's +-0.05 band; 2026-10-13 read -0.009) |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (53 rows, but overlapping windows are not independent; 130 rows across all horizons), newest 1m run date 2026-10-16 (32 days ago, bound 40) |
| Context signals | up to 1 effective observations of 8 needed. Not enough for a t-statistic yet. |
| Inputs vs SEC filings | total assets 433/433 within 0.5%; operating cash flow 402/418 within 0.5%; net income 388/427 within 0.5%; market value 452/487 within 5%; 0 scored a quarter behind a filing over 30 days old; filings cached 7 days ago |
| Top 25, worth a look | HST (#1: net income 1.29x after-tax operating income); EXPE (#2: equity 6.2% of assets); DXCM (#8: operating cash flow 36% off the filing); PSX (#9: net income 1.28x after-tax operating income); TPR (#10: equity 5.2% of assets); CF (#17: net income 15% off the filing); WDC (#25: net income 2.13x after-tax operating income) |
| Data run timing | **18 min** total; slowest `fetch.tickers` 662 s, `fetch.retry` 127 s, `context.sec_insider` 95 s; **36 tickers rate-limited** |
| Code-session login | 2026-12-03 09:01 (16 days) |
| Context coverage | options usable 491/499 (8 quotes-closed) - insider from SEC 499/499 - earnings variability 440 |

## Things that needed attention

- BASELINE main was RED on arrival: 1 pre-existing failure(s). The session did not cause these:
- tests/test_lab_reading_and_access.py::test_the_height_the_card_bound_reads_waits_for_the_page_and_sees_a_taller_card
- BASELINE fixing a red main outranks the day's focus - gate 1 below will refuse the merge until it is green.
- full output: C:\Users\smitc\OneDrive\Documents\Screener\logs\nightly-2026-11-17_060001.baseline.txt

## What changed in the repo

- `c5b5766 NIGHTLY_LOG 2026-11-17: the gates' results`
- `01d8ed8 Research engine: rerun all 32 rows on the new code hash (no statistic changed)`
- `65af5f6 NIGHTLY_LOG 2026-11-17, CLAUDE.md 0.14 and 0.13, the drilldown spot check`
- `fb7ed28 Filing cross-check: the pre-registered re-run (rule not met), changelog, inventory, rendered checks`
- `1b68dac Figures read from the annual statement are dated by it, and the stock's page says so`
- `47991e4 test: the lab's settle helper waits for the web font by name, not for 1.3 s of quiet`
- `77c95f8 brief: data run 2026-11-17`
- `5255a8b data: screener run 2026-11-17 - 499 scored, top: HST EXPE VLO MU SNDK`
- `1a24c11 test: the inline-script parse check reads a JSON-LD block as JSON, not JavaScript`
- `a2da290 Check the publish on the live site; canonical URLs, sitemap, llms.txt`
- `310d57d test: the CLI-login check's clock is pinned to its fixture's day`
- `1449e97 Unstick the loops: reinstall a missing CLI, and fix the two tests behind five refused runs`
- `3bfa1c7 test: the What-changed footnote's hand-worked example no longer needs three movers in the day's data`
- `3e4c7b7 brief: data run 2026-11-16`
- `81baea3 data: screener run 2026-11-16 - 499 scored, top: HST EXPE MU VLO SNDK`

## The session's own account

> 2026-11-17 - PRODUCT, THROUGH THE NUMBERS. Sixteen stocks were scored on last year's statements and the page dated them last quarter
> 
> **Health (rule 8, all five):** last code session ran? **failed** - `logs/nightly-2026-11-16_130000-midday.log` (failed: Required tool 'claude' is not on PATH. Aborting.). Data loop published? yes - `logs/datarun-2026-11-17_020000.log` (completed; HEALTH: PASS). Evidence base at `1m`: 5 of 8 needed at the 1m horizon (53 rows, but overlapping windows are not independent; 130 rows across all horizons), newest 1m run date 2026-10-16 (32 days ago, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab** (age **36 days**); also open: 0.10 41d; 0.14 34d.
> 
> This is the first code session to run since 2026-11-10: the eleven between aborted on the missing
> CLI (fixed by the owner-run session on 11-16). The runner started it normally.
> 
> **Tests:** before **4,319 passed / 2 skipped / 1 FAILED** (the runner's baseline,
> `logs/nightly-2026-11-17_060001.baseline.txt`) -> after: **4,334 passed / 2 skipped / 0 failed** (15 new tests). Gate 2 (dry-run) exits normally; gate 3: the three payload files are byte-identical to the base commit and `index.html` is 943 KB (the tool guard refuses `node` in this session, so the parse itself is the runner's and `tests/test_payload_parse_gate.py`'s); gate 4 clean; gate 5 (`scripts/rehearse_publish.py --changed-since 77c95f8`) **PASSED**, published files restored byte for byte.
> 
> **Owner queue / rotation:** `write_brief.open_proposals()` returns `[]`. `OWNER_FOCUS.md`'s three
> open items are the owner's own decisions (Search Console, a byline, analytics), the long-horizon
> universe expansion, and the 2026-10-20 second-reader queue, whose sub-items are all clear - none
> needed a session today. `git stash list` still shows `auto-rescue 2026-10-28_060001`, empty and
> left in place as every entry since 10-28 records. **`main` arrived red**, which outranks the focus,
> so that came first; then two dated commitments in the brief's "Coming up", both past due; and the
> second of them turned into the product day's actual finding. **0.13 (the Factor Lab) was not
> worked** - a sixth consecutive session. The deferral-streak rule excuses a session that inherits a
> broken state, which this one did, but I am not going to lean on that: see Next.
> 
> ### Did
> 
> **1. The red baseline: a web font that answered late.**
> 
> `tests/test_lab_reading_and_access.py::test_the_height_the_card_bound_reads_waits_for_the_page_and_sees_a_taller_card`
> failed with `sum(settled) == 4549` against `sum(before) + 500 == 4459`. The 90 px between them is
> the figure that module already records for Inter replacing the fallback face on the lab's cards.
> `_settled` called a page settled once a reading had held for 1.3 s - which is a statement about
> how fast Google Fonts answered. That morning it answered later, so `before` was read in the
> fallback face (3,959 px) and everything after it in Inter (4,049).
> 
> Reproduced exactly rather than argued: with 1.5 s of latency on network requests (file:// is not
> throttled, so only the font is late) the old helper returns **3,959 and the page reads 4,049 six
> seconds later**; at 0 and 0.5 s it returns 4,049
> (`research/measurements/2026-11-17-late-font-defeats-settled.py`). The helper now waits until the
> font stylesheet is applied and no face it declares is still loading, or until the request is
> known to have failed, and only then runs its stability loop. After: 4,049 at 0, 1.5, 3 and 6 s.
> 
> Two things worth carrying. My first reproduction slept inside a route handler, which blocks the
> thread the helper runs on, so the helper waited with the font and the bug vanished - a
> reproduction that cannot fail is not one. And my first version of the wait matched the page's
> bare-host `preconnect` hint instead of the stylesheet, never came true, and ran to its 20 s
> timeout on every call while returning the right number: it read as fixed and was a sleep. The new
> test asserts the state the helper reports (`in`), not only the height, and fails on the old helper
> (`MUTANT CAUGHT: ('in', 3959)`). This module is in the 02:00 publish gate, so the same flake could
> ...

---

If a run says **stopped deliberately** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

