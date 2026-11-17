# Morning Brief - Tuesday 17 November 2026, 02:23

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **failed** - last ran today |
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

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-09 (**due 8 days ago - still listed**) - The context signals' first one-month reading arrives with this run; read the brief's `context_eval` line (CLAUDE.md 0.11). The three option signals wait for 2026-12-04 (below). `CLAUDE.md`
- 2026-11-16 (**due 1 day ago - still listed**) - Re-run the filing cross-check's range table; promote total assets, equity and operating cash flow to a gate at 25 only if each stayed <= 5 on every run. `research/2026-10-14-yahoo-vs-sec.md`

## Things that needed attention

- Required tool 'claude' is not on PATH. Aborting.

## What changed in the repo

- `5255a8b data: screener run 2026-11-17 - 499 scored, top: HST EXPE VLO MU SNDK`
- `1a24c11 test: the inline-script parse check reads a JSON-LD block as JSON, not JavaScript`
- `a2da290 Check the publish on the live site; canonical URLs, sitemap, llms.txt`
- `310d57d test: the CLI-login check's clock is pinned to its fixture's day`
- `1449e97 Unstick the loops: reinstall a missing CLI, and fix the two tests behind five refused runs`
- `3bfa1c7 test: the What-changed footnote's hand-worked example no longer needs three movers in the day's data`
- `3e4c7b7 brief: data run 2026-11-16`
- `81baea3 data: screener run 2026-11-16 - 499 scored, top: HST EXPE MU VLO SNDK`
- `af6f012 brief: code session 2026-11-16-midday`
- `32836d7 brief: code session 2026-11-16`
- `186f261 brief: data run 2026-11-16`

## The session's own account

> 2026-11-17 (owner-run, after midnight) - an SEO playbook, triaged: the publish is now checked on the live site
> 
> **Health (rule 8, all five):** last code session ran? **no** - 2026-11-16 13:00 aborted on the missing
> CLI, fixed that evening (the next real test is 06:00 today). Data loop published? **yes** -
> `logs/datarun-2026-11-16_180000.log`, the 18:00 retry, live site confirmed serving
> `2026-11-16T18:00:03` by the new check. Evidence base at `1m`: **51 rows, newest 2026-10-14 (34 days
> ago, bound 40), 5 effective observations**. Priority 0: `allow_auto_apply` **false**. Top open roadmap
> item: **0.13, The Factor Lab** (opened 2026-10-12, age **36 days**).
> 
> The owner pasted a long tiered SEO/discovery playbook and asked to take what applies. Audited the
> live site first: most of the technical tier was already here (descriptions, sharing cards, 404,
> cross-links, footers). The triage, with what does not fit and why, is `plan/site-discovery.md`.
> 
> ### Shipped
> 
> 1. **`scripts/check_live_site.py`, called by `data-run.ps1` right after the push.** The loop said
>    "live dashboard refreshed" because `git push` exited 0; a failed Pages deploy afterwards was
>    invisible to every log (the owner's GitHub failure emails in early November). It range-requests
>    the live payload's first 400 bytes (cache-busted - Pages' CDN holds `max-age=600`), compares the
>    `run_timestamp` with the one pushed, waits up to 20 min, prints one line and always exits 0.
>    A stale site logs `ATTENTION - LIVE SITE NOT UPDATED` at WARN, which `write_brief.outcome` already
>    surfaces. Verified against the real site: serving the 18:00 run. `tests/test_live_site_check.py`
>    (9, fake clock, no network).
> 2. **Canonical URLs** on every page (`/` for both `index.html` and `dashboard.html`), equal to `og:url`;
>    **`sitemap.xml`**; **`llms.txt`** (the page shows a JS-less reader ~8 KB of frame and no ticker -
>    measured - so this points readers at the methodology and payload); one **`WebApplication` JSON-LD**
>    block that repeats the meta description and claims nothing the page does not say; the research
>    page's `<title>` now matches its `<h1>`. `tests/test_site_discovery.py` (8). Pages rebuilt with
>    `scripts/rebuild_page.py` (payload byte-identical) and `python v2_experiments.py report`.
> 3. **Not done, on purpose:** `robots.txt` (a project site's is never read - the host root's returns
>    404, so all is crawlable), outreach of any kind (publishing in the owner's name), pricing/trial/
>    competitor pages (nothing is sold), trend-chasing (rule 4). Owner decisions queued in
>    `OWNER_FOCUS.md`: Search Console, a byline, analytics.
> 
> ### Next
> 
> - The 06:00 session today is the first since the CLI fix; its log is the proof.
> - After the next real data run, read the `Live site:` line in `logs/datarun-*.log`.

---

If a run says **stopped deliberately** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

