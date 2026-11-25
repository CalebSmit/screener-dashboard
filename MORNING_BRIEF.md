# Morning Brief - Wednesday 25 November 2026, 13:48

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-25 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | EXPE, APA, VLO, SNDK, MU |
| Ranking's lean on growth | -0.001 (within the review's +-0.05 band; 2026-10-13 read -0.009) |
| Shipped to readers | 21 commit(s) changed what a reader sees, 6 methodology change(s), in the last 7 days |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (58 rows, but overlapping windows are not independent; 144 rows across all horizons), newest 1m run date 2026-10-23 (33 days ago, bound 40) |
| Context signals | up to 1 effective observations of 8 needed. Not enough for a t-statistic yet. |
| Inputs vs SEC filings | total assets 453/453 within 0.5%; operating cash flow 439/440 within 0.5%; net income 427/451 within 0.5%; market value 452/487 within 5%; 2 scored a quarter behind a filing over 30 days old; 1 with no filing in the cache for 165 days; filings cached 7 days ago |
| Top 25, worth a look | EXPE (#1: equity 6.2% of assets); HST (#6: net income 1.29x after-tax operating income); PSX (#10: net income 1.28x after-tax operating income); DXCM (#14: operating cash flow 36% off the filing); CF (#18: net income 15% off the filing); ES (#19: a period behind its filings); TPR (#21: equity 5.2% of assets) |
| Data run timing | **20 min** total; slowest `fetch.tickers` 646 s, `snapshot` 167 s, `fetch.retry` 134 s; **37 tickers rate-limited** |
| Code-session login | 2026-12-03 09:01 (8 days) |
| Context coverage | options usable 483/499 (16 quotes-closed) - insider from SEC 499/499 - earnings variability 440 |

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-26 (in 1 day) - Publish our own implied volatilities (built 2026-11-24, recorded beside the source field as _ctx_opt_own_*). First confirm the 11-25 02:00 run's raw fetch carries exDividendDate and the 11-25 20:00 refresh's cache has _ctx_opt_own_iv for most stocks; count own vs source coverage over the whole cache, then flip context_signals.PUBLISHED_IV_CONSTRUCTION to 'own_mid' with a changelog entry. Do not loosen MAX_REL_SPREAD or the IV bounds to raise coverage. `research/2026-11-03-open-source-quant-stack.md`
- 2026-12-04 (in 9 days) - The option signals' (expected move, ATM volatility, skew) first one-month reading: readings before 2026-11-04 recorded no price and are excluded, not spliced. `METHODOLOGY_CHANGELOG.md`
- 2026-12-07 (in 12 days) - The last Monday before the year-to-date-column decision of 2026-12-15: research what vendors and index providers do with a hole in a company's quarterly series. 26 stocks were scored on last fiscal year's flows for that reason on 2026-11-18, and it is the same question as DexCom's (may a filing do more than sit beside a scored input) at 26 times the size. Section 4; no design is proposed there. `research/2026-11-18-synthesis-what-the-screener-knows-about-its-inputs.md`

## Things that needed attention

- GATE 5 note: ENGINE CHANGED since run fc6071fe6274 was scored (run_screener.py): this rehearsal shows the new page code on that run's old scores, so the next 02:00 run is the first to put the new scores before the claims gate. A page branch keyed on a new payload field needs a test that sets the field in the browser (prompts/nightly.md, section 4).

## What changed in the repo

- `8b2b9d9 NIGHTLY_LOG 2026-11-25 (13:00): Factor Lab experiment 20, and gate results`
- `a84527e lab: experiment 20 written up (research note, plan queue row and Progress, owner item count)`
- `97db773 lab: experiment 20 states the paper's risk-adjusted gap too - about half of it survives, in its years and since`
- `6069191 page: the methodology's Rerun-the-research paragraph links experiment 20 (rebuild_page.py; payload byte-identical)`
- `8873be5 lab: experiment 20, Ang, Hodrick, Xing & Zhang's idiosyncratic-volatility sort - 3 of 6 within`
- `1c9aa54 lab: experiment 20 tolerances, written before the comparison (Ang, Hodrick, Xing & Zhang 2006, idiosyncratic volatility, Table VI Panel B)`
- `8c2fad0 brief: code session 2026-11-25`
- `f49d35d NIGHTLY_LOG 2026-11-25: gate results`
- `90d2093 NIGHTLY_LOG 2026-11-25: synthesis entry`
- `c87c8f5 exposure_shift: a scoring change records the ranking's factor exposures before and after`
- `6d154fb wip: changelog check of residual momentum on the first data run; growth-lean break recorded`
- `f0f69c8 wip: synthesis section on residual momentum's fit (note section 10)`
- `1df46b4 wip: synthesis measurement - residual momentum's fit against the other seven categories`
- `8c06fdc brief: data run 2026-11-25`
- `c1dd9e1 data: screener run 2026-11-25 - 499 scored, top: EXPE APA VLO SNDK MU`

## The session's own account

> 2026-11-25 (13:00) - QUEUE. Factor Lab experiment 20: the volatility paper's own headline sort - 3 of 6 within; the raw gap does not reproduce, about half of the risk-adjusted one does, and it survived the paper
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-25_060001.log`
> (completed). Data loop published? yes - `logs/datarun-2026-11-25_020001.log` (completed; HEALTH: PASS).
> Evidence base at `1m`: **58 rows, newest `run_date` 2026-10-23 (33 days ago, bound 40), 5 effective** of 8.
> Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, the Factor Lab** (opened
> 2026-10-12, **age 44 days**; a standing programme, last advanced **19 days** ago at session start, **0**
> after this session's Progress bullet); also open 0.10 (49 d), 0.14 (42 d). Shipped to readers: 21
> commits, 6 methodology changes in the last 7 days. The "held back" branch the line names is this
> session's own, pushed mid-session. No stash.
> **Tests:** before 4557 passed / 1 skipped (runner's baseline, 13:04, GREEN); after **4558 passed / 1 skipped**
> (the new `test_ff3_alpha_recovers_a_planted_intercept`). Gate 2 dry run exit 0. Gate 5 (`rehearse_publish.py
> --changed-since main`) PASSED twice (after the build and on the final tree). Gate 3: `node --check` is refused by
> this session's tool guard; the three payload files are byte-identical to `main`, `index.html` is 990 KB, and
> `lab_data.js` and the new part load without a console error in the lab's browser tests. Gate 4 clean.
> **Owner queue / rotation:** no proposal open (`write_brief.open_proposals()` = `[]`). The 06:00 entry's
> Next: the implied-volatility switch is dated 2026-11-26 (needs tonight's 20:00 refresh) - not today's;
> the lab's bound passing on 11-28 with Friday a retrospective - **taken now** rather than left to the
> weekend slot. That is also `OWNER_FOCUS.md`'s 2026-10-12 item (the lab), whose count is updated.
> 
> ### Did
> - **The queue's Next, `EarningsSurprise`, could not be built honestly.** Foster, Olsen & Shevlin
>   (1984), *Accounting Review* 59(4) 574-603, is on JSTOR; two searches found no working paper or open
>   copy (Stanford GSB has an abstract only). The lab reads the table before writing a tolerance, and
>   OSAP's `SignalDoc` summary is not the table. Recorded in the note's section 1; it stays first in the
>   Revisions queue.
> - **Built experiment 20 instead** - Risk's open row, Ang, Hodrick, Xing & Zhang (2006)'s headline sort
>   on idiosyncratic (FF-3 residual) volatility, on the French library's `Portfolios_Formed_on_RESVAR`.
>   - Read Table VI Panel B of NBER w10852 (pypdf, PDF p. 50); checked it against the text (p. 23).
>   - **Pre-registered in its own commit (`1c9aa54`, pushed) before any return was read**: five means
>     and the gap at experiment 5's tolerances; the volatilities left out of the count so a null matches
>     **1 of 6** (rule 8; experiment 5's null matches 6 of 11 because it scored them); prediction 3 or 4.
>   - **3 of 6 within**, as predicted: calmer three fifths match; fourth 1.25 vs 0.87; most volatile
>     0.76 vs -0.02; raw gap -0.32 (t -1.18) vs -1.06. Since 2001 value-weighted +0.09 (t 0.24).
>   - **Measured, stated, not scored** (no tolerance written first): the gap's three-factor alpha,
>     -0.54 (NW t -3.7) 1963-2000 and -0.58 (t -2.2) since 2001, size loading above 1 - about half the
>     paper's -1.31, and it persisted. The raw miss is composition. `factor_lab.ff3_alpha` (pure Python,
>     equal to a numpy OLS/Newey-West to 3dp, planted-intercept test); the page's open detail and
>     takeaway are checked against the published series in `tests/test_factor_lab.py`.
>   - The five things moved together: `EXPERIMENTS`, `TAKEAWAYS`, `CATEGORY_MAP` (Risk), the
>     methodology's "Rerun the research" paragraph (overview regenerated by `scripts/regenerate_overview.py`,
>     page rebuilt by `scripts/rebuild_page.py` - payload byte-identical), `ASSET_VERSION` 31 and both
>     `?v=` tags. Plus `after_the_paper`'s row.
>   - **Looked at it:** `lab.html#idiosyncratic-volatility` at 1440 and 375 px - no console errors, no
>     horizontal overflow, menu "20 experiments", verdict "3 of 6 ... would match 1 of these 6"; page
> ...

---

If a run says **refused by its publish gates** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

