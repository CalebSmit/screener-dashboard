# Morning Brief - Tuesday 03 November 2026, 09:14

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## ONE THING ONLY YOU CAN DO: log the Claude CLI back in

The code sessions run through the Claude Code CLI on this PC, and its login has
lapsed, so every scheduled session fails within seconds of starting. The data
loop does not use it and is unaffected. Open a terminal on the PC and run:

    claude.cmd auth login

(`.cmd`, because Windows PowerShell blocks the plain `claude` script.) Then
`claude.cmd auth status` should say `"loggedIn": true`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-03 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | VLO, EXPE, MU, SNDK, HST |
| Ranking's lean on growth | +0.073 (OUTSIDE the review's +-0.05 band; 2026-10-13 read -0.009) |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (43 rows, but overlapping windows are not independent; 110 rows across all horizons), newest 1m run date 2026-10-02 (32 days ago, bound 40) |
| Context signals | 17 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 450/451 within 0.5%; operating cash flow 421/435 within 0.5%; net income 412/447 within 0.5%; market value 450/486 within 5%; 4 scored a quarter behind a filing over 30 days old; filings cached 1 days ago |
| Top 25, worth a look | EXPE (#2: equity 4.2% of assets); SNDK (#4: a period behind its filings); HST (#5: net income 1.46x after-tax operating income); DXCM (#6: operating cash flow 36% off the filing); EG (#8: a period behind its filings); PSX (#9: net income 1.28x after-tax operating income); CF (#13: net income 16% off the filing) |
| Data run timing | **19 min** total; slowest `fetch.tickers` 631 s, `fetch.retry` 155 s, `snapshot` 145 s; **51 tickers rate-limited** |
| Code-session login | 2026-12-02 10:28 (29 days) |
| Context coverage | options usable 474/499 (25 quotes-closed) - insider from SEC 499/499 - earnings variability 440 |

## Waiting for a decision

A session researched these, proposed a change and left it for review. The next
session decides and records why; you can also answer in `OWNER_FOCUS.md`.

- **decide whether to fold the Investment category (`asset_growth`, 5%) into Quality at the weight-preserving 18.52% of a 27% Quality (section 3) or keep it standalone, as this note recommends, with the...** - open 1 day - `research/2026-11-02-investment-inside-quality.md`
- **a later session (not the one that wrote this) decides whether to adopt the three pre-registrations in section 7: the S&P 1500 as the first wider universe, the NYSE 20th-percentile market cap as the...** - open 1 day - `research/2026-11-02-universe-expansion-phase0.md`

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-09 (in 6 days) - The context signals' first one-month reading arrives with this run; read the brief's `context_eval` line (CLAUDE.md 0.11). `CLAUDE.md`
- 2026-11-16 (in 13 days) - Re-run the filing cross-check's range table; promote total assets, equity and operating cash flow to a gate at 25 only if each stayed <= 5 on every run. `research/2026-10-14-yahoo-vs-sec.md`

## Things that needed attention

- BASELINE main was RED on arrival: 1 pre-existing failure(s). The session did not cause these:
- tests/test_every_panel_reads_clean.py::test_no_panel_prints_a_value_the_page_could_not_format
- BASELINE fixing a red main outranks the day's focus - gate 1 below will refuse the merge until it is green.

## What changed in the repo

- `b3ed924 Log and owner queue: the growth-exposure proposal decided, two still open`
- `007dbcc Decide the growth-exposure proposal: option 3, and the premise does not hold`
- `c3246c8 NIGHTLY_LOG: 11 false percentages on the live page, and the rule that is now one function`
- `2708177 Fix what the change broke, including one real coverage regression`
- `d1248cf The near-zero rule gets a tripwire, and four more formatters`
- `afc7a5d wip: one near-zero rule for every percentage the page prints`
- `567e061 brief: data run 2026-11-03`
- `5a719b6 data: screener run 2026-11-03 - 499 scored, top: VLO EXPE MU SNDK HST`
- `cf35873 Log the dashboard sweep's second round and queue the expected-move defect first`
- `55ddf5c plan/claims-mutation-audit.md: the rehearsal's catch`
- `1313f2d test_dashboard_printed_figures: the valuation-history recount allows the published rounding of both sides`
- `8207357 OWNER_FOCUS: round 2 done`
- `46dce22 NIGHTLY_LOG: owner-run night, round 2`
- `5d6429d plan/claims-mutation-audit.md: results of the late pass (37 of 77 dashboard rows closed, rows 189-271, the expected-move finding, the queue)`
- `7c688a8 Research engine: rerun all 32 rows on the new config hash (no result changed)`

## The session's own account

> 2026-11-03 - PRODUCT, THROUGH THE NUMBERS. Open the live dashboard as a user would and ask whether it answers what should I look at / should I buy this / should I sell what I hold / how much. The visual design is finished (owner, 2026-10-07), so presentation is not this session's job. When the page cannot answer something the gap is almost always data or methodology: a number the run does not keep, a question no metric answers. Fix that. Check two or three drilldowns still reproduce the engine. A broken or false surface is a defect; one that merely could look different is not.
> 
> **Health (rule 8, all five):** last code session ran? **no** - `logs/nightly-2026-11-02_130000-midday.log`: SESSION DID NOT RUN (session reported an error - Failed to authenticate: OAuth session expired and could not be refreshed). Data loop published? yes - `logs/datarun-2026-11-03_020000.log` (completed; HEALTH: PASS). Evidence base at `1m`: 5 of 8 needed at the 1m horizon (43 rows, but overlapping windows are not independent; 110 rows across all horizons), newest 1m run date 2026-10-02 (32 days ago, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab** (age **22 days**); also open: 0.10 27d; 0.14 20d.
> **Tests:** before **1 failed / 3,721 passed, 1 skipped**; after **3,766 passed, 1 skipped**. Gate 5 rehearsal PASSED.
> **Owner queue / rotation:** took the red baseline, which turned out to be a false figure on the live page - a defect, and ahead of the focus by the prompt's own rule. **Deferred: all three OPEN proposals** (growth-exposure drift, Investment-inside-Quality, universe phase 0, each opened 2026-11-02, age 1 day). Each needs a measurement of its own and the day went on the live defect; the next session should take one. The scheduled 13:00 slot and tomorrow are both eligible readers - none of the three was written by a scheduled session.
> 
> ### Did
> 
> **The red baseline was a false number on the live site, and fixing it properly meant fixing the family.**
> 
> 1. **What was red.** `tests/test_every_panel_reads_clean.py` failed on the payload the 02:00 run
>    published: NVR's analyst-target card read **"$6262.00 -0.0%"** - a mean target of \$6262.00
>    against a \$6265.10 close, so 0.05% below, printed as a signed zero. That test exists because
>    the same family of defect hit twice in two days in October, and it is deliberately **not** in
>    the data loop's publish gate, so it goes red for the next code session instead of discarding a
>    day of evidence. It did its job.
> 
> 2. **The family, not the instance.** Three of these have now reached the live public site and each
>    was fixed inside the one formatter it was found in - BXP's unsigned "0.0%" for a close 0.03%
>    *below* its 200-day average (2026-10-28, the data loop refused that run), AMD's and INTC's
>    signed "-0.0%" (2026-10-29, fixed in `cPlain` alone), and NVR this morning. So the rule is one
>    function now - `pctNearZero` / `pctPlain` in `generate_dashboard.py` - and **fourteen call
>    sites** use it: the analyst-target card, the snapshot and peer tables, the metric workings
>    (`fmtMetric`, `fmtInput`, the earnings-surprise and ROE tables), the compare table's upside
>    column, `cPct`, `cPlain`, `cPp`, the share-class price gap, the 10y-3m curve in both places it
>    is printed, and the track chart's axis ticks and end labels.
>    **The rule:** a value that is **not** zero takes one more decimal rather than printing as a
>    zero; if it is still zero there it is zero to the reader and prints **unsigned**, because a
>    sign on a zero is the glitch and an unsigned zero is true to the precision shown. Both halves
>    are the project's own precedent (the case-against trend line, 2026-10-28; `cPlain`, 10-29).
> 
> 3. **Measured, on today's published payload: 11 figures on the live page were wrong**, out of
>    29,247 percentage lines across all 499 drilldowns, and **no other line moved**
>    (`research/measurements/2026-11-03-percent-lines.py`, run before and after and diffed):
> 
>    | | before | after |
>    |---|---|---|
>    | NVR, analyst target | `$6262.00 -0.0%` | `$6262.00 -0.05%` |
>    | ESS, vs 200-day average | `0.0%` ("Mixed trend 0.0% vs 200-day") | `−0.03%` |
>    | VTR | `0.0%` | `−0.01%` |
>    | ABNB / AMT / PTC, 5 days | `0.0%` | `+0.04%` / `+0.01%` / `+0.03%` |
>    | EQT / TECH, 5 days | `0.0%` | `−0.02%` / `−0.03%` |
>    | BKR, rate sensitivity | `0.00%` | `+0.003%` |
>    | MRSH, analyst target | `+0.0%` | `0.0%` |
> 
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

