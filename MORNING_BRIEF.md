# Morning Brief - Tuesday 20 October 2026, 14:19

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-10-20T13:21:50.724316 |
| Stocks scored | 501 |
| With a price | 501/501 |
| With an analyst target | 497/501 |
| Top 5 | EXPE, APA, INCY, HST, EG |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (33 rows, but overlapping windows are not independent; 88 rows across all horizons), newest 2026-10-12 |
| Context signals | Context signals: 9 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 425/426 within 0.5%; operating cash flow 407/411 within 0.5%; net income 398/421 within 0.5%; market value 447/487 within 5%; filings cached 4 days ago |
| Top 25, worth a look | EXPE (#1: equity 4.2% of assets); HST (#2: net income 1.46x after-tax operating income); CF (#8: net income 16% off the filing); MU (#15: a period behind its filings) |
| Context coverage | options usable 483/501 - insider from SEC 501/501 - earnings variability 443 |

## Waiting for a decision

A session researched these, proposed a change and left it for review. The next
session decides and records why; you can also answer in `OWNER_FOCUS.md`.

- **score each company once: in `run_screener.py`, after `compute_metrics` and before `compute_sector_percentiles`, group universe rows by SEC CIK (falling back to identical `longName`), keep the line with the higher `avg_daily_dollar_volume` (S&P DJI's Designated Listing / FTSE Russell's pricing vehicle; on 2026-10-19 GOOGL, FOXA, NWSA) and remove the other lines from percentiles, scoring and ranking; publish them as unranked entries pointing to the company's ranked line with their own price and the price gap; shipped with a no-two-rows-per-CIK test, the page sentence and its claim, the generated methodology sentence and a changelog entry (section 9). Measured on the 2026-10-19 run: top 25 unchanged; FOXA 47 -> 28, GOOGL 226 -> 197, NWSA 61 -> 52; other Communication Services stocks +1.56 composite points on average** - open 1 day - `research/2026-10-19-share-classes.md`

## What changed in the repo

- `358cd72 nightly 2026-10-20 (13:00) - bank ROE/ROA proposal ACCEPTED, with the live run that publishes it`
- `a474e4d measure: what the bank ROE/ROA change did on the run that published it`
- `b844ff2 methodology: bank ROE and ROA are net income over AVERAGE balances, not the source's ratios`
- `0ab353e measure: review of the bank ROE/ROA proposal - 46 of 46 available, 37 of 37 against the filings`
- `ec17b70 brief: code session 2026-10-20`
- `688260b log: 2026-10-20 session - bank P/B proposal reviewed and accepted`
- `2e74078 engine: rerun complete on the 2026-10-20 code (30 of 30), and no number moved`
- `742d5e2 engine: rerun on the 2026-10-20 code (chunk 7 of 8)`
- `760b429 engine: rerun on the 2026-10-20 code (chunk 6 of 8)`
- `5ec38de engine: rerun on the 2026-10-20 code (chunk 5 of 8)`
- `698afc2 engine: rerun on the 2026-10-20 code (chunk 4 of 8)`
- `a2770e2 engine: rerun on the 2026-10-20 code (chunk 3 of 8)`
- `81f978e engine: rerun on the 2026-10-20 code (chunk 2 of 8)`
- `16a6413 engine: rerun on the 2026-10-20 code (chunk 1 of 8)`
- `9c1a8d6 methodology: bank P/B scored as market value over common equity, published`

## The session's own account

> 2026-10-20 (13:00 session) - QUEUE. The bank ROE/ROA proposal reviewed and ACCEPTED: 60% of bank Quality rested on two ratios with no published definition
> 
> **Health (rule 8, all five):** last code session ran? **yes** - `logs/nightly-2026-10-20_060001.log`
> ends "Run complete: shipped to main", tagged `good/2026-10-20`. Data loop published? **yes** -
> `logs/datarun-2026-10-20_020001.log` ends "Data loop complete", HEALTH: PASS. Evidence base at
> `1m` = **33 rows, newest 2026-09-18 (32 days ago, bound 40), 4 effective**. Priority 0:
> `allow_auto_apply` **false**, 4 of 8 effective observations - unchanged, and nothing here rests on
> it. Top open roadmap item: **0.11, the context layer** (opened 2026-10-08, **age 12 days**) -
> deferred again today, see below.
> **Tests:** before **2,830 passed, 1 skipped**; after **2,852 passed, 1 skipped** (+22, exactly the
> new module; no pre-existing failures either side).
> **Owner queue / rotation:** I am the day's **second** session, so I read the 06:00 entry first and
> took its **Next** verbatim: decide the second of the three proposals opened 2026-10-19. That is
> also an open-proposal obligation, which outranks the rotation (`prompts/nightly.md` section 1).
> **Deferred:** the context layer (0.11) and the third proposal (share classes). Nothing there was
> started, so nothing is half-done. I did not revisit or reverse anything the 06:00 session decided.
> 
> ### Did
> 
> **Reviewed, re-measured and ACCEPTED the bank ROE/ROA proposal**
> (`research/2026-10-19-bank-roe-roa.md` section 7; the review is now section 9 of that note).
> ROE and ROA are 35 and 25 of the bank Quality table's 100 points - about **13.2 composite points
> for each of the 46 bank-like stocks** - and both were read straight from the data source's
> `.info` `returnOnEquity` / `returnOnAssets`.
> 
> - **Why that was wrong, and not merely untidy.** Neither the provider nor `yfinance` documents
>   either field. The engine threw out `earningsGrowth` for exactly this reason - "a black-box input
>   with no verifiable definition". Reverse-engineered, the source's ROE is *consolidated* net income
>   over average *total* equity **including minority interests**.
> - **The decisive measurement, against the companies' own SEC filings** (TTM `NetIncomeLoss` over
>   averaged `StockholdersEquity` / `Assets` from the weekly `companyfacts` cache): the new
>   construction agrees **within 0.5% for 37 of 37** comparable stocks (ROA 36 of 37, worst 0.79%);
>   the source's ratios agree **within 2% for 21 (ROE) and 12 (ROA) of 35**. The change makes the
>   metric agree with the primary source.
> - **Now:** `roe = netIncome / ((totalEquity + totalEquity_prior) / 2)`, `roa` the same on total
>   assets, both balances required positive, **no fallback** to the source's ratio and none to an
>   ending balance (the Phase 13 F19 rule). The prior balances are the ones `_year_ago_position`
>   matches by date (330-400 days).
> - **Shipped with its frontend, same commit** (CLAUDE.md 0.8c): `metric_lineage` LINEAGE / RECOMPUTE
>   / one **exact** EQUATIONS template each, `pit_record` (both fields leave `FIELD_CLASSES` and
>   `market_fields` stops building them - they were a *third* construction, on ending balances),
>   `scoring_schema` 13 -> 14, the generated methodology paragraph, `run_audit`'s field provenance,
>   claim `overview.bank_roe_roa`, `METHODOLOGY_CHANGELOG.md`, `tests/test_bank_roe_roa.py` (22).
> - **A full screener run was made in the session** so the fix is live now rather than at 02:00:
>   **501 scored**, `check_run_health` **PASS**, `scripts/check_published_claims.py` **PASS (1,152
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

