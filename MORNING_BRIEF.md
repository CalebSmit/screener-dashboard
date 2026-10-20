# Morning Brief - Tuesday 20 October 2026, 08:06

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-10-20T06:16:57.643100 |
| Stocks scored | 501 |
| With a price | 501/501 |
| With an analyst target | 497/501 |
| Top 5 | EXPE, HST, APA, INCY, VLO |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (33 rows, but overlapping windows are not independent; 88 rows across all horizons), newest 2026-10-12 |
| Context signals | Context signals: 9 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 425/426 within 0.5%; operating cash flow 407/411 within 0.5%; net income 398/421 within 0.5%; market value 447/487 within 5%; filings cached 4 days ago |
| Top 25, worth a look | EXPE (#1: equity 4.2% of assets); HST (#2: net income 1.46x after-tax operating income); CF (#8: net income 16% off the filing); MU (#15: a period behind its filings) |
| Context coverage | options usable 489/501 (12 quotes-closed) - insider from SEC 501/501 - earnings variability 443 |

## Waiting for a decision

A session researched these, proposed a change and left it for review. The next
session decides and records why; you can also answer in `OWNER_FOCUS.md`.

- **for bank-like stocks, `roe = netIncome / ((totalEquity + totalEquity_prior) / 2)` and `roa = netIncome / ((totalAssets + totalAssets_prior) / 2)` (NaN unless both balances are positive; no fallback to Yahoo's `returnOnEquity` / `returnOnAssets` or to an ending balance), shipped with `metric_lineage` LINEAGE / RECOMPUTE / EQUATIONS, `pit_record.FIELD_CLASSES`, `scoring_schema` 12, the generated methodology text and a changelog entry (section 7)** - open 1 day - `research/2026-10-19-bank-roe-roa.md`
- **score each company once: in `run_screener.py`, after `compute_metrics` and before `compute_sector_percentiles`, group universe rows by SEC CIK (falling back to identical `longName`), keep the line with the higher `avg_daily_dollar_volume` (S&P DJI's Designated Listing / FTSE Russell's pricing vehicle; on 2026-10-19 GOOGL, FOXA, NWSA) and remove the other lines from percentiles, scoring and ranking; publish them as unranked entries pointing to the company's ranked line with their own price and the price gap; shipped with a no-two-rows-per-CIK test, the page sentence and its claim, the generated methodology sentence and a changelog entry (section 9). Measured on the 2026-10-19 run: top 25 unchanged; FOXA 47 -> 28, GOOGL 226 -> 197, NWSA 61 -> 52; other Communication Services stocks +1.56 composite points on average** - open 1 day - `research/2026-10-19-share-classes.md`

## What changed in the repo

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
- `a23efa2 wip: bank P/B is market value over common equity at the balance-sheet date`
- `2cb2060 review: the bank price-to-book proposal re-measured on a second run and a fresh probe`
- `4dd1e9a brief: data run 2026-10-20`
- `ce4f70e data: screener run 2026-10-20 - 501 scored, top: EXPE HST APA INCY VLO`
- `dfc7ad0 engine: rerun complete on the final 2026-10-19 code (30 of 30, published figures unchanged); evening log addendum`

## The session's own account

> 2026-10-20 (06:00 session) - QUEUE. The bank price-to-book proposal reviewed and ACCEPTED: Berkshire's P/B was 0.0010, not 1.52, and had been since March
> 
> **Health (rule 8, all five):** last code session ran? **the 06:00 one ran and shipped**
> (`logs/nightly-2026-10-19_060001.log` - its gates failed at the time and an owner-run session
> finished and merged its work the same day); the 13:00 slot **stood down** for the owner marker, as
> designed. Data loop published? **yes** - `logs/datarun-2026-10-20_020001.log` ends "Data loop
> complete", HEALTH: PASS, 501 scored. Evidence base at `1m` = **33 rows, newest 2026-09-18 (32 days
> ago, bound 40), 4 effective**. Priority 0: `allow_auto_apply` **false**, 4 of 8 observations. Top
> open roadmap item: **0.11, the context layer** (opened 2026-10-08, **age 12 days**) - deferred
> today, see below.
> **Tests:** before **2,817 passed, 1 skipped**; after **2,830 passed, 1 skipped** (13 new).
> **Owner queue / rotation:** `MORNING_BRIEF.md` listed **three** proposals waiting for a decision,
> all opened 2026-10-19, so deciding one was the day's first job ahead of the Tuesday focus
> (`prompts/nightly.md` section 1). Took the bank price-to-book one, because it was the only one of
> the three carrying a **demonstrably false number on the live site**. The context-layer item
> (0.11) and the other two proposals were not started; nothing there is half-done.
> 
> ### Did
> 
> **Reviewed, re-measured and ACCEPTED the bank price-to-book proposal**
> (`research/2026-10-19-bank-price-to-book.md` section 8, raised by the 2026-10-19 session). Bank P/B
> is 60 of the bank Valuation table's 100 points - about **13.2 composite points for each of the 46
> bank-like stocks** - and it read the data source's `.info` `priceToBook`, which is
> `currentPrice / bookValue` with `bookValue` a **per-share** figure.
> 
> - **The live defect, confirmed on this morning's published payload before anything was changed:**
>   BRK-B's `pb_ratio` was **0.0010191567** at the **98.9th** percentile of the bank P/B ranking -
>   the cheapest bank-like stock by a factor of nearly a thousand - and it was ranked **65th** on the
>   strength of it. Its book value is per **Class A** share ($522,226) against a **Class B** price
>   (about $532). Checked against the company's own filing rather than another of the source's rows:
>   filed stockholders' equity of **$747,910,000,000** (equal to the probe's common equity *and* to
>   the run's own `totalEquity`) over a market value of $1,139,349,913,600 is **1.5234**. The
>   published figure was **1,494.8x** too low, which is the 1,500:1 A-to-B ratio. It had been
>   published every weekday since March: at least the 91.4th percentile in **all 52** improvement
>   snapshots carrying it, median 98.3.
> - **Now:** `pb_ratio = marketCap / commonEquity`, both > 0 required, where `commonEquity` is a new
>   raw field - the quarterly balance sheet's "Common Stock Equity" at the same column `totalEquity`
>   is read from. **No fallback** to `priceToBook` or `bookValue` (the Phase 13 F19 rule). A
>   company-level ratio has no per-share unit to get wrong, and it puts the metric on the same
>   balance sheet as the rest of the record - the source's book was a quarter newer than everything
>   else scored for **18 of the 46**.
> - **Shipped with its frontend, same commit** (CLAUDE.md 0.8c): `metric_lineage` LINEAGE /
>   RECOMPUTE / EQUATIONS, `pit_record.FIELD_CLASSES`, `scoring_schema` 12 -> 13, the generated
>   methodology paragraph, claim `overview.bank_price_to_book`, `METHODOLOGY_CHANGELOG.md`.
> - **A full screener run was made in the session** so the new field is in the live payload rather
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

