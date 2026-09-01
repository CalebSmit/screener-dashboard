# Morning Brief - Tuesday 01 September 2026, 06:23

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-09-01T02:00:03.870132 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | HST, EXPE, APA, CF, VLO |
| Evidence for weight changes | 3 of 8 needed at the 1m horizon (8 rows, but overlapping windows are not independent; 29 rows across all horizons), newest 2026-08-25 |

## What changed in the repo

- `97d5f10 docs: the winsorization rationale was false, so correct it rather than soften it`
- `f1f5855 test: lock down the no-winsorization property and the megacap regression`
- `4d47c42 fix: stop clipping metric values the screener then ranks`
- `c2cf5c4 brief: data run 2026-09-01`
- `c916b5e data: screener run 2026-09-01 - 502 scored, top: HST EXPE APA CF VLO`
- `490c7cf brief: code session 2026-08-31`
- `52f0d0d log: 2026-08-31 research session`
- `be47a76 docs: research/README described the old split rotation`
- `0603712 research: the size factor in a large-cap-only universe`
- `a5671f4 brief: data run 2026-08-31`
- `f4db5cd data: screener run 2026-08-31 - 502 scored, top: HST EXPE EIX APA CF`

## The session's own account

> 2026-09-01 - PRODUCT. Open the live dashboard as a user would. Does it answer what should I look at / should I buy this / should I sell what I hold / how much? Read plan/dashboard-inventory.md before building anything - the most likely failure is rebuilding what exists. Ship a dashboard change, or write down precisely what it cannot answer and why.
> 
> ### Health numbers (rule 8)
> 
> | Check | Reading |
> |---|---|
> | Last code session ran? | `logs/nightly-2026-08-31_060001.log` - "Run complete: shipped to main" |
> | Data loop published? | `logs/datarun-2026-09-01_020001.log` - "Data loop complete", HEALTH: PASS, 502 scored |
> | Evidence base | **29 rows, newest 2026-08-25, 3 effective observations at `1m`** (8 raw) |
> | Priority 0 | Fixed 2026-08-24, holding |
> 
> **Tests:** before 853/853, after **872/872**
> **Data loop:** healthy - ran 02:00, all coverage and dispersion checks passed, published to main
> **Owner queue:** empty - nothing under **Open** in `OWNER_FOCUS.md`. Nothing deferred.
> **Rotation:** ISO week 36, Tuesday. Product day - and the work was a product defect,
> so no swap was needed.
> 
> ### Did
> 
> **Fixed the live site publishing false numbers.** The dashboard showed AAPL,
> NVDA, MSFT, GOOG, GOOGL and AMZN with an identical market capitalisation of
> **$2,802.0B**. Nvidia's true figure is **$5,331.2B** - understated by 47%, or
> $2.5 trillion. This is a product-day finding in the most literal sense: it is
> what a user sees, and it is wrong.
> 
> The cause was `winsorize_metrics()`, which clipped the top and bottom 1% of
> every metric onto one boundary value four lines before the ranking step. The
> clipped number was then published as the stock's `raw` value. Removed; replaced
> by `flag_metric_outliers()`, which reports the same tails into the data-quality
> log and does not touch the frame. Changelog 2026-09-01;
> `tests/test_no_winsorization.py`, 18 tests.
> 
> This was the item the 2026-08-31 session identified as the next session's work.
> It is done, and the diagnosis it left was right in substance - though its two
> headline numbers were slightly off and are corrected here: the tie value is
> **$2,802.0B**, not $2,873.8B, and on the current payload the signature covers
> **33 continuous metrics / 301 collapsed cells**, not 27 / 282.
> 
> **Measured blast radius**, all on the published `dashboard_data.js` from today's
> 02:00 run:
> 
> - **301 (stock, metric) cells across 33 continuous metrics, on 159 of 502
>   stocks**, carried a clipped value instead of the fetched one.
> - **58 (metric, sector) tie groups** collapsed two or more stocks onto a single
>   percentile. Worst: 4 Energy names shared one `beta` rank in a 21-stock sector,
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

