# Morning Brief - Tuesday 08 September 2026, 02:12

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-09-08T02:00:04.268668 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | HST, EXPE, APA, CAH, VLO |
| Evidence for weight changes | 3 of 8 needed at the 1m horizon (8 rows, but overlapping windows are not independent; 34 rows across all horizons), newest 2026-09-01 |

## What changed in the repo

- `c72d9c1 data: screener run 2026-09-08 - 502 scored, top: HST EXPE APA CAH VLO`
- `51cfa15 brief: code session 2026-09-07`
- `cde9089 log: 2026-09-07 research session - the Revisions category has no revisions`
- `7de239c docs: correct the false "revisions data requires FactSet/Refinitiv" claim`
- `d09e7bc research: the Revisions category contains no revisions`
- `2aa2102 brief: data run 2026-09-07`
- `e2afbe4 data: screener run 2026-09-07 - 502 scored, top: HST EXPE APA CAH VLO`

## The session's own account

> 2026-09-07 - RESEARCH. Take one specific thing - a factor, a metric, a threshold, a construction rule - and learn it properly, from the literature AND from documented practice, in this one session. Real citations, effect sizes, the conditions the effect held under, and how quant shops and institutional screens actually handle it. Where academia and practice disagree, say so and say why. A dated note in research/, complete today. No production code.
> 
> ### Health numbers (rule 8, all five)
> 
> | Check | Reading |
> |---|---|
> | Last code session ran? | `logs/nightly-2026-09-04_060001.log` - "Run complete: shipped to main" (09-05/09-06 were the weekend) |
> | Data loop published? | `logs/datarun-2026-09-07_020001.log` - "Data loop complete", HEALTH: PASS, 502 scored |
> | Evidence base | **33 rows, newest 2026-08-31, 3 effective observations at `1m`** (8 raw) against a gate of 8 |
> | Priority 0 | DONE 2026-08-24, not reopened |
> | Top open roadmap item | **Priority 4, deterministic per-stock summaries - owner directive 2026-08-10, open 28 days** |
> 
> **Tests:** before 965/965, after 965/965 (no pre-existing failures; no tests added - research day)
> **Data loop:** healthy. Evidence base moved 32 -> 33 rows, newest 08-28 -> 08-31.
> **Owner queue / rotation:** `OWNER_FOCUS.md` **Open** is empty. Nothing deferred.
> ISO week 37, Monday - research day, taken as the focus.
> 
> ### Did
> 
> **Researched the Revisions category and found it contains no revisions.**
> `research/2026-09-07-revisions-category-has-no-revisions.md`.
> 
> The category carries **10% of the composite**. All five scored metrics are past
> earnings surprises, a price-target *level*, or short interest. The three
> surprise metrics come from the same four rows of `Ticker.earnings_history`
> (`factor_engine.py:1089-1130`) and are **78% of the category = 7.8% of the
> composite**.
> 
> Three findings, in order of how much they should change what we do:
> 
> **1. The category's public rationale rests on an effect documented as absent in
> this universe.** `SCREENER_OVERVIEW.md:149` justifies it with "When a company
> consistently beats earnings estimates, the stock price usually follows - but
> with a lag, which creates an opportunity." That is post-earnings announcement
> drift. Martineau (2022, *Critical Finance Review* 11(3-4)) finds PEAD
> **non-existent for all-but-microcap stocks since 2006**, with the 2016-2019
> 60-day coefficient significantly *negative*; the return moved to the
> announcement date (large-stock BHAR[0,1] ~20bps in 1984-1990 -> ~120bps in
> 2016-2019). His surprise measure is the analyst-estimate kind this screener
> computes, and every S&P 500 name is "all-but-microcap".
> 
> **2. The claim that a revisions metric is impossible is false, and it was on the
> public methodology page.** `config.yaml` and `SCREENER_OVERVIEW.md` (twice) said
> forward-EPS revisions "would require a paid data source like FactSet or
> Refinitiv I/B/E/S". Measured today against the installed yfinance 0.2.66:
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

