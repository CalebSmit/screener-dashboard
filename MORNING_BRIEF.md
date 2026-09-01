# Morning Brief - Tuesday 01 September 2026, 02:12

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

- `c916b5e data: screener run 2026-09-01 - 502 scored, top: HST EXPE APA CF VLO`
- `490c7cf brief: code session 2026-08-31`
- `52f0d0d log: 2026-08-31 research session`
- `be47a76 docs: research/README described the old split rotation`
- `0603712 research: the size factor in a large-cap-only universe`
- `a5671f4 brief: data run 2026-08-31`
- `f4db5cd data: screener run 2026-08-31 - 502 scored, top: HST EXPE EIX APA CF`

## The session's own account

> 2026-08-31 - RESEARCH. Take one specific thing - a factor, a metric, a threshold, a construction rule - and learn it properly, from the literature AND from documented practice, in this one session. Real citations, effect sizes, the conditions the effect held under, and how quant shops and institutional screens actually handle it. Where academia and practice disagree, say so and say why. A dated note in research/, complete today. No production code.
> 
> ### Health numbers (rule 8)
> 
> | Check | Reading |
> |---|---|
> | Last code session ran? | `logs/nightly-2026-08-29_121115.log` - catch-up session, shipped to main |
> | Data loop published? | `logs/datarun-2026-08-31_020001.log` - "Data loop complete", HEALTH: PASS, 502 scored |
> | Evidence base | **28 rows, newest 2026-08-24, 3 effective observations at `1m`** (8 raw) |
> | Priority 0 | Fixed 2026-08-24, holding |
> 
> **Tests:** before 853/853, after 853/853
> **Data loop:** healthy - ran 02:00, all coverage/dispersion checks passed, published to main
> **Owner queue:** empty - nothing under **Open** in `OWNER_FOCUS.md`, so the
> rotation stood. Nothing was deferred.
> **Rotation:** ISO week 36, Monday. Research day.
> 
> **The research note is written. That is the whole session.** It had been skipped
> five consecutive times, each skip for a real defect and each defensible in
> isolation, but the rotation had produced exactly **one** note in a month. Today
> nothing was broken, so there was no excuse.
> 
> ### Did
> 
> Wrote `research/2026-08-31-size-factor-in-a-large-cap-universe.md`. The question:
> the screener spends 5% of composite on `size_log_mcap` = `-ln(mcap)` inside the
> S&P 500, a universe with no small caps. Does a size tilt belong here, and is this
> the right way to build one?
> 
> **Answer: keep it at 5%, and the reason is better than the one it had.** Four
> findings, all measured on the published payload from today's 02:00 run:
> 
> - **The tilt is junk-seeking on average, exactly as the literature predicts.**
>   The 50 names it promotes most vs the 50 it demotes: median cap $15.0B vs
>   $264.1B, quality 49.3 vs 55.6, risk 44.9 vs 60.1, volatility 0.33 vs 0.30.
>   That is the pattern Asness et al. (2018) identify as the reason raw SMB fails,
>   and that MSCI concedes in its own Low Size brochure.
> - **But the composite already controls for the junk where the product points.**
>   Comparing the top 50 with and without the size category: quality moves
>   **-0.20**, risk **-0.47**, median cap $38.5B -> $30.4B. The 22% quality weight
>   removes the junk before size can promote it. Combined with the S&P 500's own
>   GAAP-profitability entry gate, this screener is running much closer to the
>   quality-controlled version of the factor (t = 4.89) than the raw one
>   (t = 1.23). **That defence did not exist before today; the weight was
>   previously unexamined.**
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

