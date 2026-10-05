# Morning Brief - Monday 05 October 2026, 06:12

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-10-05T02:00:06.261856 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | EXPE, HST, BBY, VLO, CAH |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (23 rows, but overlapping windows are not independent; 68 rows across all horizons), newest 2026-09-28 |

## What changed in the repo

- `46c92bf log: 2026-10-05 research session`
- `f2db1f0 research: fundamental reporting lag - filed-date alignment rule for backtest v2 step 3`
- `56e3738 brief: data run 2026-10-05`
- `5a4bb9e data: screener run 2026-10-05 - 502 scored, top: EXPE HST BBY VLO CAH`

## The session's own account

> 2026-10-05 - RESEARCH. Take one specific thing - a factor, a metric, a threshold, a construction rule - and learn it properly, from the literature AND from documented practice, in this one session. Real citations, effect sizes, the conditions the effect held under, and how quant shops and institutional screens actually handle it. Where academia and practice disagree, say so and say why. A dated note in research/, complete today. No production code.
> 
> **Health (rule 8, all five):** last code session ran? **yes** -
> `logs/nightly-2026-10-02_060001.log` ends "Run complete: shipped to main",
> tagged `good/2026-10-02` | data loop published? **yes** -
> `logs/datarun-2026-10-05_020001.log` "HEALTH: PASS - safe to publish", ends
> "Data loop complete", 502 scored, top EXPE HST BBY VLO CAH | evidence base at
> `1m` = **23 rows, newest 2026-09-04 (31 days ago, bound 40), 4 effective** -
> the middle of the 30-33-day steady state, healthy | priority 0 **fixed
> 2026-08-24, not weakened** (`_effective_observations()` still gates,
> `allow_auto_apply` still false, 4 effective against a gate of 8) | top open
> roadmap item: **priority 3, backtest v2 - 41 days old**, and today produced
> the alignment rule its step 3 needs, so it advanced rather than aged.
> 
> **Tests:** before **1680/1680**, after **1680/1680** (unchanged - no
> production code, per the Monday rule; the session adds three research files)
> 
> **Owner queue / rotation:** `OWNER_FOCUS.md` **Open is empty**, so nothing to
> claim or move to Done. Took the nominal **Monday research** focus and pointed
> it at the top roadmap item: backtest v2 step 3 needs a rule for aligning
> fundamentals with prices, and no note in `research/` covered it. Nothing was
> deferred - both loops healthy, baseline green at 1680/1680.
> 
> ### Did
> 
> **Researched the fundamental reporting lag - the construction rule for *when*
> accounting data becomes usable to a screen - from the literature, documented
> practice, and a live measurement of this exact universe.**
> `research/2026-10-05-fundamental-reporting-lag.md`;
> `research/measurements/2026-10-05-edgar-reporting-lag.py` (+ committed JSON).
> 
> - **The literature is already two-tier.** Fama & French (1992) impose a
>   six-month minimum gap - a deliberately conservative guess from an era
>   without machine-readable filing dates. Hou, Xue & Zhang (2020) refine it:
>   earnings usable from the announcement date (RDQ), everything else lagged 4
>   months, because the economy-wide median reporting lag in their sample was
>   46-52 days and only ~37% of earnings announcements carry a balance sheet.
>   Asness & Frazzini (2013) add the other half of the ratio: lag the
>   fundamental because you must, **never lag the price** - their timely-price
>   value construction earns 305-378 bps/yr of alpha vs five-factor models.
> - **Practice buys dates, not conventions.** Compustat Point-in-Time is the
>   paid product; SEC deadlines (60-day 10-K / 40-day 10-Q for large
>   accelerated filers, i.e. the whole S&P 500) are the legal bound; and SEC
>   EDGAR's XBRL `companyconcept` API gives every fact with its own `filed`
>   date free - **verified by a live call this session**: 146 USD facts for
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

