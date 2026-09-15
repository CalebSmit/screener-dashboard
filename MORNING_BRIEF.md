# Morning Brief - Tuesday 15 September 2026, 02:13

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **failed** - last ran today |
| Dashboard data from | 2026-09-15T02:00:04.708829 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | EXPE, HST, VLO, APA, CAH |
| Evidence for weight changes | 3 of 8 needed at the 1m horizon (11 rows, but overlapping windows are not independent; 42 rows across all horizons), newest 2026-09-08 |

## What changed in the repo

- `3072180 data: screener run 2026-09-15 - 502 scored, top: EXPE HST VLO APA CAH`
- `8a14103 brief: code session 2026-09-14`
- `58cee59 docs: north-star plan corrected by today's research, plus the session log`
- `463760f research: sell discipline, hold bands, and what the evidence forbids`
- `caf830a brief: data run 2026-09-14`
- `6b8d071 data: screener run 2026-09-14 - 502 scored, top: EXPE HST VLO APA CAH`

## The session's own account

> 2026-09-14 - RESEARCH. Take one specific thing - a factor, a metric, a threshold, a construction rule - and learn it properly, from the literature AND from documented practice, in this one session. Real citations, effect sizes, the conditions the effect held under, and how quant shops and institutional screens actually handle it. Where academia and practice disagree, say so and say why. A dated note in research/, complete today. No production code.
> 
> **Health (rule 8, all five):**
> 
> | Check | Reading |
> |---|---|
> | Last code session ran? | **Yes** - `logs/nightly-2026-09-11_060001.log` ends "Run complete: shipped to main", tagged `good/2026-09-11` |
> | Data loop published? | **Yes** - `logs/datarun-2026-09-14_020001.log` ends "Data loop complete", HEALTH: PASS, 502 scored, top EXPE HST VLO APA CAH |
> | Evidence base | **41 rows, newest 2026-09-07, 3 effective observations at `1m`** (11 raw) against a gate of 8 |
> | Priority 0 | Fixed 2026-08-24, not weakened today. Research-only session; `_effective_observations()` and every scoring path untouched |
> | Top open roadmap item | **Priority 5, the sell-side workflow - 40 days old.** Not built today, but this session is the research that unblocks it - see *Owner queue / rotation* |
> 
> **Tests:** before 1193/1193, after **1193/1193**. No production code changed;
> the run is a no-new-failures check, not a claim of new coverage.
> 
> **Owner queue / rotation:** `OWNER_FOCUS.md` **Open** is empty, so nothing was
> deferred. Monday's focus taken as written.
> 
> **On priority 5, and why this counts as progress on it.** Its age has now been
> written down for five consecutive sessions, each time as "still untouched".
> Monday is research day and priority 5 is a build item, so the two do not compete
> directly - but the topic was chosen so that they stop pulling against each
> other. `plan/dashboard-north-star.md` parked "what sell disciplines have
> evidence behind them?" as Monday research question 3, and priority 5 is the
> build that question exists to inform. That question is now answered. Thursday
> can build from a note instead of from intuition, which is the whole point of
> having a research day ahead of a build day.
> 
> ### Did
> 
> **One research note, complete today:
> `research/2026-09-14-sell-discipline-and-hold-bands.md`.** Five papers and three
> index-provider methodologies, read from primary sources rather than summaries -
> `pypdf` against the downloaded PDFs, because `WebFetch` cannot read a PDF and
> returns a confident "I cannot extract this" that is easy to mistake for "the
> source does not say".
> 
> **The headline finding is that selling is the part of the process where
> documented professional skill disappears.** Akepanidtaworn, Di Mascio, Imas &
> Schmidt (2023, *JF* 78(6)) track 783 institutional portfolios averaging $573M,
> 2000-2016, 4.4M trades. Against counterfactuals built from the managers' own
> holdings: buys beat a random-buy counterfactual by **over +100 bp/year**; sells
> **underperform a factor-neutral random-sell counterfactual by -80 bp/year**.
> That deficit is larger than the fee these managers charge.
> 
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

