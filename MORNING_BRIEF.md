# Morning Brief - Wednesday 16 September 2026, 02:13

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-09-16T02:00:02.876409 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | EXPE, HST, VLO, CAH, BBY |
| Evidence for weight changes | 3 of 8 needed at the 1m horizon (11 rows, but overlapping windows are not independent; 43 rows across all horizons), newest 2026-09-09 |

## What changed in the repo

- `fdd85a4 data: screener run 2026-09-16 - 502 scored, top: EXPE HST VLO CAH BBY`
- `cbeb38f brief: code session 2026-09-15`
- `6cedc64 docs: record the sell-side surface and what it deliberately left out`
- `c9b870d dashboard: a sell-side surface - My Holdings`
- `2ba6586 summary: say which category moved, not just how far`
- `2934891 brief: data run 2026-09-15`
- `3072180 data: screener run 2026-09-15 - 502 scored, top: EXPE HST VLO APA CAH`

## The session's own account

> 2026-09-15 - PRODUCT. Open the live dashboard as a user would. Does it answer what should I look at / should I buy this / should I sell what I hold / how much? Read plan/dashboard-inventory.md before building anything - the most likely failure is rebuilding what exists. Ship a dashboard change, or write down precisely what it cannot answer and why.
> 
> **Health (rule 8, all five):**
> 
> | Check | Reading |
> |---|---|
> | Last code session ran? | **Yes** - `logs/nightly-2026-09-14_060001.log` ends "Run complete: shipped to main", tagged `good/2026-09-14` |
> | Data loop published? | **Yes** - `logs/datarun-2026-09-15_020001.log` ends "Data loop complete", HEALTH: PASS, 502 scored, top EXPE HST VLO APA CAH |
> | Evidence base | **42 rows, newest 2026-09-08, 3 effective observations at `1m`** (11 raw) against a gate of 8. Up from 41 rows yesterday - moving |
> | Priority 0 | Fixed 2026-08-24, not weakened. No scoring path, weight, threshold or `_effective_observations()` call was touched today |
> | Top open roadmap item | **Priority 5, the sell-side workflow - 41 days old. The list half shipped today.** The remaining half (a hold band) is blocked on measurement, not design - see below. Next unblocked item is **Priority 3, backtest v2**, whose plan file dates to 2026-08-25 - **21 days** |
> 
> **Tests:** before **1193/1193**, after **1264/1264**. 71 new tests, zero
> failures either side.
> 
> **Owner queue / rotation:** `OWNER_FOCUS.md` **Open** is empty, so nothing was
> deferred. Tuesday's product focus taken as written, and it pointed at the same
> place the roadmap did: Priority 5 is a *product* gap and Tuesday is the product
> day, so for once the rotation and the north star wanted the same thing.
> 
> **On running ahead of Wednesday's synthesis.** Monday's note (2026-09-14) parked
> five design questions for the 09-16 synthesis. I built anyway, and only the part
> that none of those questions gate: the **list**, with no threshold of any kind.
> Question 1 - band width - is untouched and still open, which is the point.
> `CLAUDE.md` records nine consecutive sessions that produced real work and
> shipped no north-star item because something smaller always looked more urgent;
> deferring a 41-day-old product item on the product day, when the research it was
> waiting for landed yesterday, would have been the tenth.
> 
> ### Did
> 
> **Shipped the sell-side workflow's list half: a "My Holdings" panel.** Between
> Top 5 and What Changed. A `localStorage` list under `screener_holdings_v1`
> holding **tickers and nothing else**, rendering every saved name each run as a
> card: rank, composite, an eight-category score-and-delta strip, and the review
> sentences already baked into `stock_detail[t]["summary"]`. Above it, a
> concentration line - names, sectors, largest sector share, how many sit inside
> the top 25 and the top 100, how many carry a trap flag.
> 
> **It costs nothing in payload.** It is a *view* over fields `stock_detail`
> already carried. `plan/dashboard-inventory.md` warns that the likeliest failure
> here is rebuilding what exists; the useful version of heeding that was noticing
> that `stock_summary.py` already produces build-time, advice-screened sentences
> covering what changed, what is flagged and what the score rests on. The panel
> renders those rather than composing its own prose in the browser, which keeps
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

