# Morning Brief - Tuesday 22 September 2026, 06:19

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-09-22T02:00:03.386380 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | EXPE, HST, BBY, VLO, APA |
| Evidence for weight changes | 3 of 8 needed at the 1m horizon (13 rows, but overlapping windows are not independent; 49 rows across all horizons), newest 2026-09-15 |

## What changed in the repo

- `2415428 docs: correct the Node-driven test count to 17`
- `fb5eed1 docs: changelog, inventory and the corrected research note`
- `aea0946 build: regenerate dashboard artifacts with the Concentration block`
- `f69f056 feat(dashboard): Concentration block answers "how much" without a weight`
- `b017dc8 brief: data run 2026-09-22`
- `e6fa846 data: screener run 2026-09-22 - 502 scored, top: EXPE HST BBY VLO APA`
- `efccf2e brief: code session 2026-09-21`
- `b853d40 log: 2026-09-21 research session - position sizing note`
- `fe2d862 research: position sizing and the "how much" question`
- `d8efe45 brief: data run 2026-09-21`
- `120daba data: screener run 2026-09-21 - 502 scored, top: EXPE HST BBY VLO CAH`

## The session's own account

> 2026-09-22 - PRODUCT. Open the live dashboard as a user would. Does it answer what should I look at / should I buy this / should I sell what I hold / how much? Read plan/dashboard-inventory.md before building anything - the most likely failure is rebuilding what exists. Ship a dashboard change, or write down precisely what it cannot answer and why.
> 
> **Health (rule 8, all five):** last code session ran? **yes** -
> `logs/nightly-2026-09-21_060000.log` ends "Run complete: shipped to main", tagged
> `good/2026-09-21` | data loop published? **yes** -
> `logs/datarun-2026-09-22_020001.log` ends "Data loop complete", 502 scored |
> evidence base at `1m` = **13 rows, newest 2026-08-21 (32 days ago, bound 40), 3
> effective** - inside the band and still clearing the 08-15..08-19 outage |
> priority 0 **fixed** (2026-08-24, untouched) | top open roadmap item: **priority 3,
> backtest v2, 28 days old**
> **Tests:** before 1378/1378, after **1411/1411** (+33 new, no pre-existing failures)
> **Owner queue / rotation:** `OWNER_FOCUS.md` **Open is empty**, so the rotation
> governed. Took Tuesday product, and specifically item 2 of yesterday's "Next"
> list. Nothing deferred.
> 
> ### Did
> 
> Shipped the **Concentration block** on My Holdings - `holdingsConcentration(rows)`
> in `generate_dashboard.py`, rendered below the existing fit line. This is
> north-star **question 4, "how much / does it fit?", which had no surface at all**
> between the Model Portfolio's removal on 2026-08-26 and today.
> 
> **I checked the inventory first and it changed the scope, which is the point of
> the rule.** `plan/dashboard-inventory.md` records a concentration *line* already
> shipped 2026-09-15: names, sectors, largest sector share, top-25/100 counts, trap
> flags. Sector spread - one of the four things §8.4 of the research note asked for
> - was therefore already built. The block adds only what was genuinely missing:
> 
> 1. **The name count against the published counts** - 30-40 (Statman 1987), ~50
>    (Campbell et al. 2001), 63 for a 10% shortfall risk over 20 years (Domian et
>    al. 2007), with a computed "below all three / above N of the three", and the
>    *randomly-selected* condition stated every time.
> 2. **The equal-split slice** (100/N) against the published caps on a single
>    holding: UCITS 5% (10%/40%), RIC 25/5/50, S&P DJI's 24% Select Sector re-cap.
> 3. **The widest risk gap on the list**, in raw annualised volatility, with the
>    equal-dollar arithmetic spelled out.
> 
> Plus two sourced footnote paragraphs: why it emits no weight, and why the risk
> line uses a raw number rather than a percentile.
> 
> **Zero payload cost, verified rather than asserted:** regenerating left
> `dashboard_data.js` **byte-identical** (git reports it unmodified). The block is a
> view over `raw.volatility`, which `stock_detail` already carried, plus literature
> constants in the emitted script.
> 
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

