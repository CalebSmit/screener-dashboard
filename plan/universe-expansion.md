# Beyond the S&P 500: a phased plan to add NASDAQ and NYSE stocks

**Opened 2026-10-28, owner direction** (in chat, owner-run session): *"right now we only have the
S&P 500 stocks, and that's good. There's been some really good testing done on these stocks. But
over time, I would really like to get the NASDAQ and the New York Stock Exchange tickers in here as
well ... we would have to phase it in bit by bit."*

This is a long-horizon item. The point of this file is that each phase is small enough for one or
a few sessions, has a measured exit test, and **never changes what the S&P 500 ranking publishes
until a phase says so out loud.** `OWNER_FOCUS.md` carries the item; this file carries the queue.

## What stays fixed throughout

- **The S&P 500 ranking stays exactly what it is** - same universe, same ranks, same
  `signals.json` - until the owner decides otherwise. A broader universe is published *beside* it,
  never instead of it. The S&P 500 page is the one the testing, the research notes and the
  evidence series are about.
- **The bot keeps trading the S&P 500 ranking.** `signals.json`'s top 40 and the payload contract
  (`tests/test_bot_payload_contract.py`) are not touched by any phase here. Pointing the bot at a
  wider universe is a separate owner decision with its own bot ROADMAP item.
- **Evidence series are per universe.** The live IC history, snapshots, context log and the
  research engine's ledger are S&P 500 series. A second universe gets its own; mixing them would
  break the one-observation-per-run-date rule and every effective-observation count.
- **No paid data** (owner, 2026-10-15). Every source below is free.
- **Methodology is research-led** (CLAUDE.md rule 4). The weights were researched for large
  companies (`research/2026-10-12-value-and-momentum-among-large-companies.md`,
  `research/2026-10-20-size-weight.md`). Whether they carry to smaller companies is a research
  question for phase 0, not an assumption.

## Why this cannot be done in one step - measured 2026-10-28

| What | Today (S&P 500) | At ~1,500 stocks | At ~4,000-5,000 |
|---|---|---|---|
| `dashboard_data.js` | 6.2 MB (~90% is `stock_detail`) | ~18 MB | ~55-65 MB |
| Yahoo fetch (`fetch.tickers` + `fetch.retry`) | 795 s on 2026-10-28 | ~40 min | ~2 h+ |
| SEC Form 4 refresh | 98-400 s | ~3x | ~10x |
| Repo growth per run (measured 0.36 MB/version) | ~7.6 MB/month | ~3x | ~10x |

Plus constants that encode "about 500": `scripts/check_run_health.py` `MIN_SCORED = 490`,
`scripts/data-run.ps1` `$universe = 500`, `scripts/rehearse_publish.py` `MIN_UNIVERSE = 495`,
`universe_history.py` `MIN/MAX_PLAUSIBLE_MEMBERS = 495/515`, the input gates' top-25/top-40
continuity rules, `const N = D.table_data.length || 500` in the page, the methodology text
("about 500 stocks") and at least five test modules with a `>= 490`-style floor. Each is right for
the S&P 500 and must become a property of *the universe being run*, not a literal.

## The phases

### Phase 0 - Research (a Monday): what changes below the large caps

**WRITTEN 2026-11-02 (owner-run evening): `research/2026-11-02-universe-expansion-phase0.md`.
DECIDED 2026-11-03 (13:00 scheduled session, which did not write it) - all three pre-registrations
ADOPTED, two with a correction the review measured; note section 9.** Phase 0 is CLOSED.

### The three decisions, pre-registered. Do not re-fit these to a result.

They are written here, dated and committed, for the same reason the momentum panic rule's three
numbers are (`tests/test_momentum_panic_rule.py`): a floor chosen after seeing what it admits is
not a floor. `tests/test_universe_expansion_preregistration.py` fails if any of them changes
without this file and the note changing with it.

1. **Universe order: the S&P 1500 first** (phases 2-3), then the all-listed universe above the
   floor (phase 4). Measured: the free lists give 503 + 400 + 603 = **1,505-1,506 names**.
2. **Phase-4 floor: the NYSE 20th-percentile market cap** from Ken French's `ME_Breakpoints` file,
   the most recent June row, applied from July and held twelve months. **July 2026 - June 2027:
   $1,224.13M** (June 2026 row, n = 1,079 NYSE stocks) - verified independently from the file,
   exact. Plus Russell's exclusions (price >= $1, US exchange, no SPACs, LPs, trusts, closed-end
   funds or ETFs).
   **Adopted for one reason only: it is the line Fama & French (2008), Hou/Xue/Zhang (2020) and
   Jensen/Kelly/Pedersen (2023) all draw, so their findings apply above it.** The note's two other
   arguments for it are struck, measured over 2015-2026: the agreement with S&P's $1.2B addition
   floor is **this June's coincidence** (June 2025: $870M; June 2020: $484M), and the line is far
   from stable - **+112.1% in June 2021, -34.4% in 2022, +40.7% in June 2026**.
   **Consequence to carry into phase 4:** the universe churns at its bottom edge every July, by a
   different amount each year, so the input gates below must be re-derived **from a July replay**,
   never from an average month.
3. **Percentiles within `(Sector, size_band)`**, with the sub-ten-peer fallback changed from
   universe-wide to **band**-wide. The S&P 500 comes out byte-identical - measured, not argued:
   **0 of 23,092 percentile values differ**, against **20,151 differing (up to 68.93 points)** for
   a genuine two-band split, so the check has teeth.
   **The fallback's cost, measured:** 1 of 301 weight-bearing cells falls back today
   (`beneish_m_score` in Financials); **27 of 903 guaranteed after the split, 35 with availability
   applied** - and every guaranteed one is in **S&P 400 Communication Services (6 members)**, the
   only sector-band group under ten members anywhere in the S&P 1500. **The thin group is in the
   middle band**, so phasing the 600 in last does not defer it.

**Resolved for phase 1:** the note's section 7 says both "keep 2e9 for the S&P 500" *and* "set 0
for S&P index bands", which contradict each other for the one band whose byte-identity is the exit
test. Measured: **0 of 503 members sit below $2B; the smallest is MOS at $6.050B**, p1 $8.796B. So
**one rule - `min_market_cap: 0` for every S&P index band, the 500 included** - is byte-identical
today and does not need a special case. Reopen only if a member comes within 2x of the floor; the
guard against an absurd market cap belongs in `check_run_health.py`, not in a filter that silently
drops index members.

**Left open on purpose, not deferred:** section 4's choice between coverage rules (a) MSCI-style
renormalising, (b) Russell's shrinkage schedule and (c) a minimum share of category weight. It is
a **present-day** methodology change - today it would move one S&P 500 stock (Loews, whose whole
Revisions category rests on days-to-cover) and six more partly - so it gets its own changelog
entry decided on its own merits, not as a side effect of phase 3.

One complete note in `research/`, literature and practice, ending in a `**Proposal status: OPEN**`
line. Questions it must answer:

1. **Which universe first.** The S&P 1500 (adds the MidCap 400 and SmallCap 600) keeps the same
   index committee rules, including the positive-earnings entry screen, so the screener's
   assumptions carry furthest; an "all NYSE + NASDAQ common stocks above a floor" universe is what
   the owner ultimately asked for. Expectation to test, not assume: S&P 1500 first.
2. **Microcaps.** The anomaly literature finds many published effects concentrate in the smallest
   stocks and weaken or vanish once those are controlled for (to verify and quote with effect
   sizes: Fama & French 2008, "Dissecting Anomalies"; Hou, Xue & Zhang 2020, "Replicating
   Anomalies"). How practitioners set the floor - market-cap and dollar-volume minimums, NYSE
   breakpoints - and what number to pre-register.
3. **Percentiles.** Today a metric is ranked within its sector across ~500 large companies. With
   small companies in the pool, is a stock ranked against its sector, its sector *and* size band,
   or something else? What MSCI, FTSE Russell and S&P do.
4. **Data coverage.** Analyst targets and estimate revisions thin out in small caps; how
   practitioners score a stock with no analyst coverage, and what that does to the Revisions
   category (currently weighted) and the coverage filter.
5. **Which existing research notes stop applying** below large caps, by name.

### Phase 1 - Make the universe a parameter (no published change)

Engineering only; the S&P 500 page must come out byte-identical (`git diff` on both payloads
empty after a rebuild from the same run).

**This is the next stage. It is engineering only; no decision above is re-opened here.**

**Progress, 2026-11-08 (06:00 weekend session): the `stock_detail` shard bullet - the mechanism
and its first slice, measured at -19.8% of first paint.**
`stock_detail`'s sheet-only fields now ship in a **third payload file**, `dashboard_detail.js`
(`generate_dashboard.split_detail`, `DETAIL_KEYS = ("about", "industry")`), fetched when the
first sheet is laid out instead of at first paint.

**Why those two fields and not the obvious ones.** Measured before anything was built
(`research/measurements/2026-11-08-stock-detail-field-weights.py`, committed, which prices every
field by removing it from the whole payload and re-gzipping): `stock_detail` is **84.3% of the
1.547 MB gzipped first-paint payload**, and the heaviest single field in the whole payload is
**`about` at 19.3%** - ahead of `inp` (13.8%), `raw` (12.5%), `pct` (8.4%) and `summary` (7.9%).
The *shape* of the shard, though, is constrained by how the page reads the data: **~85 call sites
read `D.stock_detail[t]` synchronously for an arbitrary ticker** (search, peers, holdings,
movers, compare, the hash router), so a per-stock on-demand fetch would have to make every one of
them asynchronous. Enumerating the read sites by enclosing function instead gives the rule this
bullet now follows: **one read site, inside the sheet, and no number.** `about` and `industry`
are read only by `renderAbout`; `financials` is also read by `buildPeerRow`, `eq4` by
`renderReporting` (a first-paint strip), and `raw`/`pct`/`inp`/`summary` by the compare drawer,
the movers strip and the calculation traces. Those are the next slices and each needs its own
measurement - the mechanism is now there for them.

**Result: first paint 1.547 -> 1.240 MB gzipped, -307 KB, -19.8%**, with 0.222 MB served on
demand and `dashboard_context.js` byte-identical. No number, score, rank or sentence moved: the
prose is display-only by the 2026-08-26 decision and nothing scores it.

**Guarded in the commit that created it**, which is the 2026-10-16 lesson (the context payload
was published unguarded for eight days because adding a file meant remembering two PowerShell
scripts): both runners' size/header/`node --check` specs, `$DataArtifacts`,
`check_published_claims.py`'s `PUBLISHED_ARTIFACTS`, `rehearse_publish.PUBLISHED`,
`rebuild_page.PAYLOADS` (which now checks the page asks for the committed file's own hash) and
`run_screener`'s publish copy. One new tripwire generalises the next time:
`tests/test_rehearse_publish.py` derives the rehearsal's install list from `$DataArtifacts`, so a
payload file the rehearsal does not install - and would therefore rehearse *as committed* - fails
the suite. `tests/test_detail_payload_split.py`, 21 tests.

**Measured rather than assumed, and it corrects the obvious claim:** "a reader who never opens a
stock never pays for it" is **false**. The idle pre-warm lays the top stock's sheet out on an
untouched page, so it pulls the file - after both payloads. The measured request order is
`dashboard_data.js`, `dashboard_context.js`, `dashboard_detail.js`, which is what the tests
assert (an order, true on any connection, rather than a timing). First paint is what the split
protects; the pre-warm is there so the first open is instant, and that now includes the text.
**Six mutations a reader could plausibly ship, 7 of 7 caught**
(`research/measurements/2026-11-08-detail-split-mutations.py`): the split copying instead of
moving, every stock carried including empty ones, the file loaded from the start-up block, the
once-only guard lost, the waiting state back to an empty block, the loader not merging into
`stock_detail`, and the code runner not naming the file.

**What phase 4 still needs from this bullet, in order:** `financials` (4.7% of first paint),
then `inp`, `raw` and `pct` behind the calculation traces, and only then **bucketing**: one file
for the whole universe is right at 500 stocks and wrong at 5,000, so the parts become ~32-stock
buckets keyed by a function of the ticker, which is a localized change once every read site goes
through the loader. Until then the shard is "off first paint", not yet "flat in the universe".

**`self_metrics` is gone rather than sharded (2026-11-08, 13:00 session), and that corrects the
step this bullet named.** It was listed above as the other half of the next slice. It did not
need to move, because it held nothing of its own: the peer table's nine columns were built
**two ways** - the open stock's row from `self_metrics`, every peer's row from `buildPeerRow` in
JS over `raw` + `financials` - and every stock is on both sides of that split, being the subject
of its own sheet and a peer in its sector-mates'. Measured first
(`research/measurements/2026-11-08-peer-row-two-paths.py`): **0 of 4,491 cells disagreed** on the
2026-11-06 payload, so nothing a reader sees moved, and the two expressions for P/E were
genuinely different code (`round(1/ey, 1)` on an unrounded yield in Python against
`Math.round(10/ey)/10` on the published 4dp one in JS). So the subject's row goes through
`buildPeerRow` too and the block is not published: **first paint 1.240 -> 1.223 MB gzipped,
-17 KB**, with no fetch, no waiting state and nothing new on demand - the cheapest slice in this
bullet, and the only one that makes the page simpler rather than more conditional.
`tests/test_peer_row_one_path.py`; the check that would have caught the duplication reads one
company's row from its own sheet and from another stock's and requires the cells to be identical.

**Progress, 2026-11-07 (13:00 session): the per-band gates bullet - the health gate shipped, the
auto-disable's trigger deliberately not moved.**
`check_run_health.MIN_TARGET_COVERAGE` (0.50) is now applied **inside each size band** as well as
to the universe (`check_target_coverage_by_band`), with the band read from the run's own
`01_raw_metrics.parquet` so the payload stays byte-identical and a run that carries no band column
produces a **note**, never a failure. It is strictly stronger than the universe-wide check it sits
beside and never weaker - the universe rate is a weighted mean of the band rates, so a
universe-wide failure implies some band below the same floor - and measured on the published run
(`research/measurements/2026-11-07-target-coverage-by-band.py`): the **thinnest market-cap decile
of today's S&P 500 is 98.0% covered, 1.96x the floor**, so the gate cannot fire on this universe
however it is banded; and with the 40 smallest stocks stripped of their targets and made their own
band the universe-wide check passes at **91.6%** while the per-band check fails, so the band path
is not decoration. `tests/test_per_band_gates.py` (64 tests).

**`MIN_PRICE_COVERAGE` and `MIN_CATEGORY_COVERAGE` stay universe-wide, and the reason is not
caution.** Analyst coverage is the one input here whose *availability* is a function of company
size (Bhushan 1989; Hong, Lim & Stein 2000). A price and a price history are served for every
listed stock, so both of those guard a **feed failure**, which appears across the whole universe
or not at all; banding them would only add ways to fail.

**`CATEGORY_AUTO_DISABLE_PCT` (30) is measured per band and still triggered universe-wide, on
purpose.** For the health gate "per band" is engineering - the action is to discard the run. For
the auto-disable the action is **to change the weights**, and there are three incompatible
per-band rules: **(a)** keep the universe mean deciding, one weighting for everyone; **(b)** any
band below the floor disables the category for the whole run - throwing a category away for 500
well-covered large companies because 600 small ones lack it; **(c)** disable per band - which
gives the bands different weight vectors, so a composite means something different in each, on top
of percentiles that are already within band. Which is right depends on how far Revisions and
Investment coverage actually falls, which is **phase 2's** measurement and does not exist yet. So
`run_screener.weighted_coverage_pct_by_band` records the per-band figures now (`stats`,
printed from the band a second one exists) and **choosing between (a), (b) and (c) is a
methodology change with its own `METHODOLOGY_CHANGELOG.md` entry**, never a side effect of
phase 2 or 3. `tests/test_per_band_gates.py` fails if the trigger becomes per-band without it.

**Measured against the mutation standard, because a test module written the same day as its
feature is the easiest place for a check to read as coverage and not be one
(`plan/claims-mutation-audit.md`, nine passes):** six mutations a reader could plausibly ship -
the per-band check as a no-op, the floor becoming exclusive, every run taking the one-band door,
blank bands kept as bands, `df` for `g` in the per-band figure (one character), and the
auto-disable gaining a `by_band=` argument - **6 of 6 caught**
(`research/measurements/2026-11-07-per-band-gate-mutations.py`). And the number that says what
the module is worth: with it deselected, the whole suite reports **4,130 passed, 1 skipped**
against the feature being a complete no-op. Nothing else in the project can see it.
**Verified past the suite:** the published run copied to a temp directory with two real bands
written into its own `01_raw_metrics.parquet` (bottom 40 by market cap against the other 462),
through `check_run_health.py --run-dir`: both bands judged, and **457/459 + 40/40 = 497/499**
reconciles with the universe-wide line, so no stock is lost between the two checks.

**A phase-3 obligation this creates**, beside the three claims the 06:00 session recorded:
`SCREENER_OVERVIEW.md`'s fetch-reliability paragraph says the gate "refuses a run with ... 
analyst-target coverage below 50%". True today, and exactly true, because the band is the
universe. The session that publishes a second band says *per band* there, in the same commit.

**Not done, and why:** `universe_history.MIN/MAX_PLAUSIBLE_MEMBERS` per index. The bound is
already a property of a `Universe` (`CURRENT.min_members`/`max_members`, 2026-11-03); what is
missing is an S&P 400 and an S&P 600 `Universe` to carry one, and those arrive with phase 2's
constituent-list source. Making `validate_membership` take a `universe=` argument before there is
a second universe to pass would also edit `universe_history.py`, which is in
`v2_experiments.RESULT_PATHS` - a 32-row, ~25-minute research-engine rerun for a parameter with
one legal value. Do it in the commit that defines those two universes, and pay the rerun once.

**Progress, 2026-11-07 (06:00 weekend session): the `size_band` bullet, shipped.**
`universe.Universe` carries a `size_band` and `universe.assign_size_bands()` sets the column;
`run_screener` calls it once the universe is settled, so every artifact from `01_raw_metrics`
onward carries it. `factor_engine.compute_sector_percentiles` groups by `(Sector, size_band)`
with the **band**-wide fallback below `SECTOR_MIN_PEERS` (unchanged at 10), and `Composite_Pct`
is ranked within band. One universe, so one band, so nothing published moved - **measured, not
argued**: `tests/test_size_bands.py` (13 tests) compares the production function with a
transcription of the sector-only code it replaced, on the published run - **0 of 23,092
percentile values differ** - and a half-and-half cap split moves **20,140** of them, so the
identity cannot pass vacuously. (Phase 0's script reported 20,151 for the same split on the
2026-11-03 run; the figure is a property of that morning's data, not a constant.) The composite
percentile gets the same pair: unchanged with one band, **492 of 499** moved by the split.
Verified beyond the suite:
a five-ticker live run through `run_screener.py` (the band reaches all three artifacts, calc
trace reproduces, overview byte-identical) and `scripts/rehearse_publish.py` (all four published
files unchanged byte for byte).
`universe.py` joined `v2_experiments.RESULT_PATHS` the same session - it now decides what each
stock is ranked against, so an edit there moves every engine number - and the 32 ledger rows were
rerun: **no statistic moved**, only the run count and the code hash.
**Still not started:** the `config.yaml` block, the per-band gates, and the `stock_detail` shard.
**A phase-3 obligation this created:** three registered claims still say a thin sector falls back
"across the universe" (`drilldown.peer_rank`, `overview.percentile_rule`, `overview.glossary`).
They are true today because the band *is* the universe; the session that publishes a second band
must reword all three in the same commit.

**Progress, 2026-11-03 night (owner-run): the first slice of the first bullet.** `universe.py` defines
the S&P 500 (`members 495-515`, `ranked >= 490`) and `check_run_health.MIN_SCORED`,
`rehearse_publish.MIN_UNIVERSE`, `universe_history.MIN/MAX_PLAUSIBLE_MEMBERS` and two tests' floors
read it; `check_run_health`'s output on the latest run is byte-identical before and after, and the
research engine's 32 rows were rerun because `universe_history.py` is hashed. **Left literal on
purpose** and named in `tests/test_universe_definition.py`: `data-run.ps1`'s `$universe = 500` (the
data loop's own script, not edited hours before a run) and the page's `|| 500` fallback. Not started:
the `config.yaml` block (the definition lives in Python until a second universe needs a switch),
`size_band`, the per-band gates and the `stock_detail` shard.

- A `universe:` block in `config.yaml` naming the source and the plausible size band; every
  constant listed above derived from it. Tests that hard-code 490 read the same value.
- **`min_market_cap: 0` for every S&P index band** (decision 3's resolution above), so index rules
  govern membership. One rule, no special case for the 500.
- **A `size_band` column** set at fetch time: the S&P parent index (500 / 400 / 600) in phases
  2-3; NYSE breakpoints in phase 4 (above the 80th percentile, 50th-80th, 20th-50th - JKP's mega /
  large / small). `compute_sector_percentiles` groups by `(Sector, size_band)`; the sub-ten-peer
  fallback becomes band-wide; `Composite_Pct` is computed within band. `SECTOR_MIN_PEERS` stays
  at 10 - only its scope changes.
- **Per-band, not universe-wide:** `check_run_health.MIN_TARGET_COVERAGE` (0.50) and
  `run_screener.CATEGORY_AUTO_DISABLE_PCT` (30), because a universe-wide average hides a thin
  bottom band. `universe_history.MIN/MAX_PLAUSIBLE_MEMBERS` become per index (~400 and ~600 with
  the same tolerance). `MIN_CATEGORY_COVERAGE` (0.90) stays universe-wide: it guards a feed
  failure, not a band. **(The health gate shipped 2026-11-07; the auto-disable's *trigger* is a
  methodology decision for phase 2, with its three options written down in the progress note
  above - the per-band figures are measured and recorded now. The membership bounds wait for
  phase 2's S&P 400/600 `Universe` objects, so one research-engine rerun pays for both.)**
- **The input gates are re-derived per universe from that universe's own replay, never copied** -
  `RANK_MIN_SPEARMAN` (0.90), `RANK_MAX_NEW_IN_TOP` (8) and the top-25/40 rules were set by
  replaying S&P 500 pages (`research/measurements/2026-10-22-input-gates-replay.py`). For phase 4
  the replay must include a **July**, where the annual floor reset churns the bottom edge.
- `stock_detail` sharded out of `dashboard_data.js` and loaded per stock (the Factor Lab's
  `LAZY_KEYS` pattern), so first paint does not grow with the universe. Keep the bot's fields
  (`stock_detail[t].price`) where the contract says they are, or move them with the bot's tests.
- Fetch cost per ticker measured from `PHASE TIMING`, with the projection written down.

**Exit test:** the four gates, `rehearse_publish.py`, and an unchanged S&P 500 payload.

### Phase 2 - A shadow run of the S&P 400 + 600 (scored, not published)

- Universe source: the free S&P 400/600 constituent lists (same pattern as
  `factor_engine.get_sp500_tickers`, with a committed fallback file).
- Run at a time that cannot collide with the 02:00 loop or the repo lock (or a weekend), into its
  own run directory; nothing staged, nothing pushed.
- Measure and write down: fetch time and failures, per-metric coverage (expect analyst inputs to
  fall), `sec_crosscheck` agreement, how `check_run_health.py` judges it, and how far the
  mid/small-cap ranks move day to day (the Spearman continuity the bot gates use).
- **Added by the phase-0 decision**, per band: the analyst-count distribution; the share of stocks
  whose Revisions score would rest on `short_interest_ratio` alone; how often
  `forward_eps_growth` takes the `fy2_over_trailing` fallback; how many S&P 600 members sit below
  the NYSE 20th percentile (members are not removed when they shrink); the share below $700M float
  (the reporting-lag note's premise); per-band `sec_crosscheck` counts.
- **The three marginal percentile cells**, which are sector-ranked today only because availability
  is high: `accruals` in S&P 400 Utilities (15 members, needs 67% availability), S&P 400 Consumer
  Staples (14, needs 71%) and S&P 600 Utilities (14, needs 71%). Report whether they held.

**Exit test:** two weeks of shadow runs with the numbers in a research note, and the health gate
either passing them or a written reason it should judge them differently.

### Phase 3 - Publish the S&P 1500 beside the S&P 500

- A universe selector on the page ("S&P 500" default, "S&P 1500"), each with its own ranks and
  its own claims; the methodology page says plainly which evidence was gathered on which universe.
- Its own evidence series from its first published day.
- Methodology changes, if phase 0 called for any (a floor, size-band percentiles), each with a
  changelog entry, one at a time.

**Exit test:** the four gates, the data loop's claims gate covering both universes, and the S&P
500 payload and `signals.json` unchanged.

### Phase 4 - All NYSE and NASDAQ common stocks above the floor

- Universe source: the NASDAQ Trader symbol directory (`nasdaqlisted.txt`, `otherlisted.txt`,
  free, with ETF and test-issue flags) or SEC's `company_tickers_exchange.json`; exclude ETFs,
  funds, preferreds, warrants, units, SPACs and ADRs unless phase 0 argued otherwise; apply the
  pre-registered floor (decision 2 above: **$1,224.13M** for July 2026 - June 2027, re-read from
  the June file each year and held twelve months).
- **Before the first July rebalance**, re-derive the input gates from a replay that spans a July.
  The floor moved +112.1%, -34.4% and +40.7% in three of the last six years, so the annual reset
  churns the bottom of the universe and an average-month continuity threshold will read it as a
  broken run.
- The fetch will need its own time budget and probably its own scheduled task; the throttling
  record (`research/2026-10-26-data-run-timing.md`) says a second full fetch within hours is what
  got throttled, so schedule it far from 02:00.

**Exit test:** set when phase 3 has run for some weeks; not before.

## Not in scope here

- Changing the bot's universe.
- Any weight change justified by a backtest (CLAUDE.md rule 5 binds until 2027-02-11), and no
  point-in-time history is available yet for the S&P 400/600, so the research engine stays on the
  S&P 500.
