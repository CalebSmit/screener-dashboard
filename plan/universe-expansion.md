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

- A `universe:` block in `config.yaml` naming the source and the plausible size band; every
  constant listed above derived from it. Tests that hard-code 490 read the same value.
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
  pre-registered floor.
- The fetch will need its own time budget and probably its own scheduled task; the throttling
  record (`research/2026-10-26-data-run-timing.md`) says a second full fetch within hours is what
  got throttled, so schedule it far from 02:00.

**Exit test:** set when phase 3 has run for some weeks; not before.

## Not in scope here

- Changing the bot's universe.
- Any weight change justified by a backtest (CLAUDE.md rule 5 binds until 2027-02-11), and no
  point-in-time history is available yet for the S&P 400/600, so the research engine stays on the
  S&P 500.
