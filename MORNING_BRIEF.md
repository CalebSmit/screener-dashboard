# Morning Brief - Thursday 19 November 2026, 06:58

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-19 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | SNDK, EXPE, HST, APA, MPC |
| Ranking's lean on growth | +0.042 (within the review's +-0.05 band; 2026-10-13 read -0.009) |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (54 rows, but overlapping windows are not independent; 133 rows across all horizons), newest 1m run date 2026-10-19 (31 days ago, bound 40) |
| Context signals | up to 1 effective observations of 8 needed. Not enough for a t-statistic yet. |
| Inputs vs SEC filings | total assets 459/459 within 0.5%; operating cash flow 444/445 within 0.5%; net income 433/457 within 0.5%; market value 453/487 within 5%; 2 scored a quarter behind a filing over 30 days old; 1 with no filing in the cache for 165 days; filings cached 1 days ago |
| Top 25, worth a look | EXPE (#2: equity 6.2% of assets); HST (#3: net income 1.29x after-tax operating income); PSX (#8: net income 1.28x after-tax operating income); DXCM (#9: operating cash flow 36% off the filing); TPR (#12: equity 5.2% of assets); CF (#18: net income 15% off the filing) |
| Data run timing | **19 min** total; slowest `fetch.tickers` 645 s, `fetch.retry` 162 s, `context.sec_insider` 101 s; **52 tickers rate-limited** |
| Code-session login | 2026-12-03 09:01 (14 days) |
| Context coverage | options usable 494/499 (5 quotes-closed) - insider from SEC 499/499 - earnings variability 440 |

## What changed in the repo

- `69ef668 NIGHTLY_LOG 2026-11-19: the universe block, the floor that never fired, the gates' results, what is next`
- `07c7572 Research engine: all 32 rows rerun on the new config - 27,225 statistics, 0 moved; only the config hashes differ`
- `a27370b Changelog, plans, CLAUDE.md: the market-cap floor is 0 and never fired; wrong market caps in the engine's history recorded`
- `6911cd0 wip: the universe block is read - an undefined index is refused, and a market-cap floor that is set is applied`
- `f949d3a brief: data run 2026-11-19`
- `3017bd6 data: screener run 2026-11-19 - 499 scored, top: SNDK EXPE HST APA MPC`
- `e365cfc brief: code session 2026-11-18-midday`
- `4e34d62 NIGHTLY_LOG 2026-11-18 (13:00): where the missing filings went, the gates' results, what is next`
- `04e3f32 Research note, changelog, CLAUDE.md 0.14: where twenty companies' filings went`
- `6b9d514 wip: the SEC refresh follows a combined filing to the co-registrant whose record holds it`
- `9f638b9 brief: code session 2026-11-18`
- `0a4e4c0 NIGHTLY_LOG 2026-11-18: the synthesis, the gates' results, what is next`
- `1aecd56 The synthesis note commits to the date COMING_UP gives it`
- `e0f2c28 Synthesis note, changelog, CLAUDE.md 0.14: what the screener knows about its inputs`
- `cfeb7bc wip: the health line counts the stocks the filing check cannot see`

## The session's own account

> 2026-11-19 - BUILD. The universe block in `config.yaml` is read now - and the $2B floor it held had never excluded a stock
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-18_130000-midday.log` (completed). Data loop published? yes - `logs/datarun-2026-11-19_020001.log` (completed; HEALTH: PASS). Evidence base at `1m`: 5 of 8 needed at the 1m horizon (54 rows, but overlapping windows are not independent; 133 rows across all horizons), newest 1m run date 2026-10-19 (31 days ago, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab** (age **38 days**; a standing programme, last advanced **13 days** ago per `plan/factor-lab.md` (bound 21)); also open: 0.10 43d; 0.14 36d.
> 
> **Tests:** before **4,374 passed / 1 skipped / 0 failed** (4:03, my own baseline; yesterday's three
> cache-dependent skips are running again, as predicted). After **4,393 passed / 1 skipped / 0
> failed** (4:38) - 19 new cases, all in `tests/test_universe_config_block.py`. Gate 2 (dry-run)
> passes. Gate 3: no payload or page file touched; the rehearsal's own parse of all three payload
> files passed (a standalone `node --check` was refused by the tool guard this session, so I did not
> run it separately). Gate 5, `scripts/rehearse_publish.py --changed-since origin/main`:
> **REHEARSAL PASSED**, every published file unchanged byte for byte. Gate 4 clean.
> 
> **Owner queue / rotation:** first session today. `git stash list` empty; no branch held back;
> `open_proposals()` is `[]`. `OWNER_FOCUS.md` read: the top open item (2026-11-17, three decisions)
> is the owner's alone; the next is **universe expansion (2026-10-28)**, whose plan names the
> `config.yaml` `universe:` block as next in phase 1 - that is what I took, and it is also what a
> BUILD day is for (the week's research - Tuesday's dating fix, Wednesday's synthesis - shipped its
> own builds and left dated reviews, not a pending change). Its Progress line is updated; the item
> stays open (phases 1-4 continue). 0.13 is not deferred (13 days against 21; tomorrow is its day).
> Nothing deferred for a gate or a loop.
> 
> **Yesterday's Next 2, checked first:** today's health line reads total assets **459/459**, "**1**
> with no filing in the cache for 165 days" (was 20), and the three note tests run again (suite
> back to 1 skipped). "Its filings are flagged stale" appears in **0** of the four published files.
> **Not done:** adding that sentence to `claims.FORBIDDEN` - `claims.py` builds the page, and I kept
> today's diff to one subject. It is Next 3.
> 
> ### Did
> 
> **1. The finding.** The plan's resolution said `min_market_cap: 2e9 -> 0` is byte-identical
> *today* because no member is under $2B. Checking whether that also held for the research
> engine's history (S&P members were under $2B in 2012) I found it held for a different reason:
> **the floor had never fired, anywhere.** `factor_engine.apply_universe_filters` acted only "if
> `marketCap` in df.columns", and both callers (`run_screener.py`, `backtest_v2.rank_month`) hand
> it `compute_metrics`'s frame, which carries `size_log_mcap` and no `marketCap`. Measured
> (`research/measurements/2026-11-19-market-cap-floor-never-fires.py`): the shipped filter returns
> **502 of 502 rows at floors of $2 billion, $1 trillion and $1 quadrillion**. `config.yaml` said
> `# $2B minimum` since February; the Blueprint says it "eliminates micro-cap illiquidity".
> `tests/test_edge_cases.py::TestUniverseFilters` passed throughout - it builds its frame by hand
> with the column production does not have. Same shape as the 2026-11-09 `financials` shard miss
> (settled row "gates"): the test and production disagreed about the input, and the test won.
> Likewise `universe.index` was read by no code; its comment offered `R1000` and `R2000`, and
> either silently ran the S&P 500.
> 
> **2. The build** (phase 1's `config.yaml` bullet):
> ...

---

If a run says **stopped deliberately** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

