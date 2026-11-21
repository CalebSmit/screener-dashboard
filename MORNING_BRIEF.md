# Morning Brief - Saturday 21 November 2026, 06:51

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-20 18:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | EXPE, VLO, MU, APA, SNDK |
| Ranking's lean on growth | +0.054 (OUTSIDE the review's +-0.05 band; 2026-10-13 read -0.009) |
| Shipped to readers | 14 commit(s) changed what a reader sees, 8 methodology change(s), in the last 7 days |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (55 rows, but overlapping windows are not independent; 135 rows across all horizons), newest 1m run date 2026-10-20 (32 days ago, bound 40) |
| Context signals | up to 1 effective observations of 8 needed. Not enough for a t-statistic yet. |
| Inputs vs SEC filings | total assets 458/458 within 0.5%; operating cash flow 443/444 within 0.5%; net income 432/456 within 0.5%; market value 452/487 within 5%; 2 scored a quarter behind a filing over 30 days old; 1 with no filing in the cache for 165 days; filings cached 2 days ago - **see the ATTENTION note below** |
| Top 25, worth a look | EXPE (#1: equity 6.2% of assets); HST (#7: net income 1.29x after-tax operating income); PSX (#8: net income 1.28x after-tax operating income); TPR (#9: equity 5.2% of assets); DXCM (#11: operating cash flow 36% off the filing); ES (#18: a period behind its filings); CF (#23: net income 15% off the filing); WDC (#25: net income 2.13x after-tax operating income) |
| Data run timing | **21 min** total; slowest `fetch.tickers` 644 s, `context.options_pass` 182 s, `fetch.retry` 149 s; **46 tickers rate-limited** |
| Code-session login | 2026-12-03 09:01 (12 days) |
| Context coverage | options usable 494/499 - insider from SEC 499/499 - earnings variability 440 |

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-23 (in 2 days) - Build residual momentum in place of `jensens_alpha` (ACCEPTED 2026-11-21, weight 25 unchanged) in a 06:00 session (0.15's section 4 is committed - done 2026-11-21): section 9's build list - 37-month history on the same request with the Risk inputs trimmed to 13, `^SP500TR` to 4y, lineage/equation/claims, backtest v2, the engine rerun, a throwaway full run before the changelog entry. `research/2026-11-20-jensens-alpha-in-momentum.md`
- 2026-11-24 (in 3 days) - Build the implied-volatility inversion ACCEPTED on 2026-11-04 and dropped from every Next list since 11-05 (found 2026-11-19). Section 1d's build order: exDividendDate into the core .info fetch, a throwaway full run, QuantLib pinned and installed by a scheduled session (never from the desktop app: its AppData is sandboxed). A 06:00 session with nothing else in it. `research/2026-11-03-open-source-quant-stack.md`
- 2026-12-04 (in 13 days) - The option signals' (expected move, ATM volatility, skew) first one-month reading: readings before 2026-11-04 recorded no price and are excluded, not spliced. `METHODOLOGY_CHANGELOG.md`

## Things that needed attention

- ok   ATTENTION - 1 stock(s) now carry a market value implying a share count over 10% from the one their filings report, which the published page does not: SNDK +639.2%. Usually a completed acquisition or a split: the market value is right and every valuation ratio built on it has moved, while the earnings behind those ratios have not. Not a refusal - see research/2026-11-09-share-counts-and-corporate-actions.md.

## What changed in the repo

- `2e9da36 NIGHTLY_LOG 2026-11-21: gates recorded`
- `01063c3 NIGHTLY_LOG 2026-11-21 (weekend): Jensen proposal decided, 0.15 answered`
- `5758fa7 inventory: the hero's second tested sentence and the research section`
- `fd9541f 0.15's results on the page: the hero's tested line and a research-page section`
- `4c9583e 0.15 answered: D, a teaching tool with no return claim`
- `129ee13 0.15: fix the measurement's method and script before the first run`
- `86db426 Decide the Jensen's-alpha proposal: ACCEPTED (residual momentum), build after 0.15's measurement`
- `9250a6a brief: data run 2026-11-20`
- `81f1d48 data: screener run 2026-11-20 - 499 scored, top: EXPE VLO MU APA SNDK`
- `072a500 brief: code session 2026-11-20-midday`
- `a467255 NIGHTLY_LOG 2026-11-20 (13:00): gates recorded`
- `4a1d52f Research: Jensen's alpha in Momentum is the 12-month return with a beta tilt; residual momentum proposed (OPEN)`
- `3a5d0a1 wip: research note - Jensen's alpha in Momentum (practice section pending)`
- `5377901 wip: measure how much of Jensen's alpha is the 12-month return Momentum already scores`
- `4d0f7ba brief: code session 2026-11-20`

## The session's own account

> 2026-11-21 - WEEKEND. The Jensen's-alpha proposal decided (ACCEPTED: residual momentum), and 0.15 answered: D, a teaching tool with no return claim - now on both pages
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-20_130000-midday.log`
> (completed). Data loop published? yes - `logs/datarun-2026-11-20_180003.log` (completed; HEALTH: PASS) -
> the 18:00 retry, the first run under the 11-19 missing-category rule to reach the site, so the previous
> entry's Next 1 is confirmed. Evidence base at `1m`: **55 rows, newest 2026-10-20 (32 days ago, bound 40),
> 5 effective of 8**. Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.15, What the
> ranking is for** (opened 2026-11-19, **age 2 days**) - answered this session, two page changes remain open
> under it, one shipped here; also open 0.13 (40 d, a standing programme), 0.10 (45 d), 0.14 (38 d).
> Shipped to readers: 13 commits, 8 methodology changes in 7 days.
> **Tests:** baseline 4,447 passed / 1 skipped (dry run OK). After: **4,455 passed / 1 skipped** (+8, the
> new `tests/test_screen_and_horizon.py`); dry run PASSED; payload parse gate 20/20, index.html 949,749
> bytes; rehearsal (gate 5) PASSED (published files restored byte for byte); tree clean.
> **Owner queue / rotation:** `git stash list` empty; no owner-session marker; no earlier entry today. Two
> open proposals in the brief: the Jensen's-alpha one (a session that did not write it - me) and 0.15's
> Monday measurement. The rules make deciding a proposal the first job, so that came first; it then had
> nothing left blocking it but 0.15, and 0.15 is both the top roadmap item and an owner-agreed priority, so
> I did it two days before its Monday rather than leave the weekend on something smaller.
> 
> ### Did
> - **Decided `research/2026-11-20-jensens-alpha-in-momentum.md`: ACCEPTED, option C (CAPM residual
>   momentum, weight 25 unchanged), not built today.** Section 9 of the note. Done against its own
>   checklist rather than read:
>   - **Re-read the sources from the manuscript** (pypdf on `data/papers/blitz2011.pdf`): Table 2, Table 8
>     (largest 10%: Sharpe 0.36 -> 0.60, t 2.75 -> 5.32), Table 7 (no skip) and the p. 12 alpha exclusion
>     all reproduce; the sub-agent's unchecked Table 3 reading also reproduces (2000-2009: residual +4.65%,
>     t 1.18; total-return -8.54%).
>   - **Measured the fetch cost** (`research/measurements/2026-11-21-residual-momentum-fetch-cost.py` +
>     `.json`): `period="37mo"` is the same single request as today's `13mo`; median 0.126 s -> 0.163 s on
>     30 tickers (order alternated), 274 -> 774 bars. **Coverage 495 of 503** - short: FDXF, GEV, HONA, Q,
>     RDDT, SNDK, SOLV, VYLR (SNDK is in today's top five; they fall back to the existing within-category
>     share). `fetch_market_closes` pulls `^SP500TR` at `3y`, three weeks short of the needed close -
>     lengthen that one request. Neither refusal condition is met.
>   - **What the second reader added:** Blitz et al. is long-short, and this project already learned
>     (stop-loss note, 2026-10-22) that momentum's crash risk sits mostly in the losers leg a long-only
>     book never holds - so the changelog must not quote 0.36 -> 0.60 as what a reader should expect, and
>     the decision does not rest on it. It rests on: no source uses the current construction; C is a
>     construction change with no weight moved (D would move 3.25 composite points onto two 0.9-alike
>     metrics); it stops rewarding low beta a second time on top of Risk.
>   - **Not built today** because it collided with 0.15 (changing the composite two days after its
>     criteria were pre-registered, before they were measured) and is a full 06:00 session's build
>     (fetch trim, market series, lineage/equation/claims, backtest v2, engine rerun, throwaway full run).
>     `COMING_UP` carries it for **2026-11-23**, the slot 0.15 vacated.
> - **0.15 answered - `research/2026-11-19-what-the-ranking-is-for.md` sections 4-5.** The criteria were
>   fixed on 11-19; the operational choices they left open (eras, decile cut, which SE, window
> ...

---

If a run says **refused by its publish gates** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

