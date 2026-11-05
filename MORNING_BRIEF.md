# Morning Brief - Thursday 05 November 2026, 02:21

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-05 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | VLO, MPC, EXPE, MU, SNDK |
| Ranking's lean on growth | +0.060 (OUTSIDE the review's +-0.05 band; 2026-10-13 read -0.009) |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (44 rows, but overlapping windows are not independent; 112 rows across all horizons), newest 1m run date 2026-10-05 (31 days ago, bound 40) |
| Context signals | 19 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 436/437 within 0.5%; operating cash flow 403/420 within 0.5%; net income 397/433 within 0.5%; market value 446/485 within 5%; 4 scored a quarter behind a filing over 30 days old; filings cached 3 days ago |
| Top 25, worth a look | EXPE (#3: equity 4.2% of assets); HST (#6: net income 1.46x after-tax operating income); DXCM (#7: operating cash flow 36% off the filing); PSX (#8: net income 1.28x after-tax operating income); CF (#13: net income 16% off the filing) |
| Data run timing | **19 min** total; slowest `fetch.tickers` 660 s, `fetch.retry` 121 s, `context.sec_insider` 107 s; **38 tickers rate-limited** |
| Code-session login | 2026-12-03 09:01 (28 days) |
| Context coverage | options usable 476/499 (23 quotes-closed) - insider from SEC 499/499 - earnings variability 439 |

## Waiting for a decision

A session researched these, proposed a change and left it for review. The next
session decides and records why; you can also answer in `OWNER_FOCUS.md`.

- **decide whether to log (not show, not score) one line per stock - the number of 13F filers holding it and the change on the prior quarter, dated by quarter end and filing date - as a context-log...** - open 2 days - `research/2026-11-03-following-the-smart-money.md`
- **register in `research/experiments/registry.yaml`, before running it, a research-engine experiment that drops `short_interest_ratio` from Revisions (and a second that scores FINRA days to cover from...** - open 2 days - `research/2026-11-03-short-interest-days-to-cover.md`

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-09 (in 4 days) - The context signals' first one-month reading arrives with this run; read the brief's `context_eval` line (CLAUDE.md 0.11). The three option signals wait for 2026-12-04 (below). `CLAUDE.md`
- 2026-11-16 (in 11 days) - Re-run the filing cross-check's range table; promote total assets, equity and operating cash flow to a gate at 25 only if each stayed <= 5 on every run. `research/2026-10-14-yahoo-vs-sec.md`

## What changed in the repo

- `7d7c841 data: screener run 2026-11-05 - 499 scored, top: VLO MPC EXPE MU SNDK`
- `4eb5ebf Log: the stale agent worktrees are removed`
- `6188976 Log: the data loop's blind spots - where the rate limit hits, a landed refusal, retention, named stocks`
- `2780d72 Browser tests open a stock the payload carries, so an index change cannot discard a night's data`
- `425b492 Data loop: keep the screener's whole output and where the rate limit hit; a landed refusal is not "held back"`
- `86b207d brief: code session 2026-11-04-midday`
- `60e7d49 Log: the implied-volatility proposal is accepted, and the fetch has no ex-date`
- `ef9bc75 Decide the implied-volatility proposal: ACCEPTED, and the switch costs one day not a month`
- `3074106 Record which construction an implied volatility came from, so adopting the inversion splices nothing`
- `fb27f1c Pre-register the QuantLib review's replication before computing anything`
- `9f3c66a Log: the 06:00 work landed; a refused branch is named to the next session`
- `ca674ac Runner: a refused branch is named to the next session, and a failed gate keeps its output`
- `3d76cfd brief: code session 2026-11-04`
- `39c7080 Log: the auto-rescue stash is not empty - git stash show -p cannot see an -u stash`
- `05030b7 Log and owner queue: the Investment fold decided, and the page rebuilt to carry it`

## The session's own account

> 2026-11-04 (owner-run, evening) - "keep finding improvements please, be creative": the data loop's blind spots
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-04_130001-midday.log`
> (completed). Data loop published? yes - `logs/datarun-2026-11-04_020001.log` (completed; HEALTH: PASS).
> Evidence base at `1m`: 5 of 8 effective observations needed (43 rows), newest 1m run date 2026-10-02
> (**33 days**, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The
> Factor Lab** (age **23 days**). **Tests:** 3,896 passed / 1 skipped at the morning's merge; **3,908
> passed / 1 skipped** on this tree (parallel, 3:15). All five gates passed in a worktree outside
> OneDrive; pushed while no session ran; tag `good/2026-11-04-owner-2`.
> 
> ### Did
> 
> 1. **The 02:00 run rate-limits itself, every night - and nothing recorded where.** Seven consecutive
>    02:00 runs: 39-51 tickers rate-limited, a 125-155 s retry pass, every ticker recovered. The
>    2026-11-02 run had **no fetch of any kind for 72 hours before it** (no `runs/` folder since 10-30,
>    no owner session, no weekend option refresh) and still read 45 - the control the timing note's
>    section 8.5 asked for, so the "leftover budget from an earlier run" reading is refuted
>    (`research/2026-10-26-data-run-timing.md` section 8.7). The data loop logged only the last 25 lines
>    of the screener's output, so the batch-by-batch throttle was discarded every morning. Now:
>    - `data-run.ps1` saves the whole output as `logs/datarun-<stamp>.screener.txt` (never fatal);
>    - `meta.json` records `rate_limited_positions` / `_tickers` / `rate_limit_errors`
>      (`run_screener.rate_limit_detail`; provenance only, `factor_engine.py` untouched so no ledger
>      rerun).
>    No setting changed: the note says what a week of both must show before one does.
> 2. **A refused session whose work later landed is no longer reported as held back.** The 13:00
>    session's health line said "held back by its gates" about the 06:00 run whose work this morning's
>    owner session had merged, and that session spent its opening looking for a branch. The verdict now
>    asks git whether the newest commit the runner listed is on `origin/main` ("held back by its gates,
>    and that work has since reached main"; the health line says **yes**). A sha git cannot place keeps
>    the warning. Tested in a bare-origin sandbox.
> 3. **Log retention counts runs, not files.** `prune_artifacts.prune_logs(keep=60)` counted every file,
>    so each new sidecar (transcripts, the morning's `.gate1/.gate5.txt`, tonight's `.screener.txt`)
>    silently shortened the history kept and could prune a run's saved gate output while its log
>    survived. Now a run's sidecars live and die with its `.log`.
> 4. **An index change can no longer discard a night's data through a browser test.** Four modules in
>    the 02:00 publish gate opened EXPE, APA, HST and BBY by name (the queued "37 named-ticker
>    references"); a deletion from the S&P 500 would have failed the gate on every run. `tests/examples.py`
>    returns the named stock unchanged while the payload carries it - every test reads exactly as before
>    today - else the best-ranked stock with a full sheet. **Measured on a copy of the payload with those
>    four removed: the old modules failed 6 tests, the new ones none that open a stock** (two others
>    failed on that doctored copy because removing stocks moves the median and the profile ranks - an
>    artifact of the surgery, not of a name). JPM and MSFT stay named: their tests assert what only
>    those companies are. Tripwire: `tests/test_browser_examples.py` (caught the old APA line when put
>    back; one false positive, a test that deletes FOX itself, exempted by rule).
> 
> ...

---

If a run says **stopped deliberately** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

