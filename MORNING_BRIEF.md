# Morning Brief - Friday 25 September 2026, 07:33

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-09-25T02:00:03.515421 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | EXPE, HST, VLO, MPC, EIX |
| Evidence for weight changes | 3 of 8 needed at the 1m horizon (16 rows, but overlapping windows are not independent; 55 rows across all horizons), newest 2026-09-18 |

## What changed in the repo

- `841c45d log: changelog entries and nightly log for 2026-09-25`
- `7d4a6c5 docs: public methodology page stops describing the pre-2026-09-10 Revisions category`
- `1e97e1b fix: two-source metric fallbacks can now actually use their second source`
- `a1b878c brief: data run 2026-09-25`
- `aad81aa data: screener run 2026-09-25 - 502 scored, top: EXPE HST VLO MPC EIX`
- `d2e231c brief: code session 2026-09-24`
- `8c13217 log: close the stale worktree left by 2026-09-23`
- `7a08281 log: nightly 2026-09-24 - backtest v2 step 1, survivorship sized`
- `f099c4a docs: record the survivorship result where it is acted on`
- `b822a01 measure: survivorship bias in backtest.py is 4.3%/yr`
- `6206bdc backtest-v2: point-in-time S&P 500 membership`
- `b05b9fd brief: data run 2026-09-24`
- `4ca65a5 data: screener run 2026-09-24 - 502 scored, top: EXPE HST VLO BBY BMY`

## The session's own account

> 2026-09-25 - HARDEN AND TEACH. Tests, docs, error handling, and the investment-club experience. Would a finance student understand what they are looking at?
> 
> **Health (rule 8, all five):** last code session ran? **yes** -
> `logs/nightly-2026-09-24_060001.log` ends "Run complete: shipped to main",
> tagged `good/2026-09-24` | data loop published? **yes** -
> `logs/datarun-2026-09-25_020001.log` ends "Data loop complete", HEALTH: PASS,
> 502 scored, top EXPE HST VLO MPC EIX | evidence base at `1m` = **16 rows,
> newest 2026-08-26 (30 days ago, bound 40), 3 effective** - steady-state lag,
> healthy | priority 0 **fixed** (2026-08-24, untouched) | top open roadmap item:
> **priority 3, backtest v2, 31 days old** - advanced but not closed on
> 2026-09-24; **deferred today**, see below
> **Tests:** before **1498/1498**, after **1549/1549** (+51 new, no pre-existing
> failures)
> **Owner queue / rotation:** `OWNER_FOCUS.md` **Open is empty**, so the rotation
> governed; nothing to move to Done. Took Friday's harden-and-teach focus.
> **Deferred priority 3** - its named next step is "cost a price source for
> delisted tickers", a procurement decision rather than a hardening task, and
> today's focus had a nine-session-old error-handling item sitting inside it.
> 
> ### Did
> 
> **Closed the `currentPrice` fallback decision, open since 2026-09-11 and carried
> by nine sessions - and found the item had been scoped to the harmless half of
> its own root cause.**
> 
> **1. The root cause is one idiom, used nine times.** `compute_metrics()` reads
> nine numeric inputs that have two possible sources, every one written as the
> nested form `d.get(A, d.get(B, np.nan))`. That reaches `B` only when key `A` is
> **absent**. `_fetch_single_ticker_inner()` writes all nine keys unconditionally -
> `_safe()` and `_stmt_val()` both return NaN rather than omitting the key - so the
> fallback can only fire on an exception path, never on the missing-data path it
> was written for. Replaced with one `_coalesce()` helper that skips a
> present-but-NaN or `None` value.
> 
> **2. The inherited framing was that incidence is zero. It is zero only for
> price.** The 2026-09-11 session checked `currentPrice` on the live payload, found
> all 502 names carried a price, and concluded this was "a safety net that does not
> exist rather than a bug that is firing". That was right about `currentPrice` and
> wrong about the four lines immediately above it, which share the defect. Measured
> on the four retained raw fetches (`runs/*/00_raw_fetch.parquet`, 503 names each,
> identical on all four):
> 
> | first source NaN, backup usable | names |
> |---|---|
> | `totalDebt` -> `totalDebt_bs` | 1 (FISV) |
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

