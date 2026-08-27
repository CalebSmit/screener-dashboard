# Morning Brief - Wednesday 26 August 2026, 19:50

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **failed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-08-26T19:38:23.969875 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Evidence for weight changes | 2 of 8 needed at the 1m horizon (6 rows, but overlapping windows are not independent; 23 rows across all horizons), newest 2026-08-14 |

## What changed in the repo

- `d6074a9 data: screener run 2026-08-26 - 502 scored, top: MAA DOC KIM REG UDR`
- `5494086 brief: data run 2026-08-26`
- `4a622a8 product: Top 5 first, What Changed and Factor Analytics collapsed by default`
- `253be1d product: remove the model portfolio, give each stock an "about"`
- `9bed64f brief: code session 2026-08-26`
- `f055475 docs: record the fix, and correct the record it was diagnosed from`
- `3bf9798 guard: bound the blast radius of a withheld price series`
- `bdda9a7 fix: refuse a price series that mixes two split scales`
- `f997570 brief: data run 2026-08-26`
- `0a61fd4 data: screener run 2026-08-26 - 502 scored, top: HST EXPE APA EIX CF`
- `a32f390 brief: evening session 2026-08-25`
- `f2f5c74 chore: remove the last things that needed a human`

## The session's own account

> 2026-08-26 (evening) - Owner-directed: the model portfolio leaves the dashboard, stocks gain an "about"
> 
> **Not a scheduled session.** The owner asked for two specific changes in an
> interactive session and, separately, asked *how he is supposed to tell this
> routine what to focus on*. That question turned out to be the most important
> part of the evening: until tonight there was no answer. See "The channel" below.
> 
> ### Health numbers (rule 8)
> 
> | Check | Reading |
> |---|---|
> | Last code session ran? | `logs/nightly-2026-08-26_060001.log` - ran, shipped to main, tagged `good/2026-08-26` |
> | Data loop published? | `logs/datarun-2026-08-26_020001.log` - HEALTH: PASS, 0 fetch failures, 502 scored, published |
> | Evidence base | **23 rows, newest 2026-08-14, 2 effective observations at `1m`** (6 raw) |
> | Priority 0 | Fixed 2026-08-24, still holding |
> 
> The evidence base has not moved since 08-24 by row count. The 08-26 morning
> session recorded why: the next eligible snapshot becomes computable on
> **2026-08-27** with five queued behind it, and wrote itself a tripwire - if the
> count has not moved by tomorrow's session, rule 8 bites and that becomes the
> work regardless of rotation. **That tripwire is still armed and this session
> did not touch it.** Tomorrow: check `live_ic_history.csv` first.
> 
> ### The channel (the part worth keeping)
> 
> The owner had no way to direct this routine. `CLAUDE.md` priorities are written
> *by sessions, for sessions*; the weekly rotation is fixed; and he is explicitly
> not reading diffs. So a request like tonight's could only ever reach the system
> by him opening a chat and asking - which does not scale and leaves no record.
> 
> `OWNER_FOCUS.md` is now that channel: plain English, **Open** and **Done**
> headings, read during Orient *before* the rotation is consulted. Open items
> outrank the day's nominal focus. Only two things outrank an owner item - a
> stalled data loop and the ship gates - and the prompt now requires a session
> that defers one to *say so in the log*, because an unmentioned deferral is
> indistinguishable from an ignored request.
> 
> Wired into `prompts/nightly.md` (step 1) and `CLAUDE.md`. Pinned by
> `tests/test_owner_focus.py` (7 tests) - including that the reference appears
> between "## 1. Orient" and "## 2. Baseline", so it cannot drift to a position
> after the work is already chosen. **A silent channel looks exactly like an
> empty one**, which is the same shape as the evidence base sitting at 3 rows for
> 183 days while every run reported success.
> 
> While in `prompts/nightly.md` I also corrected two stale claims it was still
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

