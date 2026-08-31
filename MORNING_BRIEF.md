# Morning Brief - Monday 31 August 2026, 02:12

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **failed** - last ran yesterday |
| Dashboard data from | 2026-08-31T02:00:03.976670 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | HST, EXPE, EIX, APA, CF |
| Evidence for weight changes | 3 of 8 needed at the 1m horizon (8 rows, but overlapping windows are not independent; 28 rows across all horizons), newest 2026-08-24 |

## What changed in the repo

- `f4db5cd data: screener run 2026-08-31 - 502 scored, top: HST EXPE EIX APA CF`

## The session's own account

> 2026-08-29 (evening) - Owner-run: the stagger goes live, and a standing rule changes
> 
> The morning's CATCH-UP session fixed the logon-trigger collision (see above)
> but left `register-tasks.ps1` for the owner to run by hand, reasoning that the
> script unregisters both tasks before re-adding them and a failure partway
> through would leave the machine with neither.
> 
> Told this, the owner's answer was direct: *"never have it leave things for me
> to do, it should figure it out on its own, after all, it should be self
> improving."*
> 
> ### Did
> 
> **Ran it and verified, in that order.** `powershell -ExecutionPolicy Bypass
> -File scripts/register-tasks.ps1` re-registered both tasks; immediately
> confirmed with `Get-ScheduledTask` / `Get-ScheduledTaskInfo` rather than
> trusting the script's own summary line - `Screener Data Run` triggers at
> `delay=PT3M`, `Nightly Screener Improvement` at `delay=PT20M`, both `Ready`,
> `StartWhenAvailable`/`WakeToRun` intact. The morning session's caution about a
> partial failure was reasonable; the missing step was verifying afterward, not
> declining to run it - the script is idempotent, so a bad outcome is fixed by
> running it again, not by asking someone else to.
> 
> **The more durable change is rule 11.** Added to `CLAUDE.md`: apply a
> machine-level fix and verify it in the same session, rather than leaving a
> command for the owner. Also added to `prompts/nightly.md`, read at the very
> top before Orient, so it is standing operating instruction for every future
> session - not a note that only helps because a human happened to be in the
> loop tonight. The one exception written into both: if verification genuinely
> needs something outside the session's reach, the *next* session inherits it,
> never the owner.
> 
> ### Also confirmed, unrelated to the above
> 
> Re-verified all four ship gates independently rather than trusting the
> morning's `good/2026-08-29-1231` tag: **853/853 tests, tree clean**. The
> morning session's own nightly log (`logs/nightly-2026-08-29_121115.log`) is
> truncated after "Invoking Claude Code..." - almost certainly because an
> interactive `tail -f` watch on that exact file (mine, checking on the session's
> progress) held it open the whole time PowerShell's `Add-Content` tried to write
> to it, the same non-fatal `IOException` seen once before on 2026-08-26. The
> session's actual work - five commits, tests, the tag - was unaffected, since
> `$ErrorActionPreference = 'Continue'` makes a failed log write non-terminating.
> Noted so it isn't mistaken for a hang next time: **do not hold a scheduled
> run's own log file open with a live tail while it may still be writing to it.**
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

