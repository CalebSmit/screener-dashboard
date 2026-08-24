# Morning Brief - Monday 24 August 2026, 02:11

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## THE ROUTINE IS NOT RUNNING

- **Code session** has not run since 2 days ago

Nothing below is current. A loop that stops firing writes no log, so
the rest of this page describes the last run that *did* happen, not
today. Most likely cause: the PC rebooted and nobody logged back in -
the tasks only run while a user is signed in. See NIGHTLY_LOG.md
2026-08-20 and `scripts/register-tasks.ps1`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran 2 days ago |
| Dashboard data from | 2026-08-24T02:00:03.959537 |
| Stocks scored | 501 |
| With a price | 501/501 |
| With an analyst target | 497/501 |
| Top 5 | HST, EXPE, APA, EIX, CF |
| Evidence for weight changes | 3 of 8 needed, newest 2026-02-22 - STALE, nothing new in 183 days |

## What changed in the repo

- `37ab0c9 data: screener run 2026-08-24 - 501 scored, top: HST EXPE APA EIX CF`

## The session's own account

> 2026-08-21 (later) - Acting on the retrospective's owner-flagged items
> 
> Owner-run session, not a scheduled one. The 2026-08-21 retrospective ended with
> five items it could not action itself. Three are now done.
> 
> **Health numbers:** last code session **ran and shipped** (06:16, `good/2026-08-21-0616`);
> data loop **published** 02:12; `live_ic_history.csv` = **3 rows, newest
> 2026-02-22**; priority 0 **unfixed**.
> 
> ### Did
> 
> **1. Prompts moved out of `.claude/`.** `prompts/nightly.md` and
> `prompts/retrospective.md`. The retrospective could not edit its own templates -
> `.claude/` is treated as sensitive - so the two prompt changes it wanted were
> left undone and `nightly.md` kept describing a rotation that no longer existed.
> A self-improving routine that cannot edit its own instructions is only half a
> loop. Runner updated to `prompts\$TemplateName`.
> 
> **2. `prompts/nightly.md` rotation synced to `CLAUDE.md`.** It still described
> Monday component-research / Tuesday practitioner-research. The runner injects
> the focus string so behaviour was already correct, but the prompt body
> contradicted it - a session would read one thing in its instructions and
> another in `CLAUDE.md`. Now: Mon research (both halves, one session), Tue
> product, Wed synthesis, Thu build, Fri harden.
> 
> **3. `scripts/register-tasks.ps1`** - the scheduled tasks are in version control
> at last. Open since 08-10, called the highest-value infrastructure item. Creates
> both tasks from a single definition, idempotent, with the at-logon catch-up
> included, and documents *why* `LogonType` is `InteractiveToken` (the alternative
> needs a stored password, which this project will not do). Supersedes
> `add-catchup-trigger.ps1`.
> 
> **4. The brief now shouts when a loop stops.** `write_brief.py` read the newest
> log and reported its outcome as if current - so a loop that stopped firing
> entirely kept the status page reporting the last successful run. That is how
> 08-17..20 passed unnoticed and how the machine sat at a login screen for six
> days while the brief said everything was fine. Each loop now shows *when it
> last ran*, and a `THE ROUTINE IS NOT RUNNING` banner appears above everything
> else when either has been quiet for 2+ days. Verified against a synthetic
> 5-day-old log and a missing log; does not fire for today's runs.
> 
> ### Corrected
> 
> **The retrospective's "~$6/session" is wrong** and is now flagged as such in
> `CLAUDE.md`. The owner runs Claude Max: sessions draw on included subscription
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

