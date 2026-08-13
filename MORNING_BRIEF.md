# Morning Brief - Thursday 13 August 2026, 02:00

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **stopped deliberately** |
| Code session (6 AM) | **failed** |
| Dashboard data from | 2026-08-12T08:04:59.455861 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | HST, EXPE, EIX, APA, CF |
| Evidence for weight changes | 3 of 8 needed |

## Things that needed attention

- Run is DEGRADED. Refusing to publish.
- Run discarded by health check - the live dashboard is unchanged.
- This folder is NOT trusted by the Claude Code CLI.
- The session would run but be denied python, pytest and git.
- Fix (interactively, with no other Claude session open):
- cd "C:\Users\smitc\OneDrive\Documents\Screener"
- & "$env:APPDATA\npm\claude.cmd"
- then answer YES to the trust prompt and /exit.
- Skipping today's run rather than burning a session on a jammed workspace.

## What changed in the repo

- `ad5a610 brief: data run 2026-08-12`
- `dc618ef data: screener run 2026-08-12 - 502 scored, top: HST EXPE EIX APA CF`
- `dd5098c fix: catch-up for runs missed while logged out`
- `895a8a0 governance: backtest decides nothing until 2027-02-11; research is the basis`

## The session's own account

> 2026-08-12 - Nothing ran. The PC rebooted overnight and nobody was logged in.
> 
> **Tests:** 526 -> 530 passed
> **Data loop:** did not fire | **Code loop:** did not fire
> 
> ### What happened
> 
> No logs for 2026-08-12 at all. Both tasks show Enabled/Ready, last run 08-11,
> next run **08-13** - today's 02:00 and 06:00 slots passed without firing.
> 
> `System Boot Time: 8/12/2026, 12:47:15 AM` - the machine restarted overnight,
> almost certainly a Windows update. It was powered on through both windows, so
> this was not a sleep or network problem.
> 
> **Cause:** both tasks use `LogonType: InteractiveToken` - they run *only while
> a user is logged on*. After the update reboot the machine sat at the login
> screen with no user session, so neither task could start. Nothing errored;
> there was simply nothing to write a log.
> 
> This was a known limitation from setup (running logged-out needs a stored
> password, which was deliberately avoided) but nothing guarded against it.
> 
> ### Fixed
> 
> **Run-once-per-day markers.** Both scripts now write
> `logs/.datarun-last-success` / `logs/.nightly-last-success` on a successful
> run and exit immediately if today's already succeeded. A *failed* run leaves no
> marker, so it is correctly retried. Both take `-Force` to override.
> 
> **`scripts/add-catchup-trigger.ps1`** adds an at-logon trigger (3 min delay) to
> each task, so a missed run is picked up when the owner next logs in. The
> markers make that safe - logging in repeatedly cannot re-run the loop.
> 
> *Requires the owner to run it:* modifying scheduled-task triggers is a
> persistence change and was correctly refused when attempted automatically.
> 
> **Static checks now cover 6 scripts, 24 assertions.**
> 
> ### Noticed, not fixed
> 
> The cleanest root-cause fix is the Windows setting *"Use my sign-in info to
> automatically finish setting up after an update or restart"* (Settings ->
> Accounts -> Sign-in options). That restores the user session after an update
> reboot, which is what these tasks need. It is a Windows account setting, not a
> repo change - owner action.
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

