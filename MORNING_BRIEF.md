# Morning Brief - Tuesday 11 August 2026, 06:00

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **failed** |
| Code session (6 AM) | **failed** |
| Dashboard data from | 2026-08-11T02:00:03.400246 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 499/502 |
| Top 5 | HST, EXPE, EIX, APA, CF |
| Evidence for weight changes | 3 of 8 needed |

## Things that needed attention

- Unhandled error: The term 'Write-NativeOutput' is not recognized as the name of a cmdlet, function, script file, or operable program. Check the spelling of the name, or if a path was included, verify that the path is correct and try again.
- at <ScriptBlock>, C:\Users\smitc\OneDrive\Documents\Screener\scripts\data-run.ps1: line 203
- Brief step failed (non-fatal): The term 'Write-NativeOutput' is not recognized as the name of a cmdlet, function, script file, or operable program. Check the spelling of the name, or if a path was included, verify that the path is correct and try again.
- This folder is NOT trusted by the Claude Code CLI.
- The session would run but be denied python, pytest and git.
- Fix (interactively, with no other Claude session open):
- cd "C:\Users\smitc\OneDrive\Documents\Screener"
- & "$env:APPDATA\npm\claude.cmd"
- then answer YES to the trust prompt and /exit.
- Skipping today's run rather than burning a session on a jammed workspace.

## What changed in the repo

- `29b80c2 plan: record 8 candidate dashboard improvements with real cost estimates`
- `467c2ae plan: replace the API-key chatbot with generated per-stock summaries`
- `22a4968 feat: health gate against degraded runs + morning brief after every run`
- `6c8837e docs: log the missing-market-cap gap found during verification (16 stocks, unlogged)`
- `710621b data: full refresh - restore prices, analyst targets and correct rankings`
- `83be6c3 fix: fail fast on untrusted workspace, self-heal a dirty tree`
- `828bceb docs: correct the trust fix - fix-trust.ps1 does not persist, CLI must grant trust itself`
- `84e81e9 fix: restore dashboard methodology rendering; stop data loop bypassing gates`
- `0c9aa58 data: screener run 2026-08-10`

## The session's own account

> 2026-08-10 - RESEARCH. The priority-0 fix would have armed the engine on two overlapping observations.
> 
> **Tests:** before -/- , after -/- - **could not be run, see below**
> **Data loop:** stalled - no run today (Mon 2026-08-10); last successful run 2026-08-07
> **Code loop:** ran, but **could not execute Python at all**
> 
> ### The blocker, first, because it changes what this session could be
> 
> This is the first autonomous session to get past the runner and actually start
> work (2026-08-06 aborted on a `gh` false negative, 2026-08-07 on no network).
> It immediately hit the next link in the chain: **the unattended session cannot
> run Python.**
> 
> ```
> python --version                            -> works (3.12.10)
> python -c "print('hello')"                  -> denied
> python -m pytest tests/ test_screener.py -q -> denied
> python run_screener.py --dry-run            -> denied
> WebSearch / WebFetch                        -> denied
> ```
> 
> All of those are in `.claude/settings.json` -> `permissions.allow`. They are
> denied anyway, which means the project's permission settings are **not being
> applied** to the scheduled run. `scripts/fix-trust.ps1` predicts this symptom
> exactly in its own header: folder trust is keyed by path in
> `%USERPROFILE%\.claude.json`, the desktop app writes it with backslashes, the
> CLI reads it with forward slashes, and an untrusted workspace "ignores its
> permission settings".
> 
> **Ship gates 1 and 2 were therefore impossible to run, so nothing merged to
> `main` today.** `main` is untouched.
> 
> `git add` / `git commit` / `git push` are denied by the same cause, so **this
> session could not commit its own work.** The files below are sitting
> uncommitted in the working tree:
> 
> ```
>  M CLAUDE.md
>  M NIGHTLY_LOG.md
> ?? ACTION_REQUIRED.md
> ?? research/2026-08-10-ic-evidence-independence.md
> ```
> 
> **This will jam the loop.** `scripts/nightly-screener.ps1` checks
> `git status --porcelain` before starting (line ~192) and refuses to run on a
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

