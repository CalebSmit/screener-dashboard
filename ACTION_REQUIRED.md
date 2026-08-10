# ACTION REQUIRED — the autonomous loop is jammed

**Written by the code-loop session of 2026-08-10. Delete this file once the two
commands below have been run.**

## What happened

The scheduled Claude Code session cannot use its own permissions. Everything in
`.claude/settings.json` → `permissions.allow` is denied at runtime:

```
python --version                            -> works
python -c "print('hello')"                  -> DENIED
python -m pytest tests/ test_screener.py -q -> DENIED
python run_screener.py --dry-run            -> DENIED
git add / git commit / git push             -> DENIED
WebSearch / WebFetch                        -> DENIED
```

This is the folder-trust path mismatch that `scripts/fix-trust.ps1` was written
for: Claude Code keys trust by path in `%USERPROFILE%\.claude.json`, the desktop
app writes it with backslashes, the CLI reads it with forward slashes, and an
untrusted workspace ignores its permission settings.

**Consequences:**

- Ship gates 1 and 2 cannot be executed, so no session may merge to `main`.
- The session could not commit its own work, so the working tree is dirty.
- `scripts/nightly-screener.ps1` refuses to start on a dirty tree, so **the
  6:00 AM session will not run again until this is cleared.**

## Fix — CORRECTED 2026-08-10 evening

**`scripts/fix-trust.ps1` does not work, and should not be relied on.** It was
run successfully on 2026-08-06 and the flag was verified `true`; by 2026-08-10
it was `false` again. Patching `hasTrustDialogAccepted` from outside does not
stick, because Claude Code rewrites `.claude.json` from its own internal state
and resets a project it has never actually been trusted in.

Trust must be granted through the CLI's **own** dialog, in an interactive
session started **from inside this folder**. Note the earlier interactive login
was run from `C:\WINDOWS\system32`, so the CLI never saw this project at all —
which is why it was never prompted for.

**Close any other Claude Code session using this folder first** (including a
desktop-app conversation open on this project), or the CLI will report the
workspace is already in use. Then:

```powershell
cd "C:\Users\smitc\OneDrive\Documents\Screener"
& "$env:APPDATA\npm\claude.cmd"
```

Answer **yes** to the trust prompt, then `/exit`.

Verify it stuck — this must print `trusted=True` for the **forward-slash** key:

```powershell
python -c "import json,os;d=json.load(open(os.path.expanduser('~/.claude.json'),encoding='utf-8'));[print(k,v.get('hasTrustDialogAccepted')) for k,v in d['projects'].items() if 'Screener' in k]"
```

**The commit step below is no longer needed** — the 18:51 data run on
2026-08-10 swept this session's uncommitted work into `main` via `git add -A`
(commit `0c9aa58`). That was itself a bug: the data loop was acting as a back
door around the code loop's ship gates. Fixed in `84e81e9`, which restricts the
data loop to staging only the artifacts it owns.

So `CLAUDE.md`, `NIGHTLY_LOG.md`, and the research note are already on `main`.
Nothing to commit; the branch `nightly/2026-08-10` can be deleted.

Once trust is verified working, delete this file and commit that deletion.

## Was the session wasted?

No. It produced the most important finding since the loop was set up:

**`research/2026-08-10-ic-evidence-independence.md`** — the priority-0 fix that
`CLAUDE.md` told every session to make first would, on its own, have made the
tool *worse*. It unblocks a backfill of 11 IC observations that contain at most
**2 non-overlapping** 30-day return windows, inflating the significance
t-statistic by **2.35×** and pushing a borderline factor through the
`min_ic_ir_for_auto_apply` gate at a false p of 0.049. With
`allow_auto_apply: true`, the engine would have begun rewriting factor weights
on that. `CLAUDE.md` priority 0 has been rewritten with a **STOP** and a 5-step
package that fixes it properly.

Read that note before doing any further work on the improvement engine.

## Also worth fixing while you are here

The scheduled-task definitions are **not in version control**
(`grep -rn "Register-ScheduledTask" .` finds nothing). The 2:00 AM data loop and
6:00 AM code loop exist only as hand-made Task Scheduler entries on one machine.
The data loop silently did not fire on Monday 2026-08-10 — there is no
`logs/datarun-2026-08-10*` — while the 6:00 AM code loop did. A missing
`WakeToRun` / `StartWhenAvailable` is the likely cause. Committing a
registration script would make both loops reproducible and fix the missed run.
