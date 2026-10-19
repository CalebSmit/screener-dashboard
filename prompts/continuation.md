**The runner has resumed your session once, because it ended with uncommitted work.** You are
still on branch `{{BRANCH}}`. `git status --porcelain` when your last reply ended:

```
{{STATUS}}
```

This routine runs under `claude --print`. **Ending a reply without a tool call ends the session**:
there is no later turn, so a background command's "you will be notified when it completes" is never
delivered, and every background command you started was killed when the session ended. Whatever it
was producing is incomplete - check, do not assume.

This is the only continuation you get, it is turn-capped, and the runner's ship gates run as soon
as it ends, on exactly what you leave. Finish by **{{DEADLINE}}** (local time):

1. **Restart, in the foreground, anything a killed job was producing.** For the research engine,
   `python scripts/rerun_experiments.py --limit 4` runs only the rows that are not yet current and
   fits one foreground call (about 8 minutes); repeat until it prints `STALE REMAINING: 0`, and
   commit `research/experiments/ledger.jsonl` and the report after each call. For anything else,
   start it in the background and wait with `python scripts/wait_for.py` (it blocks for at most
   9.5 minutes; call it again until it reports `DONE`).
2. **Run the tests that cover what you changed**, then commit with explicit paths (never
   `git add -A`) and push the branch. Do not merge or push `main` - the runner does that after the
   gates pass.
3. **If it cannot be finished in time, leave the tree clean anyway**: commit what is sound, revert
   what is not, and say in `NIGHTLY_LOG.md` exactly what is left and how to resume it, as the
   entry's Next. A clean tree with an honest Next is a good outcome; a dirty one is the failure
   this continuation exists for.
4. **Your final reply comes after** `git status --porcelain` prints nothing and `git push` has
   succeeded - never before, and never while a background command is still running.
