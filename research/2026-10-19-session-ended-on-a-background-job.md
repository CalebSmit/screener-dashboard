# 2026-10-19 - Why two 06:00 sessions ended with uncommitted work

A process finding, not a methodology one. It changes no score and no weight.

## The symptom

Two of the last three 06:00 code sessions ended with uncommitted work. Gate 4 (clean tree) and
gate 1 (tests) failed, and the runner's salvage path committed the leftovers to `nightly/*` for a
later session to finish.

| Session | Claude ran (local) | Left in the tree | Gate 1 |
|---|---|---|---|
| 2026-10-18 06:00 weekend | 06:00:08 - 06:13:50 | `ledger.jsonl` (M), `research/measurements/2026-10-18-provenance-rerun-effect.py` (??) | 5 failed (`test_v2_experiments` provenance tests) |
| 2026-10-19 06:00 | 06:00:10 - 07:03:46 | `research.html`, `RESULTS.md`, `ledger.jsonl`, `summary.json` (all M) | 4 failed (same module) |

Source: `logs/nightly-2026-10-18_060001-weekend.log`, `logs/nightly-2026-10-19_060001.log`.

## What it was not

- **Not a usage limit (429), turn cap or time limit.** Both CLI results read `is_error: false`,
  `api_error_status: null`, `subtype: success`, `stop_reason: end_turn`,
  `terminal_reason: completed` (`logs/nightly-2026-10-19_060001.json`: 222 turns, 3,810 s;
  `...-10-18_060001-weekend.json`: 102 turns, 816 s). Neither full transcript
  (`~/.claude/projects/.../7ec2c2aa-....jsonl`, `12c39559-....jsonl`) contains an API-error message.
  The runner passes no `--max-turns`, and the task's 4-hour `ExecutionTimeLimit` was nowhere near.
- **Not a tool error that stopped work.** Several commands were refused by the tool guard (below),
  but each was followed by more work; none ended the session.

## What it was

Both sessions **chose to end their turn while a background job was running**, expecting to be
woken when it finished. Under `claude --print` that cannot happen: the session ends at the first
reply without a tool call, and the CLI's exit kills the background job.

**2026-10-19.** Registering one experiment (`drop-revenue-growth`) changed `registry.yaml`, which is
in `v2_experiments.RESULT_PATHS`, so every ledger row's `code_hash` went stale and the provenance
tests went red until all 29 were rerun. The transcript (UTC; local is UTC-5):

- 11:40:16Z - `python v2_experiments.py run > logs/_rerun_all.out 2>&1; echo "EXIT=$?"` in the
  background: refused ("contains multiple operations").
- 11:40:22Z - `python v2_experiments.py run`, `run_in_background: true`: started, "You will be
  notified when it completes."
- 12:01:38Z - a `Monitor` wait loop over the task output: refused ("contains shell syntax that
  cannot be statically analyzed"). The model's reply: "The background task will notify on
  completion."
- 12:03:18Z - progress check: 14 rows written.
- 12:03:41Z - final reply, `end_turn`: "the engine rerun is at 14/29 and will notify me when it
  finishes ... I'll finish the rerun, regenerate the report, verify that no number moved, then run
  gates 1 and 4 and push." The CLI exited at 07:03:46 local.

The salvage commit `0524d36` holds exactly 15 rows on the new `code_hash` `eb7acc42822f`: the single
`drop-revenue-growth` run (06:38:55) and the full rerun's first 14, the last stamped **07:02:56**.
At 1.6 minutes per experiment the 15th was due at about 07:04:30; it never came. The job died with
the session.

**2026-10-18.** Same shape, earlier. 11:13:31Z - an `until ...; do sleep 60; done` wait in the
background: refused ("contains shell syntax that cannot be statically analyzed"). 11:13:45Z - final
reply, `end_turn`: "The rerun is underway (1 of 29 rows written when last checked); its completion
will notify me. Rather than poll, I'll wait for that."

**The counter-example that shows the fix.** The 2026-10-18 13:00 session ran the same 29-experiment
rerun twice and finished cleanly, because it waited **in the foreground** with
`python -c "... time.sleep(55) ..."` polling loops, each under ten minutes. One of its waits set a
900 s timeout and was moved to the background at 600 s ("Command did not complete within its 600s
timeout and was moved to the background") - the Bash tool's ceiling, and the same trap in another
form.

## Why the existing rule did not catch it

`prompts/nightly.md` said "Commit as you go. A usage limit can end this session at any moment."
Both sessions did commit as they went (20 commits on 10-19). The rule is written for an *external*
interruption; the failure here is the model's own mistaken belief that a later turn exists. Nothing
told the session that, under `--print`, its own reply is the end.

## Measured cost of the rerun

From the 2026-10-19 rows: baseline built in ~1.7 min (06:40:22 -> 06:42:05), then ~1.6 min per
experiment, so ~47 minutes for 29. The engine appends each row as it is computed, so the work was
resumable - but the only command reran everything from the first experiment, so 14 finished rows
would have been recomputed. Applying `scripts/rerun_experiments.py`'s rule to the salvaged ledger
selects 15 ids to finish (14 experiments and the evaluation) instead of 30.

## What changed

1. **`prompts/nightly.md`** - "How this session ends": never end a reply while a background job runs
   or the tree is dirty; wait in the foreground; the measured rerun cost; a stated deadline
   (`{{DEADLINE}}`). `prompts/retrospective.md` carries the short form.
2. **`scripts/wait_for.py`** - the 10-18 13:00 session's working pattern as one command the tool
   guard accepts: blocks up to 540 s (capped at 570, under the 600 s ceiling) for text or a line
   count in a file; exit 3 means call again.
3. **`scripts/rerun_experiments.py`** - reruns only the rows not on the current code, record
   contents, baseline weighting and method version, through the engine's own `run` /
   `run_evaluations`; `--limit 4` (~8 min) fits one foreground call; `--status` changes nothing and
   distinguishes "stale against the tree" from "not level with each other" (only the second is what
   the report and tests flag). Not in `RESULT_PATHS`, so adding it moves no `code_hash`.
4. **`scripts/nightly-screener.ps1`** - (a) the prompt's `{{DEADLINE}}` = task start + 180 min,
   inside the 4-hour `ExecutionTimeLimit`, which stops the whole run including gates and salvage;
   (b) if the session ran but left uncommitted work, resume the **same** session once
   (`claude --print --resume <session_id> --max-turns 60`) with `prompts/continuation.md`, only if
   it can start within 190 min of the task's start. The four gates, the salvage commit and
   `revert-bad-merge.ps1` run afterwards exactly as before; a failed or skipped continuation is a
   WARN. It is not attempted after a session that did not run (`$SessionFailed`, e.g. a 429).

Pinned by `tests/test_session_ends_clean.py`.

## Not verified here, and why

The continuation path has not executed in production: this session could not run PowerShell or
the scheduled task. The static tests check its placement and guards, and the repo's PowerShell
parser test parses the script. The next session that ends dirty is the first live test - its log
will show `CONTINUATION:` lines, and its JSON is `logs/nightly-<stamp>-continuation.json`.
