"""Rerun only the research-engine rows that are out of date - so a rerun cut off halfway resumes.

WHY THIS EXISTS - 2026-10-19 (research/2026-10-19-session-ended-on-a-background-job.md).

``python v2_experiments.py run`` reruns every registered experiment from the start. A rerun is
routine - registering one experiment changes ``registry.yaml``, which is in ``RESULT_PATHS``, so
every row's ``code_hash`` goes stale and the provenance tests stay red until all of them are rerun
(plan/research-engine.md) - and it is long: measured on 2026-10-19, about 1.7 minutes to build the
baseline and then about 1.6 minutes per experiment, so roughly 47 minutes for 29.

Twice in two days a session was cut off inside that rerun (14 of 29 rows written on 2026-10-19, 1
of 29 on 2026-10-18), and both times the rows already written were thrown away in effect, because
the only command available starts again from the first experiment. The engine already appends each
row the moment it is computed, so the work is resumable; nothing was using that.

This picks the registered experiments (and evaluations) whose newest ledger row does not match the
current code, record contents, baseline weighting and method version, and runs only those - all of
them, or the first ``--limit`` - through the engine's own ``run`` / ``run_evaluations``, then
rewrites the report. It is a scheduler over the engine, never a second implementation of it: every
number still comes from ``v2_experiments``. Rows it writes are ordinary ledger rows.

    python scripts/rerun_experiments.py --status     # what is stale; changes nothing
    python scripts/rerun_experiments.py --limit 4    # ~8 minutes: fits one foreground call
    python scripts/rerun_experiments.py              # everything stale, then the report

The last line it prints is always ``STALE REMAINING: <n>``, which is what
``scripts/wait_for.py --contains "STALE REMAINING:"`` waits on when the run is backgrounded. Commit
the ledger after every call: ``ledger.jsonl`` is not in ``RESULT_PATHS``, so committing it does not
change ``code_hash`` and the next call carries on where this one stopped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# What makes a ledger row current. ``code_hash`` covers every file in ``RESULT_PATHS``, the registry
# (and so each transform) included; ``content_hash`` the monthly records and their sources;
# ``base_config_hash`` the baseline weighting; ``method_version`` how a comparison is computed. The
# report flags a mixture of any of the first three and of the fourth - so these are the fields a
# rerun has to bring level.
PROVENANCE_KEYS = ("code_hash", "content_hash", "base_config_hash", "method_version")


def latest_rows(ledger: Path) -> dict[str, dict]:
    """The newest row per id - the row the report reads (``v2_experiments.report``)."""
    latest: dict[str, dict] = {}
    if not ledger.exists():
        return latest
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[row["id"]] = row
    return latest


def stale_ids(ids: list[str], latest: dict[str, dict], current: dict) -> list[str]:
    """``ids`` (in registry order) whose newest row is missing or differs on any provenance key."""
    return [i for i in ids
            if i not in latest or any(latest[i].get(k) != current[k] for k in PROVENANCE_KEYS)]


def rows_are_level(ids: list[str], latest: dict[str, dict]) -> bool:
    """Whether every id has a row and all those rows share one provenance - what the report checks."""
    if any(i not in latest for i in ids):
        return False
    return len({tuple(latest[i].get(k) for k in PROVENANCE_KEYS) for i in ids}) <= 1


def current_provenance(X) -> dict:
    return {"code_hash": X.code_fingerprint()["code_hash"],
            "content_hash": X.content_fingerprint()["content_hash"],
            "base_config_hash": X.config_hash(X.base_config(), {"kind": "baseline"}),
            "method_version": X.METHOD_VERSION}


def plan(X, limit: int | None) -> tuple[list[str], list[str], int]:
    """(experiments to run now, evaluations to run now, stale count before running)."""
    current = current_provenance(X)
    latest = latest_rows(X.LEDGER)
    exps = stale_ids([e["id"] for e in X.load_registry()], latest, current)
    evs = stale_ids([e["id"] for e in X.load_evaluations()], latest, current)
    total = len(exps) + len(evs)
    if limit is not None:
        exps = exps[:max(limit, 0)]
        evs = evs if len(exps) < limit else []          # evaluations go once the experiments are level
        evs = evs[:max(limit - len(exps), 0)]
    return exps, evs, total


def remaining(X) -> int:
    current = current_provenance(X)
    latest = latest_rows(X.LEDGER)
    return (len(stale_ids([e["id"] for e in X.load_registry()], latest, current))
            + len(stale_ids([e["id"] for e in X.load_evaluations()], latest, current)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rerun only the stale research-engine rows.")
    ap.add_argument("--status", action="store_true", help="report what is stale and change nothing")
    ap.add_argument("--limit", type=int, default=None,
                    help="run at most this many stale ids (experiments first, then evaluations)")
    a = ap.parse_args(argv)
    if a.limit is not None and a.limit < 1:
        ap.error("--limit must be at least 1")

    sys.path.insert(0, str(ROOT))
    import v2_experiments as X     # research tooling only; never imported by production

    exps, evs, total = plan(X, a.limit)
    if a.status:
        stale = plan(X, None)
        ids = [e["id"] for e in X.load_registry()] + [e["id"] for e in X.load_evaluations()]
        level = rows_are_level(ids, latest_rows(X.LEDGER))
        print("stale experiments:", ", ".join(stale[0]) or "none")
        print("stale evaluations:", ", ".join(stale[1]) or "none")
        # Not the same question. The report and its tests ask whether the rows are level with EACH
        # OTHER; "stale" asks whether they match the tree as it is now. A change that cannot move a
        # number (2026-10-18: a report-wording fix to v2_experiments.py) leaves every row stale and
        # level - which is not by itself a reason to spend 47 minutes rerunning.
        print(f"rows level with each other: {'yes' if level else 'NO - the report flags a mixture; rerun'}")
        print(f"STALE REMAINING: {total}", flush=True)
        return 0
    print(f"{total} stale; running {len(exps)} experiment(s) and {len(evs)} evaluation(s) now", flush=True)
    if exps:
        X.run(exps)
    if evs:
        X.run_evaluations(evs)
    X.report()
    print(f"wrote {X.RESULTS.relative_to(X.ROOT)} - commit the ledger and the report now", flush=True)
    print(f"STALE REMAINING: {remaining(X)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
