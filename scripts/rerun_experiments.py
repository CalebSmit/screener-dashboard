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

This picks the registered experiments (and evaluations, and controls) whose newest ledger row does
not match the current code, record contents, baseline weighting and method version, and runs only
those - all of them, or the first ``--limit`` - through the engine's own ``run`` /
``run_evaluations`` / ``run_controls``, then rewrites the report. It is a scheduler over the engine,
never a second implementation of it: every number still comes from ``v2_experiments``. Rows it
writes are ordinary ledger rows.

A control (2026-10-25) carries one more input than the four provenance keys can see - the arm's own
pre-built scores file - so its recorded SHA-256 is compared with the file on disk as well, and an
arm that has not been built yet is printed rather than run (see ``stale_controls``).

    python scripts/rerun_experiments.py --status     # what is stale; changes nothing
    python scripts/rerun_experiments.py --limit 4    # ~8 minutes: fits one foreground call
    python scripts/rerun_experiments.py              # everything stale, then the report
    python scripts/rerun_experiments.py --jobs 4     # the same, in four worker processes

``--jobs N`` (2026-10-26; default 1, which is exactly the serial behaviour above; capped at
``MAX_JOBS``). The stale ids are dealt to worker processes (``assign``: experiments round-robin in
registry order, the evaluations and the controls each as one group, since each group builds its own
inputs once). Each worker runs its share through the engine's own ``run`` / ``run_evaluations`` /
``run_controls`` with ``v2_experiments.LEDGER`` pointed at a private temp file, so it is still a
scheduler: no number is computed here, and nothing in ``RESULT_PATHS`` changed to make it possible.
**Only this parent process opens the real ledger**, and it appends a worker's row only when every
row before it in the serial order (experiments, then evaluations, then controls, each in registry
order) is also finished - so the ledger is always a prefix of what a serial run would have written,
in the same order, each row the engine's own line. Cut off, the next call carries on exactly as
after a cut-off serial run; a worker that fails, or exits without writing a row it was given, stops
the run with that prefix appended and an error, never a ledger with a hole in it. A worker exits by
itself when the parent dies (``_exit_with_parent``).

Measured 2026-10-26 on 10 ids (baseline, five experiments, both evaluations, the control), serial
and ``--jobs 4`` each into a scratch ledger: **every field of every row identical** except
``run_at`` - and identical, except the code provenance (``config.yaml`` moved that day), to the real
ledger's rows for the same ids from the full serial run of 2026-10-25. Serial 1,109 s, ``--jobs 4``
754 s, both on a machine other work was loading heavily (another full test suite with six workers
ran throughout). The subset is build-heavy - four workers build the baseline four times - so the
full rerun gains more: from the serial per-row times (build ~140 s, experiment ~115 s, evaluation
group ~140 s, control ~139 s) and the ~1.5x slowdown each worker showed running beside three others,
the 32 rows extrapolate to **~63 min serial and ~27 min at ``--jobs 4``** under that load (the
2026-10-19 serial figure on a quieter machine was ~47 min). Peak memory **~2.0 GB per worker**
(1.95-2.02 GB, the serial process 2.02 GB), which is what sets ``MAX_JOBS``.

The last line it prints is always ``STALE REMAINING: <n>``, which is what
``scripts/wait_for.py --contains "STALE REMAINING:"`` waits on when the run is backgrounded. Commit
the ledger after every call: ``ledger.jsonl`` is not in ``RESULT_PATHS``, so committing it does not
change ``code_hash`` and the next call carries on where this one stopped.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
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


def stale_controls(X, latest: dict[str, dict], current: dict) -> tuple[list[str], list[str]]:
    """(controls to rerun, controls that cannot be run yet and why).

    A control carries one input the four provenance keys above cannot see: the arm's own pre-built
    scores file. ``content_hash`` covers ``cache/v2_metrics/`` - the point-in-time records, which are
    the *baseline* side of the pairing - so rebuilding the arm would leave a control row looking
    current while the numbers under it had moved. Its SHA-256 is recorded in the row
    (``control_data.scores_sha256``) and compared with the file on disk here.

    An arm that has not been built is reported, never run: ``run_controls`` would raise and take the
    whole rerun down with it, and a scheduler that cannot finish because one optional row is
    unbuilt is worse than one that says which row to build.
    """
    todo, blocked = [], []
    for e in X.load_controls():
        p = ROOT / e["scores"]
        if not p.is_file():
            blocked.append(f"{e['id']}: {e['scores']} not built - run: {e.get('built_by', '?')}")
            continue
        now = X._file_sha256(p)[:16]
        row = latest.get(e["id"])
        if row is None or any(row.get(k) != current[k] for k in PROVENANCE_KEYS) \
                or (row.get("control_data") or {}).get("scores_sha256") != now:
            todo.append(e["id"])
    return todo, blocked


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


def plan(X, limit: int | None) -> tuple[list[str], list[str], list[str], int, list[str]]:
    """(experiments, evaluations, controls to run now; stale count before running; blocked controls)."""
    current = current_provenance(X)
    latest = latest_rows(X.LEDGER)
    exps = stale_ids([e["id"] for e in X.load_registry()], latest, current)
    evs = stale_ids([e["id"] for e in X.load_evaluations()], latest, current)
    ctrls, blocked = stale_controls(X, latest, current)
    total = len(exps) + len(evs) + len(ctrls)
    if limit is not None:
        room = max(limit - len(exps), 0)
        exps = exps[:max(limit, 0)]
        evs = evs[:room]                   # evaluations go once the experiments are level
        ctrls = ctrls[:max(room - len(evs), 0)]
    return exps, evs, ctrls, total, blocked


def remaining(X) -> int:
    current = current_provenance(X)
    latest = latest_rows(X.LEDGER)
    return (len(stale_ids([e["id"] for e in X.load_registry()], latest, current))
            + len(stale_ids([e["id"] for e in X.load_evaluations()], latest, current))
            + len(stale_controls(X, latest, current)[0]))


# --------------------------------------------------------------------------- --jobs N (2026-10-26)
# Worker processes, each running a share of the stale ids through the engine's own ``run`` /
# ``run_evaluations`` / ``run_controls`` with ``LEDGER`` pointed at a private temp file. Only this
# (parent) process ever appends to the real ledger, and only the longest run of finished rows that
# starts where it left off in the serial order - see ``run_parallel``.

# The most workers ``--jobs`` will start. Measured 2026-10-26 (the docstring): a worker peaks at
# 1.95-2.02 GB - every monthly record plus the baseline ranking, the same as the serial process's
# 2.02 GB - so four take ~8 GB of a 32 GB machine that other work shares, and four of its six cores.
# Raise it only with a new measurement.
MAX_JOBS = 4
ENGINE = "v2_experiments"
WORKER_FLAG = "--worker"

# Relative cost of the pieces of work, in units of one experiment's ranking. Measured serially on
# 2026-10-26 (a loaded machine): an experiment 113-145 s (whole-market 55 s), the baseline build
# ~140 s, the evaluation group ~140 s and the control ~139 s - each group is one build of its own
# inputs (``bot_month_inputs`` / the baseline ranking) plus about a second a row, which is why a
# group goes to one worker whole: split, it would be built twice.
BUILD_COST = 1.2
EXPERIMENT_COST = 1.0
EVALUATION_COST = 0.01     # per evaluation, on top of the group's build
CONTROL_COST = 0.01        # per control, on top of the group's build


def serial_order(exps: list[str], evs: list[str], ctrls: list[str]) -> list[str]:
    """The order a serial run writes rows in: experiments in registry order, then evaluations, then
    controls (``main``'s three calls, each of which walks its own section in registry order)."""
    return list(exps) + list(evs) + list(ctrls)


def assign(exps: list[str], evs: list[str], ctrls: list[str], jobs: int) -> list[dict]:
    """Split the work between at most ``jobs`` workers, deterministically.

    Experiments are dealt round-robin in registry order, so the workers finish rows roughly in the
    serial order and the prefix the parent may append grows steadily instead of waiting on one long
    chunk. The evaluations, then the controls, go as one group each to the least-loaded worker (ties
    to the later worker, which was dealt fewer experiments)."""
    units = len(exps) + bool(evs) + bool(ctrls)
    n = max(1, min(jobs, units))
    work = [{"worker": w, "experiments": [], "evaluations": [], "controls": []} for w in range(n)]
    load = [0.0] * n
    for k, i in enumerate(exps):
        w = k % n
        if not work[w]["experiments"]:
            load[w] += BUILD_COST
        work[w]["experiments"].append(i)
        load[w] += EXPERIMENT_COST
    for key, ids, each in (("evaluations", evs, EVALUATION_COST), ("controls", ctrls, CONTROL_COST)):
        if ids:
            w = min(range(n), key=lambda j: (load[j], -j))
            work[w][key] = list(ids)
            load[w] += BUILD_COST + each * len(ids)
    for w, a in enumerate(work):
        a["est_cost"] = round(load[w], 2)
    return [a for a in work if a["experiments"] or a["evaluations"] or a["controls"]]


def peak_memory_mb() -> float | None:
    """This process's peak resident memory in MB (Windows: PeakWorkingSetSize), or None."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class _Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            c = _Counters()
            c.cb = ctypes.sizeof(c)
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters),
                                                    wintypes.DWORD]
            if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return None
            return round(c.PeakWorkingSetSize / 2**20, 1)
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except Exception:          # a measurement, never a reason to fail a run
        return None


def _wait_for_exit(pid: int) -> None:
    """Block until process ``pid`` has ended (returns at once if it is already gone)."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        h = k32.OpenProcess(0x00100000, False, pid)          # SYNCHRONIZE
        if not h:
            return
        k32.WaitForSingleObject(h, 0xFFFFFFFF)               # INFINITE
        return
    while os.getppid() == pid:
        time.sleep(2)


def _exit_with_parent(parent_pid: int) -> None:
    """End this worker when the parent goes, so a session cut off mid-rerun does not leave workers
    computing rows nobody will append.

    It waits on the parent's *process*, not on a pipe from it. The first version blocked a thread
    reading stdin, and on Windows that hung every real worker at its first ``git`` call: a child
    process inherits the standard handles, and duplicating a pipe handle with a synchronous read
    pending on it blocks (measured 2026-10-26: four workers at 0.2 s of CPU after nine minutes; the
    fake engine in the tests never started a subprocess, so they passed).
    ``test_a_worker_that_starts_a_subprocess_does_not_hang`` pins it."""
    def watch():
        try:
            _wait_for_exit(parent_pid)
        except Exception:
            return                     # cannot watch: the worker just runs to the end
        os._exit(3)
    threading.Thread(target=watch, daemon=True).start()


def worker_main(spec: dict) -> int:
    """One worker: its share of the ids through the engine's own functions, rows to its own file."""
    sys.path.insert(0, str(ROOT))
    X = importlib.import_module(spec.get("engine") or ENGINE)
    mine = Path(spec["ledger"]).resolve()
    if mine == Path(X.LEDGER).resolve():
        raise SystemExit("a worker must never write the real ledger - only the parent appends to it")
    X.LEDGER = mine
    t0 = time.monotonic()
    if spec["experiments"]:            # never an empty list: the engine reads an empty one as "all"
        X.run(spec["experiments"])
    if spec["evaluations"]:
        X.run_evaluations(spec["evaluations"])
    if spec["controls"]:
        X.run_controls(spec["controls"])
    print(f"worker {spec['worker']}: {len(spec['experiments']) + len(spec['evaluations']) + len(spec['controls'])}"
          f" row(s) in {time.monotonic() - t0:.0f}s, peak memory {peak_memory_mb()} MB", flush=True)
    return 0


class WorkerFailed(SystemExit):
    pass


def _complete_rows(path: Path) -> list[str]:
    """The rows a worker has finished: only lines ending in a newline (the engine writes each row
    and its newline in one call, so a line without one is a row still being written)."""
    if not path.exists():
        return []
    data = path.read_bytes()
    # the engine writes in text mode ("\r\n" on Windows); the parent appends in text mode too, so the
    # real ledger gets the same bytes a serial run would have written
    return [ln.rstrip(b"\r").decode("utf-8") for ln in data.split(b"\n")[:-1] if ln.strip()]


def run_parallel(X, exps: list[str], evs: list[str], ctrls: list[str], jobs: int, *,
                 engine: str = ENGINE, poll_seconds: float = 2.0) -> int:
    """Run the stale ids in up to ``jobs`` worker processes; return the number of rows appended.

    The guarantee: at every moment, ``X.LEDGER`` gains exactly the rows a serial run would have
    written by then, in the same order (``serial_order``), each a byte-for-byte copy of the line the
    engine wrote in its worker - nothing beyond a gap, nothing twice, nothing partial. The parent is
    the only process that opens the real ledger, and it appends a row only once every row before it
    in the serial order is also finished. So a run cut off at any point leaves a prefix of the serial
    run's rows, which ``plan`` then reads as current and the next call carries on after. Finished
    rows past the first unfinished one are lost on a cut-off (at most about ``jobs`` of them, since
    experiments are dealt round-robin) - the price of keeping the ledger in one order.

    A worker that exits non-zero stops the run: the prefix is appended, the other workers are
    killed, and ``WorkerFailed`` is raised - as a serial run would have stopped at that row. A worker
    that exits cleanly without writing every row it was given is a failure too, never a silent drop.
    """
    order = serial_order(exps, evs, ctrls)
    work = assign(exps, evs, ctrls, jobs)
    ledger = Path(X.LEDGER)
    tmp = Path(tempfile.mkdtemp(prefix="rerun-experiments-"))
    owner: dict[str, int] = {}
    for a in work:
        a["engine"] = engine
        a["ledger"] = str(tmp / f"worker-{a['worker']}.jsonl")
        for i in a["experiments"] + a["evaluations"] + a["controls"]:
            owner[i] = a["worker"]
    assert sorted(owner) == sorted(order) and len(owner) == len(order), "every id goes to exactly one worker"
    print(f"--jobs {jobs}: {len(work)} worker(s); "
          + "; ".join(f"worker {a['worker']}: {len(a['experiments'])} exp, {len(a['evaluations'])} eval, "
                      f"{len(a['controls'])} control (est. {a['est_cost']} units)" for a in work), flush=True)
    procs: dict[int, subprocess.Popen] = {}
    finished: dict[str, str] = {}
    appended = 0

    def collect() -> None:
        for a in work:
            for line in _complete_rows(Path(a["ledger"])):
                rid = json.loads(line)["id"]
                if owner.get(rid) != a["worker"]:
                    raise WorkerFailed(f"worker {a['worker']} wrote a row for {rid!r}, which it was not given")
                if rid in finished and finished[rid] != line:
                    raise WorkerFailed(f"worker {a['worker']} wrote {rid!r} twice")
                finished[rid] = line

    def append_prefix() -> None:
        nonlocal appended
        lines = []
        while appended + len(lines) < len(order) and order[appended + len(lines)] in finished:
            lines.append(finished[order[appended + len(lines)]])
        if lines:
            ledger.parent.mkdir(parents=True, exist_ok=True)
            with ledger.open("a", encoding="utf-8") as f:
                f.write("".join(ln + "\n" for ln in lines))
            appended += len(lines)

    try:
        for a in work:
            spec = {k: a[k] for k in ("worker", "engine", "ledger", "experiments", "evaluations", "controls")}
            spec["parent_pid"] = os.getpid()
            procs[a["worker"]] = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), WORKER_FLAG, json.dumps(spec)],
                stdin=subprocess.DEVNULL, cwd=str(ROOT))
        while True:
            codes = {w: p.poll() for w, p in procs.items()}
            collect()
            append_prefix()
            bad = {w: c for w, c in codes.items() if c not in (None, 0)}
            if bad:
                raise WorkerFailed(
                    f"worker(s) {sorted(bad)} failed (exit {sorted(set(bad.values()))}); appended the "
                    f"{appended} row(s) before the first unfinished one "
                    f"({order[appended] if appended < len(order) else 'none'!r}) and stopped "
                    "the rest - rerun to carry on")
            if all(c == 0 for c in codes.values()):
                break
            time.sleep(poll_seconds)
        collect()
        append_prefix()
        if appended < len(order):
            raise WorkerFailed(f"workers exited cleanly but wrote no row for {order[appended:]} - "
                               f"appended the {appended} before it")
        return appended
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.kill()
        for p in procs.values():
            try:
                p.wait(timeout=30)
            except Exception:
                pass
        shutil.rmtree(tmp, ignore_errors=True)


def execute(X, exps: list[str], evs: list[str], ctrls: list[str], jobs: int = 1) -> None:
    """Run the given ids: serially in this process (``jobs`` 1, or only one piece of work - exactly
    the three calls this script always made), or in worker processes (``run_parallel``). Either way
    every row comes from the engine's own functions and reaches ``X.LEDGER`` in the serial order."""
    if jobs > 1 and len(assign(exps, evs, ctrls, jobs)) > 1:
        X.refuse_edited_tree()     # once, here, rather than as N identical worker failures
        run_parallel(X, exps, evs, ctrls, jobs)
        return
    if exps:
        X.run(exps)
    if evs:
        X.run_evaluations(evs)
    if ctrls:
        X.run_controls(ctrls)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == [WORKER_FLAG]:
        spec = json.loads(argv[1])
        if spec.get("parent_pid"):
            _exit_with_parent(int(spec["parent_pid"]))
        code = 1
        try:
            code = worker_main(spec)
        except SystemExit as e:
            if e.code not in (None, 0) and not isinstance(e.code, int):
                print(e.code, file=sys.stderr)
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        except BaseException:
            import traceback
            traceback.print_exc()
        finally:
            # leave through os._exit, without waiting on the watcher thread or the finalizers (the
            # first, stdin-reading watcher crashed the interpreter's shutdown with 0xC0000005)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(code)
    ap = argparse.ArgumentParser(description="Rerun only the stale research-engine rows.")
    ap.add_argument("--status", action="store_true", help="report what is stale and change nothing")
    ap.add_argument("--limit", type=int, default=None,
                    help="run at most this many stale ids (experiments first, then evaluations)")
    ap.add_argument("--jobs", type=int, default=1,
                    help=f"worker processes (default 1: serial; at most {MAX_JOBS}); rows reach the "
                         "ledger in the serial order either way")
    a = ap.parse_args(argv)
    if a.limit is not None and a.limit < 1:
        ap.error("--limit must be at least 1")
    if a.jobs < 1:
        ap.error("--jobs must be at least 1")
    jobs = min(a.jobs, MAX_JOBS, os.cpu_count() or 1)
    if jobs < a.jobs:
        print(f"--jobs {a.jobs} capped at {jobs} (MAX_JOBS {MAX_JOBS}, {os.cpu_count()} cores)", flush=True)

    sys.path.insert(0, str(ROOT))
    import v2_experiments as X     # research tooling only; never imported by production

    exps, evs, ctrls, total, blocked = plan(X, a.limit)
    if a.status:
        stale = plan(X, None)
        ids = ([e["id"] for e in X.load_registry()] + [e["id"] for e in X.load_evaluations()]
               + [e["id"] for e in X.load_controls() if e["id"] in latest_rows(X.LEDGER)])
        level = rows_are_level(ids, latest_rows(X.LEDGER))
        print("stale experiments:", ", ".join(stale[0]) or "none")
        print("stale evaluations:", ", ".join(stale[1]) or "none")
        print("stale controls:", ", ".join(stale[2]) or "none")
        for b in stale[4]:
            print("control not runnable:", b)
        # Not the same question. The report and its tests ask whether the rows are level with EACH
        # OTHER; "stale" asks whether they match the tree as it is now. A change that cannot move a
        # number (2026-10-18: a report-wording fix to v2_experiments.py) leaves every row stale and
        # level - which is not by itself a reason to spend 47 minutes rerunning.
        print(f"rows level with each other: {'yes' if level else 'NO - the report flags a mixture; rerun'}")
        print(f"STALE REMAINING: {total}", flush=True)
        return 0
    for b in blocked:
        print("control not runnable:", b, flush=True)
    print(f"{total} stale; running {len(exps)} experiment(s), {len(evs)} evaluation(s) and "
          f"{len(ctrls)} control(s) now", flush=True)
    execute(X, exps, evs, ctrls, jobs)
    X.report()
    print(f"wrote {X.RESULTS.relative_to(X.ROOT)} - commit the ledger and the report now", flush=True)
    print(f"STALE REMAINING: {remaining(X)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
