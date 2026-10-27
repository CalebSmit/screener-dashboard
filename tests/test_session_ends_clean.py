"""An unattended session must not end its reply while work is running or uncommitted.

WHY THIS FILE EXISTS - 2026-10-19 (research/2026-10-19-session-ended-on-a-background-job.md).

Under ``claude --print`` the session ends at the first reply with no tool call. On 2026-10-18 and
2026-10-19 the 06:00 session started the research engine's 29-experiment rerun in the background,
wrote that it "will notify me when it finishes", and ended: no 429, no turn cap, no time limit -
``stop_reason: end_turn``, ``terminal_reason: completed``. The CLI's exit killed the rerun (1 of 29
rows, then 14 of 29), the gates found a dirty tree, gates 1 and 4 failed and the day's work went to
the salvage path.

The fix is in three places and each is pinned here: the prompt states the rule and the deadline;
two helpers make the right behaviour one command (a foreground wait, a resumable rerun); and the
runner resumes a session that still ends dirty once, *before* the gates and without changing them.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "nightly-screener.ps1"
REGISTER = ROOT / "scripts" / "register-tasks.ps1"
NIGHTLY = ROOT / "prompts" / "nightly.md"
RETRO = ROOT / "prompts" / "retrospective.md"
CONTINUATION = ROOT / "prompts" / "continuation.md"
SOURCE = RUNNER.read_text(encoding="utf-8-sig")
CODE = re.sub(r"(?m)#.*$", "", re.sub(r"<#.*?#>", "", SOURCE, flags=re.S))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _continuation_block() -> str:
    start = CODE.index("if (-not $SessionFailed) {")
    end = CODE.index('Write-Log "--- Ship gates ---"')
    assert start < end
    return CODE[start:end]


# ---------------------------------------------------------------------------- the prompt


def test_the_prompt_says_a_reply_without_a_tool_call_ends_the_session():
    text = NIGHTLY.read_text(encoding="utf-8")
    assert "claude --print" in text
    assert "ends the moment you write a reply that makes no tool call" in text
    assert "Never end a reply while a background command you started is still running" in text
    assert "git status --porcelain" in text


def test_the_prompt_names_the_foreground_wait_and_the_resumable_rerun():
    text = NIGHTLY.read_text(encoding="utf-8")
    assert "python scripts/wait_for.py" in text
    assert "python scripts/rerun_experiments.py" in text
    assert "47 minutes" in text, "the measured cost of a full rerun is what makes the trade visible"


def test_the_prompt_states_a_deadline_the_runner_fills_in():
    text = NIGHTLY.read_text(encoding="utf-8")
    assert "{{DEADLINE}}" in text
    assert "Replace('{{DEADLINE}}', $DeadlineText)" in CODE


def test_the_retrospective_carries_the_same_rule():
    text = RETRO.read_text(encoding="utf-8")
    assert "a reply with no tool call ends the session" in text
    assert "scripts/wait_for.py" in text


@pytest.mark.parametrize("path", [NIGHTLY, RETRO, CONTINUATION], ids=lambda p: p.name)
def test_every_placeholder_in_a_prompt_is_one_the_runner_replaces(path: Path):
    used = set(re.findall(r"\{\{([A-Z]+)\}\}", path.read_text(encoding="utf-8")))
    replaced = set(re.findall(r"Replace\('\{\{([A-Z]+)\}\}'", CODE))
    assert used <= replaced, f"{path.name} uses placeholders the runner never fills: {used - replaced}"


def test_the_continuation_prompt_forbids_the_same_mistake_and_the_merge():
    text = CONTINUATION.read_text(encoding="utf-8")
    assert "Ending a reply without a tool call ends the session" in text
    assert "{{STATUS}}" in text
    assert "Do not merge or push `main`" in text
    assert "never `git add -A`" in text or "never\n   `git add -A`" in text


# ---------------------------------------------------------------------------- the runner


def test_the_continuation_runs_after_the_session_and_before_the_gates():
    invoke = CODE.index("claude --print --permission-mode acceptEdits --output-format json")
    resume = CODE.index("--resume $sid")
    gates = CODE.index('Write-Log "--- Ship gates ---"')
    gate4 = CODE.index("GATE 4 clean tree")
    assert invoke < resume < gates < gate4, (
        "the continuation must sit between the session and the gates - never after them, so the "
        "gates always judge what it leaves")


def test_the_continuation_is_one_bounded_resume_of_the_same_session():
    block = _continuation_block()
    assert CODE.count("--resume") == 1, "exactly one continuation, never a loop"
    assert "--max-turns $ContinuationMaxTurns" in block
    assert "$RunStart.AddMinutes($ContinuationCutoffMinutes)" in block
    assert "Get-SessionId -TranscriptPath $JsonLog" in block
    assert "Invoke-Native 'git' @('status', '--porcelain')" in block, "it fires only on a dirty tree"


def test_the_continuation_cannot_publish_and_does_not_replace_any_gate():
    block = _continuation_block()
    for banned in ("'merge'", "'push'", "'reset'", "'stash'", "'checkout'", "SuccessMarker", "exit "):
        assert banned not in block, f"the continuation block must not contain {banned}"
    # every gate, the salvage commit and the revert path are still downstream of it
    tail = CODE[CODE.index('Write-Log "--- Ship gates ---"'):]
    for needed in ("GATE 1 tests", "GATE 2 dry-run", "GATE 3 dashboard artifacts", "GATE 4 clean tree",
                   "salvaged from an interrupted session", "revert-bad-merge.ps1"):
        assert needed in tail, needed


def test_a_session_that_never_ran_gets_no_continuation():
    """A 429 or an empty transcript is a lost session, not an unfinished one - resuming it would
    spend quota the usage-limit stand-down exists to protect."""
    assert _continuation_block().startswith("if (-not $SessionFailed) {")


def test_the_session_id_is_validated_before_it_reaches_a_command_line():
    fn = CODE[CODE.index("function Get-SessionId"):]
    fn = fn[:fn.index("\n}\n") if "\n}\n" in fn else fn.index("\r\n}\r\n")]
    assert "session_id" in fn
    assert "[0-9a-fA-F]{8}-" in fn and "{12}$" in fn


def _limits_minutes() -> list[int]:
    """The ExecutionTimeLimit of every task that runs nightly-screener.ps1."""
    src = REGISTER.read_text(encoding="utf-8-sig")
    out = []
    for spec in re.split(r"@\{", src)[1:]:
        if "'nightly-screener.ps1'" not in spec:
            continue
        m = re.search(r"Limit\s*=\s*\(New-TimeSpan -(Hours|Minutes) (\d+)\)", spec)
        assert m, "every code slot must declare its ExecutionTimeLimit"
        out.append(int(m.group(2)) * (60 if m.group(1) == "Hours" else 1))
    return out


def _runner_minutes(name: str) -> int:
    m = re.search(rf"\${name}\s*=\s*(\d+)", CODE)
    assert m, name
    return int(m.group(1))


def test_the_time_budget_fits_inside_every_code_slots_task_limit():
    """Task Scheduler stops the whole run at the limit - gates, salvage and brief included. The
    session's deadline, then a continuation that can still start, then ~6 min of gates (279 s of
    tests on 2026-10-19) and the brief must all fit; 45 minutes is the allowance after the cutoff."""
    limits = _limits_minutes()
    assert len(limits) == 3, "the 06:00, midday and weekend slots"
    budget = _runner_minutes("SessionBudgetMinutes")
    cutoff = _runner_minutes("ContinuationCutoffMinutes")
    assert budget < cutoff
    assert all(cutoff + 45 <= lim for lim in limits), (budget, cutoff, limits)
    assert 10 <= _runner_minutes("ContinuationMaxTurns") <= 100


def test_run_start_is_taken_before_the_lock_wait():
    """The four hours run from the task's start, so the deadline must too."""
    assert CODE.index("$RunStart") < CODE.index("Enter-RepoLock -LogDir")
    assert CODE.index("$RunStart") < CODE.index("Invoking Claude Code")


# ---------------------------------------------------------------------------- wait_for.py


def test_wait_for_returns_as_soon_as_the_marker_appears(tmp_path):
    W = _load("wait_for")
    f = tmp_path / "out.txt"
    f.write_text("a\nSTALE REMAINING: 0\n", encoding="utf-8")
    lines = []
    assert W.wait(f, "STALE REMAINING:", None, 60, 5, out=lambda s, **k: lines.append(s)) == 0
    assert lines[-1].startswith("DONE")


def test_wait_for_gives_up_inside_its_budget_and_says_call_again(tmp_path):
    W = _load("wait_for")
    t = {"now": 0.0}
    lines = []
    rc = W.wait(tmp_path / "missing.txt", "x", None, 100, 30, clock=lambda: t["now"],
                sleep=lambda s: t.__setitem__("now", t["now"] + s), out=lambda s, **k: lines.append(s))
    assert rc == W.STILL_WAITING
    assert t["now"] <= 100
    assert "Call this again in the foreground" in lines[-1]


def test_wait_for_counts_lines_and_never_exceeds_the_bash_ceiling(tmp_path):
    W = _load("wait_for")
    f = tmp_path / "ledger.jsonl"
    f.write_text("{}\n{}\n\n{}\n", encoding="utf-8")
    assert W.condition_met(f, None, 3)[0] is True
    assert W.condition_met(f, None, 4)[0] is False
    assert W.MAX_SECONDS_CAP < 600
    assert W.main(["--file", str(f)]) == 2, "a wait with no condition is an error, not forever"


# ---------------------------------------------------------------------------- rerun_experiments.py


def _row(i, code="c1", content="d1", base="b1", method=2):
    return {"id": i, "code_hash": code, "content_hash": content, "base_config_hash": base,
            "method_version": method}


CURRENT = {"code_hash": "c1", "content_hash": "d1", "base_config_hash": "b1", "method_version": 2}


def test_a_rerun_skips_what_is_current_and_reads_the_newest_row(tmp_path):
    R = _load("rerun_experiments")
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("\n".join(json.dumps(r) for r in [
        _row("a", code="old"), _row("b", code="old"), _row("a"),     # a was rerun; b was not
    ]) + "\n", encoding="utf-8")
    latest = R.latest_rows(ledger)
    assert latest["a"]["code_hash"] == "c1"
    assert R.stale_ids(["a", "b", "c"], latest, CURRENT) == ["b", "c"]


@pytest.mark.parametrize("key", ["code_hash", "content_hash", "base_config_hash", "method_version"])
def test_any_provenance_difference_makes_a_row_stale(key):
    R = _load("rerun_experiments")
    row = _row("a")
    row[key] = "other"
    assert R.stale_ids(["a"], {"a": row}, CURRENT) == ["a"]


def test_level_is_a_different_question_from_stale():
    R = _load("rerun_experiments")
    old = {i: _row(i, code="old") for i in ("a", "b")}
    assert R.rows_are_level(["a", "b"], old) is True
    assert R.stale_ids(["a", "b"], old, CURRENT) == ["a", "b"]
    mixed = {"a": _row("a"), "b": _row("b", code="old")}
    assert R.rows_are_level(["a", "b"], mixed) is False
    assert R.rows_are_level(["a", "b", "c"], old) is False, "a missing row is not level"


def test_a_limited_rerun_does_experiments_first_and_evaluations_once_they_are_level(tmp_path):
    R = _load("rerun_experiments")
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps(_row("e1")) + "\n", encoding="utf-8")

    class X:
        LEDGER = ledger
        METHOD_VERSION = 2

        @staticmethod
        def load_registry():
            return [{"id": i} for i in ("e1", "e2", "e3", "e4")]

        @staticmethod
        def load_evaluations():
            return [{"id": "v1"}]

        @staticmethod
        def code_fingerprint():
            return {"code_hash": "c1"}

        @staticmethod
        def content_fingerprint():
            return {"content_hash": "d1"}

        @staticmethod
        def base_config():
            return {}

        @staticmethod
        def config_hash(cfg, tr):
            return "b1"

        @staticmethod
        def load_controls():
            # a control whose arm is not built in this tmp dir: reported, never run, so it
            # cannot take a rerun down with it (2026-10-25)
            return [{"id": "ctl", "scores": "cache/nope.parquet", "built_by": "python x.py"}]

    # (experiments, evaluations, controls, stale total, blocked controls). The ordering is the
    # point: experiments first, then evaluations once the experiments are level, then controls -
    # a control is paired against the baseline, so it is worth nothing until that row is current.
    assert R.plan(X, 2)[:4] == (["e2", "e3"], [], [], 4)
    assert R.plan(X, 4)[:4] == (["e2", "e3", "e4"], ["v1"], [], 4)
    assert R.plan(X, None)[:4] == (["e2", "e3", "e4"], ["v1"], [], 4)
    # the unbuilt arm is named with the command that builds it rather than silently dropped
    blocked = R.plan(X, None)[4]
    assert len(blocked) == 1 and "ctl" in blocked[0] and "python x.py" in blocked[0]


def test_the_rerun_helper_is_outside_the_engines_result_paths():
    """Adding or editing it must not move a ledger row's code_hash - that is what forced the
    2026-10-19 rerun in the first place."""
    sys.path.insert(0, str(ROOT))
    import v2_experiments as X
    assert not any("rerun_experiments" in p or "wait_for" in p for p in X.RESULT_PATHS)
    assert "ledger.jsonl" not in " ".join(X.RESULT_PATHS), "committing the ledger mid-rerun must not stale it"


# ---------------------------------------------------------------------------- rerun_experiments.py --jobs N
# 2026-10-26: worker processes, so the ~47-minute full rerun is not one serial chain. What these pin
# is the scheduler's contract, on a fake engine run in real worker processes: rows reach the real
# ledger in the serial order and nowhere else first; only the parent opens it; a worker that fails or
# drops a row stops the run with the serial prefix appended, never a ledger with a hole in it.

FAKE_ENGINE = '''
import json, os, time
from pathlib import Path

LEDGER = Path(os.environ["FAKE_REAL_LEDGER"])
CONF = json.loads(os.environ.get("FAKE_ENGINE_CONF", "{}"))


def _row(i, kind):
    log = Path(os.environ["FAKE_WRITES_DIR"]) / f"{os.getpid()}.log"
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": i, "ledger": str(LEDGER)}) + "\\n")
    time.sleep(CONF.get("delay", {}).get(i, 0))
    if i in CONF.get("fail", []):
        raise RuntimeError("boom " + i)
    if i in CONF.get("skip", []):
        return
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": i, "kind": kind, "pid": os.getpid()}) + "\\n")


def run(ids):
    for i in ids:
        _row(i, "experiment")


def run_evaluations(ids):
    for i in ids:
        _row(i, "evaluation")


def run_controls(ids):
    for i in ids:
        _row(i, "control")
'''

EXPS = ["baseline", "e1", "e2", "e3", "e4", "e5", "e6"]
EVS = ["v1", "v2"]
CTRLS = ["c1"]


@pytest.fixture
def fake_engine(tmp_path, monkeypatch):
    """A fake engine importable by the worker processes, and the 'real' ledger it defaults to."""
    (tmp_path / "fake_rerun_engine.py").write_text(FAKE_ENGINE, encoding="utf-8")
    real = tmp_path / "real-ledger.jsonl"
    writes = tmp_path / "writes"
    writes.mkdir()
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv("FAKE_REAL_LEDGER", str(real))
    monkeypatch.setenv("FAKE_WRITES_DIR", str(writes))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "fake_rerun_engine", raising=False)
    X = importlib.import_module("fake_rerun_engine")

    def configure(**conf):
        monkeypatch.setenv("FAKE_ENGINE_CONF", json.dumps(conf))

    def writes_made():
        out = []
        for p in writes.glob("*.log"):
            out += [json.loads(ln) | {"pid": p.stem} for ln in p.read_text(encoding="utf-8").splitlines()]
        return out

    yield X, real, configure, writes_made
    sys.modules.pop("fake_rerun_engine", None)


def _ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [json.loads(ln)["id"] for ln in path.read_text(encoding="utf-8").splitlines()]


def test_parallel_work_is_dealt_deterministically_and_every_id_once():
    R = _load("rerun_experiments")
    work = R.assign(EXPS, EVS, CTRLS, 3)
    assert work == R.assign(EXPS, EVS, CTRLS, 3), "the split must not depend on anything but its inputs"
    dealt = [i for a in work for i in a["experiments"] + a["evaluations"] + a["controls"]]
    assert sorted(dealt) == sorted(EXPS + EVS + CTRLS) and len(dealt) == len(set(dealt))
    # experiments round-robin in registry order, so rows finish roughly in the serial order
    assert [a["experiments"] for a in work] == [["baseline", "e3", "e6"], ["e1", "e4"], ["e2", "e5"]]
    # each group builds its own inputs once, so it is never split between workers
    assert sum(bool(a["evaluations"]) for a in work) == 1 and sum(bool(a["controls"]) for a in work) == 1
    assert R.serial_order(EXPS, EVS, CTRLS) == EXPS + EVS + CTRLS
    assert len(R.assign(["e1"], [], [], 4)) == 1, "never more workers than pieces of work"
    assert R.MAX_JOBS <= 4, "a worker holds every monthly record; raise this only with a new measurement"


def test_jobs_1_and_a_single_piece_of_work_run_serially_in_process(monkeypatch):
    R = _load("rerun_experiments")
    calls = []

    class X:
        run = staticmethod(lambda ids: calls.append(("run", ids)))
        run_evaluations = staticmethod(lambda ids: calls.append(("evaluate", ids)))
        run_controls = staticmethod(lambda ids: calls.append(("control", ids)))

    monkeypatch.setattr(R, "run_parallel", lambda *a, **k: pytest.fail("must not start workers"))
    R.execute(X, ["baseline", "e1"], ["v1"], ["c1"], jobs=1)
    assert calls == [("run", ["baseline", "e1"]), ("evaluate", ["v1"]), ("control", ["c1"])]
    calls.clear()
    R.execute(X, ["baseline"], [], [], jobs=4)            # one piece of work: no worker to share it
    assert calls == [("run", ["baseline"])]


def test_parallel_rows_reach_the_ledger_in_the_serial_order_and_only_through_the_parent(fake_engine):
    R = _load("rerun_experiments")
    X, real, configure, writes_made = fake_engine
    # the first worker is the slowest, so the others finish rows far out of serial order
    configure(delay={"baseline": 1.5, "e3": 0.5})
    assert R.run_parallel(X, EXPS, EVS, CTRLS, 3, engine="fake_rerun_engine", poll_seconds=0.05) == 10
    assert _ids(real) == EXPS + EVS + CTRLS
    writes = writes_made()
    assert len(writes) == 10
    assert all(Path(w["ledger"]).resolve() != real.resolve() for w in writes), \
        "a worker opened the real ledger - only the parent may append to it"
    by_worker: dict[str, set] = {}
    for w in writes:
        by_worker.setdefault(w["pid"], set()).add(w["ledger"])
    files = [next(iter(v)) for v in by_worker.values()]
    assert all(len(v) == 1 for v in by_worker.values()) and len(set(files)) == len(files) == 3, \
        "one private file per worker"
    # every row was written by a worker process, then copied by the parent
    rows = [json.loads(ln) for ln in real.read_text(encoding="utf-8").splitlines()]
    assert {r["kind"] for r in rows} == {"experiment", "evaluation", "control"}
    assert all(r["pid"] != os.getpid() for r in rows)


def test_a_failed_worker_stops_the_run_with_the_serial_prefix_appended(fake_engine):
    R = _load("rerun_experiments")
    X, real, configure, _ = fake_engine
    real.write_text(json.dumps({"id": "older"}) + "\n", encoding="utf-8")
    configure(fail=["e4"], delay={"e5": 0.3, "e6": 0.3})
    with pytest.raises(SystemExit, match="failed"):
        R.run_parallel(X, EXPS, EVS, CTRLS, 3, engine="fake_rerun_engine", poll_seconds=0.05)
    # everything before e4 in the serial order, nothing after it even if it finished
    assert _ids(real) == ["older", "baseline", "e1", "e2", "e3"]


def test_a_worker_that_drops_a_row_is_an_error_not_a_shorter_ledger(fake_engine):
    R = _load("rerun_experiments")
    X, real, configure, _ = fake_engine
    configure(skip=["v1"])
    with pytest.raises(SystemExit, match="no row for"):
        R.run_parallel(X, EXPS, EVS, CTRLS, 2, engine="fake_rerun_engine", poll_seconds=0.05)
    assert _ids(real) == EXPS


def test_a_worker_refuses_to_write_the_real_ledger(fake_engine):
    R = _load("rerun_experiments")
    X, real, _, _ = fake_engine
    spec = {"worker": 0, "engine": "fake_rerun_engine", "ledger": str(real),
            "experiments": ["e1"], "evaluations": [], "controls": []}
    with pytest.raises(SystemExit, match="never write the real ledger"):
        R.worker_main(spec)
    assert not real.exists()


def test_a_worker_stops_when_its_parent_goes(fake_engine, tmp_path):
    """A session cut off mid-rerun kills the parent; its workers must not run on for half an hour."""
    _, _, configure, _ = fake_engine
    configure(delay={"e1": 60})
    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    spec = {"worker": 0, "engine": "fake_rerun_engine", "ledger": str(tmp_path / "w.jsonl"),
            "experiments": ["e1"], "evaluations": [], "controls": [], "parent_pid": parent.pid}
    p = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "rerun_experiments.py"), "--worker",
                          json.dumps(spec)], stdin=subprocess.DEVNULL)
    try:
        time.sleep(1.0)
        assert p.poll() is None, "the worker must run while its parent does"
        parent.kill()                                # the session is cut off
        assert p.wait(timeout=20) == 3
    finally:
        for proc in (p, parent):
            if proc.poll() is None:
                proc.kill()
    assert not (tmp_path / "w.jsonl").exists()


def test_a_worker_that_starts_a_subprocess_does_not_hang(fake_engine, tmp_path):
    """The real engine shells out to git for every row. The first parent-watcher read the worker's
    stdin, and on Windows a pending pipe read blocks the handle duplication a child process needs:
    every real worker hung at its first git call while this file's fake engine, which started no
    process, passed. So the fake engine here does what the real one does."""
    X, real, _, _ = fake_engine
    (tmp_path / "fake_rerun_engine.py").write_text(
        FAKE_ENGINE.replace("def run(ids):\n",
                            "def run(ids):\n    import subprocess, sys\n"
                            "    subprocess.run([sys.executable, '-c', 'pass'], check=True, timeout=30)\n"),
        encoding="utf-8")
    R = _load("rerun_experiments")
    t0 = time.monotonic()
    R.run_parallel(X, ["baseline", "e1"], [], [], 2, engine="fake_rerun_engine", poll_seconds=0.05)
    assert time.monotonic() - t0 < 60
    assert _ids(real) == ["baseline", "e1"]
