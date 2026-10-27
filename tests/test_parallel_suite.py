"""The suite runs in parallel where pytest-xdist is installed, and exactly as before where not.

2026-10-26 (owner-run night). Gate 1 took 6:10-7:20 serially and runs about three times in
every scheduled code session, fourteen a week, under the Bash tool's 600 s ceiling; the data
loop's claims gate is a pytest run too. Both now add ``-n auto`` when xdist is importable.

What has to stay true, and is pinned here:

* **A missing plugin never fails a gate.** Without xdist, ``scripts/run_tests.py`` and
  ``scripts/check_published_claims.py`` run the serial command they always ran, byte for byte;
  where xdist cannot run the suite at all (pytest exit 3 or 4 - not a verdict on any test) the
  serial command decides. A test that *fails* is never retried.
* **The published-artifact tripwire still fires under xdist** (root ``conftest.py``), on a real
  pytest-xdist run of a throwaway project, and still restores the file.
* **The runners read the same output.** Gate 1 logs pytest's last lines and decides on the exit
  code alone; the claims gate prints the last line. Under xdist the last line is still pytest's
  ``N passed in Xs`` summary.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "nightly-screener.ps1"
SOURCE = RUNNER.read_text(encoding="utf-8-sig")
CODE = re.sub(r"(?m)#.*$", "", re.sub(r"<#.*?#>", "", SOURCE, flags=re.S))

HAS_XDIST = importlib.util.find_spec("xdist") is not None
needs_xdist = pytest.mark.skipif(
    not HAS_XDIST, reason="MISSING PLUGIN: pytest-xdist is not installed here - the parallel "
                          "path cannot be exercised (the serial path is tested regardless)")

SUMMARY = re.compile(r"^\d+ passed( in [\d.]+s)?")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run_tests = _load("run_tests", ROOT / "scripts" / "run_tests.py")
claims_gate = _load("check_published_claims_for_parallel", ROOT / "scripts" / "check_published_claims.py")

ARGS = ["tests/", "test_screener.py", "-q"]
SERIAL = [sys.executable, "-m", "pytest", *ARGS]


# --------------------------------------------------------------------------- scripts/run_tests.py


def test_without_xdist_the_command_is_the_serial_one_byte_for_byte(monkeypatch):
    monkeypatch.setattr(run_tests, "xdist_available", lambda: False)
    mode, cmd = run_tests.plan(ARGS)
    assert cmd == SERIAL
    assert "not importable" in mode


def test_with_xdist_the_only_change_is_the_worker_count(monkeypatch):
    monkeypatch.setattr(run_tests, "xdist_available", lambda: True)
    _, cmd = run_tests.plan(ARGS)
    assert cmd == SERIAL + ["-n", "auto"]


@pytest.mark.parametrize("extra", [["-n", "4"], ["-n4"], ["-nauto"], ["--numprocesses=2"],
                                   ["--dist", "loadfile"], ["-p", "no:xdist"], ["-pno:xdist"]])
def test_a_caller_that_chooses_its_own_distribution_is_left_alone(monkeypatch, extra):
    monkeypatch.setattr(run_tests, "xdist_available", lambda: True)
    _, cmd = run_tests.plan(ARGS + extra)
    assert cmd == SERIAL + extra


def test_serial_can_be_forced(monkeypatch):
    monkeypatch.setattr(run_tests, "xdist_available", lambda: True)
    assert run_tests.plan(ARGS, force_serial=True)[1] == SERIAL


def _calls(monkeypatch, codes):
    seen = []
    it = iter(codes)
    monkeypatch.setattr(run_tests.subprocess, "call", lambda cmd, **k: (seen.append(cmd), next(it))[1])
    return seen


@pytest.mark.parametrize("rc", [3, 4], ids=["internal-error", "usage-error"])
def test_xdist_failing_to_run_the_suite_falls_back_to_serial(monkeypatch, rc, capsys):
    monkeypatch.setattr(run_tests, "xdist_available", lambda: True)
    seen = _calls(monkeypatch, [rc, 0])
    assert run_tests.main(ARGS) == 0
    assert seen == [SERIAL + ["-n", "auto"], SERIAL]
    assert "rerunning serially" in capsys.readouterr().out


@pytest.mark.parametrize("rc", [0, 1, 2, 5])
def test_a_verdict_from_the_parallel_run_is_final(monkeypatch, rc):
    """A failing test (1) is never retried - a retry would turn a real failure into a coin toss."""
    monkeypatch.setattr(run_tests, "xdist_available", lambda: True)
    seen = _calls(monkeypatch, [rc])
    assert run_tests.main(ARGS) == rc
    assert len(seen) == 1


def test_the_serial_path_runs_once_and_returns_pytests_code(monkeypatch):
    monkeypatch.setattr(run_tests, "xdist_available", lambda: False)
    seen = _calls(monkeypatch, [4])
    assert run_tests.main(ARGS) == 4
    assert seen == [SERIAL]


# --------------------------------------------------------------------------- the claims gate


def _gate_cmd_serial():
    return [sys.executable, "-m", "pytest", *claims_gate.module_paths(),
            "-q", "--no-header", "-p", "no:cacheprovider"]


def test_the_claims_gate_without_xdist_runs_its_pre_parallel_command():
    assert claims_gate.pytest_command(False) == _gate_cmd_serial()
    assert claims_gate.pytest_command(True) == _gate_cmd_serial() + ["-n", "auto"]


class _Proc:
    def __init__(self, rc, out):
        self.returncode, self.stdout, self.stderr = rc, out, ""


def _gate_runs(monkeypatch, results):
    seen = []
    it = iter(results)
    monkeypatch.setattr(claims_gate.subprocess, "run", lambda cmd, **k: (seen.append(cmd), next(it))[1])
    return seen


def test_the_claims_gate_runs_serially_when_xdist_is_absent(monkeypatch, capsys):
    monkeypatch.setattr(claims_gate, "xdist_available", lambda: False)
    seen = _gate_runs(monkeypatch, [_Proc(0, "1452 passed in 190.0s\n")])
    assert claims_gate.run([]) == 0
    assert seen == [_gate_cmd_serial()]
    assert "PASS (1452 passed in 190.0s) [serial]" in capsys.readouterr().out


@pytest.mark.parametrize("rc", [3, 4])
def test_the_claims_gate_falls_back_to_serial_when_xdist_cannot_run(monkeypatch, rc, capsys):
    monkeypatch.setattr(claims_gate, "xdist_available", lambda: True)
    seen = _gate_runs(monkeypatch, [_Proc(rc, "INTERNALERROR\n"), _Proc(0, "1452 passed in 190.0s\n")])
    assert claims_gate.run([]) == 0
    assert seen == [_gate_cmd_serial() + ["-n", "auto"], _gate_cmd_serial()]
    assert "[serial]" in capsys.readouterr().out


def test_a_failed_claim_under_xdist_still_refuses_and_is_not_retried(monkeypatch, capsys):
    monkeypatch.setattr(claims_gate, "xdist_available", lambda: True)
    out = ("FAILED tests/test_overview_claims.py::test_x - AssertionError\n"
           "1 failed, 1451 passed in 120.0s\n")
    seen = _gate_runs(monkeypatch, [_Proc(1, out)])
    assert claims_gate.run([]) == 1
    assert len(seen) == 1
    printed = capsys.readouterr().out
    assert "FAIL (pytest exit 1)" in printed and "test_overview_claims" in printed


def test_the_data_loop_still_reads_the_same_exit_codes():
    """data-run.ps1 is unchanged: 0 publish, 3 warn, anything else discard."""
    data = (ROOT / "scripts" / "data-run.ps1").read_text(encoding="utf-8-sig")
    assert "Invoke-Native 'python' @('scripts/check_published_claims.py')" in data
    assert claims_gate.PYTEST_MISSING == 3


# --------------------------------------------------------------------------- gate 1's wiring


def test_gate_1_runs_the_suite_through_the_helper():
    assert "@('scripts/run_tests.py', 'tests/', 'test_screener.py', '-q')" in CODE


def test_gate_1_falls_back_to_the_exact_serial_command_if_the_helper_is_gone():
    block = CODE[CODE.index("$gate1Args = @('-m'"):CODE.index("GATE 1 tests: PASS")]
    assert "$gate1Args = @('-m', 'pytest', 'tests/', 'test_screener.py', '-q')" in block
    assert "Test-Path (Join-Path $RepoPath 'scripts\\run_tests.py')" in block
    assert "Invoke-Native 'python' $gate1Args" in block


def test_gate_1_still_decides_on_the_exit_code_alone():
    block = CODE[CODE.index("Invoke-Native 'python' $gate1Args"):CODE.index("GATE 2 dry-run")]
    assert "if ($t.ExitCode -eq 0)" in block


@pytest.mark.parametrize("prompt", ["nightly.md", "continuation.md"])
def test_the_prompts_name_the_parallel_command(prompt):
    assert "python scripts/run_tests.py tests/ test_screener.py -q" in (ROOT / "prompts" / prompt).read_text(
        encoding="utf-8")


def test_xdist_is_pinned_in_requirements():
    reqs = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert re.search(r"(?m)^pytest-xdist==\d+\.\d+\.\d+", reqs)


# --------------------------------------------------------------------------- the tripwire, for real

WRITER = '''
from pathlib import Path
ROOT = Path(__file__).resolve().parent


def test_one_writes_a_published_file():
    p = ROOT / "sp500_tickers.json"
    p.write_text(p.read_text(encoding="utf-8") + "polluted", encoding="utf-8")


def test_two_creates_one_that_was_absent():
    (ROOT / "signals.json").write_text("{}", encoding="utf-8")


def test_three_writes_nothing():
    assert True
'''

CLEAN = '''
def test_one():
    assert True


def test_two():
    assert True


def test_three():
    assert True
'''


def _project(tmp_path: Path, body: str) -> Path:
    """A throwaway project carrying the repo's own root conftest.py and its protected files."""
    proj = tmp_path / "proj"
    (proj / "validation").mkdir(parents=True)
    shutil.copy2(ROOT / "conftest.py", proj / "conftest.py")
    (proj / "sp500_tickers.json").write_text('["AAA"]', encoding="utf-8")
    (proj / "validation" / "data_quality_log.csv").write_text("a,b\n", encoding="utf-8")
    (proj / "factor_output.xlsx").write_bytes(b"xlsx")
    (proj / "test_throwaway.py").write_text(body, encoding="utf-8")
    return proj


def _pytest(proj: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *extra],
                          cwd=proj, capture_output=True, text=True, timeout=300)


@pytest.mark.parametrize("mode", [pytest.param(["-n", "2"], marks=needs_xdist, id="xdist"),
                                  pytest.param(["-p", "no:xdist"], id="serial")])
def test_the_tripwire_fails_the_run_and_restores_the_files(tmp_path, mode):
    proj = _project(tmp_path, WRITER)
    r = _pytest(proj, *mode)
    out = r.stdout + r.stderr
    assert r.returncode == 1, out
    assert "A test wrote published artifacts" in out and "sp500_tickers.json" in out, out
    assert (proj / "sp500_tickers.json").read_text(encoding="utf-8") == '["AAA"]', "not restored"
    assert not (proj / "signals.json").exists(), "a created artifact was not removed"


@needs_xdist
def test_a_clean_parallel_run_passes_and_ends_on_pytests_summary(tmp_path):
    """No false alarm under xdist, and the last line is the one both runners log."""
    proj = _project(tmp_path, CLEAN)
    r = _pytest(proj, "-n", "2")
    assert r.returncode == 0, r.stdout + r.stderr
    last = [ln for ln in r.stdout.splitlines() if ln.strip()][-1]
    assert SUMMARY.match(last), last
