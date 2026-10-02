"""The data loop checks the claims in what it is about to publish.

WHY THIS FILE EXISTS - 2026-10-02 retrospective.

`prompts/retrospective.md` section 1 says to compare the two runners against
each other, because "where one is stricter than the other about the same
artifact, the weaker one is usually the bug, and it is usually the one that
publishes more often". Ship gate 1 - the full test suite - is the clearest case
left: `nightly-screener.ps1` runs it before it will merge anything, and
`data-run.ps1` ran **no tests at all** while publishing to the live GitHub Pages
site five mornings a week.

The failure is documented, not hypothetical. On 2026-09-28 the 02:00 data run
(`2e08f62`) regenerated the public methodology page, reverted the four
corrections the 2026-09-25 session had shipped, and published them - including a
fetch-failure rate that session had measured at 0 of 9,036 and the page put at
"10-25% of tickers". Measured at that exact commit by this retrospective:

    pytest tests/test_overview_claims.py   ->  12 failed, 2 passed in 0.36s

The evidence to refuse that publish was already committed. The loop doing the
publishing never asked for it.

Three properties these tests exist to keep, all of them deliberate:

* **Narrow, not the whole suite.** The subset is ~15s against the suite's 124s,
  and more importantly a red `main` already stops the code loop from merging. It
  must not also stop the data loop accumulating evidence, so the gate is scoped
  to failures that mean *this publish* is unsafe.
* **A failure discards the run**, exactly like every gate above it in
  `data-run.ps1`. The live site keeps the last good version. Warning and
  publishing anyway would reproduce 2026-09-28 with a log line attached.
* **A missing pytest warns and continues** (exit 3), the `node --check` fallback
  precedent from 2026-09-18. A gate that can only fail closed jams an unattended
  loop, and a jammed loop costs days of evidence.
"""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA_RUNNER = ROOT / "scripts" / "data-run.ps1"
CODE_RUNNER = ROOT / "scripts" / "nightly-screener.ps1"
GATE = ROOT / "scripts" / "check_published_claims.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("check_published_claims", GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gate = _load_gate()


def _source(path):
    return path.read_text(encoding="utf-8-sig")


def _code(path):
    """Script source with comments stripped, so assertions cannot match prose."""
    text = _source(path)
    text = re.sub(r"<#.*?#>", "", text, flags=re.S)
    return re.sub(r"(?m)#.*$", "", text)


class _Proc:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# ---------------------------------------------------------------------------
# The module list. A gate that quietly shrinks is worse than no gate.
# ---------------------------------------------------------------------------


def test_every_listed_module_exists():
    assert gate.missing_modules() == [], (
        "scripts/check_published_claims.py names test modules that are not there"
    )


def test_every_module_carries_a_reason():
    for mod, why in gate.MODULES:
        assert why.strip(), f"{mod} is in the gate with no stated reason"


def test_the_listed_modules_collect():
    """Catches a renamed test class or a module that no longer imports."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *gate.module_paths(),
         "--collect-only", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "test" in proc.stdout


def test_a_missing_module_fails_the_gate_rather_than_passing_it(monkeypatch, capsys):
    monkeypatch.setattr(gate, "MODULES", (("tests/test_does_not_exist.py", "x"),))
    assert gate.run([]) == 1
    assert "missing" in capsys.readouterr().out.lower()


def test_list_prints_every_module(capsys):
    assert gate.run(["--list"]) == 0
    out = capsys.readouterr().out
    for mod, _ in gate.MODULES:
        assert mod in out


# ---------------------------------------------------------------------------
# Membership tripwire. The 2026-09-28 regression was specifically a false claim
# in SCREENER_OVERVIEW.md, so a test asserting on that file must be in the gate.
# ---------------------------------------------------------------------------

# This module names the file in its own docstring; it is about the gate, not a
# claim in the document. The only exemption, and it is stated rather than quiet.
_TRIPWIRE_EXEMPT = {"tests/test_published_claims_gate.py"}


def test_every_test_asserting_on_the_overview_is_in_the_gate():
    listed = set(gate.module_paths())
    offenders = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        rel = f"tests/{path.name}"
        if rel in listed or rel in _TRIPWIRE_EXEMPT:
            continue
        if "SCREENER_OVERVIEW.md" in path.read_text(encoding="utf-8", errors="replace"):
            offenders.append(rel)
    assert offenders == [], (
        "these modules assert on the published methodology page but the data loop "
        "does not run them before publishing it: " + ", ".join(offenders)
    )


def test_the_gate_speaks_for_artifacts_the_data_loop_actually_publishes():
    code = _code(DATA_RUNNER)
    start = code.find("$DataArtifacts")
    assert start != -1
    block = code[start : code.find(")", start)]
    for artifact in gate.PUBLISHED_ARTIFACTS:
        assert artifact in block, (
            f"{artifact} is in the gate's scope but data-run.ps1 does not stage it"
        )


# ---------------------------------------------------------------------------
# Exit-code contract. The runner branches on all three, so each one is pinned.
# ---------------------------------------------------------------------------


def test_passing_tests_report_safe_to_publish(monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc(0, "237 passed in 9.3s\n"))
    assert gate.run([]) == 0
    assert "PASS" in capsys.readouterr().out


def test_failing_tests_refuse_the_publish(monkeypatch, capsys):
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: _Proc(1, "FAILED tests/test_overview_claims.py::x\n1 failed\n"),
    )
    assert gate.run([]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "test_overview_claims" in out


def test_a_missing_pytest_is_not_a_verdict(monkeypatch, capsys):
    """Exit 3, so the runner warns and keeps publishing. See the docstring."""
    def boom(*a, **k):
        raise FileNotFoundError("pytest")
    monkeypatch.setattr(subprocess, "run", boom)
    assert gate.run([]) == gate.PYTEST_MISSING == 3
    assert "NOT checked" in capsys.readouterr().out


@pytest.mark.parametrize("rc", [4, 5], ids=["usage-error", "collected-nothing"])
def test_pytest_not_actually_running_the_tests_is_not_a_pass(monkeypatch, rc, capsys):
    """Exit 5 means zero tests ran. Reading that as 'safe' is how a gate dies."""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc(rc, "no tests ran\n"))
    assert gate.run([]) == 1
    assert "NOT checked" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The wiring in data-run.ps1. These are the assertions that fail against the
# pre-change runner.
# ---------------------------------------------------------------------------


def test_the_data_runner_invokes_the_gate():
    assert "scripts/check_published_claims.py" in _code(DATA_RUNNER), (
        "data-run.ps1 publishes the methodology page without checking its claims"
    )


def test_the_gate_runs_after_regeneration_and_before_the_commit():
    """Ordering is the point: regenerate, check, then publish. Never otherwise."""
    code = _code(DATA_RUNNER)
    regen_at = code.find("'generate_dashboard.py'")
    gate_at = code.find("'scripts/check_published_claims.py'")
    commit_at = code.find("'commit', '-m'")
    push_at = code.find("'push', 'origin', 'main'")
    assert -1 not in (regen_at, gate_at, commit_at, push_at)
    assert regen_at < gate_at < commit_at < push_at


def test_a_failed_claim_discards_the_run():
    code = _code(DATA_RUNNER)
    idx = code.find("'scripts/check_published_claims.py'")
    assert idx != -1
    window = code[idx : idx + 1200]
    assert "Stop-Run" in window, (
        "a contradicted claim must stop the data run, not be logged past"
    )
    assert "'checkout', '--', '.'" in window, (
        "the regenerated artifacts must be discarded so the live site keeps the "
        "last good version"
    )


def test_an_unavailable_pytest_does_not_jam_the_loop():
    code = _code(DATA_RUNNER)
    idx = code.find("'scripts/check_published_claims.py'")
    window = code[idx : idx + 1200]
    assert re.search(r"ExitCode\s*-eq\s*3", window), (
        "data-run.ps1 must treat exit 3 (pytest unavailable) as a warning, not a "
        "failure - a gate that can only fail closed jams an unattended loop"
    )
    warn_at = window.find("ExitCode -eq 3")
    stop_at = window.find("Stop-Run")
    assert warn_at < stop_at, "the exit-3 branch must come before the failure branch"


def test_a_skipped_check_is_reported_not_silent():
    source = _source(DATA_RUNNER)
    assert re.search(r"NOT verified[^\"']*\"\s*'WARN'", source), (
        "silent degradation reads identically to coverage; say it in the log"
    )


# ---------------------------------------------------------------------------
# The other asymmetry found the same way: the code loop checked that the payload
# is the payload, and the daily publish path did not.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("runner", [DATA_RUNNER, CODE_RUNNER], ids=["data", "code"])
def test_both_runners_check_the_payload_is_the_payload(runner):
    """A correctly sized, validly parsing file that is not the payload is a
    blank dashboard. node --check cannot see that."""
    assert r"^window\.SCREENER_DATA=" in _code(runner), (
        f"{runner.name} does not verify the payload's opening assignment"
    )


def test_the_data_runner_refuses_a_payload_with_the_wrong_header():
    code = _code(DATA_RUNNER)
    idx = code.find(r"^window\.SCREENER_DATA=")
    assert idx != -1
    window = code[idx : idx + 500]
    assert "Stop-Run" in window


def test_the_header_check_runs_before_the_commit():
    code = _code(DATA_RUNNER)
    header_at = code.find(r"^window\.SCREENER_DATA=")
    commit_at = code.find("'commit', '-m'")
    assert -1 not in (header_at, commit_at)
    assert header_at < commit_at


# ---------------------------------------------------------------------------
# The documented gate and the implemented gate must agree - the failure shape
# that let the 2026-09-18 hole survive two retrospectives.
# ---------------------------------------------------------------------------


def test_claude_md_records_that_the_data_loop_checks_published_claims():
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "check_published_claims" in text, (
        "CLAUDE.md must describe this gate, or the next session will not know it "
        "exists and may 'simplify' it away"
    )
