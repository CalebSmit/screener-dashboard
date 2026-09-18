"""`dashboard_data.js` is parsed before it is published, on both publish paths.

WHY THIS FILE EXISTS - 2026-09-18 retrospective.

`CLAUDE.md`'s ship gate 3 and `prompts/nightly.md` section 4 both describe the
gate as "`dashboard_data.js` parses". Neither runner parsed it. The code loop
regex-matched the first line and checked a 100 KB size floor; the data loop -
**which is what actually publishes the payload to GitHub Pages, five days a
week** - checked only the size floor.

Measured before writing any of this, against the live 5 MB payload:

| payload              | node --check | passed the old checks |
|----------------------|--------------|-----------------------|
| full                 | 0            | yes                   |
| truncated to 50%     | **1**        | **yes**               |
| header only          | 1            | no                    |

So a 2.5 MB half-written payload published a blank dashboard to the public site
with every check green. The 2026-08-21 retrospective identified this hole and
deliberately declined to fix it, because neither PowerShell nor node could be
executed in that session and a gate that can only fail closed jams the loop.
Both run now, so it is verified rather than deferred.

The parse is *additional*. Where node is missing, both runners fall back to the
checks they already had and log a warning - strictly stricter than before, and
still unable to jam an unattended loop. Section 4 of `prompts/retrospective.md`
permits making a gate stricter; it forbids the reverse.

One thing these tests deliberately do NOT assert: that the payload ends in a
particular terminator. Dropping the final `;` leaves valid JavaScript with
complete data, and `node --check` accepts it - correctly. A check that fires on
a harmless condition is the failure shape fixed on 2026-09-01 (the permanent
bank-metrics alarm) and avoided again on 2026-09-17 (arming input churn at 2
metrics, not 1). It trains a reader to ignore the gate.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA_RUNNER = ROOT / "scripts" / "data-run.ps1"
CODE_RUNNER = ROOT / "scripts" / "nightly-screener.ps1"
PAYLOAD = ROOT / "dashboard_data.js"

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not available")


def _source(path):
    return path.read_text(encoding="utf-8-sig")


def _code(path):
    """Script source with comments stripped, so assertions cannot match prose."""
    text = _source(path)
    text = re.sub(r"<#.*?#>", "", text, flags=re.S)
    return re.sub(r"(?m)#.*$", "", text)


# ---------------------------------------------------------------------------
# The behaviour being protected: node really does refuse what the old checks
# passed. These run the parser rather than trusting the table in the docstring.
# ---------------------------------------------------------------------------


def _old_checks_pass(data: bytes, min_bytes: int = 100_000) -> bool:
    """The checks both runners had before 2026-09-18, reimplemented exactly."""
    if len(data) < min_bytes:
        return False
    first_line = data.split(b"\n", 1)[0].replace(b" ", b"")
    return first_line.startswith(b"window.SCREENER_DATA=")


def _node_check(path) -> int:
    return subprocess.run(
        [NODE, "--check", str(path)], capture_output=True, text=True, timeout=120
    ).returncode


@needs_node
def test_the_live_payload_parses():
    """If this fails the published site is broken right now, not the gate."""
    assert PAYLOAD.exists(), "no dashboard_data.js to check"
    assert _node_check(PAYLOAD) == 0


@needs_node
def test_a_half_truncated_payload_passes_the_old_checks_and_fails_the_parse(tmp_path):
    """The exact hole. This is the test the gate exists for."""
    src = PAYLOAD.read_bytes()
    truncated = src[: len(src) // 2]
    victim = tmp_path / "dashboard_data.js"
    victim.write_bytes(truncated)

    # It is big, and it opens with the right assignment...
    assert len(truncated) > 100_000
    assert _old_checks_pass(truncated) is True
    # ...and it is not JavaScript.
    assert _node_check(victim) != 0


@needs_node
def test_a_payload_missing_its_tail_object_fails_the_parse(tmp_path):
    """A different truncation point, so the test is not pinned to one offset."""
    src = PAYLOAD.read_bytes()
    victim = tmp_path / "dashboard_data.js"
    victim.write_bytes(src[: int(len(src) * 0.9)])
    assert _old_checks_pass(src[: int(len(src) * 0.9)]) is True
    assert _node_check(victim) != 0


@needs_node
def test_dropping_the_final_semicolon_is_not_treated_as_damage(tmp_path):
    """Documented non-goal: valid JS stays valid. See the module docstring."""
    src = PAYLOAD.read_bytes()
    victim = tmp_path / "dashboard_data.js"
    victim.write_bytes(src.rstrip()[:-1])
    assert _node_check(victim) == 0


# ---------------------------------------------------------------------------
# Both publish paths carry the parse. These are the assertions that fail
# against the pre-change scripts.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("runner", [DATA_RUNNER, CODE_RUNNER], ids=["data", "code"])
def test_runner_parses_the_payload_before_publishing(runner):
    code = _code(runner)
    assert "node" in code, f"{runner.name} does not invoke node at all"
    assert re.search(r"'node'\s*@\(\s*'--check'", code), (
        f"{runner.name} must run `node --check` on the payload before publishing"
    )


@pytest.mark.parametrize("runner", [DATA_RUNNER, CODE_RUNNER], ids=["data", "code"])
def test_the_parse_is_skipped_rather_than_fatal_when_node_is_absent(runner):
    """An unattended loop must not be jammed by a missing interpreter."""
    code = _code(runner)
    assert re.search(r"Get-Command\s+node\s+-ErrorAction\s+SilentlyContinue", code), (
        f"{runner.name} must probe for node instead of assuming it"
    )


@pytest.mark.parametrize("runner", [DATA_RUNNER, CODE_RUNNER], ids=["data", "code"])
def test_a_missing_node_is_reported_not_silent(runner):
    """Silent degradation reads identically to coverage. Say it in the log."""
    source = _source(runner)
    warn = re.search(r"node not found[^\"']*\"\s*'WARN'", source)
    assert warn, f"{runner.name} must log a WARN when it skips the parse"


def test_the_data_runner_refuses_to_publish_an_unparseable_payload():
    """The data loop's failure mode is refusing to publish, not warning."""
    code = _code(DATA_RUNNER)
    idx = code.find("'--check'")
    assert idx != -1
    window = code[idx : idx + 600]
    assert "Stop-Run" in window, (
        "a payload that does not parse must stop the data run, not be logged past"
    )


def test_the_code_runner_fails_gate_three_on_an_unparseable_payload():
    """The code loop's failure mode is failing the gate, so the merge is refused."""
    code = _code(CODE_RUNNER)
    idx = code.find("'--check'")
    assert idx != -1
    window = code[idx : idx + 600]
    assert "$g3 = $false" in window, (
        "a payload that does not parse must clear $g3 so gate 3 records a failure"
    )


def test_the_parse_runs_before_the_data_runner_commits():
    """Ordering is the whole point: check, then publish. Never the reverse."""
    code = _code(DATA_RUNNER)
    parse_at = code.find("'--check'")
    commit_at = code.find("'commit', '-m'")
    push_at = code.find("'push', 'origin', 'main'")
    assert -1 not in (parse_at, commit_at, push_at)
    assert parse_at < commit_at < push_at


def test_the_parse_runs_before_the_code_runner_merges():
    code = _code(CODE_RUNNER)
    parse_at = code.find("'--check'")
    merge_at = code.find("'merge', '--no-ff'")
    assert -1 not in (parse_at, merge_at)
    assert parse_at < merge_at


# ---------------------------------------------------------------------------
# The documented gate and the implemented gate must agree. They did not for
# four weeks, which is how the hole survived two retrospectives that both
# read the runners.
# ---------------------------------------------------------------------------


def test_claude_md_still_claims_the_payload_parses():
    """If someone softens the promise, this fails and points at the runner."""
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "dashboard_data.js` parses" in text


# ---------------------------------------------------------------------------
# Housekeeping: the data loop keeps its own object store packed.
# ---------------------------------------------------------------------------


def test_the_data_runner_repacks_and_never_dies_doing_it():
    code = _code(DATA_RUNNER)
    assert "'gc'" in code and "'--auto'" in code, "the data loop should repack"
    idx = code.find("'gc'")
    # Only what handles the gc result - a Stop-Run before it belongs to the push.
    window = code[idx : idx + 300]
    assert "Stop-Run" not in window, "repacking must never end the run"
    assert "'WARN'" in window, "a failed repack should be logged, not swallowed"
    push_at = code.find("'push', 'origin', 'main'")
    assert push_at < idx, "repack after publishing, never before"
