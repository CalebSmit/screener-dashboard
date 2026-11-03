"""The data loop warns, in the morning brief, when the code loop's CLI login has lapsed.

2026-11-01..02: four scheduled code sessions died in two seconds each on "OAuth session expired",
and nothing told the owner until he asked. ``scripts/check_cli_login.py`` reads ``claude auth
status`` from the 02:00 data loop - before the day's first code session - and
``write_brief.login_alert`` turns either loop's evidence into one instruction at the top of the
brief. Pinned here: what counts as logged out, that the check can never fail the data loop, that
the brief's alert clears once a newer data run reads OK, and that the instruction names the shim
Windows PowerShell does not block.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import check_cli_login as cl  # noqa: E402
import write_brief as wb  # noqa: E402

OK_STATUS = '{\n  "loggedIn": true,\n  "authMethod": "claude.ai"\n}'
OUT_STATUS = '{\n  "loggedIn": false\n}'


# --------------------------------------------------------------------------- the check
def test_status_is_read_from_the_json_the_cli_prints():
    assert cl.classify(OK_STATUS) == "ok"
    assert cl.classify(OUT_STATUS) == "logged_out"
    assert cl.classify("some banner\n" + OK_STATUS + "\n") == "ok"


def test_anything_unreadable_is_unknown_not_logged_out():
    """A missing or garbled status must not tell the owner to log in when he may not need to."""
    for text in ("", "error: unknown command", "{not json}", '{"other": 1}', '{"loggedIn": "yes"}'):
        assert cl.classify(text) in ("unknown", "logged_out")
    assert cl.classify("") == "unknown"
    assert cl.classify('{"other": 1}') == "unknown"


def test_the_logged_out_line_carries_the_marker_the_brief_reads():
    line = cl.message("logged_out")
    assert line.startswith(cl.LOGGED_OUT)
    assert any(m in line for m in wb.LOGIN_MARKERS)
    assert "claude.cmd auth login" in line
    assert not any(m in cl.message("ok") for m in wb.LOGIN_MARKERS)
    assert not any(m in cl.message("unknown", "x") for m in wb.LOGIN_MARKERS)


def test_the_check_never_fails_and_never_logs_in():
    src = (ROOT / "scripts" / "check_cli_login.py").read_text(encoding="utf-8")
    assert re.findall(r"return (\S+)", src.split("def main")[1]) == ["0", "0", "0"]
    # one process is started, and it is `claude auth status` - never `auth login`
    assert re.findall(r"subprocess\.run\(\[([^\]]*)\]", src) == ['exe, "auth", "status"']
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_cli_login.py")],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0
    assert r.stdout.strip().count("\n") == 0                         # one line, whatever it says


def test_the_data_loop_runs_it_as_a_note_after_the_network_check():
    ps = (ROOT / "scripts" / "data-run.ps1").read_text(encoding="utf-8")
    i = ps.index("scripts/check_cli_login.py")
    assert ps.index('Write-Log "Network is up."') < i < ps.index("Enter-RepoLock -LogDir")
    block = ps[ps.rindex("# --- Is the code loop's CLI logged in?", 0, i):ps.index("# --- Take the repo", i)]
    assert "Stop-Run" not in block and "exit" not in block.lower().replace("exits 0", "")
    assert "'WARN'" in block                                         # the brief collects WARN lines


def test_the_code_runner_names_the_fix_when_a_session_dies_on_the_login():
    ps = (ROOT / "scripts" / "nightly-screener.ps1").read_text(encoding="utf-8")
    i = ps.index("Failed to authenticate|OAuth")
    assert ps.index('Write-Log "SESSION DID NOT RUN: $SessionError"') < i
    assert "claude.cmd auth login" in ps[i:i + 400]
    # the message the brief used to print as two fragments is one line now
    assert "No success marker will be written,\" 'ERROR'" not in ps


# --------------------------------------------------------------------------- the brief
def test_the_brief_leads_with_the_login_when_a_session_died_on_it():
    code = ["[13:04:04] [ERROR] SESSION DID NOT RUN: session reported an error - Failed to "
            "authenticate: OAuth session expired and could not be refreshed"]
    alert = wb.login_alert([], code, data_is_newer=False)
    assert alert and alert[0].startswith("## ")
    assert "claude.cmd auth login" in "\n".join(alert)


def test_the_brief_warns_from_the_data_loop_before_any_session_fails():
    data = ["[02:00:05] [WARN]     " + cl.message("logged_out")]
    assert wb.login_alert(data, ["[06:00:00] [INFO] shipped to main"], data_is_newer=True)


def test_a_newer_ok_reading_clears_an_older_failure():
    code = ["[13:04:04] [ERROR] SESSION DID NOT RUN: Failed to authenticate: OAuth session expired"]
    data = ["[02:00:05] [INFO]     " + cl.message("ok")]
    assert wb.login_alert(data, code, data_is_newer=True) == []
    assert wb.login_alert(data, code, data_is_newer=False)            # the failure is the newer


def test_no_alert_on_an_ordinary_morning():
    assert wb.login_alert(["[02:00:05] [INFO]     " + cl.message("ok")],
                          ["[06:30:00] [INFO] shipped to main"], data_is_newer=False) == []
