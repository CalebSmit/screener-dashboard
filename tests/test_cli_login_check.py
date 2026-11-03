"""The data loop warns, in the morning brief, before and after the code loop's CLI login lapses.

2026-11-01..02: four scheduled code sessions died in two seconds each on "OAuth session expired",
and nothing told the owner until he asked. The login lasts about thirty days and its end is a
stored timestamp, so ``scripts/check_cli_login.py`` reads that one field from the 02:00 data loop
and ``write_brief.login_alert`` turns it into one instruction at the top of the brief - a week
ahead, and again once it has run out. Pinned here: the thresholds, that the check reads no token
and never runs the CLI (a status call can rotate the tokens of the session running the suite),
that it can never fail the data loop, that the brief clears once a newer run reads OK, and that
the instruction names the shim Windows PowerShell does not block.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import check_cli_login as cl  # noqa: E402
import write_brief as wb  # noqa: E402

NOW = datetime(2026, 11, 2, 2, 0)


def _creds(tmp_path: Path, expires: datetime | None, **extra) -> Path:
    p = tmp_path / ".credentials.json"
    body = {"claudeAiOauth": {"accessToken": "SECRET-ACCESS", "refreshToken": "SECRET-REFRESH",
                              "expiresAt": 1, **extra}}
    if expires is not None:
        body["claudeAiOauth"]["refreshTokenExpiresAt"] = int(expires.timestamp() * 1000)
    p.write_text(json.dumps(body), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- the check
def test_the_expiry_is_read_from_the_one_field_that_holds_it(tmp_path):
    end = datetime(2026, 12, 2, 10, 28)
    assert cl.refresh_expiry(_creds(tmp_path, end)) == end
    assert cl.refresh_expiry(_creds(tmp_path, None)) is None              # field missing
    assert cl.refresh_expiry(tmp_path / "absent.json") is None
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert cl.refresh_expiry(tmp_path / "bad.json") is None


def test_the_thresholds():
    assert cl.classify(NOW + timedelta(days=30), NOW) == "ok"
    assert cl.classify(NOW + timedelta(days=cl.WARN_DAYS, seconds=1), NOW) == "ok"
    assert cl.classify(NOW + timedelta(days=cl.WARN_DAYS), NOW) == "expiring"
    assert cl.classify(NOW + timedelta(hours=1), NOW) == "expiring"
    assert cl.classify(NOW, NOW) == "logged_out"
    assert cl.classify(NOW - timedelta(days=1), NOW) == "logged_out"
    assert cl.classify(None, NOW) == "unknown"                             # never "log in" on a guess
    assert cl.WARN_DAYS == 7


def test_the_lines_carry_the_markers_the_brief_reads():
    end = NOW + timedelta(days=3)
    out, soon = cl.message("logged_out", NOW - timedelta(days=1), NOW), cl.message("expiring", end, NOW)
    assert out.startswith(cl.LOGGED_OUT) and any(m in out for m in wb.LOGIN_MARKERS)
    assert soon.startswith(cl.EXPIRING) and "2026-11-05 02:00" in soon and "in 3 days" in soon
    for line in (out, soon):
        assert "claude.cmd auth login" in line
    ok = cl.message("ok", NOW + timedelta(days=29), NOW)
    assert "valid until 2026-12-01 02:00 (29 days)" in ok
    for quiet in (ok, cl.message("unknown")):
        assert not any(m in quiet for m in wb.LOGIN_MARKERS) and "ATTENTION" not in quiet


def test_it_reads_no_token_and_never_runs_the_cli(tmp_path, monkeypatch, capsys):
    """The tokens never reach the output, and no process is started: a `claude auth status` from
    inside the suite could rotate the tokens of the live session running it."""
    src = (ROOT / "scripts" / "check_cli_login.py").read_text(encoding="utf-8")
    assert "subprocess" not in src and "auth status\"" not in src and "accessToken" not in src
    monkeypatch.setattr(cl, "credentials_path", lambda: _creds(tmp_path, NOW + timedelta(days=20)))
    assert cl.main() == 0
    out = capsys.readouterr().out
    assert "SECRET" not in out and out.count("\n") == 1 and out.startswith("CLI login: OK")


def test_it_never_fails_its_caller(tmp_path, monkeypatch, capsys):
    def boom():
        raise RuntimeError("anything at all")
    monkeypatch.setattr(cl, "refresh_expiry", boom)
    assert cl.main() == 0
    assert "could not check" in capsys.readouterr().out
    assert re.findall(r"return (\S+)", (ROOT / "scripts" / "check_cli_login.py")
                      .read_text(encoding="utf-8").split("def main")[1]) == ["0"]


def test_the_data_loop_runs_it_as_a_note_after_the_network_check():
    ps = (ROOT / "scripts" / "data-run.ps1").read_text(encoding="utf-8")
    i = ps.index("scripts/check_cli_login.py")
    assert ps.index('Write-Log "Network is up."') < i < ps.index("Enter-RepoLock -LogDir")
    block = ps[ps.rindex("# --- Is the code loop's CLI logged in?", 0, i):ps.index("# --- Take the repo", i)]
    assert "Stop-Run" not in block and "exit" not in block.lower().replace("exits 0", "")
    assert "'WARN'" in block and "CLI LOGGED OUT|CLI LOGIN EXPIRES" in block


def test_the_code_runner_names_the_fix_when_a_session_dies_on_the_login():
    ps = (ROOT / "scripts" / "nightly-screener.ps1").read_text(encoding="utf-8")
    i = ps.index("Failed to authenticate|OAuth")
    assert ps.index('Write-Log "SESSION DID NOT RUN: $SessionError"') < i
    assert "claude.cmd auth login" in ps[i:i + 400]
    # the message the brief used to print as two fragments is one line now
    assert "No success marker will be written,\" 'ERROR'" not in ps


# --------------------------------------------------------------------------- the brief
def _data(line: str) -> list[str]:
    return ["[02:00:05] [INFO]     " + line]


def test_the_brief_leads_with_the_login_when_a_session_died_on_it():
    code = ["[13:04:04] [ERROR] SESSION DID NOT RUN: session reported an error - Failed to "
            "authenticate: OAuth session expired and could not be refreshed"]
    alert = wb.login_alert([], code, data_is_newer=False)
    assert alert and alert[0].startswith("## ")
    assert "claude.cmd auth login" in "\n".join(alert)


def test_the_brief_warns_from_the_data_loop_before_any_session_fails():
    gone = _data(cl.message("logged_out", NOW - timedelta(hours=2), NOW))
    assert wb.login_alert(gone, ["[06:00:00] [INFO] shipped to main"], data_is_newer=True)


def test_the_brief_warns_a_week_ahead_and_names_the_date():
    soon = _data(cl.message("expiring", NOW + timedelta(days=5), NOW))
    alert = wb.login_alert(soon, ["[06:00:00] [INFO] shipped to main"], data_is_newer=True)
    assert alert and "2026-11-07 02:00" in alert[0] and "claude.cmd auth login" in "\n".join(alert)
    assert wb.login_row(soon) == "**runs out 2026-11-07 02:00 (in 5 days)**"


def test_a_newer_ok_reading_clears_an_older_failure():
    code = ["[13:04:04] [ERROR] SESSION DID NOT RUN: Failed to authenticate: OAuth session expired"]
    data = _data(cl.message("ok", NOW + timedelta(days=29), NOW))
    assert wb.login_alert(data, code, data_is_newer=True) == []
    assert wb.login_alert(data, code, data_is_newer=False)            # the failure is the newer


def test_no_alert_on_an_ordinary_morning_and_the_glance_row_says_how_long():
    data = _data(cl.message("ok", NOW + timedelta(days=29), NOW))
    assert wb.login_alert(data, ["[06:30:00] [INFO] shipped to main"], data_is_newer=False) == []
    assert wb.login_row(data) == "2026-12-01 02:00 (29 days)"
    assert wb.login_row(_data(cl.message("unknown"))) == ""


def test_the_script_runs_standalone():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_cli_login.py")],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0
    assert r.stdout.startswith(("CLI login", "ATTENTION - CLI")), r.stdout
    assert r.stdout.strip().count("\n") == 0
