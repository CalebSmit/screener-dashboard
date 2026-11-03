"""When does the Claude Code CLI login the scheduled code sessions use run out? Never fails a run.

WHY THIS EXISTS - 2026-11-02. The CLI's login lapsed at 05:43 on 2026-11-01 and every scheduled
code session after it - the 11-01 weekend and midday sessions and the 11-02 06:00 and 13:00 ones -
died two seconds after starting with "OAuth session expired and could not be refreshed". Nothing
said so until the owner asked how the runs went. Signing in is never a session's to do, so the
most this routine can do is say so early.

**The lapse is predictable.** The CLI stores its login in ``~/.claude/.credentials.json`` with a
``refreshTokenExpiresAt``; after the owner signed in again on 2026-11-02 at 18:51 it read
2026-12-02 - about thirty days, and the refreshes the daily sessions make do not extend it (the
sessions ran every day up to the lapse). So this reads that one timestamp and warns a week ahead.

It reads the expiry fields and nothing else - no token is read into a variable, printed or logged -
and it deliberately does **not** run ``claude auth status``: a status call may refresh and rotate
the stored tokens, and this script is also exercised by the test suite, which a scheduled session
runs from *inside* a live CLI session. A check that could log out the session running it would be
the outage it exists to warn about.

The 02:00 data loop runs it after its network check, before any code session of the day. Output is
one line; it always exits 0 - a check that could fail the data loop over the code loop's
credentials would cost the evidence as well. ``write_brief.login_alert`` reads the two markers.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

LOGGED_OUT = "ATTENTION - CLI LOGGED OUT"
EXPIRING = "ATTENTION - CLI LOGIN EXPIRES"
WARN_DAYS = 7

# What the owner should run. In Windows PowerShell the npm `claude.ps1` shim is blocked by the
# default execution policy ("running scripts is disabled on this system" - the owner hit exactly
# that on 2026-11-02), so the instruction names the `.cmd` shim, which is not.
LOGIN_COMMAND = "claude.cmd auth login"


def credentials_path() -> Path:
    return Path(os.environ.get("USERPROFILE") or Path.home()) / ".claude" / ".credentials.json"


def refresh_expiry(path: Path | None = None) -> datetime | None:
    """The login's hard end, from the one field that holds it; None if it cannot be read."""
    path = path or credentials_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ms = data["claudeAiOauth"]["refreshTokenExpiresAt"]
        return datetime.fromtimestamp(float(ms) / 1000)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def classify(expires: datetime | None, now: datetime) -> str:
    """'ok', 'expiring', 'logged_out' or 'unknown'."""
    if expires is None:
        return "unknown"
    if expires <= now:
        return "logged_out"
    if (expires - now).total_seconds() <= WARN_DAYS * 86400:
        return "expiring"
    return "ok"


def message(state: str, expires: datetime | None = None, now: datetime | None = None) -> str:
    when = f"{expires:%Y-%m-%d %H:%M}" if expires else "?"
    days = (expires - now).days if (expires and now) else None
    if state == "ok":
        return f"CLI login: OK - valid until {when} ({days} days); the scheduled code sessions can start."
    if state == "expiring":
        return (f"{EXPIRING} {when} (in {days} days) - after that every scheduled code session fails "
                f"until the owner runs `{LOGIN_COMMAND}` in a terminal and signs in; doing it before "
                f"then loses nothing.")
    if state == "logged_out":
        return (f"{LOGGED_OUT} (the login ran out {when}) - every scheduled code session will fail "
                f"until the owner runs `{LOGIN_COMMAND}` in a terminal and signs in.")
    return "CLI login: could not check (no readable expiry in the CLI's credentials file) - not a failure."


def main() -> int:
    try:
        now = datetime.now()
        expires = refresh_expiry()
        print(message(classify(expires, now), expires, now))
    except Exception as e:  # noqa: BLE001 - a note, never the failure
        print(f"CLI login: could not check ({type(e).__name__}) - not a failure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
