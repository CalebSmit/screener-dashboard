"""Is the Claude Code CLI that the scheduled code sessions use still logged in? Never fails a run.

WHY THIS EXISTS - 2026-11-02. The CLI's login lapsed at 05:43 on 2026-11-01 and every scheduled
code session after it - the 11-01 weekend and midday sessions and the 11-02 06:00 and 13:00 ones -
died two seconds after starting with "OAuth session expired and could not be refreshed". Nothing
said so until the owner asked how the runs went: the brief listed the error among the attention
lines, and the data loop, which does not use the CLI, kept running as if all was well. Four
sessions were lost to something only the owner can fix (signing in is never a session's to do).

The 02:00 data loop runs this after its network check, before any code session of the day, so the
morning brief can say "log in" before the 06:00 session fails rather than after it. It only reads
`claude auth status`; it never logs in, never writes a file, and always exits 0 - a check that
could fail the data loop over the code loop's credentials would cost the evidence as well.

Output is one line. ``ATTENTION - CLI LOGGED OUT`` is the marker ``write_brief.login_alert`` looks
for; ``CLI login: OK`` and ``CLI login: could not check`` are informational.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys

LOGGED_OUT = "ATTENTION - CLI LOGGED OUT"
TIMEOUT_S = 60

# What the owner should run. In Windows PowerShell the npm `claude.ps1` shim is blocked by the
# default execution policy ("running scripts is disabled on this system" - the owner hit exactly
# that on 2026-11-02), so the instruction names the `.cmd` shim, which is not.
LOGIN_COMMAND = "claude.cmd auth login"


def classify(status_text: str) -> str:
    """'ok', 'logged_out' or 'unknown' from the text `claude auth status` printed."""
    try:
        start, end = status_text.index("{"), status_text.rindex("}") + 1
        data = json.loads(status_text[start:end])
    except (ValueError, json.JSONDecodeError):
        return "unknown"
    if not isinstance(data, dict) or "loggedIn" not in data:
        return "unknown"
    return "ok" if data["loggedIn"] is True else "logged_out"


def message(state: str, detail: str = "") -> str:
    if state == "ok":
        return "CLI login: OK - the scheduled code sessions can start."
    if state == "logged_out":
        return (f"{LOGGED_OUT} - every scheduled code session will fail until the owner runs "
                f"`{LOGIN_COMMAND}` in a terminal and signs in.")
    return f"CLI login: could not check ({detail or 'no readable status'}) - not a failure."


def main() -> int:
    exe = shutil.which("claude")
    if not exe:
        print(message("unknown", "`claude` is not on PATH"))
        return 0
    try:
        r = subprocess.run([exe, "auth", "status"], capture_output=True, text=True,
                           timeout=TIMEOUT_S, encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError) as e:
        print(message("unknown", type(e).__name__))
        return 0
    state = classify((r.stdout or "") + (r.stderr or ""))
    print(message(state, f"exit {r.returncode}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
