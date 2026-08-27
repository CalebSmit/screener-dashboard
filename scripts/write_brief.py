#!/usr/bin/env python3
"""Write MORNING_BRIEF.md - a plain-language summary of the last runs.

The owner asked to be told what happened each night without having to read
logs. This assembles that from what the loops already produce: the run logs,
the improvement-engine evidence, git history, and the session's own
NIGHTLY_LOG.md entry.

It is committed, so it is also readable on GitHub, and the push that carries it
triggers a GitHub notification email if the repo is watched.

Usage:
    python scripts/write_brief.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
BRIEF = ROOT / "MORNING_BRIEF.md"


def git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def newest(pattern: str) -> Path | None:
    files = sorted(LOGS.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def tail(path: Path | None, n: int = 400) -> list[str]:
    if not path or not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


def run_age(path: Path | None) -> tuple[str, int | None]:
    """How long ago this loop last ran, and whether that is alarming.

    The brief reads the *newest* log and reports its outcome. If a loop stops
    firing entirely there is no new log, so the brief keeps cheerfully
    reporting the last successful run - which is exactly how 2026-08-17..20
    passed unnoticed, and how the machine sat at a login screen for six days
    with the status page saying everything was fine. Absence has to be louder
    than the last success.
    """
    if not path or not path.exists():
        return "never", None
    days = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).days
    if days == 0:
        return "today", 0
    if days == 1:
        return "yesterday", 1
    return f"{days} days ago", days


def outcome(lines: list[str]) -> tuple[str, list[str]]:
    """Classify a run log and pull out the lines worth showing."""
    if not lines:
        return "did not run", []
    text = "\n".join(lines)
    notable = [
        ln.split("] ", 2)[-1].strip()
        for ln in lines
        if "[ERROR]" in ln or "[WARN]" in ln
    ]
    if "Data loop complete" in text or "shipped to main" in text:
        return "completed", notable
    if "Run complete (no changes)" in text:
        return "completed, nothing to change", notable
    if "discarded" in text.lower() or "SHIP GATES FAILED" in text:
        return "stopped deliberately", notable
    return "failed", notable


# The payload keys this brief reads. Checked at runtime rather than assumed,
# because every lookup below uses `.get()` with a default: when the dashboard
# dropped the `portfolio` key on 2026-08-26 the Top 5 row simply stopped being
# emitted, with no error in the run log and nothing missing that anyone would
# notice from the page itself. A silently-shortened status page is the same
# failure shape as a silently-stalled loop.
PAYLOAD_KEYS = ("kpis", "stock_detail", "table_data")


def dashboard_facts() -> dict:
    path = ROOT / "dashboard_data.js"
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
        d = json.loads(text[text.index("{"):].rstrip().rstrip(";"))
    except (OSError, ValueError):
        return {}
    detail = d.get("stock_detail") or {}
    return {
        "timestamp": d.get("kpis", {}).get("run_timestamp", "?"),
        "scored": d.get("kpis", {}).get("stocks_scored", "?"),
        "priced": f"{sum(1 for v in detail.values() if v.get('price'))}/{len(detail)}",
        "targeted": f"{sum(1 for v in detail.values() if v.get('pt_mean'))}/{len(detail)}",
        "top5": top5_tickers(d.get("table_data") or []),
        "missing_keys": [k for k in PAYLOAD_KEYS if k not in d],
    }


def top5_tickers(table: list[dict]) -> list[str]:
    """The dashboard's Top 5, computed the way the dashboard computes it.

    Until 2026-08-26 this read `portfolio.holdings`. That surface was removed
    that evening and its payload key went with it, so `d.get("portfolio", {})`
    quietly returned an empty dict, `top5` became `[]`, and the row was
    dropped. The brief has shipped without a Top 5 line ever since - the single
    most decision-relevant row on the owner's daily page, gone with no error
    anywhere, because every step of that path degrades silently.

    Mirrors `renderTop5()` in generate_dashboard.py: trap-flagged names are
    excluded, then rank order. Keep the two in step - if they disagree, the
    brief is telling the owner something the dashboard does not show.
    """
    ranked = [
        r for r in table
        if not r.get("Value_Trap_Flag")
        and not r.get("Growth_Trap_Flag")
        and r.get("Rank") is not None
        and r.get("Ticker")
    ]
    ranked.sort(key=lambda r: r["Rank"])
    return [r["Ticker"] for r in ranked[:5]]


OPTIMIZATION_HORIZON = "1m"
HORIZON_DAYS = {"1w": 7, "1m": 30, "3m": 90}


def _effective_observations(dates: list[str], horizon: str) -> int:
    """Count non-overlapping return windows among these run dates.

    Deliberately duplicated from improvement_engine._effective_observations()
    rather than imported: this script is stdlib-only on purpose, because it is
    what reports that everything else is broken. Keep the two in step.
    """
    span = HORIZON_DAYS.get(horizon, 30)
    parsed = []
    for d in dates:
        try:
            parsed.append(datetime.strptime(d, "%Y-%m-%d"))
        except ValueError:
            continue

    n_eff, window_end = 0, None
    for ts in sorted(set(parsed)):
        if window_end is None or ts >= window_end:
            n_eff += 1
            window_end = ts + timedelta(days=span)
    return n_eff


def ic_observations() -> str:
    p = ROOT / "improvement" / "live_ic_history.csv"
    if not p.exists():
        return "unknown"
    try:
        rows = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        return "unknown"

    if len(rows) < 2:
        return "0 of 8 needed - the series is empty"

    # Report the count the engine's gate actually reads: EFFECTIVE
    # (non-overlapping) observations at the optimization horizon. The raw row
    # count is not the same thing and is much larger - after the 2026-08-24
    # repair the file holds 23 rows but only 2 effective 1-month observations,
    # so "23 of 8 needed" would read as a cleared gate that is nowhere near
    # cleared. Overlapping 30-day windows are the same measurement repeated.
    parsed = []
    for r in rows[1:]:
        parts = r.split(",")
        if len(parts) >= 2:
            parsed.append((parts[0].strip(), parts[1].strip()))
    if not parsed:
        return "unknown"

    at_horizon = [d for d, h in parsed if h == OPTIMIZATION_HORIZON]
    n_eff = _effective_observations(at_horizon, OPTIMIZATION_HORIZON)

    # The date is what distinguishes slow progress from no progress. It read
    # "3 of 8 needed" on every brief from February to 2026-08-21 while the data
    # loop ran successfully every weekday, because nothing called
    # compute_live_ic(). A number that never moves looks like slow progress.
    newest = max(d for d, _ in parsed)
    age = ""
    try:
        # A 1-week observation only matures 7 days after its run date, and
        # weekends push that out, so the newest date always trails today by
        # ~7-11 days even when everything is working. 21 days is clear of that
        # and still catches a real stall inside three weeks.
        days = (datetime.now() - datetime.strptime(newest, "%Y-%m-%d")).days
        if days > 21:
            age = f" - STALE, nothing new in {days} days"
    except ValueError:
        pass

    return (
        f"{n_eff} of 8 needed at the {OPTIMIZATION_HORIZON} horizon "
        f"({len(at_horizon)} rows, but overlapping windows are not independent; "
        f"{len(parsed)} rows across all horizons), newest {newest}{age}"
    )


def last_log_entry() -> str:
    """The most recent dated section of NIGHTLY_LOG.md, lightly trimmed."""
    p = ROOT / "NIGHTLY_LOG.md"
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="replace")
    entries = re.split(r"^## ", text, flags=re.M)
    if len(entries) < 2:
        return ""
    body = entries[-1].strip()
    lines = body.splitlines()
    return "\n".join(lines[:45]) + ("\n..." if len(lines) > 45 else "")


def main() -> int:
    now = datetime.now()
    data_log = newest("datarun-*.log")
    code_log = newest("nightly-*.log")
    data_state, data_notes = outcome(tail(data_log))
    code_state, code_notes = outcome(tail(code_log))
    facts = dashboard_facts()
    data_age_pre, data_days_pre = run_age(data_log)
    code_age_pre, code_days_pre = run_age(code_log)

    since = git("log", "--oneline", "--since=36 hours ago", "--no-merges")
    commits = [c for c in since.splitlines() if c.strip()]

    out: list[str] = []
    a = out.append

    a(f"# Morning Brief - {now:%A %d %B %Y, %H:%M}")
    a("")
    a("Written automatically after each run. Newest state only - the full")
    a("history is in `NIGHTLY_LOG.md`.")
    a("")

    stalled = []
    for label, age, days in (("Data run", data_age_pre, data_days_pre),
                             ("Code session", code_age_pre, code_days_pre)):
        if days is None or days >= 2:
            stalled.append(f"**{label}** has not run since {age}")
    if stalled:
        a("## THE ROUTINE IS NOT RUNNING")
        a("")
        for s in stalled:
            a(f"- {s}")
        a("")
        a("Nothing below is current. A loop that stops firing writes no log, so")
        a("the rest of this page describes the last run that *did* happen, not")
        a("today. Most likely cause: the PC rebooted and nobody logged back in -")
        a("the tasks only run while a user is signed in. See NIGHTLY_LOG.md")
        a("2026-08-20 and `scripts/register-tasks.ps1`.")
        a("")

    a("## At a glance")
    a("")
    a("| | |")
    a("|---|---|")
    data_age, data_days = run_age(data_log)
    code_age, code_days = run_age(code_log)
    a(f"| Data run (2 AM) | **{data_state}** - last ran {data_age} |")
    a(f"| Code session (6 AM) | **{code_state}** - last ran {code_age} |")
    if facts:
        a(f"| Dashboard data from | {facts.get('timestamp', '?')} |")
        a(f"| Stocks scored | {facts.get('scored', '?')} |")
        a(f"| With a price | {facts.get('priced', '?')} |")
        a(f"| With an analyst target | {facts.get('targeted', '?')} |")
        if facts.get("top5"):
            a(f"| Top 5 | {', '.join(facts['top5'])} |")
    a(f"| Evidence for weight changes | {ic_observations()} |")
    a("")

    payload_notes = []
    if facts.get("missing_keys"):
        payload_notes.append(
            "`dashboard_data.js` no longer has: "
            + ", ".join(f"`{k}`" for k in facts["missing_keys"])
            + ". Rows above that depend on it are missing from this brief. A "
            "payload key was probably renamed or removed - see "
            "`PAYLOAD_KEYS` in `scripts/write_brief.py`."
        )

    if data_notes or code_notes or payload_notes:
        a("## Things that needed attention")
        a("")
        for n in dict.fromkeys(payload_notes + data_notes + code_notes):
            a(f"- {n}")
        a("")

    a("## What changed in the repo")
    a("")
    if commits:
        for c in commits[:15]:
            a(f"- `{c}`")
    else:
        a("- Nothing committed in the last 36 hours.")
    a("")

    entry = last_log_entry()
    if entry:
        a("## The session's own account")
        a("")
        a("> " + entry.replace("\n", "\n> "))
        a("")

    a("---")
    a("")
    a("If a run says **stopped deliberately**, that is the safety gates working:")
    a("the live dashboard was left untouched rather than published with bad data.")
    a("`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.")
    a("")

    BRIEF.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Wrote {BRIEF.name} ({BRIEF.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
