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
from datetime import datetime
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
    n = len(detail) or 1
    return {
        "timestamp": d.get("kpis", {}).get("run_timestamp", "?"),
        "scored": d.get("kpis", {}).get("stocks_scored", "?"),
        "priced": f"{sum(1 for v in detail.values() if v.get('price'))}/{len(detail)}",
        "targeted": f"{sum(1 for v in detail.values() if v.get('pt_mean'))}/{len(detail)}",
        "top5": [h["ticker"] for h in (d.get("portfolio", {}).get("holdings") or [])[:5]],
    }


def ic_observations() -> str:
    p = ROOT / "improvement" / "live_ic_history.csv"
    if not p.exists():
        return "unknown"
    try:
        rows = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        return "unknown"

    n = max(0, len(rows) - 1)
    if n == 0:
        return "0 of 8 needed - the series is empty"

    # The count alone is not enough. It read "3 of 8 needed" on every brief from
    # February to 2026-08-21 while the data loop ran successfully every weekday,
    # because the loop never calls compute_live_ic(). A number that never moves
    # looks like slow progress; the date is what shows it is no progress at all.
    dates = sorted(r.split(",", 1)[0] for r in rows[1:] if "," in r)
    newest = dates[-1] if dates else "unknown"
    age = ""
    try:
        days = (datetime.now() - datetime.strptime(newest, "%Y-%m-%d")).days
        if days > 14:
            age = f" - STALE, nothing new in {days} days"
    except ValueError:
        pass
    return f"{n} of 8 needed, newest {newest}{age}"


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

    since = git("log", "--oneline", "--since=36 hours ago", "--no-merges")
    commits = [c for c in since.splitlines() if c.strip()]

    out: list[str] = []
    a = out.append

    a(f"# Morning Brief - {now:%A %d %B %Y, %H:%M}")
    a("")
    a("Written automatically after each run. Newest state only - the full")
    a("history is in `NIGHTLY_LOG.md`.")
    a("")

    a("## At a glance")
    a("")
    a("| | |")
    a("|---|---|")
    a(f"| Data run (2 AM) | **{data_state}** |")
    a(f"| Code session (6 AM) | **{code_state}** |")
    if facts:
        a(f"| Dashboard data from | {facts.get('timestamp', '?')} |")
        a(f"| Stocks scored | {facts.get('scored', '?')} |")
        a(f"| With a price | {facts.get('priced', '?')} |")
        a(f"| With an analyst target | {facts.get('targeted', '?')} |")
        if facts.get("top5"):
            a(f"| Top 5 | {', '.join(facts['top5'])} |")
    a(f"| Evidence for weight changes | {ic_observations()} |")
    a("")

    if data_notes or code_notes:
        a("## Things that needed attention")
        a("")
        for n in dict.fromkeys(data_notes + code_notes):
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
