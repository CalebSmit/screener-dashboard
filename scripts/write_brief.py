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


# A code-session log that records no session: a dry run, or an extra slot standing down because
# the owner was working (scripts/nightly-screener.ps1 -Slot). Since 2026-10-14 there can be several
# of these a day, and the newest log is no longer necessarily the newest *session* - a 08:15 dry
# run published a brief calling the day's code session "failed".
NOT_A_SESSION = ("=== Dry run complete ===", "Standing down")


def newest_session(pattern: str = "nightly-*.log") -> Path | None:
    """The newest log that records a session that actually ran."""
    for path in sorted(LOGS.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True):
        text = "\n".join(tail(path, 60))
        if not any(marker in text for marker in NOT_A_SESSION):
            return path
    return None


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
        if "[ERROR]" in ln or "[WARN]" in ln or "ATTENTION - " in ln
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


def context_coverage() -> str:
    """How much of the context layer landed: usable option quotes (and the session they are
    from), SEC-sourced insider data, and the earnings-variability candidate. One line, so a
    session can see whether the 20:00 quote task and the SEC caches are doing their job."""
    try:
        t = (ROOT / "dashboard_context.js").read_text(encoding="utf-8")
        c = json.loads(t[t.index("{"):].rstrip().rstrip(";"))
        ctx = c.get("ctx") or {}
        n = len(ctx)
        ok = sum(1 for v in ctx.values() if v.get("os") in ("ok", "partial"))
        closed = sum(1 for v in ctx.values() if v.get("os") == "quotes-closed")
        sec = sum(1 for v in ctx.values() if (v.get("ins") or {}).get("src") == "sec")
        d = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
        dd = json.loads(d[d.index("{"):].rstrip().rstrip(";"))
        ev = sum(1 for v in (dd.get("stock_detail") or {}).values()
                 if (v.get("raw") or {}).get("earnings_variability") is not None)
        return (f"options usable {ok}/{n}" + (f" ({closed} quotes-closed)" if closed else "")
                + f" - insider from SEC {sec}/{n} - earnings variability {ev}")
    except Exception:  # noqa: BLE001 - the brief must still publish
        return "no context payload"


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


STALE_AFTER_DAYS = 40     # CLAUDE.md rule 8: the newest `1m` run date within 40 days of today


def ic_observations(today: datetime | None = None) -> str:
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
    #
    # Read at the optimization horizon and nowhere else (CLAUDE.md rule 8,
    # corrected 2026-09-18). The newest date across all horizons rises every
    # weekday as the 1-week horizon gains a row, so it cannot stand still: on
    # 2026-09-18 the `1m` date had not moved for six sessions while the whole
    # file's kept rising. This line still read the whole file, against a 21-day
    # bound, until 2026-10-29 - "newest 2026-10-21" beside a `1m` date of
    # 2026-09-28. The bound is rule 8's: 40 days (30-day horizon plus a weekend
    # is 30-33 in steady state; 40 tolerates a week of missed runs).
    if not at_horizon:
        return (f"0 of 8 needed at the {OPTIMIZATION_HORIZON} horizon - no "
                f"{OPTIMIZATION_HORIZON} rows yet ({len(parsed)} rows across all horizons)")
    newest = max(at_horizon)
    age = ""
    try:
        days = ((today or datetime.now()) - datetime.strptime(newest, "%Y-%m-%d")).days
        age = f" ({days} days ago" + (
            f" - STALE: over the {STALE_AFTER_DAYS}-day bound, find out why)" if days > STALE_AFTER_DAYS
            else f", bound {STALE_AFTER_DAYS})")
    except ValueError:
        pass

    return (
        f"{n_eff} of 8 needed at the {OPTIMIZATION_HORIZON} horizon "
        f"({len(at_horizon)} rows, but overlapping windows are not independent; "
        f"{len(parsed)} rows across all horizons), newest {OPTIMIZATION_HORIZON} "
        f"run date {newest}{age}"
    )


def run_stamp(ts) -> str:
    """The run's timestamp to the minute: "2026-10-29 02:00", not "2026-10-29T02:00:05.052320".
    Anything that does not parse is shown as it came, so the row never hides a strange value."""
    if not ts:
        return "?"
    try:
        return datetime.fromisoformat(str(ts)).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return str(ts)


def context_record() -> str:
    """One line on the record the context signals are building (context_eval.py)."""
    p = ROOT / "data" / "context_eval.json"
    try:
        sys.path.insert(0, str(ROOT))
        import context_eval
        line = context_eval.summary_line(json.loads(p.read_text(encoding="utf-8")))
        # the line is written for the data-run log, where it names itself; the brief's row label
        # already says "Context signals" (it read "Context signals | Context signals: ...")
        return re.sub(r"^Context signals:\s*", "", line)
    except Exception:  # noqa: BLE001 - the brief must still publish
        return "no context evaluation yet"


# A research note that proposes a change and leaves it for a second session to decide carries
# one line:  **Proposal status: OPEN** - <what is proposed>   The session that decides rewrites
# it (DECIDED <date> - built / refused). Until then the brief shows it with its age, because a
# proposal nobody is reminded of is a decision made by default.
PROPOSAL_OPEN = re.compile(r"^\*\*Proposal status: OPEN\*\*\s*-\s*(.+?)\s*$", re.M)


def open_proposals(research_dir: Path | None = None, today: datetime | None = None) -> list[str]:
    """One line per research note with an open proposal, oldest first. Never raises: the brief
    is written from a `finally` and must not be the thing that fails."""
    research_dir = research_dir or ROOT / "research"
    today = today or datetime.now()
    out = []
    try:
        notes = sorted(research_dir.glob("20??-??-??-*.md"))
    except OSError:
        return out
    for note in notes:
        try:
            found = PROPOSAL_OPEN.search(note.read_text(encoding="utf-8", errors="replace"))
            opened = datetime.strptime(note.name[:10], "%Y-%m-%d")
        except (OSError, ValueError):
            continue
        if not found:
            continue
        days = max(0, (today.date() - opened.date()).days)
        age = "today" if days == 0 else f"{days} day{'' if days == 1 else 's'}"
        out.append(f"**{found.group(1)}** - open {age} - `research/{note.name}`")
    return out


FILING_LINE = "inputs against SEC filings: "
TOP_LINE = "top 25, worth a look: "
PHASE_LINE = "PHASE TIMING (s): "


def top_of_ranking(lines: list[str]) -> str:
    """The health check's top-25 line (``check_top_of_ranking``), as printed, or empty."""
    line = next((ln for ln in reversed(lines) if TOP_LINE in ln), "")
    return line.split(TOP_LINE, 1)[1].strip() if line else ""


def filing_check(lines: list[str]) -> str:
    """The data run's agreement with the companies' own SEC filings, short enough for one row.

    The health check prints one line per run (``check_inputs_against_filings``, 2026-10-14); this
    keeps the four figures that sit near 100% when nothing is wrong. Empty when the run printed
    none (an older run, or the cache missing) - the row is then left out, never guessed."""
    import re
    line = next((ln for ln in reversed(lines) if FILING_LINE in ln), "")
    if not line:
        return ""
    body = line.split(FILING_LINE, 1)[1]
    keep = []
    for label, short in (("total assets", "total assets"), ("operating cash flow", "operating cash flow"),
                         ("net income", "net income"), ("market value / price vs filed shares", "market value")):
        m = re.search(re.escape(label) + r" (\d+)/(\d+) within (\S+)", body)
        if m:
            keep.append(f"{short} {m.group(1)}/{m.group(2)} within {m.group(3)}")
    # The stocks scored a quarter behind a filing over a month old (2026-10-29): 15 that morning,
    # Amazon and Bank of America among them, printed on the health line and dropped here - the one
    # figure on the line about scores that are out of date rather than inputs that disagree.
    # research/2026-10-29-ttm-fills-the-empty-quarter.md.
    late = re.search(r"(\d+) stocks scored a period behind a filing over 30 days old", body)
    if keep and late:
        keep.append(f"{late.group(1)} scored a quarter behind a filing over 30 days old")
    age = re.search(r"SEC cache (\d+) days old", body)
    if keep and age:
        keep.append(f"filings cached {age.group(1)} days ago")
    attention = any("ATTENTION - " in ln for ln in lines)
    return "; ".join(keep) + (" - **see the ATTENTION note below**" if attention and keep else "")


def run_timing(lines: list[str]) -> str:
    """How long the data run took and whether Yahoo throttled it, for one row.

    Added 2026-10-27. `research/2026-10-26-data-run-timing.md` section 5 item 6 proposed raising
    the fetch's worker count only after the rate-limited count reads 0 "on every 02:00 run for a
    month" - and the first 02:00 run that could report it reported **39**. A number nobody is shown
    is a number nobody watches, which is the failure CLAUDE.md rule 8 exists for, so the count goes
    where a session reads it first. Provenance about the machine, not about a stock: this module
    neither scores nor publishes a ranking, which is why `tests/test_phase_timing.py`'s
    `_MUST_NOT_READ` does not list it.

    Empty when the run printed no timing line (a run from before phase timing) - the row is then
    left out, never guessed.
    """
    line = next((ln for ln in reversed(lines) if PHASE_LINE in ln), "")
    if not line:
        return ""
    body = line.split(PHASE_LINE, 1)[1].strip()
    phases = {}
    for part in body.split("|"):
        bits = part.strip().rsplit(" ", 1)
        if len(bits) == 2 and re.fullmatch(r"\d+(\.\d+)?", bits[1]):
            phases[bits[0].strip()] = float(bits[1])
    total = phases.pop("total", None)
    if total is None:
        return ""
    out = [f"**{total / 60:.0f} min** total"]
    slowest = sorted(phases.items(), key=lambda kv: -kv[1])[:3]
    if slowest:
        out.append("slowest " + ", ".join(f"`{k}` {v:.0f} s" for k, v in slowest))

    limited = _rate_limited()
    if limited is not None:
        out.append(f"**{limited} tickers rate-limited**" if limited
                   else "0 tickers rate-limited")
    return "; ".join(out)


def _rate_limited() -> int | None:
    """The newest run's rate-limited ticker count, or None if it did not record one."""
    try:
        sys.path.insert(0, str(ROOT))
        import run_context
        d = run_context.latest_run_dir()
        meta = json.loads((Path(d) / "meta.json").read_text(encoding="utf-8"))
        return int(meta["phase_detail"]["fetch.tickers"]["rate_limited"])
    except Exception:  # noqa: BLE001 - the brief never fails over a missing figure
        return None


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
    code_log = newest_session("nightly-*.log")
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
    a(f"| Code session (latest) | **{code_state}** - last ran {code_age} |")
    if facts:
        a(f"| Dashboard data from | {run_stamp(facts.get('timestamp'))} |")
        a(f"| Stocks scored | {facts.get('scored', '?')} |")
        a(f"| With a price | {facts.get('priced', '?')} |")
        a(f"| With an analyst target | {facts.get('targeted', '?')} |")
        if facts.get("top5"):
            a(f"| Top 5 | {', '.join(facts['top5'])} |")
    a(f"| Evidence for weight changes | {ic_observations()} |")
    a(f"| Context signals | {context_record()} |")
    _filings = filing_check(tail(data_log))
    if _filings:
        a(f"| Inputs vs SEC filings | {_filings} |")
    _top = top_of_ranking(tail(data_log))
    if _top:
        a(f"| Top 25, worth a look | {_top} |")
    _timing = run_timing(tail(data_log))
    if _timing:
        a(f"| Data run timing | {_timing} |")
    a(f"| Context coverage | {context_coverage()} |")
    a("")

    proposals = open_proposals(today=now)
    if proposals:
        a("## Waiting for a decision")
        a("")
        a("A session researched these, proposed a change and left it for review. The next")
        a("session decides and records why; you can also answer in `OWNER_FOCUS.md`.")
        a("")
        for p in proposals:
            a(f"- {p}")
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
