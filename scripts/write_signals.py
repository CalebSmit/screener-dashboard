"""Write ``signals.json`` - the one ranking a downstream consumer may trade.

`OWNER_FOCUS.md` 2026-10-21 item 5 (screener-bot ROADMAP 1.2). The problem it
closes: ``dashboard_data.js`` is published by **two** paths. The 02:00 data
loop publishes it behind seven gates (no synthetic data, a fetch-failure
ceiling, ``check_run_health.py`` with its rank-continuity, empty-record and
price-freshness rules, a size floor, a real ``node --check`` parse, the opening
assignment, and ``check_published_claims.py``). A code session that runs the
full screener can also rewrite the page, and **none of those gates stand in
front of it** - its four ship gates are about the repository, not about the
run. The bot reads the page as GitHub Pages serves it and cannot tell the two
apart.

So the gated run writes a second, small file, and the bot trades *that*:

* it is written **after every gate has passed** and **before the commit**, so
  it exists only for a run the whole gate stack accepted;
* it is derived from the payload those gates just inspected, so the ranking in
  it is by construction the ranking that was checked - not a second read of
  the run directory that could disagree;
* ``scripts/nightly-screener.ps1`` never stages it, so a code session's page
  goes live with a ``signals.json`` that still describes the last gated run.
  The bot's own freshness rule then declines to trade, which is the intended
  outcome: the page keeps being published, it just stops being tradable.

Nothing here is a gate. If this script fails the run still publishes - a
discarded run costs a day of evidence, and a stale ``signals.json`` is already
the safe state. It fails loudly instead, and writes atomically so a reader
never sees half a file.

Usage::

    python scripts/write_signals.py --date 2026-10-24 \
        --gate run_health=pass --gate payload_parse=warn ...

``tests/test_tradable_signals.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The top of the ranking, deep enough that the bot can fill a 25-name book
# after dropping names its own rules reject (a trap flag, a missing price, a
# position it already holds). 40 is the item's number.
TOP_N = 40

PAYLOAD = "dashboard_data.js"
OUTPUT = "signals.json"
SCHEMA = 1

# Verdicts a gate may report. "pass" is the gate ran and accepted; "warn" is a
# documented fallback where the gate could not run at all (node absent, pytest
# absent) and the loop published on the checks around it - both are recorded in
# data-run.ps1. There is deliberately no "fail": this file is written only
# after every gate has accepted, so a failing verdict here would be a lie about
# how it came to exist.
VERDICTS = ("pass", "warn")


class SignalsError(RuntimeError):
    """Raised for anything that would make the file misdescribe the run."""


def load_payload(path: Path) -> dict:
    """Parse ``window.SCREENER_DATA = {...};`` the way commit_subject.py does."""
    raw = path.read_text(encoding="utf-8")
    match = re.search(r"window\.SCREENER_DATA\s*=\s*", raw)
    if not match:
        raise SignalsError(f"{path.name} is not a SCREENER_DATA payload")
    return json.loads(raw[match.end():].rstrip().rstrip(";"))


def parse_gates(pairs: list[str]) -> dict:
    """``["run_health=pass", ...]`` -> ``{"run_health": "pass", ...}``."""
    gates: dict[str, str] = {}
    for pair in pairs:
        for item in pair.split(","):          # tolerate one comma-joined argument
            item = item.strip()
            if not item:
                continue
            name, sep, verdict = item.partition("=")
            name, verdict = name.strip(), verdict.strip().lower()
            if not sep or not name:
                raise SignalsError(f"gate {item!r} is not name=verdict")
            if verdict not in VERDICTS:
                raise SignalsError(
                    f"gate {name!r} reports {verdict!r}; only {'/'.join(VERDICTS)} "
                    "are possible here - this file is written after every gate passed")
            gates[name] = verdict
    if not gates:
        raise SignalsError("no gate verdicts given - refusing to claim a run was gated")
    return gates


def top_names(payload: dict, n: int = TOP_N) -> list[dict]:
    """The top ``n`` rows by Rank, with everything the bot needs to act.

    Ordered by Rank and **not** by the payload's row order: the commit-subject
    regression of 2026-08-26 is the standing reminder that reading a ranking by
    position is how a file quietly starts naming the wrong stocks.
    """
    rows = [r for r in (payload.get("table_data") or []) if r.get("Rank") is not None]
    rows.sort(key=lambda r: (r["Rank"], str(r.get("Ticker") or "")))
    detail = payload.get("stock_detail") or {}
    out = []
    for row in rows[:n]:
        ticker = str(row.get("Ticker") or "")
        if not ticker:
            continue
        price = (detail.get(ticker) or {}).get("price")
        out.append({
            "rank": int(row["Rank"]),
            "ticker": ticker,
            "composite": row.get("Composite"),
            "sector": row.get("Sector"),
            "value_trap": bool(row.get("Value_Trap_Flag")),
            "growth_trap": bool(row.get("Growth_Trap_Flag")),
            "price": price,
            "price_date": (detail.get(ticker) or {}).get("price_date"),
        })
    if not out:
        raise SignalsError("the payload has no ranked rows - nothing to publish")
    return out


def build(payload: dict, gates: dict, date: str) -> dict:
    """The signals document. Every field comes from the gated payload."""
    kpis = payload.get("kpis") or {}

    # The audit trail (OWNER_FOCUS item 6). A run that cannot name itself is
    # not a run the bot should act on, so this is the one hard requirement:
    # without run_id and run_timestamp, "which run did I trade?" has no answer.
    missing = [k for k in ("run_timestamp", "run_id") if not kpis.get(k)]
    if missing:
        raise SignalsError(
            "the payload does not identify its run (" + ", ".join(missing) +
            ") - see run_fingerprint.py; refusing to publish an untraceable signal")

    top = top_names(payload)
    return {
        "schema": SCHEMA,
        # Named so a reader of the file knows which loop stands behind it.
        "written_by": "scripts/data-run.ps1",
        "run_date": date,
        "run_timestamp": kpis.get("run_timestamp"),
        "run_id": kpis.get("run_id"),
        "git_sha": kpis.get("git_sha"),
        "methodology_version": kpis.get("methodology_version"),
        "raw_sha256": kpis.get("raw_sha256"),
        "universe_size": kpis.get("universe_size"),
        "stocks_scored": kpis.get("stocks_scored"),
        "gates": dict(sorted(gates.items())),
        # True only when every gate actually ran. A "warn" means a check could
        # not run at all, which the consumer is entitled to treat differently.
        "all_gates_ran": all(v == "pass" for v in gates.values()),
        "top_n": len(top),
        "top": top,
    }


def write_atomic(doc: dict, path: Path) -> None:
    """Replace in one step, so a reader never sees a half-written ranking."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True, help="the run date, YYYY-MM-DD")
    ap.add_argument("--gate", action="append", default=[], metavar="NAME=VERDICT",
                    help="a gate that accepted this run; repeatable")
    ap.add_argument("--payload", default=str(ROOT / PAYLOAD))
    ap.add_argument("--out", default=str(ROOT / OUTPUT))
    args = ap.parse_args(argv)

    try:
        gates = parse_gates(args.gate)
        payload = load_payload(Path(args.payload))
        doc = build(payload, gates, args.date)
        write_atomic(doc, Path(args.out))
    except (SignalsError, OSError, ValueError) as exc:
        print(f"signals: NOT written - {exc}", file=sys.stderr)
        return 1

    print(f"signals: {doc['top_n']} names, run {doc['run_id']}, "
          f"{len(gates)} gates ({'all ran' if doc['all_gates_ran'] else 'with a warning'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
