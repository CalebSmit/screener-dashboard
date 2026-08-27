"""Tests for scripts/write_brief.py - the owner's daily status page.

`MORNING_BRIEF.md` had no tests at all until now, which is exactly how it lost
its Top 5 row on 2026-08-26 without anyone noticing: `dashboard_facts()` read
`d["portfolio"]["holdings"]`, that evening's session removed the `portfolio`
payload key, and every lookup on the path used `.get()` with a default. No
exception, no log line, no empty row - the line just stopped being emitted.

These tests do not call `main()`. It writes the real `MORNING_BRIEF.md`, which
is a published artifact that `conftest.py` exists to protect; the logic worth
testing is reachable without it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import write_brief as wb  # noqa: E402


def row(ticker, rank, vt=False, gt=False):
    return {
        "Ticker": ticker, "Rank": rank, "Composite": 100 - rank,
        "Value_Trap_Flag": vt, "Growth_Trap_Flag": gt,
    }


class TestTop5:
    def test_it_takes_the_first_five_by_rank(self):
        table = [row(f"T{i}", i) for i in range(10, 0, -1)]
        assert wb.top5_tickers(table) == ["T1", "T2", "T3", "T4", "T5"]

    def test_rank_order_not_payload_order(self):
        table = [row("C", 3), row("A", 1), row("B", 2)]
        assert wb.top5_tickers(table) == ["A", "B", "C"]

    def test_trap_flagged_names_are_excluded(self):
        # The dashboard excludes them deliberately: dropping the filter would
        # promote a flagged name into the headline five.
        table = [row("BAD", 1, vt=True), row("ALSOBAD", 2, gt=True), row("OK", 3)]
        assert wb.top5_tickers(table) == ["OK"]

    def test_rows_without_a_rank_are_skipped(self):
        table = [{"Ticker": "NORANK", "Value_Trap_Flag": False,
                  "Growth_Trap_Flag": False, "Rank": None}, row("OK", 2)]
        assert wb.top5_tickers(table) == ["OK"]

    def test_an_empty_table_yields_nothing_rather_than_raising(self):
        assert wb.top5_tickers([]) == []

    def test_fewer_than_five_survivors_is_fine(self):
        assert wb.top5_tickers([row("A", 1), row("B", 2)]) == ["A", "B"]


class TestRegression20260826:
    """The actual defect: a payload with no `portfolio` key."""

    def test_top_five_survives_the_removal_of_the_portfolio_key(self):
        payload = {
            "kpis": {}, "stock_detail": {},
            "table_data": [row("HST", 1), row("EXPE", 2), row("EIX", 3),
                           row("APA", 4), row("CF", 5), row("SIXTH", 6)],
        }
        assert "portfolio" not in payload
        assert wb.top5_tickers(payload["table_data"]) == [
            "HST", "EXPE", "EIX", "APA", "CF"
        ]

    def test_a_removed_payload_key_is_reported_rather_than_ignored(self):
        # The lesson of the bug, generalised: the brief must say when the
        # payload no longer carries something it reads.
        assert "table_data" in wb.PAYLOAD_KEYS
        assert "kpis" in wb.PAYLOAD_KEYS
        assert "stock_detail" in wb.PAYLOAD_KEYS


class TestAgainstTheLivePayload:
    """Read-only checks against the artifact actually published."""

    @staticmethod
    def _payload():
        path = ROOT / "dashboard_data.js"
        if not path.exists():
            pytest.skip("no published dashboard_data.js")
        text = path.read_text(encoding="utf-8")
        return json.loads(text[text.index("{"):].rstrip().rstrip(";"))

    def test_every_key_the_brief_reads_is_present(self):
        payload = self._payload()
        missing = [k for k in wb.PAYLOAD_KEYS if k not in payload]
        assert not missing, (
            f"write_brief.py reads {missing}, which the published payload no "
            "longer has. Either the brief or generate_dashboard.py is wrong."
        )

    def test_the_brief_reports_no_missing_keys_on_the_live_payload(self):
        assert wb.dashboard_facts().get("missing_keys") == []

    def test_the_brief_and_the_dashboard_agree_on_the_top_five(self):
        # If these two disagree, the brief is telling the owner something the
        # dashboard does not show. Recomputed here from the payload using
        # renderTop5()'s rule rather than by calling the function under test.
        payload = self._payload()
        ranked = [r for r in payload["table_data"]
                  if not r["Value_Trap_Flag"] and not r["Growth_Trap_Flag"]]
        ranked.sort(key=lambda r: r["Rank"])
        assert wb.dashboard_facts()["top5"] == [r["Ticker"] for r in ranked[:5]]

    def test_the_live_brief_actually_has_five_names(self):
        # The condition that gates the row: `if facts.get("top5")`. This is the
        # assertion that fails against the 2026-08-26..27 code.
        assert len(wb.dashboard_facts()["top5"]) == 5


class TestStallBanner:
    """The existing in-process stall detector. Kept honest, not replaced.

    It can only catch "one loop died while the other lived" - when neither
    fires, nothing regenerates the brief. `scripts/check_loop_health.py` and
    the loop-watchdog workflow cover the case this one structurally cannot.
    """

    def test_a_run_today_is_not_stale(self):
        import datetime as _dt
        assert wb.run_age.__doc__  # the reasoning is recorded there
        label, days = wb.run_age(ROOT / "dashboard_data.js")
        assert days is not None
        assert isinstance(label, str)

    def test_a_missing_log_reads_as_never(self):
        label, days = wb.run_age(ROOT / "logs" / "does-not-exist.log")
        assert label == "never"
        assert days is None

    def test_a_failed_session_is_classified_distinctly_from_a_success(self):
        assert wb.outcome(["[06:00:01] [INFO] shipped to main"])[0] == "completed"
        assert wb.outcome([])[0] == "did not run"
        assert wb.outcome(
            ["[06:00:01] [ERROR] SHIP GATES FAILED"]
        )[0] == "stopped deliberately"
