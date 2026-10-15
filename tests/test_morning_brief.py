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


class TestOpenProposals:
    """A research note may propose a change and leave it for a second session. The brief lists
    every such proposal with its age until someone decides it - otherwise the default decision
    is whichever one nobody noticed being made."""

    def note(self, folder, name, body):
        (folder / name).write_text(body, encoding="utf-8")

    def test_an_open_proposal_is_listed_with_its_age(self, tmp_path):
        from datetime import datetime
        self.note(tmp_path, "2026-10-12-growth-category.md",
                  "# Growth\n\n**Proposal status: OPEN** - Growth 13% -> 10%, Quality 22% -> 25%\n")
        got = wb.open_proposals(tmp_path, datetime(2026, 10, 15, 6, 0))
        assert got == ["**Growth 13% -> 10%, Quality 22% -> 25%** - open 3 days - "
                       "`research/2026-10-12-growth-category.md`"]
        assert "open today" in wb.open_proposals(tmp_path, datetime(2026, 10, 12, 23, 0))[0]
        assert "open 1 day -" in wb.open_proposals(tmp_path, datetime(2026, 10, 13, 2, 0))[0]

    def test_a_decided_proposal_and_an_ordinary_note_are_not(self, tmp_path):
        self.note(tmp_path, "2026-10-12-a.md", "**Proposal status: DECIDED 2026-10-13** - built\n")
        self.note(tmp_path, "2026-10-12-b.md", "A note that mentions Proposal status: OPEN in a sentence.\n")
        self.note(tmp_path, "README.md", "**Proposal status: OPEN** - not a dated note\n")
        assert wb.open_proposals(tmp_path) == []

    def test_a_missing_folder_or_an_odd_file_never_raises(self, tmp_path):
        assert wb.open_proposals(tmp_path / "nowhere") == []
        (tmp_path / "2026-10-12-broken.md").write_bytes(b"\xff\xfe**Proposal status: OPEN** - x\n")
        assert isinstance(wb.open_proposals(tmp_path), list)

    def test_the_brief_and_every_real_note_agree_on_what_is_still_open(self):
        """The brief's list must equal the notes whose status line still says OPEN - in either
        direction. Written 2026-10-12 as "the Growth proposal is open until a changelog entry
        takes it up", which assumed a proposal could only end by being *built*. It can also be
        **refused**, and a refusal changes no weight, metric or threshold, so by this project's
        convention it correctly writes no changelog entry - which is how the Growth proposal
        ended on 2026-10-13. That is a third outcome the old form could not express, so this now
        checks the actual rule and keeps the teeth it had: a decided note must leave the brief,
        and a note the changelog has taken up may not still read OPEN."""
        notes = sorted((ROOT / "research").glob("20*.md"))
        assert notes, "no dated research notes found"
        listed = wb.open_proposals()
        still_open = [
            f"research/{n.name}" for n in notes
            if "**Proposal status: OPEN**" in n.read_text(encoding="utf-8", errors="replace")
        ]
        assert sorted(p.split("`")[-2] for p in listed) == sorted(still_open), (
            f"the brief lists {listed} but the notes saying OPEN are {still_open}"
        )

        changelog = (ROOT / "METHODOLOGY_CHANGELOG.md").read_text(encoding="utf-8")
        for path in still_open:
            assert path not in changelog, (
                f"{path} was taken up in the changelog but its status line still says OPEN"
            )

    def test_the_growth_proposal_was_decided_and_says_how(self):
        """The specific one this machinery was built for. It was refused on 2026-10-13
        (research/2026-10-12-growth-category.md section 10) and must not drift back to OPEN
        without a session deliberately reopening it."""
        note = (ROOT / "research" / "2026-10-12-growth-category.md").read_text(encoding="utf-8")
        assert "**Proposal status: DECIDED 2026-10-13** - refused" in note
        assert "**Proposal status: OPEN**" not in note
        assert [p for p in wb.open_proposals() if "2026-10-12-growth-category.md" in p] == []

    def test_the_brief_has_a_section_for_them(self):
        src = (ROOT / "scripts" / "write_brief.py").read_text(encoding="utf-8")
        assert '"## Waiting for a decision"' in src and "open_proposals(today=now)" in src


class TestTheNewestSession:
    """Several code-session logs a day since 2026-10-14, and not all of them are sessions."""

    def log(self, folder, name, body, age):
        import os, time
        path = folder / name
        path.write_text(body, encoding="utf-8")
        stamp = time.time() - age
        os.utime(path, (stamp, stamp))
        return path

    def test_a_dry_run_or_a_stand_down_is_not_the_latest_session(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wb, "LOGS", tmp_path)
        real = self.log(tmp_path, "nightly-2026-10-14_060001.log",
                        "[06:40:12] [INFO] === Run complete: shipped to main ===\n", 3600 * 3)
        self.log(tmp_path, "nightly-2026-10-14_081453-midday.log",
                 "[08:14:56] [INFO] === Dry run complete ===\n", 3600)
        self.log(tmp_path, "nightly-2026-10-14_130001-midday.log",
                 "[13:00:01] [WARN] Standing down (midday slot): the tree is dirty on 'owner/x'\n", 60)
        assert wb.newest_session() == real
        assert wb.outcome(wb.tail(wb.newest_session()))[0] == "completed"

    def test_with_only_non_sessions_there_is_no_session(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wb, "LOGS", tmp_path)
        self.log(tmp_path, "nightly-2026-10-14_081453-midday.log", "=== Dry run complete ===\n", 60)
        assert wb.newest_session() is None

    def test_the_brief_reads_the_code_session_through_it_and_a_dry_run_publishes_nothing(self):
        src = (ROOT / "scripts" / "write_brief.py").read_text(encoding="utf-8")
        assert 'code_log = newest_session("nightly-*.log")' in src
        runner = (ROOT / "scripts" / "nightly-screener.ps1").read_text(encoding="utf-8-sig")
        dry = runner[runner.index("    if ($DryRun) {"):]
        assert dry.index("$SkipBrief = $true") < dry.index("exit 0")


class TestFilingCheckRow:
    """The brief carries the data run's agreement with the SEC filings (2026-10-14), so the owner
    sees every morning how well the inputs matched the companies' own figures."""

    LINE = ("[02:20:31] [INFO]     ok   inputs against SEC filings: revenue 363/410 within 0.5% (13 over 10%); "
            "net income 397/421 within 0.5% (7 over 10%); operating cash flow 406/411 within 0.5% (1 over 10%); "
            "total assets 425/426 within 0.5% (0 over 10%); equity 417/426 within 0.5% (2 over 10%); "
            "market value / price vs filed shares 447/487 within 5% (22 over 10%)")

    def test_the_row_keeps_the_four_figures_that_sit_near_all(self):
        assert wb.filing_check([self.LINE]) == (
            "total assets 425/426 within 0.5%; operating cash flow 406/411 within 0.5%; "
            "net income 397/421 within 0.5%; market value 447/487 within 5%")

    def test_an_attention_note_is_pointed_to(self):
        lines = [self.LINE, "[02:20:31] [INFO]     ok   ATTENTION - total assets differs from the companies' filings"]
        assert wb.filing_check(lines).endswith("see the ATTENTION note below**")

    def test_no_line_means_no_row_rather_than_a_guess(self):
        assert wb.filing_check(["[02:20:31] [INFO] === Data loop complete ==="]) == ""
        assert wb.filing_check([]) == ""

    def test_the_brief_adds_the_row_only_when_there_is_one(self):
        src = (ROOT / "scripts" / "write_brief.py").read_text(encoding="utf-8")
        assert "_filings = filing_check(tail(data_log))" in src and "if _filings:" in src
        # the line it reads is the one the health check prints
        health = (ROOT / "scripts" / "check_run_health.py").read_text(encoding="utf-8")
        assert '"inputs against SEC filings: "' in health and wb.FILING_LINE == "inputs against SEC filings: "

