"""Tests for scripts/check_loop_health.py - the external loop watchdog.

The thing being defended here is subtle. A watchdog that cries wolf gets
ignored, and an ignored watchdog is worse than none because it looks like
coverage. So most of these tests are about the cases where it must stay
*quiet*: weekends, a slot whose deadline has not arrived, a late catch-up run,
a single blip.

The load-bearing test is `TestRealOutage`, which replays the outage of
2026-08-17..2026-08-20 recorded by the 2026-08-21 retrospective and asserts
both that the watchdog detects it and *when* it would first have spoken.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_loop_health.py"

# Same idiom as tests/test_run_health.py and tests/test_commit_subject.py.
sys.path.insert(0, str(ROOT / "scripts"))

import check_loop_health as clh  # noqa: E402

DATA = next(l for l in clh.LOOPS if l.key == "data")
CODE = next(l for l in clh.LOOPS if l.key == "code")


def d(s: str) -> date:
    return date.fromisoformat(s)


def dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


# ---------------------------------------------------------------------------
# closed_weekdays - which days are even judgeable yet
# ---------------------------------------------------------------------------


class TestClosedWeekdays:
    def test_weekends_are_never_judged(self):
        # 2026-08-27 is a Thursday; the window reaches back over two weekends.
        got = clh.closed_weekdays(dt("2026-08-27T17:00"), deadline_hour=12)
        assert all(x.weekday() < 5 for x in got)
        assert d("2026-08-22") not in got  # Saturday
        assert d("2026-08-23") not in got  # Sunday

    def test_today_is_not_judged_before_its_deadline(self):
        # 06:00 on a Thursday: the data loop's noon deadline has not arrived,
        # so today cannot yet count as missed. Judging it would report a stall
        # every single morning.
        got = clh.closed_weekdays(dt("2026-08-27T06:00"), deadline_hour=12)
        assert d("2026-08-27") not in got

    def test_today_is_judged_once_its_deadline_passes(self):
        got = clh.closed_weekdays(dt("2026-08-27T12:00"), deadline_hour=12)
        assert d("2026-08-27") in got

    def test_the_two_loops_have_different_deadlines(self):
        # 14:00: the data loop's noon deadline has passed, the code loop's
        # 16:00 one has not. A shared deadline would either judge the code
        # loop too early or the data loop too late.
        at_2pm = dt("2026-08-27T14:00")
        assert d("2026-08-27") in clh.closed_weekdays(at_2pm, DATA.deadline_hour)
        assert d("2026-08-27") not in clh.closed_weekdays(at_2pm, CODE.deadline_hour)

    def test_oldest_first(self):
        got = clh.closed_weekdays(dt("2026-08-27T17:00"), deadline_hour=12)
        assert got == sorted(got)

    def test_lookback_bounds_the_window(self):
        got = clh.closed_weekdays(dt("2026-08-27T17:00"), 12, lookback_days=3)
        assert min(got) >= d("2026-08-24")


# ---------------------------------------------------------------------------
# assess - the verdict
# ---------------------------------------------------------------------------


def every_weekday(start: str, end: str) -> set[date]:
    cur, stop, out = d(start), d(end), set()
    while cur <= stop:
        if cur.weekday() < 5:
            out.add(cur)
        cur += timedelta(days=1)
    return out


class TestAssess:
    def test_a_healthy_loop_is_ok(self):
        beats = every_weekday("2026-08-10", "2026-08-27")
        s = clh.assess(DATA, beats, dt("2026-08-27T17:00"))
        assert s.verdict == clh.OK
        assert s.missed == []
        assert s.consecutive_missed == 0
        assert s.last_heartbeat == d("2026-08-27")

    def test_one_missed_weekday_warns_but_does_not_alarm(self):
        beats = every_weekday("2026-08-10", "2026-08-27") - {d("2026-08-27")}
        s = clh.assess(DATA, beats, dt("2026-08-27T17:00"))
        assert s.consecutive_missed == 1
        assert s.verdict == clh.WARN

    def test_two_consecutive_missed_weekdays_is_a_stall(self):
        beats = every_weekday("2026-08-10", "2026-08-27") - {
            d("2026-08-26"), d("2026-08-27")
        }
        s = clh.assess(DATA, beats, dt("2026-08-27T17:00"))
        assert s.consecutive_missed == 2
        assert s.verdict == clh.STALLED

    def test_a_weekend_gap_is_not_a_stall(self):
        # The single most important quiet case. Monday 17:00, with the last run
        # on Friday: three calendar days of silence, zero missed weekday slots.
        # A naive "days since last run >= 2" check alarms here every Monday.
        beats = every_weekday("2026-08-10", "2026-08-21")  # through Friday
        monday = dt("2026-08-24T17:00")
        s = clh.assess(DATA, beats | {d("2026-08-24")}, monday)
        assert s.verdict == clh.OK
        assert s.consecutive_missed == 0

    def test_an_old_gap_does_not_alarm_once_the_loop_recovers(self):
        # Scattered history: the outage is visible in `missed` for context, but
        # the verdict tracks *now*, so a recovered loop reads OK.
        beats = every_weekday("2026-08-10", "2026-08-27") - every_weekday(
            "2026-08-17", "2026-08-20"
        )
        s = clh.assess(CODE, beats, dt("2026-08-27T17:00"))
        assert len(s.missed) == 4
        assert s.consecutive_missed == 0
        assert s.verdict == clh.OK

    def test_a_loop_that_never_ran_reports_no_heartbeat(self):
        s = clh.assess(DATA, set(), dt("2026-08-27T17:00"))
        assert s.last_heartbeat is None
        assert s.verdict == clh.STALLED
        assert s.consecutive_missed == s.weekdays_checked

    def test_a_future_dated_commit_is_not_treated_as_a_heartbeat(self):
        # Clock skew on the scheduling machine must not let tomorrow's commit
        # vouch for today.
        s = clh.assess(DATA, {d("2026-09-30")}, dt("2026-08-27T17:00"))
        assert s.last_heartbeat is None


class TestOverall:
    def test_the_worst_loop_sets_the_verdict(self):
        healthy = clh.assess(DATA, every_weekday("2026-08-10", "2026-08-27"),
                             dt("2026-08-27T17:00"))
        dead = clh.assess(CODE, set(), dt("2026-08-27T17:00"))
        assert clh.overall([healthy, dead]) == clh.STALLED
        assert clh.overall([healthy, healthy]) == clh.OK


# ---------------------------------------------------------------------------
# The heartbeat itself
# ---------------------------------------------------------------------------


class TestHeartbeatMatching:
    def _commits(self, *subjects):
        return [(dt("2026-08-27T02:13:09+00:00"), s) for s in subjects]

    def test_real_data_loop_subjects_match(self):
        c = self._commits(
            "brief: data run 2026-08-27",
            "data: screener run 2026-08-27 - 502 scored, top: HST EXPE EIX APA CF",
        )
        assert clh.heartbeat_dates(c, DATA) == {d("2026-08-27")}

    def test_real_code_loop_subject_matches(self):
        c = self._commits("brief: code session 2026-08-27")
        assert clh.heartbeat_dates(c, CODE) == {d("2026-08-27")}

    def test_a_failed_session_still_counts_as_a_heartbeat(self):
        # "SESSION DID NOT RUN" means the loop fired and the session failed.
        # That is a different problem, already made loud by write_brief.py.
        # This file answers only "did the task fire at all".
        c = self._commits("brief: code session 2026-08-27 - SESSION DID NOT RUN")
        assert clh.heartbeat_dates(c, CODE) == {d("2026-08-27")}

    def test_an_evening_session_is_not_a_code_loop_heartbeat(self):
        # Owner-initiated interactive work proves a human was present, not that
        # the 06:00 scheduled task fired.
        c = self._commits("brief: evening session 2026-08-27")
        assert clh.heartbeat_dates(c, CODE) == set()

    def test_ordinary_commits_are_not_heartbeats(self):
        c = self._commits(
            "fix: refuse a price series that mixes two split scales",
            "docs: record the fix",
            "process: act on the retrospective's owner-flagged items",
        )
        assert clh.heartbeat_dates(c, DATA) == set()
        assert clh.heartbeat_dates(c, CODE) == set()

    def test_the_loops_do_not_vouch_for_each_other(self):
        c = self._commits("brief: data run 2026-08-27")
        assert clh.heartbeat_dates(c, DATA) == {d("2026-08-27")}
        assert clh.heartbeat_dates(c, CODE) == set()

    def test_the_commits_own_date_is_used_not_the_observers(self):
        # A 02:13 Chicago run is 07:13 UTC the same day, but a 23:28 Chicago
        # catch-up run is 04:28 UTC the *next* day. A watchdog running in UTC
        # must still credit it to 08-20, or the real 2026-08-20 catch-up would
        # read as a miss.
        late = [(datetime(2026, 8, 20, 23, 28, 50,
                          tzinfo=timezone(timedelta(hours=-5))),
                 "brief: data run 2026-08-20")]
        assert clh.heartbeat_dates(late, DATA) == {d("2026-08-20")}


class TestScheduleNow:
    def test_the_scheduling_offset_comes_from_the_newest_commit(self):
        commits = [(datetime(2026, 8, 27, 2, 13,
                             tzinfo=timezone(timedelta(hours=-5))),
                    "brief: data run 2026-08-27")]
        now = clh.schedule_now(commits)
        assert now.tzinfo is None  # naive, in scheduling-local time
        reference = datetime.now(timezone(timedelta(hours=-5))).replace(tzinfo=None)
        assert abs((now - reference).total_seconds()) < 120

    def test_no_commits_falls_back_to_system_local(self):
        now = clh.schedule_now([])
        assert now.tzinfo is None
        assert abs((now - datetime.now()).total_seconds()) < 120


# ---------------------------------------------------------------------------
# The outage this file was built for
# ---------------------------------------------------------------------------


class TestRealOutage:
    """Replay 2026-08-17..2026-08-20, as recorded by the 08-21 retrospective.

    Ground truth, from `CLAUDE.md` priority -1 and confirmed against the git
    history of `main`: the code loop produced nothing on 08-17, 08-18, 08-19 or
    08-20. The data loop was equally silent until a logon catch-up fired at
    23:28 on 08-20 (`data: screener run 2026-08-20`), which is why it is
    credited with that day and the code loop is not.
    """

    CODE_BEATS = every_weekday("2026-08-03", "2026-08-14") | {d("2026-08-21")}
    DATA_BEATS = CODE_BEATS | {d("2026-08-20")}

    def test_the_code_loop_outage_is_detected_in_full(self):
        s = clh.assess(CODE, self.CODE_BEATS, dt("2026-08-20T17:00"))
        assert s.missed[-4:] == [d("2026-08-17"), d("2026-08-18"),
                                 d("2026-08-19"), d("2026-08-20")]
        assert s.consecutive_missed == 4
        assert s.verdict == clh.STALLED

    def test_the_data_loops_catch_up_run_is_credited_to_that_day(self):
        s = clh.assess(DATA, self.DATA_BEATS, dt("2026-08-20T23:59"))
        assert d("2026-08-20") not in s.missed
        assert s.consecutive_missed == 0

    def test_it_would_have_spoken_on_the_second_day(self):
        # The measurable improvement. The outage ran six days before anyone
        # noticed by counting files. On 2026-08-18 at 17:00 - two days in -
        # the watchdog already reads STALLED.
        s = clh.assess(CODE, self.CODE_BEATS, dt("2026-08-18T17:00"))
        assert s.consecutive_missed == 2
        assert s.verdict == clh.STALLED

    def test_it_stays_quiet_on_the_first_day(self):
        # 08-17 alone is a WARN, not an alarm. One missed slot is a reboot.
        s = clh.assess(CODE, self.CODE_BEATS, dt("2026-08-17T17:00"))
        assert s.consecutive_missed == 1
        assert s.verdict == clh.WARN

    def test_it_was_quiet_on_the_friday_before(self):
        # 2026-08-14 ran (and failed on an API quota, which is a heartbeat).
        # The watchdog must not have been alarming going into the outage, or
        # the alarm on 08-18 would have been lost in noise.
        s = clh.assess(CODE, self.CODE_BEATS, dt("2026-08-14T17:00"))
        assert s.verdict == clh.OK


# ---------------------------------------------------------------------------
# The command-line surface the workflow depends on
# ---------------------------------------------------------------------------


class TestCommandLine:
    def _run(self, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )

    def test_json_output_parses_and_has_the_shape_the_workflow_reads(self):
        proc = self._run("--json")
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["verdict"] in (clh.OK, clh.WARN, clh.STALLED)
        assert {l["key"] for l in payload["loops"]} == {"data", "code"}
        for loop in payload["loops"]:
            assert set(loop) >= {
                "key", "label", "slot", "verdict", "last_heartbeat",
                "missed", "consecutive_missed", "weekdays_checked",
            }

    def test_human_output_names_both_loops(self):
        proc = self._run()
        assert proc.returncode == 0, proc.stderr
        assert "Data run" in proc.stdout
        assert "Code session" in proc.stdout

    def test_it_exits_zero_by_default_so_it_cannot_break_a_caller(self):
        assert self._run().returncode == 0

    def test_strict_maps_verdicts_to_exit_codes(self):
        assert clh.EXIT_CODES[clh.OK] == 0
        assert clh.EXIT_CODES[clh.WARN] == 0
        assert clh.EXIT_CODES[clh.STALLED] == 2

    def test_it_reads_this_repos_real_history(self):
        # An end-to-end check that the git plumbing works: this repo has run
        # both loops today, so both must show a heartbeat.
        payload = json.loads(self._run("--json").stdout)
        for loop in payload["loops"]:
            assert loop["last_heartbeat"] is not None, (
                f"{loop['key']} found no heartbeat in real history - the "
                "subject patterns or the git ref resolution have drifted"
            )


WORKFLOW = ROOT / ".github" / "workflows" / "loop-watchdog.yml"


@pytest.fixture(scope="module")
def wf():
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


class TestWorkflow:
    """Static checks on the external observer.

    None of this can be executed locally - the workflow only ever really runs
    on GitHub. That is exactly why it is worth pinning statically: a typo in
    the embedded JavaScript, or a missing permission, would otherwise surface
    as silence at the moment the watchdog was supposed to speak, which is
    indistinguishable from the failure it exists to report.
    """

    WORKFLOW = WORKFLOW

    def _triggers(self, wf):
        # PyYAML reads the bare key `on:` as the boolean True. GitHub does not;
        # accept either so this test is not pinning a YAML quirk.
        return wf.get("on", wf.get(True))

    def test_the_workflow_exists(self):
        assert self.WORKFLOW.exists()

    def test_it_runs_on_a_weekday_schedule_and_on_demand(self, wf):
        triggers = self._triggers(wf)
        assert "workflow_dispatch" in triggers
        crons = [c["cron"] for c in triggers["schedule"]]
        assert crons, "no cron - a watchdog nobody triggers watches nothing"
        for cron in crons:
            minute, hour, _, _, dow = cron.split()
            assert dow == "1-5", "the loops are Mon-Fri; weekend alarms are noise"
            # Must land after the latest deadline (16:00 local) even in CST
            # (UTC-6), i.e. at or after 22:00 UTC. Earlier and it would judge
            # the code loop before its deadline had passed.
            assert int(hour) >= 22, f"{cron} runs before the code loop's deadline"

    def test_it_can_open_issues_but_not_write_to_the_repo(self, wf):
        assert wf["permissions"]["issues"] == "write"
        assert wf["permissions"]["contents"] == "read", (
            "the watchdog must never be able to push - it only observes"
        )

    def test_it_checks_out_full_history(self, wf):
        checkout = next(s for s in wf["jobs"]["check"]["steps"]
                        if "actions/checkout" in str(s.get("uses", "")))
        assert checkout["with"]["fetch-depth"] == 0, (
            "a shallow clone hides the heartbeat commits and reports a false stall"
        )

    def test_it_invokes_the_checker_this_module_tests(self, wf):
        run_steps = " ".join(s.get("run", "") for s in wf["jobs"]["check"]["steps"])
        assert "scripts/check_loop_health.py" in run_steps
        assert "--json" in run_steps

    def test_the_embedded_javascript_is_syntactically_valid(self, wf):
        node = shutil.which("node")
        if not node:
            pytest.skip("node not available")
        step = next(s for s in wf["jobs"]["check"]["steps"]
                    if "github-script" in str(s.get("uses", "")))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wf.mjs"
            # github-script runs the body inside an async wrapper, so top-level
            # `await` is legal there; reproduce that to check it honestly.
            path.write_text(
                "async function main(github, context, core){\n"
                + step["with"]["script"] + "\n}\n",
                encoding="utf-8",
            )
            proc = subprocess.run([node, "--check", str(path)],
                                  capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr

    def test_it_only_opens_an_issue_when_stalled(self, wf):
        step = next(s for s in wf["jobs"]["check"]["steps"]
                    if "github-script" in str(s.get("uses", "")))
        script = step["with"]["script"]
        assert "'stalled'" in script
        assert "issues.create" in script
        assert "state: 'closed'" in script, "it must close the issue on recovery"

    def test_it_reuses_one_issue_rather_than_opening_many(self, wf):
        step = next(s for s in wf["jobs"]["check"]["steps"]
                    if "github-script" in str(s.get("uses", "")))
        script = step["with"]["script"]
        assert "issues.update" in script
        assert "state: 'open'" in script, "it must look for an existing issue first"


class TestRenderedAdvice:
    def test_a_stall_report_points_at_the_registration_script(self):
        dead = clh.assess(CODE, set(), dt("2026-08-27T17:00"))
        text = clh.render([dead], clh.STALLED, dt("2026-08-27T17:00"))
        assert "register-tasks.ps1" in text
        assert "logged on" in text

    def test_a_healthy_report_does_not_give_troubleshooting_advice(self):
        ok = clh.assess(DATA, every_weekday("2026-08-10", "2026-08-27"),
                        dt("2026-08-27T17:00"))
        text = clh.render([ok], clh.OK, dt("2026-08-27T17:00"))
        assert "register-tasks.ps1" not in text
        assert "Both loops are firing." in text
