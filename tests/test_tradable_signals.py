"""Only the gated run is tradable: ``signals.json``.

`OWNER_FOCUS.md` 2026-10-21 item 5 / screener-bot ROADMAP 1.2. The reasoning is
in ``scripts/write_signals.py``'s docstring. What these tests defend is the
asymmetry that makes the file mean anything:

* ``data-run.ps1`` writes it **after every gate** and **before the commit**,
  and stages it;
* ``nightly-screener.ps1`` never writes or stages it, so a code session's
  regenerated page is published with a ``signals.json`` that still describes
  the last gated run.

Lose either half and the file becomes a second copy of the ranking with a
reassuring name on it. The ordering half is checked the way
``tests/test_gate_ordering.py`` and ``tests/test_payload_parse_gate.py`` check
theirs - by position in the script - because there is no way to run an
unattended PowerShell publish inside the suite.
"""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import write_signals as W  # noqa: E402

DATA_RUN = ROOT / "scripts" / "data-run.ps1"
CODE_RUN = ROOT / "scripts" / "nightly-screener.ps1"


def _code(path: Path) -> str:
    """Script text with comments stripped, so a mention in prose is not a call."""
    src = path.read_text(encoding="utf-8-sig")
    return re.sub(r"(?m)#.*$", "", re.sub(r"<#.*?#>", "", src, flags=re.S))


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------

def _payload(n: int = 60) -> dict:
    rows, detail = [], {}
    for i in range(n):
        t = "T%02d" % i
        rows.append({"Ticker": t, "Rank": i + 1, "Composite": 90.0 - i,
                     "Sector": "Energy", "Value_Trap_Flag": i == 3,
                     "Growth_Trap_Flag": False})
        detail[t] = {"price": 10.0 + i, "price_date": "2026-10-23"}
    return {
        "table_data": rows,
        "stock_detail": detail,
        "kpis": {"run_timestamp": "2026-10-24T02:00:00", "run_id": "abc123",
                 "git_sha": "f" * 40, "methodology_version": "v7",
                 "raw_sha256": "a" * 64, "universe_size": n, "stocks_scored": n},
    }


@pytest.fixture()
def doc():
    return W.build(_payload(), {"run_health": "pass", "payload_parse": "pass"}, "2026-10-24")


def test_it_publishes_the_top_forty_in_rank_order(doc):
    assert doc["top_n"] == 40 == len(doc["top"])
    assert [r["rank"] for r in doc["top"]] == list(range(1, 41))
    assert doc["top"][0]["ticker"] == "T00"


def test_the_ranking_is_read_by_rank_not_by_row_order():
    """The 2026-08-26 commit-subject regression, in a file that places trades.

    A payload serialised in some other order must still yield the real top.
    """
    payload = _payload()
    payload["table_data"].reverse()
    doc = W.build(payload, {"run_health": "pass"}, "2026-10-24")
    assert [r["ticker"] for r in doc["top"][:3]] == ["T00", "T01", "T02"]


def test_each_name_carries_what_the_bot_needs_to_act(doc):
    first = doc["top"][0]
    assert set(first) == {"rank", "ticker", "composite", "sector",
                          "value_trap", "growth_trap", "price", "price_date"}
    assert first["price"] == 10.0
    assert doc["top"][3]["value_trap"] is True


def test_it_carries_the_audit_trail_of_the_run_it_came_from(doc):
    assert doc["run_id"] == "abc123"
    assert doc["git_sha"] == "f" * 40
    assert doc["methodology_version"] == "v7"
    assert doc["raw_sha256"] == "a" * 64
    assert doc["run_timestamp"] == "2026-10-24T02:00:00"


def test_a_payload_that_cannot_name_its_run_is_refused():
    """Without run_id there is no answer to "which run did I trade?" - and the
    bot's input gates are built on being able to answer it."""
    payload = _payload()
    payload["kpis"].pop("run_id")
    with pytest.raises(W.SignalsError, match="run_id"):
        W.build(payload, {"run_health": "pass"}, "2026-10-24")


def test_an_empty_ranking_is_refused_rather_than_published_empty():
    payload = _payload()
    payload["table_data"] = []
    with pytest.raises(W.SignalsError):
        W.build(payload, {"run_health": "pass"}, "2026-10-24")


def test_a_short_universe_publishes_what_there_is():
    doc = W.build(_payload(12), {"run_health": "pass"}, "2026-10-24")
    assert doc["top_n"] == 12


# ---------------------------------------------------------------------------
# Gate verdicts
# ---------------------------------------------------------------------------

def test_a_warning_verdict_is_recorded_and_flagged():
    """node or pytest absent is a documented fallback, not a silent pass - the
    consumer is entitled to know a check did not run."""
    doc = W.build(_payload(), {"payload_parse": "warn", "run_health": "pass"}, "2026-10-24")
    assert doc["gates"]["payload_parse"] == "warn"
    assert doc["all_gates_ran"] is False


def test_all_gates_ran_is_true_only_when_every_verdict_is_pass(doc):
    assert doc["all_gates_ran"] is True


def test_a_failing_verdict_cannot_be_recorded():
    """This file exists only after every gate accepted. A 'fail' in it would be
    a statement that cannot be true, so it is rejected at the boundary."""
    with pytest.raises(W.SignalsError, match="after every gate"):
        W.parse_gates(["run_health=fail"])


def test_no_gates_at_all_is_refused():
    with pytest.raises(W.SignalsError, match="refusing to claim"):
        W.parse_gates([])


def test_gate_arguments_parse_both_ways():
    assert W.parse_gates(["a=pass", "b=warn"]) == {"a": "pass", "b": "warn"}
    assert W.parse_gates(["a=pass,b=warn"]) == {"a": "pass", "b": "warn"}
    with pytest.raises(W.SignalsError):
        W.parse_gates(["nonsense"])


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def test_the_file_is_replaced_in_one_step_and_leaves_no_scratch(tmp_path):
    out = tmp_path / "signals.json"
    out.write_text('{"stale": true}', encoding="utf-8")
    W.write_atomic({"fresh": True}, out)
    assert json.loads(out.read_text()) == {"fresh": True}
    assert list(tmp_path.iterdir()) == [out], "a .tmp file was left behind"


def test_main_writes_the_file_from_a_payload_file(tmp_path):
    payload = tmp_path / "dashboard_data.js"
    payload.write_text("window.SCREENER_DATA=" + json.dumps(_payload()) + ";", encoding="utf-8")
    out = tmp_path / "signals.json"
    rc = W.main(["--date", "2026-10-24", "--gate", "run_health=pass",
                 "--payload", str(payload), "--out", str(out)])
    assert rc == 0
    assert json.loads(out.read_text())["top_n"] == 40


def test_main_fails_without_writing_when_the_payload_is_not_a_payload(tmp_path):
    bad = tmp_path / "dashboard_data.js"
    bad.write_text("<html>404</html>", encoding="utf-8")
    out = tmp_path / "signals.json"
    assert W.main(["--date", "2026-10-24", "--gate", "run_health=pass",
                   "--payload", str(bad), "--out", str(out)]) == 1
    assert not out.exists(), "a failed run must leave the previous signals.json alone"


# ---------------------------------------------------------------------------
# The two loops: who may write it
# ---------------------------------------------------------------------------

GATE_MARKERS = [
    ("synthetic data", "$GateVerdicts['synthetic_data']"),
    ("fetch failure rate", "$GateVerdicts['fetch_failure_rate']"),
    ("run health", "$GateVerdicts['run_health']"),
    ("dashboard generation", "$GateVerdicts['dashboard_generated']"),
    ("payload size floor", "$GateVerdicts['payload_size']"),
    ("payload parse", "$GateVerdicts['payload_parse']"),
    ("payload header", "$GateVerdicts['payload_header']"),
    ("published claims", "$GateVerdicts['published_claims']"),
]


def test_the_data_loop_records_a_verdict_for_every_gate_it_runs():
    code = _code(DATA_RUN)
    for name, marker in GATE_MARKERS:
        assert marker in code, f"the {name} gate records no verdict"


def test_the_write_comes_after_every_gate_and_before_the_commit():
    """The ordering is the claim. Written earlier it would describe a run that
    had not yet been accepted; written after the commit it would not be in it.
    """
    code = _code(DATA_RUN)
    write_at = code.index("write_signals.py")
    for name, marker in GATE_MARKERS:
        assert code.index(marker) < write_at, f"signals.json is written before the {name} gate"
    commit_at = code.index("'commit', '-m'")
    assert write_at < commit_at, "signals.json is written after the commit that publishes it"


def test_the_data_loop_stages_it():
    block = _code(DATA_RUN)
    start = block.index("$DataArtifacts")
    assert "'signals.json'" in block[start:block.index(")", start)]


def test_a_failed_write_does_not_discard_the_run():
    """A 2 KB side file must not cost a day of evidence. The previous
    signals.json stays, which the bot reads as stale - the safe state."""
    code = _code(DATA_RUN)
    after = code[code.index("write_signals.py"):]
    branch = after[:after.index("$DataArtifacts")]
    assert "Stop-Run" not in branch, "a failed signals write discards the run"
    assert "'WARN'" in branch


def test_the_code_loop_never_writes_or_stages_it():
    """The mechanism. A code session's full run may republish the page; it must
    not republish the tradable ranking, or the gate stack means nothing."""
    code = _code(CODE_RUN)
    assert "signals.json" not in code
    assert "write_signals" not in code


def test_nothing_in_the_pipeline_writes_it_except_the_one_script():
    """A second writer anywhere in the repo would reopen the hole silently."""
    writers = []
    for py in list(ROOT.glob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        if py.name == "write_signals.py":
            continue
        text = py.read_text(encoding="utf-8", errors="ignore")
        # The literal filename appearing in a write context. Prose mentions in
        # docstrings are fine; an open()/write_text()/to_json() is not.
        for line in text.splitlines():
            if "signals.json" in line and re.search(r"open\(|write_text|write_bytes|to_json|dump", line):
                writers.append(f"{py.name}: {line.strip()}")
    assert writers == [], "another module writes signals.json: " + "; ".join(writers)


def test_the_tripwire_protects_it_from_the_test_suite():
    """conftest.py fails the session if a test writes it - a test that did
    would be asserting a run was gated when it was not."""
    import conftest
    assert ROOT / "signals.json" in conftest.PROTECTED_PATHS
