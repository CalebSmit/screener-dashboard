"""The run's time, per phase - provenance only (2026-10-26).

research/2026-10-26-data-run-timing.md: the 02:00 loop took 27 minutes and nothing said where.
``phase_timing.PhaseClock`` books the run into contiguous named laps, ``run_screener`` records
them in the run's ``meta.json`` and prints one line that the data loop's 25-line tail captures.
These tests pin the four things that make that record trustworthy and harmless: the laps add up
to the whole, the record never moves a key the gates read, the line is the last thing printed,
and nothing that scores or publishes reads it.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

import phase_timing
from phase_timing import PhaseClock

ROOT = Path(__file__).resolve().parent.parent


class _FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def test_laps_are_contiguous_and_add_up_to_the_total():
    fc = _FakeClock()
    c = PhaseClock(clock=fc)
    fc.advance(2.0)
    c.lap("universe")
    fc.advance(450.0)
    c.lap("fetch.tickers")
    fc.advance(0.3)
    c.lap("scoring")
    assert c.seconds == {"universe": 2.0, "fetch.tickers": 450.0, "scoring": pytest.approx(0.3)}
    assert sum(c.seconds.values()) == pytest.approx(c.total())
    assert c.total() == pytest.approx(452.3)


def test_a_phase_lapped_twice_accumulates():
    fc = _FakeClock()
    c = PhaseClock(clock=fc)
    fc.advance(5)
    c.lap("context")
    fc.advance(1)
    c.lap("scoring")
    fc.advance(7)
    c.lap("context")
    assert c.seconds["context"] == 12
    assert list(c.seconds) == ["context", "scoring"]   # first-lap order, not last


def test_meta_record_and_summary_line():
    fc = _FakeClock()
    c = PhaseClock(clock=fc)
    fc.advance(450.04)
    c.lap("fetch.tickers")
    c.note("fetch.tickers", n=503, workers=3, sum_ticker_seconds=1178.0)
    fc.advance(0.2)
    c.lap("scoring")
    fc.advance(0.3)
    c.lap("artifact.raw_fetch")
    fc.advance(181)
    c.lap("snapshot")
    meta = c.as_meta()
    assert meta["phase_seconds"] == {"fetch.tickers": 450.0, "scoring": 0.2,
                                     "artifact.raw_fetch": 0.3, "snapshot": 181.0}
    assert meta["phase_detail"]["fetch.tickers"]["n"] == 503
    assert meta["phase_total_seconds"] == pytest.approx(631.5, abs=0.1)
    json.dumps(meta)                                    # meta.json must be able to hold it
    line = c.summary_line()
    assert line.startswith("PHASE TIMING (s): total 632 | ")
    assert "fetch.tickers 450" in line and "snapshot 181" in line
    # sub-second phases are folded into one "other" figure, so the line stays one line
    assert "scoring" not in line and "raw_fetch" not in line
    assert line.endswith("| other 0")                   # 0.2 + 0.3 s
    assert "\n" not in line


def test_update_metadata_adds_keys_and_never_moves_one(tmp_path, monkeypatch):
    """The health check and the generator key on start_time / end_time / run_id; the timing
    record is added after they are written and must not be able to change them."""
    import run_context
    monkeypatch.setattr(run_context, "RUNS_DIR", tmp_path)
    ctx = run_context.RunContext(run_id="t1")
    try:
        ctx.save_metadata({"cli_flags": {"tickers": None}})
        before = json.loads((tmp_path / "t1" / "meta.json").read_text())
        ctx.phases.lap("fetch.tickers")
        ctx.update_metadata({**ctx.phases.as_meta(), "start_time": "1999-01-01T00:00:00",
                             "run_id": "other"})
        after = json.loads((tmp_path / "t1" / "meta.json").read_text())
    finally:
        for h in list(ctx.log.handlers):
            h.close()
            ctx.log.removeHandler(h)
    for k, v in before.items():
        assert after[k] == v, k
    assert "fetch.tickers" in after["phase_seconds"]
    assert set(after) - set(before) == {"phase_seconds", "phase_detail", "phase_total_seconds"}


def _main_body():
    tree = ast.parse((ROOT / "run_screener.py").read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main").body


def test_the_summary_is_the_last_line_main_prints():
    """scripts/data-run.ps1 copies only the screener's last 25 output lines into the loop's log;
    the timing line is only visible there if nothing is printed after it."""
    last = _main_body()[-1]
    src = ast.unparse(last)
    assert src.startswith("print(") and "summary_line()" in src, src
    ps1 = (ROOT / "scripts" / "data-run.ps1").read_text(encoding="utf-8")
    assert re.search(r"\$run\.Output \| Select-Object -Last 25", ps1), \
        "the tail the summary line relies on changed - re-check where the line must go"


def test_every_stage_of_a_full_run_is_lapped():
    src = (ROOT / "run_screener.py").read_text(encoding="utf-8")
    names = set(re.findall(r'clock\.lap\("([a-z_.]+)"\)', src))
    expected = {
        "universe", "fetch.probe", "fetch.risk_free", "fetch.market", "fetch.tickers",
        "fetch.retry", "fetch.sec_roe_history", "context.valuation_history",
        "context.options_pass", "context.sec_insider", "context.rate_sensitivity",
        "artifact.raw_fetch", "metrics", "checks", "scoring", "diagnostics",
        "score_cache_write", "portfolio", "excel", "dq_log_and_meta", "overview",
        "context.market_backdrop", "context.track_record", "context.signal_log",
        "context.eval", "dashboard", "snapshot",
    }
    assert expected <= names, sorted(expected - names)


# Modules that score, publish or hand the ranking to the bot. Timing is provenance about the
# machine, not about a stock: none of them may read it.
_MUST_NOT_READ = [
    "factor_engine.py", "portfolio_constructor.py", "generate_dashboard.py", "stock_summary.py",
    "calc_trace.py", "metric_lineage.py", "claims.py", "run_fingerprint.py", "history.py",
    "context_signals.py", "scripts/write_signals.py", "scripts/check_run_health.py",
]


@pytest.mark.parametrize("rel", _MUST_NOT_READ)
def test_no_scoring_or_publishing_module_reads_the_timing(rel):
    text = (ROOT / rel).read_text(encoding="utf-8")
    for token in ("phase_timing", "phase_seconds", "phase_detail", "phase_total_seconds",
                  "PhaseClock"):
        assert token not in text, f"{rel} reads {token}"


def test_module_has_no_scoring_imports():
    src = Path(phase_timing.__file__).read_text(encoding="utf-8")
    imported = {a.name.split(".")[0] for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(ast.parse(src))
                 if isinstance(n, ast.ImportFrom) and n.module}
    assert imported <= {"__future__", "time", "typing"}, imported
