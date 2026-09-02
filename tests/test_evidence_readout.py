"""The data loop's daily evidence readout must quote EFFECTIVE observations.

Why this file exists
--------------------
``CLAUDE.md`` rule 8 requires every session to record the evidence base, and
says explicitly that quoting the raw row count instead of the effective
(non-overlapping) count "is how this went wrong the first time".

The 2026-09-02 02:00 data run logged::

    Improvement engine: 30 raw IC row(s) (effective count unavailable).

That is the fallback branch.  The primary path - an inline ``python -c``
here-string in ``scripts/data-run.ps1`` - had failed, and the fallback
substituted the raw row count.  So the single number the loop prints daily,
the one read without thinking, was the misleading one.  At that moment the
true readout was **3 effective** observations at the ``1m`` optimization
horizon against a gate of 8; "30" reads as though the gate were long cleared.

The logic could not be tested while it lived in a here-string.  It is now
``scripts/report_evidence.py``, and these tests are the point of moving it.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import report_evidence  # noqa: E402

DATA_RUN = ROOT / "scripts" / "data-run.ps1"


# ---------------------------------------------------------------------------
# The readout itself
# ---------------------------------------------------------------------------

def test_readout_leads_with_the_effective_count():
    line = report_evidence.build_readout()
    assert "effective" in line
    # The effective count must come before the row count in the sentence -
    # whichever number is read first is the one that sticks.
    assert line.index("effective") < line.index("rows")


def test_readout_names_the_optimization_horizon_and_the_gate():
    line = report_evidence.build_readout()
    assert "horizon" in line
    assert "propose a weight change" in line


def test_readout_reports_both_numbers_so_the_gap_is_visible():
    """Effective AND raw, because the gap between them is the whole point."""
    import improvement_engine as ie
    import pandas as pd

    ic = pd.read_csv(ie.LIVE_IC_HISTORY_PATH)
    horizon = ie._get_governance_config()["optimization_horizon"]
    at_horizon = ic.loc[ic["horizon"].astype(str) == horizon, "run_date"]
    effective = ie._effective_observations(at_horizon, horizon)

    line = report_evidence.build_readout()
    assert line.startswith(f"{effective} effective")
    assert f"({len(at_horizon)} rows)" in line


def test_effective_count_does_not_exceed_the_row_count():
    """Non-overlapping observations can never outnumber the rows they come
    from. A regression that inverted the two would be caught here."""
    import improvement_engine as ie
    import pandas as pd

    ic = pd.read_csv(ie.LIVE_IC_HISTORY_PATH)
    horizon = ie._get_governance_config()["optimization_horizon"]
    at_horizon = ic.loc[ic["horizon"].astype(str) == horizon, "run_date"]
    assert ie._effective_observations(at_horizon, horizon) <= len(at_horizon)


def test_script_runs_standalone_and_exits_zero():
    """It is invoked as a bare path from PowerShell, so it must work with no
    arguments, from any working directory, without the repo on sys.path."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "report_evidence.py")],
        capture_output=True, text=True, cwd=str(ROOT.parent),
    )
    assert proc.returncode == 0, proc.stderr
    assert "effective" in proc.stdout


def test_failure_does_not_fall_back_to_a_raw_count(monkeypatch, capsys):
    """The defect being fixed: on failure the loop must NOT print a row count.

    An authoritative-looking wrong number is worse than an admitted gap.
    """
    monkeypatch.setattr(
        report_evidence, "build_readout",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert report_evidence.main() == 1
    captured = capsys.readouterr()
    assert "unavailable" in captured.err
    assert captured.out.strip() == ""


# ---------------------------------------------------------------------------
# The caller
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data_run_source():
    return DATA_RUN.read_text(encoding="utf-8-sig")


def test_data_run_invokes_the_script_not_an_inline_snippet(data_run_source):
    assert "scripts\\report_evidence.py" in data_run_source
    assert "_effective_observations" not in data_run_source, (
        "the readout logic moved into scripts/report_evidence.py so it could "
        "be tested; it must not be inlined back into the PowerShell script"
    )


def test_data_run_no_longer_reports_a_raw_row_count(data_run_source):
    """Fails against the pre-2026-09-02 script."""
    assert "raw IC row(s)" not in data_run_source
    assert "Measure-Object -Line" not in data_run_source.split(
        "Report evidence accumulation")[-1].split("SuccessMarker")[0]


def test_data_run_failure_branch_is_a_warning(data_run_source):
    """A silent INFO line is how this went unnoticed for a day."""
    tail = data_run_source.split("Report evidence accumulation")[-1]
    block = tail.split("SuccessMarker")[0]
    assert "'WARN'" in block
