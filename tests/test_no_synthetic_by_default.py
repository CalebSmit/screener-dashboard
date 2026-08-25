"""The screener must refuse to fabricate data when its data source is down.

2026-08-06: the 02:00 data run executed with no network. `run_factor_engine`
caught the failed connectivity probe, set `USE_SAMPLE = True`, and generated
"sector-realistic sample values" for all 503 tickers. It then produced a
completely normal-looking 2.6 MB dashboard payload reporting
`stocks_scored: 503, avg_composite: 50.5`, and committed it. The only reason
invented stock scores did not reach the public site is that the push failed on
the same dead network.

Nothing in the output distinguished it from a real run. The size checks passed
- a fully synthetic run produces a normally-sized payload - and the run
reported "0 issues" at the summary level. The single tell was buried in
`validation/data_quality_log.csv`.

`scripts/data-run.ps1` and `scripts/check_run_health.py` gate on this
downstream, but that only protects the scheduled loop; anyone invoking the
screener directly still got fiction. These tests pin the refusal at source.
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_SCREENER = ROOT / "run_screener.py"
CLI = ROOT / "cli.py"


def _source(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_opt_in_flag_is_on_the_parser_that_actually_runs():
    """run_screener.py defines its own parse_args(); cli.py is not used by it.

    cli.py has a near-identical parser imported only by tests/test_cli.py.
    A flag added there alone is inert - `args.allow_synthetic` would never
    exist, and the getattr default would silently refuse every run including
    the deliberate opt-in. Pin the flag to the live parser.
    """
    for path in (RUN_SCREENER, CLI):
        src = _source(path)
        assert "--allow-synthetic" in src, f"{path.name} must expose --allow-synthetic"
        m = re.search(r'add_argument\(\s*"--allow-synthetic"([^)]*)\)', src, re.S)
        assert m, f"could not parse the --allow-synthetic argument in {path.name}"
        assert 'action="store_true"' in m.group(1), (
            f"{path.name}: --allow-synthetic must be store_true so it defaults False"
        )


def test_the_live_parser_defaults_the_flag_off():
    """Parse an empty argv through the real parser and check the default."""
    import subprocess, sys
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0,'.'); sys.argv=['run_screener.py'];"
         "import run_screener as r; a=r.parse_args(); print(a.allow_synthetic)"],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, f"parse_args failed: {out.stderr[-400:]}"
    assert out.stdout.strip() == "False", (
        f"--allow-synthetic must default to False, got {out.stdout.strip()!r}"
    )


def test_synthetic_path_is_guarded():
    """USE_SAMPLE must not be reachable without the explicit opt-in."""
    src = _source(RUN_SCREENER)
    i = src.index("USE_SAMPLE = True")
    guard_region = src[max(0, i - 1800):i]
    assert "allow_synthetic" in guard_region, (
        "USE_SAMPLE = True is set without checking allow_synthetic first - "
        "a failed fetch would silently fabricate data again"
    )
    assert "SystemExit(2)" in guard_region, (
        "the un-opted-in path must exit non-zero rather than continue"
    )


def test_refusal_precedes_fabrication():
    """The exit must come before USE_SAMPLE, not after."""
    src = _source(RUN_SCREENER)
    exit_pos = src.index("SystemExit(2)")
    sample_pos = src.index("USE_SAMPLE = True")
    assert exit_pos < sample_pos, (
        "SystemExit(2) appears after USE_SAMPLE = True, so fabrication happens "
        "before the refusal can take effect"
    )


def test_fabricated_output_is_labelled_when_opted_in():
    """Even the opt-in path must say loudly that the output is not real."""
    src = _source(RUN_SCREENER)
    i = src.index("USE_SAMPLE = True")
    region = src[max(0, i - 700):i]
    assert "FABRICATED" in region.upper(), (
        "the --allow-synthetic path must state plainly that its output is "
        "fabricated; the 08-06 run printed only 'Generating sector-realistic "
        "sample data', which reads like a normal pipeline step"
    )


def test_run_screener_still_parses():
    """A syntax error here means the 02:00 loop does not run at all."""
    ast.parse(_source(RUN_SCREENER))
    ast.parse(_source(CLI))
