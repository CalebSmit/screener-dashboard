"""Every universe-size constant reads ``universe.py`` (phase 1 of plan/universe-expansion.md, 2026-11-03).

The S&P 500's size used to be literal in each place that checked it, so a second universe would
have needed every copy found by hand. These pin that each reads the one definition, that the
values did not move (the phase's rule: nothing published changes), and that the literals left on
purpose are still listed - so finishing them is a visible step, not a forgotten one.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import universe  # noqa: E402


def test_the_sp500_definition_is_unchanged():
    u = universe.SP500
    assert (u.min_members, u.max_members, u.min_ranked) == (495, 515, 490)
    assert universe.CURRENT is universe.SP500
    assert u.min_ranked <= u.min_members <= u.max_members


def test_every_size_constant_reads_the_definition():
    import check_run_health
    import rehearse_publish
    import universe_history
    u = universe.CURRENT
    assert check_run_health.MIN_SCORED == u.min_ranked
    assert rehearse_publish.MIN_UNIVERSE == u.min_members
    assert (universe_history.MIN_PLAUSIBLE_MEMBERS, universe_history.MAX_PLAUSIBLE_MEMBERS) == (
        u.min_members, u.max_members)


_ASSIGN = re.compile(r"^\s*(MIN_SCORED|MIN_UNIVERSE|MIN_PLAUSIBLE_MEMBERS|MAX_PLAUSIBLE_MEMBERS)\s*=\s*\d")


def test_no_module_writes_the_size_as_a_literal_again():
    hits = []
    for p in [*ROOT.glob("*.py"), *ROOT.joinpath("scripts").glob("*.py")]:
        for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if _ASSIGN.match(line):
                hits.append(f"{p.name}:{n}: {line.strip()}")
    assert not hits, hits


# Left literal on purpose on 2026-11-03, and named here so finishing them is a visible step.
REMAINING = (
    ("scripts/data-run.ps1", "$universe = 500"),
)


def test_the_page_reads_no_literal_universe_size():
    """The rank chart's fallback was ``D.table_data.length || 500`` (closed 2026-11-03 night)."""
    src = (ROOT / "generate_dashboard.py").read_text(encoding="utf-8")
    assert "|| 500" not in src


def test_the_literals_left_for_later_are_still_where_the_plan_says():
    for rel, text in REMAINING:
        src = (ROOT / rel).read_text(encoding="utf-8-sig", errors="replace")
        assert text in src, f"{rel} no longer contains {text!r}: drop it from REMAINING and universe.py's note"
