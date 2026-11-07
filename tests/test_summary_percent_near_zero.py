"""The summary sentences print a percentage near zero by the page's own rule (2026-11-06).

``stock_summary._fmt_metric`` mirrors the page's ``fmtMetric``, but when the JS formatters were
brought under one near-zero rule (``pctNearZero``, 2026-11-03, ``tests/test_percent_near_zero.py``)
the Python mirror was not. The 18:34 data run on 2026-11-06 then published LNT's summary as
"... the 60th on 12-1M Return (-0.0%)." and turned gate 1 red with no code change
(``tests/test_every_panel_reads_clean.py``). These cases are the JS rule's, worked by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stock_summary as ss  # noqa: E402

CASES = [
    (-0.0004, "-0.04%"),     # LNT's shape: not zero, so one more decimal, sign kept
    (0.0004, "0.04%"),
    (-0.00001, "0.0%"),      # still zero at two decimals: zero to the reader, unsigned
    (0.0, "0.0%"),
    (-0.0, "0.0%"),
    (0.123, "12.3%"),        # ordinary values are unchanged
    (-0.05, "-5.0%"),
    (-0.0003, "-0.03%"),
]


@pytest.mark.parametrize("value, want", CASES)
def test_a_percentage_near_zero_follows_the_pages_rule(value, want):
    assert ss._fmt_metric(value, "pct") == want


def test_no_summary_sentence_can_print_a_signed_zero():
    for v in (-1e-9, -0.00004, -0.0000499, -0.00049):
        assert not ss._fmt_metric(v, "pct").startswith("-0.0%"), v
        assert "-0.00%" not in ss._fmt_metric(v, "pct"), v
