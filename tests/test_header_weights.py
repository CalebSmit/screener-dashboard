"""The rankings table's column tooltips state the weights the run used - read from the run's
published weight tables, never typed into the page.

2026-10-09: removing operating leverage from Quality left the Quality tooltip saying
"operating leverage (8%)" - all six category tooltips carried weights typed by hand, so any
weight change silently made the page disagree with the engine (CLAUDE.md row 0.8c).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_dashboard as gd  # noqa: E402

CATS = ("valuation", "quality", "growth", "momentum", "risk", "revisions")


def _headers(html: str) -> dict:
    out = {}
    for th in re.findall(r"<th [^>]*data-wcat=[^>]*>", html):
        out[re.search(r'data-wcat="([a-z]+)"', th).group(1)] = re.search(r'title="([^"]*)"', th).group(1)
    return out


def test_every_weighted_category_header_takes_its_weights_from_the_payload():
    heads = _headers(gd.generate_html())
    assert set(heads) == set(CATS)
    for cat, title in heads.items():
        assert "@W@" in title, cat
        assert not re.search(r"\(\d+(\.\d+)?%\)", title), f"{cat}: a weight is typed into the page: {title}"


def test_the_bank_clause_is_filled_where_banks_have_their_own_table():
    heads = _headers(gd.generate_html())
    assert "@BANK@" in heads["valuation"] and "@BANK@" in heads["quality"]
    js = gd.generate_html()
    assert "weightList(P.generic)" in js and "weightList(P.bank)" in js
