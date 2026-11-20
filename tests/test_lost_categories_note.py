"""The health check names a top-40 stock that lost a category since the last published run.

2026-11-19: SanDisk's price history failed to download, it rose 7th -> 1st, and the brief said
nothing. `scripts/check_run_health.lost_categories`. A note, never a refusal.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import check_run_health as h  # noqa: E402

CATS = h.CATEGORY_KEYS


def _row(t, rank, missing=()):
    r = {"Ticker": t, "Rank": rank}
    r.update({f"{c}_score": (None if c in missing else 50.0) for c in CATS})
    return r


def test_a_top_stock_that_lost_two_categories_is_named():
    today = {"table_data": [_row("SNDK", 1, ("momentum", "risk")), _row("EXPE", 2)]}
    before = {"table_data": [_row("SNDK", 7), _row("EXPE", 1)]}
    assert h.lost_categories(today, before) == ["SNDK (#1: momentum, risk)"]


def test_a_gap_it_already_had_or_a_stock_below_the_top_is_not():
    today = {"table_data": [_row("A", 3, ("investment",)), _row("B", 41, ("momentum",)),
                            _row("NEW", 5, ("risk",))]}
    before = {"table_data": [_row("A", 3, ("investment",)), _row("B", 30)]}
    assert h.lost_categories(today, before) == []


def test_no_published_baseline_says_nothing_rather_than_guessing():
    assert h.lost_categories({"table_data": [_row("X", 1, ("risk",))]}, None) == []


def test_the_note_is_wired_into_main_and_never_refuses():
    import inspect
    src = inspect.getsource(h.main)
    assert "check_lost_categories(payload, published, res)" in src
    body = inspect.getsource(h.check_lost_categories)
    assert "res.note(" in body and "res.fail" not in body
