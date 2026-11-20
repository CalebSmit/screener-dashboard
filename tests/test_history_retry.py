"""A fetch whose price history came back empty is retried once (2026-11-19, owner-run review).

On 2026-11-19 SanDisk's `Ticker.history()` alone came back empty. That is not an `_error`, so the
existing retry pass never saw it; its Momentum and Risk categories dropped out, the composite
renormalised over the other six, and it went from #7 (71.02) to #1 (79.09) on the public page and
in `signals.json`. `research/2026-11-19-fresh-eyes-review.md`, finding 1.
"""
import math

import run_screener as rs


def _rec(t, price=100.0, **kw):
    r = {"Ticker": t, "price_latest": price}
    r.update(kw)
    return r


def test_selects_only_successful_fetches_without_history():
    raw = [
        _rec("AAA"),
        _rec("SNDK", price=float("nan")),
        {"Ticker": "NOP"},                       # key absent altogether
        _rec("ERR", price=float("nan"), _error="boom"),  # the error pass owns this one
        _rec("NONE", price=None),
    ]
    assert rs.missing_price_history(raw) == ["SNDK", "NOP", "NONE"]


def test_nothing_missing_means_nothing_to_retry():
    assert rs.missing_price_history([_rec("A"), _rec("B")]) == []


def test_merge_takes_a_refetch_only_when_it_has_history():
    raw = [_rec("AAA"), _rec("SNDK", price=float("nan"), marker="old"),
           _rec("KVUE", price=float("nan"), marker="old")]
    retry = [_rec("SNDK", price=776.22, marker="new"),
             _rec("KVUE", price=float("nan"), marker="new")]
    merged, got = rs.merge_recovered_history(raw, retry)
    by = {r["Ticker"]: r for r in merged}
    assert got == ["SNDK"]
    assert by["SNDK"]["marker"] == "new" and by["SNDK"]["price_latest"] == 776.22
    # Still empty on the retry: the original record stays, never swapped for an equal or worse one.
    assert by["KVUE"]["marker"] == "old" and math.isnan(by["KVUE"]["price_latest"])
    assert by["AAA"]["price_latest"] == 100.0
    assert [r["Ticker"] for r in merged] == ["AAA", "SNDK", "KVUE"]  # order kept


def test_a_refetch_that_errors_is_ignored():
    raw = [_rec("SNDK", price=float("nan"), marker="old")]
    merged, got = rs.merge_recovered_history(raw, [_rec("SNDK", price=5.0, _error="429")])
    assert got == [] and merged[0]["marker"] == "old"


def test_the_retry_is_wired_into_the_fetch_and_bounded():
    """The pass runs after the error retry and is skipped for an outage-sized gap."""
    import inspect
    src = inspect.getsource(rs.run_factor_engine)
    assert "missing_price_history(raw)" in src
    assert "merge_recovered_history(raw, hist_raw)" in src
    assert src.index("Retry pass:") < src.index("History retry:")
    assert 0 < rs.HISTORY_RETRY_MAX_SHARE <= 0.10
