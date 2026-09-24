"""Tests for point-in-time S&P 500 membership reconstruction.

No test here touches the network. The module is deliberately split so that
everything with judgement in it — parsing, validation, staleness, the
survivorship arithmetic — is a pure function over text, and only the thin
fetch wrappers need a live connection.

The tests that matter most are the ones pinning a *specific* failure the
reconstruction could have had and a reader would not have noticed:

* the constituent table's column layout changed mid-window — the "SEC filings"
  column was dropped in 2023, shifting everything after it — so any positional
  assumption about the table is unsafe. (The symbol column itself stayed at
  position 0 throughout; that was checked, not assumed, and the test below
  says so rather than overclaiming a bug that did not happen.)
* the same page carried a 269-row "selected changes" table that a
  find-any-table parser would happily return as a universe;
* a short or vandalised parse looks exactly like a real index contraction
  unless something refuses it.
"""

import json

import pytest

import universe_history as uh


# ---------------------------------------------------------------------------
# Fixtures: the two real table layouts, reduced to a few rows each
# ---------------------------------------------------------------------------
def _rows(symbols, extra_cells):
    return "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in [s] + extra_cells) + "</tr>"
        for s in symbols
    )


def pre2023_html(symbols):
    """Layout used through 2022: Symbol, Security, SEC filings, GICS Sector."""
    return (
        "<table><tr><th>Symbol</th><th>Security</th><th>SEC filings</th>"
        "<th>GICS Sector</th></tr>"
        + _rows(symbols, ["Some Company Inc.", "reports", "Industrials"])
        + "</table>"
    )


def post2023_html(symbols):
    """Layout from 2023: the SEC filings column was dropped."""
    return (
        "<table><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th>"
        "<th>GICS Sub-Industry</th></tr>"
        + _rows(symbols, ["Some Company Inc.", "Industrials", "Conglomerates"])
        + "</table>"
    )


def changes_table_html():
    """The 'selected changes' table: a two-row header, hence MultiIndex."""
    return (
        "<table>"
        "<thead>"
        "<tr><th colspan='1'>Date</th><th colspan='2'>Added</th>"
        "<th colspan='2'>Removed</th><th colspan='1'>Reason</th></tr>"
        "<tr><th>Date</th><th>Ticker</th><th>Security</th>"
        "<th>Ticker</th><th>Security</th><th>Reason</th></tr>"
        "</thead><tbody>"
        "<tr><td>June 4, 2021</td><td>ABC</td><td>A Corp</td>"
        "<td>XYZ</td><td>X Corp</td><td>Acquired</td></tr>"
        "</tbody></table>"
    )


def universe(n, prefix="AA"):
    """n distinct plausible tickers."""
    out = []
    for i in range(n):
        out.append(f"{prefix}{i:03d}")
    return out


# ---------------------------------------------------------------------------
# normalize_ticker
# ---------------------------------------------------------------------------
def test_dot_becomes_dash_matching_factor_engine():
    # factor_engine.get_sp500_tickers does .str.replace(".", "-"); if this
    # module disagreed, every share-class name would look like a delisting.
    assert uh.normalize_ticker("BRK.B") == "BRK-B"
    assert uh.normalize_ticker("BF.B") == "BF-B"


def test_normalisation_matches_factor_engine_on_the_real_list():
    import factor_engine

    fallback = json.loads((uh.ROOT / "sp500_tickers.json").read_text())
    raw = [r["Ticker"] for r in fallback]
    # factor_engine's own rule, applied to the same strings
    expected = [t.replace(".", "-") for t in raw]
    assert [uh.normalize_ticker(t) for t in raw] == expected
    assert factor_engine is not None


def test_footnote_markers_are_stripped():
    assert uh.normalize_ticker("BF.B[1]") == "BF-B"
    assert uh.normalize_ticker("AAPL[note 2]") == "AAPL"


def test_whitespace_and_nbsp_are_stripped():
    assert uh.normalize_ticker("\xa0AAPL ") == "AAPL"
    assert uh.normalize_ticker("  MSFT\n") == "MSFT"


def test_case_is_normalised():
    assert uh.normalize_ticker("aapl") == "AAPL"


# ---------------------------------------------------------------------------
# _symbol_column — the layout-change defect
# ---------------------------------------------------------------------------
def test_symbol_column_found_in_both_real_layouts():
    import io

    import pandas as pd

    for html in (pre2023_html(["AAPL"]), post2023_html(["AAPL"])):
        frame = pd.read_html(io.StringIO(html))[0]
        assert uh._symbol_column(frame) == "Symbol"


def test_column_layout_is_not_stable_across_the_window():
    """Why the parser keys on name — and the limit of that claim.

    Position 2 is 'SEC filings' before 2023 and 'GICS Sector' after, so any
    positional assumption about the table is unsafe. The *symbol* column
    happens to sit at position 0 in both layouts, so a position-0 parser would
    not have been wrong; this test records both facts so a later reader does
    not inherit an overstated justification.
    """
    import io

    import pandas as pd

    pre = pd.read_html(io.StringIO(pre2023_html(["AAPL"])))[0]
    post = pd.read_html(io.StringIO(post2023_html(["AAPL"])))[0]
    assert list(pre.columns)[2] != list(post.columns)[2]   # layout moved
    assert list(pre.columns)[0] == list(post.columns)[0]   # but symbol did not
    assert uh._symbol_column(pre) == uh._symbol_column(post) == "Symbol"


def test_symbol_column_accepts_older_header_wording():
    import io

    import pandas as pd

    html = "<table><tr><th>Ticker symbol</th><th>Security</th></tr>" \
           "<tr><td>AAPL</td><td>Apple</td></tr></table>"
    frame = pd.read_html(io.StringIO(html))[0]
    assert uh._symbol_column(frame) == "Ticker symbol"


def test_symbol_column_returns_none_when_absent():
    import io

    import pandas as pd

    html = "<table><tr><th>Date</th><th>Reason</th></tr>" \
           "<tr><td>2021-01-01</td><td>Acquired</td></tr></table>"
    frame = pd.read_html(io.StringIO(html))[0]
    assert uh._symbol_column(frame) is None


# ---------------------------------------------------------------------------
# parse_constituents
# ---------------------------------------------------------------------------
def test_parses_pre2023_layout():
    assert uh.parse_constituents(pre2023_html(["AAPL", "BRK.B"])) == ["AAPL", "BRK-B"]


def test_parses_post2023_layout():
    assert uh.parse_constituents(post2023_html(["AAPL", "BRK.B"])) == ["AAPL", "BRK-B"]


def test_changes_table_is_not_mistaken_for_a_universe():
    """A first-table-wins parser returns 269 rows of index *changes* here."""
    html = changes_table_html() + post2023_html(["AAPL", "MSFT"])
    assert uh.parse_constituents(html) == ["AAPL", "MSFT"]


def test_constituent_table_found_when_it_is_not_first():
    html = ("<table><tr><th>Date</th></tr><tr><td>2021-01-01</td></tr></table>"
            + post2023_html(["AAPL"]))
    assert uh.parse_constituents(html) == ["AAPL"]


def test_no_symbol_column_anywhere_raises():
    html = "<table><tr><th>Date</th></tr><tr><td>2021-01-01</td></tr></table>"
    with pytest.raises(uh.UniverseParseError):
        uh.parse_constituents(html)


def test_no_tables_at_all_raises():
    with pytest.raises(uh.UniverseParseError):
        uh.parse_constituents("<html><body><p>nothing here</p></body></html>")


def test_blank_cells_are_dropped_not_kept_as_nan():
    html = ("<table><tr><th>Symbol</th><th>Security</th></tr>"
            "<tr><td>AAPL</td><td>Apple</td></tr>"
            "<tr><td></td><td>Orphan row</td></tr></table>")
    assert uh.parse_constituents(html) == ["AAPL"]


# ---------------------------------------------------------------------------
# parse_ticker_ciks — separating renames from removals
# ---------------------------------------------------------------------------
def cik_html(pairs):
    rows = "".join(
        f"<tr><td>{s}</td><td>A Corp</td><td>{c}</td></tr>" for s, c in pairs
    )
    return ("<table><tr><th>Symbol</th><th>Security</th><th>CIK</th></tr>"
            + rows + "</table>")


def test_ciks_are_parsed_by_ticker():
    assert uh.parse_ticker_ciks(cik_html([("MMM", 66740), ("ABT", 1800)])) == {
        "MMM": 66740, "ABT": 1800}


def test_zero_padded_ciks_compare_equal():
    """Padding varies across revisions; a padded CIK is the same registrant."""
    a = uh.parse_ticker_ciks(cik_html([("MMM", "0000066740")]))
    b = uh.parse_ticker_ciks(cik_html([("MMM", "66740")]))
    assert a == b == {"MMM": 66740}


def test_cik_keys_are_normalised_like_tickers():
    assert "BRK-B" in uh.parse_ticker_ciks(cik_html([("BRK.B", 1067983)]))


def test_missing_cik_column_returns_empty_not_an_error():
    """Callers degrade to the gross figure knowingly rather than crashing."""
    assert uh.parse_ticker_ciks(post2023_html(["AAPL"])) == {}


def test_unparseable_cik_rows_are_skipped_not_guessed():
    out = uh.parse_ticker_ciks(cik_html([("MMM", 66740), ("BAD", "n/a")]))
    assert out == {"MMM": 66740}


def test_changes_table_is_skipped_when_looking_for_ciks():
    html = changes_table_html() + cik_html([("MMM", 66740)])
    assert uh.parse_ticker_ciks(html) == {"MMM": 66740}


def test_cik_identifies_a_rename_that_symbol_and_name_both_miss():
    """The case the decomposition exists for.

    Anthem/ANTM became Elevance Health/ELV: ticker changed, company name
    changed, registrant did not. Only the CIK links them.
    """
    then = uh.parse_ticker_ciks(cik_html([("ANTM", 1156039)]))
    now = uh.parse_ticker_ciks(cik_html([("ELV", 1156039)]))
    assert then["ANTM"] == now["ELV"]
    assert set(then) & set(now) == set()  # no symbol overlap at all


def test_html_with_no_tables_returns_empty_ciks():
    assert uh.parse_ticker_ciks("<p>nothing</p>") == {}


# ---------------------------------------------------------------------------
# validate_membership
# ---------------------------------------------------------------------------
def test_real_sized_universe_passes():
    uh.validate_membership(universe(503), as_of="2024-09-30")
    uh.validate_membership(universe(505), as_of="2020-06-30")


def test_the_269_row_changes_table_is_refused():
    """The specific number the wrong table would have produced."""
    with pytest.raises(uh.ImplausibleUniverseError) as exc:
        uh.validate_membership(universe(269), as_of="2021-03-31")
    assert "269" in str(exc.value)


def test_truncated_parse_is_refused():
    with pytest.raises(uh.ImplausibleUniverseError):
        uh.validate_membership(universe(120))


def test_empty_parse_is_refused():
    with pytest.raises(uh.ImplausibleUniverseError):
        uh.validate_membership([])


def test_oversized_parse_is_refused():
    # Two tables concatenated, or a page carrying a different index
    with pytest.raises(uh.ImplausibleUniverseError):
        uh.validate_membership(universe(1006))


def test_band_edges_are_inclusive():
    uh.validate_membership(universe(uh.MIN_PLAUSIBLE_MEMBERS))
    uh.validate_membership(universe(uh.MAX_PLAUSIBLE_MEMBERS))
    with pytest.raises(uh.ImplausibleUniverseError):
        uh.validate_membership(universe(uh.MIN_PLAUSIBLE_MEMBERS - 1))
    with pytest.raises(uh.ImplausibleUniverseError):
        uh.validate_membership(universe(uh.MAX_PLAUSIBLE_MEMBERS + 1))


def test_duplicates_are_refused():
    dupes = universe(503)
    dupes[7] = dupes[0]
    with pytest.raises(uh.ImplausibleUniverseError) as exc:
        uh.validate_membership(dupes, as_of="2022-01-31")
    assert dupes[0] in str(exc.value)


def test_error_message_says_it_is_a_parse_failure_not_an_index_change():
    """A short universe looks like a real contraction; the message must not."""
    with pytest.raises(uh.ImplausibleUniverseError) as exc:
        uh.validate_membership(universe(300))
    assert "not an index change" in str(exc.value)


# ---------------------------------------------------------------------------
# staleness and calendar
# ---------------------------------------------------------------------------
def test_staleness_counts_whole_days():
    assert uh._staleness_days("2021-03-31", "2021-03-29T19:48:21Z") == 2
    assert uh._staleness_days("2020-01-31", "2020-01-31T02:38:19Z") == 0


def test_staleness_of_a_week_old_revision():
    # The real worst case observed in the window: 2023-03-31 -> 2023-03-24
    assert uh._staleness_days("2023-03-31", "2023-03-24T20:44:59Z") == 7


def test_month_ends_are_month_ends():
    ends = uh.month_ends("2020-01-01", "2020-04-15")
    assert ends == ["2020-01-31", "2020-02-29", "2020-03-31"]


def test_month_ends_includes_leap_day_february():
    assert "2020-02-29" in uh.month_ends("2020-01-01", "2020-12-31")
    assert "2021-02-28" in uh.month_ends("2021-01-01", "2021-12-31")


def test_month_ends_excludes_an_incomplete_final_month():
    # 2026-09-24 is mid-month; September must not appear as a month-end
    assert "2026-09-30" not in uh.month_ends("2026-01-01", "2026-09-24")


# ---------------------------------------------------------------------------
# UniverseSnapshot
# ---------------------------------------------------------------------------
def _snap(as_of="2021-03-31", tickers=("AAPL", "MSFT"), revid=1014924736):
    return uh.UniverseSnapshot(
        as_of=as_of, tickers=tuple(tickers), revid=revid,
        revision_timestamp="2021-03-29T19:48:21Z", staleness_days=2,
    )


def test_snapshot_reports_its_size_and_source():
    s = _snap()
    assert s.n_members == 2
    assert str(s.revid) in s.source_url
    assert s.source_url.startswith("https://en.wikipedia.org/")


def test_snapshot_round_trips_through_dict():
    s = _snap()
    assert uh.UniverseSnapshot.from_dict(s.to_dict()) == s


def test_snapshot_dict_carries_provenance_not_just_tickers():
    d = _snap().to_dict()
    for key in ("revid", "revision_timestamp", "staleness_days", "source_url"):
        assert key in d, f"{key} missing — provenance is a defensibility feature"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def test_cache_round_trip(tmp_path):
    path = tmp_path / "u.json"
    snaps = {"2021-03-31": _snap()}
    uh.save_cache(snaps, path)
    loaded = uh.load_cache(path)
    assert loaded == snaps


def test_cache_is_written_in_date_order_so_it_diffs_cleanly(tmp_path):
    path = tmp_path / "u.json"
    uh.save_cache({
        "2021-06-30": _snap("2021-06-30"),
        "2020-01-31": _snap("2020-01-31"),
        "2021-03-31": _snap("2021-03-31"),
    }, path)
    dates = [s["as_of"] for s in json.loads(path.read_text())["snapshots"]]
    assert dates == sorted(dates)


def test_missing_cache_file_reads_as_empty(tmp_path):
    assert uh.load_cache(tmp_path / "absent.json") == {}


def test_wrong_schema_version_is_refused(tmp_path):
    path = tmp_path / "u.json"
    path.write_text(json.dumps({"schema": 99, "snapshots": []}))
    with pytest.raises(uh.UniverseHistoryError):
        uh.load_cache(path)


def test_cache_records_the_regeneration_command(tmp_path):
    path = tmp_path / "u.json"
    uh.save_cache({"2021-03-31": _snap()}, path)
    assert "--refresh" in json.loads(path.read_text())["note"]


# ---------------------------------------------------------------------------
# members_on
# ---------------------------------------------------------------------------
def test_cache_hit_does_not_touch_the_network(tmp_path, monkeypatch):
    path = tmp_path / "u.json"
    uh.save_cache({"2021-03-31": _snap()}, path)

    def boom(*a, **k):
        raise AssertionError("network used despite a cache hit")

    monkeypatch.setattr(uh, "_session", boom)
    monkeypatch.setattr(uh, "fetch_snapshot", boom)
    assert uh.members_on("2021-03-31", cache_path=path).n_members == 2


def test_cache_miss_offline_raises_with_the_fix_in_the_message(tmp_path):
    with pytest.raises(uh.UniverseHistoryError) as exc:
        uh.members_on("2021-03-31", cache_path=tmp_path / "u.json",
                      allow_network=False)
    assert "--refresh" in str(exc.value)


def test_cache_miss_offline_names_the_date(tmp_path):
    with pytest.raises(uh.UniverseHistoryError) as exc:
        uh.members_on("2022-07-29", cache_path=tmp_path / "u.json",
                      allow_network=False)
    assert "2022-07-29" in str(exc.value)


# ---------------------------------------------------------------------------
# survivorship_gap — the arithmetic the measurement rests on
# ---------------------------------------------------------------------------
def test_gap_names_the_companies_todays_list_deletes():
    snap = _snap(tickers=("AAPL", "MSFT", "XYZ_GONE"))
    gap = uh.survivorship_gap(snap, ["AAPL", "MSFT", "NEWCO"])
    assert gap["dropped"] == ["XYZ_GONE"]
    assert gap["n_dropped"] == 1
    assert gap["n_survivors"] == 2
    assert gap["n_then"] == 3
    assert gap["pct_dropped"] == pytest.approx(100 / 3)


def test_gap_is_zero_when_nothing_left_the_index():
    snap = _snap(tickers=("AAPL", "MSFT"))
    gap = uh.survivorship_gap(snap, ["AAPL", "MSFT", "NEWCO"])
    assert gap["n_dropped"] == 0
    assert gap["pct_dropped"] == 0.0


def test_gap_ignores_additions_because_they_are_not_survivorship():
    """Names added *after* the date are not a survivorship problem.

    A backtest built on today's list wrongly includes them at that date, which
    is look-ahead, not survivorship. This function measures one thing.
    """
    snap = _snap(tickers=("AAPL",))
    gap = uh.survivorship_gap(snap, ["AAPL", "NEW1", "NEW2", "NEW3"])
    assert gap["n_dropped"] == 0
    assert gap["n_then"] == 1


def test_gap_handles_an_empty_universe_without_dividing_by_zero():
    snap = _snap(tickers=())
    assert uh.survivorship_gap(snap, ["AAPL"])["pct_dropped"] == 0.0


# ---------------------------------------------------------------------------
# The committed artifact — guards a future refresh from committing garbage
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def committed():
    if not uh.CACHE_PATH.exists():
        pytest.skip(f"{uh.CACHE_PATH} not present")
    return uh.load_cache(uh.CACHE_PATH)


def test_committed_cache_covers_the_backtest_window(committed):
    assert "2020-01-31" in committed, "backtest starts 2020-01-01"
    assert len(committed) >= 70, f"only {len(committed)} month-ends cached"


def test_every_committed_snapshot_is_in_band(committed):
    for as_of, snap in committed.items():
        uh.validate_membership(snap.tickers, as_of=as_of)


def test_every_committed_snapshot_carries_a_revision_id(committed):
    for as_of, snap in committed.items():
        assert snap.revid > 0, f"{as_of} has no revision id to check against"


def test_no_committed_snapshot_is_unacceptably_stale(committed):
    stale = {a: s.staleness_days for a, s in committed.items()
             if s.staleness_days > uh.MAX_STALENESS_DAYS}
    assert not stale, f"revisions older than the date they describe: {stale}"


def test_staleness_is_never_negative(committed):
    """A revision from after the date it describes would be look-ahead."""
    bad = {a: s.staleness_days for a, s in committed.items()
           if s.staleness_days < 0}
    assert not bad, f"revision postdates its as_of: {bad}"


def test_committed_snapshots_are_sorted_and_unique(committed):
    dates = list(committed)
    assert dates == sorted(set(dates))


def test_membership_changes_over_the_window(committed):
    """If every snapshot were identical the reconstruction would be doing
    nothing — which is exactly the bug it exists to fix."""
    first = committed[min(committed)]
    last = committed[max(committed)]
    assert set(first.tickers) != set(last.tickers)
