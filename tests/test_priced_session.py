"""The header names the session the published prices are from (claim ``dashboard.priced_session``,
2026-11-06).

The page said "Data as of Nov 5, 2026, 2:00 AM CST" and nothing else about time. That run was priced
at the Nov 4 close, and the S&P 500 fell 4.4% on Nov 5, so on the day a reader most needed to know
how old the prices were, the page could not say. Each stock's ``price_date`` was already published
(the input gate reads it); the header and the footer now name the session most of them carry, and
how many do when not all of them do.

``generate_dashboard.priced_session`` is the rule in Python and ``pricedSession`` the same rule in
the page. These tests hold the two to each other on the published payload and on hand-worked cases,
and hold what the header prints to the payload, asserting its premise rather than which date a given
morning happened to have.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_dashboard as gd  # noqa: E402
from tests import sheet_warm  # noqa: E402

try:
    from playwright import sync_api
except ImportError:  # pragma: no cover
    sync_api = None


def _payload(name: str) -> dict:
    p = ROOT / name
    if not p.exists():
        pytest.skip(f"MISSING DATA: no published {name}")
    s = p.read_text(encoding="utf-8")
    return json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))


def _pretty(iso: str) -> str:
    d = date.fromisoformat(iso)
    return d.strftime("%b ") + str(d.day)


HAND = [
    ({"A": {"price_date": "2026-11-04"}, "B": {"price_date": "2026-11-04"}, "C": {"price_date": "2026-11-03"}},
     {"date": "2026-11-04", "n": 2, "of": 3}),
    # a tie goes to the later session
    ({"A": {"price_date": "2026-11-03"}, "B": {"price_date": "2026-11-04"}},
     {"date": "2026-11-04", "n": 1, "of": 2}),
    # a stock with no date counts in "of" and in nothing else
    ({"A": {"price_date": "2026-11-04"}, "B": {"price_date": None}, "C": {}},
     {"date": "2026-11-04", "n": 1, "of": 3}),
    ({"A": {"price_date": ""}, "B": {}}, None),
    # an empty date is no date, even where it outnumbers the real ones
    ({"A": {"price_date": ""}, "B": {"price_date": ""}, "C": {"price_date": "2026-11-04"}},
     {"date": "2026-11-04", "n": 1, "of": 3}),
    ({}, None),
]


@pytest.mark.parametrize("detail,want", HAND)
def test_a_hand_worked_set_of_price_dates(detail, want):
    assert gd.priced_session(detail) == want


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data() -> dict:
    return _payload("dashboard_data.js")


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    if sync_api is None:
        pytest.skip("playwright not installed")
    for f in ("dashboard_data.js", "dashboard_context.js"):
        if not (ROOT / f).exists():
            pytest.skip(f"MISSING DATA: no published {f}")
    d = tmp_path_factory.mktemp("priced")
    (d / "index.html").write_text(gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t",
                                                   data_version="v"), encoding="utf-8")
    for f in ("dashboard_data.js", "dashboard_context.js"):
        shutil.copy(ROOT / f, d / f)
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        pg = b.new_page(viewport={"width": 1440, "height": 1000})
        errors: list[str] = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto((d / "index.html").as_uri(), wait_until="load", timeout=60000)
        pg.wait_for_function("typeof pricedSession === 'function' && typeof D !== 'undefined' && "
                             "document.querySelector('#hero-eyebrow').textContent.indexOf('data as of') >= 0",
                             timeout=60000)
        sheet_warm.settle(pg)
        yield pg
        pg.close()
        b.close()
        assert not errors, errors


def test_the_page_and_the_python_rule_agree_on_the_published_payload(page, data):
    want = gd.priced_session(data["stock_detail"])
    # The premise: the payload carries price dates at all, or there is nothing to name.
    assert want is not None and want["n"] > 0
    assert page.evaluate("() => pricedSession()") == want


@pytest.mark.parametrize("detail,want", HAND)
def test_the_page_rule_on_the_hand_worked_dates(page, detail, want):
    got = page.evaluate("""sd => { const keep = D.stock_detail; D.stock_detail = sd;
        try { return pricedSession(); } finally { D.stock_detail = keep; } }""", detail)
    assert got == want, detail


def test_the_header_and_the_footer_name_the_published_session(page, data):
    ps = gd.priced_session(data["stock_detail"])
    want = f"prices at the {_pretty(ps['date'])} close"
    if ps["n"] < ps["of"]:
        want += f" for {ps['n']} of {ps['of']} stocks"
    eyebrow = page.inner_text("#hero-eyebrow").replace(" ", " ")
    foot = page.inner_text("#gen-time").replace(" ", " ")
    assert want.upper() in eyebrow.upper(), (eyebrow, want)
    assert foot.endswith(", " + want), (foot, want)


def test_a_partly_priced_run_says_how_many_stocks_the_session_covers(page):
    got = page.evaluate("""sd => { const keep = D.stock_detail; D.stock_detail = sd;
        try { return pricedSessionText(); } finally { D.stock_detail = keep; } }""", HAND[0][0])
    assert got == "prices at the Nov 4 close for 2 of 3 stocks"
    got = page.evaluate("""sd => { const keep = D.stock_detail; D.stock_detail = sd;
        try { return pricedSessionText(); } finally { D.stock_detail = keep; } }""", HAND[3][0])
    assert got == ""
