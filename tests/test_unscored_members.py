"""A member the run fetched and could not score is named and told why (2026-10-29).

HONA and VYLR were S&P 500 members the 2026-10-29 run left unscored (the source returned too
little of their data), and they were absent from the payload altogether. A reader who searched
either, or held one, was told only that a company "that has left the index, or whose data could
not be fetched" is not in the ranking - the first half false for both. The run records them in its
own ``universe.json`` ("failed"); the generator now publishes them as ``unscored`` and the search,
the palette and My Holdings say what is true.

**The page tests use stand-in tickers, not HONA and VYLR** - corrected 2026-10-30. They named the
two real tickets, and being unscored is a property of one morning's fetch: the 2026-10-30 run
scored HONA, ``notInRunText('hona')`` correctly returned '' because the stock *is* in the run, and
both page tests failed on a payload that was right. A red ``main`` blocks every merge, so a test
that pins which stock the source happened to fail yesterday costs the project a day. The branch
under test is "this ticker is in ``D.unscored``", which any ticker can stand for;
``test_the_stand_ins_are_absent_from_the_published_run`` fails if a stand-in ever stops standing in.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_dashboard as gd  # noqa: E402

try:
    from playwright import sync_api
except ImportError:  # pragma: no cover
    sync_api = None


# --------------------------------------------------------------------------- the generator
def _run(tmp_path, failed, names):
    run = tmp_path / "run"
    run.mkdir()
    (run / "universe.json").write_text(json.dumps({"scored": ["AAA"], "failed": failed}), encoding="utf-8")
    raw = run / "00_raw_fetch.parquet"
    pd.DataFrame({"Ticker": list(names), "longName": list(names.values())}).to_parquet(raw)
    return run, raw


def test_the_run_s_own_failed_list_is_published_with_names(tmp_path):
    run, raw = _run(tmp_path, ["HONA", "VYLR"], {"AAA": "Aaa Corp", "HONA": "Honeywell Aerospace Inc."})
    assert gd._run_unscored(run, raw) == {"HONA": "Honeywell Aerospace Inc.", "VYLR": None}


def test_a_ticker_the_table_carries_is_never_listed(tmp_path):
    run, raw = _run(tmp_path, ["HONA", "AAA"], {"HONA": "Honeywell Aerospace Inc."})
    block = gd._unscored_block({"unscored": gd._run_unscored(run, raw)}, [{"Ticker": "AAA"}])
    assert block == {"HONA": "Honeywell Aerospace Inc."}


def test_a_missing_or_broken_record_never_stops_a_build(tmp_path):
    assert gd._run_unscored(tmp_path, None) == {}
    (tmp_path / "universe.json").write_text("{not json", encoding="utf-8")
    assert gd._run_unscored(tmp_path, None) == {}
    (tmp_path / "universe.json").write_text(json.dumps({"failed": ["HONA"]}), encoding="utf-8")
    assert gd._run_unscored(tmp_path, tmp_path / "missing.parquet") == {"HONA": None}


def test_the_payload_carries_the_block():
    src = (ROOT / "generate_dashboard.py").read_text(encoding="utf-8")
    assert '"unscored": _unscored_block(run_data, table_data)' in src
    assert '"unscored": _run_unscored(run_dir, raw_path)' in src


# --------------------------------------------------------------------------- the page
# Stand-ins for the two real cases, so the assertions do not depend on which stock the source
# failed to serve this morning. QZZA has a recorded name, QZZB has none, ZZZZ is not a member.
MEMBER_NAMED, MEMBER_UNNAMED, NOT_A_MEMBER = "QZZA", "QZZB", "ZZZZ"


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    if sync_api is None:
        pytest.skip("MISSING DEP: playwright not installed")
    for f in ("dashboard_data.js", "dashboard_context.js"):
        if not (ROOT / f).exists():
            pytest.skip(f"MISSING DATA: no published {f}")
    d = tmp_path_factory.mktemp("unscored")
    (d / "index.html").write_text(gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t",
                                                   data_version="v"), encoding="utf-8")
    for f in ("dashboard_data.js", "dashboard_context.js"):
        shutil.copy(ROOT / f, d / f)
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        ctx = b.new_context(viewport={"width": 1280, "height": 900})
        ctx.add_init_script(
            "try { localStorage.setItem('screener_holdings_v1', JSON.stringify(%s)); } catch (e) {}"
            % json.dumps([MEMBER_NAMED, NOT_A_MEMBER])
        )
        pg = ctx.new_page()
        errors: list[str] = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto((d / "index.html").as_uri(), wait_until="load", timeout=60000)
        pg.wait_for_function("typeof notInRunText === 'function' && typeof D !== 'undefined'", timeout=60000)
        pg.evaluate(
            "u => { D.unscored = u; renderHoldings(); }",
            {MEMBER_NAMED: "Quartz Zenith Aerospace Inc.", MEMBER_UNNAMED: None},
        )
        yield pg
        b.close()
        assert not errors, errors


def test_the_stand_ins_are_absent_from_the_published_run(page):
    """The three tickers above must match no scored stock, or every assertion below is vacuous.

    This is the guard the original tests lacked: they named HONA and VYLR, the two stocks the
    2026-10-29 fetch happened to fail on, and went red the morning the source served HONA again.
    """
    for t in (MEMBER_NAMED, MEMBER_UNNAMED, NOT_A_MEMBER):
        assert page.evaluate(
            "t => !(D.stock_detail || {})[t] && !(D.share_classes || {})[t]"
            " && !(D.table_data || []).some(r => r.Ticker === t)",
            t,
        ), f"{t} is in the published run; pick a stand-in that is not"


def test_a_search_for_an_unscored_member_names_it_and_says_why(page):
    txt = page.evaluate("t => notInRunText(t.toLowerCase())", MEMBER_NAMED)
    assert txt.startswith("Quartz Zenith Aerospace Inc. is in the S&amp;P 500, but this run left it unscored")
    assert "left the index" not in txt
    # no name recorded: the ticker stands in for it
    assert page.evaluate("t => notInRunText(t)", MEMBER_UNNAMED).startswith(
        f"{MEMBER_UNNAMED} is in the S&amp;P 500")


def test_a_ticker_that_is_not_a_member_keeps_the_general_reason(page):
    assert "has left the index" in page.evaluate("t => notInRunText(t)", NOT_A_MEMBER)


def test_a_held_unscored_member_says_why_and_a_departed_one_keeps_its_reason(page):
    cards = page.evaluate("[...document.querySelectorAll('.holding-note-unscored')].map(e => e.textContent)")
    held = next(c for c in cards if c.startswith(MEMBER_NAMED))
    gone = next(c for c in cards if c.startswith(NOT_A_MEMBER))
    assert "Quartz Zenith Aerospace Inc. is in the S&P 500, but this run left it unscored" in held
    assert "left the S&P 500" not in held
    assert "has left the S&P 500 or its data could not be fetched" in gone
