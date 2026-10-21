"""What the page says about a share class it does not rank - checked on what the code does.

Scoring each company once (2026-10-21) removes three tickers from the ranking. Without a
sentence of their own the page would tell a reader who searches FOX, or who holds it, that the
company "has left the index, or [its] data could not be fetched" - which is false, and is the
class of defect the context layer's close-out was built to stop (CLAUDE.md 0.11: "a claim whose
check tests the plumbing instead of the assertion is an unchecked claim").

So these tests render the **generator's own page** and call its functions, rather than grepping
the generated source for a string, and the payload half is asserted by behaviour on constructed
inputs. Claims: ``overview.share_classes``, ``drilldown.share_class_second_line``.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_dashboard as gd  # noqa: E402
import run_screener  # noqa: E402
import share_classes as sc  # noqa: E402

try:
    from playwright import sync_api
except Exception:  # noqa: BLE001
    sync_api = None

with open(ROOT / "config.yaml", encoding="utf-8") as _fh:
    CFG = yaml.safe_load(_fh)
DECLARED = list(CFG["share_classes"]["designated"])
PAIRS = {"GOOGL": "GOOG", "FOXA": "FOX", "NWSA": "NWS"}


# ----------------------------------------------------------------- the methodology paragraph
def _overview() -> str:
    return run_screener.build_screener_overview(CFG)


def test_the_page_names_the_declared_lines_and_only_those():
    """The sentence is generated from `config.yaml`, so it cannot name a line the run dropped."""
    text = _overview()
    i = text.index("One company, one line")
    para = text[i:i + 2200]
    for t in DECLARED:
        assert f"**{t}**" in para, f"{t} is declared but not named on the page"
    for kept, second in PAIRS.items():
        if kept in DECLARED:
            assert f"**{second}**" not in para, f"{second} is not ranked but the page names it"


def test_the_page_states_the_switching_margin_from_config():
    margin = float(CFG["share_classes"]["challenge_margin"])
    text = _overview()
    assert f"{margin:.0%}" in text[text.index("One company, one line"):][:2200]
    assert "declared in `config.yaml` rather than re-picked" in text


def test_turning_it_off_changes_what_the_page_says():
    """A page that keeps the sentence when the rule is off would be a false claim."""
    cfg = {**CFG, "share_classes": {**CFG["share_classes"], "score_once_per_company": False}}
    text = run_screener.build_screener_overview(cfg)
    assert "is switched off in this run's configuration" in text
    assert "Each company is therefore scored" not in text


def test_the_claim_detects_on_the_generated_page_and_not_on_the_off_variant():
    import claims
    claim = next(c for c in claims.CLAIMS if c.id == "overview.share_classes")
    assert re.search(claim.detect, _overview())
    cfg = {**CFG, "share_classes": {**CFG["share_classes"], "score_once_per_company": False}}
    assert not re.search(claim.detect, run_screener.build_screener_overview(cfg))


# -------------------------------------------------------------------------- the payload block
def _row(ticker, rank, company):
    return {"Ticker": ticker, "Rank": rank, "Company": company}


def test_the_payload_carries_the_second_lines():
    run_data = {"share_classes": {"secondary": {
        "FOX": {"scored_as": "FOXA", "price": 57.22, "price_gap_pct": -10.80,
                "company": "Fox Corporation"}}}}
    out = gd._share_classes_block(run_data, [_row("FOXA", 35, "Fox Corporation")])
    assert out == {"FOX": {"scored_as": "FOXA", "company": "Fox Corporation", "rank": 35,
                           "price": 57.22, "price_gap_pct": -10.80}}


def test_a_second_line_whose_company_is_unranked_says_nothing():
    """If the designated line is itself absent - a dropped fetch - the page must not claim a
    rank it does not have. Saying nothing falls back to the generic 'not in this run' text."""
    run_data = {"share_classes": {"secondary": {
        "FOX": {"scored_as": "FOXA", "price": 57.22, "price_gap_pct": -10.8}}}}
    assert gd._share_classes_block(run_data, [_row("AAPL", 1, "Apple Inc.")]) == {}


def test_no_share_class_record_is_an_empty_block_not_a_crash():
    assert gd._share_classes_block({}, [_row("AAPL", 1, "Apple")]) == {}
    assert gd._share_classes_block({"share_classes": {}}, []) == {}


def test_the_block_rides_the_first_payload_not_the_context_file():
    """The search palette and My Holdings run before `dashboard_context.js` loads."""
    assert "share_classes" not in gd.CONTEXT_KEYS


# ------------------------------------------------------- the price gap IS the whole difference
def test_the_price_gap_is_the_only_valuation_difference():
    """The page tells a reader the second line is "the same claim on the same earnings" at a
    price gap. That is only honest because every valuation ratio differs between two lines of
    one company by exactly the ratio of their prices - measured here on the newest run."""
    runs = glob.glob(str(ROOT / "runs" / "*" / "02_outliers_flagged.parquet"))
    if not runs:
        pytest.skip("no run artifacts present")
    run = Path(sorted(runs, key=os.path.getmtime)[-1]).parent
    m = pd.read_parquet(run / "02_outliers_flagged.parquet").set_index("Ticker")
    raw = pd.read_parquet(run / "00_raw_fetch.parquet").set_index("Ticker")
    checked = 0
    for kept, second in PAIRS.items():
        if kept not in m.index or second not in m.index:
            continue      # a run made after the rule shipped carries one line per company
        ey = m.loc[kept, "earnings_yield"] / m.loc[second, "earnings_yield"]
        pr = raw.loc[second, "currentPrice"] / raw.loc[kept, "currentPrice"]
        assert ey == pytest.approx(pr, rel=1e-4), f"{kept}/{second}: {ey} vs {pr}"
        checked += 1
    if checked == 0:
        pytest.skip("this run already scores each company once")


# ---------------------------------------------------------------------- the page, rendered
@pytest.fixture(scope="module")
def page(tmp_path_factory):
    """The generator's own page with the published payloads beside it."""
    if sync_api is None:
        pytest.skip("playwright not installed")
    for f in ("dashboard_data.js", "dashboard_context.js"):
        if not (ROOT / f).exists():
            pytest.skip(f"no published {f}")
    d = tmp_path_factory.mktemp("scpage")
    (d / "index.html").write_text(
        gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t", data_version="v"),
        encoding="utf-8")
    for f in ("dashboard_data.js", "dashboard_context.js"):
        shutil.copy(ROOT / f, d / f)
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        errors: list[str] = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto((d / "index.html").as_uri(), wait_until="load", timeout=60000)
        pg.wait_for_function("typeof notInRunText === 'function' && typeof D !== 'undefined'",
                             timeout=60000)
        # The payload under test: FOXA is ranked, FOX is its other listed class.
        pg.evaluate("""() => {
            D.share_classes = {FOX: {scored_as: 'FOXA', company: 'Fox Corporation',
                                     rank: 35, price: 57.22, price_gap_pct: -10.8}};
            if (D.stock_detail) delete D.stock_detail.FOX;
        }""")
        yield pg
        b.close()
        assert not errors, errors


def _txt(page, js):
    """What the reader sees: the HTML these functions return, rendered to text."""
    html = page.evaluate(js)
    rendered = page.evaluate("h => { const d = document.createElement('div');"
                             " d.innerHTML = h; return d.textContent; }", html)
    return re.sub(r"\s+", " ", rendered).strip()


def test_a_second_class_is_not_called_a_departed_stock(page):
    t = _txt(page, "notInRunText('FOX')")
    assert "second listed share class of Fox Corporation" in t
    assert "scored once, on its more-traded class FOXA, ranked #35 here" in t
    assert "left the index" not in t and "could not be fetched" not in t


def test_the_sentence_names_the_providers_whose_rule_it_is(page):
    t = _txt(page, "notInRunText('FOX')")
    assert "S&P DJI" in t and "FTSE Russell" in t


def test_the_sentence_states_the_price_gap_and_its_direction(page):
    t = _txt(page, "notInRunText('FOX')")
    assert "trades 10.8% below FOXA for the same claim on the same earnings" in t
    page.evaluate("D.share_classes.FOX.price_gap_pct = 9.5")
    assert "trades 9.5% above FOXA" in _txt(page, "notInRunText('FOX')")
    page.evaluate("D.share_classes.FOX.price_gap_pct = null")
    assert "trades" not in _txt(page, "notInRunText('FOX')")
    page.evaluate("D.share_classes.FOX.price_gap_pct = -10.8")


def test_a_ticker_this_run_really_did_not_score_keeps_the_generic_sentence(page):
    t = _txt(page, "notInRunText('ZZZX')")
    assert "left the index" in t
    assert "share class" not in t


def test_the_holdings_card_names_the_ranked_line(page):
    """CLAUDE.md row 5: a saved name renders every time. A second share class must render as
    its company's other class, not as a departed stock, and keep its remove button."""
    html = page.evaluate("unscoredHoldingCard('FOX')")
    assert "second listed share class of Fox Corporation" in re.sub(r"\s+", " ", html)
    assert "left the index" not in html
    assert "removeHolding(" in html and ">#35<" in html


def test_a_held_name_that_really_left_still_gets_the_departed_card(page):
    html = page.evaluate("unscoredHoldingCard('ZZZX')")
    assert "has left the S&amp;P 500 or its data could not be fetched" in html
    assert "share class" not in html


def test_the_lookup_accepts_the_dotted_spelling(page):
    page.evaluate("D.share_classes['BF-A'] = {scored_as: 'BF-B', company: 'Brown-Forman', "
                  "rank: 7, price: 1, price_gap_pct: 0};")
    assert "second listed share class" in _txt(page, "notInRunText('BF.A')")
    page.evaluate("delete D.share_classes['BF-A']")
