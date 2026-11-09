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
def two_lines(m, raw, kept, second):
    """What two listed lines of one company share, and what separates them.

    Both are claims on one set of filings, so every company-level figure behind a ratio is the
    same for each line. ``earnings_yield`` is that one net income over each line's **market
    value**, so the ratio between the lines is the ratio of their market values;
    ``dividend_yield`` is one dividend per share over each line's **price**, so its ratio is the
    ratio of their prices. Those two coincide exactly - which is what makes the page's
    price-gap wording the whole of the difference - *while both of the source's records count
    the same shares*. The share count behind a market value is ``marketCap / currentPrice``.
    """
    mc_k, mc_s = float(raw.loc[kept, "marketCap"]), float(raw.loc[second, "marketCap"])
    p_k, p_s = float(raw.loc[kept, "currentPrice"]), float(raw.loc[second, "currentPrice"])
    out = {"shares_kept": mc_k / p_k, "shares_second": mc_s / p_s,
           "mc_gap": mc_s / mc_k,        # the gap earnings_yield reads
           "price_gap": p_s / p_k}       # the gap the page states
    out["one_share_count"] = out["shares_kept"] == pytest.approx(out["shares_second"], rel=1e-4)
    for name, expect in (("earnings_yield", out["mc_gap"]), ("dividend_yield", out["price_gap"])):
        a, b = m.loc[kept, name], m.loc[second, name]
        out[name] = None if (pd.isna(a) or pd.isna(b) or b == 0) else (a / b, expect)
    return out


def audit_pairs(m, raw, ranked, pairs=None):
    """Hold every pair of lines present in `m` to what the page says about them.

    Raises ``AssertionError`` on a violation; returns the pairs whose two records agree on a
    share count and those whose do not. Taking the three frames as arguments is what lets every
    branch be exercised on constructed inputs below rather than only on the branches the
    morning's data happens to reach.
    """
    checked, one_count_off = [], []
    for kept, second in (pairs or PAIRS).items():
        if kept not in m.index or second not in m.index:
            continue      # only one of the two lines was fetched at all
        g = two_lines(m, raw, kept, second)
        for name in ("earnings_yield", "dividend_yield"):
            if g[name] is None:
                continue                      # a non-payer, or a figure this run did not get
            got, expect = g[name]
            assert got == pytest.approx(expect, rel=1e-4), \
                f"{kept}/{second} {name}: {got} vs {expect}"
        if g["one_share_count"]:
            assert g["mc_gap"] == pytest.approx(g["price_gap"], rel=1e-4), \
                f"{kept}/{second}: {g['mc_gap']} vs {g['price_gap']}"
            checked.append((kept, second))
        else:
            one_count_off.append((kept, second, g["shares_kept"], g["shares_second"]))
        # Whichever basis each record is on, only one line of the company may be ranked. Two
        # ranked lines plus two share counts would put two inconsistent valuations of one
        # company on the page, and that is the state this must refuse.
        assert kept in ranked, f"{kept} is the designated line and is not ranked"
        assert second not in ranked, f"{second} is a second line and was ranked"
    return checked, one_count_off


def test_the_price_gap_is_the_only_valuation_difference():
    """The page tells a reader the second line is "the same claim on the same earnings" at a
    price gap. Measured on the newest run: one company's two lines share every figure from its
    filings, so each ratio differs between them by exactly its own gap - market values for a
    yield on market value, prices for a yield on price - and while both records count the same
    shares those are one number, which is the price gap the page states.

    **The share count is a premise, so it is asserted and not assumed** (CLAUDE.md "data->red").
    A corporate action can reach one line's record before the other's: Fox completed its
    acquisition of Roku on 2026-11-06 (8-K of that date, "the Closing Date"), the source added
    the 144,500,000 newly issued Class A shares to FOXA and left FOX's market value on the
    pre-merger count, and on 2026-11-09 the two lines implied 566,301,414 and 421,801,414
    shares. The old form of this test read one metric against the price gap alone, so it failed
    and **discarded a night of the data loop's evidence** (`logs/datarun-2026-11-09_020001.log`).
    Nothing published was wrong: `02_outliers_flagged` keeps every fetched line, the second line
    is dropped before the percentiles, so no ratio of FOX's is scored, ranked or published - and
    the page's sentence is about the two prices, which were both current. That the company is
    scored once is the fact which makes the divergence harmless, so that is what gets asserted
    here rather than tolerated silently.
    """
    runs = glob.glob(str(ROOT / "runs" / "*" / "02_outliers_flagged.parquet"))
    if not runs:
        pytest.skip("no run artifacts present")
    run = Path(sorted(runs, key=os.path.getmtime)[-1]).parent
    m = pd.read_parquet(run / "02_outliers_flagged.parquet").set_index("Ticker")
    raw = pd.read_parquet(run / "00_raw_fetch.parquet").set_index("Ticker")
    ranked = set(pd.read_parquet(run / "03_percentiles.parquet")["Ticker"])
    checked, one_count_off = audit_pairs(m, raw, ranked)
    if not checked and not one_count_off:
        pytest.skip("this run fetched only one line per company")


def _frames(mc_kept, mc_second, price_kept, price_second, ey_kept, ey_second):
    """Two constructed lines of one company, in the shape the run artifacts have."""
    m = pd.DataFrame({"Ticker": ["AAA", "BBB"],
                      "earnings_yield": [ey_kept, ey_second],
                      "dividend_yield": [1.0 / price_kept, 1.0 / price_second]}).set_index("Ticker")
    raw = pd.DataFrame({"Ticker": ["AAA", "BBB"],
                        "marketCap": [mc_kept, mc_second],
                        "currentPrice": [price_kept, price_second]}).set_index("Ticker")
    return m, raw


def test_one_share_count_makes_the_market_value_gap_the_price_gap():
    """The state the page's wording describes: both records count 100 shares."""
    m, raw = _frames(1000.0, 800.0, 10.0, 8.0, 50.0 / 1000, 50.0 / 800)
    g = two_lines(m, raw, "AAA", "BBB")
    assert g["one_share_count"] and g["shares_kept"] == 100.0
    assert g["mc_gap"] == pytest.approx(g["price_gap"])
    assert g["earnings_yield"][0] == pytest.approx(g["earnings_yield"][1])
    assert g["dividend_yield"][0] == pytest.approx(g["dividend_yield"][1])


def test_two_share_counts_separate_the_two_gaps_and_are_detected():
    """Fox on 2026-11-09, in miniature: the second line's record is 1.5x short of the first's
    share count, so the market-value gap and the price gap part company - and the yields still
    each track their own gap exactly, which is what says the filings behind them are one."""
    m, raw = _frames(1500.0, 800.0, 10.0, 8.0, 50.0 / 1500, 50.0 / 800)
    g = two_lines(m, raw, "AAA", "BBB")
    assert not g["one_share_count"]
    assert (g["shares_kept"], g["shares_second"]) == (150.0, 100.0)
    assert g["mc_gap"] != pytest.approx(g["price_gap"], rel=1e-4)
    assert g["earnings_yield"][0] == pytest.approx(g["earnings_yield"][1])
    assert g["dividend_yield"][0] == pytest.approx(g["dividend_yield"][1])


def test_a_yield_off_its_own_gap_is_caught_on_either_basis():
    """The assertion that carries the weight: two lines of one company cannot have two net
    incomes. It must fire whether or not the share counts agree."""
    for mc_kept in (1000.0, 1500.0):          # one share count, then two
        m, raw = _frames(mc_kept, 800.0, 10.0, 8.0, 50.0 / mc_kept, 61.0 / 800)
        g = two_lines(m, raw, "AAA", "BBB")
        assert g["earnings_yield"][0] != pytest.approx(g["earnings_yield"][1], rel=1e-4)


def test_a_line_with_no_dividend_is_handled_rather_than_divided_by():
    m, raw = _frames(1000.0, 800.0, 10.0, 8.0, 0.05, 0.0625)
    m["dividend_yield"] = [float("nan"), 0.0]
    g = two_lines(m, raw, "AAA", "BBB")
    assert g["dividend_yield"] is None and g["earnings_yield"] is not None


# --------------------------------- and the audit itself, on inputs chosen rather than fetched
PAIR = {"AAA": "BBB"}


def _good(mc_kept=1000.0):
    """A pair the audit must accept, on one share count or on two."""
    return _frames(mc_kept, 800.0, 10.0, 8.0, 50.0 / mc_kept, 50.0 / 800)


def test_the_audit_accepts_a_sound_pair_on_either_basis_and_says_which():
    for mc_kept, one_count in ((1000.0, True), (1500.0, False)):
        m, raw = _good(mc_kept)
        checked, off = audit_pairs(m, raw, {"AAA"}, PAIR)
        assert bool(checked) is one_count and bool(off) is not one_count


def test_the_audit_refuses_two_ranked_lines_of_one_company():
    """The state that would put two inconsistent valuations of one company on the page."""
    m, raw = _good()
    with pytest.raises(AssertionError, match="second line and was ranked"):
        audit_pairs(m, raw, {"AAA", "BBB"}, PAIR)


def test_the_audit_refuses_a_designated_line_that_was_not_ranked():
    m, raw = _good()
    with pytest.raises(AssertionError, match="designated line and is not ranked"):
        audit_pairs(m, raw, {"ZZZ"}, PAIR)


def test_the_audit_refuses_two_net_incomes_on_either_basis():
    for mc_kept in (1000.0, 1500.0):
        m, raw = _frames(mc_kept, 800.0, 10.0, 8.0, 50.0 / mc_kept, 61.0 / 800)
        with pytest.raises(AssertionError, match="earnings_yield"):
            audit_pairs(m, raw, {"AAA"}, PAIR)


def test_the_audit_refuses_two_dividends_per_share_on_either_basis():
    for mc_kept in (1000.0, 1500.0):
        m, raw = _good(mc_kept)
        m.loc["BBB", "dividend_yield"] *= 1.02
        with pytest.raises(AssertionError, match="dividend_yield"):
            audit_pairs(m, raw, {"AAA"}, PAIR)


def test_the_audit_reports_nothing_when_only_one_line_was_fetched():
    """The skip the run-reading test takes - and it must not be reachable while both lines are
    there, or a fetch that silently lost a line would read as a clean pass."""
    m, raw = _good()
    assert audit_pairs(m.drop(index="BBB"), raw, {"AAA"}, PAIR) == ([], [])
    assert audit_pairs(m, raw, {"AAA"}, PAIR) != ([], [])


# ---------------------------------------------------------------------- the page, rendered
@pytest.fixture(scope="module")
def page(tmp_path_factory):
    """The generator's own page with the published payloads beside it."""
    if sync_api is None:
        pytest.skip("playwright not installed")
    for f in ("dashboard_data.js", "dashboard_context.js", "dashboard_detail.js"):
        if not (ROOT / f).exists():
            pytest.skip(f"no published {f}")
    d = tmp_path_factory.mktemp("scpage")
    (d / "index.html").write_text(
        gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t", data_version="v"),
        encoding="utf-8")
    for f in ("dashboard_data.js", "dashboard_context.js", "dashboard_detail.js"):
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


def test_a_company_name_ending_in_a_stop_does_not_print_two(page):
    """Seen on the live page, 2026-10-21: "a second listed share class of Alphabet Inc.."."""
    page.evaluate("D.share_classes.FOX.company = 'Alphabet Inc.'")
    t = _txt(page, "notInRunText('FOX')")
    assert "Alphabet Inc. The company is scored once" in t
    assert "Inc.." not in t
    page.evaluate("D.share_classes.FOX.company = 'Fox Corporation'")


def test_the_lookup_accepts_the_dotted_spelling(page):
    page.evaluate("D.share_classes['BF-A'] = {scored_as: 'BF-B', company: 'Brown-Forman', "
                  "rank: 7, price: 1, price_gap_pct: 0};")
    assert "second listed share class" in _txt(page, "notInRunText('BF.A')")
    page.evaluate("delete D.share_classes['BF-A']")
