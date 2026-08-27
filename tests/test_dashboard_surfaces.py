"""What the dashboard publishes, and what it deliberately no longer publishes.

Two owner-directed changes on 2026-08-26 (evening) are pinned here:

1. **The Model Portfolio surface was removed.** It was a renamed subset of
   ``table_data`` - every column it carried already existed there - and it
   shipped no position weights, so it never answered "how much" either. The
   suite passed 676/676 both before and after that removal, which is the point:
   nothing tested this surface, so nothing would have noticed it silently
   coming back or half-leaving. A half-removal is the dangerous state, because
   ``D.portfolio`` would then be ``undefined`` at render time and take the whole
   script down with it - a blank public page, per ``tests/test_dashboard_js.py``.

2. **Business descriptions were added to the drilldown.** They ride along in the
   ``.info`` dict the fetch already pulls, so the cost is payload, not API
   calls. They are display-only: never scored, never ranked.

See ``METHODOLOGY_CHANGELOG.md`` 2026-08-26 (evening).
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import generate_dashboard as g  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

CATEGORIES = ["valuation", "quality", "growth", "momentum",
              "risk", "revisions", "size", "investment"]


def _frame() -> pd.DataFrame:
    """A universe small enough to reason about, wide enough to render.

    ``CCC`` is value-trap flagged and ``DDD`` growth-trap flagged so the Top 5
    exclusion rule has something to exclude. ``BBB`` carries a NaN description
    and ``DDD`` an empty one - the two ways a provider omits the field.
    """
    df = pd.DataFrame({
        "Ticker": ["AAA", "BBB", "CCC", "DDD"],
        "Company": ["Alpha Inc", "Beta Corp", "Gamma Ltd", "Delta SA"],
        "Sector": ["Information Technology", "Financials", "Energy", "Utilities"],
        "Composite": [80.0, 60.0, 55.0, 40.0],
        "Rank": [1, 2, 3, 4],
        "Value_Trap_Flag": [False, False, True, False],
        "Growth_Trap_Flag": [False, False, False, True],
        "_about": ["Alpha Inc designs and sells things worldwide.",
                   np.nan,
                   "Gamma Ltd explores for and produces crude oil.",
                   ""],
        "_industry": ["Software - Infrastructure", "Banks - Diversified", "Oil & Gas E&P", ""],
    })
    for cat in CATEGORIES:
        df[cat + "_score"] = 50.0
    return df


@pytest.fixture(scope="module")
def payload() -> dict:
    data = g.prepare_dashboard_data({
        "df": _frame(), "meta": {"run_date": "2026-08-26"},
        "weights": {}, "sens_df": None, "corr_df": None, "cfg": {},
    })
    return json.loads(data)


@pytest.fixture(scope="module")
def html() -> str:
    return g.generate_html()


# ---------------------------------------------------------------------------
# 1. The model portfolio is gone - from the payload and from the front end
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["portfolio", "spx_weights"])
def test_removed_keys_absent_from_payload(payload, key):
    """``spx_weights`` went with it: the portfolio-vs-SPX chart was its only
    consumer, and a sector split of the S&P 500 against itself is a tautology."""
    assert key not in payload


def test_payload_still_carries_the_surfaces_that_replaced_it(payload):
    """Everything the portfolio panel showed is reachable from ``table_data``."""
    assert len(payload["table_data"]) == 4
    row = payload["table_data"][0]
    for col in ["Ticker", "Company", "Sector", "Composite", "Rank"]:
        assert col in row
    for cat in CATEGORIES:
        assert cat + "_score" in row


@pytest.mark.parametrize("symbol", [
    "D.portfolio",          # would be undefined at runtime and kill the script
    "renderPortfolio",
    "renderSectorAlloc",
    "sector-alloc-chart",
    "portfolio-table",
    "portfolio-kpis",
    'id="sec-portfolio"',
    "D.spx_weights",
])
def test_front_end_has_no_dangling_portfolio_reference(html, symbol):
    assert symbol not in html


def test_defensibility_section_survived_the_removal(html):
    """Weight sensitivity lived inside the portfolio *section* in the source
    layout but is a protected feature (CLAUDE.md rule 7). It must still render."""
    assert 'id="sec-defensibility"' in html
    assert "sensitivity-table" in html
    assert "correlation-heatmap" in html


# ---------------------------------------------------------------------------
# 2. Top 5 is re-sourced from the ranking, and still excludes traps
# ---------------------------------------------------------------------------

def _code_only(block: str) -> str:
    """Strip ``//`` comment lines so an assertion tests behaviour, not prose."""
    lines = block.split("\n")
    return "\n".join(ln for ln in lines if not ln.lstrip().startswith("//"))


def test_top5_reads_table_data_not_holdings(html):
    block = _code_only(html[html.index("function renderTop5()"):][:1600])
    assert "D.table_data" in block
    assert "holdings" not in block


def test_top5_still_excludes_trap_flagged_names(html):
    """Dropping the trap filter would promote a flagged stock into the
    headline five - the one behaviour of the old portfolio worth keeping."""
    block = html[html.index("function renderTop5()"):][:1600]
    assert "Value_Trap_Flag" in block
    assert "Growth_Trap_Flag" in block


def test_trap_exclusion_matches_the_rule_the_js_applies(payload):
    """The Python-side equivalent of the JS filter, so the intent is pinned
    even though the selection itself happens in the browser."""
    eligible = [r for r in payload["table_data"]
                if not r["Value_Trap_Flag"] and not r["Growth_Trap_Flag"]]
    eligible.sort(key=lambda r: r["Rank"])
    assert [r["Ticker"] for r in eligible] == ["AAA", "BBB"]


# ---------------------------------------------------------------------------
# 3. Business descriptions reach the drilldown
# ---------------------------------------------------------------------------

def test_fetch_asks_for_the_business_summary():
    """It must come off the ``.info`` dict already being fetched. Pulling it
    from a second endpoint would multiply the API cost of every run, and the
    data loop is already rate-limited by Yahoo."""
    src = (REPO / "factor_engine.py").read_text(encoding="utf-8")
    assert 'rec["longBusinessSummary"]' in src
    fetch = src[src.index("def _fetch_single_ticker_inner"):]
    fetch = fetch[:fetch.index("# ---- quarterly financial statements")]
    assert '_safe(info, "longBusinessSummary"' in fetch


def test_description_reaches_stock_detail(payload):
    assert payload["stock_detail"]["AAA"]["about"].startswith("Alpha Inc designs")
    assert payload["stock_detail"]["AAA"]["industry"] == "Software - Infrastructure"


@pytest.mark.parametrize("ticker", ["BBB", "DDD"])
def test_missing_description_becomes_empty_string(payload, ticker):
    """NaN must not reach the browser as the string "nan" - the front end
    tests truthiness to decide whether to show the block at all."""
    assert payload["stock_detail"][ticker]["about"] == ""


def test_no_nan_text_leaked_anywhere(payload):
    for ticker, detail in payload["stock_detail"].items():
        for field in ("about", "industry"):
            assert detail[field].strip().lower() != "nan", ticker


def test_description_is_not_scored(payload):
    """Display-only. If it ever appears among the scored metrics, the screener
    would be ranking on prose."""
    for detail in payload["stock_detail"].values():
        assert "about" not in detail["raw"]
        assert "about" not in detail["pct"]
        assert "longBusinessSummary" not in detail["raw"]


@pytest.mark.parametrize("hook", [
    'id="modal-about"',
    'id="modal-about-text"',
    'id="modal-about-toggle"',
    'id="modal-industry"',
    "function renderAbout",
    "function toggleAbout",
    "renderAbout(s);",
])
def test_about_ui_hooks_are_present(html, hook):
    assert hook in html


def test_about_overflow_is_measured_after_layout(html):
    """Regression: the first cut measured ``scrollHeight`` inside
    ``renderAbout``, which runs while the modal is still ``display:none``.
    Both heights read 0, so "Show more" was hidden on every stock and long
    descriptions were permanently truncated with no way to expand them."""
    block = html[html.index("function renderAbout"):]
    block = block[:block.index("function toggleAbout")]
    assert "requestAnimationFrame" in block
    measure = block[block.index("requestAnimationFrame"):]
    assert "scrollHeight" in measure, "overflow test must run inside the rAF callback"


def test_about_names_its_source(html):
    """Provenance is the product (CLAUDE.md). A reader must be able to tell
    this paragraph is the provider's words, not the screener's judgement."""
    block = html[html.index('id="modal-about"'):][:1200]
    assert "Yahoo Finance" in block
    assert re.search(r"not scored", block)


# ---------------------------------------------------------------------------
# 4. Section order and default collapse state (owner request, 2026-08-26 eve)
# ---------------------------------------------------------------------------

def _section_pos(html: str, sec_id: str) -> int:
    marker = 'id="%s"' % sec_id
    assert marker in html, sec_id
    return html.index(marker)


def test_top5_comes_before_what_changed(html):
    """The owner reads the ranking first and the deltas second. Section order
    in the emitted HTML *is* the reading order - there is no ordering layer."""
    assert _section_pos(html, "sec-top5") < _section_pos(html, "sec-changed")


def test_what_changed_still_precedes_the_universe_table(html):
    assert _section_pos(html, "sec-changed") < _section_pos(html, "sec-universe")


@pytest.mark.parametrize("sec_id", ["sec-changed", "sec-analytics", "sec-defensibility"])
def test_collapsed_by_default(html, sec_id):
    """Owner request: the landing view is Top 5 and the full table. Everything
    else is one click away rather than scrolled past."""
    tag = html[_section_pos(html, sec_id) - 200:_section_pos(html, sec_id) + 60]
    section = tag[tag.rindex("<section"):]
    assert "collapsed" in section, "%s should be collapsed by default" % sec_id


@pytest.mark.parametrize("sec_id", ["sec-top5", "sec-universe"])
def test_open_by_default(html, sec_id):
    """These two are the landing view. Collapsing them would leave a visitor
    looking at nothing but headers."""
    tag = html[_section_pos(html, sec_id) - 200:_section_pos(html, sec_id) + 60]
    section = tag[tag.rindex("<section"):]
    assert "collapsed" not in section, "%s must not be collapsed" % sec_id


def test_what_changed_reveal_does_not_fight_the_collapse(html):
    """`renderChanged()` un-hides the section with `style.display` when history
    exists. That must not also expand it - the two mechanisms are independent,
    and clearing `collapsed` there would silently undo the default."""
    block = html[html.index("function renderChanged()"):]
    block = block[:block.index("// Range switch")]
    assert "sec.style.display = ''" in block
    assert "classList.remove('collapsed')" not in block
