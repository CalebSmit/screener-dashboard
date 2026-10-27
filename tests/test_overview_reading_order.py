"""The methodology page's reading order, for a reader who reads one screen (2026-10-27).

WHY THIS EXISTS - CLAUDE.md priority 7 asks whether a student can open this on a phone and
understand it. Read at 375px the way a sophomore in an investment club would, the methodology
page opened on metric counts ("No stock is scored on all 31. Most use 27 general metrics...")
and reached the one sentence a first reader most needs - that the ranking has not been shown to
predict returns - only in Limitation 3, some 500 lines down. Its seven-step summary sat at the
very end, and published as a single run-on paragraph, because Python-Markdown needs a blank line
before a list: so did "Additional outputs" and the factor-exposure list.

The page now opens, inside the Overview, on what the tool is, how to use it and what it cannot
do; the seven steps follow; then a glossary of the words the rest of the page relies on. These
tests hold that order, keep every list a list, keep the summary's pointers into the Limitations
pointing at the right items, and check the summary's and glossary's computational sentences
against the engine (claims ``overview.at_a_glance_flags``, ``overview.at_a_glance_tested`` and
``overview.glossary``). Everything is read from the generator, never from the committed file
(rule 10); ``tests/test_overview_is_generated.py`` holds the committed file to it.
"""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

import markdown
import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import factor_engine as fe  # noqa: E402
import run_screener  # noqa: E402
import stock_summary  # noqa: E402


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def page(cfg) -> str:
    """The document the next data run will publish."""
    return run_screener.build_screener_overview(cfg)


def _h2_block(page: str, title: str) -> str:
    start = page.index(f"\n## {title}\n")
    nxt = page.find("\n## ", start + 4)
    return page[start: nxt if nxt != -1 else len(page)]


def _h3_block(block: str, title: str) -> str:
    start = block.index(f"\n### {title}\n")
    nxt = block.find("\n### ", start + 5)
    return block[start: nxt if nxt != -1 else len(block)]


# ---------------------------------------------------------------------------
# the order
# ---------------------------------------------------------------------------

def test_the_overview_opens_on_the_summary_then_how_to_use_it_then_its_limits(page):
    h2 = re.findall(r"^## (.+)$", page, re.M)
    assert h2[:3] == ["Overview", "Words this page uses", "Data sources"], h2[:3]
    overview = _h2_block(page, "Overview")
    h3 = re.findall(r"^### (.+)$", overview, re.M)
    assert h3 == ["If you read nothing else", "How to use it", "What it cannot do",
                  "What it measures", "In seven steps"], h3
    # nothing but the heading stands between the Overview's title and the summary
    assert overview.lstrip("\n").startswith("## Overview\n\n### If you read nothing else\n")


def test_the_seven_steps_are_stated_once(page):
    assert page.count("It does this in seven steps") == 1
    assert "\n## Summary\n" not in page


def test_the_summary_is_short_and_carries_no_advice_language(page):
    overview = _h2_block(page, "Overview")
    glance = _h3_block(overview, "If you read nothing else")
    bullets = [ln for ln in glance.splitlines() if ln.startswith("- **")]
    assert 4 <= len(bullets) <= 7, len(bullets)
    assert len(glance.split()) < 400, len(glance.split())
    front = glance + _h3_block(overview, "How to use it") + _h3_block(overview, "What it cannot do")
    assert stock_summary.advice_terms_in(front) == [], stock_summary.advice_terms_in(front)


def test_the_limitations_are_named_from_the_top_and_open_with_the_three_to_read(page):
    lim = _h2_block(page, "Limitations")
    lead = lim.split("\n1. ")[0]
    assert "If you read only three, read 3" in lead and "10" in lead and "11" in lead


# ---------------------------------------------------------------------------
# every list a list, every pointer to the right limitation
# ---------------------------------------------------------------------------

def test_every_list_on_the_page_renders_as_a_list(page):
    """A list straight after a line of text renders as one paragraph in Python-Markdown - the
    seven steps, 'Additional outputs' and the factor-exposure list all published that way."""
    html = markdown.markdown(page, extensions=["tables", "fenced_code"])
    broken = [p[:80] for p in re.findall(r"<p>((?:(?!</p>).)*?)</p>", html, re.S)
              if re.search(r"\n(\d+\.|-|\*) ", p)]
    assert broken == [], broken


# The item each pointer from the summary means, by the words that open its title.
LIMITATION_TOPICS = {1: "Data source", 3: "Point-in-time", 5: "EPS revisions reach back only 90 days",
                     10: "Not investment advice", 11: "The published premia"}


def _limitation_titles(page: str) -> dict[int, str]:
    lim = _h2_block(page, "Limitations")
    return {int(n): t for n, t in re.findall(r"^(\d+)\. \*\*([^*]+?):?\*\*", lim, re.M)}


def test_every_pointer_into_the_limitations_lands_on_the_item_it_means(page):
    titles = _limitation_titles(page)
    assert sorted(titles) == list(range(1, len(titles) + 1)), titles
    for n, words in LIMITATION_TOPICS.items():
        assert titles[n].startswith(words), (n, titles[n])
    front = _h2_block(page, "Overview")
    cited = {int(n) for n in re.findall(r"Limitation (\d+)", front)}
    lead_text = _h2_block(page, "Limitations").split("\n1. ")[0]
    lead = {int(n) for n in re.findall(r"\b(\d+) \(", lead_text)}
    assert cited and cited <= set(LIMITATION_TOPICS), cited
    assert lead == {3, 10, 11}, lead


# ---------------------------------------------------------------------------
# the glossary
# ---------------------------------------------------------------------------

# Where each glossary term is used on the rest of the page. A new term with no entry here fails,
# so the glossary cannot grow words the page does not use.
USED_AS = {
    "Sector percentile": ["sector percentile", "sector-relative percentile"],
    "Category score": ["category score"],
    "Composite": ["composite"],
    "Points": ["points"],
    "Rank": ["rank"],
    "Coverage": ["coverage"],
    "Trap flag": ["value trap", "growth trap"],
    "Bank-like stock": ["bank-like"],
    "LTM and MRQ": ["LTM", "MRQ"],
    "Enterprise value (EV)": ["Enterprise Value", "EV/EBITDA"],
    "Factor and premium": ["factor", "premium"],
    "Backtest": ["backtest"],
    "Look-ahead bias": ["look-ahead"],
    "Survivorship bias": ["left the index"],
    "Rank correlation and IC": ["rank correlation", "information-coefficient"],
    "Percentage point and basis point": ["percentage points", "bps"],
}


def _glossary(page: str) -> dict[str, str]:
    block = _h2_block(page, "Words this page uses")
    return dict(re.findall(r"^- \*\*(.+?)\*\* – (.+)$", block, re.M))


def test_the_glossary_defines_the_terms_a_first_reader_meets(page):
    terms = _glossary(page)
    for needed in ("Sector percentile", "Composite", "Trap flag", "Look-ahead bias",
                   "Survivorship bias", "Rank correlation and IC"):
        assert needed in terms, needed
    assert "*sector-relative*" in terms["Sector percentile"]
    assert "information coefficient (IC)" in terms["Rank correlation and IC"]


def test_every_glossary_term_is_one_the_page_uses(page):
    terms = _glossary(page)
    assert set(terms) == set(USED_AS), set(terms) ^ set(USED_AS)
    rest = page.replace(_h2_block(page, "Words this page uses"), "")
    for term, uses in USED_AS.items():
        for u in uses:
            assert u.lower() in rest.lower(), (term, u)


def test_the_glossary_percentile_is_sector_relative_and_higher_is_better():
    """Twelve stocks in each of two sectors. EV/EBITDA is lower-is-better: the cheapest in a
    sector gets the highest percentile, and the same value earns a different percentile in a
    sector with different peers."""
    assert fe.METRIC_DIR["ev_ebitda"] is False and fe.METRIC_DIR.get("fcf_yield", True) is True
    n = 12
    df = pd.DataFrame({
        "Ticker": [f"A{i}" for i in range(n)] + [f"B{i}" for i in range(n)],
        "Sector": ["Utilities"] * n + ["Energy"] * n,
        "ev_ebitda": list(np.linspace(5, 16, n)) + list(np.linspace(10, 32, n)),
        "fcf_yield": list(np.linspace(0.01, 0.12, n)) * 2,
    })
    out = fe.compute_sector_percentiles(df.copy())
    a = out[out["Sector"] == "Utilities"].set_index("Ticker")
    assert a["ev_ebitda_pct"].idxmax() == "A0" and a["ev_ebitda_pct"].idxmin() == f"A{n - 1}"
    assert a["fcf_yield_pct"].idxmax() == f"A{n - 1}"
    both = out.set_index("Ticker")
    assert both.loc["A5", "ev_ebitda"] == both.loc["B0", "ev_ebitda"] == 10.0
    assert both.loc["A5", "ev_ebitda_pct"] != both.loc["B0", "ev_ebitda_pct"]
    assert 0 < out["ev_ebitda_pct"].min() and out["ev_ebitda_pct"].max() < 100


def test_the_glossary_category_score_is_the_weighted_average_of_percentiles(cfg):
    prof = fe.metric_weight_profiles(cfg, "valuation")["generic"]
    live = {m: w for m, w in prof.items() if w > 0}
    pcts = {m: 10.0 + 20.0 * i for i, m in enumerate(live)}
    row = {f"{m}_pct": pcts.get(m, np.nan) for m in fe.METRIC_COLS}
    df = pd.DataFrame([row])
    out = fe.compute_category_scores(df, copy.deepcopy(cfg))
    want = sum(pcts[m] * w for m, w in live.items()) / sum(live.values())
    assert out.loc[0, "valuation_score"] == pytest.approx(want, abs=1e-6)


def test_the_glossary_points_example_and_rank_are_the_engines(page, cfg):
    w = cfg["factor_weights"]["valuation"]
    term = _glossary(page)["Points"]
    assert f"A Valuation score of 60 at a {w}% weight adds {60 * w / 100:.1f} points" in term
    df = pd.DataFrame({"Ticker": ["LOW", "TOP", "MID"], "Composite": [41.0, 77.5, 63.2]})
    assert fe.rank_stocks(df).set_index("Ticker")["Rank"].to_dict() == {"TOP": 1, "MID": 2, "LOW": 3}


# ---------------------------------------------------------------------------
# the dashboard's guide walks the same path
# ---------------------------------------------------------------------------

def test_the_guide_walks_a_first_visitor_from_the_ranking_to_one_stock_to_following_it():
    """Three steps, in the order a first visit goes: the ranking and its key, one stock's sheet,
    then comparing and following - and it hands on to the Methodology's summary."""
    import generate_dashboard as gd
    html = gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t", data_version="v")
    guide = html[html.index('<section class="guide"'):html.index("<!-- Top 5 Stocks -->")]
    steps = re.findall(r"<li><span class=\"guide-n\">\d</span><div><strong>([^<]+)</strong>", guide)
    assert steps == ["Start with the ranking.", "Open a stock.", "Compare and follow."], steps
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", guide))
    for named in ("Why it ranks here", "The workings", "Compare", "Holdings", "My holdings"):
        assert named in text, named
    assert "openScoreKey()" in guide and "openMethodology()" in guide
    assert "changes neither the score nor the rank" in text
    # the sheet's tools are called what the guide calls them
    assert 'id="mt-compare"' in html and ">Compare</button>" in html
    assert re.search(r'id="mt-hold"[^>]*>.*?Holdings</button>', html)
    assert len(text.split()) < 140, len(text.split())


# ---------------------------------------------------------------------------
# the summary's two computational sentences
# ---------------------------------------------------------------------------

def test_a_flag_changes_neither_the_score_nor_the_rank_in_the_engine(cfg):
    """A stock built to be a value trap and one built to be a growth trap keep the composite and
    the rank they had before the flags were applied."""
    n = 60
    cats = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]
    rng = np.random.default_rng(7)
    df = pd.DataFrame({c + "_score": rng.uniform(5, 95, n) for c in cats})
    df.loc[0, ["valuation_score", "quality_score", "momentum_score"]] = [99, 1, 1]
    df.loc[1, ["growth_score", "revisions_score"]] = [99, 1]
    df["Composite"] = rng.uniform(20, 80, n)
    before = fe.rank_stocks(df.copy())[["Composite", "Rank"]]
    flagged = fe.apply_growth_trap_flags(fe.apply_value_trap_flags(df.copy(), cfg), cfg)
    assert flagged.loc[0, "Value_Trap_Flag"] and flagged.loc[1, "Growth_Trap_Flag"]
    after = fe.rank_stocks(flagged)[["Composite", "Rank"]]
    pd.testing.assert_frame_equal(before, after)


def _summary(tmp_path, lo, hi, better, mean=0.0):
    p = tmp_path / f"summary_{lo}_{hi}_{better}.json"
    p.write_text(json.dumps({"better": better, "baseline_ic": {"pooled": mean},
                             "baseline_ic_ci95": {"pooled": [lo, hi]},
                             "months": {"pooled": 176}}), encoding="utf-8")
    return p


def test_the_history_line_says_no_better_than_chance_only_when_the_summary_does(tmp_path):
    line = run_screener.history_test_line
    chance = "did no better than chance"
    assert chance in line(_summary(tmp_path, -0.02, 0.02, 0))
    # an interval clear of zero, either way, or an alternative that cleared the bar: no verdict
    above = line(_summary(tmp_path, 0.01, 0.06, 0, mean=0.035))
    assert chance not in above and "+0.035" in above and "+0.010 to +0.060" in above
    assert chance not in line(_summary(tmp_path, -0.06, -0.01, 0, mean=-0.035))
    assert chance not in line(_summary(tmp_path, -0.02, 0.02, 1))
    missing = line(tmp_path / "absent.json")
    assert "has not been shown" in missing and chance not in missing
    for text in (above, missing):
        assert stock_summary.advice_terms_in(text) == []


def test_the_published_summary_line_is_the_committed_summarys(page):
    s = json.loads((ROOT / "research" / "experiments" / "summary.json").read_text(encoding="utf-8"))
    assert run_screener._typeset_md(run_screener.history_test_line()) in page
    lo, hi = s["baseline_ic_ci95"]["pooled"]
    said = "did no better than chance" in page
    assert said == (lo <= 0 <= hi and int(s["better"]) == 0)
    assert f"Rerun on {s['months']['pooled']} months" in page or not said
