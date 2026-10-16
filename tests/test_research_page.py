"""The public research page (``research.html``, 2026-10-15): what the research engine found.

Pinned: the committed page is exactly what ``research_page.build`` makes from the committed ledger
(it is generated - rule 10 - and a hand-edit or a stale page fails here); its headline numbers are
the engine's published summary; it gives no advice; its links resolve; nothing in production imports
the generator; and the dashboard sends readers to it and to the Factor Lab from the hero, the
footer, the search palette and the track record.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import research_page as rp  # noqa: E402
import v2_experiments as X  # noqa: E402
from stock_summary import advice_terms_in  # noqa: E402


def _latest() -> dict:
    latest = {}
    for line in X.LEDGER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[r["id"]] = r
    return latest


def _rebuilt() -> str:
    reg, latest = X.load_registry(), _latest()
    return rp.build(reg, latest, X.summary_for_page(reg, latest), X.detectable, X.load_evaluations())


def test_the_committed_page_is_what_the_generator_makes_from_the_ledger():
    assert rp.PAGE.read_text(encoding="utf-8") == _rebuilt(), \
        "research.html is stale or hand-edited: run `python v2_experiments.py report` and commit it"


def test_its_headline_numbers_are_the_engine_s_summary():
    page = rp.PAGE.read_text(encoding="utf-8")
    s = json.loads(X.SUMMARY.read_text(encoding="utf-8"))
    assert f">{rp.num(s['baseline_ic']['presample'])}</span> over 2012&ndash;2019" in page
    assert f">{rp.num(s['baseline_ic']['all'])}</span> over 2020&ndash;2026" in page
    assert f"of the {s['alternatives']} alternative weightings cleared the bar" in page
    assert page.count('class="dot-row"') == s["alternatives"]


def test_signs_and_zero_are_printed_honestly():
    assert rp.num(0.00003) == "0.000" and rp.num(-0.0028) == "&minus;0.003" and rp.num(0.028) == "+0.028"
    assert rp.months_needed(0.05, 0.12, 3.0) == round(((3.0 + 0.84) * 0.12 / 0.05) ** 2)


def test_the_page_gives_no_advice_and_says_so():
    page = rp.PAGE.read_text(encoding="utf-8")
    text = re.sub(r"<style.*?</style>", " ", page, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    assert advice_terms_in(text) == []
    assert "It is not investment advice" in text
    assert "has not been shown to predict returns" in text


def test_its_links_resolve():
    page = rp.PAGE.read_text(encoding="utf-8")
    for href in re.findall(r'href="([^"#:]+\.html)"', page):
        assert (ROOT / href).exists(), href
    assert 'href="index.html"' in page and 'href="lab.html"' in page


def test_no_production_module_imports_the_generator():
    pat = re.compile(r"^\s*(import research_page|from research_page)\b", re.M)
    for p in ROOT.glob("*.py"):
        if p.name in ("research_page.py", "v2_experiments.py", "conftest.py"):
            continue
        assert not pat.search(p.read_text(encoding="utf-8", errors="replace")), p.name


def test_the_dashboard_sends_readers_to_the_evidence_and_the_lab():
    import generate_dashboard as g
    html = g.generate_html(methodology_html="", data_timestamp="2026-10-15 02:00 UTC", data_version="x")
    hero = html[html.index('class="hero-actions"'):]
    hero = hero[:hero.index("</div>")]
    assert 'href="research.html"' in hero and 'href="lab.html"' in hero
    footer = html[html.index('class="footer-links"'):]
    assert 'href="research.html"' in footer[:footer.index("</nav>")]
    assert "window.location.href = 'research.html'" in html                      # the search palette
    assert 'the ranking tested on 2012&ndash;2026 history</a>' in html           # the track record
