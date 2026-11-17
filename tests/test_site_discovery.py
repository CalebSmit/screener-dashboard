"""How the site is found, previewed and read by machines - kept true (2026-11-17).

Owner-run session, prompted by an article on running a site's search presence with an agent. Most of
its technical basics were already here (descriptions, sharing cards, a 404 page, cross-links). What
was missing, and what this pins:

* **one address per page.** Pages serves the dashboard as ``/``, ``/index.html`` and
  ``/dashboard.html``; each page now names its canonical address, equal to its ``og:url``.
* **``sitemap.xml`` lists exactly the canonical pages** - it is only read if submitted in Search
  Console (a project site's ``robots.txt`` would sit where no crawler looks), so the one thing it
  must never be is wrong.
* **the structured data repeats only what the page already says** - its description *is* the
  meta description, its url the canonical.
* **a page's title says what its heading says** (the research page's did not).
* **``llms.txt`` points only at files that exist**, names the eight categories ``config.yaml``
  weights, and says the tool is not investment advice.

The committed pages are read here, not the generators, because ``tests/test_rebuild_page.py`` and
``tests/test_research_page.py`` already fail if either committed page differs from its generator.
Nothing here reads a figure the data loop changes, so the 02:00 run cannot turn it red.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SITE = "https://calebsmit.github.io/screener-dashboard/"
PAGES = {"index.html": SITE, "dashboard.html": SITE,
         "lab.html": SITE + "lab.html", "research.html": SITE + "research.html"}


def _head(name: str) -> str:
    s = (ROOT / name).read_text(encoding="utf-8")
    return s[:s.index("</head>")]


def _attr(head: str, pattern: str) -> list[str]:
    return [html.unescape(m) for m in re.findall(pattern, head)]


def test_every_page_names_one_canonical_address_equal_to_its_sharing_url():
    for name, want in PAGES.items():
        head = _head(name)
        canon = _attr(head, r'<link rel="canonical" href="([^"]+)"')
        assert canon == [want], (name, canon)
        assert _attr(head, r'<meta property="og:url" content="([^"]+)"') == [want], name


def test_the_sitemap_lists_exactly_the_canonical_pages():
    xml = re.sub(r"<!--.*?-->", "", (ROOT / "sitemap.xml").read_text(encoding="utf-8"), flags=re.S)
    assert sorted(re.findall(r"<loc>([^<]+)</loc>", xml)) == sorted(set(PAGES.values()))
    assert "<lastmod>" not in xml


def test_the_404_page_stays_out_of_the_index():
    assert '<meta name="robots" content="noindex">' in _head("404.html")


def test_the_structured_data_says_only_what_the_page_says():
    head = _head("index.html")
    blocks = re.findall(r'<script type="application/ld\+json">(.*?)</script>', head, re.S)
    assert len(blocks) == 1
    ld = json.loads(blocks[0])
    assert ld["@type"] == "WebApplication" and ld["url"] == SITE
    desc = _attr(head, r'<meta name="description" content="([^"]+)"')
    assert [ld["description"]] == desc
    title = _attr(head, r"<title>([^<]+)</title>")[0]
    assert title.startswith(ld["name"])
    # every claim the block makes has to be visible on the page; these are not
    for unseen in ("isAccessibleForFree", "aggregateRating", "offers", "author", "review"):
        assert unseen not in ld


def test_a_page_title_says_what_its_heading_says():
    for name in ("research.html", "lab.html", "index.html"):
        s = (ROOT / name).read_text(encoding="utf-8")
        title = html.unescape(re.search(r"<title>([^<]+)</title>", s).group(1))
        h1 = html.unescape(re.sub(r"<[^>]+>", "", re.search(r"<h1[^>]*>(.*?)</h1>", s, re.S).group(1)))
        h1 = re.sub(r"\s+", " ", h1).strip()
        assert title.split(" · ")[0] in (h1, h1.rstrip("."), "Factor Lab"), (name, title, h1)


# --------------------------------------------------------------------------- llms.txt
LLMS = ROOT / "llms.txt"


def _links() -> list[str]:
    return re.findall(r"\]\((https?://[^)]+)\)", LLMS.read_text(encoding="utf-8"))


def test_llms_txt_points_only_at_files_that_exist():
    links = _links()
    assert len(links) >= 8
    for url in links:
        if url.startswith(SITE):
            rel = url[len(SITE):] or "index.html"
            assert (ROOT / rel).is_file(), url
        else:
            m = re.fullmatch(r"https://github.com/CalebSmit/screener-dashboard(?:/(?:tree|blob)/main/(.+))?", url)
            assert m, url
            if m.group(1):
                assert (ROOT / m.group(1)).exists(), url


def test_llms_txt_names_the_weighted_categories_and_disclaims_advice():
    text = LLMS.read_text(encoding="utf-8")
    assert text.startswith("# Multi-Factor Screener\n")
    weights = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["factor_weights"]
    summary = text.split("\n> ", 1)[1].split("\n", 1)[0]
    named = re.search(r"eight categories - (.+?) - each", summary).group(1)
    names = [n.strip().replace("analyst revisions", "revisions")
             for n in re.split(r",|\band\b", named) if n.strip()]
    assert sorted(names) == sorted(weights) and len(weights) == 8
    assert "not investment advice" in summary


def test_llms_txt_describes_the_payload_the_way_it_is_built():
    """The payload's opening and the two keys it names - not its size or contents, which move."""
    with open(ROOT / "dashboard_data.js", encoding="utf-8") as fh:
        start = fh.read(200)
    assert start.startswith('window.SCREENER_DATA = {"kpis": {"run_timestamp": ')
    text = LLMS.read_text(encoding="utf-8")
    assert "`window.SCREENER_DATA = {...}`" in text and "`kpis.run_timestamp`" in text
    assert "`table_data`" in text
    assert not re.search(r"\b\d+(\.\d+)? ?MB\b", text)          # a size here would go stale
