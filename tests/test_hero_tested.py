"""The home page says what the 15-year test found, beside the track record (2026-11-19).

The hero led with a bold nine-month "+10.5%" while `research.html` says no edge has been shown.
`research/2026-11-19-fresh-eyes-review.md` finding 2; claim `page.hero_tested`.
"""
import json
import re
from pathlib import Path

import generate_dashboard as g

ROOT = Path(__file__).resolve().parent.parent


def _summary(tmp_path, lo, hi, better, mean=0.0, months=176):
    p = tmp_path / "summary.json"
    p.write_text(json.dumps({"baseline_ic": {"pooled": mean},
                             "baseline_ic_ci95": {"pooled": [lo, hi]},
                             "months": {"pooled": months}, "better": better}), encoding="utf-8")
    return p


def test_no_edge_only_while_the_summary_says_so(tmp_path):
    assert "no measurable edge" in g.history_verdict(_summary(tmp_path, -0.018, 0.015, 0))
    # an interval clear of zero, or an alternative that beat it: state the figure, no verdict
    above = g.history_verdict(_summary(tmp_path, 0.004, 0.06, 0, mean=0.032))
    assert "no measurable edge" not in above and "+0.032" in above and "+0.004 to +0.060" in above
    assert "no measurable edge" not in g.history_verdict(_summary(tmp_path, -0.01, 0.02, 1))
    assert "not yet been tested" in g.history_verdict(tmp_path / "missing.json")


def test_the_committed_page_carries_the_committed_summarys_verdict():
    page = (ROOT / "index.html").read_text(encoding="utf-8")
    m = re.search(r'<p class="hero-note hero-tested" id="hero-tested">(.*?) <a href="research.html">', page)
    assert m, "the hero's history-test line is missing from index.html"
    # the one-month verdict, then (since 2026-11-21) the two narrower tests' sentence when they count
    assert m.group(1) == g.hero_tested_line()
    assert m.group(1).startswith(g.history_verdict())


def test_the_line_carries_no_advice_language():
    import stock_summary
    assert not stock_summary.advice_terms_in(g.history_verdict())
