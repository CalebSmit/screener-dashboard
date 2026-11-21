"""The two narrower tests behind CLAUDE.md 0.15, as the home page and the research page state them.

`research/2026-11-19-what-the-ranking-is-for.md` pre-registered (section 3) a screen test (A: do the
top 40 avoid the next month's bottom decile?) and a slow-forecast test (B: rank IC at 6 and 12
months), and section 5 asked for both results on the page. Claims `page.hero_screen` and
`research.screen_and_horizon` (2026-11-21). Every sentence here is driven by the measurement's own
file, so each test builds a file by hand and checks the words follow it - including the cases the
2026-11-21 run did not produce.
"""
import copy
import json
import re
from pathlib import Path

import generate_dashboard as g
import research_page as rp
import stock_summary

ROOT = Path(__file__).resolve().parent.parent

BASE = {
    "counts": True,
    "A": {"holds": False, "pooled": {"ci": [0.0909, 0.1048]},
          "per_era": {"2012-2019": {"mean": 0.0979, "upper_one_sided_95": 0.1061},
                      "2020-2026": {"mean": 0.0978, "upper_one_sided_95": 0.1071}}},
    "B": {"6m": {"verdict": "undecidable", "windows": 29, "pooled": {"mean": -0.007, "ci": [-0.047, 0.033]},
                 "detectable_80pct": 0.057, "same_sign": False},
          "12m": {"verdict": "undecidable", "windows": 14, "pooled": {"mean": 0.004, "ci": [-0.042, 0.051]},
                  "detectable_80pct": 0.065, "same_sign": False},
          "holds": False},
    "D_is_the_answer": True,
}


def _file(tmp_path, d):
    p = tmp_path / "m.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def _with(**changes):
    d = copy.deepcopy(BASE)
    for path, value in changes.items():
        node = d
        keys = path.split("__")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value
    return d


def test_a_result_whose_control_was_not_seen_is_not_published(tmp_path):
    assert g.screen_verdict(_file(tmp_path, _with(counts=False))) == ""
    assert g.screen_verdict(tmp_path / "missing.json") == ""
    assert rp.purpose_section(g.screen_test_reading(_file(tmp_path, _with(counts=False)))) == ""


def test_the_screen_words_follow_the_rule_and_the_interval(tmp_path):
    as_chance = g.screen_verdict(_file(tmp_path, BASE))
    assert "as likely as any stock" in as_chance
    held = g.screen_verdict(_file(tmp_path, _with(A__holds=True, A__pooled={"ci": [0.05, 0.08]})))
    assert "less likely than chance" in held and "as likely" not in held
    worse = g.screen_verdict(_file(tmp_path, _with(A__pooled={"ci": [0.11, 0.13]})))
    assert "more likely than chance" in worse
    # below 10% on the pooled interval but the rule (both eras) not met: never "less likely"
    near = g.screen_verdict(_file(tmp_path, _with(A__pooled={"ci": [0.085, 0.099]})))
    assert "about as likely" in near and "less likely than chance" not in near


def test_the_horizon_words_follow_each_verdict(tmp_path):
    assert "over six and twelve months the history is too short to tell" in g.screen_verdict(_file(tmp_path, BASE))
    mixed = g.screen_verdict(_file(tmp_path, _with(B__12m__verdict="holds")))
    assert "over six months the history is too short to tell" in mixed
    assert "over twelve it ordered returns better than chance in both periods" in mixed
    failed = g.screen_verdict(_file(tmp_path, _with(B__6m__verdict="fails", B__12m__verdict="fails")))
    assert "over six and twelve months it showed no edge the test could have seen" in failed


def test_the_research_section_prints_the_files_own_figures(tmp_path):
    sec = rp.purpose_section(g.screen_test_reading(_file(tmp_path, BASE)))
    assert '<section id="screen-and-horizon">' in sec
    assert "9.8% of the time in 2012&ndash;2019 and 9.8% in 2020&ndash;2026" in sec
    assert "it was 10.6% and 10.7%" in sec
    assert "&minus;0.007</span> across 29 separate windows (95% interval &minus;0.047 to +0.033)" in sec
    assert "+0.004</span> across 14 separate windows (95% interval &minus;0.042 to +0.051)" in sec
    assert "0.057</span> at six months and" in sec and "0.065</span> at twelve months" in sec
    assert "At neither horizon did the two periods agree on the sign." in sec
    assert "teaching tool" in sec


def test_the_research_section_claims_only_what_the_file_says(tmp_path):
    # periods that agree on the sign: no sentence saying they did not
    agree = _with(B__6m__same_sign=True)
    assert "agree on the sign" not in rp.purpose_section(g.screen_test_reading(_file(tmp_path, agree)))
    # nothing undecidable: no power sentence
    decided = _with(B__6m__verdict="fails", B__12m__verdict="fails")
    assert "could reliably see" not in rp.purpose_section(g.screen_test_reading(_file(tmp_path, decided)))
    # not D: no teaching-tool conclusion
    not_d = _with(D_is_the_answer=False)
    assert "teaching tool" not in rp.purpose_section(g.screen_test_reading(_file(tmp_path, not_d)))


def test_the_committed_measurement_is_what_both_pages_read():
    """The real file: the run counted, and the hero's clause is the research page's clause."""
    r = g.screen_test_reading()
    assert r is not None, "the 0.15 measurement is missing or its positive control was not seen"
    assert g.horizon_clause(r) in rp.purpose_section(r)
    assert g.screen_verdict().endswith(g.horizon_clause(r) + ".")


def test_the_committed_research_page_carries_the_section():
    page = (ROOT / "research.html").read_text(encoding="utf-8")
    assert rp.purpose_section(g.screen_test_reading()).strip() in page


def test_no_advice_language():
    assert not stock_summary.advice_terms_in(g.screen_verdict())
    assert not stock_summary.advice_terms_in(re.sub(r"<[^>]+>", " ", rp.purpose_section(g.screen_test_reading())))
