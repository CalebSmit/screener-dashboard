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


# --------------------------------------------------------------------------- the 2026-10-16 audit
import copy  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _page_text(page: str) -> str:
    text = re.sub(r"<style.*?</style>", " ", page, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text))


def test_the_size_sentence_claims_no_more_than_the_coverage_data_can_show():
    """Item 1: coverage rises with the calendar almost exactly, so the across-era relation cannot tell
    survivorship from time. Recomputed here from the ledger and the committed coverage file, the page
    must print the within-era correlations and call survivorship a hypothesis it cannot separate."""
    page = rp.PAGE.read_text(encoding="utf-8")
    assert "disappears where coverage is fullest" not in page
    assert "follows a gap in the data rather than the market" not in page
    latest = _latest()
    cov = pd.Series({m: v["coverage"] for m, v in json.loads(rp.COVERAGE.read_text(encoding="utf-8"))["months"].items()})
    df = pd.DataFrame({"s": pd.Series(latest["only-size"]["ic_by_month"]),
                       "b": pd.Series(latest["baseline"]["ic_by_month"]), "c": cov}).dropna().sort_index()
    assert len(df) == 176
    r_time = np.corrcoef(df["c"], np.arange(len(df)))[0, 1]
    assert r_time > 0.99                                    # coverage is, for practical purposes, the calendar
    pre = df[df.index <= "2019-12-31"]
    r_s, r_b = pre["s"].corr(pre["c"]), pre["b"].corr(pre["c"])
    r_d = (pre["s"] - pre["b"]).corr(pre["c"])
    t_d = r_d * np.sqrt((len(pre) - 2) / (1 - r_d ** 2))
    assert f">{rp.num(r_s)}</span> and <span class=\"num\">{rp.num(r_b)}</span>" in page
    assert f">{r_time:.3f}</span>" in page
    text = _page_text(page)
    assert "Survivorship could explain it" in text and "cannot separate that explanation from the passage of time" in text
    if abs(t_d) < 2:
        assert "about as much as the current weighting's did" in text


def test_every_outcome_known_screen_is_marked_and_the_pre_registration_sentence_is_qualified():
    """Item 2: the seven screens whose 2020-2026 outcome was committed before registration carry a
    dagger wherever they are named, and step 1 says only their 2012-2019 test is blind."""
    page = rp.PAGE.read_text(encoding="utf-8")
    flagged = [e for e in X.load_registry() if e.get("outcome_known_at_registration")]
    assert len(flagged) == 7
    for e in flagged:
        assert page.count(rp.display_label(e)) >= 2, e["id"]          # the chart and the table
        assert rp.display_label(e).endswith(rp.KNOWN_MARK)
    step1 = page[page.index("before</em> it was run."):]
    step1 = step1[:step1.index("</li>")]
    assert "seven of them" in step1 and "only the 2012&ndash;2019 test is blind" in step1


def test_the_page_says_the_pairing_is_on_the_shared_stocks_and_states_the_universe_difference():
    """Method version 2 (2026-10-17, plan/research-engine.md 8a) reversed what this guards.

    Until then the pairing really was by month only, and this test rightly forbade "on the same
    stocks" and required the page to say so. Now both sides are recomputed on each month's
    intersection, so the page must say *that*, must still state how many stocks each side could
    rank (the universe difference has not gone away - it is what makes the intersection smaller),
    and must state the overlap the comparison was actually measured over, read from the ledger.
    """
    page = rp.PAGE.read_text(encoding="utf-8")
    per = json.loads(rp.PER_CONFIG.read_text(encoding="utf-8"))["configs"]
    assert "paired by month and measured on the stocks the two rankings share" in page
    assert "The pairing is by month:" not in page and "judged on fewer stocks" not in page
    assert f'ranked <span class="num">{per["baseline"]["sample"]["mean_stocks"]:.0f}</span> stocks a month' in page
    assert "pairing on the overlap" in page or "in the overlap" in page


def test_every_headline_figure_carries_its_interval_from_the_ledger():
    """Item 4: each era's IC with its 95% interval from the ledger's own standard error (Student's t),
    and the pooled 176-month estimate with its interval and what it could detect."""
    page = rp.PAGE.read_text(encoding="utf-8")
    base = _latest()["baseline"]
    for era, lab in (("presample", "2012&ndash;2019"), ("all", "2020&ndash;2026")):
        st = base["summary"][era]["ic"]
        q = X.t_critical(0.05, st["n"] - 1)
        lo, hi = round(st["mean"] - q * st["se"], 4), round(st["mean"] + q * st["se"], 4)
        assert (f'>{rp.num(st["mean"])}</span> over {lab} (95% interval '
                f'<span class="num">{rp.num(lo)} to {rp.num(hi)}</span>)') in page
    series = pd.Series(base["ic_by_month"])
    m, se = round(series.mean(), 5), round(series.std(ddof=1) / np.sqrt(len(series)), 5)
    q = X.t_critical(0.05, len(series) - 1)
    assert f'Pooled over all <span class="num">{len(series)}</span> months it was <span class="num">{rp.num(m)}</span>' in page
    assert f'<span class="num">{rp.num(round(m - q * se, 4))} to {rp.num(round(m + q * se, 4))}</span>' in page
    assert f'<span class="num">{X.detectable(se, 1, len(series)):.3f}</span> at a t-statistic of 3' in page
    # The formatter, which is the half of this the loop above cannot cover: a true minus sign, a
    # leading + on a positive, three decimals. NOT a literal interval - both lines here used to
    # pin the 2026-10-20 audit's own figures ("&minus;0.023 to +0.017" and "&minus;0.027 to
    # +0.027"), and on 2026-10-24 a legitimate data fix moved the record those intervals are
    # computed from, so both went red while the page was perfectly correct. The derived loop above
    # is the stricter check anyway: it reads the ledger and fails if the page disagrees with it at
    # *any* data version, where a literal can only say whether the data is the data it was written
    # against. A tripwire that fires when nothing is wrong is one that gets ignored (the same
    # lesson as the 2026-10-18 dirty-marker bug, plan/research-engine.md).
    assert rp.num(-0.0229) == "&minus;0.023"
    assert rp.num(0.0173) == "+0.017"
    assert rp.num(0.0) == "0.000"


def test_the_band_and_the_pass_bar_are_named_separately_and_a_t_of_3_5_is_never_called_indistinguishable():
    """Item 4: the band is the 80%-power detectable difference; the pass bar is drawn on its own,
    and a row beyond the bar is described as beyond it."""
    page = rp.PAGE.read_text(encoding="utf-8")
    assert "cannot be told apart from zero" not in page
    assert "smallest difference the test could reliably detect" in page and "pass bar" in page
    assert page.count('class="tick"') == 2 * page.count('class="dot-row"')
    reg, latest = X.load_registry(), copy.deepcopy(_latest())
    target = "drop-growth"
    latest[target]["summary"]["presample"]["ic_vs_baseline"].update({"mean": 0.035, "se": 0.01, "t": 3.5, "n": 96})
    built = rp.build(reg, latest, X.summary_for_page(reg, latest), X.detectable, X.load_evaluations())
    aria = re.search(r'aria-label="Without Growth: [^"]*"', built).group(0)
    assert "beyond the pass bar" in aria and "inside" not in aria
    assert "cannot be told apart" not in built


def test_the_trading_rule_sentence_follows_each_period_s_sign_and_t():
    """Item 5: the comparison is the average scored stock (survivors), labelled as such, with
    intervals; the sentence is generated, so it cannot say "in every period" for a gap that is
    distinguishable in one period only."""
    page = rp.PAGE.read_text(encoding="utf-8")
    assert "Against the equal-weighted S&amp;P&nbsp;500" not in page and "in every period" not in page
    assert "Against the average scored stock" in page and "members still listed today" in page
    b = _latest()["bot-rule-v1.1"]["summary"]
    for k, lab in (("presample", "2012&ndash;2019"), ("discovery", "2020 &ndash; mid-2023"), ("holdout", "mid-2023 &ndash; 2026")):
        st = b[k]["excess"]
        word = "below" if st["mean"] < 0 else "above"
        verdict = "larger than chance readily produces" if X.p_value(st) < 0.05 else "within the noise"
        assert re.search(rf"In {re.escape(lab)} it returned <span class=\"num\">[0-9.]+%</span> a month {word} "
                         rf"that average \(t&nbsp;{st['t']:+.2f}\), a gap {verdict}\.", page), k
    flipped = copy.deepcopy(b)
    flipped["discovery"]["excess"].update({"t": -0.5, "mean": -0.001})
    assert "2020 &ndash; mid-2023 it returned" in rp.bot_sentence(flipped, [("discovery", "2020 &ndash; mid-2023")], X.p_value)
    assert "within the noise" in rp.bot_sentence(flipped, [("discovery", "2020 &ndash; mid-2023")], X.p_value)


def test_the_stop_loss_arm_is_shown_beside_the_rule_with_a_verdict_that_can_say_it_failed():
    """OWNER_FOCUS 2026-10-22: one pre-registered stop, reported beside the unstopped rule.

    The verdict sentence is generated from the two summaries, so it is checked by feeding it
    numbers - a stop that cut both figures everywhere, one that cut neither, and one that cut them
    in some periods only. A sentence that could only ever say "the prediction held" would be
    checking plumbing, not the claim."""
    page = rp.PAGE.read_text(encoding="utf-8")
    sev = next(e for e in X.load_evaluations() if e["rule"].get("stop_loss_pct"))
    latest = _latest()
    assert sev["id"] in latest, "the registered stop arm has no ledger row - rerun the evaluations"
    import html as _html
    assert "The same rule with a 10% stop-loss" in page
    assert "never searched over" in page            # one level, stated on the page itself
    assert "SSRN&nbsp;2407199" in page
    # The page states the construction in its own words, because the registered text uses the word
    # BANNED_TERMS forbids and its prediction carries a phrase this page's own tripwire forbids -
    # and a registered prediction may never be edited once it has run. So: not quoted verbatim,
    # all four differences still stated, and the registered text pointed at by path.
    assert _html.escape(sev["construction_note"]) not in page
    assert _html.escape(sev["prediction"]) not in page
    for difference in ("lowest price", "intraday range", "triggering close",
                       "Treasury-bill interest", "long-short momentum sort"):
        assert difference in page, difference
    assert "research/experiments/registry.yaml" in page
    assert "research/2026-10-22-stop-loss-on-the-bots-rule.md" in page
    assert "in all three periods" in page           # what was predicted, in page words
    assert f"{int(sev['rule']['stop_loss_pct'])}% or more below its price at the start" in page
    assert f"a further {int(sev['rule']['stop_loss_pct']) // 2}% down" in page

    periods = [("presample", "2012&ndash;2019"), ("discovery", "2020 &ndash; mid-2023"),
               ("holdout", "mid-2023 &ndash; 2026")]
    base = {k: {"worst_month": -0.10, "max_drawdown": -0.20, "turnover_a_year": 1.0,
                "positions_stopped": 0} for k, _ in periods}
    better = {k: dict(v, worst_month=-0.05, max_drawdown=-0.10, turnover_a_year=2.0,
                      positions_stopped=10) for k, v in base.items()}
    worse = {k: dict(v, worst_month=-0.20, max_drawdown=-0.30, turnover_a_year=2.0,
                     positions_stopped=10) for k, v in base.items()}
    mixed = dict(worse, discovery=better["discovery"])
    assert "The prediction held" in rp.stop_verdict(base, better, periods)
    assert "prediction failed" in rp.stop_verdict(base, worse, periods)
    half = rp.stop_verdict(base, mixed, periods)
    assert "half failed" in half and "2020 &ndash; mid-2023" in half
    # one of the two figures better and the other worse is not "it worked"
    one = {k: dict(v, worst_month=-0.05, max_drawdown=-0.30) for k, v in base.items()}
    assert "prediction failed" in rp.stop_verdict(base, one, periods)
    assert "2.0x" in rp.stop_verdict(base, better, periods)

    # and the page says whichever of those the measured rows actually support
    b, s = latest["bot-rule-v1.1"]["summary"], latest[sev["id"]]["summary"]
    assert rp.stop_verdict(b, s, periods) in page


def test_within_sector_screens_are_labelled_so_on_the_page_with_their_note():
    """Item 6: single-metric screens and Size on its own are within-sector versions of market-wide
    papers; the page labels them and carries the note."""
    page = rp.PAGE.read_text(encoding="utf-8")
    import html as _html
    for e in X.load_registry():
        if e.get("construction_note"):
            assert rp.display_label(e).startswith(_html.escape(rp.label(e["id"]) + rp.WITHIN_SECTOR)), e["id"]
            assert rp.display_label(e) in page, e["id"]
            assert _html.escape(e["construction_note"]) in page, e["id"]
    for name in ("12-1 month return alone (within sector)", "Size alone (within sector)",
                 "Low volatility alone (within sector)", "Nearness to the 52-week high (within sector)"):
        assert name in page


# --------------------------------------------------------------------------- reading it (2026-10-27)
def test_the_contents_list_names_every_section_in_order_and_nothing_else():
    """Thirteen thousand pixels on a phone, and until 2026-10-27 its fifth section could be reached
    only by scrolling past the first four. The list is read off the generator's own sections."""
    page = _rebuilt()
    sections = re.findall(r'<section id="([a-z-]+)">\s*<h2>(.*?)</h2>', page)
    start = page.index('<nav class="toc"')
    toc = page[start:page.index("</nav>", start)]
    assert re.findall(r'<a href="#([a-z-]+)">(.*?)</a>', toc) == sections
    assert len(sections) >= 5 and page.count("<section") == len(sections)    # every section is addressable
    assert page.index('class="fine terms"') < start < page.index('id="how-tested"')
    foot = page[page.index('<nav class="next-nav"'):]
    assert 'href="#top"' in foot and 'href="lab.html"' in foot and 'href="index.html"' in foot
    assert 'id="top"' in page


def test_on_a_phone_its_controls_are_44_px_and_on_paper_it_is_ink():
    import pytest
    sync_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")
    with sync_api.sync_playwright() as pw:
        try:
            browser = pw.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        try:
            page = browser.new_page(viewport={"width": 375, "height": 812})
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(rp.PAGE.as_uri(), wait_until="load", timeout=60000)
            page.wait_for_selector(".toc a", timeout=8000)
            short = page.evaluate("""[...document.querySelectorAll('.bar nav a, .toc a, details summary, .next-nav a')]
                .map(e => [e.textContent.trim().slice(0, 30), e.getBoundingClientRect().height]).filter(x => x[1] < 43.5)""")
            assert short == []
            assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
            page.click(".toc a[href='#limits']")                        # a contents link lands on its section
            page.wait_for_function("location.hash === '#limits'", timeout=8000)
            page.wait_for_function("Math.abs(document.getElementById('limits').getBoundingClientRect().top) < 200", timeout=8000)
            page.emulate_media(media="print")
            assert page.evaluate("getComputedStyle(document.body).backgroundColor") == "rgb(255, 255, 255)"
            assert page.evaluate("getComputedStyle(document.querySelector('h1')).color") == "rgb(17, 17, 17)"
            for sel in (".bar", ".toc", ".next-nav"):
                assert page.evaluate(f"getComputedStyle(document.querySelector({sel!r})).display") == "none", sel
            assert errors == []
        finally:
            browser.close()
