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
    # 2026-10-29: this line read `== round(((3.0 + 0.84) * 0.12 / 0.05) ** 2)`, which is the
    # function's own expression copied - it passes whatever that expression becomes
    # (plan/claims-mutation-audit.md, row 59). The figure is a literal now: (3.84 x 0.12 / 0.05)^2
    # = 84.93 -> 85, against 36 with the power term dropped and 23 at the 5% level.
    assert rp.months_needed(0.05, 0.12, 3.0) == 85
    assert rp.months_needed(0.05, 0.12, 1.96) == 45


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
import math  # noqa: E402

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
                         rf"that average \(t&nbsp;{re.escape(rp.num(st['t'], 2))}\), a gap {verdict}\.", page), k
    flipped = copy.deepcopy(b)
    flipped["discovery"]["excess"].update({"t": -0.5, "mean": -0.001})
    assert "2020 &ndash; mid-2023 it returned" in rp.bot_sentence(flipped, [("discovery", "2020 &ndash; mid-2023")], X.p_value)
    assert "within the noise" in rp.bot_sentence(flipped, [("discovery", "2020 &ndash; mid-2023")], X.p_value)


def test_the_trading_rules_monthly_figure_is_the_ledgers_own_number():
    """Closes `bot-monthly-excess-figure` (claims-mutation audit, opened 2026-10-29).

    The test above matches the percentage as ``[0-9.]+%``, so the sign, the words and the
    t-statistic were pinned and the published figure itself was a wildcard: it could print any
    number at all, in any of the three periods, and nothing on this page or anywhere in the suite
    would have failed. Here each period's figure is derived from the ledger row independently -
    ``abs(mean) x 100`` to two decimals, the magnitude, with the direction carried by the word -
    and the whole span, markup included, must appear.

    It reads the *generated* page, not the committed one (the 2026-10-29 structural fix): a
    check against the committed file is green until someone regenerates, which is a drift
    tripwire wearing a value check's name.
    """
    page = _rebuilt()
    b = _latest()["bot-rule-v1.1"]["summary"]
    seen = 0
    for k, lab in (("presample", "2012&ndash;2019"), ("discovery", "2020 &ndash; mid-2023"),
                   ("holdout", "mid-2023 &ndash; 2026")):
        mean = b[k]["excess"]["mean"]
        want = f'In {lab} it returned <span class="num">{abs(mean) * 100:.2f}%</span> a month'
        assert want in page, f"{k}: expected {want!r}"
        # The magnitude is printed and the direction is the word, so a sign flip must not be
        # able to leave the figure looking right.
        assert f'>{abs(mean) * 100:.2f}%</span> a month {"below" if mean < 0 else "above"}' in page
        seen += 1
    assert seen == 3, "all three periods must be stated"


def test_the_survivorship_coverage_percentages_are_the_two_years_means():
    """Closes `bot-coverage-by-year` (opened 2026-10-29). The sentence under the trading-rule
    table is this page's caveat about survivorship - the two percentages are the point of it, and
    `test_the_trading_rule_sentence...` pinned only the words 'members still listed today'.

    The generated page, for the same reason as the test above."""
    import pytest
    page = _rebuilt()
    if not rp.COVERAGE.exists():
        pytest.skip(f"MISSING DATA: {rp.COVERAGE.name}")
    mc = (json.loads(rp.COVERAGE.read_text(encoding="utf-8")) or {}).get("months") or {}
    if not mc:
        pytest.skip("MISSING DATA: the coverage-by-month measurement has no months")
    ys = sorted({m[:4] for m in mc})
    first, last = (float(np.mean([v["coverage"] for m, v in mc.items() if m.startswith(y)])) * 100
                   for y in (ys[0], ys[-1]))
    assert (f'Those are the members still listed today: <span class="num">{first:.0f}%</span> of '
            f"each month's members in {ys[0]}, rising to "
            f'<span class="num">{last:.0f}%</span> in {ys[-1]}') in page
    assert first < last, "the sentence says 'rising to' - it must be generated, not written down"


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


# --------------------------------------------------------------------------- round two (2026-10-27)
def test_the_plain_answer_is_given_only_while_every_interval_includes_zero():
    """Claim ``research.plain_answer``: the student's-words line under the headline says no edge has
    been shown, so it may appear only while every interval (each era and pooled) includes zero - and
    must go, with "Not measurably.", the day one does not."""
    reg, latest = X.load_registry(), _latest()
    summary = X.summary_for_page(reg, latest)
    ci = summary["baseline_ic_ci95"]
    every_zero = all(lo <= 0 <= hi for lo, hi in ci.values())
    page = rp.build(reg, latest, summary, X.detectable, X.load_evaluations())
    assert ("In plain terms:" in page) == every_zero == ("<strong>Not measurably.</strong>" in page)
    if every_zero:                                    # the line sits under the headline, before the statistics
        assert page.index("<strong>Not measurably.</strong>") < page.index("In plain terms:") \
            < page.index("The ranking's average monthly rank correlation")
    moved = copy.deepcopy(summary)
    moved["baseline_ic_ci95"]["pooled"] = [0.004, 0.03]
    built = rp.build(reg, latest, moved, X.detectable, X.load_evaluations())
    assert "In plain terms:" not in built and "No edge has been shown" not in built
    assert "<strong>Not reliably.</strong>" in built
    assert rp.plain_answer(True) and rp.plain_answer(False) == ""
    assert advice_terms_in(re.sub(r"<[^>]+>", " ", rp.plain_answer(True))) == []


def test_the_check_on_the_headline_chart_says_what_it_is():
    """'Fed today's filings / lead over the real ranking' meant nothing until two cards later; the
    row names a ranking that cheats, and the caption gives the scale's ends and its zero."""
    page = _rebuilt()
    fig = page[page.index('<figure class="fp"'):page.index("</figure>")]
    if "fp-c" in fig:
        assert "A ranking that cheats" in fig and "today&rsquo;s filings for every past month" in fig
        assert "Fed today&rsquo;s filings" not in fig
    assert "runs from &minus;1 to +1" in fig and "no relation" in fig


def test_one_banner_one_main_one_footer_and_links_in_sentences_are_underlined():
    """axe (round two): no main landmark, every word outside a landmark, and links told from their
    sentence by colour alone (1.12:1 against the foot's grey)."""
    page = _rebuilt()
    body = page[page.index("<body>"):]
    assert body.count("<main") == 1 and body.count('<header class="bar"') == 1 and body.count("<footer") == 1
    assert body.index('<header class="bar"') < body.index("<main") < body.index('id="how-tested"') \
        < body.index("</main>") < body.index('<footer class="foot"')
    assert body.count("<header") == 1                 # the banner is the only one
    import pytest
    sync_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")
    with sync_api.sync_playwright() as pw:
        try:
            browser = pw.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(rp.PAGE.as_uri(), wait_until="load", timeout=60000)
            page.wait_for_selector("main", timeout=8000)
            bare = page.evaluate("""[...document.querySelectorAll('p a, li a, .foot > a')]
                .filter(a => !a.closest('.toc, .next-nav'))
                .filter(a => !getComputedStyle(a).textDecorationLine.includes('underline')).map(a => a.textContent)""")
            assert bare == []
            assert page.evaluate("document.querySelectorAll('p a, .foot > a').length") >= 3
            outside = page.evaluate("""[...document.querySelectorAll('h1, h2, p, .answer')]
                .filter(e => !e.closest('main, header, footer, nav')).map(e => e.tagName + ' ' + e.textContent.slice(0, 40))""")
            assert outside == []
        finally:
            browser.close()


# ------------------------------------------- the 2026-10-28 mutation audit (rows 92-94, 98, 99)
# `research.plain_answer`'s only check recomputed the rule from the code and then asked the page to
# agree with it (`every_zero = all(lo <= 0 <= hi ...)`), so four mutations of the claim's own
# arithmetic went through it green: the boundary `<=` -> `<`, the intervals' confidence level
# 5% -> 10% in either of the two places that builds them, and the record the pooled estimate
# covers. The quantiles below are written as literals on purpose - reading them back out of
# `v2_experiments` is what made the old check blind (plan/claims-mutation-audit.md, row 59's rule).
def _synthetic_baseline() -> dict:
    """A baseline ledger row with hand-chosen statistics, so the intervals the page prints can be
    checked against figures written here rather than against the engine's recomputation of them.
    Its monthly series covers the record exactly - 176 month-ends, 2012-01 to 2026-08 - with one
    month either side of it that no interval may include."""
    idx = list(pd.date_range("2012-01-31", "2026-08-31", freq="ME").strftime("%Y-%m-%d"))
    months = {m: (0.01 if i % 2 else -0.01) for i, m in enumerate(idx)}
    months["2011-12-31"] = months["2026-09-30"] = 0.5          # outside the record on either side
    return {"id": "baseline", "months": len(idx), "data_hash": "synthetic", "method_version": 2,
            "ic_by_month": months,
            "summary": {"presample": {"ic": {"mean": 0.0, "se": 0.01, "n": 38}},
                        "all": {"ic": {"mean": 0.002, "se": 0.005, "n": 80}}}}


def test_the_intervals_behind_the_plain_answer_are_95_percent_over_the_whole_record():
    """Claim ``research.plain_answer`` says "every 95% interval on the ranking's monthly rank
    correlation (each era and pooled)". Two-sided 5% Student's t: 2.0262 on 37 df, 1.9905 on 79,
    1.9736 on 175 (the first is pinned against the textbook in
    ``test_the_t_distribution_matches_known_quantiles``)."""
    summary = X.summary_for_page(X.load_registry(), {"baseline": _synthetic_baseline()})
    ci = summary["baseline_ic_ci95"]
    assert ci["presample"] == [-0.0203, 0.0203]              # 0.0 +- 2.0262 x 0.01
    assert ci["all"] == [-0.008, 0.012]                      # 0.002 +- 1.9905 x 0.005
    # the pooled estimate is the whole record and nothing outside it: either 0.5 month on its own
    # would move the mean to 0.0028 and carry the interval clear of zero
    assert summary["months"]["pooled"] == 176
    assert summary["baseline_ic"]["pooled"] == 0.0
    assert ci["pooled"] == [-0.0015, 0.0015]                 # 0.0 +- 1.9736 x 0.00075593
    # and the figure beside them: the smallest mean this record could detect 80% of the time at
    # |t| >= 3 is (3 + 0.8434) x 0.00075593, not 3 x it
    assert summary["pooled_detectable"] == 0.0029


def test_an_interval_that_only_just_includes_zero_still_includes_it():
    """Row 92: the rule is ``lo <= 0 <= hi``. An endpoint landing exactly on zero - which rounding
    the published interval to four decimals makes a real possibility - still includes zero, so the
    plain answer and "Not measurably." both stand; a lower bound a ten-thousandth above zero ends
    both."""
    reg, latest = X.load_registry(), _latest()
    base = X.summary_for_page(reg, latest)
    middling = {"all": [-0.01, 0.01], "pooled": [-0.02, 0.01]}
    for era, expected in (([-0.02, 0.0], True),           # upper bound exactly on zero
                          ([0.0, 0.02], True),            # lower bound exactly on zero
                          ([0.0001, 0.02], False)):       # the first interval that excludes it
        summary = copy.deepcopy(base)
        summary["baseline_ic_ci95"] = {"presample": era, **middling}
        page = rp.build(reg, latest, summary, X.detectable, X.load_evaluations())
        assert ("In plain terms:" in page) is expected, era
        assert ("<strong>Not measurably.</strong>" in page) is expected, era
        assert ("<strong>Not reliably.</strong>" in page) is (not expected), era
        assert ("Every interval includes zero." in page) is expected, era


def test_every_t_statistic_carries_a_real_minus_as_the_means_beside_it_do():
    """Round three, 2026-10-27: the t-statistics printed a hyphen ("t -0.67") beside means printed with
    a real minus ("&minus;0.09%"), so two signs on one row differed in shape and width."""
    page = _rebuilt()
    body = re.sub(r"<style.*?</style>", " ", page, flags=re.S)
    # the generator's own t-statistics (prose quoted from the registry is the registry's, untouched here)
    assert not re.search(r'\(t&nbsp;-\d|t <span class="num">-\d|<span class="t">t -\d|aria-label="[^"]*, t -\d', body)
    assert not re.search(r'<span class="t">\((?:t )?-\d', body)
    assert re.search(r"\(t(?:&nbsp;| )&minus;\d", body)                     # and the negative ones are there to check


# ---------------------------------------- "How long would it take to prove an edge?" (2026-10-29)
# The three sentences in that section each say how a number is computed and none of them was in
# `claims.py`, so they sat outside the publish gate's derivation and outside the mutation audit
# (plan/claims-mutation-audit.md, next-list item 3). They are registered now as
# `research.ic_swing`, `research.months_to_prove` and `research.detectable_difference`, and these
# are their checks. Every expected figure below is a literal: reading it back out of
# `research_page` or `v2_experiments` is what made the audit's row 59 class of check blind.

SWING = re.compile(r"swings by about <span class=\"num\">([\d.]+)</span>")
MDE = re.compile(r"reliably see was about <span class=\"num\">([\d.]+)</span> over 2012&ndash;2019 "
                 r"and <span class=\"num\">([\d.]+)</span> over the held-out months")
STAT = re.compile(r'<div class="stat"><div class="v">(\d+)</div><div class="k">([^<]+)</div>')


def _row_ids(reg: list, latest: dict) -> list[str]:
    """The alternatives `build` keeps, in id order - the rows whose standard errors decide the
    two "smallest difference" figures."""
    return sorted(e["id"] for e in reg
                  if e["id"] != "baseline" and not e.get("void") and latest.get(e["id"])
                  and "ic_vs_baseline" in latest[e["id"]]["summary"].get("presample", {}))


def _spread(ids: list[str], plateau: float, below: tuple, above: tuple, mean_middle: float) -> dict:
    """A standard error per row, chosen so the *median* of them is a figure written here.

    A plateau wide enough to cover the middle index whatever the row count, two rows below it and
    two above - so the median is ``plateau`` while the mean, the smallest and the largest are all
    something else. The row sitting in the middle of the **mean** ordering (the order ``build``
    sorts the rows into) gets its own value, so a pick that takes the middle row without
    re-sorting by standard error reads a different number.
    """
    se = {i: plateau for i in ids}
    se[ids[0]], se[ids[1]] = below
    se[ids[-2]], se[ids[-1]] = above
    se[ids[len(ids) // 2]] = mean_middle
    return se


P_SE, P_N = 0.004, 38           # 2012-2019: the median alternative's standard error and months
H_SE, H_N = 0.006, 41           # the held-out months: a different figure and a different count


def _mde_fixture(latest: dict, reg: list, baseline_ic: dict | None = None) -> dict:
    """Every alternative's comparison statistics replaced with hand-chosen ones. Everything else
    on the page - the registry, the control, the bot rule, the coverage - is the real thing, so
    this exercises the generator the data loop runs and not a reduced copy of it."""
    lat = copy.deepcopy(latest)
    ids = _row_ids(reg, latest)
    p_se = _spread(ids, P_SE, (0.001, 0.002), (0.050, 0.100), 0.030)
    h_se = _spread(ids, H_SE, (0.0005, 0.0008), (0.060, 0.120), 0.040)
    for k, i in enumerate(ids):
        s = lat[i]["summary"]
        for period, mean, se, n in (("presample", round(0.05 - 0.001 * k, 5), p_se[i], P_N),
                                    ("holdout", round(-0.05 + 0.001 * k, 5), h_se[i], H_N)):
            s[period]["ic_vs_baseline"] = {
                "n": n, "mean": mean, "se": se, "t": round(mean / se, 2), "p": 0.5,
                "ci95": [round(mean - 2 * se, 5), round(mean + 2 * se, 5)]}
    if baseline_ic:
        lat["baseline"]["summary"].update({p: {"ic": st} for p, st in baseline_ic.items()})
    return lat


def test_the_smallest_difference_the_test_could_see_is_the_median_alternatives_own_error():
    """Claim ``research.detectable_difference``. The pair of figures is the engine's
    ``detectable`` applied to the **median** alternative's standard error *in each period*, over
    that period's own month count, with the multiple-testing correction taken over the registered
    count the page was handed.

    Checked by watching what the generator asks for: the injected ``detectable`` returns a value
    that encodes both arguments, so a figure built from the mean, the smallest or the largest
    standard error, from the wrong period, or over the wrong number of months prints differently.
    """
    reg, latest = X.load_registry(), _latest()
    summary = X.summary_for_page(reg, latest)
    seen = []

    def spy(se, n_tests, n=None):
        seen.append((se, n_tests, n))
        return round(se * 100 + ((n or 0) - P_N) * 0.001, 6)

    page = rp.build(reg, _mde_fixture(latest, reg), summary, spy, X.load_evaluations())
    # 0.004 x 100 + 0 = 0.400; 0.006 x 100 + 3 x 0.001 = 0.603. Swapping the periods' standard
    # errors gives 0.600/0.403 and swapping their month counts 0.403/0.600; the mean standard
    # error (0.0062/0.0076) gives 0.620/0.763, the smallest 0.100/0.050 and the largest
    # 10.000/12.003; the middle row of the mean ordering gives 3.000/4.003.
    assert MDE.search(page).groups() == ("0.400", "0.603")
    assert (P_SE, summary["registered"], P_N) in seen            # the Bonferroni count is the
    assert (H_SE, summary["registered"], H_N) in seen            # registered one the page was given


def test_the_smallest_difference_the_test_could_see_is_the_engines_own_bar():
    """The same claim's arithmetic, against figures written here rather than recomputed.

    At 29 registered comparisons the bar is Bonferroni-corrected Student's t and the power term
    is the one-sided 80% quantile on the same degrees of freedom: (3.37935 + 0.85144) x 0.004 =
    0.01692 over 38 months, and (3.35978 + 0.85070) x 0.006 = 0.02526 over 41. Printed to three
    decimals that is 0.017 and 0.025 - against 0.014/0.020 if the power term were dropped and
    0.012/0.018 with no correction for the number of comparisons.
    """
    reg, latest = X.load_registry(), _latest()
    summary = copy.deepcopy(X.summary_for_page(reg, latest))
    summary["registered"] = 29                   # a literal, so the ledger growing cannot move it
    page = rp.build(reg, _mde_fixture(latest, reg), summary, X.detectable, X.load_evaluations())
    assert MDE.search(page).groups() == ("0.017", "0.025")


BASELINE_IC = {
    # se x sqrt(n) per period, each rounding to a different two-decimal figure, so the sentence
    # names which record it is taken over: 2020-2026 is the one the claim states.
    "all": {"n": 100, "mean": -0.0003, "se": 0.01, "t": -0.03, "p": 0.98},         # -> 0.10
    "presample": {"n": 36, "mean": -0.00188, "se": 0.02, "t": -0.09, "p": 0.93},   # -> 0.12
    "holdout": {"n": 25, "mean": 0.00417, "se": 0.03, "t": 0.14, "p": 0.89},       # -> 0.15
    "discovery": {"n": 49, "mean": -0.00434, "se": 0.04, "t": -0.11, "p": 0.91},   # -> 0.28
}


def test_the_monthly_swing_is_the_standard_deviation_of_the_2020_2026_record():
    """Claim ``research.ic_swing``: "the ranking's monthly correlation swings by about X from one
    month to the next" is the **standard deviation** of the baseline's monthly rank correlation
    over 2020-2026, recovered from the ledger row's own standard error as se x sqrt(n).

    0.01 x sqrt(100) = 0.10. The other three periods in the fixture print 0.12, 0.15 and 0.28, so
    the figure names its record; se / sqrt(n) would print 0.00 and se x n 1.00.
    """
    reg, latest = X.load_registry(), _latest()
    lat = _mde_fixture(latest, reg, baseline_ic=BASELINE_IC)
    page = rp.build(reg, lat, X.summary_for_page(reg, latest), X.detectable, X.load_evaluations())
    assert SWING.search(page).group(1) == "0.10"
    # the fixture really does discriminate: four periods, four printed figures
    assert len({f"{st['se'] * math.sqrt(st['n']):.2f}" for st in BASELINE_IC.values()}) == 4


def test_the_months_to_prove_an_edge_are_that_swing_at_eighty_percent_power():
    """Claim ``research.months_to_prove``: each count is ((z + 0.84) x the monthly swing / the
    correlation being shown)^2, at 80% power, with z = 1.96 for "the usual 5% level" and 3 for a
    t-statistic of 3 - and each stat is paired with the figures its own label names.

    At a swing of 0.10: (2.80 x 0.10 / 0.05)^2 = 31.36 -> 31, (3.84 x 0.10 / 0.05)^2 = 58.98 ->
    59, (3.84 x 0.10 / 0.03)^2 = 163.84 -> 164. Dropping the power term gives 15/36/100; reading
    the swing off 2012-2019 instead gives 45/85/236; the two z values swapped would swap the
    first two.
    """
    reg, latest = X.load_registry(), _latest()
    lat = _mde_fixture(latest, reg, baseline_ic=BASELINE_IC)
    page = rp.build(reg, lat, X.summary_for_page(reg, latest), X.detectable, X.load_evaluations())
    assert STAT.findall(page) == [
        ("31", "months of record to show a monthly correlation of 0.05 at the usual 5% level"),
        ("59", "months for the same with a t-statistic of 3"),
        ("164", "months to show 0.03 with a t-statistic of 3"),
    ]


# --------------------------------------------------------------- the overlap (2026-10-29, 13:00)
# Claim ``research.pairing_overlap``. Found by the sweep in
# ``tests/test_claims_register.py`` section 5 and named before that in
# ``plan/claims-mutation-audit.md``: the paragraph states four figures and
# ``test_the_page_says_the_pairing_is_on_the_shared_stocks_and_states_the_universe_difference``
# pinned one of them (the baseline's stocks a month) plus the words around the rest.

NARROW_ID = "fcf-yield-only"


def _overlap_fixture(tmp_path, monkeypatch, *, baseline=440.0, narrow=400.0, decoy=430.0,
                     common=None, outside=None):
    """A controlled ``PER_CONFIG`` and ledger for the universe paragraph.

    ``decoy`` is a second alternative ranked *wider* than ``narrow``, so a page that
    took the first, the last or the widest alternative would name the wrong screen.
    ``outside`` is an overlap figure dated outside ``X.SAMPLE``; it must not move the
    mean, which is the "wrong record" mutation rows 89-105 found twice.
    """
    reg, latest = X.load_registry(), copy.deepcopy(_latest())
    assert NARROW_ID in latest and "equal-categories" in latest, "fixture ids left the ledger"
    per = {"configs": {
        "baseline": {"sample": {"months": 80, "mean_stocks": baseline}},
        NARROW_ID: {"sample": {"months": 80, "mean_stocks": narrow}},
        "equal-categories": {"sample": {"months": 80, "mean_stocks": decoy}},
    }}
    path = tmp_path / "per_config.json"
    path.write_text(json.dumps(per), encoding="utf-8")
    monkeypatch.setattr(rp, "PER_CONFIG", path)
    months = {"2020-01-31": common[0], "2020-02-29": common[1]} if common else {}
    if outside is not None:
        months["2019-12-31"] = outside            # before X.SAMPLE[0]
        months["2026-09-30"] = outside            # after X.SAMPLE[1]
    latest[NARROW_ID]["n_common_by_month"] = months
    page = rp.build(reg, latest, X.summary_for_page(reg, _latest()), X.detectable,
                    X.load_evaluations())
    return page[page.index("A weighting can only rank what it has enough data to score"):][:1200]


def test_the_universe_paragraph_names_the_narrowest_screen_and_both_its_own_figures(tmp_path, monkeypatch):
    """The two counts the sentence states: what the current weighting can rank, and what the
    *narrowest* alternative can. 440 and 400 against a decoy at 430 - so picking the widest
    alternative, or the first in the file, prints 430 and names Equal weight on every category.
    """
    para = _overlap_fixture(tmp_path, monkeypatch, common=[300.0, 300.0])
    assert 'ranked <span class="num">440</span> stocks a month on average' in para
    assert 'alternative, Free-cash-flow yield alone (within sector), <span class="num">400</span>.' in para
    assert "430" not in para and "Equal weight on every category" not in para


def test_the_overlap_is_the_mean_of_the_screens_own_monthly_counts_inside_the_record(tmp_path, monkeypatch):
    """"The two had N companies a month in the overlap" is the mean of that screen's
    ``n_common_by_month`` over ``X.SAMPLE`` - not its month count, not the screen's own universe,
    and not the baseline's.

    290 and 310 give 300. The two months dated outside the record carry 999 each: including them
    would print 650, and taking the maximum 999. 400 (the screen's universe) and 440 (the
    baseline's) are both present in the paragraph for other reasons, so this assertion pins the
    *overlap* position specifically.
    """
    para = _overlap_fixture(tmp_path, monkeypatch, common=[290.0, 310.0], outside=999.0)
    assert 'The two had <span class="num">300</span> companies a month in the overlap' in para
    assert "999" not in para and "650" not in para


def test_a_near_subset_says_which_side_the_pairing_holds_and_what_it_gives_up(tmp_path, monkeypatch):
    """The other branch, and the point of it: when the overlap is essentially the screen's own
    universe, the sentence says pairing holds the *baseline* to the smaller set, and states the
    stocks only the baseline can score - the baseline's universe minus the overlap.

    Overlap 396 of the screen's 400: 440 - 396 = 44. The screen's own 400 - 396 = 4 is the
    other subtraction available and prints as "4 a month are unique to it", so a swap of the
    two is visible. The narrower-screen clause must not claim fewer than one when four are.
    """
    para = _overlap_fixture(tmp_path, monkeypatch, common=[396.0, 396.0])
    assert '(<span class="num">4</span> a month are unique to it)' in para
    assert ('so pairing on the overlap holds the current weighting to that smaller set instead of '
            'crediting it with the <span class="num">44</span> only it can score.') in para
    assert "companies a month in the overlap" not in para
    assert "fewer than one a month is unique to it" not in para


def test_the_near_subset_branch_turns_at_ninety_nine_percent_of_the_screens_universe(tmp_path, monkeypatch):
    """The threshold itself, at and either side of it - the state that separates ``>=`` from ``>``.

    0.99 x 400 = 396. At 396 exactly the page takes the near-subset branch; at 395.9 it reports
    the overlap instead. Both branches are reachable, which is what makes the sentence a statement
    about the data rather than a fixed piece of prose.
    """
    at = _overlap_fixture(tmp_path, monkeypatch, common=[396.0, 396.0])
    assert "pairing on the overlap holds" in at and "in the overlap, and that" not in at
    below = _overlap_fixture(tmp_path, monkeypatch, common=[395.8, 396.0])
    assert "in the overlap, and that" in below and "pairing on the overlap holds" not in below
    above = _overlap_fixture(tmp_path, monkeypatch, common=[400.0, 400.0])
    assert "fewer than one a month is unique to it" in above


def test_the_paragraph_says_nothing_when_the_screens_months_are_missing(tmp_path, monkeypatch):
    """A row with no ``n_common_by_month`` must leave the overlap sentence out rather than print a
    mean of nothing - the universe counts still stand, because they come from the committed
    measurement and not from the ledger."""
    para = _overlap_fixture(tmp_path, monkeypatch, common=None)
    assert 'ranked <span class="num">440</span> stocks a month' in para
    assert "in the overlap" not in para and "unique to it" not in para
