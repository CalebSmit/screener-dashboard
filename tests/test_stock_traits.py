"""What sets a stock apart from its sector, in words (claim ``dashboard.traits``, 2026-11-05).

The leader cards and the top of every stock's sheet name the categories that *stand out* (a
category score of ``TRAIT_HI`` or more) and *trail* (``TRAIT_LO`` or less), in the score key's own
direction. The page computes that in JS (``traitsOf``); ``generate_dashboard.traits`` is the same
rule in Python. These tests hold the two to each other on every published stock, hold the words to
the score key, and hold what the page prints to the payload's own scores - asserting their premise
from the payload, never which stock happens to stand out on a given morning.
"""
from __future__ import annotations

import html as htmllib
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_dashboard as gd  # noqa: E402
from stock_summary import advice_terms_in  # noqa: E402
from tests import sheet_warm  # noqa: E402

try:
    from playwright import sync_api
except ImportError:  # pragma: no cover
    sync_api = None

CATS = [k for k, *_ in gd.SCORE_KEY]


def _payload(name: str) -> dict:
    p = ROOT / name
    if not p.exists():
        pytest.skip(f"MISSING DATA: no published {name}")
    s = p.read_text(encoding="utf-8")
    return json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))


def _text(h: str) -> str:
    return re.sub(r"\s+", " ", htmllib.unescape(re.sub(r"<[^>]+>", " ", h or ""))).strip()


def _row_scores(r: dict) -> dict:
    return {c: r.get(f"{c}_score") for c in CATS}


# ---------------------------------------------------------------------------
# The rule and its words
# ---------------------------------------------------------------------------

def test_every_category_has_words_for_both_ends_and_none_is_advice():
    assert set(gd.TRAIT_WORDS) == set(CATS)
    for c, (hi, lo) in gd.TRAIT_WORDS.items():
        assert hi and lo and hi != lo, c
        for w in (hi, lo):
            assert not advice_terms_in(w), (c, w)
            assert not re.search(r"\d", w), (c, w)


def test_the_words_run_the_score_keys_way():
    """Each 'stands out' phrase says what the key says a *higher* score means; a hand-kept pairing,
    so rewording either side without the other fails here."""
    key = {k: text for k, _s, _n, text in gd.SCORE_KEY}
    pairs = {
        "valuation": ("lower price", "Lower price"), "quality": ("more profitable", "More profitable"),
        "growth": ("faster growth", "Faster growth"), "momentum": ("stronger price trend", "Stronger price trend"),
        "risk": ("calmer price", "Calmer price"), "revisions": ("improving", "Improving"),
        "size": ("smaller company", "Smaller company"), "investment": ("slower growth in total assets", "Slower asset growth"),
    }
    for c, (in_key, in_word) in pairs.items():
        assert in_key in key[c].lower(), (c, key[c])
        assert gd.TRAIT_WORDS[c][0].startswith(in_word), (c, gd.TRAIT_WORDS[c])


def test_a_hand_worked_set_of_scores():
    """Thresholds inclusive at 70 and 30, highest and lowest first, ties in the key's order, at most
    three and two, unscored categories neither."""
    sc = {"valuation": 70.0, "quality": 95.0, "growth": 70.0, "momentum": 69.99, "risk": 30.0,
          "revisions": 88.0, "size": 2.0, "investment": None}
    assert gd.traits(sc) == {"hi": ["quality", "revisions", "valuation"], "lo": ["size", "risk"]}
    assert gd.traits({"size": 30.01, "risk": 50}) == {"hi": [], "lo": []}
    assert gd.traits({}) == {"hi": [], "lo": []}
    assert gd.traits({"risk": 10.0, "size": 10.0, "investment": 5.0}) == {"hi": [], "lo": ["investment", "risk"]}
    assert gd.traits({"growth": True, "quality": float("nan")}) == {"hi": [], "lo": []}


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data() -> dict:
    return _payload("dashboard_data.js")


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    if sync_api is None:
        pytest.skip("playwright not installed")
    for f in ("dashboard_data.js", "dashboard_context.js", "dashboard_detail.js"):
        if not (ROOT / f).exists():
            pytest.skip(f"MISSING DATA: no published {f}")
    d = tmp_path_factory.mktemp("traits")
    (d / "index.html").write_text(gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t",
                                                   data_version="v"), encoding="utf-8")
    for f in ("dashboard_data.js", "dashboard_context.js", "dashboard_detail.js"):
        shutil.copy(ROOT / f, d / f)
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        pg = b.new_page(viewport={"width": 1440, "height": 1000})
        errors: list[str] = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto((d / "index.html").as_uri(), wait_until="load", timeout=60000)
        pg.wait_for_function("typeof traitsOf === 'function' && typeof D !== 'undefined' && "
                             "document.querySelector('#top5-row .top5-card .traits') !== null", timeout=60000)
        sheet_warm.settle(pg)
        yield pg
        pg.close()
        b.close()
        assert not errors, errors


def test_the_page_and_the_python_rule_agree_on_every_published_stock(page, data):
    rows = data["table_data"]
    got = page.evaluate("rows => rows.map(r => traitsOf(r))", [_row_scores(r) for r in rows])
    bad = [(r["Ticker"], g, gd.traits(_row_scores(r))) for r, g in zip(rows, got)
           if g != gd.traits(_row_scores(r))]
    assert not bad, bad[:5]
    # The premise: the rule names something on this page at all, at both ends.
    assert any(g["hi"] for g in got) and any(g["lo"] for g in got)


def test_each_leader_card_prints_the_words_for_its_own_scores(page, data):
    rows = sorted((r for r in data["table_data"]
                   if not r.get("Value_Trap_Flag") and not r.get("Growth_Trap_Flag")),
                  key=lambda r: r["Rank"])[:5]
    cards = page.evaluate("""() => [...document.querySelectorAll('#top5-row .top5-card')].map(c => ({
        t: c.querySelector('.top5-ticker').textContent,
        hi: [...c.querySelectorAll('.tr-hi')].map(e => e.textContent),
        lo: [...c.querySelectorAll('.tr-lo')].map(e => e.textContent) }))""")
    assert [c["t"] for c in cards] == [r["Ticker"] for r in rows]
    for c, r in zip(cards, rows):
        want = gd.traits(_row_scores(r))
        assert c["hi"] == [gd.TRAIT_WORDS[k][0] for k in want["hi"]], (c, want)
        assert c["lo"] == [gd.TRAIT_WORDS[k][1] for k in want["lo"]], (c, want)


_SHEET_JS = """t => { openStockDetail(t); const h = document.querySelector('#modal-score-row .msr-traits');
    const out = { hi: [...h.querySelectorAll('.tr-hi')].map(e => [e.querySelector('.tr-cat').textContent, e.lastChild.textContent]),
                  lo: [...h.querySelectorAll('.tr-lo')].map(e => [e.querySelector('.tr-cat').textContent, e.lastChild.textContent]),
                  text: h.innerText }; closeModal(); return out; }"""


def test_the_sheet_prints_each_named_category_with_its_published_score(page, data):
    """Through the reader's own route (openStockDetail), for the top and bottom of the ranking and
    the median stock: each chip names its category and the score it reads, to one decimal."""
    rows = sorted(data["table_data"], key=lambda r: r["Rank"])
    for r in (rows[0], rows[len(rows) // 2], rows[-1]):
        t = r["Ticker"]
        sc = data["stock_detail"][t]["cat_scores"]
        want = gd.traits(sc)
        got = page.evaluate(_SHEET_JS, t)
        label = {k: n for k, _s, n, _t in gd.SCORE_KEY}
        # The score as the page prints it everywhere else (fmt, one decimal), the score cards' own.
        shown = lambda k: page.evaluate("v => fmt(v, 'score')", sc[k])  # noqa: E731
        assert all(shown(k) == f"{sc[k]:.1f}" for k in want["hi"] + want["lo"] if round(sc[k] * 20) != sc[k] * 20)
        assert got["hi"] == [[f"{label[k]} {shown(k)}", gd.TRAIT_WORDS[k][0]] for k in want["hi"]], (t, got)
        assert got["lo"] == [[f"{label[k]} {shown(k)}", gd.TRAIT_WORDS[k][1]] for k in want["lo"]], (t, got)
        if want["hi"] or want["lo"]:
            assert f"a category scoring {gd.TRAIT_HI} or more stands out, {gd.TRAIT_LO} or less trails" in got["text"]
        else:
            assert got["text"].strip() == (f"No category scores {gd.TRAIT_HI} or more, or {gd.TRAIT_LO} or "
                                           "less, against its sector."), (t, got)


def test_a_stock_with_nothing_at_either_end_says_so(page):
    h = page.evaluate("sc => traitsHtml(sc, true)", {c: 50.0 for c in CATS})
    assert _text(h) == (f"No category scores {gd.TRAIT_HI} or more, or {gd.TRAIT_LO} or less, "
                        "against its sector.")
    h = page.evaluate("sc => traitsHtml(sc, false)", dict({c: 50.0 for c in CATS}, size=4.0))
    assert _text(h).startswith(f"No category scores {gd.TRAIT_HI} or more against its sector. Trails "
                               "Larger company"), _text(h)


def test_the_page_rule_on_the_hand_worked_scores(page):
    """The boundaries and ties no morning's data is guaranteed to reach, through the page's own
    traitsOf: 70 and 30 count, 69.99 and 30.01 do not, ties fall in the key's order."""
    cases = [
        {"valuation": 70.0, "quality": 95.0, "growth": 70.0, "momentum": 69.99, "risk": 30.0,
         "revisions": 88.0, "size": 2.0, "investment": None},
        {"size": 30.01, "risk": 50},
        {"risk": 10.0, "size": 10.0, "investment": 5.0},
        {"growth": 80.0, "valuation": 80.0, "momentum": 80.0, "quality": 80.0},
    ]
    for sc in cases:
        assert page.evaluate("sc => traitsOf(sc)", sc) == gd.traits(sc), sc
    assert gd.traits(cases[3])["hi"] == ["valuation", "quality", "growth"]


def test_a_holding_card_carries_the_words_for_its_own_scores(page, data):
    """My Holdings cards carry the same strip, read from the stock's own published scores."""
    rows = sorted(data["table_data"], key=lambda r: r["Rank"])
    for r in (rows[0], rows[-1]):
        t = r["Ticker"]
        got = page.evaluate("""t => { const h = document.createElement('div');
            h.innerHTML = holdingCard({t: t, s: D.stock_detail[t]});
            const tr = h.querySelector('.holding-traits');
            return tr && { hi: [...tr.querySelectorAll('.tr-hi')].map(e => e.textContent),
                           lo: [...tr.querySelectorAll('.tr-lo')].map(e => e.textContent) }; }""", t)
        want = gd.traits(data["stock_detail"][t]["cat_scores"])
        assert got == {"hi": [gd.TRAIT_WORDS[k][0] for k in want["hi"]],
                       "lo": [gd.TRAIT_WORDS[k][1] for k in want["lo"]]}, (t, got)


def test_the_compare_view_carries_the_words_for_each_stock(page, data):
    """Compare sets the same strip under each stock's composite, in the view's own (rank) order."""
    rows = sorted(data["table_data"], key=lambda r: r["Rank"])
    pick = [rows[-1]["Ticker"], rows[0]["Ticker"], rows[len(rows) // 2]["Ticker"]]
    got = page.evaluate("""ts => { UX.compare = ts.slice(); openCompare();
        const cells = [...document.querySelectorAll('#cmp-body .cmp-traits td')].map(td => ({
            hi: [...td.querySelectorAll('.tr-hi')].map(e => e.textContent),
            lo: [...td.querySelectorAll('.tr-lo')].map(e => e.textContent),
            none: !!td.querySelector('.tr-none') }));
        const heads = [...document.querySelectorAll('#cmp-body .cmp-head .cmp-t')].map(e => e.textContent);
        closeCompare(); UX.compare = []; return { cells, heads }; }""", pick)
    order = sorted(pick, key=lambda t: data["stock_detail"][t]["rank"])
    assert got["heads"] == order
    for t, cell in zip(order, got["cells"]):
        want = gd.traits(data["stock_detail"][t]["cat_scores"])
        assert cell["hi"] == [gd.TRAIT_WORDS[k][0] for k in want["hi"]], (t, cell)
        assert cell["lo"] == [gd.TRAIT_WORDS[k][1] for k in want["lo"]], (t, cell)
        assert cell["none"] == (not want["hi"]), (t, cell)
