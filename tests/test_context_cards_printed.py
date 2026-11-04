"""The context cards' printed figures, read back from the page for every stock.

WHY THIS EXISTS - ``plan/claims-mutation-audit.md``, queue item 4 (2026-11-03, owner-run evening).
The insider rules are pinned where they are computed (``insider_activity.summarise_rows``, claims
``context.insider_trades`` and ``context.insider_plan``), and the 52-week figures where they are
fetched; what no registered check read was the **printing** - which published field lands in which
slot of the Insider trades card, the case against's insider sentence and the Trend card's 52-week
line. Wrong field (buys printed as sales, the trade count as the insider count), wrong copy of a
figure printed twice (the plan share), a dropped branch ("None", the singular): all invisible to a
check that stops at the payload.

So each test runs **the generator's own page** with the published payloads, renders the card for
every stock, and compares each slot with the payload's own field - the share and the "net sale"
rule worked out here independently - plus hand-worked fixture stocks for the branches today's data
may not take. ``cMoney`` and ``cPrice`` are pinned once on hand-worked values, so the per-stock
checks can ask the page to format a payload value without trusting the formatter blind. Nothing
here names a stock or reads the clock (CLAUDE.md rows "data->red" and "clock->red").
"""
from __future__ import annotations

import html as htmllib
import json
import math
import re
import shutil
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_dashboard as gd  # noqa: E402

try:
    from playwright import sync_api
except Exception:  # noqa: BLE001
    sync_api = None


def _payload(name: str) -> dict:
    p = ROOT / name
    if not p.exists():
        pytest.skip(f"MISSING DATA: no published {name}")
    s = p.read_text(encoding="utf-8")
    return json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))


@pytest.fixture(scope="module")
def ctx() -> dict:
    return _payload("dashboard_context.js")["ctx"]


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    """The generator's own page (never the committed file) with both published payloads."""
    if sync_api is None:
        pytest.skip("playwright not installed")
    for f in ("dashboard_data.js", "dashboard_context.js"):
        if not (ROOT / f).exists():
            pytest.skip(f"MISSING DATA: no published {f}")
    d = tmp_path_factory.mktemp("ctxcards")
    (d / "index.html").write_text(gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t",
                                                   data_version="v"), encoding="utf-8")
    for f in ("dashboard_data.js", "dashboard_context.js"):
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
        pg.wait_for_function("typeof renderContext === 'function' && typeof caseItems === 'function' && "
                             "typeof D !== 'undefined' && window.SCREENER_CONTEXT !== undefined && "
                             "Object.values(D.stock_detail).some(s => s.ctx)", timeout=60000)
        yield pg
        b.close()
        assert not errors, errors


def _text(h: str) -> str:
    return re.sub(r"\s+", " ", htmllib.unescape(re.sub(r"<[^>]+>", " ", h or ""))).strip()


def _jsround(x: float) -> int:
    """JavaScript's Math.round (half up), not Python's half-even."""
    return math.floor(x + 0.5)


def _fixed(x: float, f: int) -> str:
    """Number.prototype.toFixed for x >= 0: the exact binary value, ties to the larger n."""
    return str(Decimal(x).quantize(Decimal(1).scaleb(-f), rounding=ROUND_HALF_UP))


def _money(v: float) -> str:
    a = abs(v)
    if a >= 1e9:
        return f"${_fixed(v / 1e9, 1)}B"
    if a >= 1e6:
        return f"${_fixed(v / 1e6, 1)}M"
    if a >= 1e3:
        return f"${_fixed(v / 1e3, 0)}K"
    return f"${_fixed(v, 0)}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


# --------------------------------------------------------------------------- the formatters, once
@pytest.mark.parametrize("v, want", [
    (999.0, "$999"), (1000.0, "$1K"), (45678.0, "$46K"), (999_499.0, "$999K"),
    (1_234_567.0, "$1.2M"), (12_250_000.0, "$12.3M"), (2_500_000_000.0, "$2.5B"),
])
def test_money_is_printed_by_the_rule_the_page_states(page, v, want):
    assert _text(page.evaluate("v => cMoney(v)", v)) == want == _money(v)


@pytest.mark.parametrize("v, want", [(100.125, "$100.13"), (99.994, "$99.99"), (5.0, "$5.00")])
def test_prices_print_to_the_cent(page, v, want):
    assert _text(page.evaluate("v => cPrice(v)", v)) == want == f"${_fixed(v, 2)}"


# --------------------------------------------------------------------------- the Insider trades card
_CARD_JS = """(tickers) => { const out = {};
    for (const t of tickers) { const s = D.stock_detail[t]; renderContext(t, s);
        const pairs = [...document.querySelectorAll('#modal-context .ctx-pair > div')].map(d => ({
            k: (d.querySelector('.ctx-fk') || {}).textContent, v: (d.querySelector('.ctx-fv') || {}).innerHTML,
            s: (d.querySelector('.ctx-fs') || {}).innerHTML }));
        out[t] = pairs.filter(p => p.k && p.k.startsWith('Open-market')); }
    return out; }"""


def _expected_card(ins: dict) -> dict:
    """What each slot must say, from the published summary alone."""
    buy = ("None", "") if not ins.get("buy_n") else (
        _money(ins["buy_value"]), f"{_plural(ins['buy_people'], 'insider')}, {_plural(ins['buy_n'], 'trade')}")
    if not ins.get("sell_n"):
        sell = ("None", "")
    else:
        sub = _plural(ins["sell_people"], "insider")
        if ins.get("sell_planned_value") is not None and (ins.get("sell_value") or 0) > 0:
            sub += f", {_jsround(100 * ins['sell_planned_value'] / ins['sell_value'])}% on pre-set plans"
        sell = (_money(ins["sell_value"]), sub)
    return {"Open-market buys, 90 days": buy, "Open-market sales, 90 days": sell}


def _check_card(got: list, ins: dict) -> list[str]:
    want = _expected_card(ins)
    seen = {g["k"]: (_text(g["v"]), _text(g["s"])) for g in got}
    return [f"{k}: printed {seen.get(k)!r}, the payload says {w!r}" for k, w in want.items() if seen.get(k) != w]


def test_every_insider_card_prints_the_published_summary_in_its_own_slots(page, ctx):
    tickers = [t for t, c in ctx.items() if c.get("ins")]
    if not tickers:
        pytest.skip("MISSING DATA: no stock in the published context carries an insider summary")
    got = page.evaluate(_CARD_JS, tickers)
    bad = {t: e for t in tickers if (e := _check_card(got[t], ctx[t]["ins"]))}
    assert not bad, f"{len(bad)} of {len(tickers)} cards misprint: {dict(list(bad.items())[:5])}"


_FIXTURE_JS = """(args) => { const [t, ins] = args; const s = JSON.parse(JSON.stringify(D.stock_detail[t]));
    s.ctx = Object.assign({}, s.ctx || {}, {ins: ins}); renderContext(t, s);
    return [...document.querySelectorAll('#modal-context .ctx-pair > div')].map(d => ({
        k: (d.querySelector('.ctx-fk') || {}).textContent, v: (d.querySelector('.ctx-fv') || {}).innerHTML,
        s: (d.querySelector('.ctx-fs') || {}).innerHTML })).filter(p => p.k && p.k.startsWith('Open-market')); }"""

_HAND = [
    # no buys; one seller; a source without the plan flag (Yahoo) - no share, never "0%"
    ({"buy_n": 0, "buy_value": 0, "buy_people": 0, "sell_n": 1, "sell_value": 512_000.0, "sell_people": 1,
      "sell_planned_value": None, "src": "yahoo"},
     {"Open-market buys, 90 days": ("None", ""), "Open-market sales, 90 days": ("$512K", "1 insider")}),
    # one buyer, one trade; sales two-thirds on plans (2/3 -> 66.67% -> 67%)
    ({"buy_n": 1, "buy_value": 2_000_000.0, "buy_people": 1, "sell_n": 3, "sell_value": 3_000_000.0,
      "sell_people": 2, "sell_planned_value": 2_000_000.0, "src": "sec"},
     {"Open-market buys, 90 days": ("$2.0M", "1 insider, 1 trade"),
      "Open-market sales, 90 days": ("$3.0M", "2 insiders, 67% on pre-set plans")}),
    # a half rounds up, as the page's Math.round does (1/8 -> 12.5% -> 13%)
    ({"buy_n": 4, "buy_value": 950.0, "buy_people": 3, "sell_n": 2, "sell_value": 8_000.0, "sell_people": 2,
      "sell_planned_value": 1_000.0, "src": "sec"},
     {"Open-market buys, 90 days": ("$950", "3 insiders, 4 trades"),
      "Open-market sales, 90 days": ("$8K", "2 insiders, 13% on pre-set plans")}),
    # nothing in 90 days
    ({"buy_n": 0, "buy_value": 0, "buy_people": 0, "sell_n": 0, "sell_value": 0, "sell_people": 0,
      "sell_planned_value": None, "src": "sec"},
     {"Open-market buys, 90 days": ("None", ""), "Open-market sales, 90 days": ("None", "")}),
]


@pytest.mark.parametrize("ins, want", _HAND)
def test_a_hand_worked_insider_card(page, ctx, ins, want):
    t = next(iter(ctx))
    got = page.evaluate(_FIXTURE_JS, [t, {"recent": [], **ins}])
    assert {g["k"]: (_text(g["v"]), _text(g["s"])) for g in got} == want
    assert _expected_card(ins) == want, "the independent reading disagrees with the hand-worked one"


# --------------------------------------------------------------------------- the case against
# The sentence's own tests live in tests/test_case_against.py with the rest of its claim's checks;
# the rule for what it must say is here, beside the card's.
def _net_sale(ins: dict) -> bool:
    return (ins.get("sell_n") or 0) > 0 and (not ins.get("buy_n") or (ins.get("sell_value") or 0) > (ins.get("buy_value") or 0))


def _expected_case(ins: dict) -> str:
    plan = ""
    if ins.get("sell_planned_value") is not None and (ins.get("sell_value") or 0) > 0:
        plan = f", {_jsround(100 * ins['sell_planned_value'] / ins['sell_value'])}% on pre-set plans"
    tail = (f", against {_money(ins['buy_value'])} of purchases." if ins.get("buy_n")
            else " and no open-market purchases.")
    return (f"Officers and directors made {_money(ins['sell_value'])} of open-market sales in the last 90 days "
            f"({_plural(ins['sell_people'], 'insider')}{plan}){tail} The panel gives the evidence that sales "
            "have historically said less than purchases.")



# --------------------------------------------------------------------------- the 52-week line
_TREND_JS = """(tickers) => { const out = {};
    for (const t of tickers) { const s = D.stock_detail[t]; renderContext(t, s);
        const f = [...document.querySelectorAll('#modal-context .ctx-fact')].find(e =>
            (e.querySelector('.ctx-fk') || {}).textContent === 'From 52-week high');
        out[t] = f ? [f.querySelector('.ctx-fv').innerHTML, f.querySelector('.ctx-fs').innerHTML,
                      s.ctx.hid ? cDate(s.ctx.hid) : null] : null; }
    return out; }"""
_PCT = re.compile(r"^([+−]?)(\d+(?:\.\d+)?)%$")


def test_every_52_week_line_prints_its_own_high_low_and_distance(page, ctx):
    tickers = [t for t, c in ctx.items() if c.get("hi")]
    if not tickers:
        pytest.skip("MISSING DATA: no stock in the published context carries a 52-week high")
    got = page.evaluate(_TREND_JS, tickers)
    bad = {}
    for t in tickers:
        c = ctx[t]
        if got[t] is None:
            bad[t] = "no 52-week line although the payload has a high"
            continue
        fv, fs, day = _text(got[t][0]), _text(got[t][1]), got[t][2]
        m = _PCT.match(fv)
        if not m:
            bad[t] = f"distance printed as {fv!r}"
            continue
        printed = float(m.group(2)) * (-1 if m.group(1) == "−" else 1)
        decimals = len(m.group(2).split(".")[1]) if "." in m.group(2) else 0
        true = (c["px"] / c["hi"] - 1) * 100
        if abs(printed - true) > 0.5 * 10 ** -decimals + 1e-9 or (m.group(1) == "" and printed != 0):
            bad[t] = f"distance {fv} for close {c['px']} against high {c['hi']} ({true:+.4f}%)"
            continue
        want = f"high ${_fixed(c['hi'], 2)}" + (f" on {_text(day)}" if c.get("hid") else "") + f", low ${_fixed(c['lo'], 2)}"
        if fs != want:
            bad[t] = f"printed {fs!r}, the payload says {want!r}"
    assert not bad, f"{len(bad)} of {len(tickers)}: {dict(list(bad.items())[:5])}"
