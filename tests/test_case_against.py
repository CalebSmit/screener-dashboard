"""'The case against' in the stock sheet: every line is a fact the page already publishes.

WHY THIS EXISTS - owner, 2026-10-27 ("Yes make all changes necessary"), approving the proposal in
``research/2026-10-27-tradingagents-ideas.md`` section 5: one block at the top of the drilldown that
only **collects** what the sheet already shows against a stock - its weakest weighted categories,
trap and caveat flags, insider net selling, a close below the 200-day average, a report inside the
review window, a mean analyst target below the price, inputs that disagree with the filing. No new
number, no new threshold, no verdict.

That is a claim about *every* stock, so it is checked on every stock: the generator's own page is
rendered with the committed payloads, ``caseItems`` is read for all ~500 names in one call, and each
line is compared with the payload it came from - the same figure (to the page's own rounding), the
same published sentence where it reuses one, and present exactly when its condition holds. Claim
``drilldown.case_against``; CLAUDE.md "T0a's standing constraint".
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import claims  # noqa: E402
import generate_dashboard as gd  # noqa: E402
import stock_summary  # noqa: E402

try:
    from playwright import sync_api
except Exception:  # noqa: BLE001
    sync_api = None

CATS = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]
XNAMES = [("ta", "total assets"), ("eq", "equity"), ("rev", "revenue"), ("ni", "net income"),
          ("ocf", "operating cash flow")]
GO = {"weakest": "section-contribution", "flags": "modal-summary", "target": "section-price-targets",
      "trend": "section-context", "insider": "section-context", "report": "modal-summary",
      "caveats": "modal-summary", "churn": "modal-summary", "filing": "section-provenance",
      "behind": "section-provenance"}


def _payload(name: str) -> dict:
    p = ROOT / name
    if not p.exists():
        pytest.skip(f"MISSING DATA: no published {name}")
    s = p.read_text(encoding="utf-8")
    return json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))


@pytest.fixture(scope="module")
def data():
    return _payload("dashboard_data.js")


@pytest.fixture(scope="module")
def ctx():
    return _payload("dashboard_context.js")["ctx"]


def _site(tmp_path_factory) -> Path:
    for f in ("dashboard_data.js", "dashboard_context.js"):
        if not (ROOT / f).exists():
            pytest.skip(f"MISSING DATA: no published {f}")
    d = tmp_path_factory.mktemp("case")
    (d / "index.html").write_text(gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t",
                                                   data_version="v"), encoding="utf-8")
    for f in ("dashboard_data.js", "dashboard_context.js"):
        shutil.copy(ROOT / f, d / f)
    return d


@pytest.fixture(scope="module")
def browser():
    if sync_api is None:
        pytest.skip("playwright not installed")
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        yield b
        b.close()


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    return _site(tmp_path_factory)


@pytest.fixture(scope="module")
def page(browser, site):
    """The generator's own page (not the committed file), with the context file loaded."""
    pg = browser.new_page(viewport={"width": 1440, "height": 1000})
    errors: list[str] = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto((site / "index.html").as_uri(), wait_until="load", timeout=60000)
    pg.wait_for_function("typeof caseItems === 'function' && typeof D !== 'undefined' && "
                         "Object.values(D.stock_detail).some(s => s.ctx)", timeout=60000)
    yield pg
    pg.close()
    assert not errors, errors


@pytest.fixture(scope="module")
def cases(page) -> dict:
    """caseItems for every published stock, read in one call."""
    return dict(page.evaluate("Object.keys(D.stock_detail).map(t => [t, caseItems(D.stock_detail[t])])"))


def _sm(s: dict) -> dict:
    out = {}
    for f in s.get("summary") or []:
        out.setdefault(f.get("k"), f.get("t", ""))
    return out


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def _money(text: str) -> float:
    m = re.fullmatch(r"\$([\d.]+)([BMK]?)", text)
    assert m, text
    return float(m.group(1)) * {"B": 1e9, "M": 1e6, "K": 1e3, "": 1.0}[m.group(2)]


def _eff_weights(fw: dict, scores: dict) -> dict:
    total = sum(fw.get(c) or 0 for c in CATS if (fw.get(c) or 0) > 0 and _num(scores.get(c)))
    return {c: ((fw.get(c) or 0) / total * 100 if total and (fw.get(c) or 0) > 0 and _num(scores.get(c)) else 0)
            for c in CATS}


def _days(data, s) -> int | None:
    d = (s.get("earn") or {}).get("d")
    if not d:
        return None
    return (date.fromisoformat(d[:10]) - date.fromisoformat(str(data["kpis"]["run_timestamp"])[:10])).days


# --------------------------------------------------------------------------- every line, every stock
def test_every_line_is_the_payloads_own_fact_for_every_stock(data, ctx, cases):
    sd, fw = data["stock_detail"], data["weights"]["factor_weights"]
    assert len(cases) == len(sd) >= 490
    labels = stock_summary.CAT_LABELS
    seen = set()
    for t, s in sd.items():
        got = {x["k"]: x for x in cases[t]["items"]}
        assert len(got) == len(cases[t]["items"]), (t, "a kind listed twice")
        seen |= set(got)
        for k, x in got.items():
            assert x["go"] == GO[k], (t, k)
        sm, fl, scores = _sm(s), s.get("flags") or {}, s.get("cat_scores") or {}

        # 1. the two lowest weighted categories, with the tiles' score and points
        ew = _eff_weights(fw, scores)
        live = sorted((c for c in CATS if ew[c] > 0 and _num(scores.get(c))),
                      key=lambda c: (scores[c], CATS.index(c)))[:2]
        parts = re.findall(r"(\w+) at ([\d.]+) \(([\d.]+|—) of ([\d.]+) points\)", got["weakest"]["text"])
        assert [p[0] for p in parts] == [labels[c] for c in live], t
        for c, (_, sc, pts, w) in zip(live, parts):
            assert abs(float(sc) - scores[c]) <= 0.05 + 1e-9, (t, c)
            assert abs(float(pts) - s["contrib"][c]) <= 0.05 + 1e-9, (t, c)
            assert abs(float(w) - ew[c]) <= 0.05 + 1e-9, (t, c)
        assert got["weakest"]["text"].startswith("Its two weakest categories are " if len(live) == 2
                                                 else "Its weakest category is "), t

        # 2. trap and accounting flags: the summary's own sentence
        flagged = bool(s.get("vt") or s.get("gt") or fl.get("beneish_flag") or fl.get("channel_stuffing"))
        assert ("flags" in got) == flagged, t
        if flagged:
            assert got["flags"]["text"] == sm["flags"] and sm["flags"].startswith("Flagged as"), t

        # 3. mean analyst target below the last price: the summary's own sentence
        below = _num(s.get("price")) and s["price"] > 0 and _num(s.get("pt_mean")) and s["pt_mean"] < s["price"]
        assert ("target" in got) == bool(below), t
        if below:
            assert got["target"]["text"] == sm["target"] and "below the current price" in sm["target"], t

        # 4. the close below its 200-day average: the Trend card's figure
        c = ctx.get(t) or {}
        under = _num(c.get("px")) and _num(c.get("s200")) and c["s200"] > 0 and c["px"] < c["s200"]
        assert ("trend" in got) == bool(under), t
        if under:
            m = re.fullmatch(r"Its last close is −([\d.]+)% against its 200-day average of \$([\d.]+)"
                             r"( \((Uptrend|Downtrend|Mixed trend)\))?\.", got["trend"]["text"])
            assert m, (t, got["trend"]["text"])
            assert abs(float(m.group(1)) - abs(c["px"] / c["s200"] - 1) * 100) <= 0.05 + 1e-9, t
            assert abs(float(m.group(2)) - c["s200"]) <= 0.005 + 1e-9, t

        # 5. insider net selling: the Insider card's figures
        ins = c.get("ins")
        net = bool(ins) and (ins.get("sell_n") or 0) > 0 and (
            not ins.get("buy_n") or (ins.get("sell_value") or 0) > (ins.get("buy_value") or 0))
        assert ("insider" in got) == net, t
        if net:
            m = re.match(r"Officers and directors made (\$[\d.]+[BMK]?) of open-market sales in the last "
                         r"90 days \((\d+) insiders?(, (\d+)% on pre-set plans)?\)", got["insider"]["text"])
            assert m, (t, got["insider"]["text"])
            sv = ins["sell_value"]
            # the Insider card's cMoney: one decimal for $B and $M, none for $K and dollars
            unit = m.group(1)[-1]
            half = {"B": 0.05e9, "M": 0.05e6, "K": 500.0}.get(unit, 0.5)
            assert abs(_money(m.group(1)) - sv) <= half + 1e-6, t
            assert int(m.group(2)) == ins["sell_people"], t
            if m.group(4) is not None:
                assert int(m.group(4)) == round(100 * ins["sell_planned_value"] / sv), t
            assert ("of purchases" in got["insider"]["text"]) == bool(ins.get("buy_n")), t

        # 6. a report 0-14 days after the run: the summary's own sentence
        d = _days(data, s)
        soon = d is not None and 0 <= d <= 14
        assert ("report" in got) == soon, t
        if soon:
            assert got["report"]["text"] == sm["earnings"], t
            said = re.search(r"(\d+) days after this run|the (day) after this run|the day of (this) run", sm["earnings"])
            assert said and (int(said.group(1)) if said.group(1) else 1 if said.group(2) else 0) == d, t

        # 7. Read with care: the summary's own confidence sentence
        caveat = any(not _num(scores.get(k)) for k in CATS) or bool(fl.get("stale_data")) or bool(s.get("eps_mismatch"))
        assert ("caveats" in got) == caveat, t
        if caveat:
            assert got["caveats"]["text"] == sm["confidence"], t
        assert ("churn" in got) == ("input_churn" in sm), t
        if "input_churn" in sm:
            assert got["churn"]["text"] == ((sm["change"] + " ") if "change" in sm else "") + sm["input_churn"], t

        # 8. the filing check's own 2%, and a period the run is behind
        xk = s.get("xchk") or {}
        differ = [(n, abs(xk[k])) for k, n in XNAMES if _num(xk.get(k)) and abs(xk[k]) > 0.02]
        assert ("filing" in got) == bool(differ), t
        if differ:
            shown = re.findall(r"([a-z ]+?) \(([\d.]+)%\)", got["filing"]["text"].split("quarters, ", 1)[1])
            assert [n.strip().removeprefix("and ").strip() for n, _ in shown] == [n for n, _ in differ], t
            for (_, v), (_, a) in zip(shown, differ):
                assert abs(float(v) - a * 100) <= (0.05 if a < 0.1 else 0.5) + 1e-9, t
        nf = s.get("newer_filing") or {}
        assert ("behind" in got) == bool(nf.get("end")), t
        if nf.get("end"):
            assert got["behind"]["text"].startswith(f"The company filed its {nf['form']} for the period to "), t
    # the universe exercises most kinds; a kind no stock carries today is still checked above
    assert {"weakest", "flags", "trend", "insider", "caveats"} <= seen, seen


def test_each_check_is_listed_cleared_or_unavailable(data, cases, page):
    """No silent gaps: every check ends in exactly one of a listed line, 'nothing to list' or
    'not available' - so a stock with nothing against it says so instead of showing a blank."""
    rules = {
        "flags": ("flags", "no trap or accounting flag", None),
        "target": ("target", "average analyst target at or above the price", "analyst targets"),
        "trend": ("trend", "last close at or above its 200-day average", "the 200-day average"),
        "insider": ("insider", "no net insider sales in 90 days", "insider trades"),
        "report": ("report", "no report within 14 days of this run", "the next earnings date"),
        "caveats": ("caveats", "no missing category, stale filing or EPS disagreement", None),
        "filing": ("filing", "inputs agree with its SEC filing within 2%", "the filing check"),
    }
    for t, r in cases.items():
        keys = {x["k"] for x in r["items"]}
        assert "weakest" in keys, t
        assert not r["wait"], (t, "the context file is loaded")
        for name, (k, clear, na) in rules.items():
            hits = [k in keys, clear in r["clear"], na is not None and na in r["na"]]
            assert sum(hits) == 1, (t, name, hits)
        assert set(r["clear"]) <= {v[1] for v in rules.values()}, t
        assert set(r["na"]) <= {v[2] for v in rules.values() if v[2]} | {"the price trend"}, t
    # rendered: the quietest stock says plainly that the checks came back empty
    quiet = min(cases, key=lambda t: (len(cases[t]["items"]), -len(cases[t]["clear"]), data["stock_detail"][t]["rank"]))
    page.evaluate("t => openStockDetail(t)", quiet)
    text = page.evaluate("document.getElementById('modal-case').innerText")
    assert "Checked, nothing to list:" in text, quiet
    assert page.evaluate("document.getElementById('modal-case').hidden") is False
    page.evaluate("closeModal()")


def test_no_line_is_advice_or_a_forbidden_claim(cases):
    texts = set()
    for r in cases.values():
        texts.update(x["text"] for x in r["items"])
        texts.update(r["clear"])
        texts.update(r["na"])
    js = gd._js_drilldown()
    section = js[js.index("THE CASE AGAINST"):js.index("function renderCaseAgainst")]
    texts.update(m.group(1) for m in re.finditer(r"'((?:[^'\\]|\\.)*)'", section))
    html = gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t", data_version="v")
    block = html[html.index('id="modal-case"'):html.index('id="modal-case-list"')]
    texts.add(re.sub(r"<[^>]+>", " ", block))
    assert len(texts) > 100
    for text in texts:
        assert not stock_summary.advice_terms_in(text), text
        assert not claims.forbidden_hits(text), text


def test_the_cut_offs_are_the_ones_the_page_already_states():
    js = gd._js_drilldown()
    page_js = gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t", data_version="v")
    # Reporting soon's longer window, and the filing check's agreement band
    assert "const CASE_REPORT_DAYS = 14;" in js
    assert "[14, 'Next 14 days']" in page_js
    assert "const CASE_FILING_TOL = 0.02;" in js
    assert "if (a <= 0.02) agree.push(p[1]);" in page_js
    # the only numeric literals in the block's logic: the window, the band, the two weakest, 100
    # for a percentage and the '0.1' the filing panel uses to pick its decimals
    section = js[js.index("const CASE_CATS"):js.index("function renderCaseAgainst")]
    code = re.sub(r"'(?:[^'\\]|\\.)*'", "''", section)
    code = re.sub(r"//[^\n]*", "", code)
    assert set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?", code)) <= {"14", "0.02", "0", "1", "2", "100", "0.1"}


def test_context_lines_wait_for_the_context_file(browser, site):
    """Before dashboard_context.js lands the block says so; when it lands the lines appear, the way
    Before you decide re-renders."""
    pg = browser.new_page(viewport={"width": 1440, "height": 1000})
    held = []
    pg.route("**/dashboard_context.js*", lambda route: held.append(route))
    pg.goto((site / "index.html").as_uri(), wait_until="domcontentloaded", timeout=60000)
    pg.wait_for_function("typeof openStockDetail === 'function' && typeof D !== 'undefined' && D.table_data", timeout=60000)
    # Wait until the context request has actually reached the route and is held, rather than a
    # fixed 200 ms: the page asks for the file at the end of its init, but the route handler runs
    # only when that request's event reaches this process, which a loaded machine delayed past
    # 200 ms - and then `assert held` below failed although nothing was wrong with the page
    # (2026-10-27, research/2026-10-27-flaky-tests.md).
    deadline = time.monotonic() + 30
    while not held and time.monotonic() < deadline:
        pg.wait_for_timeout(50)
    data = _payload("dashboard_data.js")
    ctx = _payload("dashboard_context.js")["ctx"]
    # a stock whose context lists something, so the arrival is visible
    t = next(t for t, s in data["stock_detail"].items()
             if (ctx.get(t) or {}).get("s200") and ctx[t]["px"] < ctx[t]["s200"])
    pg.evaluate("t => openStockDetail(t)", t)
    before = pg.evaluate("document.getElementById('modal-case').innerText")
    assert "Loading the price trend and insider trades" in before
    assert "200-day average of" not in before
    assert held, "the context request was not held"
    held[0].continue_()
    pg.wait_for_function("!/Loading the price trend/.test(document.getElementById('modal-case').innerText)", timeout=30000)
    after = pg.evaluate("document.getElementById('modal-case').innerText")
    assert "200-day average of" in after
    # and a stock with no context once the file is in: named as not available, never as clear
    r = pg.evaluate("t => caseItems(Object.assign({}, D.stock_detail[t], {ctx: undefined}))", t)
    assert "the price trend" in r["na"] and "insider trades" in r["na"]
    assert not {"trend", "insider"} & {x["k"] for x in r["items"]}
    pg.close()


def test_a_close_just_below_its_200_day_average_still_reads_as_below(page, data):
    """2026-10-28: BXP closed 0.03% under its 200-day average, the line read "Its last close is
    0.0% against...", with no sign, and the data loop's claims gate refused the whole run. A gap
    too small for one decimal takes two, and the line is always signed."""
    t = next(iter(data["stock_detail"]))
    for px, s200, want in ((60.889999, 60.909738, "−0.03%"), (99.0, 100.0, "−1.0%"),
                           (99.95, 100.0, "−0.05%"), (99.8, 100.0, "−0.2%")):
        r = page.evaluate("([t, px, s200]) => caseItems(Object.assign({}, D.stock_detail[t], "
                          "{ctx: {px: px, s200: s200}}))", [t, px, s200])
        line = next(x["text"] for x in r["items"] if x["k"] == "trend")
        assert line.startswith(f"Its last close is {want} against"), (px, s200, line)


def test_a_small_percent_in_the_context_panels_keeps_its_sign(page):
    """2026-10-29, an all-panel sweep of the live page: AMD's five-year earnings-yield range read
    "-0.0% – 3.2%" (its low was a loss) and INTC's median "-0.0%". `cPlain` formats every context
    panel's percent; a value that rounds to zero at its precision now takes one more decimal."""
    got = page.evaluate("[-0.0003, 0.0002, -0.00049, 0.000001, -0.000001, 0, 0.0123, -0.05, 0.0004]"
                        ".map(v => cPlain(v))")
    assert got == ["-0.03%", "0.02%", "-0.05%", "0.0%", "0.0%", "0.0%", "1.2%", "-5.0%", "0.04%"], got
    assert page.evaluate("cPlain(-0.0003, 2)") == "-0.03%"
    assert page.evaluate("cPlain(-0.00003, 2)") == "-0.003%"
    assert not any(s.startswith("-0.0%") for s in got)


def test_each_line_opens_the_section_that_shows_it(page, cases):
    """Each kind of line, clicked, brings its section to the top of the sheet (opened if folded),
    and the navigation layer still wraps openStockDetail (UX.current follows it)."""
    for k, target in GO.items():
        t = next((t for t, r in cases.items() if any(x["k"] == k for x in r["items"])), None)
        if t is None:
            continue
        page.evaluate("t => openStockDetail(t)", t)
        assert page.evaluate("UX.current") == t
        page.click(f'#modal-case .case-item[data-k="{k}"]')
        # wait for the smooth scroll to settle rather than a fixed 900 ms, which a loaded parallel
        # suite overran (2026-10-27); the assertions below are unchanged
        try:
            page.wait_for_function("""id => { const el = document.getElementById(id), b = el && el.closest('.modal-body');
                if (!b || el.classList.contains('collapsed')) return false;
                const top = el.getBoundingClientRect().top - b.getBoundingClientRect().top;
                return top >= -4 && top <= 0.6 * b.clientHeight; }""", arg=target, timeout=8000)
        except Exception:  # noqa: BLE001 - the assertions below report what was seen
            pass
        top, collapsed, height = page.evaluate("""id => { const el = document.getElementById(id);
            const b = el.closest('.modal-body');
            return [el.getBoundingClientRect().top - b.getBoundingClientRect().top, el.classList.contains('collapsed'),
                    b.clientHeight]; }""", target)
        assert not collapsed, (k, target)
        # at the top of the sheet; a last section that is still unfolding when the scroll is computed
        # can stop short of the top (goToModal, as the tab strip's own Data tab does), but in view
        assert -4 <= top <= 0.6 * height, (k, target, top, height)
        page.evaluate("closeModal()")


def test_the_block_sits_after_the_summary_with_its_own_tab():
    html = gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t", data_version="v")
    i_sum, i_case, i_about = (html.index(f'id="{x}"') for x in ("modal-summary", "modal-case", "modal-about"))
    assert i_sum < i_case < i_about
    assert '<h3 class="case-title" id="case-title">The case against</h3>' in html
    assert "goToModal('modal-case')" in html
    # display only: nothing in a scoring module reads it
    for mod in ("factor_engine.py", "run_screener.py", "portfolio_constructor.py", "stock_summary.py"):
        assert "caseItems" not in (ROOT / mod).read_text(encoding="utf-8"), mod


def test_the_claim_is_registered_and_resolves():
    c = claims.claim("drilldown.case_against")
    assert c.surface == claims.DRILLDOWN
    assert callable(c.resolve())
    assert all(x.startswith("tests/test_case_against.py::") for x in c.checked_by)
