"""The Market backdrop's pictures say what its words say.

WHY THIS EXISTS - owner request, 2026-11-05: the panel's words were good but its visuals did not
explain anything (a 120px sparkline with no axis, and a tick on a bare line). The panel now draws,
for each of the five readings, a gauge of the zones its label comes from with today's value on it,
and for each of the ten series a three-year chart with its ten-year median, zero for a yield
spread and the 2% goal for inflation, plus the ten-year range with its ends.

A picture can contradict its own sentence - the gauge could light "Normal" beside a pill that says
"Tight", or put today's line in a zone it is not in. So:

* the word and the zones come from **one table**, ``market_context.READING_BANDS``: ``readings()``
  takes its word from ``band_state`` and the page is built with the same table written in, never a
  second copy of the cuts (the old inline comparisons are pinned here, so moving them into the
  table changed no word);
* the browser checks run **the generator's own page** on the published payloads and assert
  relationships, never values: the lit zone is the reading's own word, today's line sits inside it,
  the range marker sits at the series' own percentile. Nothing here names a date, a level or a
  series' state (CLAUDE.md rows "data->red" and "clock->red").
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_dashboard as gd  # noqa: E402
import market_context as mc  # noqa: E402

try:
    from playwright import sync_api
except Exception:  # noqa: BLE001
    sync_api = None

MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# --------------------------------------------------------------------------- one table, no copies
def _old_rule(k: str, x: float) -> str:
    """The comparisons ``readings()`` made inline until 2026-11-05, verbatim."""
    if k == "curve":
        return "inverted" if x < 0 else ("flat" if x < 0.5 else "positive")
    if k == "credit":
        return "tight" if x < 33 else ("wide" if x > 67 else "normal")
    if k == "vol":
        return "calm" if x < 15 else ("elevated" if x > 25 else "normal")
    if k == "inflation":
        return "above target" if x > 2.5 else ("near target" if x >= 1.5 else "below target")
    if k == "jobs":
        return "triggered" if x >= 0.5 else "not triggered"
    raise KeyError(k)


@pytest.mark.parametrize("k, values", [
    ("curve", [-2.0, -0.01, 0.0, 0.01, 0.49, 0.5, 0.51, 3.0]),
    ("credit", [0.0, 32.9, 33.0, 33.1, 50.0, 66.9, 67.0, 67.1, 100.0]),
    ("vol", [9.0, 14.99, 15.0, 15.01, 24.99, 25.0, 25.01, 80.0]),
    ("inflation", [-1.0, 1.49, 1.5, 1.51, 2.0, 2.49, 2.5, 2.51, 9.0]),
    ("jobs", [-0.3, 0.0, 0.49, 0.5, 0.51, 1.2]),
])
def test_the_band_table_gives_every_word_the_old_comparisons_gave(k, values):
    """Moving the cuts into one table changed no published word, at either side of any edge."""
    for x in values:
        assert mc.band_state(k, x) == _old_rule(k, x), (k, x)


def test_readings_take_their_word_from_the_band_table():
    summ = {"T10Y3M": {"last": -0.4}, "BAA10Y": {"last": 2.1, "pct_10y": 70.0},
            "VIXCLS": {"last": 31.0, "median_10y": 17.0}, "CPIAUCSL": {"last": 2.5, "date": "2026-09-01"},
            "UNRATE": {"last": 4.4}}
    by = {r["k"]: r for r in mc.readings(summ, 0.5, "real-time")}
    reads = {"curve": -0.4, "credit": 70.0, "vol": 31.0, "inflation": 2.5, "jobs": 0.5}
    assert set(by) == set(reads) == set(mc.READING_BANDS)
    for k, x in reads.items():
        assert by[k]["state"] == mc.band_state(k, x), k


def test_every_band_names_the_field_it_is_read_from():
    for k, b in mc.READING_BANDS.items():
        assert b["read"] in ("last", "pct_10y", "sahm"), k
        assert (b["series"] is None) == (b["read"] == "sahm"), k
        assert b["series"] is None or b["series"] in mc.SERIES, k
        assert len(b["states"]) == len(b["cuts"]) + 1, k
        assert [c[0] for c in b["cuts"]] == sorted(c[0] for c in b["cuts"]), k


def _page_html() -> str:
    return gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t", data_version="v")


def test_the_page_is_built_with_the_band_table_itself():
    html = _page_html()
    assert "__MK_BANDS__" not in html
    start = html.index("const MK_BANDS = ") + len("const MK_BANDS = ")
    table, end = json.JSONDecoder().raw_decode(html[start:])
    assert html[start + end] == ";"
    assert table == mc.READING_BANDS


def test_every_caution_colour_is_a_word_a_band_can_give():
    html = _page_html()
    line = next(ln for ln in html.splitlines() if "const MK_CAUTION = " in ln)
    caution = json.loads(line.split("=", 1)[1].strip().rstrip(";").replace("'", '"'))
    words = {w for b in mc.READING_BANDS.values() for w in b["states"]}
    assert caution and set(caution) <= words


# --------------------------------------------------------------------------- the page, in a browser
@pytest.fixture(scope="module")
def page(tmp_path_factory):
    """The generator's own page (never the committed file) with both published payloads."""
    if sync_api is None:
        pytest.skip("playwright not installed")
    for f in ("dashboard_data.js", "dashboard_context.js"):
        if not (ROOT / f).exists():
            pytest.skip(f"MISSING DATA: no published {f}")
    d = tmp_path_factory.mktemp("mkvis")
    (d / "index.html").write_text(_page_html(), encoding="utf-8")
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
        pg.wait_for_function("typeof renderMarket === 'function' && typeof D !== 'undefined' && "
                             "window.SCREENER_CONTEXT !== undefined", timeout=60000)
        if not pg.evaluate("!!(D.market && D.market.series)"):
            b.close()
            pytest.skip("MISSING DATA: the published context carries no market backdrop")
        pg.evaluate("goToSection('sec-market')")
        pg.wait_for_selector("#market-body .mk-grid", state="attached", timeout=15000)
        yield pg
        b.close()
        assert not errors, errors


_GAUGES_JS = """() => [...document.querySelectorAll('#market-body .mk-reading')].map(r => {
    const g = r.querySelector('.mk-gauge');
    const pct = el => parseFloat((el.getAttribute('style').match(/left:([-0-9.]+)%/) || [])[1]);
    const wid = el => parseFloat((el.getAttribute('style').match(/width:([-0-9.]+)%/) || [])[1]);
    return { pill: r.querySelector('.mk-state').textContent,
             gauge: !!g,
             on: g ? [...g.querySelectorAll('.mk-z.on')].map(z => ({ t: z.getAttribute('title'), l: pct(z), w: wid(z) })) : [],
             x: g ? pct(g.querySelector('.mk-g-x')) : null };
})"""


def test_each_gauge_lights_the_zone_its_reading_names_and_today_sits_inside_it(page):
    readings = page.evaluate(_GAUGES_JS)
    published = page.evaluate("D.market.readings || []")
    assert len(readings) == len(published)
    gauged = [r for r in readings if r["gauge"]]
    # every reading the band table covers has a value to draw (the payload carries it), so each gets one
    assert len(gauged) == sum(1 for r in published if r["k"] in mc.READING_BANDS)
    for r, pub in zip(readings, published):
        if not r["gauge"]:
            continue
        assert len(r["on"]) == 1, r
        z = r["on"][0]
        assert z["t"] == pub["state"] == r["pill"], (z, pub["state"], r["pill"])
        assert z["l"] - 0.01 <= r["x"] <= z["l"] + z["w"] + 0.01, (pub["k"], z, r["x"])


def test_every_series_with_a_history_is_charted_and_its_range_marker_is_its_own_percentile(page):
    got = page.evaluate("""() => [...document.querySelectorAll('#market-body .mk-card')].map(c => ({
        k: (c.querySelector('.mk-ch') || {dataset: {}}).dataset.k || null,
        label: c.querySelector('.mk-label').textContent,
        left: (c.querySelector('.mk-pct-track i') || {getAttribute: () => null}).getAttribute('style') }))""")
    series = page.evaluate("D.market.series")
    by_label = {s["label"]: (k, s) for k, s in series.items()}
    assert {g["label"] for g in got} == set(by_label), "every published series gets one card"
    for g in got:
        k, s = by_label[g["label"]]
        has_line = sum(isinstance(v, (int, float)) for v in (s.get("spark") or [])) >= 3
        assert (g["k"] == k) if has_line else g["k"] is None, g
        if isinstance(s.get("pct_10y"), (int, float)):
            assert g["left"] == f"left:{s['pct_10y']}%", g


def test_hovering_a_chart_reads_that_week_off_it(page):
    k = page.evaluate("""() => { const el = [...document.querySelectorAll('#market-body .mk-ch')].find(e => {
        const v = D.market.series[e.dataset.k].spark; return typeof v[v.length - 1] === 'number'; });
        return el ? el.dataset.k : null; }""")
    if k is None:
        pytest.skip("MISSING DATA: no charted series ends on a reading")
    s = page.evaluate(f"D.market.series[{json.dumps(k)}]")
    el = page.locator(f'#market-body .mk-ch[data-k="{k}"]')
    el.scroll_into_view_if_needed()
    box = el.bounding_box()
    el.hover(position={"x": box["width"] - 0.5, "y": box["height"] / 2})
    tip = el.locator(".mk-tip")
    assert tip.is_visible()
    y, m, d = s["spark_d"][-1].split("-")
    want_date = f"{MON[int(m) - 1]} {int(d)}, {y}"
    want_value = page.evaluate("([u, v]) => mkAt({unit: u}, v)", [s["unit"], s["spark"][-1]])
    assert tip.inner_text().replace("·", "|").split("|")[0].strip() == want_date
    assert want_value in tip.inner_text()
    page.mouse.move(1, 1)


# --------------------------------------------------------------------------- hand-worked gauges
_ONE_GAUGE_JS = """([r, M]) => { const d = document.createElement('div'); d.innerHTML = mkGauge(r, M);
    const pct = el => el ? parseFloat((el.getAttribute('style').match(/left:([-0-9.]+)%/) || [])[1]) : null;
    const on = d.querySelector('.mk-z.on');
    return { on: on ? on.getAttribute('title') : null, warn: !!(on && on.classList.contains('warn')),
             x: pct(d.querySelector('.mk-g-x')), onL: pct(on),
             onW: on ? parseFloat(on.getAttribute('style').match(/width:([-0-9.]+)%/)[1]) : null,
             now: (d.querySelector('.mk-g-now') || {}).textContent || null,
             ends: (d.querySelector('.mk-g-ends') || {}).textContent || '' }; }"""


def test_an_inverted_curve_lights_its_caution_zone_with_today_inside_it(page):
    M = {"series": {"T10Y3M": {"unit": "pp", "last": -0.4, "min_10y": -1.89, "max_10y": 2.27, "median_10y": 0.52}}}
    got = page.evaluate(_ONE_GAUGE_JS, [{"k": "curve", "state": mc.band_state("curve", -0.4), "title": "Yield curve"}, M])
    assert got["on"] == "inverted" and got["warn"]
    assert got["onL"] <= got["x"] <= got["onL"] + got["onW"]
    assert got["now"] == "−0.40pp"   # a true minus, as the series cards print it
    assert "10-yr low" in got["ends"] and "off this scale" not in got["ends"]


def test_a_long_tail_is_cut_off_and_says_so(page):
    """The VIX's ten-year high (2020) would squeeze every zone into a corner; the scale stops past
    the last cut and the end label says the high is off it."""
    M = {"series": {"VIXCLS": {"unit": "", "last": 20.8, "min_10y": 9.14, "max_10y": 82.69, "median_10y": 17.0}}}
    got = page.evaluate(_ONE_GAUGE_JS, [{"k": "vol", "state": mc.band_state("vol", 20.8), "title": "Volatility"}, M])
    assert got["on"] == "normal" and not got["warn"]
    assert got["onL"] <= got["x"] <= got["onL"] + got["onW"]
    assert "off this scale" in got["ends"]


def test_credit_is_drawn_on_its_percentile_scale(page):
    """Credit's word is set by its ten-year percentile, so its gauge runs 0-100 on that scale."""
    M = {"series": {"BAA10Y": {"unit": "pp", "last": 1.54, "pct_10y": 8.8, "min_10y": 1.36, "max_10y": 4.31, "median_10y": 1.93}}}
    got = page.evaluate(_ONE_GAUGE_JS, [{"k": "credit", "state": mc.band_state("credit", 8.8), "title": "Credit spreads"}, M])
    assert got["on"] == "tight" and got["x"] == pytest.approx(8.8, abs=0.01)
    assert got["now"] == "1.54pp"


def test_the_sahm_gauge_reads_the_indicator_and_has_no_ten_year_ends(page):
    got = page.evaluate(_ONE_GAUGE_JS, [{"k": "jobs", "state": mc.band_state("jobs", 0.5), "title": "Labor market"},
                                        {"series": {}, "sahm": 0.5}])
    assert got["on"] == "triggered" and got["warn"]
    assert got["onL"] <= got["x"] <= got["onL"] + got["onW"]
    assert got["now"] == "+0.50pp" and got["ends"] == ""


def test_a_reading_with_nothing_to_draw_draws_nothing(page):
    assert page.evaluate("mkGauge({k: 'vol', state: 'normal', title: 'V'}, {series: {}})") == ""
    assert page.evaluate("mkGauge({k: 'nope', state: 'x', title: 'N'}, {series: {}})") == ""
