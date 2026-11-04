"""The figures the dashboard's JavaScript prints into sentences, read back from the page.

WHY THIS EXISTS - `plan/claims-mutation-audit.md`, "Results, 2026-11-02 (owner-run night)": the
dashboard sweep found 84 computed sentences and left 77 open for one reason. Most dashboard claims
pin the **engine's** number or the **payload's** (row 0.8's design), and the JavaScript that formats
that number into a sentence is a second step no registered check ran: wrong field, wrong scale,
wrong copy of a figure printed twice - all invisible to a check that stops at the payload.

So every test here runs **the generator's own page** (``generate_dashboard.generate_html()``, never
the committed ``index.html``) with the published payloads, calls the function that prints the
sentence, parses the figure out of the text it wrote, and compares it with a value worked out here
from the engine's rule or the payload's inputs - for every stock where the sentence is per stock.
A branch today's data may not take (a coverage discount, a rejected history, no lead to erase) has
a hand-worked fixture with literal expected text, so the branch is checked whatever the morning's
data did. Nothing here names a stock, a run folder or a day (CLAUDE.md row "data->red"), and
nothing reads the clock.

Claims: ``drilldown.composite_chain``, ``drilldown.peer_rank``, ``drilldown.unscored_category``,
``context.track_turnover``, ``context.track_luck``. The data loop's publish gate runs this module
because those claims name it (``scripts/check_published_claims.py`` derives its membership from
``claims.py``).
"""

from __future__ import annotations

import html as htmllib
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_dashboard as gd  # noqa: E402
from factor_engine import METRIC_DIR, SECTOR_MIN_PEERS, metric_weight_profiles  # noqa: E402

try:
    from playwright import sync_api
except Exception:  # noqa: BLE001
    sync_api = None

CATS = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]
NUM = r"(-?\d+(?:\.\d+)?)"


def _payload(name: str) -> dict:
    p = ROOT / name
    if not p.exists():
        pytest.skip(f"MISSING DATA: no published {name}")
    s = p.read_text(encoding="utf-8")
    return json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))


@pytest.fixture(scope="module")
def data() -> dict:
    return _payload("dashboard_data.js")


@pytest.fixture(scope="module")
def track() -> dict:
    T = _payload("dashboard_context.js").get("track") or {}
    if not T.get("available"):
        pytest.skip("MISSING DATA: the published context file carries no track record")
    return T


@pytest.fixture(scope="module")
def cfg() -> dict:
    import yaml
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
def page(browser, tmp_path_factory):
    """The generator's own page with the published payloads, context file loaded."""
    for f in ("dashboard_data.js", "dashboard_context.js"):
        if not (ROOT / f).exists():
            pytest.skip(f"MISSING DATA: no published {f}")
    d = tmp_path_factory.mktemp("printed")
    (d / "index.html").write_text(gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t",
                                                   data_version="v"), encoding="utf-8")
    for f in ("dashboard_data.js", "dashboard_context.js"):
        shutil.copy(ROOT / f, d / f)
    pg = browser.new_page(viewport={"width": 1440, "height": 1000})
    errors: list[str] = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto((d / "index.html").as_uri(), wait_until="load", timeout=60000)
    pg.wait_for_function("typeof renderContribVisual === 'function' && typeof D !== 'undefined' && "
                         "window.SCREENER_CONTEXT !== undefined && D.track !== undefined", timeout=60000)
    yield pg
    pg.close()
    assert not errors, errors


def _text(h: str) -> str:
    """What a reader sees of an HTML fragment: tags dropped, entities decoded, spaces collapsed."""
    return re.sub(r"\s+", " ", htmllib.unescape(re.sub(r"<[^>]+>", " ", h))).strip()


def _jsround(x: float) -> int:
    """JavaScript's Math.round (half up), not Python's half-even."""
    return math.floor(x + 0.5)


def _ordinal(n: int) -> str:
    v = n % 100
    return f"{n}{'th' if 11 <= v <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


# ---------------------------------------------------------------------------
# drilldown.composite_chain - "Category points add up to X", the discount line, the composite
# ---------------------------------------------------------------------------
_CHAIN_JS = """(stocks) => { const C = %s; const out = {};
    for (const t of stocks) { renderContribVisual(D.stock_detail[t], C);
        out[t] = document.getElementById('contrib-total').innerHTML; }
    return out; }""" % json.dumps(CATS)

_TOTAL = re.compile(r"Category points add up to " + NUM)
_DISC = re.compile(r"Coverage discount — values for (\d+) of (\d+) applicable metrics \((\d+)%\), below the "
                   r"(\d+)% threshold, so the composite is reduced by " + NUM + r"% −" + NUM)
_COMPOSITE = re.compile(r"Composite Score: " + NUM + r" / 100")


def _chain(page, tickers) -> dict:
    return {t: _text(h) for t, h in page.evaluate(_CHAIN_JS, list(tickers)).items()}


def test_every_drilldown_prints_the_published_composite_and_the_chain_to_it(page, data, cfg):
    """For every stock: the total is the sum of its category points, the composite is the engine's
    (``table_data``), and the discount line - present exactly when a discount applied - prints the
    coverage counts, the configured threshold, the engine's rate for that coverage and the cut that
    takes the total to the composite."""
    cd = cfg["data_quality"]["coverage_discount"]
    published = {r["Ticker"]: r["Composite"] for r in data["table_data"]}
    sd = data["stock_detail"]
    tickers = [t for t in sd if t in published and published[t] is not None]
    assert len(tickers) >= 400, f"only {len(tickers)} scored stocks to read - not a normal run"
    printed = _chain(page, tickers)
    problems, discounted = [], 0
    for t in tickers:
        s, txt, comp = sd[t], printed[t], published[t]
        m_total, m_comp = _TOTAL.search(txt), _COMPOSITE.search(txt)
        if not (m_total and m_comp):
            problems.append(f"{t}: chain unreadable: {txt!r}")
            continue
        total, shown = float(m_total.group(1)), float(m_comp.group(1))
        pts = sum(v for v in s["contrib"].values() if v is not None)
        if abs(total - pts) > 0.05 + 1e-6:
            problems.append(f"{t}: total {total} is not the category points' sum {pts:.3f}")
        if abs(shown - comp) > 0.05 + 1e-6:
            problems.append(f"{t}: composite printed {shown}, the engine's is {comp}")
        cov = s.get("cov") or {}
        disc = cov.get("disc") or 0
        m_disc = _DISC.search(txt)
        if (m_disc is not None) != (disc > 0):
            problems.append(f"{t}: discount line {'present' if m_disc else 'absent'} with disc={disc}")
            continue
        if not disc:
            # no discount: the points and the composite are one figure, to the payload's rounding
            if abs(total - comp) > 0.1:
                problems.append(f"{t}: undiscounted, yet total {total} differs from composite {comp}")
            continue
        discounted += 1
        n, of, cpct, thr, rate, cut = m_disc.groups()
        n, of = int(n), int(of)
        if (n, of) != (cov["n"], cov["of"]):
            problems.append(f"{t}: printed {n} of {of}, payload {cov['n']} of {cov['of']}")
        if int(cpct) != _jsround(cov["n"] / cov["of"] * 100):
            problems.append(f"{t}: coverage printed {cpct}%, is {cov['n'] / cov['of']:.2%}")
        if int(thr) != _jsround(cd["threshold"] * 100):
            problems.append(f"{t}: threshold printed {thr}%, configured {cd['threshold']}")
        engine_rate = cd["penalty_rate"] * (cd["threshold"] - cov["n"] / cov["of"])
        if abs(float(rate) - engine_rate * 100) > 0.05 + 1e-4:
            problems.append(f"{t}: rate printed {rate}%, the configured rule gives {engine_rate:.4%}")
        if abs(float(cut) - (pts - comp)) > 0.05 + 1e-6:
            problems.append(f"{t}: cut printed {cut}, points {pts:.2f} - composite {comp}")
        if abs(pts * (1 - engine_rate) - comp) > 0.06:
            problems.append(f"{t}: points x (1 - rate) = {pts * (1 - engine_rate):.2f}, composite {comp}")
    assert not problems, f"{len(problems)} problems ({discounted} discounted stocks):\n" + "\n".join(problems[:30])


def _fixture_stock(page, ticker_like: str, patch: str) -> str:
    """Render the chain for a copy of a published stock with ``patch`` (JS on ``s``) applied, under a
    temporarily replaced ``D.weights.coverage_discount``; the page's own state is restored."""
    return page.evaluate("""([t, patch, C]) => { const keep = D.weights.coverage_discount;
        try { const s = structuredClone(D.stock_detail[t]); (new Function('s', 'D', patch))(s, D);
              renderContribVisual(s, C);
              return document.getElementById('contrib-total').innerHTML; }
        finally { D.weights.coverage_discount = keep; } }""", [ticker_like, patch, CATS])


def _any_ticker(data) -> str:
    return next(iter(data["stock_detail"]))


def test_the_discount_line_prints_a_hand_worked_case(page, data):
    """A branch today's data may not take, worked by hand: points 10 + 12.5 + 7.25 + 8 + 4.25 + 5
    + 2 + 1 = 50.00; 18 of 25 metrics is 72%; a threshold of 85% at rate 0.2 cuts 0.2 x 0.13 = 2.6%,
    so the composite is 48.70 and the cut 1.30. The threshold is set to 85% here so a page that
    printed a fixed 80 instead of the configured figure is caught."""
    patch = """s.contrib = {valuation: 10, quality: 12.5, growth: 7.25, momentum: 8, risk: 4.25,
                            revisions: 5, size: 2, investment: 1};
               Object.keys(s.contrib).forEach(c => { s.cat_scores[c] = 50; });
               s.cov = {n: 18, of: 25, disc: 0.026}; s.composite = 48.7;
               D.weights.coverage_discount = {enabled: true, threshold: 0.85, rate: 0.2};"""
    txt = _text(_fixture_stock(page, _any_ticker(data), patch))
    assert "Category points add up to 50.0" in txt, txt
    assert ("Coverage discount — values for 18 of 25 applicable metrics (72%), below the 85% threshold, "
            "so the composite is reduced by 2.6% −1.3") in txt, txt
    assert "Composite Score: 48.7 / 100" in txt, txt


def test_an_undiscounted_chain_prints_no_discount_line(page, data):
    patch = """s.contrib = {valuation: 10, quality: 12.5, growth: 7.25, momentum: 8, risk: 4.25,
                            revisions: 5, size: 2, investment: 1};
               Object.keys(s.contrib).forEach(c => { s.cat_scores[c] = 50; });
               s.cov = {n: 25, of: 25}; s.composite = 50;"""
    txt = _text(_fixture_stock(page, _any_ticker(data), patch))
    assert txt.startswith("Category points add up to 50.0") or "Category points add up to 50.0" in txt, txt
    assert "Coverage discount" not in txt and "Composite Score: 50.0 / 100" in txt, txt


# ---------------------------------------------------------------------------
# drilldown.peer_rank - "Ranked Nth of M in <sector>; sector median ...; That is the Pth percentile."
# ---------------------------------------------------------------------------
_RANK_JS = """() => { const out = {}; const L = Object.keys(D.lineage || {});
    for (const t of Object.keys(D.stock_detail)) { const s = D.stock_detail[t]; out[t] = {};
        for (const m of L) { if (s.raw[m] == null || s.pct[m] == null) continue;
            const h = metricDetailHtml(m, s, null);
            const k = h.indexOf('<span class="wk-k">Rank</span>');
            if (k < 0) { out[t][m] = null; continue; }
            const e = h.indexOf('</div></div>', k);
            out[t][m] = h.slice(k, e); } }
    return out; }"""
_RANK = re.compile(r"Rank Ranked (\d+)(?:st|nd|rd|th) of (\d+) in (.+?), (lower|higher) being better"
                   r"(?:; sector median (.+?) , middle half (.+?) to (.+?))?\. That is the (\d+(?:st|nd|rd|th)) percentile\.$")


def _num_of(printed: str, fmt: str) -> tuple[float, float]:
    """The value a ``fmtMetric`` printing stands for, in raw units, and half its display step."""
    if fmt == "pct":
        return float(printed.rstrip("%")) / 100, 0.0005
    if fmt == "bp":
        return float(printed.replace(" bp", "")) / 10000, 0.00005
    if fmt == "int":
        return float(printed), 0.5
    if fmt == "score":
        return float(printed), 0.05
    if fmt == "ratio":
        return float(printed), 0.005
    return float(printed), 1e-9


def test_every_metric_row_prints_its_own_rank_peer_count_and_sector_figures(page, data):
    """For every stock and every metric row the page can open: the rank is one plus the number of
    peers with a better value in the engine's direction (``METRIC_DIR``), counted over the sector
    when it has ``SECTOR_MIN_PEERS`` values and over the universe otherwise; the median and middle
    half are the sector's own (numpy's linear quartiles of the published raw values); the
    percentile printed is the stock's published percentile, rounded."""
    sd = data["stock_detail"]
    meta = data.get("metric_meta") or {}
    printed = page.evaluate(_RANK_JS)
    by_sector: dict[tuple[str, str], list[float]] = {}
    for s in sd.values():
        for m, v in (s.get("raw") or {}).items():
            if v is not None:
                by_sector.setdefault((s["sector"], m), []).append(v)
    universe: dict[str, list[float]] = {}
    for (sec, m), vals in by_sector.items():
        universe.setdefault(m, []).extend(vals)
    problems, rows, fallbacks, with_stats = [], 0, 0, 0
    for t, per in printed.items():
        s = sd[t]
        for m, h in per.items():
            rows += 1
            if h is None:
                problems.append(f"{t}/{m}: no Rank line")
                continue
            txt = _text(h)
            g = _RANK.search(txt)
            if not g:
                problems.append(f"{t}/{m}: unreadable: {txt!r}")
                continue
            rank, n, where, word, med, q1, q3, pctl = g.groups()
            if m not in METRIC_DIR:
                problems.append(f"{t}/{m}: the engine has no direction for this metric")
                continue
            higher = METRIC_DIR[m]
            sector_vals = by_sector.get((s["sector"], m), [])
            fallback = len(sector_vals) < SECTOR_MIN_PEERS
            peers = universe[m] if fallback else sector_vals
            mine = s["raw"][m]
            better = sum(1 for v in peers if (v > mine if higher else v < mine))
            if (int(rank), int(n)) != (better + 1, len(peers)):
                problems.append(f"{t}/{m}: printed {rank} of {n}, the rule gives {better + 1} of {len(peers)}")
            if word != ("higher" if higher else "lower"):
                problems.append(f"{t}/{m}: says {word} is better, the engine ranks {'higher' if higher else 'lower'}")
            want_where = (f"the whole universe (fewer than {SECTOR_MIN_PEERS} stocks in {s['sector']} have a value)"
                          if fallback else s["sector"])
            if where != want_where:
                problems.append(f"{t}/{m}: ranked 'in {where}', should be 'in {want_where}'")
            if pctl != _ordinal(_jsround(s["pct"][m])):
                problems.append(f"{t}/{m}: printed the {pctl} percentile, published {s['pct'][m]}")
            if fallback:
                fallbacks += 1
                if med is not None:
                    problems.append(f"{t}/{m}: universe-ranked row prints sector figures")
                continue
            if med is None:
                problems.append(f"{t}/{m}: sector-ranked row prints no sector median")
                continue
            with_stats += 1
            fmt = (meta.get(m) or {}).get("fmt")
            want = np.percentile(np.array(sector_vals, dtype=float), [25, 50, 75])
            for label, got, w in (("lower quartile", q1, want[0]), ("median", med, want[1]),
                                  ("upper quartile", q3, want[2])):
                v, half = _num_of(got.strip(), fmt)
                tol = half + 2e-4 + 1e-9 * max(1.0, abs(w))
                if abs(v - w) > tol and not (fmt not in ("pct", "bp", "int", "score", "ratio")
                                             and math.isclose(v, w, rel_tol=1e-3, abs_tol=2e-4)):
                    problems.append(f"{t}/{m}: sector {label} printed {got}, the sector's is {w:.6g}")
    assert rows > 5000, f"only {rows} metric rows read - the page or the payload has changed shape"
    assert not problems, (f"{len(problems)} problems in {rows} rows ({fallbacks} universe-ranked, "
                          f"{with_stats} with sector figures):\n" + "\n".join(problems[:30]))


# ---------------------------------------------------------------------------
# drilldown.unscored_category - the tile's count, the workings' count, the weight shared out
# ---------------------------------------------------------------------------
_UNSCORED_JS = """([t, c, patch]) => { const s = structuredClone(D.stock_detail[t]);
    if (patch) (new Function('s', 'c', patch))(s, c);
    const w = unscoredWhy(c, s);
    renderContribVisual(s, %s);
    const rows = [...document.querySelectorAll('#contrib-visual .contrib-row-na')]
        .map(r => r.innerHTML);
    return {short: w.short, long: w.long, metrics: w.metrics, rows: rows}; }""" % json.dumps(CATS)

_SHORT = re.compile(r"^Not scored: (?:none of its (\d+) metrics has data|its one metric has no data)$")
_LONG = re.compile(r"^(?:None of the (\d+) metrics this category weights for this stock has data in this run "
                   r"\((.+)\)|Its one weighted metric, (.+), has no data for this stock in this run), so the "
                   r"category is not scored and its weight is shared across the others\.")
_SHARED = re.compile(r"Not scored for this stock — its (\d+(?:\.\d)?)% was shared out across the categories below\.")


def _weighted(cfg, c: str, s: dict) -> list[str]:
    table = metric_weight_profiles(cfg, c)[((s.get("wp") or {}).get(c)) or "generic"]
    return [m for m, w in table.items() if w and w > 0]


def _check_unscored(got: dict, cfg, data, t: str, c: str, s: dict) -> list[str]:
    out = []
    weighted = _weighted(cfg, c, s)
    pub = data["weights"]["profiles"][c][((s.get("wp") or {}).get(c)) or "generic"]
    if sorted(weighted) != sorted(m for m, w in pub.items() if w and w > 0):
        out.append(f"{t}/{c}: the published profile is not the engine's")
    n = len(weighted)
    ms, ml = _SHORT.match(got["short"]), _LONG.match(got["long"])
    if not ms or not ml:
        return out + [f"{t}/{c}: unreadable: {got['short']!r} / {got['long']!r}"]
    if int(ms.group(1) or 1) != n or int(ml.group(1) or 1) != n:
        out.append(f"{t}/{c}: printed {ms.group(1) or 1} / {ml.group(1) or 1} metrics, the engine weights {n}")
    named = ml.group(2) or ml.group(3)
    parts = re.split(r", | and ", named) if ml.group(2) else [named]
    labels = {((data.get("metric_meta") or {}).get(m) or {}).get("label") or m for m in weighted}
    if len(parts) != n or set(parts) != labels or got["metrics"] != parts:
        out.append(f"{t}/{c}: names {parts}, the engine's weighted metrics are {sorted(labels)}")
    fw = data["weights"]["factor_weights"][c]
    shared = [_SHARED.search(_text(r)) for r in got["rows"]]
    shared = [float(x.group(1)) for x in shared if x]
    if not any(abs(v - fw) <= 0.05 + 1e-9 for v in shared):
        out.append(f"{t}/{c}: weight shared out printed {shared}, this run's {c} weight is {fw}")
    return out


def test_every_unscored_category_prints_the_engines_count_and_the_runs_weight(page, data, cfg):
    """For every published stock with an unscored category: the tile's and the workings' counts are
    the number of metrics carrying weight in the engine's table for this stock
    (``metric_weight_profiles``), the workings name exactly those, and the contribution row prints
    this run's weight for the category as the weight shared out."""
    sd = data["stock_detail"]
    cases = [(t, c) for t, s in sd.items() for c in CATS if s["cat_scores"].get(c) is None
             and not ((s.get("flags") or {}).get("px_rejected") and c in ("momentum", "risk"))]
    problems = []
    for t, c in cases:
        got = page.evaluate(_UNSCORED_JS, [t, c, ""])
        problems += _check_unscored(got, cfg, data, t, c, sd[t])
    assert not problems, f"{len(problems)} problems in {len(cases)} unscored categories:\n" + "\n".join(problems)


@pytest.mark.parametrize("cat", ["growth", "revisions", "quality"])
def test_an_unscored_category_made_by_hand_prints_the_engines_count(page, data, cfg, cat):
    """The branch on any day: a copy of a scored stock with every weighted metric of ``cat`` blanked
    (and the category's score removed, as the engine would) must print the engine's count, names and
    weight - whether or not the morning's data left any category unscored. Quality runs on a stock's
    own table (bank, Piotroski-conditional or generic), so the count is not a constant."""
    sd = data["stock_detail"]
    t = next((t for t, s in sd.items() if all(s["cat_scores"].get(c) is not None for c in CATS)), None)
    assert t, "no fully scored stock in the payload"
    weighted = _weighted(cfg, cat, sd[t])
    patch = ("s.cat_scores[c] = null; s.contrib[c] = 0; "
             + "".join(f"s.pct[{json.dumps(m)}] = null; s.raw[{json.dumps(m)}] = null; " for m in weighted))
    got = page.evaluate(_UNSCORED_JS, [t, cat, patch])
    probs = _check_unscored(got, cfg, data, t, cat, {**sd[t]})
    assert not probs, probs


# ---------------------------------------------------------------------------
# context.track_turnover / context.track_luck - the track record panel
# ---------------------------------------------------------------------------
_TRACK_JS = """(T) => { const keep = D.track; try { if (T) D.track = T; renderTrack();
        return document.getElementById('track-body').innerHTML; }
    finally { D.track = keep; renderTrack(); } }"""


def _panel(page, T=None) -> str:
    return _text(page.evaluate(_TRACK_JS, T))


def _binom_tail(k: int, n: int) -> float:
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def _break_even(T: dict) -> float | None:
    """The claim's equation, solved here: (1 + top) x prod(1 - 2 c TO_k) = 1 + RSP."""
    tos = [p["turnover"] for p in T["periods"] if p.get("turnover") is not None]
    top, rsp = T["total"]["top"], T["total"]["RSP"]
    if not tos or top <= rsp:
        return None
    lo, hi = 0.0, 0.5 / max(tos)
    for _ in range(200):
        c = (lo + hi) / 2
        v = (1 + top) * math.prod(1 - 2 * c * x for x in tos)
        lo, hi = (c, hi) if v > 1 + rsp else (lo, c)
    return (lo + hi) / 2


def _pct_of(text: str) -> float:
    return float(text.rstrip("%")) / 100


def test_the_track_record_panel_prints_the_records_own_figures(page, track):
    """The turnover tile and the costs paragraph, the coin-toss share, and the day-by-day lead, read
    from the panel the page renders and set against the published record: the mean, least and most
    traded are those of the periods' own turnovers; the coin-toss share is the binomial tail of the
    periods ahead of RSP; the break-even cost solves the claim's equation on the published totals and
    turnovers; the years to two are (2 / t)^2 x days / 252 on the published t and days."""
    T = track
    txt = _panel(page)
    problems = []
    tos = [p["turnover"] for p in T["periods"] if p.get("turnover") is not None]
    TO = T.get("turnover")
    if TO:
        assert tos, "a turnover summary with no period carrying a turnover"
        m = re.search(r"Traded at each rebalance (\d+(?:\.\d)?%) one way, on average; (\S+%) to (\S+%)", txt)
        m2 = re.search(r"At each rebalance the top 25 traded (\S+%) of its value on average, one way", txt)
        if not (m and m2):
            problems.append(f"turnover unreadable: {txt[:400]!r}")
        else:
            mean, lo, hi = (_pct_of(x) for x in m.groups())
            want = sum(tos) / len(tos)
            for label, got, w in (("mean", mean, want), ("least", lo, min(tos)), ("most", hi, max(tos)),
                                  ("paragraph's mean", _pct_of(m2.group(1)), want)):
                if abs(got - w) > 0.005 + 1e-4:
                    problems.append(f"turnover {label} printed {got:.4f}, the periods give {w:.4f}")
        be = _break_even(T)
        mb = re.search(r"a cost of (\S+%) on every dollar traded would have erased its lead over RSP", txt)
        if be is None:
            if mb:
                problems.append("a break-even cost is printed with no lead over RSP to erase")
        elif not mb:
            problems.append(f"no break-even cost printed; the equation gives {be:.5f}")
        elif abs(_pct_of(mb.group(1)) - be) > 0.00005 + 0.0001:
            problems.append(f"break-even printed {mb.group(1)}, the equation gives {be:.5%}")
    beat = sum(1 for p in T["periods"] if p.get("top") is not None and p.get("RSP") is not None
               and p["top"] > p["RSP"])
    compared = sum(1 for p in T["periods"] if p.get("top") is not None and p.get("RSP") is not None)
    mc = re.search(r"Periods ahead of RSP (\d+) of (\d+) a coin toss does at least this well (\d+)% of the time", txt)
    if (beat, compared) != (T["beat_rsp"], T["periods_compared"]):
        problems.append(f"the record's {T['beat_rsp']} of {T['periods_compared']} is not its periods' {beat} of {compared}")
    if not mc:
        problems.append(f"coin-toss share unreadable: {txt[:600]!r}")
    else:
        if (int(mc.group(1)), int(mc.group(2))) != (beat, compared):
            problems.append(f"printed {mc.group(1)} of {mc.group(2)}, the periods give {beat} of {compared}")
        if int(mc.group(3)) != _jsround(_binom_tail(beat, compared) * 100):
            problems.append(f"coin toss printed {mc.group(3)}%, the binomial tail is {_binom_tail(beat, compared):.4f}")
    X = T.get("excess")
    if X and X.get("t") is not None:
        mx = re.search(r"Measured day by day against RSP over (\d+) trading days, the top 25’s lead is (-?\d+\.\d) "
                       r"standard errors from zero; two is the usual minimum for telling skill from luck"
                       r"(?:, and even if the lead went on at the same pace it would take about (\d+\.\d) years "
                       r"of record to get there)?\.", txt)
        if not mx:
            problems.append(f"the excess sentence is unreadable: {txt[-2500:]!r}")
        else:
            days, t, years = mx.groups()
            if int(days) != X["days"] or abs(float(t) - X["t"]) > 0.05 + 1e-9:
                problems.append(f"printed {days} days and t {t}, published {X['days']} and {X['t']}")
            if X["t"] > 0:
                f = lambda tt: (2 / tt) ** 2 * X["days"] / 252  # noqa: E731
                lo_y, hi_y = f(X["t"] + 0.005), f(max(X["t"] - 0.005, 1e-9))
                if years is None or not (lo_y - 0.05 - 1e-9 <= float(years) <= hi_y + 0.05 + 1e-9):
                    problems.append(f"years to two printed {years}, (2/t)^2 x days / 252 is "
                                    f"{lo_y:.2f}..{hi_y:.2f}")
            elif years is not None:
                problems.append(f"a lead of t={X['t']} prints years to two ({years}); there is no lead")
    assert not problems, "\n".join(problems)


def _fixture_track(track: dict, **changes) -> dict:
    T = json.loads(json.dumps(track))
    for k, v in changes.items():
        T[k] = v
    return T


def test_the_track_panel_prints_a_hand_worked_record(page, track):
    """Literal figures, worked by hand, so the printing is checked whatever the record holds:
    turnovers 20%, 35%, 50% (mean 35%), a break-even cost of 1.23%, 3 of 4 periods ahead of RSP
    (the tail of 3 or more of 4 fair tosses is 5/16, 31%), and a lead of t = 1.0 over 100 days, whose
    years to two are 4 x 100 / 252 = 1.6."""
    T = _fixture_track(track, turnover={"rebalances": 3, "mean": 0.35, "min": 0.2, "max": 0.5,
                                        "break_even_cost": 0.0123, "traded": 2.1, "basis": "x"},
                       beat_rsp=3, periods_compared=4, beat_rsp_chance=0.3125,
                       excess={"days": 100, "t": 1.0, "years_to_two": 1.6})
    txt = _panel(page, T)
    for want in ("Traded at each rebalance 35% one way, on average; 20% to 50%",
                 "Periods ahead of RSP 3 of 4 a coin toss does at least this well 31% of the time",
                 "At each rebalance the top 25 traded 35% of its value on average, one way; a cost of 1.23% "
                 "on every dollar traded would have erased its lead over RSP.",
                 "Measured day by day against RSP over 100 trading days, the top 25’s lead is 1.0 standard "
                 "errors from zero; two is the usual minimum for telling skill from luck, and even if the lead "
                 "went on at the same pace it would take about 1.6 years of record to get there."):
        assert want in txt, (want, txt)


def test_the_track_panel_drops_the_clauses_it_has_no_figure_for(page, track):
    """No lead to erase: no break-even clause. No positive lead: no years clause. No turnover: the
    costs sentence alone."""
    T = _fixture_track(track, turnover={"rebalances": 3, "mean": 0.35, "min": 0.2, "max": 0.5,
                                        "break_even_cost": None, "traded": 2.1, "basis": "x"},
                       excess={"days": 100, "t": -0.4, "years_to_two": None})
    txt = _panel(page, T)
    assert "At each rebalance the top 25 traded 35% of its value on average, one way. " in txt, txt
    assert "a cost of" not in txt and "years of record" not in txt, txt
    assert "lead is -0.4 standard errors from zero; two is the usual minimum for telling skill from luck. " in txt, txt
    txt = _panel(page, _fixture_track(track, turnover=None))
    assert "Traded at each rebalance" not in txt and "No trading costs or taxes are deducted. " in txt, txt
    assert "At each rebalance the top 25 traded" not in txt, txt


def test_an_unscored_category_gives_up_this_runs_weight_not_the_default(page, data, cfg):
    """In the momentum panic state the run scores Momentum at 13 x 0.70 = 9.1% and splits the freed
    3.9 points between Quality and Valuation (22 + 1.95 = 23.95 each). A stock whose Momentum is
    unscored gives up the 9.1% it would have carried that run, so the row must say 9.1%, not the
    configured 13%. Worked by hand on a copy of the weights; the page's own are restored."""
    sd = data["stock_detail"]
    t = next((t for t, s in sd.items() if all(s["cat_scores"].get(c) is not None for c in CATS)), None)
    assert t, "no fully scored stock in the payload"
    weighted = _weighted(cfg, "momentum", sd[t])
    patch = ("s.cat_scores[c] = null; s.contrib[c] = 0; "
             + "".join(f"s.pct[{json.dumps(m)}] = null; s.raw[{json.dumps(m)}] = null; " for m in weighted))
    rows = page.evaluate("""([t, patch, C]) => { const keep = D.weights.factor_weights;
        try { D.weights.factor_weights = Object.assign({}, keep, {momentum: 9.1, quality: 23.95, valuation: 23.95});
              const s = structuredClone(D.stock_detail[t]); (new Function('s', 'c', patch))(s, 'momentum');
              renderContribVisual(s, C);
              return [...document.querySelectorAll('#contrib-visual .contrib-row-na')].map(r => r.innerHTML); }
        finally { D.weights.factor_weights = keep; } }""", [t, patch, CATS])
    texts = [_text(r) for r in rows]
    assert any("Not scored for this stock — its 9.1% was shared out across the categories below." in x
               for x in texts), texts


_PEER_FIXTURE_JS = """([k, m, st, probe]) => { const added = [];
    const keepStats = D.sector_stats['Fixture Sector'];
    try { for (let i = 1; i <= k; i++) { const t = '__FX' + i; added.push(t);
              D.stock_detail[t] = {sector: 'Fixture Sector', raw: {[m]: i / 100}, pct: {[m]: i * 10}}; }
          if (st) D.sector_stats['Fixture Sector'] = {[m]: st};
          return metricDetailHtml(m, D.stock_detail['__FX' + probe], null); }
    finally { added.forEach(t => delete D.stock_detail[t]);
              if (keepStats === undefined) delete D.sector_stats['Fixture Sector'];
              else D.sector_stats['Fixture Sector'] = keepStats; } }"""


def _fixture_rank(page, k, st, probe=4, m="roic") -> str:
    h = page.evaluate(_PEER_FIXTURE_JS, [k, m, st, probe])
    i = h.index('<span class="wk-k">Rank</span>')
    return _text(h[i:h.index("</div></div>", i)])


def test_a_sector_with_exactly_the_minimum_prints_its_own_rank_and_figures(page, data):
    """Branches today's data does not take (every sector has well over ten values for every metric):
    ten fixture stocks with ROIC 1%..10% form a sector of exactly ``SECTOR_MIN_PEERS``, so it is
    ranked within itself. The stock at 4% has six better, so it is 7th of 10; the sector figures are
    the published ones (set here to 3.0%, 5.5% and 8.0%); its percentile is the published 40."""
    assert SECTOR_MIN_PEERS == 10 and data["sector_min_peers"] == 10
    assert (data["metric_meta"].get("roic") or {}).get("fmt") == "pct" and METRIC_DIR["roic"] is True
    txt = _fixture_rank(page, 10, [10, 0.03, 0.055, 0.08])
    assert txt == ("Rank Ranked 7th of 10 in Fixture Sector, higher being better; sector median 5.5% , "
                   "middle half 3.0% to 8.0%. That is the 40th percentile."), txt


def test_a_sector_below_the_minimum_prints_the_universe_rank(page, data):
    """Nine fixture stocks: below the minimum, so the stock is ranked against the whole universe -
    every published ROIC plus the nine - with no sector figures, and the sentence says why."""
    vals = [s["raw"]["roic"] for s in data["stock_detail"].values() if s["raw"].get("roic") is not None]
    assert len(vals) > 100, "the payload carries too few ROIC values to rank against"
    better = sum(1 for v in vals if v > 0.04) + 5            # 5%..9% among the nine
    txt = _fixture_rank(page, 9, [9, 0.03, 0.05, 0.07])
    assert txt == (f"Rank Ranked {_ordinal(better + 1)} of {len(vals) + 9} in the whole universe (fewer than 10 "
                   "stocks in Fixture Sector have a value), higher being better. That is the 40th percentile."), txt


@pytest.mark.parametrize("n,word", [(11, "11th"), (12, "12th"), (13, "13th"), (21, "21st"), (22, "22nd"),
                                    (23, "23rd"), (111, "111th"), (101, "101st")])
def test_the_ordinals_the_rank_sentence_prints(page, n, word):
    assert page.evaluate("n => ordinal(n)", n) == word


# ---------------------------------------------------------------------------
# drilldown.composite_distribution - the strip beside the composite in the sheet's masthead
# ---------------------------------------------------------------------------
_DIST_JS = """() => { const out = {};
    for (const t of Object.keys(D.stock_detail)) { const s = D.stock_detail[t];
        if (typeof s.composite !== 'number') continue;
        const h = compDistHtml(s.composite);
        const lab = (h.match(/aria-label="([^"]*)"/) || [])[1];
        const cap = (h.match(/<span class="cd-cap"[^>]*>([^<]*)<[/]span>/) || [])[1];
        out[t] = [lab === undefined ? null : lab, cap === undefined ? null : cap,
                  (h.match(/class="cd-on"/g) || []).length,
                  (h.match(/<rect x="([^"]*)"[^>]*class="cd-on"/) || [])[1] || null]; }
    return out; }"""
_DIST = re.compile(r"^Composite (\d+(?:\.\d)?) among the (\d+) published composites(?:; the median is (\d+(?:\.\d)?))?$")


def test_every_masthead_strip_prints_the_composite_the_count_and_the_median(page, data):
    """For every stock: the strip's label states the engine's composite, the number of published
    composites, and their median - computed here from ``table_data`` - and the caption under the
    strip prints the same median; exactly one bin is marked."""
    import statistics
    comps = [r["Composite"] for r in data["table_data"] if isinstance(r.get("Composite"), (int, float))]
    by_t = {r["Ticker"]: r["Composite"] for r in data["table_data"]}
    assert len(comps) >= 400, f"only {len(comps)} published composites - not a normal run"
    median = statistics.median(comps)
    n_detail = sum(1 for s in data["stock_detail"].values() if isinstance(s.get("composite"), (int, float)))
    printed = page.evaluate(_DIST_JS)
    problems = []
    vals = [s["composite"] for s in data["stock_detail"].values() if isinstance(s.get("composite"), (int, float))]
    lo = math.floor(min(vals) / 5) * 5
    hi = max(lo + 5, math.ceil(max(vals) / 5) * 5)
    w = (hi - lo) / 40
    for t, (lab, cap, on, x) in printed.items():
        g = _DIST.match(htmllib.unescape(lab or ""))
        if not g:
            problems.append(f"{t}: label unreadable: {lab!r}")
            continue
        score, n, med = g.groups()
        if abs(float(score) - by_t[t]) > 0.05 + 1e-9:
            problems.append(f"{t}: composite printed {score}, the engine's is {by_t[t]}")
        if int(n) != n_detail or int(n) != len(comps):
            problems.append(f"{t}: printed {n} published composites; the payload carries {n_detail}, the table {len(comps)}")
        if med is None or abs(float(med) - median) > 0.05 + 0.005:
            problems.append(f"{t}: median printed {med}, the published composites' median is {median:.3f}")
        if cap != f"median {med}":
            problems.append(f"{t}: the caption prints {cap!r}, the label's median is {med}")
        k = max(0, min(39, math.floor((data["stock_detail"][t]["composite"] - lo) / w)))
        if on != 1 or x != f"{k * 4 + 0.5:.1f}":
            problems.append(f"{t}: {on} bins marked, at x={x}; its composite falls in bin {k} (x={k * 4 + 0.5:.1f})")
    assert len(printed) == n_detail
    assert not problems, f"{len(problems)} problems:\n" + "\n".join(problems[:30])


def test_a_strip_with_no_published_median_says_none(page, data):
    """The branch a run without ``median_composite`` takes: no median in the label, no caption."""
    t = _any_ticker(data)
    h = page.evaluate("""(t) => { const keep = D.kpis.median_composite;
        try { delete D.kpis.median_composite; return compDistHtml(D.stock_detail[t].composite); }
        finally { D.kpis.median_composite = keep; } }""", t)
    assert "the median is" not in h and "cd-cap" not in h and "among the" in h, h


# ---------------------------------------------------------------------------
# overview.momentum_regime - "Momentum rule, this run." on the Methodology dialog
# ---------------------------------------------------------------------------
_RUN_NOTE_JS = """(m) => { const keep = D.weights.momentum_regime;
    try { hydrateMethodology();
          document.querySelectorAll('#methodology-modal .method-run-note').forEach(e => e.remove());
          if (m) D.weights.momentum_regime = m;
          methodologyRunNote();
          const e = document.querySelector('#methodology-modal .method-run-note');
          return e ? e.innerHTML : null; }
    finally { D.weights.momentum_regime = keep;
              document.querySelectorAll('#methodology-modal .method-run-note').forEach(e => e.remove());
              methodologyRunNote(); } }"""
_REGIME = r"over (\d+) months, with realised volatility of (\d+\.\d)% a year over (\d+) trading days\. "


def _run_note(page, m=None) -> str:
    h = page.evaluate(_RUN_NOTE_JS, m)
    assert h is not None, "the Methodology dialog wrote no run note"
    return _text(h)


def test_the_run_note_prints_the_runs_own_regime_reading(page, data, cfg):
    """The run's reading against the engine's published one and the configured rule: the return and
    its sign, the two windows and the threshold as configured, and the panic sentence exactly when
    the return is negative and the volatility above the threshold (strict, as the engine applies it)."""
    m = data["weights"].get("momentum_regime") or {}
    rc = cfg["momentum_regime"]
    txt = _run_note(page)
    if m.get("state") == "UNKNOWN" or m.get("vol") is None or m.get("ret") is None:
        assert "could not be read" in txt, txt
        return
    g = re.search(r": (−|\+)(\d+\.\d)% " + _REGIME, txt)
    assert g, txt
    sign, ret, months, vol, days = g.groups()
    assert (sign == "−") == (m["ret"] < 0) and abs(float(ret) - abs(m["ret"]) * 100) <= 0.05 + 1e-9, (txt, m)
    assert abs(float(vol) - m["vol"] * 100) <= 0.05 + 1e-9, (txt, m)
    assert (int(months), int(days)) == (rc["return_window_months"], rc["vol_window_days"]), txt
    thr = f"{rc['vol_threshold'] * 100:.1f}%"
    panic = m["ret"] < 0 and m["vol"] > rc["vol_threshold"]
    assert (m["state"] == "PANIC") == panic, f"the published state {m['state']} is not the rule's reading"
    want = (f"The return is negative and volatility is above {thr}, so the market is in the panic state and "
            "Momentum’s weight is cut for this run." if panic else
            f"The panic state needs a negative return and volatility above {thr}, so Momentum’s weight is unchanged.")
    assert txt.endswith(want), (txt, want)


_FIX_REGIME = {"ret": -0.0512, "vol": 0.2534, "vol_threshold": 0.206, "return_window_months": 24,
               "vol_window_days": 126, "as_of": "2026-03-02", "source": "x"}


@pytest.mark.parametrize("m,want", [
    ({**_FIX_REGIME, "state": "PANIC"},
     "−5.1% over 24 months, with realised volatility of 25.3% a year over 126 trading days. The return is "
     "negative and volatility is above 20.6%, so the market is in the panic state and Momentum’s weight is cut "
     "for this run."),
    ({**_FIX_REGIME, "state": "NORMAL", "vol": 0.152, "ret": -0.031},
     "−3.1% over 24 months, with realised volatility of 15.2% a year over 126 trading days. The panic state "
     "needs a negative return and volatility above 20.6%, so Momentum’s weight is unchanged."),
    ({**_FIX_REGIME, "state": "NORMAL", "vol": 0.2534, "ret": 0.0875, "vol_threshold": 0.25,
      "return_window_months": 18, "vol_window_days": 100},
     "+8.8% over 18 months, with realised volatility of 25.3% a year over 100 trading days. The panic state "
     "needs a negative return and volatility above 25.0%, so Momentum’s weight is unchanged."),
    ({**_FIX_REGIME, "state": "UNKNOWN", "vol": None},
     "The index history could not be read, so the momentum regime rule made no change."),
])
def test_the_run_note_prints_a_hand_worked_reading(page, m, want):
    """Each branch with literal figures, whatever the morning's regime: the panic state, a negative
    return without the volatility, a positive return under other windows and threshold (so a page
    that printed the configured figures instead of the run's would be caught), and no reading."""
    txt = _run_note(page, m)
    assert txt.endswith(want), (txt, want)


# ---------------------------------------------------------------------------
# history.run_overview - the What Changed footnote and the section tile's movers counts
# ---------------------------------------------------------------------------
_FOOT_JS = """([k, mv]) => { const keep = H.movers[k], keepR = changedRange;
    try { if (mv) H.movers[k] = mv; setChangedRange(k);
          return document.getElementById('changed-footnote').innerHTML; }
    finally { H.movers[k] = keep; setChangedRange(keepR); } }"""
_TILE_JS = """(movers) => { const keep = H.movers;
    try { if (movers) H.movers = movers; renderTileViz();
          const v = document.querySelector('#sec-changed .tile-viz');
          return v ? [v.getAttribute('aria-label'), v.querySelector('.tv-cap').innerHTML] : null; }
    finally { H.movers = keep; renderTileViz(); } }"""


def _counts(history: dict, k: str) -> tuple[int, int]:
    """Stocks whose rank moved the material threshold or more, each way - the rule the panel states."""
    thr = history["noise"]["material_threshold"]
    drs = [d[k]["dr"] for d in history["delta"].values() if (d.get(k) or {}).get("dr") is not None]
    return sum(1 for x in drs if x >= thr), sum(1 for x in drs if x <= -thr)


def _history(data) -> dict:
    H = data.get("history") or {}
    if not (H.get("available") and H.get("movers")):
        pytest.skip("MISSING DATA: the payload carries no comparable history")
    return H


def test_the_whats_changed_footnote_prints_the_runs_own_counts(page, data):
    """For each comparison window the payload carries: the up and down counts are the stocks whose
    rank moved the material threshold or more (counted here from each stock's published rank
    change), the shown counts are the lengths of the two lists, and the round-trip share is the
    payload's count over all material movers."""
    H = _history(data)
    problems = []
    for k, mv in H["movers"].items():
        if k not in (H.get("compare") or {}):
            continue
        up, down = _counts(H, k)
        txt = _text(page.evaluate(_FOOT_JS, [k, None]))
        g = re.search(r"(\d+) stocks moved up materially and (\d+) moved down; showing the largest (\d+) and (\d+)\.", txt)
        if not g:
            problems.append(f"{k}: unreadable: {txt[:300]!r}")
            continue
        got = tuple(int(x) for x in g.groups())
        want = (up, down, min(len(mv["up"]), up), min(len(mv["down"]), down))
        if got != want:
            problems.append(f"{k}: printed {got}, the rank changes give {want}")
        r = re.search(r"(\d+) of (\d+) are flagged round-trip", txt)
        if (r is not None) != bool(mv.get("n_round_trip")):
            problems.append(f"{k}: round-trip sentence {'present' if r else 'absent'} with n_round_trip={mv.get('n_round_trip')}")
        elif r and (int(r.group(1)), int(r.group(2))) != (mv["n_round_trip"], up + down):
            problems.append(f"{k}: printed {r.group(1)} of {r.group(2)} round-trip, want {mv['n_round_trip']} of {up + down}")
    assert not problems, "\n".join(problems)


def test_the_whats_changed_footnote_prints_a_hand_worked_window(page, data):
    """Lists shorter than the counts, a round trip, and nothing down - worked by hand."""
    H = _history(data)
    k = next(iter(H["movers"]))
    mv = {**H["movers"][k], "up": H["movers"][k]["up"][:3], "down": [], "n_up": 5, "n_down": 0, "n_round_trip": 2}
    txt = _text(page.evaluate(_FOOT_JS, [k, mv]))
    assert "5 stocks moved up materially and 0 moved down; showing the largest 3 and 0." in txt, txt
    assert "2 of 5 are flagged round-trip" in txt, txt
    txt = _text(page.evaluate(_FOOT_JS, [k, {**mv, "n_round_trip": 0}]))
    assert "flagged round-trip" not in txt, txt


def test_the_whats_changed_tile_prints_the_runs_own_counts(page, data):
    """The tile reads the ~1-month window where there is one, else the last run; its counts are the
    rule's, out of the run's universe."""
    H = _history(data)
    k = "m1" if H["movers"].get("m1") else "prev"
    up, down = _counts(H, k)
    N = data["kpis"]["universe_size"]
    got = page.evaluate(_TILE_JS, None)
    assert got, "the What Changed section drew no tile"
    label, cap = got
    assert label == f"{up} of {N} stocks moved up materially and {down} moved down", (label, k, up, down, N)
    assert _text(cap) == f"{up} up of {N} {down} down", (cap, k)


def test_the_whats_changed_tile_prints_a_hand_worked_window(page, data):
    """Without a ~1-month window the tile reads the last run's: 7 up and 4 down of the universe."""
    H = _history(data)
    prev = H["movers"].get("prev") or next(iter(H["movers"].values()))
    got = page.evaluate(_TILE_JS, {"prev": {**prev, "n_up": 7, "n_down": 4}})
    N = data["kpis"]["universe_size"]
    assert got and got[0] == f"7 of {N} stocks moved up materially and 4 moved down", got
    assert _text(got[1]) == f"7 up of {N} 4 down", got


# ---------------------------------------------------------------------------
# context.rate_sensitivity - the drilldown's "Sensitivity to interest rates" card
# ---------------------------------------------------------------------------
_RATE_JS = """([tickers, patch]) => { const out = {}; const keepSr = (D.market || {}).sector_rate;
    try { for (const t of tickers) { let s = D.stock_detail[t];
              if (patch) { s = structuredClone(s); (new Function('s', 'D', patch))(s, D); }
              renderContext(t, s);
              const h = document.getElementById('modal-context').innerHTML;
              const i = h.indexOf('<span>Sensitivity to interest rates</span>');
              out[t] = i < 0 ? null : h.slice(i, h.indexOf('</p></div>', i)); } }
    finally { if (D.market) D.market.sector_rate = keepSr; }
    return out; }"""
# Two decimals, or three where the slope would round to zero at two - `pctNearZero`'s rule, which
# `cPct` has taken since 2026-11-03 (BKR's +0.003% printed as an unsigned "0.00%" until then).
_RATE = re.compile(r"^Sensitivity to interest rates ([+−]?)(\d+\.\d\d\d?)% How the stock has typically moved for each "
                   r"\+0\.10 percentage point rise in the 10-year Treasury yield \(a large day for the 10-year\), from "
                   r"the past (\d+) months of daily moves(?:; (.+) median ([+−]?)(\d+\.\d\d\d?)%)?\. Rate moves explained "
                   r"(\d+(?:\.\d)?)% of its daily swings(, very little, so read the figure loosely)?\.?$")


def _signed(sign: str, v: str) -> float:
    return -float(v) if sign == "−" else float(v)


def _check_near_zero(sign: str, printed: str, want: float, what: str, slack: float = 0.0) -> list[str]:
    """A figure printed by ``cPct``, checked against the value and against the page's near-zero
    rule (``generate_dashboard.pctNearZero``, 2026-11-03): the decimals shown bound the error, a
    figure that prints as a zero is **unsigned**, and a value that is not zero does not print as
    a zero while a decimal would show it."""
    out = []
    dec = len(printed.split(".")[1]) if "." in printed else 0
    if abs(_signed(sign, printed) - want) > 0.5 * 10 ** -dec + 1e-6 + slack:
        out.append(f"printed {sign}{printed}% for {what}, which is {want:.4f}%")
    if float(printed) == 0:
        if sign:
            out.append(f"printed a signed zero, {sign}{printed}%, for {what}")
        if round(want, dec + 1) != 0:
            out.append(f"printed {printed}% for {what}, which is {want:.4f}% - a decimal would show it")
    return out


def _check_rate(txt: str, c: dict, sector: str, sector_median: float | None) -> list[str]:
    g = _RATE.match(txt)
    if not g:
        return [f"unreadable: {txt!r}"]
    out = []
    sg, big, months, sec, ssg, smed, r2, loose = g.groups()
    out += _check_near_zero(sg, big, c["rb"] * 10, "the stock's slope")
    if int(months) != _jsround(c["rn"] / 21):
        out.append(f"printed {months} months, {c['rn']} trading days is {c['rn'] / 21:.2f}")
    if (sec is None) != (sector_median is None):
        out.append(f"sector clause {'absent' if sec is None else 'present'}, sector median {sector_median}")
    elif sec is not None:
        if sec != sector:
            out.append(f"printed '{sec} median', the stock's sector is {sector}")
        out += _check_near_zero(ssg, smed, sector_median * 10, f"{sector}'s median slope", 2e-4)
    whole = c["rr2"] * 100
    # cPlain's rule: a whole percent; a non-zero share that rounds to 0 gets one decimal, and one
    # still 0 at one decimal reads "0"
    want = str(_jsround(whole))
    if want == "0" and whole != 0:
        one = _jsround(whole * 10) / 10
        want = f"{one:.1f}" if one else "0"
    if r2 != want:
        out.append(f"explained printed {r2}%, the squared correlation is {whole:.3f}% (want {want}%)")
    if bool(loose) != (c["rr2"] < 0.05):
        out.append(f"'very little' clause {'present' if loose else 'absent'} at rr2={c['rr2']}")
    return out


def test_every_rate_card_prints_the_stocks_slope_window_sector_median_and_share(page, data):
    """For every stock with a rate reading: the slope per +0.10 point, the window in months (21
    trading days each), the sector median - recomputed here as the median slope of the sector's
    stocks, shown only for a sector of five or more - and the share of daily swings explained, with
    the 'read loosely' clause exactly under 5%."""
    import statistics
    ctx = _payload("dashboard_context.js")["ctx"]
    sd = data["stock_detail"]
    have = [t for t in sd if (ctx.get(t) or {}).get("rb") is not None]
    assert len(have) >= 400, f"only {len(have)} stocks carry a rate reading"
    by_sector: dict[str, list[float]] = {}
    for t in have:
        by_sector.setdefault(sd[t]["sector"], []).append(ctx[t]["rb"])
    med = {k: (statistics.median(v) if len(v) >= 5 else None) for k, v in by_sector.items()}
    printed = page.evaluate(_RATE_JS, [have, ""])
    problems = []
    for t in have:
        if printed.get(t) is None:
            problems.append(f"{t}: no rate card")
            continue
        problems += [f"{t}: {p}" for p in _check_rate(_text(printed[t]), ctx[t], sd[t]["sector"], med[sd[t]["sector"]])]
    assert not problems, f"{len(problems)} problems:\n" + "\n".join(problems[:30])


@pytest.mark.parametrize("rb,rn,rr2,sr,want", [
    (0.2312, 252, 0.031, None,
     "+2.31% How the stock has typically moved for each +0.10 percentage point rise in the 10-year Treasury yield "
     "(a large day for the 10-year), from the past 12 months of daily moves. Rate moves explained 3% of its daily "
     "swings, very little, so read the figure loosely."),
    (-0.0874, 273, 0.12, {"median": -0.05, "n": 7},
     "−0.87% How the stock has typically moved for each +0.10 percentage point rise in the 10-year Treasury yield "
     "(a large day for the 10-year), from the past 13 months of daily moves; Fixture Sector median −0.50%. Rate "
     "moves explained 12% of its daily swings."),
    (0.0, 273, 0.004, None, "0.00% How the stock"),
    (0.05, 273, 0.004, None, "explained 0.4% of its daily swings, very little"),
])
def test_a_rate_card_worked_by_hand(page, data, rb, rn, rr2, sr, want):
    """Literal figures for each branch: no sector median, a sector median, a zero slope, and a share
    explained too small to print as a whole percent."""
    t = _any_ticker(data)
    patch = (f"s.ctx = Object.assign({{}}, s.ctx || {{}}, {{rb: {rb}, rn: {rn}, rr2: {rr2}}}); s.sector = 'Fixture Sector'; "
             + (f"D.market.sector_rate = Object.assign({{}}, D.market.sector_rate, {{'Fixture Sector': {json.dumps(sr)}}});"
                if sr else ""))
    got = page.evaluate(_RATE_JS, [[t], patch])[t]
    assert got is not None, "no rate card"
    assert want in _text(got), (want, _text(got))


# ---------------------------------------------------------------------------
# context.valuation_history - "Against its own five years"
# ---------------------------------------------------------------------------
_VH_JS = """(vhs) => { const out = {};
    for (const t of Object.keys(vhs)) { const h = vhCard({vh: vhs[t]});
        out[t] = [...h.matchAll(/<div class="ctx-facts">([\\s\\S]*?)<\\/div><\\/div><\\/div>/g)].map(m => m[1]); }
    return out; }"""
_VH = re.compile(r"^(Earnings yield|Free-cash-flow yield) (-?\d+\.\d+)% (above every one|below every one|the highest|"
                 r"the lowest|higher than at (\d+)%) of its past (\d+) month-ends Five-year range (-?\d+\.\d+)% – "
                 r"(-?\d+\.\d+)% median (-?\d+\.\d+)% \(dashed line\)\. ")


def _pc(printed: str) -> tuple[float, float]:
    """A cPlain printing as a fraction, and half its last printed digit."""
    return float(printed) / 100, 0.5 * 10 ** -len(printed.split(".")[1]) / 100


def _check_vh_row(txt: str, y: dict) -> list[str]:
    g = _VH.match(txt)
    if not g:
        return [f"unreadable: {txt[:200]!r}"]
    out = []
    _label, now, words, above, n, lo, hi, med = g.groups()
    past = [v / 10000 for v in y["s"] if v is not None]
    # the series is rounded to basis points and `now` to five decimals, so a month-end within
    # 0.55bp of today's yield cannot be called lower or higher from the published figures (the
    # 2026-11-02 regenerated page has a month-end at 1007bp against a yield of 0.10065)
    bp = 0.55 / 10000 + 1e-12
    if int(n) != len(past) or int(n) != y["n"]:
        out.append(f"printed {n} month-ends; the series has {len(past)}, the record says {y['n']}")
    for name, got, want in (("now", now, y["now"]), ("low", lo, min(past)), ("high", hi, max(past)),
                            ("median", med, float(np.median(past)))):
        v, half = _pc(got)
        if abs(v - want) > half + bp + 1e-9:
            out.append(f"{name} printed {got}%, the series gives {want:.5f}")
    if y["now"] > y["hi"]:
        want_words = "above every one"
    elif y["now"] < y["lo"]:
        want_words = "below every one"
    elif y["now"] == y["hi"]:
        want_words = "the highest"
    elif y["now"] == y["lo"]:
        want_words = "the lowest"
    else:
        sure = sum(1 for v in past if v < y["now"] - bp)
        maybe = sum(1 for v in past if v < y["now"] + bp)
        # the record publishes the share to one decimal and the page rounds that, so a share of
        # 53.45% prints 54 (double rounding - half a point at most; noted in the audit, not fixed)
        ok = {_jsround(f) for k in range(sure, maybe + 1)
              for f in (100 * k / len(past), _jsround(1000 * k / len(past)) / 10, round(100 * k / len(past), 1))}
        if above is None or int(above) not in ok:
            out.append(f"printed 'higher than at {above}%', the series gives {sorted(ok)}")
        want_words = words if above is not None else "higher than at"
    if not words.startswith(want_words.split(" at")[0]) and words != want_words:
        out.append(f"printed '{words}', the record's now/lo/hi give '{want_words}'")
    return out


def test_every_valuation_history_row_prints_its_own_series(page):
    """For every stock with a five-year record: today's yield, the share of past month-ends with a
    lower yield (recounted here from the published monthly series, to its basis-point rounding),
    the number of month-ends, and the range and median of that same series."""
    ctx = _payload("dashboard_context.js")["ctx"]
    vhs = {t: c["vh"] for t, c in ctx.items() if c.get("vh") and (c["vh"].get("ey") or c["vh"].get("fy"))}
    assert len(vhs) >= 300, f"only {len(vhs)} stocks carry a five-year record"
    printed = page.evaluate(_VH_JS, vhs)
    problems, rows = [], 0
    for t, vh in vhs.items():
        ys = [vh[k] for k in ("ey", "fy") if vh.get(k)]
        if len(printed[t]) != len(ys):
            problems.append(f"{t}: {len(printed[t])} rows printed for {len(ys)} yields")
            continue
        for y, h in zip(ys, printed[t]):
            rows += 1
            problems += [f"{t}: {p}" for p in _check_vh_row(_text(h), y)]
    assert not problems, f"{len(problems)} problems in {rows} rows:\n" + "\n".join(problems[:30])


def _vh_fixture(now, n=48):
    s = [100 + 10 * i for i in range(n)]                      # 1.00% .. 5.70%, rising
    s[3] = None                                               # a month-end with no figure
    vals = [v / 10000 for v in s if v is not None]
    return {"m0": "2022-01-01", "ey": {"now": now, "pct": round(100 * sum(v < now for v in vals) / len(vals), 1),
                                       "lo": min(vals), "med": float(np.median(vals)), "hi": max(vals),
                                       "n": len(vals), "s": s}}


@pytest.mark.parametrize("now,words", [
    (0.03, "higher than at 40%"),        # 19 of the 47 month-ends (1.0%..2.9%, less the missing 1.3%) - 40.4
    (0.057, "the highest"), (0.06, "above every one"), (0.01, "the lowest"), (0.005, "below every one"),
])
def test_a_valuation_history_row_worked_by_hand(page, now, words):
    """47 month-ends from 1.0% to 5.7% in 0.1-point steps with one missing: each branch of the
    sentence, the count of month-ends with data, and the range and median (3.4%) of the series."""
    txt = _text(page.evaluate(_VH_JS, {"X": _vh_fixture(now)})["X"][0])
    assert f"{words} of its past 47 month-ends Five-year range 1.0% – 5.7% median 3.4% (dashed line)." in txt, txt


# ---------------------------------------------------------------------------
# dashboard.score_key / dashboard.trap_flag_key / overview.momentum_regime - the KPI row, the
# score key's median line, the analytics headline and the not-in-the-run message
# ---------------------------------------------------------------------------
_KPI_JS = """(w) => { const keep = D.weights;
    try { if (w) D.weights = Object.assign({}, keep, w); renderKPIs();
          return {kpi: document.getElementById('kpi-row').innerHTML,
                  median: (document.getElementById('sk-median') || {}).textContent || null,
                  analytics: (document.getElementById('meta-analytics') || {}).textContent || null,
                  absent: notInRunText('QZQZ')}; }
    finally { D.weights = keep; renderKPIs(); if (typeof fillTrackKpi === 'function') fillTrackKpi(); } }"""


def _card(kpi_html: str, label: str) -> str | None:
    i = kpi_html.find(f'<div class="kpi-label">{label}</div>')
    return None if i < 0 else _text(kpi_html[i:kpi_html.find('<div class="kpi-card"', i + 1) if
                                    kpi_html.find('<div class="kpi-card"', i + 1) > 0 else None])


def test_the_kpi_row_and_the_score_key_print_the_runs_own_counts(page, data):
    """The trap card's counts and shares of the scored universe (counted here from ``table_data``'s
    flags), the score key's median and universe (the median of the published composites), the
    analytics headline's sector and rate (each sector's flagged share, recounted), and the size the
    not-in-the-run message gives the ranking."""
    import statistics
    rows = data["table_data"]
    n = len(rows)
    assert n >= 400, f"only {n} ranked stocks"
    vt = sum(1 for r in rows if r.get("Value_Trap_Flag"))
    gt = sum(1 for r in rows if r.get("Growth_Trap_Flag"))
    got = page.evaluate(_KPI_JS, None)
    w = data["weights"]
    moved = bool(w.get("factor_weights_adjusted")) and w["factor_weights"].get("momentum") != w["base_factor_weights"].get("momentum")
    if not moved:
        card = _card(got["kpi"], "Trap flags")
        assert card == (f"Trap flags {vt} / {gt} value / growth · {_jsround(vt / n * 100)}% and "
                        f"{_jsround(gt / n * 100)}% of stocks What a flag means"), card
    med = statistics.median(r["Composite"] for r in rows)
    g = re.fullmatch(r" The median this run is (\d+\.\d), so a composite above it is in the upper half of the (\d+) stocks\.",
                     got["median"] or "")
    assert g, got["median"]
    assert abs(float(g.group(1)) - med) <= 0.05 + 0.005 and int(g.group(2)) == n, (got["median"], med, n)
    tot, flg = {}, {}
    for r in rows:
        tot[r["Sector"]] = tot.get(r["Sector"], 0) + 1
        flg[r["Sector"]] = flg.get(r["Sector"], 0) + bool(r.get("Value_Trap_Flag"))
    rate = {s: 100 * flg[s] / tot[s] for s in tot}
    g = re.fullmatch(r"Most value‑trap flags: (.+), (\d+\.\d)% of the sector", got["analytics"] or "")
    assert g, got["analytics"]
    top = max(rate.values())
    assert g.group(1) in rate and abs(rate[g.group(1)] - top) <= 0.05 + 1e-9, (g.groups(), top)
    assert abs(float(g.group(2)) - top) <= 0.05 + 1e-9, (g.groups(), top)
    assert _text(got["absent"]).startswith(f"The ranking covers the {n} S&P 500 companies this run scored."), got["absent"]


def test_the_momentum_weight_card_prints_a_hand_worked_cut(page):
    """The KPI card the regime rule's panic state puts in place of the trap card: 13% cut to 9.1%."""
    got = page.evaluate(_KPI_JS, {"factor_weights_adjusted": True,
                                  "factor_weights": {"valuation": 23.95, "quality": 23.95, "growth": 13, "momentum": 9.1,
                                                     "risk": 10, "revisions": 10, "size": 5, "investment": 5},
                                  "base_factor_weights": {"valuation": 22, "quality": 22, "growth": 13, "momentum": 13,
                                                          "risk": 10, "revisions": 10, "size": 5, "investment": 5}})
    assert _card(got["kpi"], "Momentum weight") == ("Momentum weight 9.1% cut from 13% by the momentum regime rule this run"), got["kpi"]
    assert _card(got["kpi"], "Trap flags") is None


# ---------------------------------------------------------------------------
# context.track_record - the panel's three remaining counts (claims audit queue item 3, 2026-11-03):
# the holding periods, the changelog ticks, and the picks left out of each basket
# ---------------------------------------------------------------------------
_BASKET = {"top": "top-25", "q1": "top-fifth", "q5": "bottom-fifth"}


def _unpriced_clause(T: dict) -> str | None:
    """The 'they are N of the M picks in the X baskets' clause, from the record's own fields."""
    picks, unp = T.get("picks") or {}, T.get("unpriced") or {}
    parts = [f"{len(unp[k])} of the {picks[k]} picks in the {_BASKET[k]} baskets"
             for k in ("top", "q1", "q5") if (unp.get(k) or []) and picks.get(k)]
    return "; they are " + " and ".join(parts) + ". " if parts else None


def test_the_track_panel_prints_its_own_period_tick_and_left_out_counts(page, track):
    txt = _panel(page)
    assert f"Every holding period ({len(track['periods'])})" in txt, txt[:200]
    marks = track.get("methodology_changes") or []
    assert f"The methodology changelog has entries on {len(marks)} dates inside this window" in txt
    # the count is of dates, as the sentence says: distinct, inside the record's own window
    dates = [m["date"] for m in marks]
    assert len(set(dates)) == len(dates) and all(track["start"] <= d <= track["end"] for d in dates)
    clause = _unpriced_clause(track)
    if clause:
        assert clause.strip() in txt, (clause, txt[-600:])


def test_the_track_panel_prints_hand_worked_counts(page, track):
    """Literal counts: 3 periods, 2 changelog dates, and 1 of 75 / 2 of 300 picks left out - with
    one name in two baskets listed once, and an empty basket left out of the clause."""
    T = _fixture_track(track, periods=track["periods"][:3],
                       methodology_changes=[{"date": track["start"], "title": "a"}, {"date": track["end"], "title": "b"}],
                       unpriced={"top": ["AAA"], "q1": ["AAA", "BBB"], "q5": []},
                       picks={"top": 75, "q1": 300, "q5": 300})
    txt = _panel(page, T)
    assert "Every holding period (3)" in txt
    assert "The methodology changelog has entries on 2 dates inside this window" in txt
    assert "(AAA, BBB)" in txt
    assert "; they are 1 of the 75 picks in the top-25 baskets and 2 of the 300 picks in the top-fifth baskets. " in txt, txt[-700:]
    assert "bottom-fifth baskets" not in txt


def test_the_changelog_ticks_are_one_per_dated_heading_date(tmp_path):
    """track_record.methodology_dates: several entries on one day are one tick; a heading with no
    date, or a date inside the text, is not one."""
    import track_record
    p = tmp_path / "CHANGELOG.md"
    p.write_text("# Log\n## 2026-10-01 - first\ntext 2026-10-05 in prose\n## 2026-10-01 (evening) - second\n"
                 "## Entry format\n## 2026-09-30 - earlier\n", encoding="utf-8")
    got = track_record.methodology_dates(p)
    assert [m["date"] for m in got] == ["2026-09-30", "2026-10-01"]
    assert got[1]["title"] == "first"


# ---------------------------------------------------------------------------
# drilldown.percentile_convention - the note that says how many metrics are "lower is better"
# (claims audit queue item 6, 2026-11-03)
# ---------------------------------------------------------------------------
def test_the_convention_note_prints_the_scorers_direction_count_and_true_examples(page, data):
    """The count is the payload's, the payload's directions are the scorer's (METRIC_DIR False is
    lower-is-better), and the three metrics the sentence names as examples really are inverted."""
    mm = data["metric_meta"]
    for k, m in mm.items():
        if k in METRIC_DIR:
            assert (m.get("dir") == "lower") == (METRIC_DIR[k] is False), k
    n = sum(1 for m in mm.values() if m.get("dir") == "lower")
    txt = _text(page.evaluate("PCTILE_CONVENTION"))
    assert (f"For the {n} of {len(mm)} metrics marked \u2193 better (EV/EBITDA, Beta and PEG among them) "
            "a low raw value earns a high percentile.") in txt, txt
    for key in ("ev_ebitda", "beta", "peg_ratio"):
        assert mm[key]["dir"] == "lower" and METRIC_DIR[key] is False, key


# ---------------------------------------------------------------------------
# drilldown.composite_distribution - the rankings strip's own title and label (claims audit queue
# item 5, 2026-11-03): its company count, range and median, against the ranking table
# ---------------------------------------------------------------------------
def test_the_distribution_strip_prints_the_rankings_count_range_and_median(page, data):
    vals = [r["Composite"] for r in data["table_data"]
            if isinstance(r.get("Composite"), (int, float)) and math.isfinite(r["Composite"])]
    if len(vals) < 10:
        pytest.skip("MISSING DATA: fewer than 10 scored companies - the strip hides itself")
    got = page.evaluate("""() => { renderDist(); return [document.getElementById('dist-title').textContent,
        document.getElementById('dist-plot').getAttribute('aria-label')]; }""")
    med = data["kpis"]["median_composite"]
    assert got[0] == f"{len(vals)} companies, one dot each"
    want = (f"{len(vals)} companies by composite score, from {_fixed1(min(vals))} to {_fixed1(max(vals))}, "
            f"median {_fixed1(med)}.")
    assert got[1].startswith(want), (got[1], want)
    # the published median is the median of the ranking it describes (to the table's rounding)
    assert abs(float(np.median(vals)) - med) <= 0.051, (float(np.median(vals)), med)


def _fixed1(x: float) -> str:
    """toFixed(1): the exact binary value, ties to the larger magnitude."""
    from decimal import ROUND_HALF_UP, Decimal
    q = Decimal(abs(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return ("-" if x < 0 and q != 0 else "") + str(q)


def test_the_distribution_strip_counts_only_scored_companies(page, data):
    """A company with no composite is in the table but not on the strip: a branch today's fully
    scored ranking may not take, so it is built by hand."""
    vals = [r["Composite"] for r in data["table_data"]
            if isinstance(r.get("Composite"), (int, float)) and math.isfinite(r["Composite"])]
    got = page.evaluate("""() => { const keep = D.table_data;
        try { D.table_data = keep.concat([{Ticker: 'ZZZ', Composite: null, Rank: null}]); renderDist();
              return document.getElementById('dist-title').textContent; }
        finally { D.table_data = keep; renderDist(); } }""")
    assert got == f"{len(vals)} companies, one dot each"


# ---------------------------------------------------------------------------
# drilldown.metric_weights - every workings row's weight lines and points equation (claims audit
# queue item 2, 2026-11-03): the share after rescaling, the points and the printed lines, worked
# out here from the published weight table and percentiles for every stock, category and metric
# ---------------------------------------------------------------------------
_WK_JS = """(cats) => { const out = [];
    for (const t of Object.keys(D.stock_detail)) { const s = D.stock_detail[t];
        renderCategoryDetails(t, s, cats);
        for (const c of cats) { const wk = categoryWorkings(c, s); if (!wk) continue;
            // the weight line as the page rendered it: the title on each row's weight cell
            const shown = {};
            document.querySelectorAll('#cat-detail-' + c + ' tr.metric-row').forEach(tr => {
                const cell = tr.querySelector('.metric-weight');
                shown[tr.dataset.metric] = cell ? cell.getAttribute('title') : null; });
            for (const r of wk.rows) {
                let pts = null;
                if (r.pct !== null && (D.lineage || {})[r.metric]) {
                    const h = metricDetailHtml(r.metric, s, r);
                    const k = h.indexOf('<span class="wk-k">Points</span>');
                    pts = k < 0 ? '' : h.slice(k, h.indexOf('</div></div>', k));
                }
                out.push([t, c, r.metric, wk.wp.pid, r.metric in shown ? shown[r.metric] : '(no row rendered)', pts]);
            } } }
    return out; }"""


def _fixed_n(x: float, n: int) -> str:
    """Number.prototype.toFixed(n): the exact binary value, ties away from zero."""
    from decimal import ROUND_HALF_UP, Decimal
    q = Decimal(abs(x)).quantize(Decimal(1).scaleb(-n), rounding=ROUND_HALF_UP)
    return ("-" if x < 0 and q != 0 else "") + str(q)


def _fmt_weight(w: float) -> str:
    return (str(_jsround(w)) if abs(w - _jsround(w)) < 0.05 else _fixed_n(w, 1)) + "%"


def test_every_workings_row_prints_the_weight_and_points_worked_out_from_the_published_tables(page, data):
    """For each row: the configured weight is the published table's (the stock's own profile), the
    share is that weight over the total of the weighted metrics with data, the points are percentile
    x share, and the printed lines say exactly that. Each scored category's points add up to its
    published score (the build's own guarantee, calc_trace)."""
    profiles = data["weights"]["profiles"]
    sd = data["stock_detail"]
    got = page.evaluate(_WK_JS, CATS)
    assert got, "no workings rows rendered"
    by: dict[tuple[str, str], list] = {}
    for t, c, m, pid, tip, pts in got:
        by.setdefault((t, c), []).append((m, pid, tip, pts))
    problems, n_rows, n_eq = [], 0, 0
    for (t, c), rows in by.items():
        s = sd[t]
        pid = (s.get("wp") or {}).get(c) or "generic"
        table = profiles[c].get(pid) or profiles[c]["generic"]
        total = sum(w for mm, w in table.items() if w and w > 0 and s["pct"].get(mm) is not None)
        cat_points = 0.0
        for m, page_pid, tip, pts in rows:
            n_rows += 1
            w = table[m]
            p = s["pct"].get(m)
            if page_pid != pid:
                problems.append(f"{t}/{c}: page used profile {page_pid}, the payload says {pid}")
            if p is None:
                want = f"Configured {_fmt_weight(w)}; no data for this stock, so it carries no weight here."
            else:
                share = w / total * 100
                points = p * share / 100
                cat_points += points
                want = (f"Configured {_fmt_weight(w)} of this weighting; {_fmt_weight(share)} of this score "
                        "after rescaling to the metrics that have data.")
                if pts is not None:
                    n_eq += 1
                    eq = (f"Points {_fixed_n(p, 2)} percentile × {_fmt_weight(share)} weight = "
                          f"{_fixed_n(points, 2)} points toward the category score")
                    if _text(pts) != eq:
                        problems.append(f"{t}/{c}/{m}: printed {_text(pts)!r}, worked out {eq!r}")
            if tip != want:
                problems.append(f"{t}/{c}/{m}: printed {tip!r}, worked out {want!r}")
        sc = (s.get("cat_scores") or {}).get(c)
        if sc is not None and abs(cat_points - sc) > 0.06:
            problems.append(f"{t}/{c}: rows add up to {cat_points:.3f}, published score {sc}")
    assert n_rows > 1000 and n_eq > 1000, (n_rows, n_eq)
    assert not problems, f"{len(problems)} of {n_rows} rows: {problems[:6]}"


def test_a_hand_worked_workings_row(page, data):
    """Three weighted metrics configured 50 / 30 / 20, the 30 with no data: the others rescale to
    71.4% and 28.6% of the score; a percentile of 80.0 at 71.4% is 57.14 points."""
    t = next(iter(data["stock_detail"]))
    got = page.evaluate("""(t) => { const keep = D.weights.profiles.valuation;
        const s = JSON.parse(JSON.stringify(D.stock_detail[t]));
        try { D.weights.profiles.valuation = { generic: { ev_ebitda: 50, fcf_yield: 30, earnings_yield: 20 } };
              s.wp = Object.assign({}, s.wp || {}, { valuation: 'generic' });
              s.pct = Object.assign({}, s.pct, { ev_ebitda: 80.0, fcf_yield: null, earnings_yield: 40.0 });
              const wk = categoryWorkings('valuation', s);
              const r = wk.rows.find(x => x.metric === 'ev_ebitda');
              const h = metricDetailHtml('ev_ebitda', s, r);
              const k = h.indexOf('<span class="wk-k">Points</span>');
              return [wk.rows.map(x => [x.metric, x.share, x.points]), h.slice(k, h.indexOf('</div></div>', k))]; }
        finally { D.weights.profiles.valuation = keep; } }""", t)
    rows = {m: (sh, pt) for m, sh, pt in got[0]}
    assert rows["fcf_yield"] == (None, None)
    assert rows["ev_ebitda"][0] == pytest.approx(500 / 7) and rows["ev_ebitda"][1] == pytest.approx(80 * 50 / 70)
    assert rows["earnings_yield"][0] == pytest.approx(200 / 7)
    assert _text(got[1]) == "Points 80.00 percentile × 71.4% weight = 57.14 points toward the category score"


# ---------------------------------------------------------------------------
# drilldown.row_equation - the workings' summary lines for the scores that are sums or statistics
# of parts (claims audit queue item 2, 2026-11-03): each line's arithmetic redone here from the
# inputs the payload publishes beside it (``pio``, ``eq4``, ``roe5``), and checked to give the
# value that was scored
# ---------------------------------------------------------------------------
_SURPRISES = re.compile(r"Latest surprise ([+−]?)([\d.]+)% − the quarter before ([+−]?)([\d.]+)% = ")
_SUM_METRICS = ["piotroski_f_score", "analyst_surprise", "earnings_acceleration", "consecutive_beat_streak",
                "earnings_variability"]
_SUMS_JS = """(ms) => { const out = {};
    for (const t of Object.keys(D.stock_detail)) { const s = D.stock_detail[t]; out[t] = {};
        for (const m of ms) {
            if (s.raw[m] == null) continue;
            const meta = D.metric_meta[m] || { fmt: 'ratio' };
            const h = metricDetailHtml(m, s, null);
            const sums = [...h.matchAll(/<div class="wk-sum">([\\s\\S]*?)<\\/div>/g)].map(x => x[1]);
            const head = (h.match(/The last (\\d+) quarters, oldest first/) || [null, null])[1];
            const ok = (h.match(/Rebuilt from the inputs above: the same ([\\s\\S]*?) that was scored/) || [null, null])[1];
            out[t][m] = { sums: sums, head: head, ok: ok, eq: equationHtml(m, s, meta), raw: fmtMetric(s.raw[m], meta.fmt) };
        } }
    return out; }"""


def _ints(txt: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", txt)]


def test_every_sum_of_parts_line_redoes_its_own_arithmetic(page, data):
    """Piotroski: passed and testable counted from the nine published signals, the score equal to the
    passes (the engine does not rescale). Surprises: the median of the valid quarters, the latest minus
    the one before, and the recency-weighted beat share x 10, each from the published quarters and
    equal to the scored value. ROE: the sample standard deviation of the five published ROEs. And the
    quarter count is the number of published quarters."""
    sd = data["stock_detail"]
    got = page.evaluate(_SUMS_JS, _SUM_METRICS)
    problems, checked = [], {m: 0 for m in _SUM_METRICS}
    for t, per in got.items():
        s = sd[t]
        for m, g in per.items():
            raw = s["raw"][m]
            sums = [_text(x) for x in g["sums"]]
            if m == "piotroski_f_score" and s.get("pio"):
                sig = s["pio"]
                n, k = sig.count("1"), sum(c != "-" for c in sig)
                want = f"{n} passed of {k} testable = score {g['raw']}"
                if want not in sums or raw != n or k < 6:
                    problems.append(f"{t}/{m}: {sums} for signals {sig}, raw {raw}")
                if _text(g["eq"]) != f"{n} of {k} testable signals passed = {_text(g['raw'])}":
                    problems.append(f"{t}/{m}: row line {_text(g['eq'])!r}")
                checked[m] += 1
            q = s.get("eq4") or []
            if m in ("analyst_surprise", "earnings_acceleration", "consecutive_beat_streak") and q:
                if g["head"] is None or int(g["head"]) != len(q):
                    problems.append(f"{t}/{m}: quarter count {g['head']} for {len(q)} published")
                vals = [r[3] for r in q if r[3] is not None]
                if m == "analyst_surprise":
                    ok = abs(float(np.median(vals)) - raw) < 1e-9
                    line = next((x for x in sums if x.startswith("Median of the")), "")
                    ok &= line.startswith(f"Median of the {len(vals)} valid surprises (") and line.endswith(f"= {_text(g['raw'])}")
                elif m == "earnings_acceleration":
                    ok = q[-1][3] is not None and q[-2][3] is not None and abs((q[-1][3] - q[-2][3]) - raw) < 1e-9
                    line = next((x for x in sums if x.startswith("Latest surprise")), "")
                    ok &= line.endswith(f"= {_text(g['raw'])}")
                    # each printed surprise is its own quarter's: the newest first, then the one before
                    mm_ = _SURPRISES.match(line)
                    if ok and mm_:
                        for (sign, mag), want_v in zip((mm_.group(1, 2), mm_.group(3, 4)), (q[-1][3], q[-2][3])):
                            v = float(mag) * (-1 if sign == "−" else 1)
                            step = 0.5 * 10 ** -(len(mag.split(".")[1]) if "." in mag else 0)
                            ok &= abs(v - 100 * want_v) <= step + 1e-9
                    ok &= mm_ is not None
                else:
                    L = len(q)
                    pos = [(4 - (L - 1 - i), r[3]) for i, r in enumerate(q) if r[3] is not None]
                    beat, every = [w for w, v in pos if v > 0], [w for w, _ in pos]
                    ok = abs(10 * sum(beat) / sum(every) - raw) < 1e-9
                    line = next((x for x in sums if "with data" in x), "")
                    head, _, tail = line.partition(" of ")
                    ok &= (_ints(head.split(" (")[0]) == beat if beat else head == "No quarter with data beat its estimate")
                    ok &= _ints(tail.split("with data")[0]) == every and line.endswith(f"= {_text(g['raw'])}")
                if not ok:
                    problems.append(f"{t}/{m}: {sums} from quarters {q}, raw {raw}")
                checked[m] += 1
            if m == "earnings_variability" and s.get("roe5"):
                roes = [r[3] for r in s["roe5"]]
                ok = None not in roes and len(roes) == 5 and abs(float(np.std(roes, ddof=1)) - raw) < 1e-9
                ok &= f"Sample standard deviation of the five ROEs = {_text(g['raw'])}" in sums
                if not ok:
                    problems.append(f"{t}/{m}: {sums} from ROEs {roes}, raw {raw}")
                checked[m] += 1
            if g["ok"] is not None and _text(g["ok"]) != _text(g["raw"]):
                problems.append(f"{t}/{m}: 'the same {_text(g['ok'])} that was scored' against {_text(g['raw'])}")
    assert all(n >= 100 for n in checked.values()), checked
    assert not problems, f"{len(problems)}: {problems[:5]}"


# ---------------------------------------------------------------------------
# history.run_overview / the rank history - the history and count lines (claims audit queue item 5,
# 2026-11-03 night): each count recomputed here from the published history and earnings dates
# ---------------------------------------------------------------------------
_HIST_JS = """() => { const out = {};
    for (const t of Object.keys(H.series || {})) { renderStockHistory(t);
        const head = document.querySelector('#modal-history .rank-chart-head span');
        const note = document.querySelector('#modal-history .modal-note');
        out[t] = head && document.getElementById('section-history').style.display !== 'none'
            ? [head.textContent, note ? note.textContent : null] : null; }
    return out; }"""
_HIST_HEAD = re.compile(r"^Rank (\d+) on (.+?) → (\d+) on (.+?), across (\d+) comparable runs$")


def test_every_rank_history_line_is_the_stocks_own_series(page, data):
    """For every stock with two or more ranked runs: the first and last ranks and dates are its own
    series', the run count is the number of runs that ranked it, the closing rank is today's
    published rank when it was ranked today, and 'Based on N comparable runs' counts the dates."""
    H = data.get("history") or {}
    if not (H.get("available") and H.get("series")):
        pytest.skip("MISSING DATA: the payload carries no comparable history")
    rank_now = {r["Ticker"]: r.get("Rank") for r in data["table_data"]}
    got = page.evaluate(_HIST_JS)
    pretty = dict(zip(H["dates"], page.evaluate("ds => ds.map(prettyDate)", H["dates"])))
    problems, shown = [], 0
    for t, ser in H["series"].items():
        pts = [(i, r) for i, r in enumerate(ser["r"]) if r is not None]
        if len(pts) < 2:
            if got.get(t) is not None:
                problems.append(f"{t}: a history line with fewer than two ranked runs")
            continue
        shown += 1
        if got.get(t) is None:
            problems.append(f"{t}: no history line")
            continue
        m = _HIST_HEAD.match(re.sub(r"\s+", " ", got[t][0]).strip())
        (i0, r0), (i1, r1) = pts[0], pts[-1]
        want = (str(r0), pretty[H["dates"][i0]], str(r1), pretty[H["dates"][i1]], str(len(pts)))
        if not m or m.groups() != want:
            problems.append(f"{t}: printed {got[t][0]!r}, its series says {want}")
        if i1 == len(H["dates"]) - 1 and rank_now.get(t) is not None and r1 != rank_now[t]:
            problems.append(f"{t}: history ends at rank {r1}, today's table says {rank_now[t]}")
        if got[t][1] != f"Based on {len(H['dates'])} comparable runs.":
            problems.append(f"{t}: {got[t][1]!r}")
    assert shown > 400, shown
    assert not problems, f"{len(problems)}: {problems[:5]}"


def test_the_whats_changed_caption_and_footnote_count_their_own_runs(page, data):
    """'Built from N comparable runs, first to last' counts the published dates; the noise band's
    'N paired runs' is the number of consecutive dates no more than history.MAX_NOISE_PAIR_GAP_DAYS
    apart, recounted here; its observation count is printed as published and is at least what the
    published series alone can supply (departed stocks add more)."""
    import history
    H = data.get("history") or {}
    if not H.get("available"):
        pytest.skip("MISSING DATA: the payload carries no comparable history")
    got = page.evaluate("() => { renderChanged(); return [document.getElementById('changed-caption').textContent, "
                        "document.getElementById('changed-footnote').textContent]; }")
    d = H["dates"]
    first, last = page.evaluate("ds => ds.map(prettyDate)", [d[0], d[-1]])
    assert f"Built from {len(d)} comparable runs, {first} to {last}." in got[1]
    n = H["noise"]
    if n.get("source") != "measured":
        assert "a default used until there are enough paired runs" in got[0]
        return
    from datetime import date as _date
    gaps = [(_date.fromisoformat(b) - _date.fromisoformat(a)).days for a, b in zip(d, d[1:])]
    pairs = sum(g <= history.MAX_NOISE_PAIR_GAP_DAYS for g in gaps)
    assert n["n_pairs"] == pairs, (n["n_pairs"], pairs)
    lower = 0
    for i, g in enumerate(gaps):
        if g <= history.MAX_NOISE_PAIR_GAP_DAYS:
            lower += sum(1 for s in H["series"].values() if s["r"][i] is not None and s["r"][i + 1] is not None)
    assert n["n_observations"] >= lower, (n["n_observations"], lower)
    obs = f"{n['n_observations']:,}"
    assert (f"past ±{n['material_threshold']} ranks, the 95th percentile of ordinary run-to-run variation "
            f"measured across {pairs} paired runs ({obs} observations).") in got[0], got[0]


def test_the_reporting_soon_counts_are_the_earnings_dates_within_7_and_14_days(page, data):
    """Counted here from each ranked stock's published earnings date against the run's own date."""
    from datetime import date as _date
    run = _date.fromisoformat(data["kpis"]["run_timestamp"][:10])
    days = []
    for r in data["table_data"]:
        e = (data["stock_detail"].get(r["Ticker"]) or {}).get("earn") or {}
        if e.get("d"):
            days.append((_date.fromisoformat(e["d"]) - run).days)
    a, b = sum(0 <= x <= 7 for x in days), sum(0 <= x <= 14 for x in days)
    got = page.evaluate("() => { reportingMeta(); return document.getElementById('meta-reporting').textContent; }")
    assert got == f"{a} report within 7 days of this run · {b} within 14"


def test_a_hand_worked_rank_history_that_stopped_before_today(page, data):
    """A stock last ranked two runs ago: the closing rank and date are that run's, not today's."""
    H = data.get("history") or {}
    if not (H.get("available") and len(H.get("dates") or []) >= 4):
        pytest.skip("MISSING DATA: fewer than four comparable runs")
    n = len(H["dates"])
    r = [None] * n
    r[0], r[1], r[n - 3] = 40, 35, 12
    got = page.evaluate("""(r) => { H.series.ZZZ = { r: r, c: r.map(() => null) };
        try { renderStockHistory('ZZZ'); return document.querySelector('#modal-history .rank-chart-head span').textContent; }
        finally { delete H.series.ZZZ; } }""", r)
    d0, dl = page.evaluate("ds => ds.map(prettyDate)", [H["dates"][0], H["dates"][n - 3]])
    assert re.sub(r"\s+", " ", got).strip() == f"Rank 40 on {d0} \u2192 12 on {dl}, across 3 comparable runs"


def test_the_reporting_soon_window_edges_by_hand(page, data):
    """Reports on day 7 count in both windows, day 8 only in the 14-day one, day 15 in neither, and
    yesterday's in neither - set on three stocks by hand and counted against the unchanged rest."""
    from datetime import date as _date, timedelta
    run = _date.fromisoformat(data["kpis"]["run_timestamp"][:10])
    picks = [r["Ticker"] for r in data["table_data"][:4]]
    when = {picks[0]: run + timedelta(days=7), picks[1]: run + timedelta(days=8),
            picks[2]: run + timedelta(days=15), picks[3]: run - timedelta(days=1)}
    days = []
    for r in data["table_data"]:
        e = (data["stock_detail"].get(r["Ticker"]) or {}).get("earn") or {}
        d = when.get(r["Ticker"]) or (_date.fromisoformat(e["d"]) if e.get("d") else None)
        if d:
            days.append((d - run).days)
    a, b = sum(0 <= x <= 7 for x in days), sum(0 <= x <= 14 for x in days)
    got = page.evaluate("""(w) => { const keep = {};
        for (const [t, d] of Object.entries(w)) { const s = D.stock_detail[t]; keep[t] = s.earn; s.earn = Object.assign({}, s.earn || {}, { d: d }); }
        try { reportingMeta(); return document.getElementById('meta-reporting').textContent; }
        finally { for (const t of Object.keys(keep)) D.stock_detail[t].earn = keep[t]; reportingMeta(); } }""",
                        {t: d.isoformat() for t, d in when.items()})
    assert got == f"{a} report within 7 days of this run \u00b7 {b} within 14"


# ---------------------------------------------------------------------------
# Single-stock figures (claims audit queue item 6, 2026-11-03 night): the price-target strip, the
# share-class price gap and the compare sentence, each recomputed from the payload
# ---------------------------------------------------------------------------
_PT_JS = """() => { const out = {};
    for (const t of Object.keys(D.stock_detail)) { const s = D.stock_detail[t];
        if (!s.pt_mean && !s.pt_high && !s.pt_low) continue;
        renderPriceTargets(s);
        out[t] = [...document.querySelectorAll('#modal-price-targets .pt-card')].map(c =>
            [c.querySelector('.pt-card-label').textContent, c.querySelector('.pt-card-value').textContent]); }
    return out; }"""
_PT_VAL = re.compile(r"^\$(\d+\.\d{2})(?: ([+-]?)(\d+(?:\.\d+)?)%)?$")


def test_every_price_target_strip_prints_its_targets_and_their_distance_from_the_price(page, data):
    """Each card's dollars are the payload's to the cent; each percentage is (target - price) /
    price to its printed precision with the right sign; and the average target's distance is the
    price-target upside the engine scored, where it scored one - one figure on two surfaces."""
    sd = data["stock_detail"]
    got = page.evaluate(_PT_JS)
    assert len(got) > 400, len(got)
    problems = []
    for t, cards in got.items():
        s = sd[t]
        p = s["price"]
        by = dict(cards)
        if by.get("Current Price") != f"${_fixed_n(p, 2)}":
            problems.append(f"{t}: price {by.get('Current Price')!r} for {p}")
        for label, key in (("Avg Target", "pt_mean"), ("Low Target", "pt_low"), ("High Target", "pt_high")):
            v = s.get(key)
            if not v:
                continue
            m = _PT_VAL.match(re.sub(r"\s+", " ", by.get(label, "")).strip())
            if not m or m.group(1) != _fixed_n(v, 2) or m.group(3) is None:
                problems.append(f"{t}/{label}: {by.get(label)!r} for {v}")
                continue
            printed = float(m.group(3)) * (-1 if m.group(2) == "-" else 1)
            step = 0.5 * 10 ** -(len(m.group(3).split(".")[1]) if "." in m.group(3) else 0)
            true = (v - p) / p * 100
            if abs(printed - true) > step + 1e-9 or (m.group(2) == "" and printed != 0):
                problems.append(f"{t}/{label}: {printed}% printed, {true:.4f}% true")
            if key == "pt_mean":
                u = (s.get("raw") or {}).get("price_target_upside")
                if u is not None and abs(100 * u - true) > 0.05:
                    problems.append(f"{t}: the card's upside {true:.3f}% is not the scored {100 * u:.3f}%")
    assert not problems, f"{len(problems)}: {problems[:6]}"


def test_every_share_class_line_states_the_price_gap_and_rank_of_the_line_scored(page, data):
    """The second line's gap is its published price against the scored line's, and the rank is the
    scored line's rank in today's table."""
    sc = data.get("share_classes") or {}
    if not sc:
        pytest.skip("MISSING DATA: the payload names no second share class")
    sd = data["stock_detail"]
    rank = {r["Ticker"]: r.get("Rank") for r in data["table_data"]}
    for t, e in sc.items():
        base = sd[e["scored_as"]]["price"]
        assert e["price_gap_pct"] == pytest.approx(100 * (e["price"] / base - 1), abs=0.006), t
        assert e["rank"] == rank[e["scored_as"]], t
        txt = _text(page.evaluate("([t, sc]) => shareClassText(t, sc)", [t, e]))
        mag = page.evaluate("v => pctNearZero(v, 1).mag", e["price_gap_pct"])
        side = "below" if e["price_gap_pct"] < 0 else "above"
        assert f"It trades {mag}% {side} {e['scored_as']} for the same claim on the same earnings." in txt, txt
        assert f"ranked #{e['rank']} here" in txt, txt


_GAP = re.compile(r"^(\S+) scores (\d+\.\d) composite points (above|below) (\S+)(?:\. The largest differences: (.+))?\.$")


def test_the_compare_sentence_is_the_composite_gap_and_its_two_largest_category_differences(page, data):
    """For pairs across the ranking: the gap is the difference of the published composites, and the
    two categories named are the two largest differences in published category points (at least
    0.05), each with its own signed difference."""
    labels = page.evaluate("() => CAT_LABELS")
    cats = page.evaluate("() => UX_CATS")
    rows = sorted((r for r in data["table_data"] if r.get("Rank")), key=lambda r: r["Rank"])
    tick = [r["Ticker"] for r in rows]
    pairs = [(tick[i], tick[i + 1]) for i in range(0, 60)] + [(tick[i], tick[-1 - i]) for i in range(0, 40)]
    got = page.evaluate("ps => ps.map(([a, b]) => gapSentence(a, b))", pairs)
    sd = data["stock_detail"]
    problems = []
    for (a, b), h in zip(pairs, got):
        # inline tags only here (<strong>), so they are dropped without adding a space
        lede = htmllib.unescape(re.sub(r"<[^>]+>", "", re.search(r'<p class="gap-lede">(.*?)</p>', h).group(1))).strip()
        m = _GAP.match(lede)
        gap = sd[a]["composite"] - sd[b]["composite"]
        parts = sorted(((c, (sd[a]["contrib"].get(c) or 0) - (sd[b]["contrib"].get(c) or 0)) for c in cats),
                       key=lambda x: -abs(x[1]))
        lead = [f"{labels[c]} ({'+' if v >= 0 else chr(0x2212)}{_fixed_n(abs(v), 1)})" for c, v in parts if abs(v) >= 0.05][:2]
        want = (a, _fixed_n(abs(gap), 1), "above" if gap >= 0 else "below", b, " and ".join(lead) if lead else None)
        if not m or m.groups() != want:
            problems.append(f"{a} vs {b}: printed {lede!r}, worked out {want}")
    assert not problems, f"{len(problems)} of {len(pairs)}: {problems[:4]}"


# ---------------------------------------------------------------------------
# My Holdings' rank strip (claims audit queue item 5, 2026-11-03 night): the saved names' ranks,
# lowest first, "of N" with N the number ranked, and the axis's two ends
# ---------------------------------------------------------------------------
def test_the_holdings_strip_prints_the_saved_names_ranks_of_the_number_ranked(page, data):
    rows = sorted((r for r in data["table_data"] if r.get("Rank")), key=lambda r: r["Rank"])
    n = data["kpis"]["universe_size"]
    # the premise "of N" rests on: N is the number ranked, and the ranks run 1 to N
    assert n == len(rows) == rows[-1]["Rank"] and rows[0]["Rank"] == 1, (n, len(rows), rows[-1]["Rank"])
    saved = [rows[299]["Ticker"], rows[9]["Ticker"], "ZZZZ-NOT-RANKED", rows[0]["Ticker"]]
    got = page.evaluate("""(saved) => { const keep = holdings.slice();
        try { holdings = saved; renderTileViz();
              const v = document.querySelector('#sec-holdings .tile-viz');
              return [v.getAttribute('aria-label'), v.querySelector('.tv-cap').textContent,
                      [...v.querySelectorAll('.tv-strip i')].map(i => i.style.left)]; }
        finally { holdings = keep; renderTileViz(); } }""", saved)
    assert got[0] == f"Your saved names rank #1, #10, #300 of {n}"
    assert got[1] == f"#1Where they rank#{n}"
    # positions on the 1-to-N scale (the browser normalises "60.0%" to "60%", so compare numbers)
    want_left = sorted(float(_fixed_n((r - 1) / (n - 1) * 100, 1)) for r in (300, 10, 1))
    assert sorted(float(x.rstrip("%")) for x in got[2]) == want_left, got[2]


# ---------------------------------------------------------------------------
# The phone table's "Show the next N" (claims audit queue item 5, 2026-11-03 night): stepping
# through the whole ranking, each button states how many rows it will add and the count it reaches
# ---------------------------------------------------------------------------
def test_the_phone_table_states_each_next_page_and_reaches_every_row(browser, data, tmp_path):
    for f in ("dashboard_data.js", "dashboard_context.js"):
        shutil.copy(ROOT / f, tmp_path / f)
    (tmp_path / "index.html").write_text(gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t",
                                                          data_version="v"), encoding="utf-8")
    pg = browser.new_page(viewport={"width": 375, "height": 812})
    try:
        pg.goto((tmp_path / "index.html").as_uri(), wait_until="load", timeout=60000)
        pg.wait_for_function("typeof updateTableMore === 'function' && typeof D !== 'undefined' && "
                             "!document.getElementById('table-more').hidden", timeout=60000)
        n = len([r for r in data["table_data"]])
        step = pg.evaluate("PHONE_STEP")
        k, seen = step, 0
        while k < n:
            got = pg.evaluate("""() => { const b = document.getElementById('table-more');
                return [b.querySelector('.tm-count').textContent, b.querySelector('.tm-next').textContent,
                        b.querySelector('.tm-all').hidden]; }""")
            nxt = min(step, n - k)
            assert got[0] == f"Showing {k} of {n}" and got[1] == f"Show the next {nxt}", (k, got)
            assert got[2] is (nxt >= n - k), (k, got)
            pg.evaluate("showMoreRows(false)")
            k += nxt
            seen += 1
        assert pg.evaluate("document.getElementById('table-more').hidden") is True
        assert pg.evaluate("shownRows()") == n and seen == -(-(n - step) // step)
    finally:
        pg.close()
