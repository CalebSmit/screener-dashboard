"""The Factor Lab: a paper's result, rerun on public data, with the gap to the paper shown.

``lab.html`` lets a reader change the conditions of a published experiment and recalculates the
result in the browser. That is a page computing its own numbers, which the dashboard is
forbidden to do (CLAUDE.md row 0.8), so the lab carries its own version of that rule and these
tests are it:

1. **Python is the reference.** ``factor_lab.py`` defines every calculation; ``lab_engine.js``
   must reproduce it on the published data, at the paper's settings and away from them.
2. **The comparison with the paper is computed, never typed.** The verdicts published in
   ``lab_data.js`` are rebuilt here from the published series and must match.
3. **Every experiment shows its four parts** - the claim, the conditions, the difference from
   the reference, and what is still unverified - and the tolerances live in the build.
4. **It decides nothing in the screener.** No scoring module imports it.

No test here touches the network; ``factor_lab.fetch`` is never called.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import factor_lab as fl  # noqa: E402
from stock_summary import advice_terms_in  # noqa: E402

NODE = shutil.which("node")
PAGE = ROOT / "lab.html"
ENGINE = ROOT / "lab_engine.js"


@pytest.fixture(scope="module")
def published() -> dict:
    return fl.load_published()


# --------------------------------------------------------------------------- the arithmetic
def test_stats_are_the_papers_columns():
    x = [2.0, -1.0, 3.0, 0.0, -4.0, 6.0]
    s = fl.stats(x)
    mean = sum(x) / 6
    sd = math.sqrt(sum((v - mean) ** 2 for v in x) / 5)
    assert s["mean"] == pytest.approx(12 * mean) and s["std"] == pytest.approx(math.sqrt(12) * sd)
    assert s["sharpe"] == pytest.approx(12 * mean / (math.sqrt(12) * sd))
    assert (s["max"], s["min"], s["n"]) == (6.0, -4.0, 6)
    # a symmetric series has no skew; a two-point distribution has excess kurtosis of -2
    assert fl.stats([-1.0, 1.0, -1.0, 1.0])["skew"] == pytest.approx(0)
    assert fl.stats([-1.0, 1.0, -1.0, 1.0])["kurt"] == pytest.approx(-2)
    assert fl.stats([1.0, 2.0]) is None and fl.stats([3.0, 3.0, 3.0]) is None


def test_the_risk_forecast_uses_only_days_before_the_month():
    dates = ["19270103", "19270104", "19270105", "19270201", "19270202", "19270301"]
    daily = [1.0, -1.0, 2.0, 50.0, 3.0, None]            # percent; a wild day inside February
    got = fl.vol_forecasts(dates, daily, ["192701", "192702", "192703"], window=3)
    assert got[0] is None                                 # no history before January
    # February's forecast is January's three days: the 50% day is not known yet
    assert got[1] == pytest.approx(math.sqrt(252 * (0.01 ** 2 + 0.01 ** 2 + 0.02 ** 2) / 3))
    # March's is the last three days before March, which now includes it
    assert got[2] == pytest.approx(math.sqrt(252 * (0.02 ** 2 + 0.5 ** 2 + 0.03 ** 2) / 3))


def test_exposure_is_target_over_forecast_and_the_limit_binds():
    data = {"months": ["200001", "200002", "200003", "200004"], "wml": [10.0, -20.0, 5.0, 4.0],
            "vol": {"126": [None, 0.24, 0.06, 0.12]}}
    r = fl.run(data, "200001", "200012", window=126, target=0.12)
    assert r["months"] == ["200002", "200003", "200004"]          # no forecast, no month - in either series
    assert r["weights"] == pytest.approx([0.5, 2.0, 1.0])
    assert r["managed"] == pytest.approx([-10.0, 10.0, 4.0]) and r["plain"] == [-20.0, 5.0, 4.0]
    capped = fl.run(data, "200001", "200012", window=126, target=0.12, cap=1.0)
    assert capped["weights"] == pytest.approx([0.5, 1.0, 1.0])
    assert fl.run(data, "200003", "200003")["plain_stats"] is None   # one month measures nothing


def test_a_library_file_is_read_by_its_table_title(tmp_path):
    body = ("This file was created using the 202608 CRSP database.\n\n"
            "  Value Weight Returns -- Monthly\n,Lo PRIOR,PRIOR 2,Hi PRIOR\n"
            "192701,  -3.32,  1.00,  -0.24\n192702,   7.39,  1.00,   7.04\n192703, -99.99,  1.00,   5.50\n\n"
            "  Average Equal Weighted Returns -- Monthly\n,Lo PRIOR,PRIOR 2,Hi PRIOR\n192701, 9, 9, 9\n")
    z = tmp_path / "f.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("f.csv", body)
    dates, rows, vintage = fl.read_table(z, "Value Weight Returns -- Monthly")
    assert dates == ["192701", "192702", "192703"] and rows[0] == [-3.32, 1.0, -0.24]
    assert "202608 CRSP" in vintage
    wml = fl.winners_minus_losers(rows)
    assert wml[0] == pytest.approx(3.08) and wml[2] is None       # a missing leg is not a return of 105%
    with pytest.raises(ValueError):
        fl.read_table(z, "No Such Table")


# --------------------------------------------------------------------------- what is published
def test_the_published_series_is_whole(published):
    m = published["momentum"]
    months = m["months"]
    assert len(months) == len(m["wml"]) > 1000
    for a, b in zip(months, months[1:]):                          # consecutive calendar months
        ya, ma, yb, mb = int(a[:4]), int(a[4:]), int(b[:4]), int(b[4:])
        assert (yb, mb) == ((ya, ma + 1) if ma < 12 else (ya + 1, 1)), (a, b)
    assert sorted(m["vol"]) == sorted(str(w) for w in fl.VOL_WINDOWS)
    for w, series in m["vol"].items():
        assert len(series) == len(months)
        known = [v for v in series if v is not None]
        assert len(known) > len(months) - 14 and all(0.01 < v < 3 for v in known), w
    assert all(-95 < v < 60 for v in m["wml"])
    att = published["attribution"]
    assert "Kenneth R. French" in att["text"] and "Fama" in att["text"] and att["url"].startswith("https://")


def test_the_comparison_with_the_paper_is_computed_not_typed(published):
    for built, exp in zip(published["experiments"], fl.EXPERIMENTS):
        assert built["reference"] == json.loads(json.dumps(exp["reference"]))   # the paper's figures, as recorded
        again = fl.reproduce(published["momentum"], exp)
        assert (again["within"], again["total"]) == (built["result"]["within"], built["result"]["total"])
        for a, b in zip(again["rows"], built["result"]["rows"]):
            assert (a["leg"], a["stat"], a["within"]) == (b["leg"], b["stat"], b["within"])
            assert a["here"] == pytest.approx(b["here"]) and a["diff"] == pytest.approx(b["here"] - b["paper"])
            assert b["within"] == (abs(b["diff"]) <= b["tolerance"])
        assert built["result"]["total"] == 14


def test_every_experiment_shows_its_four_parts():
    for exp in fl.EXPERIMENTS:
        assert len(exp["claim"]) > 80 and exp["paper"]["doi"] and exp["paper"]["year"]
        assert len(exp["conditions"]) >= 4 and all(len(c) == 2 and c[1] for c in exp["conditions"])
        assert len(exp["open_details"]) >= 1 and exp["reference"]["source"]
        assert set(exp["reference"]["plain"]) == set(exp["reference"]["managed"]) == set(fl.TOLERANCES)
        assert exp["in_the_screener"]


def test_the_tolerances_are_the_recorded_ones():
    """Widening one to turn an 'outside' into a 'within' is the lab's version of loosening a
    bar to pass a build. Change the reasoning in factor_lab.py first, then this."""
    assert fl.TOLERANCES == {"max": 3.0, "min": 3.0, "mean": 1.0, "std": 1.0, "kurt": 1.0,
                             "skew": 0.15, "sharpe": 0.05}


# --------------------------------------------------------------------------- the browser's copy
SETTINGS = [
    {"start": "192703", "end": "201112", "window": 126, "target": 0.12, "cap": None},   # the paper's
    {"start": "201201", "end": "209912", "window": 126, "target": 0.12, "cap": None},   # after it
    {"start": "192901", "end": "193912", "window": 21, "target": 0.20, "cap": 1.0},
    {"start": "200001", "end": "201012", "window": 252, "target": 0.06, "cap": 1.5},
    {"start": "192701", "end": "209912", "window": 63, "target": 0.12, "cap": 2.0},
]


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_the_browser_engine_reproduces_the_reference(published, tmp_path):
    script = tmp_path / "run.js"
    script.write_text(
        "const E = require(process.argv[2]);\n"
        "const text = require('fs').readFileSync(process.argv[3], 'utf8');\n"
        "const D = JSON.parse(text.slice(text.indexOf('{'), text.lastIndexOf('}') + 1));\n"
        "const out = JSON.parse(process.argv[4]).map(o => { const r = E.run(D.momentum, o);\n"
        "  return { n: r.months.length, first: r.months[0], plain: r.plain_stats, managed: r.managed_stats,\n"
        "           weights: r.weight_stats, grow: E.growth(r.managed).pop(), fall: E.deepestFall(r.plain) }; });\n"
        "const X = D.experiments[0];\n"
        "const rows = E.compare(X.reference, E.run(D.momentum, {start: X.reference.start, end: X.reference.end,\n"
        "  window: X.reference.window, target: X.reference.target, cap: X.reference.cap}), D.tolerances, D.stat_labels);\n"
        "console.log(JSON.stringify({ out, rows }));\n", encoding="utf-8")
    res = subprocess.run([NODE, str(script), str(ENGINE), str(fl.OUT), json.dumps(SETTINGS)],
                         capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    got = json.loads(res.stdout)
    for o, js in zip(SETTINGS, got["out"]):
        py = fl.run(published["momentum"], **o)
        assert js["n"] == len(py["months"]) > 100 and js["first"] == py["months"][0]
        for leg in ("plain", "managed"):
            for k, v in py[leg + "_stats"].items():
                assert js[leg][k] == pytest.approx(v, rel=1e-9, abs=1e-9), (o, leg, k)
        for k, v in py["weight_stats"].items():
            assert js["weights"][k] == pytest.approx(v, rel=1e-9)
        assert js["grow"] == pytest.approx(fl.growth(py["managed"])[-1], rel=1e-9)
        assert js["fall"] == pytest.approx(fl.deepest_fall(py["plain"]), rel=1e-9)
    for js, built in zip(got["rows"], published["experiments"][0]["result"]["rows"]):
        assert js["within"] == built["within"] and js["here"] == pytest.approx(built["here"], rel=1e-9)


def test_the_page_shows_the_four_parts_and_runs_only_local_code():
    html = PAGE.read_text(encoding="utf-8")
    for part in ("The claim being tested", "The conditions used", "Against the paper",
                 "What is still needed to call it verified", "Run it yourself"):
        assert part in html, part
    assert re.findall(r'<script src="([^"]+)"', html) == ["lab_data.js", "lab_engine.js"]
    assert 'name="viewport"' in html and "E.run(M, S)" in html
    # the page prints the code it ran and warns if its own numbers leave the build's
    assert "E.run.toString()" in html and "disagrees with the build" in html
    assert "never used to set the screener's own" in html


def test_the_lab_gives_no_advice():
    visible = re.sub(r"<style.*?</style>", " ", PAGE.read_text(encoding="utf-8"), flags=re.S)
    words = json.dumps(fl.EXPERIMENTS) + " " + re.sub(r"<[^>]+>", " ", visible)
    assert advice_terms_in(words) == []
    assert "Not investment advice" in visible


def test_the_methodology_points_at_the_experiment_it_cites():
    """Checked in the generator, not the committed page (CLAUDE.md rule 10)."""
    import yaml
    import run_screener
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    text = run_screener.build_screener_overview(cfg)
    assert "[Factor Lab](https://calebsmit.github.io/screener-dashboard/lab.html)" in text


# --------------------------------------------------------------------------- what it must not become
def test_no_scoring_module_reads_the_lab():
    """The lab reruns published factor portfolios. It is not evidence for the screener's own
    parameters and must not become an input to them."""
    for path in ROOT.glob("*.py"):
        if path.name in ("factor_lab.py",):
            continue
        src = path.read_text(encoding="utf-8", errors="replace")
        assert not re.search(r"^\s*(import factor_lab|from factor_lab)\b", src, flags=re.M), path.name
        assert "lab_data.js" not in src or path.name == "run_screener.py", path.name


def test_the_librarys_own_files_are_not_republished():
    assert "data/lab/" in (ROOT / ".gitignore").read_text(encoding="utf-8")
    git = shutil.which("git")
    if git is None:
        pytest.skip("git not installed")
    tracked = subprocess.run([git, "ls-files", "data/lab"], cwd=ROOT, capture_output=True, text=True).stdout
    assert tracked.strip() == ""


# --------------------------------------------------------------------------- in a real browser
def test_changing_a_setting_recalculates_the_result():
    sync_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        try:
            for width in (1440, 375):
                page = browser.new_page(viewport={"width": width, "height": 900})
                errors: list[str] = []
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.goto(PAGE.as_uri(), wait_until="load", timeout=60000)
                page.wait_for_function("window.LAB_STATE && window.LAB_STATE().result", timeout=30000)
                before = page.evaluate("window.LAB_STATE().result.managed_stats.sharpe")
                assert page.evaluate("document.getElementById('x-mismatch').hidden") is True
                assert page.evaluate("document.getElementById('c-reset').hidden") is True
                page.click("#c-preset button[data-i='1']")              # after the paper's sample
                after = page.evaluate("window.LAB_STATE()")
                assert after["settings"]["start"] == "201201" and after["result"]["managed_stats"]["sharpe"] != before
                assert "Jan 2012" in page.inner_text("#x-read")
                assert page.evaluate("document.getElementById('c-reset').hidden") is False
                page.click("#c-reset")
                assert page.evaluate("window.LAB_STATE().result.managed_stats.sharpe") == pytest.approx(before)
                assert page.evaluate("document.querySelectorAll('.chart svg path').length") == 3
                assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
                assert errors == []
                page.close()
        finally:
            browser.close()
