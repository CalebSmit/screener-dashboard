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
    mk = published["market"]
    assert len(mk["mktrf"]) == len(months)
    assert sorted(mk["vol"]) == sorted(str(w) for w in fl.MARKET_VOL_WINDOWS)
    assert sorted(mk["trail"]) == sorted(str(n) for n in fl.BEAR_LOOKBACKS)
    assert all(len(s) == len(months) for s in list(mk["vol"].values()) + list(mk["trail"].values()))
    assert mk["trail"]["24"][0] is not None and mk["vol"]["126"][0] is not None     # January 1927 has a reading
    fa = published["factors"]
    assert fa["months"][0] == "196307" and all(len(fa[k]) == len(fa["months"]) > 700 for k in fl.FACTOR_KEYS)
    for a, b in zip(fa["months"], fa["months"][1:]):
        ya, ma, yb, mb = int(a[:4]), int(a[4:]), int(b[:4]), int(b[4:])
        assert (yb, mb) == ((ya, ma + 1) if ma < 12 else (ya + 1, 1)), (a, b)
    att = published["attribution"]
    assert "Kenneth R. French" in att["text"] and "Fama" in att["text"] and att["url"].startswith("https://")


def test_the_comparison_with_the_paper_is_computed_not_typed(published):
    assert [x["id"] for x in published["experiments"]] == [x["id"] for x in fl.EXPERIMENTS]
    for built, exp in zip(published["experiments"], fl.EXPERIMENTS):
        assert built["reference"] == json.loads(json.dumps(exp["reference"]))   # the paper's figures, as recorded
        again = (fl.reproduce_crashes(published["momentum"], published["market"], exp)
                 if exp["kind"] == "crashes" else fl.reproduce_factors(published["factors"], exp)
                 if exp["kind"] == "factors" else fl.reproduce(published["momentum"], exp))
        assert (again["within"], again["total"]) == (built["result"]["within"], built["result"]["total"])
        for a, b in zip(again["rows"], built["result"]["rows"]):
            assert a["label"] == b["label"] and a["within"] == b["within"]
            assert a["here"] == pytest.approx(b["here"]) and a["diff"] == pytest.approx(b["here"] - b["paper"])
            assert b["within"] == (abs(b["diff"]) <= b["tolerance"])
    assert [x["result"]["total"] for x in published["experiments"]] == [14, 15, 21]
    crash = published["experiments"][1]["result"]
    again = fl.reproduce_crashes(published["momentum"], published["market"], fl.EXPERIMENTS[1])
    assert [(c["n"], c["mean"]) for c in again["cells"]] == [(c["n"], pytest.approx(c["mean"])) for c in crash["cells"]]
    assert [w["month"] for w in again["worst"]] == [w["month"] for w in crash["worst"]] and len(crash["worst"]) == 15


def test_every_experiment_shows_its_four_parts():
    for exp in fl.EXPERIMENTS:
        assert len(exp["claim"]) > 80 and exp["paper"]["doi"] and exp["paper"]["year"]
        assert len(exp["conditions"]) >= 4 and all(len(c) == 2 and c[1] for c in exp["conditions"])
        assert len(exp["open_details"]) >= 1 and exp["reference"]["source"]
        if exp["kind"] in ("crashes", "factors"):
            figs = exp["reference"]["figures"]
            assert len({f["key"] for f in figs}) == len(figs) >= 10
            assert all(f["tolerance"] > 0 and f["where"] and f["label"] for f in figs)
        else:
            assert set(exp["reference"]["plain"]) == set(exp["reference"]["managed"]) == set(fl.TOLERANCES)
        assert exp["in_the_screener"] and exp["topic"] and exp["kind"] in ("risk-managed", "crashes", "factors")


def test_the_tolerances_are_the_recorded_ones():
    """Widening one to turn an 'outside' into a 'within' is the lab's version of loosening a
    bar to pass a build. Change the reasoning in factor_lab.py first, then this."""
    assert fl.TOLERANCES == {"max": 3.0, "min": 3.0, "mean": 1.0, "std": 1.0, "kurt": 1.0,
                             "skew": 0.15, "sharpe": 0.05}
    # experiment 2's were written before its comparison was first run (plan/factor-lab.md rule 3)
    assert {f["key"]: (f["paper"], f["tolerance"]) for f in fl.EXPERIMENTS[1]["reference"]["figures"]} == {
        "wml_mean": (17.9, 1.0), "wml_std": (30.0, 1.0), "wml_sharpe": (0.60, 0.05),
        "mkt_mean": (7.7, 0.3), "mkt_std": (18.8, 0.3), "beta0": (-0.576, 0.05), "alpha0": (1.852, 0.10),
        "beta_bear": (-1.131, 0.15), "alpha_bear": (-2.04, 0.35), "bear_months": (183, 5),
        "gamma0": (1.955, 0.15), "gamma_bear": (-2.626, 0.40), "worst1": (-74.36, 3.0),
        "worst2": (-60.98, 3.0), "worst_overlap": (15, 3)}
    # experiment 3's too: one tolerance per kind of figure
    kinds = {"mean": 0.03, "std": 0.10, "t": 0.25, "corr": 0.05}
    for f in fl.EXPERIMENTS[2]["reference"]["figures"]:
        want = kinds.get(f["key"].split("_")[0])
        if want is None:
            want = {"hml_intercept": 0.05, "hml_on_cma": 0.08, "hml_r2": 0.05,
                    "rmw_intercept": 0.05, "cma_intercept": 0.05}[f["key"]]
        assert f["tolerance"] == want, f["key"]
    assert {f["key"]: f["paper"] for f in fl.EXPERIMENTS[2]["reference"]["figures"]} == {
        "mean_mktrf": 0.50, "mean_smb": 0.29, "mean_hml": 0.37, "mean_rmw": 0.25, "mean_cma": 0.33,
        "std_mktrf": 4.49, "std_smb": 3.07, "std_hml": 2.88, "std_rmw": 2.14, "std_cma": 2.01,
        "t_mktrf": 2.74, "t_smb": 2.31, "t_hml": 3.20, "t_rmw": 2.92, "t_cma": 4.07,
        "corr_hml_cma": 0.70, "hml_intercept": -0.04, "hml_on_cma": 1.04, "hml_r2": 0.51,
        "rmw_intercept": 0.43, "cma_intercept": 0.28}


def test_the_sentence_about_the_misses_is_true(published):
    """Experiment 2's verdict says every figure outside its tolerance describes the portfolio
    itself and every figure about what follows a market decline is within. If a rebuild of
    the library changes that, the sentence must change with it."""
    rows = {r["key"]: r["within"] for r in published["experiments"][1]["result"]["rows"]}
    after_a_decline = {"beta_bear", "alpha_bear", "bear_months", "gamma_bear"}
    the_portfolio = {"wml_mean", "wml_std", "wml_sharpe", "alpha0", "gamma0", "worst1", "worst2"}
    assert all(rows[k] for k in after_a_decline)
    assert {k for k, ok in rows.items() if not ok} <= the_portfolio
    assert "after a market decline is within" in fl.EXPERIMENTS[1]["verdict_note"]


def test_the_markets_state_is_known_before_the_month():
    dates = ["19270103", "19270104", "19270105", "19270106", "19270201", "19270202", "19270301"]
    daily = [1.0, -1.0, 1.0, -1.0, 40.0, 1.0, 0.0]                  # a wild day inside February
    got = fl.market_vol(dates, daily, ["192701", "192702", "192703"], window=4)
    assert got[0] is None
    m = 0.0                                                        # January's four days average zero
    assert got[1] == pytest.approx(math.sqrt(252 * (4 * 0.01 ** 2 - 4 * m * m) / 3))
    assert got[2] > 1.0                                            # March's reading has now seen it
    # the trailing return is over the months BEFORE the month, as many as exist (at least six)
    md = ["192607", "192608", "192609", "192610", "192611", "192612", "192701", "192702"]
    mr = [10.0, -10.0, 0.0, 0.0, 0.0, 0.0, -50.0, 0.0]
    tr = fl.trailing_return(md, mr, ["192612", "192701", "192702"], lookback=24)
    assert tr[0] is None                                           # only five months before December
    assert tr[1] == pytest.approx(1.1 * 0.9 - 1)                   # January's own -50% is not in it
    assert tr[2] == pytest.approx(1.1 * 0.9 * 0.5 - 1)


def test_months_are_sorted_into_four_states():
    mom = {"months": ["200001", "200002", "200003", "200004", "200005"], "wml": [2.0, -4.0, 1.0, -30.0, 6.0]}
    market = {"trail": {"24": [0.10, 0.10, -0.05, -0.05, None]},
              "vol": {"126": [0.10, 0.30, 0.10, 0.30, 0.30]}}
    r = fl.crash_run(mom, market, "200001", "200012", lookback=24, window=126, threshold=0.206)
    assert r["months"] == ["200001", "200002", "200003", "200004"] and r["state"] == [0, 1, 2, 3]
    assert [c["n"] for c in r["cells"]] == [1, 1, 1, 1]
    assert r["cells"][3]["mean"] == -30.0 and r["cells"][3]["losing"] == 1.0
    assert r["rest_mean"] == pytest.approx((2.0 - 4.0 + 1.0) / 3) and r["all_mean"] == pytest.approx(-31.0 / 4)
    # the threshold is strict, and a state nobody is in reports nothing rather than zero
    calm = fl.crash_run(mom, market, "200001", "200012", threshold=0.30)
    assert [c["n"] for c in calm["cells"]] == [2, 0, 2, 0] and calm["cells"][3]["mean"] is None


def test_a_factor_the_others_fully_explain_has_nothing_left_over():
    months = [f"2000{m:02d}" for m in range(1, 13)] + [f"2001{m:02d}" for m in range(1, 13)]
    cma = [0.5, -1.0, 2.0, 0.0, 1.5, -0.5, 1.0, -2.0, 0.5, 1.0, -1.5, 2.5] * 2
    rmw = [1.0, 0.5, -0.5, 2.0, -1.0, 0.0, 1.5, 0.5, -2.0, 1.0, 0.5, -0.5] * 2
    smb = [-1.0, 2.0, 0.5, -0.5, 1.0, 1.5, -2.0, 0.0, 1.0, -1.5, 0.5, 2.0] * 2
    mkt = [2.0, -3.0, 1.0, 4.0, -2.0, 0.5, 3.0, -1.0, -4.0, 2.5, 1.5, -0.5] * 2
    mkt[13] += 0.7                                                  # so no column repeats another
    hml = [0.25 + 1.5 * c - 0.5 * r for c, r in zip(cma, rmw)]      # value built from two of the others
    f = {"months": months, "mktrf": mkt, "smb": smb, "hml": hml, "rmw": rmw, "cma": cma}
    r = fl.factor_run(f, "200001", "200112")
    sp = r["spanned"]["hml"]
    assert sp["intercept"] == pytest.approx(0.25) and sp["r2"] == pytest.approx(1.0)
    assert sp["slopes"]["cma"] == pytest.approx(1.5) and sp["slopes"]["rmw"] == pytest.approx(-0.5)
    assert sp["slopes"]["smb"] == pytest.approx(0, abs=1e-9)
    s = r["stats"]["cma"]
    mean = sum(cma) / 24
    sd = math.sqrt(sum((v - mean) ** 2 for v in cma) / 23)
    assert s["mean"] == pytest.approx(mean) and s["t"] == pytest.approx(mean / (sd / math.sqrt(24)))
    assert r["corr"]["cma"]["cma"] == pytest.approx(1.0)
    # one year only: figures, but no regression on eleven months
    assert fl.factor_run(f, "200001", "200011")["spanned"] == {}
    assert fl.factor_run(f, "200001", "200003")["stats"]["hml"] is None


def test_least_squares_recovers_a_known_line():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    flag = [0.0, 1.0, 0.0, 1.0, 1.0]
    y = [1.5 + 2.0 * a - 3.0 * b for a, b in zip(x, flag)]
    got = fl.ols(y, [[1.0] * 5, x, flag])
    assert got == pytest.approx([1.5, 2.0, -3.0])


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


FACTOR_SETTINGS = [{"start": "196307", "end": "201312"}, {"start": "201401", "end": "209912"},
                   {"start": "200001", "end": "201012"}, {"start": "196307", "end": "209912"}]


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_the_browser_measures_the_factors_as_the_reference_does(published, tmp_path):
    script = tmp_path / "factors.js"
    script.write_text(
        "const E = require(process.argv[2]);\n"
        "const text = require('fs').readFileSync(process.argv[3], 'utf8');\n"
        "const D = JSON.parse(text.slice(text.indexOf('{'), text.lastIndexOf('}') + 1));\n"
        "console.log(JSON.stringify(JSON.parse(process.argv[4]).map(o => { const r = E.factorRun(D.factors, D.factor_keys, o);\n"
        "  return { n: r.months.length, stats: r.stats, corr: r.corr, spanned: r.spanned }; })));\n",
        encoding="utf-8")
    res = subprocess.run([NODE, str(script), str(ENGINE), str(fl.OUT), json.dumps(FACTOR_SETTINGS)],
                         capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    for o, js in zip(FACTOR_SETTINGS, json.loads(res.stdout)):
        py = fl.factor_run(published["factors"], **o)
        assert js["n"] == len(py["months"]) > 100
        for k in fl.FACTOR_KEYS:
            for stat in ("mean", "std", "t", "growth"):
                assert js["stats"][k][stat] == pytest.approx(py["stats"][k][stat], rel=1e-8), (o, k, stat)
            assert js["spanned"][k]["intercept"] == pytest.approx(py["spanned"][k]["intercept"], rel=1e-6, abs=1e-9)
            assert js["spanned"][k]["r2"] == pytest.approx(py["spanned"][k]["r2"], rel=1e-8)
            for other, slope in py["spanned"][k]["slopes"].items():
                assert js["spanned"][k]["slopes"][other] == pytest.approx(slope, rel=1e-6, abs=1e-9)
            for other in fl.FACTOR_KEYS:
                assert js["corr"][k][other] == pytest.approx(py["corr"][k][other], rel=1e-9)


CRASH_SETTINGS = [
    {"start": "192701", "end": "201303", "lookback": 24, "window": 126, "threshold": 0.206},   # the screener's rule
    {"start": "201304", "end": "209912", "lookback": 24, "window": 126, "threshold": 0.206},
    {"start": "192701", "end": "209912", "lookback": 12, "window": 63, "threshold": 0.15},
    {"start": "196307", "end": "209912", "lookback": 36, "window": 252, "threshold": 0.30},
]


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_the_browser_sorts_the_months_as_the_reference_does(published, tmp_path):
    script = tmp_path / "crash.js"
    script.write_text(
        "const E = require(process.argv[2]);\n"
        "const text = require('fs').readFileSync(process.argv[3], 'utf8');\n"
        "const D = JSON.parse(text.slice(text.indexOf('{'), text.lastIndexOf('}') + 1));\n"
        "console.log(JSON.stringify(JSON.parse(process.argv[4]).map(o => { const r = E.crashRun(D.momentum, D.market, o);\n"
        "  return { n: r.months.length, state: r.state.join(''), cells: r.cells, rest: r.rest_mean, all: r.all_mean }; })));\n",
        encoding="utf-8")
    res = subprocess.run([NODE, str(script), str(ENGINE), str(fl.OUT), json.dumps(CRASH_SETTINGS)],
                         capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    for o, js in zip(CRASH_SETTINGS, json.loads(res.stdout)):
        py = fl.crash_run(published["momentum"], published["market"], **o)
        assert js["n"] == len(py["months"]) > 100 and js["state"] == "".join(map(str, py["state"]))
        assert js["rest"] == pytest.approx(py["rest_mean"], rel=1e-9) and js["all"] == pytest.approx(py["all_mean"], rel=1e-9)
        for a, b in zip(js["cells"], py["cells"]):
            assert a["n"] == b["n"] and a["share"] == pytest.approx(b["share"])
            for k in ("mean", "worst", "losing"):
                assert a[k] == (None if b[k] is None else pytest.approx(b[k], rel=1e-9)), (o, k)


def test_the_page_shows_the_four_parts_and_runs_only_local_code():
    html = PAGE.read_text(encoding="utf-8")
    for part in ("The claim being tested", "The conditions used", "Against the paper",
                 "What is still needed to call it verified", "Run it yourself"):
        assert part in html, part
    v = f"?v={fl.ASSET_VERSION}"                                    # one version on both, so they travel together
    assert re.findall(r'<script src="([^"]+)"', html) == ["lab_data.js" + v, "lab_engine.js" + v]
    assert 'name="viewport"' in html and "E.run(M, S)" in html and "E.crashRun(M, MK, S)" in html
    assert "E.factorRun(F, KEYS, S)" in html
    # the page prints the code it ran and warns if its own numbers leave the build's
    assert "E.run.toString()" in html and "E.crashRun.toString()" in html
    assert html.count("disagrees with the build") == 3             # each experiment checks itself
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


def test_the_dashboard_has_a_way_into_the_lab():
    """In the generator (rule 10): the footer and the search palette, neither of which is a
    change to the closed header design."""
    import generate_dashboard as gd
    html = gd.generate_html()
    assert '<a href="lab.html">Factor Lab</a>' in html
    assert "label: 'Factor Lab'" in html and "window.location.href = 'lab.html'" in html


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
                # the second experiment, by its link
                page.goto(PAGE.as_uri() + "#momentum-crashes", wait_until="load", timeout=60000)
                page.wait_for_function("window.LAB_STATE && window.LAB_STATE().id === 'momentum-crashes'", timeout=30000)
                assert page.evaluate("document.getElementById('x-mismatch').hidden") is True
                at_rule = page.evaluate("window.LAB_STATE().result.cells[3].n")
                assert at_rule > 50 and page.evaluate("document.querySelectorAll('.chart .band').length") > 3
                page.evaluate("const el = document.getElementById('c-thr'); el.value = 30; el.dispatchEvent(new Event('input'))")
                assert page.evaluate("window.LAB_STATE().result.cells[3].n") < at_rule       # a stricter bar, fewer months
                assert page.evaluate("window.LAB_STATE().settings.threshold") == pytest.approx(0.30)
                page.click("#c-reset")
                assert page.evaluate("window.LAB_STATE().result.cells[3].n") == at_rule
                assert page.evaluate("document.querySelectorAll('.tabs a').length") == 3
                assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
                assert errors == []
                # the third
                page.goto(PAGE.as_uri() + "#five-factors", wait_until="load", timeout=60000)
                page.wait_for_function("window.LAB_STATE && window.LAB_STATE().id === 'five-factors'", timeout=30000)
                assert page.evaluate("document.getElementById('x-mismatch').hidden") is True
                assert "value factor" in page.inner_text("#x-read")
                page.click("#c-focus button[data-v='rmw']")
                assert "profitability factor" in page.inner_text("#x-read")
                assert page.evaluate("document.querySelectorAll('.chart svg path').length") == 5
                page.click("#c-preset button[data-i='1']")
                assert page.evaluate("window.LAB_STATE().result.months.length") < 200
                assert "short period for a factor" in page.inner_text("#x-live-note")
                assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
                assert errors == []
                page.close()
        finally:
            browser.close()
