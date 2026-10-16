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
    sz = published["size"]
    assert sz["months"][0] == "192607" and len(sz["ew"]) == len(sz["vw"]) == 10
    assert all(len(col) == len(sz["months"]) > 1100 for col in sz["ew"] + sz["vw"])
    att = published["attribution"]
    assert "Kenneth R. French" in att["text"] and "Fama" in att["text"] and att["url"].startswith("https://")


def test_the_comparison_with_the_paper_is_computed_not_typed(published):
    assert [x["id"] for x in published["experiments"]] == [x["id"] for x in fl.EXPERIMENTS]
    for built, exp in zip(published["experiments"], fl.EXPERIMENTS):
        assert built["reference"] == json.loads(json.dumps(exp["reference"]))   # the paper's figures, as recorded
        again = (fl.reproduce_crashes(published["momentum"], published["market"], exp)
                 if exp["kind"] == "crashes" else fl.reproduce_factors(published["factors"], exp)
                 if exp["kind"] == "factors" else fl.reproduce_size(published["size"], exp)
                 if exp["kind"] == "size" else fl.reproduce_sort(published["sorts"][exp["sort"]["data"]], exp)
                 if exp["kind"] == "sort"
                 else fl.reproduce_split(published[exp.get("split", {}).get("data", "split")], exp)
                 if exp["kind"] == "split"
                 else fl.reproduce_combo(published["split"], published["factors"], exp)
                 if exp["kind"] == "combo" else fl.reproduce(published["momentum"], exp))
        assert (again["within"], again["total"]) == (built["result"]["within"], built["result"]["total"])
        for a, b in zip(again["rows"], built["result"]["rows"]):
            assert a["label"] == b["label"] and a["within"] == b["within"]
            assert a["here"] == pytest.approx(b["here"]) and a["diff"] == pytest.approx(b["here"] - b["paper"])
            assert b["within"] == (abs(b["diff"]) <= b["tolerance"])
    assert [x["result"]["total"] for x in published["experiments"]] == [14, 15, 21, 11, 11, 11, 11, 18, 12, 6, 11, 11, 10]
    crash = published["experiments"][1]["result"]
    again = fl.reproduce_crashes(published["momentum"], published["market"], fl.EXPERIMENTS[1])
    assert [(c["n"], c["mean"]) for c in again["cells"]] == [(c["n"], pytest.approx(c["mean"])) for c in crash["cells"]]
    assert [w["month"] for w in again["worst"]] == [w["month"] for w in crash["worst"]] and len(crash["worst"]) == 15


def test_every_experiment_shows_its_four_parts():
    for exp in fl.EXPERIMENTS:
        assert len(exp["claim"]) > 80 and (exp["paper"].get("doi") or exp["paper"].get("url", "").startswith("https://")) and exp["paper"]["year"]
        assert len(exp["conditions"]) >= 4 and all(len(c) == 2 and c[1] for c in exp["conditions"])
        assert len(exp["open_details"]) >= 1 and exp["reference"]["source"]
        if exp["kind"] in ("crashes", "factors", "size", "sort", "split", "combo"):
            figs = exp["reference"]["figures"]
            # at least one figure per group a sort reports, plus the spread between its ends (a
            # decile sort gives 11, a quintile sort 6); every other kind reports ten or more
            least = len(exp["sort"]["groups"]) + 1 if exp["kind"] == "sort" else 10
            assert len({f["key"] for f in figs}) == len(figs) >= least
            assert all(f["tolerance"] > 0 and f["where"] and f["label"] for f in figs)
        else:
            assert set(exp["reference"]["plain"]) == set(exp["reference"]["managed"]) == set(fl.TOLERANCES)
        assert exp["in_the_screener"] and exp["topic"] and exp["kind"] in ("risk-managed", "crashes", "factors", "size", "sort", "split", "combo")
        if exp["kind"] == "sort":                                       # what the generic layout needs
            g = exp["sort"]
            assert g["data"] in fl.SORTS and len(g["groups"]) == len(fl.SORTS[g["data"]]["columns"])
            assert len(g["ends"]) == 2 and g["low"] and g["high"] and g["spread"] and g["key_accent"] and g["key_rest"]
            assert all(0 <= i < len(g["groups"]) for i in g["accent"]) and g["presets"] and g["weightings"]
            assert exp["reference"]["weighting"] in [w[0] for w in g["weightings"]]


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


def test_a_replication_that_mostly_fails_is_published_as_one(published):
    """Experiment 4 reruns a 1992 table on a different set of companies from the paper's and
    most figures miss. The tolerances were written first and stay; the page says how many
    missed and why; and the sentence that says which ones is checked here."""
    exp, built = fl.EXPERIMENTS[3], published["experiments"][3]["result"]
    assert {f["key"]: (f["paper"], f["tolerance"]) for f in exp["reference"]["figures"]} == {
        "dec1": (1.546, 0.05), "dec2": (1.29, 0.05), "dec3": (1.24, 0.05), "dec4": (1.25, 0.05),
        "dec5": (1.29, 0.05), "dec6": (1.17, 0.05), "dec7": (1.07, 0.05), "dec8": (1.10, 0.05),
        "dec9": (0.95, 0.05), "dec10": (0.890, 0.05), "small_big": (0.656, 0.08)}
    # the two derived figures are the paper's halves weighted by its own firm counts
    assert (1.64 * 772 + 1.16 * 189) / (772 + 189) == pytest.approx(1.546, abs=0.001)
    assert (0.88 * 60 + 0.90 * 64) / (60 + 64) == pytest.approx(0.890, abs=0.001)
    rows = {r["key"]: r for r in built["rows"]}
    assert built["within"] < built["total"] / 2                      # it is a failed replication, and says so
    for d in range(1, 7):                                            # "every group of smaller companies sits below"
        assert rows[f"dec{d}"]["here"] < rows[f"dec{d}"]["paper"] and not rows[f"dec{d}"]["within"]
    assert {k for k, r in rows.items() if r["within"]} <= {"dec7", "dec8", "dec9", "dec10"}
    assert "only matches are among the four largest groups" in exp["verdict_note"]


def test_the_volatility_sort_misses_where_it_was_expected_to(published):
    """Experiment 5's tolerances were written first, with the expectation that the most
    volatile fifth would miss. They are pinned here, and so is the sentence about the misses."""
    exp, built = fl.EXPERIMENTS[4], published["experiments"][4]["result"]
    assert {f["key"]: (f["paper"], f["tolerance"]) for f in exp["reference"]["figures"]} == {
        "mean1": (1.06, 0.10), "mean2": (1.15, 0.10), "mean3": (1.22, 0.10), "mean4": (0.99, 0.10),
        "mean5": (0.09, 0.10), "std1": (3.71, 0.40), "std2": (4.48, 0.40), "std3": (5.63, 0.40),
        "std4": (7.15, 0.40), "std5": (8.30, 0.40), "spread": (-0.97, 0.20)}
    rows = {r["key"]: r for r in built["rows"]}
    # "every miss is in the more volatile groups"
    assert {k for k, r in rows.items() if not r["within"]} <= {"mean4", "mean5", "std3", "std4", "std5", "spread"}
    # "its most volatile fifth ... earned far more than the paper's"
    assert rows["mean5"]["here"] > rows["mean5"]["paper"] + 0.3 and not rows["mean5"]["within"]


def test_the_earnings_yield_sort_matches_in_shape_and_misses_at_the_ends(published):
    exp, built = fl.EXPERIMENTS[5], published["experiments"][5]["result"]
    assert {f["key"]: (f["paper"], f["tolerance"]) for f in exp["reference"]["figures"]} == {
        "mean1": (0.984, 0.05), "mean2": (0.94, 0.05), "mean3": (1.03, 0.05), "mean4": (1.18, 0.05),
        "mean5": (1.22, 0.05), "mean6": (1.33, 0.05), "mean7": (1.42, 0.05), "mean8": (1.46, 0.05),
        "mean9": (1.57, 0.05), "mean10": (1.730, 0.05), "spread": (0.746, 0.08)}
    assert (1.04 * 88 + 0.93 * 90) / 178 == pytest.approx(0.984, abs=0.001)       # the halves, by firm count
    assert (1.74 * 95 + 1.72 * 91) / 186 == pytest.approx(1.730, abs=0.001)
    rows = {r["key"]: r for r in built["rows"]}
    # "the misses are small and mostly at the two ends ... the rise and the gap both match"
    missed = {k for k, r in rows.items() if not r["within"]}
    assert {"mean1", "mean10"} <= missed and len(missed & {"mean1", "mean2", "mean9", "mean10"}) >= len(missed) / 2
    assert all(abs(r["diff"]) < 0.15 for r in rows.values()) and rows["spread"]["within"]
    here = [rows[f"mean{d}"]["here"] for d in range(1, 11)]
    assert sum(b > a for a, b in zip(here, here[1:])) >= 8                         # rising, almost step by step


def test_the_beta_sort_is_flat_as_the_paper_found(published):
    exp, built = fl.EXPERIMENTS[6], published["experiments"][6]["result"]
    assert {f["key"]: (f["paper"], f["tolerance"]) for f in exp["reference"]["figures"]} == {
        "mean1": (1.20, 0.05), "mean2": (1.32, 0.05), "mean3": (1.26, 0.05), "mean4": (1.31, 0.05),
        "mean5": (1.30, 0.05), "mean6": (1.30, 0.05), "mean7": (1.23, 0.05), "mean8": (1.23, 0.05),
        "mean9": (1.33, 0.05), "mean10": (1.238, 0.05), "spread": (0.038, 0.08)}
    assert (1.34 * 165 + 1.18 * 291) / 456 == pytest.approx(1.238, abs=0.001)      # the halves, by firm count
    rows = {r["key"]: r for r in built["rows"]}
    assert rows["spread"]["within"]                                 # "the gap between the end groups ... matches"
    assert all(r["here"] < r["paper"] for r in rows.values() if not r["within"])   # "every miss sits below"


def test_a_sort_is_measured_group_by_group_and_end_to_end():
    months = [f"2000{m:02d}" for m in range(1, 13)]
    lo = [1.0, 2.0, 0.0, 1.5, 0.5, 1.0, 2.0, 0.0, 1.5, 0.5, 1.0, 1.0]
    hi = [4.0, -6.0, 5.0, -3.0, 2.0, 4.0, -6.0, 5.0, -3.0, 2.0, 0.0, 2.0]
    data = {"months": months, "vw": [lo, [0.0] * 12, hi], "ew": [hi, [0.0] * 12, lo]}
    r = fl.sort_run(data, "200001", "200012", "vw")
    assert r["groups"][0]["mean"] == pytest.approx(1.0) and r["groups"][2]["mean"] == pytest.approx(0.5)
    assert r["groups"][2]["std"] > r["groups"][0]["std"] * 4            # more risk, less return
    assert r["spread"]["mean"] == pytest.approx(-0.5)                   # last group minus first
    assert fl.sort_run(data, "200001", "200012", "ew")["spread"]["mean"] == pytest.approx(0.5)
    assert fl.sort_run(data, "200001", "200004", "vw")["groups"][0] is None


def test_the_size_groups_are_measured_as_differences_month_by_month():
    months = [f"2000{m:02d}" for m in range(1, 13)]
    base = [1.0, -2.0, 3.0, 0.5, -1.0, 2.0, 0.0, 1.5, -0.5, 2.5, -1.5, 1.0]
    size = {"months": months, "ew": [[v + (9 - d) * 0.1 + (0.3 if (d == 8 and i % 2) else 0.0) for i, v in enumerate(base)]
                                     for d in range(10)], "vw": [list(base) for _ in range(10)]}
    r = fl.size_run(size, "200001", "200012", "ew")
    assert r["deciles"][0]["mean"] == pytest.approx(sum(base) / 12 + 0.9)
    assert r["small_big"]["mean"] == pytest.approx(0.9)
    # the ninth group beats the tenth by 0.1 every month and by another 0.3 every other month
    assert r["nine_ten"]["mean"] == pytest.approx(0.25)
    d = [0.1 + (0.3 if i % 2 else 0.0) for i in range(12)]
    sd = math.sqrt(sum((v - 0.25) ** 2 for v in d) / 11)
    assert r["nine_ten"]["t"] == pytest.approx(0.25 / (sd / math.sqrt(12)))
    assert fl.size_run(size, "200001", "200012", "vw")["small_big"]["mean"] == pytest.approx(0)
    assert fl.size_run(size, "200001", "200003", "ew")["deciles"][0] is None


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


SIZE_SETTINGS = [{"start": "196307", "end": "199012", "weighting": "ew"}, {"start": "199101", "end": "209912", "weighting": "vw"},
                 {"start": "192607", "end": "209912", "weighting": "ew"}, {"start": "200001", "end": "209912", "weighting": "vw"}]


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_the_browser_measures_the_size_groups_as_the_reference_does(published, tmp_path):
    script = tmp_path / "size.js"
    script.write_text(
        "const E = require(process.argv[2]);\n"
        "const text = require('fs').readFileSync(process.argv[3], 'utf8');\n"
        "const D = JSON.parse(text.slice(text.indexOf('{'), text.lastIndexOf('}') + 1));\n"
        "console.log(JSON.stringify(JSON.parse(process.argv[4]).map(o => { const r = E.sizeRun(D.size, o);\n"
        "  return { n: r.months.length, deciles: r.deciles, small_big: r.small_big, nine_ten: r.nine_ten, eight_ten: r.eight_ten }; })));\n",
        encoding="utf-8")
    res = subprocess.run([NODE, str(script), str(ENGINE), str(fl.OUT), json.dumps(SIZE_SETTINGS)],
                         capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    for o, js in zip(SIZE_SETTINGS, json.loads(res.stdout)):
        py = fl.size_run(published["size"], **o)
        assert js["n"] == len(py["months"]) > 100
        for a, b in zip(js["deciles"] + [js["small_big"], js["nine_ten"], js["eight_ten"]],
                        py["deciles"] + [py["small_big"], py["nine_ten"], py["eight_ten"]]):
            for k in ("mean", "std", "t"):
                assert a[k] == pytest.approx(b[k], rel=1e-8, abs=1e-12), (o, k)


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_the_browser_measures_a_sort_as_the_reference_does(published, tmp_path):
    settings = [{"start": "196307", "end": "200012", "weighting": "vw"}, {"start": "200101", "end": "209912", "weighting": "ew"},
                {"start": "196307", "end": "209912", "weighting": "vw"}]
    script = tmp_path / "sort.js"
    script.write_text(
        "const E = require(process.argv[2]);\n"
        "const text = require('fs').readFileSync(process.argv[3], 'utf8');\n"
        "const D = JSON.parse(text.slice(text.indexOf('{'), text.lastIndexOf('}') + 1));\n"
        "const out = {};\n"
        "Object.keys(D.sorts).forEach(name => { out[name] = JSON.parse(process.argv[4]).map(o => { const r = E.sortRun(D.sorts[name], o);\n"
        "  return { n: r.months.length, groups: r.groups, spread: r.spread }; }); });\n"
        "console.log(JSON.stringify(out));\n", encoding="utf-8")
    res = subprocess.run([NODE, str(script), str(ENGINE), str(fl.OUT), json.dumps(settings)],
                         capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    got = json.loads(res.stdout)
    assert set(got) == set(fl.SORTS) == set(published["sorts"])
    for name, runs in got.items():
        for o, js in zip(settings, runs):
            py = fl.sort_run(published["sorts"][name], **o)
            assert js["n"] == len(py["months"]) > 100
            for a, b in zip(js["groups"] + [js["spread"]], py["groups"] + [py["spread"]]):
                for k in ("mean", "std", "t"):
                    assert a[k] == pytest.approx(b[k], rel=1e-8, abs=1e-12), (name, o, k)


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_the_browser_measures_the_halves_as_the_reference_does(published, tmp_path):
    script = tmp_path / "split.js"
    script.write_text(
        "const E = require(process.argv[2]);\n"
        "const text = require('fs').readFileSync(process.argv[3], 'utf8');\n"
        "const D = JSON.parse(text.slice(text.indexOf('{'), text.lastIndexOf('}') + 1));\n"
        "const one = data => JSON.parse(process.argv[4]).map(o => { const r = E.splitRun(data, o);\n"
        "  return { n: r.months.length, factors: r.factors }; });\n"
        "console.log(JSON.stringify({ split: one(D.split), quintiles: one(D.quintiles) }));\n", encoding="utf-8")
    res = subprocess.run([NODE, str(script), str(ENGINE), str(fl.OUT), json.dumps(FACTOR_SETTINGS)],
                         capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    got = json.loads(res.stdout)
    assert list(got["split"][0]["factors"]) == fl.SPLIT_KEYS == published["split_keys"]
    for name, keys in (("split", fl.SPLIT_KEYS), ("quintiles", fl.QUINT_KEYS)):
        for o, js in zip(FACTOR_SETTINGS, got[name]):
            py = fl.split_run(published[name], **o)
            assert js["n"] == len(py["months"]) > 100 and list(js["factors"]) == keys
            for k in keys:
                for part in ("small", "big", "gap", "long", "short"):
                    for stat in ("mean", "std", "t"):
                        assert js["factors"][k][part][stat] == pytest.approx(py["factors"][k][part][stat], rel=1e-8, abs=1e-12), (name, o, k, part)


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
    assert "E.factorRun(F, KEYS, S)" in html and "E.sizeRun(Z, S)" in html and "E.sortRun(Z, S)" in html
    assert "E.splitRun(Z, S)" in html and "E.comboRun(Z, F, S)" in html
    # the page prints the code it ran and warns if its own numbers leave the build's
    assert "E.run.toString()" in html and "E.crashRun.toString()" in html
    assert html.count("disagrees with the build") == 7             # each kind of experiment checks itself
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


# --------------------------------------------------------------------------- the front page
def test_each_takeaway_is_what_the_data_shows(published):
    """The front page gives one sentence per experiment. Each is checked here against the
    published series, so a rebuild that makes one untrue fails instead of leaving it up."""
    assert set(fl.TAKEAWAYS) == {x["id"] for x in fl.EXPERIMENTS}
    mom, fac, size = published["momentum"], published["factors"], published["size"]
    # 1. "roughly doubled its Sharpe ratio in the paper's sample ... after the sample, both earned less"
    inside, after = fl.run(mom, "192703", "201112"), fl.run(mom, "201201", "209912")
    assert 1.6 < inside["managed_stats"]["sharpe"] / inside["plain_stats"]["sharpe"] < 2.4
    for leg in ("plain_stats", "managed_stats"):
        assert after[leg]["sharpe"] < inside[leg]["sharpe"] and after[leg]["mean"] < inside[leg]["mean"]
    # 2. "that pattern reproduces closely; the levels of the paper's own portfolio do not"
    rows = {r["key"]: r["within"] for r in published["experiments"][1]["result"]["rows"]}
    assert rows["gamma_bear"] and rows["beta_bear"] and rows["bear_months"] and not rows["wml_mean"]
    cells = published["experiments"][1]["result"]["cells"]
    assert cells[3]["mean"] < 0 < cells[0]["mean"]
    # 3. "about a third of a percent a month for fifty years; since, none distinguishable from zero"
    first, since = fl.factor_run(fac, "196307", "201312"), fl.factor_run(fac, "201401", "209912")
    for k in ("smb", "hml", "rmw", "cma"):
        assert 0.2 < first["stats"][k]["mean"] < 0.45 and first["stats"][k]["t"] > 2
        assert abs(since["stats"][k]["t"]) < 2
    assert len(since["months"]) < 240                                   # "a period too short to settle it"
    # 4. "by less than the paper's table shows ... between the two largest groups the difference is small"
    rows = {r["key"]: r for r in published["experiments"][3]["result"]["rows"]}
    assert 0 < rows["small_big"]["here"] < rows["small_big"]["paper"]
    assert abs(published["experiments"][3]["result"]["nine_ten"]["mean"]) < 0.15
    # 5. "about as much as the rest, with half the volatility ... the paper's large penalty does not appear"
    r = fl.sort_run(published["sorts"]["low-volatility"], "196307", "200012", "vw")
    rest = sum(g["mean"] for g in r["groups"][1:]) / 4
    assert abs(r["groups"][0]["mean"] - rest) < 0.15
    assert 1.8 < r["groups"][4]["std"] / r["groups"][0]["std"] < 2.5
    assert r["spread"]["mean"] > -0.97 + 0.5 and abs(r["spread"]["t"]) < 2
    # 6. "in the paper's sample and for decades after; since 2007 all but closed; reversed among large companies"
    ey = published["sorts"]["earnings-yield"]
    assert fl.sort_run(ey, "196307", "199012", "ew")["spread"]["t"] > 2
    assert fl.sort_run(ey, "199101", "209912", "ew")["spread"]["t"] > 2
    late = fl.sort_run(ey, "200701", "209912", "ew")["spread"]
    assert abs(late["t"]) < 2 and late["mean"] < 0.25
    assert fl.sort_run(ey, "200701", "209912", "vw")["spread"]["mean"] < 0
    # 7. "flat in the paper's sample ... not distinguishable from zero since ... about twice as volatile"
    be = published["sorts"]["market-beta"]
    for a, b in (("196307", "199012"), ("199101", "209912"), ("196307", "209912")):
        r = fl.sort_run(be, a, b, "ew")
        assert abs(r["spread"]["t"]) < 2
        assert 1.8 < r["groups"][9]["std"] / r["groups"][0]["std"] < 2.5
    # 8. "each earned roughly half what they did among small ones in the paper's sample; since 2000
    #     only profitability has stayed distinguishable from zero there"
    sp = published["split"]
    paper = fl.split_run(sp, "196307", "201312")["factors"]
    for k in ("value", "profitability", "investment"):
        assert 0.4 < paper[k]["big"]["mean"] / paper[k]["small"]["mean"] < 0.7, k
    late = fl.split_run(sp, "200001", "209912")["factors"]
    assert late["profitability"]["big"]["t"] > 2
    assert all(abs(late[k]["big"]["t"]) < 2 for k in fl.SPLIT_KEYS if k != "profitability")
    # 9. "nearly 8% a year from 1927 to 2011, as the paper found; since 2000 under 3% a year and not
    #     distinguishable from zero; value there was never reliably different from zero"
    qd = published["quintiles"]
    rows = {r["key"]: r for r in published["experiments"][8]["result"]["rows"]}
    assert 7.5 <= rows["mean_momentum_big"]["here"] < 8.0 and rows["mean_momentum_big"]["within"]
    since = fl.split_run(qd, "200001", "209912")["factors"]
    assert since["momentum"]["big"]["mean"] * 12 < 3 and abs(since["momentum"]["big"]["t"]) < 2
    for a, b in (("192701", "201112"), ("192701", "209912"), ("200001", "209912")):
        assert abs(fl.split_run(qd, a, b)["factors"]["value"]["big"]["t"]) < 2, (a, b)


def test_the_factors_are_measured_within_each_half_of_the_market():
    """A factor among big companies is the big half's high group minus its low group - and for
    investment the low minus the high. The gap between the halves is taken month by month."""
    months = [f"2000{m:02d}" for m in range(1, 13)]
    split = {"months": months}
    for i, k in enumerate(fl.SPLIT_KEYS):
        split[k] = {"small": [1.0 + i + 0.1 * j for j in range(12)], "big": [0.5 * i - 0.05 * j for j in range(12)],
                    "long": [0.3 * i + 0.01 * j for j in range(12)], "short": [0.2 * i - 0.06 * j for j in range(12)]}
    r = fl.split_run(split, "200003", "200010")
    assert r["months"] == months[2:10]
    for i, k in enumerate(fl.SPLIT_KEYS):
        small, big = split[k]["small"][2:10], split[k]["big"][2:10]
        assert r["factors"][k]["small"]["mean"] == pytest.approx(sum(small) / 8)
        assert r["factors"][k]["big"]["mean"] == pytest.approx(sum(big) / 8)
        assert r["factors"][k]["gap"]["mean"] == pytest.approx(sum(a - b for a, b in zip(small, big)) / 8)
        assert r["factors"][k]["long"]["mean"] == pytest.approx(sum(split[k]["long"][2:10]) / 8)
        assert r["factors"][k]["short"]["mean"] == pytest.approx(sum(split[k]["short"][2:10]) / 8)
    assert fl.SPLIT_SIGN == {"value": 1, "profitability": 1, "investment": -1, "momentum": 1}
    assert set(fl.SPLIT_CATEGORY.values()) <= {c["category"] for c in fl.CATEGORY_MAP}


def test_the_large_company_experiment_is_recorded_as_it_landed(published):
    """Experiment 8's tolerances were written first, expecting nearly every figure within. Six
    missed. They stay as written, and the sentence that says where the misses are is checked."""
    exp, built = fl.EXPERIMENTS[7], published["experiments"][7]["result"]
    assert exp["id"] == "big-and-small" and exp["kind"] == "split"
    figs = {f["key"]: (f["paper"], f["tolerance"]) for f in exp["reference"]["figures"]}
    assert figs == {
        "mean_value_small": (0.53, 0.03), "t_value_small": (4.05, 0.25), "mean_value_big": (0.21, 0.03),
        "t_value_big": (1.69, 0.25), "mean_value_gap": (0.32, 0.03), "t_value_gap": (2.94, 0.25),
        "mean_profitability_small": (0.33, 0.03), "t_profitability_small": (3.06, 0.25),
        "mean_profitability_big": (0.17, 0.03), "t_profitability_big": (1.81, 0.25),
        "mean_profitability_gap": (0.16, 0.03), "t_profitability_gap": (1.48, 0.25),
        "mean_investment_small": (0.45, 0.03), "t_investment_small": (5.49, 0.25),
        "mean_investment_big": (0.22, 0.03), "t_investment_big": (2.00, 0.25),
        "mean_investment_gap": (0.23, 0.03), "t_investment_gap": (2.29, 0.25)}
    rows = {r["key"]: r for r in built["rows"]}
    missed = {k for k, r in rows.items() if not r["within"]}
    assert missed and all("value" in k or "profitability" in k for k in missed)      # "every miss is in value or profitability"
    for k in ("value", "profitability"):                                             # "a little more than the paper printed"
        assert 0 < rows[f"mean_{k}_big"]["diff"] < 0.06
    for k in ("value", "profitability", "investment"):                               # "almost exactly at two standard errors"
        assert 1.9 < rows[f"t_{k}_big"]["here"] < 2.1
    assert "Every miss is in value or profitability" in exp["verdict_note"]
    # momentum is shown but has no figure in the paper's table, so none in the comparison
    assert not any("momentum" in k for k in figs) and "momentum" in published["split"]


def test_a_big_company_factor_is_its_two_sides_added_together(published):
    """Held group minus the market, plus the market minus the shorted group, is the factor -
    every month. The front page's sentence about profitability rests on the split."""
    sp = published["split"]
    for k in fl.SPLIT_KEYS:
        assert set(sp[k]) == set(fl.SPLIT_PARTS)
        for big, held, shorted in zip(sp[k]["big"], sp[k]["long"], sp[k]["short"]):
            assert big == pytest.approx(held + shorted, abs=2e-4)                    # each is rounded to 4 places
    whole = fl.split_run(sp, "196307", "209912")["factors"]
    # "under half of it is profitable companies beating the market; the rest is unprofitable ones lagging it"
    assert 0 < whole["profitability"]["long"]["mean"] < 0.5 * whole["profitability"]["big"]["mean"]
    assert whole["profitability"]["short"]["mean"] > 0
    # value and investment among big companies are almost entirely the held side
    for k in ("value", "investment"):
        assert whole[k]["long"]["mean"] > 0.8 * whole[k]["big"]["mean"]
    html = PAGE.read_text(encoding="utf-8")
    assert "can keep only the first" in html and 'id="t-sides"' in html


def test_the_largest_fifth_experiment_is_recorded_as_it_landed(published):
    """Experiment 9: tolerances written first, most figures expected within. Five were. The
    four that describe the largest fifth all match; every miss involves the smallest fifth, and
    the sentence that says so is checked here."""
    exp, built = fl.EXPERIMENTS[8], published["experiments"][8]["result"]
    assert exp["id"] == "largest-fifth" and exp["kind"] == "split" and exp["split"]["data"] == "quintiles"
    assert exp["reference"]["scale"] == 12                                  # the paper reports percent a year
    assert {f["key"]: (f["paper"], f["tolerance"]) for f in exp["reference"]["figures"]} == {
        "mean_value_small": (11.22, 0.5), "t_value_small": (3.87, 0.25), "mean_value_big": (3.70, 0.5),
        "t_value_big": (1.90, 0.25), "mean_value_gap": (7.13, 0.5), "t_value_gap": (2.10, 0.25),
        "mean_momentum_small": (10.87, 0.5), "t_momentum_small": (4.50, 0.25), "mean_momentum_big": (7.49, 0.5),
        "t_momentum_big": (2.95, 0.25), "mean_momentum_gap": (3.42, 0.5), "t_momentum_gap": (1.56, 0.25)}
    rows = {r["key"]: r for r in built["rows"]}
    for k in ("mean_value_big", "t_value_big", "mean_momentum_big", "t_momentum_big"):
        assert rows[k]["within"], k                                          # "all four figures for the largest fifth match"
    missed = {k for k, r in rows.items() if not r["within"]}
    assert missed and all(k.endswith("_small") or k.endswith("_gap") for k in missed)   # "every miss involves the smallest fifth"
    assert "All four figures for the largest fifth" in exp["verdict_note"]
    # a year's figure is twelve average months, and the two sides add up to the spread
    qd = published["quintiles"]
    r = fl.split_run(qd, exp["reference"]["start"], exp["reference"]["end"])["factors"]
    assert rows["mean_momentum_big"]["here"] == pytest.approx(12 * r["momentum"]["big"]["mean"])
    assert list(k for k in qd if k != "months") == fl.QUINT_KEYS == exp["split"]["keys"]
    for k in fl.QUINT_KEYS:
        assert set(qd[k]) == set(fl.SPLIT_PARTS)
        for big, held, shorted in zip(qd[k]["big"], qd[k]["long"], qd[k]["short"]):
            assert big == pytest.approx(held + shorted, abs=2e-4)
    assert fl.QUINT_COLS == {"small": (0, 4), "big": (20, 24)}


def test_the_scorecard_is_computed_and_carries_the_screeners_own_weights(published):
    """The front page's table of where each category stands. Its weights are the config's - a
    weight change without a rebuild of the lab fails here - and every figure in it is rebuilt
    from the published series. The words around the figures are checked below them."""
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    card = published["scorecard"]
    assert [(r["key"], r["weight"]) for r in card] == list(cfg["factor_weights"].items())
    assert card == json.loads(json.dumps(fl.scorecard(published["split"], published["size"], cfg["factor_weights"])))
    assert all(r["measured"] and r["open"] for r in card) and set(fl.SCORECARD_OPEN) == set(cfg["factor_weights"])
    assert advice_terms_in(json.dumps(card)) == []
    sp = published["split"]
    whole, late, recent = (fl.split_run(sp, a, "209912")["factors"] for a in ("196307", "200001", "201401"))
    # valuation: "about half its size among small ones ... since 2014" close to nothing
    assert 0.35 < whole["value"]["big"]["mean"] / whole["value"]["small"]["mean"] < 0.65
    assert abs(recent["value"]["big"]["mean"]) < 0.15
    assert fl.sort_run(published["sorts"]["earnings-yield"], "200701", "209912", "vw")["spread"]["mean"] < 0
    # quality: "carries over best ... the only one still distinguishable from zero among big companies since 2000"
    ratio = {k: whole[k]["big"]["mean"] / whole[k]["small"]["mean"] for k in fl.SPLIT_KEYS}
    assert max(ratio, key=ratio.get) == "profitability"
    assert [k for k in fl.SPLIT_KEYS if abs(late[k]["big"]["t"]) >= 2] == ["profitability"]
    # momentum: "under half its size among small ones"; since 2000 not distinguishable from zero
    assert ratio["momentum"] < 0.5 and abs(late["momentum"]["big"]["t"]) < 1
    # investment: "it moves with value"
    assert fl.correlation(sp["value"]["big"], sp["investment"]["big"]) > 0.4
    # growth and revisions have nothing rerun, and say so
    by_key = {r["key"]: r for r in card}
    for k in ("growth", "revisions"):
        assert by_key[k]["experiments"] == [] and by_key[k]["measured"].startswith("Nothing.")
    # the page says, above the table, that none of this sets a weight
    html = PAGE.read_text(encoding="utf-8")
    assert "The lab does not set the screener's weights, and none has changed because of it." in html
    assert "D.scorecard.map" in html


def test_the_table_of_results_after_their_papers_is_computed(published):
    """One row per headline result: its value in the paper's sample and in the months since.
    Rebuilt here from the published series; the page's sentence counts the rows itself."""
    built = published["after_the_paper"]
    again = fl.after_the_paper(published["momentum"], published["market"], published["factors"],
                               published["size"], published["sorts"], published["split"],
                               published["quintiles"])
    assert [r["label"] for r in again] == [r["label"] for r in built] and len(built) >= 10
    assert {r["experiment"] for r in built} == {x["id"] for x in fl.EXPERIMENTS}
    for a, b in zip(again, built):
        for side in ("inside", "after"):
            assert a[side]["months"] == b[side]["months"] > 0
            assert a[side]["value"] == pytest.approx(b[side]["value"])
        assert a["weaker"] == b["weaker"]
        if b["sign"] == 0:
            assert b["weaker"] is None
        else:
            assert b["weaker"] == (b["sign"] * b["after"]["value"] < b["sign"] * b["inside"]["value"])
    # the periods do not overlap: "since" starts the month after the paper's sample ends
    assert fl.next_month("201112") == "201201" and fl.next_month("201303") == "201304"
    rm = next(r for r in built if r["label"].startswith("Risk-managed"))
    assert rm["inside"]["months"] == 1016 and rm["weaker"] is True
    html = PAGE.read_text(encoding="utf-8")
    assert "D.after_the_paper.filter(r => r.weaker !== null)" in html and "were weaker" in html


def test_the_category_map_covers_the_screeners_eight_and_nothing_else():
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert [c["key"] for c in fl.CATEGORY_MAP] == list(cfg["factor_weights"])
    ids = {x["id"] for x in fl.EXPERIMENTS}
    for c in fl.CATEGORY_MAP:
        assert set(c["experiments"]) <= ids and c["has"] and c["gap"] and c["category"]
    assert advice_terms_in(json.dumps(fl.CATEGORY_MAP) + json.dumps(fl.TAKEAWAYS)) == []
    # every study the map says the methodology cites is in the methodology the generator writes
    import run_screener
    overview = run_screener.build_screener_overview(cfg)
    for study in ("Piotroski (2000)", "Beneish (1999)", "Mohanram (2005)", "Jegadeesh & Titman, 1993",
                  "Chan, Jegadeesh & Lakonishok (1996"):
        assert study in overview, study


def test_the_front_page_lists_every_experiment_and_opens_them():
    sync_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")
    with sync_api.sync_playwright() as pw:
        try:
            browser = pw.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        try:
            for width in (1200, 375):
                page = browser.new_page(viewport={"width": width, "height": 900})
                errors: list[str] = []
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.goto(PAGE.as_uri(), wait_until="load")
                page.wait_for_function("window.LAB_STATE && window.LAB_STATE().id === 'overview'", timeout=30000)
                assert page.evaluate("document.querySelectorAll('.card').length") == len(fl.EXPERIMENTS)
                assert page.evaluate("document.querySelectorAll('#t-map tbody tr').length") == 8
                assert page.evaluate("document.querySelectorAll('#t-score tbody tr').length") == 8
                assert "22% of the score" in page.inner_text("#t-score")
                assert page.evaluate("document.querySelectorAll('#t-after tbody tr').length") >= 10
                assert "were weaker" in page.inner_text("#o-after + p")
                total = sum(len(x["reference"].get("figures", [])) or 14 for x in fl.EXPERIMENTS)
                assert f"of {total} figures" in page.inner_text("#lab")
                assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
                page.click(".card >> nth=1")
                page.wait_for_function("window.LAB_STATE().id === 'momentum-crashes'", timeout=30000)
                assert page.evaluate("window.LAB_STATE().result.cells.length") == 4
                assert errors == []
                page.close()
        finally:
            browser.close()


def test_the_methodology_says_what_the_factors_did_since_2014(published):
    """Limitation 11 on the methodology page (in the generator, rule 10) states what four
    factors have earned since 2014. It is a claim about data that will keep arriving: when a
    monthly refresh of the lab makes any part of it untrue, this fails and the sentence is
    rewritten - which is the point of checking it instead of typing numbers into it."""
    import yaml
    import run_screener
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    text = run_screener.build_screener_overview(cfg)
    para = next(l for l in text.splitlines() if l.startswith("11. **The published premia"))
    assert "lab.html#five-factors" in para and "too short to settle anything" in para
    fac = published["factors"]
    first, since = fl.factor_run(fac, "196307", "201312"), fl.factor_run(fac, "201401", "209912")
    for k in ("smb", "hml", "rmw", "cma"):
        assert 0.2 <= first["stats"][k]["mean"] <= 0.45                 # "between 0.2% and 0.45% a month"
        assert abs(since["stats"][k]["t"]) < 2                          # "none ... distinguishable from zero"
    for k in ("smb", "hml", "cma"):
        assert abs(since["stats"][k]["mean"]) < 0.2                     # "each averaged close to zero"
    assert since["stats"]["rmw"]["mean"] < first["stats"]["rmw"]["mean"]   # "profitability has earned less"
    assert len(since["months"]) < 360                                   # still short of "roughly thirty years"


def test_the_methodology_links_each_category_to_its_experiments():
    """In the generator (rule 10). The methodology cannot import the lab - no pipeline module
    may - so the links are written out there, and this keeps them in step with the lab's map."""
    import yaml
    import run_screener
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    text = run_screener.build_screener_overview(cfg)
    para = next(l for l in text.splitlines() if l.startswith("**Rerun the research.**"))
    linked = set(re.findall(r"lab\.html#([a-z-]+)\)", para))
    assert linked == {x["id"] for x in fl.EXPERIMENTS}
    for c in fl.CATEGORY_MAP:
        name = {"revisions": "Analyst Revisions"}.get(c["key"], c["category"])
        if c["experiments"]:
            part = re.search(re.escape(name) + r" [-–] (.*?)(?:; [A-Z]|\. Growth)", para)
            assert part, name
            assert set(re.findall(r"lab\.html#([a-z-]+)\)", part.group(1))) == set(c["experiments"]), name
        else:
            assert name in para.split("have nothing there yet")[0].rsplit(".", 1)[-1], name


def test_the_methodology_states_risk_and_size_as_the_lab_measures_them(published):
    """Until 2026-10-12 the methodology called the low-volatility anomaly "one of the most
    robust findings in finance". The lab's reruns support a weaker statement, and the page now
    makes that one; the Size section quotes the lab's measurement. Both are checked here against
    the data, so a refresh that changes the picture fails this instead of leaving the page wrong."""
    import yaml
    import run_screener
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    text = run_screener.build_screener_overview(cfg)
    assert "one of the most robust findings in finance" not in text
    assert "so the extra risk has not been paid for" in text and "lab.html#market-beta" in text
    # "more volatile and higher-beta stocks have not earned reliably more than calmer ones"
    for name in ("low-volatility", "market-beta"):
        for weighting in ("vw", "ew"):
            r = fl.sort_run(published["sorts"][name], "196307", "209912", weighting)
            assert r["spread"]["t"] < 2, (name, weighting)                      # riskiest minus calmest: not reliably positive
            assert r["groups"][-1]["std"] > 1.8 * r["groups"][0]["std"], (name, weighting)   # at about twice the risk
    # "the ninth size decile has out-earned the tenth by about 0.1% a month over the past century -
    #  the same sign in every period checked, and distinguishable from zero only over the full hundred years"
    size = published["size"]
    whole = fl.size_run(size, "192607", "209912", "ew")["nine_ten"]
    assert 0.05 < whole["mean"] < 0.15 and whole["t"] > 2
    assert len(fl.size_run(size, "192607", "209912", "ew")["months"]) >= 1200
    for a, b in (("196307", "199012"), ("199101", "209912"), ("200001", "209912")):
        for w in ("ew", "vw"):
            part = fl.size_run(size, a, b, w)["nine_ten"]
            assert part["mean"] > 0 and part["t"] < 2, (a, b, w)
    assert "ninth size decile has out-earned the tenth by about 0.1% a month" in text


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
                page.goto(PAGE.as_uri() + "#risk-managed-momentum", wait_until="load", timeout=60000)
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
                assert page.evaluate("document.querySelectorAll('.tabs a').length") == len(fl.EXPERIMENTS) + 1
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
                # the fourth
                page.goto(PAGE.as_uri() + "#size-effect", wait_until="load", timeout=60000)
                page.wait_for_function("window.LAB_STATE && window.LAB_STATE().id === 'size-effect'", timeout=30000)
                assert page.evaluate("document.getElementById('x-mismatch').hidden") is True
                assert page.evaluate("document.querySelectorAll('#ch-bars rect').length") == 10
                assert page.evaluate("document.querySelectorAll('#ch-bars rect.col-a').length") == 2
                # every bar reaches the zero line (a style named .bar once cut the grey ones short)
                assert page.evaluate("(() => { const r = [...document.querySelectorAll('#ch-bars rect')].map(e => e.getBoundingClientRect().bottom); return Math.max(...r) - Math.min(...r); })()") < 1.5
                ew = page.evaluate("window.LAB_STATE().result.small_big.mean")
                page.click("#c-weighting button[data-v='vw']")
                assert page.evaluate("window.LAB_STATE().result.small_big.mean") != ew
                assert page.evaluate("location.hash") == "#size-effect?weighting=vw"
                assert "Between the two largest groups" in page.inner_text("#x-read")
                assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
                assert errors == []
                # the sorted-portfolio layout, by its first user
                page.goto(PAGE.as_uri() + "#low-volatility", wait_until="load", timeout=60000)
                page.wait_for_function("window.LAB_STATE && window.LAB_STATE().id === 'low-volatility'", timeout=30000)
                assert page.evaluate("document.getElementById('x-mismatch').hidden") is True
                assert page.evaluate("document.querySelectorAll('#ch-bars rect').length") == 5
                assert page.evaluate("document.querySelectorAll('#t-live tbody tr').length") == 6
                assert "calmest fifth averaged" in page.inner_text("#x-read")
                page.click("#c-preset button[data-i='1']")
                assert page.evaluate("location.hash").startswith("#low-volatility?start=200101")
                assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
                assert errors == []
                # the factors within each half of the market
                page.goto(PAGE.as_uri() + "#big-and-small", wait_until="load", timeout=60000)
                page.wait_for_function("window.LAB_STATE && window.LAB_STATE().id === 'big-and-small'", timeout=30000)
                assert page.evaluate("document.getElementById('x-mismatch').hidden") is True
                assert page.evaluate("document.querySelectorAll('#ch-bars rect').length") == 8
                assert page.evaluate("document.querySelectorAll('#ch-bars rect.col-a').length") == 4
                assert page.evaluate("document.querySelectorAll('#t-live tbody tr').length") == 4
                assert page.evaluate("document.querySelectorAll('#t-ref tbody tr').length") == 18
                assert page.evaluate("document.querySelectorAll('#t-sides tbody tr').length") == 4
                in_sample = page.evaluate("window.LAB_STATE().result.factors.momentum.big.mean")
                page.click("#c-preset button[data-i='2']")                # since 2000
                assert page.evaluate("location.hash").startswith("#big-and-small?start=200001")
                assert page.evaluate("window.LAB_STATE().result.factors.momentum.big.mean") < in_sample
                assert "1 of the 4</strong> among big companies" in page.inner_html("#x-read")
                assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
                assert errors == []
                # the same layout on the smallest and largest fifths
                page.goto(PAGE.as_uri() + "#largest-fifth", wait_until="load", timeout=60000)
                page.wait_for_function("window.LAB_STATE && window.LAB_STATE().id === 'largest-fifth'", timeout=30000)
                assert page.evaluate("document.getElementById('x-mismatch').hidden") is True
                assert page.evaluate("document.querySelectorAll('#ch-bars rect').length") == 4
                assert page.evaluate("document.querySelectorAll('#t-live tbody tr').length") == 2
                assert page.evaluate("document.querySelectorAll('#t-ref tbody tr').length") == 12
                assert "among the largest fifth of companies" in page.inner_text("#x-read")
                in_sample = page.evaluate("window.LAB_STATE().result.factors.momentum.big.mean")
                page.click("#c-preset button[data-i='2']")                # since 2000
                assert page.evaluate("location.hash").startswith("#largest-fifth?start=200001")
                assert page.evaluate("window.LAB_STATE().result.factors.momentum.big.mean") < in_sample
                assert "none</strong> among the largest fifth" in page.inner_html("#x-read")
                assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
                assert errors == []
                page.close()
        finally:
            browser.close()


def test_a_scene_can_be_sent_as_a_link_and_a_month_read_off_the_chart():
    """The address carries the experiment and whatever settings differ from its opening ones,
    so the page in front of one reader can be sent to another. Settings a link cannot mean
    are ignored rather than obeyed."""
    sync_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")
    with sync_api.sync_playwright() as pw:
        try:
            browser = pw.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        try:
            page = browser.new_page(viewport={"width": 1200, "height": 900})
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(PAGE.as_uri() + "#momentum-crashes?start=192901&end=193912&threshold=0.3", wait_until="load")
            page.wait_for_function("window.LAB_STATE && window.LAB_STATE().id === 'momentum-crashes'", timeout=30000)
            s = page.evaluate("window.LAB_STATE().settings")
            assert (s["start"], s["end"], s["threshold"], s["lookback"]) == ("192901", "193912", 0.3, 24)
            assert page.evaluate("document.getElementById('c-reset').hidden") is False
            assert page.inner_text("#c-copy") == "Copy a link to this scene"
            # changing a setting rewrites the address; going back to the opening ones clears it
            page.click("#c-preset button[data-i='0']")
            assert page.evaluate("location.hash") == "#momentum-crashes?threshold=0.3"
            page.click("#c-reset")
            assert page.evaluate("location.hash") == "#momentum-crashes"
            # pointing at the chart reads one month, in words
            page.locator("#ch-growth svg").scroll_into_view_if_needed()
            box = page.locator("#ch-growth svg").bounding_box()
            assert page.inner_text(".chart-read") == "Point at the chart to read any month."
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            read = page.inner_text(".chart-read")
            assert re.match(r"[A-Z][a-z]{2} \d{4}: momentum returned ", read) and "$1 had become $" in read
            assert page.evaluate("document.querySelector('.chart .cursor').style.visibility") == "visible"
            # a link that asks for things that do not exist opens the experiment as it starts
            page.goto(PAGE.as_uri() + "#five-factors?focus=nonsense&start=99&end=abc&window=7", wait_until="load")
            page.wait_for_function("window.LAB_STATE && window.LAB_STATE().id === 'five-factors'", timeout=30000)
            s = page.evaluate("window.LAB_STATE().settings")
            assert (s["focus"], s["start"], s["end"]) == ("hml", "196307", "201312")
            page.goto(PAGE.as_uri() + "#risk-managed-momentum?cap=1&target=0.2&window=21", wait_until="load")
            page.wait_for_function("window.LAB_STATE && window.LAB_STATE().id === 'risk-managed-momentum'", timeout=30000)
            s = page.evaluate("window.LAB_STATE().settings")
            assert (s["cap"], s["target"], s["window"]) == (1, 0.2, 21)
            assert errors == []
        finally:
            browser.close()


def test_the_profitability_sort_rises_with_profitability_as_the_paper_found(published):
    """Experiment 10 (2026-10-15): Novy-Marx (2013) on the public operating-profitability sort. The
    tolerances were written before the comparison; the public file sorts on a narrower measure and
    keeps financial firms, so a spread below the paper's was expected and is what came out."""
    exp = next(x for x in published["experiments"] if x["id"] == "profitability")
    r = exp["result"]
    assert (r["within"], r["total"]) == (6, 6)
    assert exp["reference"]["excess"] is True and exp["reference"]["weighting"] == "vw"
    assert r["means"][-1] > r["means"][0]                          # profitable beat unprofitable
    spread = next(x for x in r["rows"] if x["key"] == "spread")
    assert 0 < spread["here"] < spread["paper"]                    # positive, and below the paper's
    data = published["sorts"]["profitability"]
    assert len(data["rf"]) == len(data["months"])                   # excess rebuilt from the page's own data


def test_the_cash_flow_yield_sort_is_compared_as_the_paper_reports_it(published):
    """Experiment 11 (2026-10-15): Lakonishok, Shleifer & Vishny (1994), Table 2 of the working paper.
    Its returns are size-adjusted and annual, so each decile is compared against the average of the
    ten, in percent a year (``relative`` and ``scale``). Tolerances were written before the run."""
    exp = next(x for x in published["experiments"] if x["id"] == "cash-flow-yield")
    ref, r = exp["reference"], exp["result"]
    assert ref["relative"] is True and ref["scale"] == 12 and ref["weighting"] == "ew"
    assert (r["within"], r["total"]) == (11, 11)
    rows = {x["key"]: x for x in r["rows"]}
    assert sum(rows[f"mean{g}"]["here"] for g in range(1, 11)) == pytest.approx(0.0, abs=1e-6)
    assert rows["mean10"]["here"] > rows["mean1"]["here"] and rows["spread"]["here"] > 0


def test_the_accruals_sort_is_published_as_the_miss_it_is(published):
    """Experiment 12 (2026-10-15): Sloan (1996), Table 6. The ends and spread were compared before the
    middle rows were read from the rendered page (the source records the order); 6 of 11 within. The
    deciles are listed highest accruals first, so the spread is the paper's hedge."""
    exp = next(x for x in published["experiments"] if x["id"] == "accruals")
    assert exp["paper"]["url"] == "https://www.jstor.org/stable/248290" and not exp["paper"].get("doi")
    r = exp["result"]
    assert (r["within"], r["total"]) == (6, 11)
    rows = {x["key"]: x for x in r["rows"]}
    assert rows["spread"]["here"] > 0 and rows["spread"]["here"] < rows["spread"]["paper"] - 4.0


COMBO_SETTINGS = [{"start": "197201", "end": "201107", "group": "big"}, {"start": "201108", "end": "209912", "group": "big"},
                  {"start": "192701", "end": "209912", "group": "all"}]


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_the_browser_measures_value_and_momentum_together_as_the_reference_does(published, tmp_path):
    """Experiment 13 (2026-10-15): the browser's comboRun is the Python combo_run, figure for figure."""
    script = tmp_path / "combo.js"
    script.write_text(
        "const E = require(process.argv[2]);\n"
        "const text = require('fs').readFileSync(process.argv[3], 'utf8');\n"
        "const D = JSON.parse(text.slice(text.indexOf('{'), text.lastIndexOf('}') + 1));\n"
        "console.log(JSON.stringify(JSON.parse(process.argv[4]).map(o => { const r = E.comboRun(D.split, D.factors, o);\n"
        "  return { n: r.months.length, value: r.value, momentum: r.momentum, combo: r.combo, corr: r.corr, corr_resid: r.corr_resid }; })));\n",
        encoding="utf-8")
    res = subprocess.run([NODE, str(script), str(ENGINE), str(fl.OUT), json.dumps(COMBO_SETTINGS)],
                         capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    for o, js in zip(COMBO_SETTINGS, json.loads(res.stdout)):
        py = fl.combo_run(published["split"], published["factors"], **o)
        assert js["n"] == len(py["months"]) > 100
        for k in ("value", "momentum", "combo"):
            for stat in ("mean_a", "std_a", "sharpe", "t"):
                assert js[k][stat] == pytest.approx(py[k][stat], rel=1e-8, abs=1e-12), (o, k, stat)
        assert js["corr"] == pytest.approx(py["corr"], rel=1e-8) and js["corr_resid"] == pytest.approx(py["corr_resid"], rel=1e-8)


def test_the_value_and_momentum_rerun_is_the_paper_s_table_on_big_companies(published):
    exp = next(x for x in published["experiments"] if x["id"] == "value-and-momentum")
    assert exp["kind"] == "combo" and exp["reference"]["group"] == "big"
    assert exp["paper"]["doi"] == "10.1111/jofi.12021"
    rows = {r["key"]: r for r in exp["result"]["rows"]}
    assert rows["corr_resid"]["paper"] == -0.53 and rows["sharpe_combo"]["paper"] == 0.63
    assert rows["corr_resid"]["here"] < 0                                 # the hedge is there, if milder
    # the mix's volatility and Sharpe ratio follow from the correlation, and the page says so once
    # (2026-10-16 audit); nothing about the figures or their tolerances changed for it
    assert {r["key"] for r in exp["result"]["rows"] if not r["within"]} == {"std_combo", "sharpe_combo", "corr_resid"}
    assert {f["key"]: (f["paper"], f["tolerance"]) for f in exp["reference"]["figures"]} == {
        "mean_value": (3.7, 1.2), "mean_momentum": (5.4, 1.2), "mean_combo": (4.6, 1.2),
        "std_value": (12.8, 2.0), "std_momentum": (16.4, 2.0), "std_combo": (7.2, 2.0),
        "sharpe_value": (0.29, 0.10), "sharpe_momentum": (0.33, 0.10), "sharpe_combo": (0.63, 0.10),
        "corr_resid": (-0.53, 0.15)}
    assert "follow mechanically from how strongly value and momentum move against each other" in exp["against_note"]
    assert "X.against_note" in PAGE.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- the 2026-10-16 audit
def _open(pw, width: int = 1200):
    """A chromium page at a width, collecting page errors; skips where no browser can start."""
    try:
        browser = pw.chromium.launch()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no usable chromium: {exc}")
    page = browser.new_page(viewport={"width": width, "height": 900})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    return browser, page, errors


def _goto(page, exp_id: str) -> None:
    page.goto(PAGE.as_uri() + "#" + exp_id, wait_until="load", timeout=60000)
    page.wait_for_function(f"window.LAB_STATE && window.LAB_STATE().id === '{exp_id}'", timeout=30000)


def test_see_every_figure_scrolls_to_the_comparison_and_keeps_the_experiment():
    """The verdict's "See every figure" link once rewrote the address to #x-against, which the
    scene reader took for an unknown experiment: the overview replaced the page. It now scrolls to
    the comparison and leaves the experiment, and its address, where they were."""
    sync_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")
    with sync_api.sync_playwright() as pw:
        browser, page, errors = _open(pw)
        try:
            for exp in fl.EXPERIMENTS:
                _goto(page, exp["id"])
                page.click(".verdict a[href='#x-against']")
                assert page.evaluate("window.LAB_STATE().id") == exp["id"]
                assert page.evaluate("location.hash") == "#" + exp["id"]
                top = page.evaluate("document.getElementById('x-against').getBoundingClientRect().top")
                assert -2 <= top < 900, (exp["id"], top)                     # in view (a scroll lands to the sub-pixel)
            # the same address typed by hand is read the same way, and the scene's address comes back
            last = fl.EXPERIMENTS[-1]["id"]
            page.evaluate("location.hash = '#x-against'")
            page.wait_for_function(f"location.hash === '#{last}'", timeout=10000)
            assert page.evaluate("window.LAB_STATE().id") == last
            assert errors == []
        finally:
            browser.close()


def test_each_experiment_says_when_its_tolerances_were_written(published):
    """Rule 3 says tolerances are written before the comparison. Three records say otherwise for
    themselves - experiment 1's were set after its first comparison, experiment 12's end values
    were corrected after theirs, experiment 13's figures had been measured an hour before - and the
    page used to print "written before" under all but the first. Each record now carries its own
    account, and the page prints it from one function."""
    assert set(fl.TOLERANCES_WRITTEN) == {"before", "after_first_comparison", "after_figures_seen"}
    assert published["tolerances_written"] == fl.TOLERANCES_WRITTEN
    for exp in fl.EXPERIMENTS:
        ref = exp["reference"]
        assert ref["tolerances_written"] in fl.TOLERANCES_WRITTEN, exp["id"]
        if ref["tolerances_written"] != "before":
            assert len(ref.get("tolerances_note", "")) > 60, exp["id"]          # what happened, not just a label
    status = {x["id"]: x["reference"]["tolerances_written"] for x in fl.EXPERIMENTS}
    assert status["risk-managed-momentum"] == "after_first_comparison"
    assert status["accruals"] == "after_first_comparison" and status["value-and-momentum"] == "after_figures_seen"
    html = PAGE.read_text(encoding="utf-8")
    assert "These tolerances were written before the comparison was first run" not in html
    assert "They were set after the first comparison was seen" not in html
    kinds = {x["kind"] for x in fl.EXPERIMENTS}
    assert html.count("${tolNote(X)}</p>") == len(kinds) == 7                  # every layout, one function
    sync_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")
    with sync_api.sync_playwright() as pw:
        browser, page, errors = _open(pw)
        try:
            for exp in fl.EXPERIMENTS:
                _goto(page, exp["id"])
                note = page.locator("#x-against ~ p.note", has_text="written down in the build").inner_text()
                assert fl.TOLERANCES_WRITTEN[exp["reference"]["tolerances_written"]] in note, exp["id"]
                assert exp["reference"].get("tolerances_note", "")[:40] in note, exp["id"]
            assert errors == []
        finally:
            browser.close()


def test_what_a_result_of_no_effect_would_match_is_published_beside_every_verdict(published):
    """A tolerance wide enough to pass a result of no effect at all says nothing about the paper.
    No tolerance moves for it (rule 3): the build counts, from the record alone, how many figures a
    null result would land within, and the page prints that beside every verdict (plan rule 8)."""
    for built, exp in zip(published["experiments"], fl.EXPERIMENTS):
        r, nulls = built["result"], fl.null_rows(exp)
        assert len(nulls) == len(r["rows"]) == r["total"]
        assert [n["paper"] for n in nulls] == [row["paper"] for row in r["rows"]]       # the same figures, in order
        assert [row["null"] for row in r["rows"]] == pytest.approx([n["null"] for n in nulls])
        assert r["null_within"] == fl.null_within(exp) == sum(
            abs(row["null"] - row["paper"]) <= row["tolerance"] for row in r["rows"])
        assert r["null_meaning"] == fl.null_meaning(exp) and r["null_meaning"]
    # the rules, on the record: experiment 1's null is the plain strategy; a sort's groups all earn
    # the paper's own average across them; a spread is zero; a volatility is left at the paper's
    rm = fl.null_rows(fl.EXPERIMENTS[0])
    assert [n["null"] for n in rm[7:]] == [fl.EXPERIMENTS[0]["reference"]["plain"][k] for k in fl.STAT_ORDER]
    ey = next(x for x in fl.EXPERIMENTS if x["id"] == "earnings-yield")
    figs, nulls = ey["reference"]["figures"], fl.null_rows(ey)
    level = sum(f["paper"] for f in figs if f["key"].startswith("mean")) / 10
    assert {f["key"]: n["null"] for f, n in zip(figs, nulls)} == pytest.approx(
        {**{f"mean{g}": level for g in range(1, 11)}, "spread": 0.0})
    lv = next(x for x in fl.EXPERIMENTS if x["id"] == "low-volatility")
    assert all(n["null"] == f["paper"] for f, n in zip(lv["reference"]["figures"], fl.null_rows(lv)) if f["key"].startswith("std"))
    # the counts themselves follow from paper values and tolerances that are pinned above
    assert {x["id"]: fl.null_within(x) for x in fl.EXPERIMENTS} == {
        "risk-managed-momentum": 7, "momentum-crashes": 7, "five-factors": 8, "size-effect": 1,
        "low-volatility": 6, "earnings-yield": 1, "market-beta": 9, "big-and-small": 0, "largest-fifth": 0,
        "profitability": 3, "cash-flow-yield": 6, "accruals": 4, "value-and-momentum": 3}
    # every sort whose paper claims an effect: the null misses the spread between the ends
    for exp in fl.EXPERIMENTS:
        if exp["kind"] == "sort" and exp["id"] != "market-beta":                      # beta's claim is "no effect"
            spread = next(n for f, n in zip(exp["reference"]["figures"], fl.null_rows(exp)) if f["key"] == "spread")
            assert abs(spread["null"] - spread["paper"]) > spread["tolerance"], exp["id"]
    html = PAGE.read_text(encoding="utf-8")
    assert "nullLine(r)" in html and "nullLine(x.result)" in html and "nullLine(X.result)" in html
    sync_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")
    with sync_api.sync_playwright() as pw:
        browser, page, errors = _open(pw)
        try:
            page.goto(PAGE.as_uri(), wait_until="load", timeout=60000)
            page.wait_for_function("window.LAB_STATE && window.LAB_STATE().id === 'overview'", timeout=30000)
            cards = page.locator(".card .null").all_inner_texts()
            assert len(cards) == len(fl.EXPERIMENTS)
            for text, exp in zip(cards, published["experiments"]):
                r = exp["result"]
                assert f"would match {r['null_within'] or 'none'} of these {r['total']}" in text
            _goto(page, "cash-flow-yield")
            assert "A result of no effect at all would match 6 of these 11." in page.inner_text(".verdict")
            assert errors == []
        finally:
            browser.close()


def test_every_printed_difference_agrees_with_its_verdict_against_the_printed_tolerance(published):
    """Experiment 8's profitability among big companies differs from the paper by 0.0325 against a
    tolerance of 0.03. Printed to two places it read "+0.03 / ±0.03 / outside", which a reader
    checking the arithmetic cannot reconcile. Every row on every experiment is read off the page
    here: the printed difference, compared with the printed tolerance, gives the printed verdict."""
    sync_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")
    num = lambda t: float(t.strip().replace("−", "-").replace("±", "").replace("+", ""))  # noqa: E731
    with sync_api.sync_playwright() as pw:
        browser, page, errors = _open(pw)
        try:
            checked = 0
            for built in published["experiments"]:
                _goto(page, built["id"])
                cells = page.evaluate("[...document.querySelectorAll('#t-ref tbody tr:not(.group)')]"
                                      ".map(tr => [...tr.cells].map(td => td.textContent))")
                assert len(cells) == built["result"]["total"], built["id"]
                for row, c in zip(built["result"]["rows"], cells):
                    diff, tol, verdict = num(c[3]), num(c[4]), c[5]
                    assert verdict == ("within" if row["within"] else "outside")
                    assert (abs(diff) <= tol) == row["within"], (built["id"], row["label"], c)
                    assert diff == pytest.approx(row["diff"], abs=0.006)              # and it is the difference
                    checked += 1
            assert checked == sum(x["result"]["total"] for x in published["experiments"])
            big = {r["key"]: r for r in published["experiments"][7]["result"]["rows"]}["mean_profitability_big"]
            assert 0.03 < big["diff"] < 0.035 and not big["within"]                   # the row the audit found
            assert errors == []
        finally:
            browser.close()


def test_the_year_pickers_keep_the_period_whole_and_inside_the_data():
    """One period helper for every layout (it was copied seven times). Picking From 2026 and then
    To 2020 used to leave 2026-2026 without a word; picking To = the data's last year gave an end
    month past the data, so the preset running to the latest month never showed pressed."""
    sync_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")
    settings = "window.LAB_STATE().settings"
    with sync_api.sync_playwright() as pw:
        browser, page, errors = _open(pw)
        try:
            for exp in fl.EXPERIMENTS:
                _goto(page, exp["id"])
                years = page.evaluate("[...document.querySelectorAll('#c-to option')].map(o => o.value)")
                page.click("#c-preset button[data-i='1']")                     # "After the paper": to the latest month
                after = page.evaluate(settings)
                middle = str(int(after["start"][:4]) + 3)
                page.select_option("#c-to", middle)
                assert page.evaluate(settings)["end"] == middle + "12"
                page.select_option("#c-to", years[-1])                          # the data's last year: its last month
                s = page.evaluate(settings)
                assert (s["start"], s["end"]) == (after["start"], after["end"]), exp["id"]
                assert page.get_attribute("#c-preset button[data-i='1']", "aria-pressed") == "true", exp["id"]
                # From the last year, then To an earlier one: the start follows the end back
                page.select_option("#c-from", years[-1])
                s = page.evaluate(settings)
                assert s["start"] == years[-1] + "01" and s["end"] == after["end"], exp["id"]
                early = years[-7]
                page.select_option("#c-to", early)
                s = page.evaluate(settings)
                assert (s["start"], s["end"]) == (early + "01", early + "12"), exp["id"]
                assert page.input_value("#c-from") == page.input_value("#c-to") == early
                # the first year starts at the data's first month, never before it
                page.select_option("#c-from", years[0])
                first = page.evaluate(settings)["start"]
                assert first[:4] == years[0] and first >= years[0] + "01"
                assert page.evaluate("window.LAB_STATE().result") is not None
            assert errors == []
        finally:
            browser.close()


def test_a_chart_can_be_read_month_by_month_from_the_keyboard():
    """The month-by-month readout answered only a pointer. Each line chart now takes focus; the
    arrow keys, Home and End move its cursor; the sentence under it is a polite live region. And no
    chart's label points at "the table below" unless the table below holds its figures."""
    sync_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")
    mon = lambda m: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][int(m[4:]) - 1] + " " + m[:4]  # noqa: E731
    with sync_api.sync_playwright() as pw:
        browser, page, errors = _open(pw)
        try:
            for exp in fl.EXPERIMENTS:
                _goto(page, exp["id"])
                labels = page.evaluate("[...document.querySelectorAll('.chart, .chart svg')].map(e => e.getAttribute('aria-label') || '')")
                if exp["kind"] not in ("sort", "split"):                         # their tables hold the bars' figures
                    assert not any("table below" in l for l in labels), (exp["id"], labels)
                if exp["kind"] not in ("risk-managed", "crashes", "factors", "combo"):
                    continue
                months = page.evaluate("window.LAB_STATE().result.months")
                svg, read = "#ch-growth svg", "#ch-growth + .chart-read"
                assert page.get_attribute(svg, "tabindex") == "0" and page.get_attribute(svg, "role") == "img"
                assert "arrow keys" in page.get_attribute(svg, "aria-label")
                assert page.get_attribute(read, "aria-live") == "polite"
                page.focus(svg)
                page.keyboard.press("End")
                assert page.inner_text(read).startswith(mon(months[-1]) + ":"), exp["id"]
                page.keyboard.press("Home")
                assert page.inner_text(read).startswith(mon(months[0]) + ":")
                page.keyboard.press("ArrowRight")
                assert page.inner_text(read).startswith(mon(months[1]) + ":")
                page.keyboard.press("ArrowLeft")
                page.keyboard.press("ArrowLeft")                                 # held at the first month
                assert page.inner_text(read).startswith(mon(months[0]) + ":")
                assert page.evaluate("document.querySelector('#ch-growth .cursor').style.visibility") == "visible"
                page.focus("#c-from")                                             # leaving the chart clears it
                assert page.inner_text(read) == "Point at the chart to read any month."
            # the size chart's figures are in no table below, so its label carries them
            _goto(page, "size-effect")
            label = page.get_attribute("#ch-bars", "aria-label")
            means = page.evaluate("window.LAB_STATE().result.deciles.map(d => d.mean.toFixed(2))")
            assert all(m.replace("-", "−") in label for m in means)
            assert errors == []
        finally:
            browser.close()

