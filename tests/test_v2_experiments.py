"""The research engine (``v2_experiments.py``): its rules are what make a result worth showing.

Pinned: the discovery and held-out periods are fixed in code and do not overlap; an experiment
must be registered before it can run; every registered transform builds the configuration it
says; a comparison with the baseline is month by month on the same months; the verdict needs the
held-out months to agree with discovery and reports the multiple-testing bar honestly; every run is
appended to the ledger; and no production module imports the engine.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import v2_experiments as X  # noqa: E402


def test_the_held_out_months_are_fixed_and_never_overlap_discovery():
    assert X.DISCOVERY == ("2020-01-31", "2023-06-30")
    assert X.HOLDOUT == ("2023-07-31", "2026-08-31")
    assert X.DISCOVERY[1] < X.HOLDOUT[0]
    assert X.HLZ_T == 3.0


def test_every_registered_experiment_has_a_prediction_and_a_source_and_builds():
    reg = X.load_registry()
    ids = [e["id"] for e in reg]
    assert len(ids) == len(set(ids)) and "baseline" in ids
    base = X.base_config()
    for e in reg:
        assert e["hypothesis"] and e["source"] and e["prediction"], e["id"]
        cfg = X.apply_transform(base, e["transform"])
        assert sum(cfg["factor_weights"].values()) > 0, e["id"]


def test_the_transforms_do_what_they_say():
    base = X.base_config()
    only = X.apply_transform(base, {"kind": "only_category", "category": "quality"})
    assert {k for k, v in only["factor_weights"].items() if v} == {"quality"}
    drop = X.apply_transform(base, {"kind": "drop_category", "category": "growth"})
    assert drop["factor_weights"]["growth"] == 0 and drop["factor_weights"]["valuation"] == base["factor_weights"]["valuation"]
    eq = X.apply_transform(base, {"kind": "equal_categories"})
    assert {v for v in eq["factor_weights"].values() if v} == {1}
    mf = X.apply_transform(base, {"kind": "metrics", "metrics": {"valuation": {"ev_ebitda": 50}, "quality": {"roic": 50}}})
    assert {k for k, v in mf["factor_weights"].items() if v} == {"valuation", "quality"}
    assert [m for m, w in mf["metric_weights"]["valuation"].items() if w] == ["ev_ebitda"]
    assert [m for m, w in mf["metric_weights"]["quality"].items() if w] == ["roic"]
    sw = X.apply_transform(base, {"kind": "config", "set": {"factor_neutralization.enabled": True}})
    assert sw["factor_neutralization"]["enabled"] is True
    assert base["factor_weights"]["growth"] > 0                     # the baseline itself is untouched
    with pytest.raises(ValueError):
        X.apply_transform(base, {"kind": "made-up"})


def test_an_unregistered_experiment_cannot_be_run():
    with pytest.raises(SystemExit, match="register before running"):
        X.run(["not-in-the-registry"])


def _pm(ic, months, base_ic=None):
    """A per-month frame. With ``base_ic`` it is a *paired* frame - the shape ``per_month`` returns
    when it was given the baseline's rankings, carrying the baseline's reading on the same stocks."""
    df = pd.DataFrame({"ic": ic, "top_minus_universe": [x / 10 for x in ic], "top": 0.01,
                       "universe": 0.0, "n": 400}, index=months)
    if base_ic is not None:
        df["base_ic"] = base_ic
        df["base_top_minus_universe"] = [x / 10 for x in base_ic]
        df["n_common"] = 400
    return df


def test_the_comparison_is_paired_month_by_month():
    months = [d.strftime("%Y-%m-%d") for d in pd.date_range("2020-01-31", "2026-08-31", freq="ME")]
    assert len(months) == 80
    rng = np.random.default_rng(1)
    base_ic = rng.normal(0, 0.12, 80)
    exp = _pm(base_ic + 0.05, months, base_ic=base_ic)  # better by exactly 0.05 every month
    s = X.summarise(exp)
    for period in ("discovery", "holdout"):
        d = s[period]["ic_vs_baseline"]
        assert d["mean"] == pytest.approx(0.05) and d["se"] == pytest.approx(0.0, abs=1e-9)
    assert s["discovery"]["ic"]["n"] + s["holdout"]["ic"]["n"] == s["all"]["ic"]["n"]


def _scores(month: str, composites: dict) -> pd.DataFrame:
    return pd.DataFrame({"month": month, "Ticker": list(composites), "Composite": list(composites.values())})


def _fwd(month: str, rets: dict) -> pd.DataFrame:
    return pd.DataFrame([rets], index=[month])


def test_a_narrow_screen_is_compared_on_the_stocks_both_rankings_carry():
    """Queue item 8a. The experiment ranks 60 of the baseline's 80 stocks; the comparison is over
    those 60 on both sides, and ``n_common`` says so."""
    month = "2024-01-31"
    tickers = [f"T{i:03d}" for i in range(80)]
    rng = np.random.default_rng(7)
    rets = dict(zip(tickers, rng.normal(0, 0.05, 80)))
    base = _scores(month, dict(zip(tickers, rng.normal(0, 1, 80))))
    narrow = _scores(month, dict(zip(tickers[:60], rng.normal(0, 1, 60))))
    pm = X.per_month(narrow, _fwd(month, rets), baseline_scores=base)
    assert pm.loc[month, "n_common"] == 60 and pm.loc[month, "n"] == 60
    # the baseline's reading in this frame is its reading on those 60, not on all 80
    full = X.per_month(base, _fwd(month, rets))
    assert pm.loc[month, "base_ic"] != pytest.approx(full.loc[month, "ic"])
    on_60 = X.per_month(base[base["Ticker"].isin(tickers[:60])], _fwd(month, rets))
    assert pm.loc[month, "base_ic"] == pytest.approx(on_60.loc[month, "ic"])
    assert pm.loc[month, "base_top_minus_universe"] == pytest.approx(on_60.loc[month, "top_minus_universe"])


def test_an_identical_ranking_differs_from_the_baseline_by_exactly_nothing():
    """The property version 1 could not give: a configuration that ranks the same stocks the same
    way must show a difference of exactly zero, whatever stocks it is missing."""
    month = "2024-02-29"
    tickers = [f"T{i:03d}" for i in range(90)]
    rng = np.random.default_rng(11)
    comps = dict(zip(tickers, rng.normal(0, 1, 90)))
    rets = dict(zip(tickers, rng.normal(0, 0.05, 90)))
    base = _scores(month, comps)
    same_but_thinner = _scores(month, {t: comps[t] for t in tickers[:55]})
    pm = X.per_month(same_but_thinner, _fwd(month, rets), baseline_scores=base)
    assert pm.loc[month, "ic"] == pytest.approx(pm.loc[month, "base_ic"])
    assert pm.loc[month, "top_minus_universe"] == pytest.approx(pm.loc[month, "base_top_minus_universe"])


def test_a_month_too_thin_to_compare_is_dropped_rather_than_half_measured():
    month = "2024-03-31"
    tickers = [f"T{i:03d}" for i in range(80)]
    rng = np.random.default_rng(3)
    rets = dict(zip(tickers, rng.normal(0, 0.05, 80)))
    base = _scores(month, dict(zip(tickers, rng.normal(0, 1, 80))))
    tiny = _scores(month, dict(zip(tickers[:X.MIN_STOCKS_A_MONTH - 1], rng.normal(0, 1, X.MIN_STOCKS_A_MONTH - 1))))
    assert X.per_month(tiny, _fwd(month, rets), baseline_scores=base).empty


def test_the_unpaired_frame_is_still_the_configurations_own_full_reading():
    """The baseline's own ledger row - the figure the methodology page publishes - is unpaired."""
    month = "2024-04-30"
    tickers = [f"T{i:03d}" for i in range(80)]
    rng = np.random.default_rng(5)
    base = _scores(month, dict(zip(tickers, rng.normal(0, 1, 80))))
    pm = X.per_month(base, _fwd(month, dict(zip(tickers, rng.normal(0, 0.05, 80)))))
    assert pm.loc[month, "n"] == 80 and "n_common" not in pm.columns and "base_ic" not in pm.columns


def test_summarise_cannot_difference_against_a_separately_measured_baseline():
    """Method version 2: the baseline a difference is taken from travels in the frame, measured on
    the same stocks. There is deliberately no parameter for passing another configuration's frame -
    that was method version 1, and it is what 8a removed."""
    months = [d.strftime("%Y-%m-%d") for d in pd.date_range("2020-01-31", "2023-06-30", freq="ME")]
    unpaired = _pm(list(np.linspace(-0.05, 0.05, len(months))), months)
    assert "ic_vs_baseline" not in X.summarise(unpaired)["discovery"]
    with pytest.raises(TypeError):
        X.summarise(unpaired, _pm([0.0] * len(months), months))


def test_the_verdict_needs_the_held_out_months_to_agree():
    def res(d_mean, d_p, h_mean, h_p, h_t):
        return {"summary": {"discovery": {"ic_vs_baseline": {"mean": d_mean, "p": d_p}},
                            "holdout": {"ic_vs_baseline": {"mean": h_mean, "p": h_p, "t": h_t}}}}
    e = {"id": "x"}
    assert X.verdict(e, res(0.03, 0.01, 0.002, 0.6, 0.5), 24) == "looked different in discovery; did not hold out"
    assert X.verdict(e, res(0.0, 0.7, 0.03, 0.01, 2.6), 24) == "different only in the held-out months - not confirmed"
    assert X.verdict(e, res(0.03, 0.01, 0.03, 0.01, 2.6), 24).endswith("but not the multiple-testing bar")
    assert X.verdict(e, res(0.03, 0.001, 0.04, 0.0001, 3.9), 24).endswith("clears the multiple-testing bar")
    assert X.verdict(e, res(0.0, 0.8, 0.0, 0.9, 0.1), 24) == "not distinguishable from the baseline"
    assert X.verdict(e, {"summary": {"holdout": {}}}, 24) == "baseline"


def test_no_production_module_imports_it():
    pat = re.compile(r"^\s*(import v2_experiments|from v2_experiments)\b", re.M)
    # research_page.py is the engine's own report writer (2026-10-16: it reads the engine's Student's t
    # and pass bar rather than re-deriving them), and is itself never imported by production -
    # tests/test_research_page.py::test_no_production_module_imports_the_generator
    for p in ROOT.glob("*.py"):
        if p.name in ("v2_experiments.py", "conftest.py", "research_page.py"):
            continue
        assert not pat.search(p.read_text(encoding="utf-8", errors="replace")), p.name


def test_a_stock_with_no_composite_is_left_out_of_the_month_not_ranked(monkeypatch):
    """2026-10-15: a narrow experiment (one category, one metric) can leave a stock with no composite;
    the first full run crashed casting its NaN rank to an integer. ``rank_month`` now drops it."""
    import backtest_v2 as v2
    fe = v2.fe
    df = pd.DataFrame({"Ticker": ["A", "B", "C"], "Sector": ["X", "X", "X"]})
    monkeypatch.setattr(fe, "applicable_coverage", lambda d, c: (pd.Series(9, index=d.index), pd.Series(9, index=d.index)))
    for name in ("apply_universe_filters", "apply_percentile_transform", "compute_category_scores"):
        monkeypatch.setattr(fe, name, lambda d, c: d)
    monkeypatch.setattr(fe, "compute_sector_percentiles", lambda d: d)
    monkeypatch.setattr(fe, "compute_composite", lambda d, c: d.assign(Composite=[70.0, np.nan, 40.0]))
    cfg = {"data_quality": {"min_data_coverage_pct": 50}, "factor_weights": {}}
    out = v2.rank_month(df, cfg, {})
    assert list(out["Ticker"]) == ["A", "C"] and list(out["Rank"]) == [1, 2]


def test_the_ledger_says_which_data_each_run_used():
    """A rebuild of the monthly records is a new vintage; the fingerprint changes with the members."""
    a = {"2024-01-31": (pd.DataFrame({"Ticker": ["A", "B"]}), {}), "2024-02-29": (pd.DataFrame({"Ticker": ["A"]}), {})}
    b = {"2024-01-31": (pd.DataFrame({"Ticker": ["A", "B"]}), {}), "2024-02-29": (pd.DataFrame({"Ticker": ["A", "C"]}), {})}
    fa, fb = X.data_fingerprint(a), X.data_fingerprint(b)
    assert fa["stock_months"] == 3 and fb["stock_months"] == 4 and fa["data_hash"] != fb["data_hash"]
    assert X.data_fingerprint(dict(reversed(list(a.items())))) == fa      # order-free


def test_every_ledger_row_says_how_its_comparison_was_measured():
    """Queue item 8a: how a comparison is computed is as much a precondition for comparing two rows
    as the data they ran on, so every row carries ``method_version`` and they all agree."""
    rows = [json.loads(line) for line in X.LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    # the ledger is append-only (rule 2), so the method-1 rows run before 8a stay in it and carry
    # no field at all - that absence *is* version 1. What must hold is that the row the report
    # reads for each id, the newest, was measured the current way.
    assert {r.get("method_version", 1) for r in rows} <= {1, X.METHOD_VERSION}
    newest = {}
    for r in rows:
        newest[r["id"]] = r
    assert newest and all(r.get("method_version") == X.METHOD_VERSION for r in newest.values()), \
        sorted(i for i, r in newest.items() if r.get("method_version") != X.METHOD_VERSION)


def test_an_experiment_row_records_what_its_comparison_covered():
    """The intersection's size and the baseline's reading on it, month by month, so the difference
    can be rebuilt from the row without a rerun. The baseline's own row carries neither."""
    latest = {}
    for line in X.LEDGER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[r["id"]] = r
    assert "n_common_by_month" not in latest["baseline"] and "base_ic_by_month" not in latest["baseline"]
    for e in X.load_registry():
        if e["transform"]["kind"] == "baseline" or e["id"] not in latest:
            continue
        r = latest[e["id"]]
        assert r["n_common_by_month"] and r["base_ic_by_month"], e["id"]
        assert set(r["n_common_by_month"]) == set(r["ic_by_month"]), e["id"]
        assert all(v >= X.MIN_STOCKS_A_MONTH for v in r["n_common_by_month"].values()), e["id"]


def test_the_report_refuses_to_mix_two_ways_of_measuring():
    """A method change is a rerun, not a footnote - the report flags a mixed ledger the way it
    flags mixed data versions, so a reader cannot compare rows measured differently."""
    assert X.method_note([2]) == "Method: every row was measured the same way (version 2)."
    mixed = X.method_note([1, 2])
    assert mixed.startswith("**Method:") and "version 1, 2" in mixed and "Rerun" in mixed
    text = X.RESULTS.read_text(encoding="utf-8")
    assert f"Method: every row was measured the same way (version {X.METHOD_VERSION})." in text
    assert "measured in more than one way" not in text


def test_the_period_before_the_sample_is_fixed_and_judged_by_its_own_rule():
    """Registered 2026-10-15 before any of its months was built (Linnainmaa & Roberts 2018)."""
    assert X.PRESAMPLE == ("2012-01-31", "2019-12-31") and X.PRESAMPLE[1] < X.DISCOVERY[0]
    assert X.SAMPLE == ("2020-01-31", "2026-08-31")
    import yaml
    reg = yaml.safe_load(X.REGISTRY.read_text(encoding="utf-8"))
    assert reg["presample"]["months"] == ["2012-01-31", "2019-12-31"] or \
        [str(m) for m in reg["presample"]["months"]] == ["2012-01-31", "2019-12-31"]

    def res(p_mean, p_p, p_t, all_mean):
        return {"summary": {"presample": {"ic_vs_baseline": {"mean": p_mean, "p": p_p, "t": p_t}},
                            "all": {"ic_vs_baseline": {"mean": all_mean}}}}
    e = {"id": "x"}
    assert X.verdict_presample(e, res(0.04, 0.0001, 3.9, 0.01), 24).endswith("the same direction as 2020-2026")
    assert X.verdict_presample(e, res(0.04, 0.0001, 3.9, -0.01), 24).endswith("the opposite direction to 2020-2026")
    assert X.verdict_presample(e, res(0.02, 0.02, 2.3, 0.01), 24).startswith("p < 0.05 in 2012-2019 only")
    assert X.verdict_presample(e, res(0.0, 0.9, 0.1, 0.01), 24) == "not distinguishable in 2012-2019"
    assert X.verdict_presample(e, {"summary": {}}, 24) == "not run on 2012-2019"
    assert X.verdict_presample({"id": "v", "void": "why"}, res(0.04, 0.0001, 3.9, 0.01), 24) == "void"


def test_a_void_experiment_says_so_instead_of_a_verdict():
    reg = {e["id"]: e for e in X.load_registry()}
    assert reg["sector-relative-composite"]["void"]
    r = {"summary": {"discovery": {"ic_vs_baseline": {"mean": 0, "p": 1}},
                     "holdout": {"ic_vs_baseline": {"mean": 0, "p": 1, "t": 0}}}}
    assert X.verdict(reg["sector-relative-composite"], r, 24).startswith("VOID - ")


def test_months_before_the_sample_never_enter_the_sample_periods():
    months = [d.strftime("%Y-%m-%d") for d in pd.date_range("2012-01-31", "2026-08-31", freq="ME")]
    pm = _pm(list(np.linspace(-0.1, 0.1, len(months))), months)
    s = X.summarise(pm)
    assert s["presample"]["ic"]["n"] == 96
    assert s["discovery"]["ic"]["n"] + s["holdout"]["ic"]["n"] == s["all"]["ic"]["n"] == 80


# --------------------------------------------------------------------------- the bot's rule
RULE = {"target_count": 2, "sell_rank": 3, "max_per_sector": 0, "cash_reserve_pct": 0,
        "cost_bps": 0, "drawdown_halt_pct": 25}


def _month(rows):
    return pd.DataFrame(rows, columns=["Ticker", "Rank", "Sector", "Value_Trap_Flag", "Growth_Trap_Flag"])


def test_the_bot_simulation_keeps_the_hold_band_and_sells_past_it():
    """screener-bot v1.1's plan_rebalance, monthly: a holding is kept while ranked inside the band."""
    inputs = {"2024-01-31": _month([("A", 1, "X", False, False), ("B", 2, "X", False, False), ("C", 3, "X", False, False)]),
              "2024-02-29": _month([("C", 1, "X", False, False), ("A", 3, "X", False, False), ("B", 4, "X", False, False)])}
    fwd = pd.DataFrame({"A": [0.10, 0.0], "B": [0.0, 0.0], "C": [0.0, 0.0]}, index=list(inputs))
    pm = X.simulate_bot(inputs, fwd, RULE)
    assert pm.loc["2024-01-31", "ret"] == pytest.approx(0.05)          # half in A (+10%), half in B
    assert pm.loc["2024-02-29", "held"] == 2                            # A kept at rank 3; B (rank 4) sold, C bought
    assert pm.loc["2024-02-29", "bought"] == 1


def test_the_bot_simulation_skips_flags_and_full_sectors_and_never_rebuys_what_it_just_sold():
    rule = dict(RULE, max_per_sector=1)
    inputs = {"2024-01-31": _month([("T", 1, "X", True, False), ("A", 2, "X", False, False),
                                    ("B", 3, "X", False, False), ("C", 4, "Y", False, False)])}
    fwd = pd.DataFrame({"A": [0.0], "B": [0.0], "C": [0.0], "T": [0.0]}, index=list(inputs))
    pm = X.simulate_bot(inputs, fwd, rule)
    assert pm.loc["2024-01-31", "bought"] == 2                          # A and C: T flagged, B sector full
    rule2 = dict(RULE, target_count=1, sell_rank=1)
    inputs2 = {"2024-01-31": _month([("A", 1, "X", False, False), ("B", 2, "X", False, False)]),
               "2024-02-29": _month([("B", 1, "X", False, False), ("A", 2, "X", False, False)])}
    fwd2 = pd.DataFrame({"A": [0.0, 0.0], "B": [0.0, 0.0]}, index=list(inputs2))
    pm2 = X.simulate_bot(inputs2, fwd2, rule2)
    assert pm2.loc["2024-02-29", "bought"] == 1 and pm2.loc["2024-02-29", "held"] == 1   # sold A, bought B


def test_the_bot_simulation_halts_new_buys_in_a_drawdown_and_charges_costs():
    rule = dict(RULE, target_count=1, sell_rank=1, cost_bps=10)
    inputs = {"2024-01-31": _month([("A", 1, "X", False, False), ("B", 2, "X", False, False)]),
              "2024-02-29": _month([("B", 1, "X", False, False), ("A", 2, "X", False, False)])}
    fwd = pd.DataFrame({"A": [-0.5, 0.0], "B": [0.0, 0.0]}, index=list(inputs))
    pm = X.simulate_bot(inputs, fwd, rule)
    assert pm.loc["2024-01-31", "ret"] == pytest.approx(0.999 * 0.5 - 1)  # 10 bps on the way in
    assert bool(pm.loc["2024-02-29", "halted"]) and pm.loc["2024-02-29", "bought"] == 0
    assert pm.loc["2024-02-29", "held"] == 0                              # the sell still happens


def test_every_verdict_says_how_small_a_difference_it_could_see():
    """Queue item 2 (2026-10-15): 'not distinguishable' is printed next to what it rules out."""
    assert X.detectable(0.01, 1) == pytest.approx((3.0 + 0.84) * 0.01, abs=1e-4)       # |t| >= 3 binds
    assert X.detectable(0.01, 1000) > X.detectable(0.01, 1)                           # Bonferroni binds
    assert X.detectable(None, 27) is None
    assert "; sees 0.0" in X._vs_cell({"mean": 0.001, "t": 0.1, "se": 0.01}, 27)


def test_the_statistics_say_when_months_are_not_independent():
    """Queue item 3 (2026-10-15): beside each plain t, the lag-1 autocorrelation and a Newey-West t."""
    rng = np.random.default_rng(3)
    iid = pd.Series(rng.normal(0.01, 0.1, 400))
    s = X._stats(iid)
    assert abs(s["ac1"]) < 0.15 and s["t_nw"] == pytest.approx(s["t"], rel=0.25)
    e = rng.normal(0, 0.1, 401)
    ar = pd.Series(0.01 + e[1:] + 0.9 * e[:-1])                       # strongly dependent months
    s2 = X._stats(ar)
    assert s2["ac1"] > 0.3 and abs(s2["t_nw"]) < abs(s2["t"])           # the honest t is smaller


def test_the_methodology_page_quotes_the_engine_s_own_summary():
    """Claim ``overview.research_engine`` (2026-10-15): the page states what the engine found, from
    ``summary.json``, which must be exactly what the engine computes from its committed ledger."""
    import yaml
    import run_screener
    latest = {}
    for line in X.LEDGER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[r["id"]] = r
    published = json.loads(X.SUMMARY.read_text(encoding="utf-8"))
    assert published == json.loads(json.dumps(X.summary_for_page(X.load_registry(), latest)))
    text = run_screener.build_screener_overview(
        yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")))
    ic = lambda x: "0.000" if abs(x) < 0.0005 else f"{x:+.3f}"                       # noqa: E731
    assert f"**{ic(published['baseline_ic']['presample'])}** in 2012-2019" in text
    assert f"**{ic(published['baseline_ic']['all'])}** in 2020-2026" in text
    assert f"the {published['alternatives']} alternative weightings registered in advance" in text
    assert "this ranking has not been shown to predict returns" in text
    if published["better"] == 0:
        assert "none of the" in text


# --------------------------------------------------------------------------- the 2026-10-16 audit
def test_the_t_distribution_matches_known_quantiles():
    """Item 7: p-values and critical values are Student's t (no scipy), pinned to textbook values."""
    assert X.t_critical(0.05, 9) == pytest.approx(2.2622, abs=1e-3)
    assert X.t_critical(0.05, 1) == pytest.approx(12.706, abs=1e-2)
    assert X.t_critical(0.05, 37) == pytest.approx(2.0262, abs=1e-3)
    assert X.t_two_sided_p(2.0, 10) == pytest.approx(0.07339, abs=1e-4)
    assert X.t_two_sided_p(1.96, None) == pytest.approx(0.05, abs=1e-3)       # the normal curve
    assert X.t_two_sided_p(0.0, 20) == pytest.approx(1.0)


def test_the_bar_at_38_held_out_months_is_the_exact_t_and_detectable_uses_it():
    crit = X.t_critical(0.05 / 27, 37)
    assert crit == pytest.approx(3.354, abs=0.005)                             # the normal curve says 3.11
    assert X.pass_bar(27, 38) == pytest.approx(crit)
    assert X.pass_bar(27, None) == pytest.approx(3.113, abs=0.005)             # the old normal-curve bar
    assert X.pass_bar(1, 38) == pytest.approx(X.HLZ_T)                         # one test: |t| >= 3 binds
    assert X.detectable(0.01, 27, 38) == pytest.approx(round((crit + X.t_critical(0.40, 37)) * 0.01, 4))
    assert X.detectable(0.01, 27, 38) > X.detectable(0.01, 27)


def test_the_verdicts_read_p_from_the_t_distribution_not_the_stored_normal_p():
    """A row written before 2026-10-16 carries a normal-approximation p; the report recomputes it."""
    st = {"n": 38, "t": 2.0, "p": 0.0455, "mean": 0.02, "se": 0.01}
    assert X.p_value(st) == pytest.approx(X.t_two_sided_p(2.0, 37)) and X.p_value(st) > 0.05
    assert X.p_value({"p": 0.03, "t": 2.2}) == 0.03                           # no month count: kept


def test_a_registered_idea_whose_outcome_was_already_on_file_must_say_so(monkeypatch):
    """Item 2: ``run`` refuses an id whose statistic a committed measurement published before it was
    registered, unless the registry discloses it - checked before any data is read."""
    undisclosed = {"id": "x", "transform": {"kind": "only_category", "category": "size"}}
    disclosed = dict(undisclosed, outcome_known_at_registration=["2020-2026"])
    monkeypatch.setattr(X, "outcome_known_before_registration", lambda e: ["research/measurements/m.json"])
    monkeypatch.setattr(X, "load_registry", lambda: [undisclosed])
    monkeypatch.setattr(X, "load_months", lambda: pytest.fail("the guard must refuse before data is read"))
    with pytest.raises(SystemExit, match="outcome already on file before registration"):
        X.run(["x"])
    X.refuse_known_outcomes([disclosed])                                     # disclosed: allowed
    monkeypatch.setattr(X, "outcome_known_before_registration", lambda e: [])
    X.refuse_known_outcomes([undisclosed])                                   # nothing on file: allowed
    assert X.statistic_keys({"kind": "only_category", "category": "size"}) == [("category_ic", "size")]
    assert X.statistic_keys({"kind": "drop_category", "category": "size"}) == []


def test_the_seven_category_screens_are_the_ones_on_file_before_registration_and_say_so():
    """Against the real history: the guard finds exactly the seven ``only-*`` entries (their category
    ICs were in backtest-v2 measurements committed at 11:31 and 16:55, the registry at 17:13), and
    every entry it finds carries the disclosure."""
    try:
        X.registered_at("baseline")
    except OSError:
        pytest.skip("no git history available")
    reg = X.load_registry()
    found = {e["id"] for e in reg if X.outcome_known_before_registration(e)}
    if not found:
        pytest.skip("shallow history: the measurement commits are not reachable")
    seven = {f"only-{c}" for c in ("valuation", "quality", "growth", "momentum", "risk", "size", "investment")}
    assert found == seven
    assert {e["id"] for e in reg if e.get("outcome_known_at_registration")} == seven
    for e in reg:
        if e["id"] in seven:
            assert e["outcome_known_at_registration"] == ["2020-2026"]


def test_results_md_marks_every_known_outcome_row():
    text = X.RESULTS.read_text(encoding="utf-8")
    for e in X.load_registry():
        mark = f"| {e['id']}{X.KNOWN_MARK} |"
        assert (mark in text) == bool(e.get("outcome_known_at_registration")), e["id"]
    assert "only their 2012-2019\nresult is a blind test" in text or "only their 2012-2019 result is a blind test" in text
    assert "on the same months" not in text and "paired by month" in text


DESCRIBES_THE_PAIRING = (ROOT / "v2_experiments.py", X.REGISTRY, ROOT / "research_page.py",
                         ROOT / "research.html", X.RESULTS,
                         ROOT / "research" / "2026-10-15-research-engine-first-batch.md")


def test_no_surface_still_says_a_narrow_screen_is_judged_on_its_own_stocks():
    """The inverse of the check this replaced, because 8a reversed the fact it guarded.

    Until method version 2 the pairing really was by month *only*, and the audit of 2026-10-16
    rightly made every surface say so - that test forbade the phrase "same stocks" anywhere
    outside a dated correction. Since 2026-10-17 both sides are recomputed on each month's
    intersection, so "same stocks" is now the true description and the *old* wording is the one
    that would mislead. The assertion is inverted rather than deleted: a surface may still carry
    the version-1 wording, but only inside a passage that dates it as history.
    """
    stale = ("pairing is by month only", "paired by month only", "judged on fewer stocks",
             "is on a different, usually smaller, set")
    for p in DESCRIBES_THE_PAIRING:
        lines = p.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if any(s in line.lower() for s in stale):
                window = "\n".join(lines[max(0, i - 6):i + 2])
                assert any(d in window for d in ("2026-10-15", "2026-10-16", "version 1")), (p.name, line)


def test_every_surface_describes_the_pairing_as_on_the_shared_stocks():
    """The positive half: the three generated surfaces a reader actually meets must say that the
    comparison is on the stocks both rankings carry, not merely that it is monthly."""
    for p in (X.RESULTS, ROOT / "research.html", X.REGISTRY):
        text = p.read_text(encoding="utf-8").lower()
        assert any(s in text for s in ("stocks the two rankings share", "stocks both rankings",
                                       "shared stocks", "intersection")), p.name


def test_within_sector_screens_carry_a_construction_note():
    """Item 6: the engine ranks every metric within its sector (``compute_sector_percentiles``), so a
    single-metric screen, and Size on its own, is a within-sector version of a market-wide paper."""
    for e in X.load_registry():
        tr = e["transform"]
        single = tr["kind"] == "metrics" and sum(len(v) for v in tr["metrics"].values()) == 1
        if single or e["id"] in ("only-size", "profitability-and-value"):
            note = (e.get("construction_note") or "").lower()
            assert "within" in note and "sector" in note, e["id"]


# --------------------------------------------------------------------------- provenance (item 8b)
# A result nobody can rebuild is not evidence. These pin what every ledger row has to be able to
# name: the exact code, the exact baseline weighting, and the *contents* - not merely the member
# lists - of the monthly records it ran on. Added 2026-10-18 with plan/research-engine.md item 8b.

def test_every_result_affecting_file_is_named_and_present():
    """The provenance list is the whole claim: if a file that can move a number is missing from it,
    every hash below is of the wrong thing. Both backtest modules, the metric engine, the
    point-in-time record builder, the membership history, the weights and the registry."""
    for rel in X.RESULT_PATHS:
        assert (ROOT / rel).is_file(), rel
    assert {"v2_experiments.py", "backtest_v2.py", "backtest_v2_returns.py", "factor_engine.py",
            "pit_record.py", "universe_history.py", "config.yaml"} <= set(X.RESULT_PATHS)
    assert any(r.endswith("registry.yaml") for r in X.RESULT_PATHS)


def test_a_run_records_the_code_it_ran_on(tmp_path):
    """``git describe --dirty`` plus a content hash per file. The hashes are the part that cannot be
    faked by moving a tag, and a file that is not there is reported missing rather than skipped."""
    fp = X.code_fingerprint()
    assert fp["code_version"]
    assert set(fp["code_hashes"]) == set(X.RESULT_PATHS)
    assert all(re.fullmatch(r"[0-9a-f]{16}", v) for v in fp["code_hashes"].values()), fp["code_hashes"]
    assert re.fullmatch(r"[0-9a-f]{12}", fp["code_hash"])
    empty = X.code_fingerprint(tmp_path)
    assert set(empty["code_hashes"].values()) == {"missing"}
    assert empty["code_hash"] != fp["code_hash"]        # the hash moves with the files it covers
    (tmp_path / "config.yaml").write_text("a: 1", encoding="utf-8")
    assert X.code_fingerprint(tmp_path)["code_hash"] != empty["code_hash"]


def test_the_engines_own_output_does_not_brand_a_row_unreproducible(monkeypatch):
    """Found by running the rerun item 8b asked for (2026-10-18). ``git describe --dirty`` reads the
    *whole* tree, and a run appends to the tracked ``ledger.jsonl`` while ``report`` rewrites
    ``RESULTS.md`` - so by the time the second entry point starts, the tree is dirty because of the
    run itself. It branded ``bot-rule-v1.1`` ``-dirty`` on a rerun where nothing but the engine's
    own output had changed, and no order of ``run`` and ``evaluate`` escapes it.

    So the marker is scoped to ``RESULT_PATHS``, exactly like the refusal: the guard and the marker
    have to mean the same thing, or one of them is lying about the same tree. An engine output is a
    result and cannot change a number."""
    monkeypatch.setattr(X, "edited_result_files", lambda root=X.ROOT: [])
    clean = X.git_describe()
    assert clean and not clean.endswith("-dirty")
    monkeypatch.setattr(X, "edited_result_files", lambda root=X.ROOT: ["config.yaml"])
    assert X.git_describe() == clean + "-dirty"          # and it still fires on a real edit
    # the engine's own artifacts are never in the scope that can mark a row dirty
    for out in ("research/experiments/ledger.jsonl", "research/experiments/RESULTS.md",
                "research/experiments/summary.json", "research.html"):
        assert out not in X.RESULT_PATHS
    # and the marker is never asked of git itself, which is where the whole-tree scope came from
    seen = []

    class _R:
        stdout = "good/2026-10-18-1-gabc1234\n"

    monkeypatch.setattr(X, "edited_result_files", lambda root=X.ROOT: [])
    monkeypatch.setattr(X.subprocess, "run", lambda cmd, **kw: seen.append(cmd) or _R())
    assert X.git_describe() == "good/2026-10-18-1-gabc1234"
    assert seen and "--dirty" not in seen[0], seen[0]


def test_the_engine_refuses_to_run_from_an_edited_tree(monkeypatch):
    """Item 8b: commit before running. A row whose code is "whatever was in the working copy that
    afternoon" cannot be rerun by anyone, including its author, and the commit is the
    pre-registration proof rule 1 rests on."""
    monkeypatch.setattr(X, "edited_result_files", lambda root=X.ROOT: ["config.yaml"])
    with pytest.raises(SystemExit, match="the tree is edited"):
        X.refuse_edited_tree()
    monkeypatch.setattr(X, "edited_result_files", lambda root=X.ROOT: [])
    X.refuse_edited_tree()                                            # a clean tree: allowed
    monkeypatch.undo()
    # the guard is scoped to files that can change a number - an edited test, plan or log must not
    # stop a run, or it would be switched off within a week
    assert all(any(p.endswith(r) for r in X.RESULT_PATHS) for p in X.edited_result_files()), \
        X.edited_result_files()


def test_run_and_evaluate_both_check_the_tree_before_reading_any_data(monkeypatch):
    """Both entry points, and before the months are loaded - the check is worthless after the fact."""
    def boom(root=X.ROOT):
        raise SystemExit("edited tree")

    for fn in (X.run, X.run_evaluations):
        monkeypatch.setattr(X, "refuse_edited_tree", boom)
        monkeypatch.setattr(X, "load_months", lambda: pytest.fail("checked before any data is read"))
        with pytest.raises(SystemExit, match="edited tree"):
            fn([])


def test_the_fingerprint_covers_the_records_contents_not_only_their_members(tmp_path):
    """The crux of item 8b. ``data_fingerprint`` hashes each month's member list, which a rebuild
    can leave identical while every number inside it moves - a metric fixed, filings re-fetched, a
    split corrected. Same tickers, different numbers, has to be a different vintage."""
    a, b = tmp_path / "a", tmp_path / "b"
    for d, value in ((a, 1.0), (b, 2.0)):
        d.mkdir()
        pd.DataFrame({"Ticker": ["A", "B"], "roic": [value, value]}).to_parquet(d / "2024-01-31.parquet")
    months = {"2024-01-31": (pd.DataFrame({"Ticker": ["A", "B"]}), {})}
    assert X.data_fingerprint(months) == X.data_fingerprint(months)          # members: identical
    fa = X.content_fingerprint(tmp_path, metrics_dir=a)
    fb = X.content_fingerprint(tmp_path, metrics_dir=b)
    assert fa["content_hash"] != fb["content_hash"]
    assert fa["content_parts"]["month_file_count"] == 1
    # the price, index, split and filing files the months were built from, and the live run whose
    # static fields they borrow - absent here, so reported missing rather than silently omitted
    assert set(X.DATA_SOURCES) <= set(fa["content_parts"])
    assert {fa["content_parts"][k] for k in X.DATA_SOURCES} == {"missing"}
    assert fa["content_parts"]["base_run"] == "unrecorded"
    assert X.build_record(tmp_path) == {}


def test_the_live_fingerprint_reads_the_real_inputs():
    """Against the real cache: every data source is found and hashed, not quietly missing."""
    if not all((ROOT / rel).is_file() for rel in X.DATA_SOURCES.values()):
        pytest.skip("the v2 cache is not built here (it is gitignored)")
    fp = X.content_fingerprint()
    parts = fp["content_parts"]
    assert parts["month_file_count"] > 100, parts["month_file_count"]
    for name in X.DATA_SOURCES:
        assert re.fullmatch(r"[0-9a-f]{16}", parts[name]), (name, parts[name])
    assert re.fullmatch(r"[0-9a-f]{12}", fp["content_hash"])


def _newest_ledger_rows() -> dict:
    newest = {}
    for line in X.LEDGER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            newest[r["id"]] = r
    return newest


def test_every_ledger_row_the_report_reads_names_its_code_and_its_baseline():
    """What item 8b is for. The ledger is append-only (rule 2), so rows written before 2026-10-18
    carry none of these fields - that absence is the record of a time when the engine could not
    say. What must hold is that the row the report reads for each id, the newest, can."""
    newest = _newest_ledger_rows()
    assert newest
    for i, r in newest.items():
        for k in ("code_version", "code_hash", "content_hash", "base_config_hash"):
            assert r.get(k), (i, k)
        assert not r["code_version"].endswith("-dirty"), i          # the guard above forbids it
    # and they all agree with each other, or they do not compare with each other
    for k in ("code_hash", "content_hash", "base_config_hash"):
        assert len({r[k] for r in newest.values()}) == 1, (k, sorted({r[k] for r in newest.values()}))


def test_the_baseline_weighting_is_hashed_separately_from_the_transformed_one():
    """``config_hash`` covers the transform too, so it differs between experiments by design and
    cannot say whether two rows started from the same ``config.yaml``. ``base_config_hash`` can."""
    cfg0 = X.base_config()
    base_h = X.config_hash(cfg0, {"kind": "baseline"})
    tr = {"kind": "factor_weights", "weights": dict(cfg0["factor_weights"], growth=50)}
    assert X.config_hash(X.apply_transform(cfg0, tr), tr) != base_h
    newest = _newest_ledger_rows()
    assert len({r["config_hash"] for r in newest.values() if "config_hash" in r}) > 1   # per transform
    assert all(r["base_config_hash"] == base_h for r in newest.values()), \
        "the ledger's baseline weighting is not the committed config.yaml - rerun"


def test_a_row_built_on_another_baseline_or_another_vintage_does_not_compare():
    """A change to ``config.yaml`` moves the baseline every row is differenced against, so such a
    row is dropped from the page's count exactly as a row of another data version is."""
    base = {"base_config_hash": "aaa", "content_hash": "bbb"}
    assert X.comparable_provenance(dict(base), base)
    assert not X.comparable_provenance(dict(base, base_config_hash="zzz"), base)
    assert not X.comparable_provenance(dict(base, content_hash="zzz"), base)
    # pre-8b rows: the fields did not exist, so claiming they disagree would be as false as
    # claiming they agree - absent on both sides stays comparable, absent on one side does not
    assert X.comparable_provenance({}, {})
    assert not X.comparable_provenance({}, base)
    assert not X.comparable_provenance(base, {})


def test_the_report_says_what_its_rows_ran_on_and_flags_a_mixture():
    """Flagged the way a mixed data version is, because the remedy is the same: rerun."""
    one = [{"code_version": "good/2026-10-18", "base_config_hash": "aaa", "content_hash": "bbb"}]
    note = X.provenance_note(one)
    assert note.startswith("Provenance: ") and "same code" in note and "good/2026-10-18" in note
    mixed = X.provenance_note(one + [dict(one[0], code_version="good/2026-10-11")])
    assert mixed.startswith("**Provenance:**") and "more than one version of the code" in mixed
    cfg = X.provenance_note(one + [dict(one[0], base_config_hash="ccc")])
    assert "more than one baseline weighting" in cfg and "config.yaml` changed" in cfg
    vintage = X.provenance_note(one + [dict(one[0], content_hash="ccc")])
    assert "contents differ" in vintage
    dirty = X.provenance_note([dict(one[0], code_version="good/2026-10-18-dirty")])
    assert "edited tree" in dirty
    assert X.provenance_note([]) == "Provenance: no runs in the ledger."
    text = X.RESULTS.read_text(encoding="utf-8")
    assert "Provenance: " in text and "**Provenance:**" not in text, "the committed report is mixed"


def test_the_build_record_names_the_live_run_the_months_borrowed_static_fields_from(tmp_path):
    """``Inputs.load`` takes the *newest* run's raw fetch for sector, industry and share class, and
    the data loop writes a new one every weekday - so two months built months apart can disagree
    about a company's sector, which is the peer group the engine ranks within. The build records
    which run it used, per month, and merges rather than replaces because builds are sharded."""
    import backtest_v2 as v2
    p = tmp_path / "_build.json"
    v2.record_build(["2024-01-31", "2024-02-29"], "20260101_000000", p)
    rec = json.loads(p.read_text(encoding="utf-8"))
    assert rec["base_run"] == "20260101_000000" and set(rec["months"]) == {"2024-01-31", "2024-02-29"}
    assert rec["commits"] and all(rec["commits"])
    v2.record_build(["2024-03-31"], "20260202_000000", p)             # a later, separate build
    rec = json.loads(p.read_text(encoding="utf-8"))
    assert len(rec["months"]) == 3                                    # merged, not replaced
    assert rec["base_run"] == "20260101_000000+20260202_000000"       # and it says so


def test_the_shards_build_records_are_merged_not_overwritten(tmp_path):
    """Shards are separate processes; a shared read-modify-write would lose whichever finished
    first, so each writes its own file and ``--merge`` folds them in."""
    import backtest_v2 as v2
    p = tmp_path / "_build.json"
    for k, m in enumerate(("2024-01-31", "2024-02-29", "2024-03-31")):
        v2.record_build([m], "20260101_000000", p.with_name(f"_build.shard{k}.json"))
    v2.merge_build_records(3, p)
    rec = json.loads(p.read_text(encoding="utf-8"))
    assert len(rec["months"]) == 3 and rec["base_run"] == "20260101_000000"
    assert not list(tmp_path.glob("_build.shard*.json"))               # consumed


def test_the_diagnostic_build_records_nothing_because_it_caches_nothing():
    """The look-ahead arm deliberately writes no monthly metrics (``build_month`` guards on
    ``filings_as_of``), so it must not stamp the build record either - that would name a base run
    for months it never wrote."""
    src = (ROOT / "backtest_v2.py").read_text(encoding="utf-8")
    assert "if f_as_of is None:\n                built.append(m)" in src
    assert "if filings_as_of is None:\n        try:\n            save_metrics" in src
