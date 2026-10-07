"""A page's arithmetic must reproduce from the page's own numbers.

WHY THIS EXISTS - 2026-10-07 (``plan/calculation-transparency.md``, stage T0b).

The drilldown prints, for every metric, a percentile and a weight, and for every
category a score. On 2026-10-06, recomputing each category score from the published
payload alone showed that **334 of 4,012 stock-category pairs did not reproduce - 275
of 502 stocks (55%)**. The page printed the *generic* metric weight; the scorer used
bank weights for banks, Piotroski-conditional weights for 220 non-banks, and
renormalised over whatever data a stock had. JPM's Valuation panel showed three
heavily weighted metrics as N/A, called P/B "Inactive" (it is the bank's main
valuation metric), and printed a score no arithmetic on screen could produce.
Separately the composite was shown as the plain sum of category points although a
coverage discount is applied after it (3 of 502 stocks).

Nothing in CI could see it, because nothing recomputed a score from what the page
publishes. These tests do - using ``calc_trace``, which reads only the payload and
shares no code with ``factor_engine``'s scoring.

Run ``research/measurements/2026-10-06-calculation-reproducibility.py`` against the
pre-change tree to see the original numbers: 3,678 of 4,012 reproduced, 500 of 502
composites.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import calc_trace  # noqa: E402
import generate_dashboard as gd  # noqa: E402
from factor_engine import (  # noqa: E402
    CAT_METRICS,
    METRIC_COLS,
    WEIGHT_PROFILE_LABELS,
    _BANK_ONLY_METRICS,
    compute_category_scores,
    compute_composite,
    compute_factor_contributions,
    load_config,
    metric_weight_profiles,
    published_weight_profiles,
)

CATS = list(CAT_METRICS)
PAYLOAD = ROOT / "dashboard_data.js"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


# ---------------------------------------------------------------------------
# A synthetic universe that exercises every weight path
# ---------------------------------------------------------------------------

def _universe(cfg: dict, n: int = 80, seed: int = 7) -> pd.DataFrame:
    """Banks, low-valuation non-banks (Piotroski-conditional), thin-coverage names,
    and ordinary names, with percentile columns already attached."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"Ticker": [f"T{i:03d}" for i in range(n)]})
    df["Sector"] = ["Financials" if i < 12 else "Industrials" for i in range(n)]
    df["_is_bank_like"] = [i < 12 for i in range(n)]
    for m in METRIC_COLS:
        df[m] = rng.normal(size=n)
        df[m + "_pct"] = rng.uniform(0, 100, size=n)
    # Missing data scattered about, so renormalisation is exercised.
    for m in METRIC_COLS:
        holes = rng.random(n) < 0.12
        df.loc[holes, m] = np.nan
        df.loc[holes, m + "_pct"] = np.nan
    # Structurally absent for the wrong stock type, as in the real pipeline.
    for m in _BANK_ONLY_METRICS:
        df.loc[~df["_is_bank_like"], [m, m + "_pct"]] = np.nan
    # A few thin-coverage names: well under the discount threshold.
    thin = df.index[-6:]
    for m in METRIC_COLS[: int(len(METRIC_COLS) * 0.6)]:
        df.loc[thin, [m, m + "_pct"]] = np.nan
    return df


def _scored(cfg: dict) -> pd.DataFrame:
    df = _universe(cfg)
    df = compute_category_scores(df, copy.deepcopy(cfg))
    df = compute_composite(df, copy.deepcopy(cfg))
    return compute_factor_contributions(df, copy.deepcopy(cfg))


def _as_payload(df: pd.DataFrame, cfg: dict) -> dict:
    """What `prepare_dashboard_data` publishes for these rows, in the same shapes."""
    def val(v):
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

    stocks = {}
    for _, row in df.iterrows():
        wp = {c: row[f"_wp_{c}"] for c in CATS if row[f"_wp_{c}"] != "generic"}
        stock = {
            "pct": {m: val(row[m + "_pct"]) for m in METRIC_COLS},
            "raw": {m: val(row[m]) for m in METRIC_COLS},
            "cat_scores": {c: val(row[f"{c}_score"]) for c in CATS},
            "contrib": {c: val(row[f"{c}_contrib"]) for c in CATS},
            "composite": val(row["Composite"]),
            "cov": {"n": int(row["_cov_present"]), "of": int(row["_cov_applicable"])},
        }
        if row["_cov_discount"]:
            stock["cov"]["disc"] = float(row["_cov_discount"])
        if wp:
            stock["wp"] = wp
        stocks[row["Ticker"]] = stock
    weights = {"profiles": published_weight_profiles(cfg),
               "factor_weights": dict(cfg["factor_weights"])}
    return {"weights": weights, "stock_detail": stocks}


# ---------------------------------------------------------------------------
# 1. the weight tables
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cat", CATS)
def test_every_profile_sums_to_one_hundred_percent(cfg, cat):
    for pid, table in metric_weight_profiles(cfg, cat).items():
        assert abs(sum(table.values()) - 1.0) < 1e-9, f"{cat}/{pid} sums to {sum(table.values())}"


def test_generic_profile_is_exactly_the_configured_weights(cfg):
    for cat in CATS:
        generic = metric_weight_profiles(cfg, cat)["generic"]
        for m in CAT_METRICS[cat]:
            assert generic[m] == cfg["metric_weights"].get(cat, {}).get(m, 0) / 100.0


def test_bank_and_piotroski_profiles_exist_and_differ_from_generic(cfg):
    val = metric_weight_profiles(cfg, "valuation")
    assert val["bank"]["pb_ratio"] > 0 == val["generic"]["pb_ratio"], (
        "banks are scored on P/B; the generic weight for it is zero")
    qual = metric_weight_profiles(cfg, "quality")
    assert qual["pio_lowval"]["piotroski_f_score"] == pytest.approx(
        qual["generic"]["piotroski_f_score"] * cfg["piotroski_conditional"]["reduction_factor"])
    assert set(qual) >= {"generic", "bank", "pio_lowval", "pio_gt"}


def test_every_profile_has_a_plain_english_reason():
    assert set(WEIGHT_PROFILE_LABELS) >= {"generic", "bank", "pio_lowval", "pio_gt"}
    assert all(len(v) > 20 for v in WEIGHT_PROFILE_LABELS.values())


# ---------------------------------------------------------------------------
# 2. the engine and the payload-only recomputation agree
# ---------------------------------------------------------------------------

def test_synthetic_universe_exercises_every_weight_path(cfg):
    df = _scored(cfg)
    assert set(df["_wp_valuation"]) == {"generic", "bank"}
    assert {"generic", "bank", "pio_lowval"} <= set(df["_wp_quality"])
    assert (df["_cov_discount"] > 0).any(), "no thin-coverage row - the discount is untested"


def test_category_scores_reproduce_from_the_published_tables(cfg):
    payload = _as_payload(_scored(cfg), cfg)
    worst = 0.0
    for t, s in payload["stock_detail"].items():
        for cat in CATS:
            pub = s["cat_scores"][cat]
            tr = calc_trace.category_trace(payload["weights"], cat, s)
            if pub is None:
                assert tr["score"] is None, f"{t}/{cat}: engine had no score, trace has one"
                continue
            worst = max(worst, abs(pub - tr["score"]))
    # Published weights are rounded to 6 decimal places of a percent, so exact
    # equality is not expected - but 1e-5 is a thousandth of the 4dp the payload
    # stores scores to.
    assert worst < 1e-5, f"payload-only recomputation differs from the engine by {worst}"


def test_composite_and_points_reproduce_including_the_coverage_discount(cfg):
    payload = _as_payload(_scored(cfg), cfg)
    discounted = 0
    for t, s in payload["stock_detail"].items():
        c = calc_trace.composite_trace(payload["weights"], s)
        if c["composite"] is None:
            continue
        assert abs(c["composite"] - s["composite"]) <= 0.0051, (
            f"{t}: recomputed {c['composite']:.4f} vs published {s['composite']:.2f}")
        assert abs(c["published_contrib_sum"] - c["pre_discount"]) <= calc_trace.CONTRIB_SUM_TOL
        discounted += 1 if c["discount"] else 0
    assert discounted >= 1


def test_verify_payload_is_clean_on_the_synthetic_universe(cfg):
    result = calc_trace.verify_payload(_as_payload(_scored(cfg), cfg))
    assert result["failing_stocks"] == 0, result["failures"]


# ---------------------------------------------------------------------------
# 3. negative controls - the check must fire on the original defect
# ---------------------------------------------------------------------------

def test_dropping_the_weight_table_choice_reproduces_the_original_defect(cfg):
    """Remove `wp` (so every stock is read with generic weights) - exactly what the
    page did until 2026-10-07. The recomputation must reject it, for banks and for
    Piotroski-adjusted names."""
    payload = _as_payload(_scored(cfg), cfg)
    for s in payload["stock_detail"].values():
        s.pop("wp", None)
    result = calc_trace.verify_payload(payload)
    assert result["failing_stocks"] > 0
    cats_hit = {p.split(":")[0] for probs in result["failures"].values() for p in probs}
    assert {"valuation", "quality"} <= cats_hit


def test_a_hidden_coverage_discount_is_caught(cfg):
    payload = _as_payload(_scored(cfg), cfg)
    hit = [t for t, s in payload["stock_detail"].items() if s["cov"].get("disc")]
    assert hit
    for t in hit:
        payload["stock_detail"][t]["cov"].pop("disc")
    result = calc_trace.verify_payload(payload)
    assert set(hit) <= set(result["failures"])
    assert any("composite" in p for t in hit for p in result["failures"][t])


def test_a_perturbed_weight_is_caught(cfg):
    payload = _as_payload(_scored(cfg), cfg)
    payload["weights"]["profiles"]["valuation"]["generic"]["earnings_yield"] += 5.0
    assert calc_trace.verify_payload(payload)["failing_stocks"] > 0


# ---------------------------------------------------------------------------
# 4. the build refuses a page that does not reproduce
# ---------------------------------------------------------------------------

def test_build_refuses_a_payload_that_does_not_reproduce(cfg):
    payload = _as_payload(_scored(cfg), cfg)
    for s in payload["stock_detail"].values():
        s.pop("wp", None)
    with pytest.raises(gd.CalculationMismatch):
        gd._reconcile_scores(payload["weights"], payload["stock_detail"], cfg)


def test_build_passes_a_payload_that_does_reproduce(cfg):
    payload = _as_payload(_scored(cfg), cfg)
    gd._reconcile_scores(payload["weights"], payload["stock_detail"], cfg)


def test_build_does_not_guess_when_a_run_has_no_weight_tables(cfg, capsys):
    payload = _as_payload(_scored(cfg), cfg)
    weights = {k: v for k, v in payload["weights"].items() if k != "profiles"}
    gd._reconcile_scores(weights, payload["stock_detail"], cfg)
    assert "not published and not checked" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 5. the cache cannot serve a table the new engine did not build
# ---------------------------------------------------------------------------

def test_scored_cache_key_is_versioned_by_scoring_schema():
    src = (ROOT / "run_context.py").read_text(encoding="utf-8")
    assert '"scoring_schema"' in src, (
        "the factor_scores cache is keyed by config alone; a warm start after an engine "
        "change served scores without the new columns on 2026-10-07")


# ---------------------------------------------------------------------------
# 6. the live payload
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def live() -> dict:
    if not PAYLOAD.exists():
        pytest.skip("dashboard_data.js not present")
    text = PAYLOAD.read_text(encoding="utf-8", errors="replace")
    return json.loads(text[text.find("{"):text.rfind("}") + 1])


def test_live_payload_reproduces_every_category_score_and_composite(live):
    """The original failure: 334 of 4,012 pairs and 3 of 502 composites."""
    result = calc_trace.verify_payload(live)
    assert result["category_pairs"] > 3900
    assert result["failing_stocks"] == 0, list(result["failures"].items())[:5]


def test_live_payload_carries_the_tables_and_each_stocks_choice(live):
    w = live["weights"]
    assert set(w["profiles"]) == set(CATS)
    assert set(w["profiles"]["quality"]) >= {"generic", "bank", "pio_lowval"}
    banks = [s for s in live["stock_detail"].values() if s["flags"]["is_bank"]]
    assert banks and all(s["wp"]["valuation"] == "bank" for s in banks)
    assert any(s.get("wp", {}).get("quality") == "pio_lowval"
               for s in live["stock_detail"].values())
    assert w["coverage_discount"]["threshold"] == pytest.approx(0.8)


def test_every_weighted_metric_is_published(live):
    pct_keys = set(next(iter(live["stock_detail"].values()))["pct"])
    for cat, profiles in live["weights"]["profiles"].items():
        for pid, table in profiles.items():
            missing = [m for m, w in table.items() if w > 0 and m not in pct_keys]
            assert not missing, f"{cat}/{pid} weights metrics the payload does not publish: {missing}"


def test_weight_choice_and_coverage_never_enter_scored_fields(live):
    s = next(iter(live["stock_detail"].values()))
    for key in ("wp", "cov"):
        assert key not in s["raw"] and key not in s["pct"]
    src = (ROOT / "factor_engine.py").read_text(encoding="utf-8")
    assert "_wp_" not in src.split("METRIC_COLS = ")[1].split("]")[0]


# ---------------------------------------------------------------------------
# the side-by-side view's gap arithmetic (claims.py: compare.composite_gap)
# The compare view (2026-10-07, owner-run UI pass 2) takes the composite gap between two
# stocks apart into category points. These hold the payload to that account.
# ---------------------------------------------------------------------------

_GAP_CATS = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]


def _points(s):
    return sum(s["contrib"].get(c) or 0.0 for c in _GAP_CATS)


def test_points_add_up_to_the_composite_for_every_undiscounted_stock(live):
    """The gap view shows one line per category and, only for a discounted pair, a discount line.
    That is a complete account of the gap only if every undiscounted stock's points add up to
    its composite within the rounding the view admits (and never show as a phantom discount)."""
    bad = []
    for t, s in live["stock_detail"].items():
        if s.get("composite") is None or (s.get("cov") or {}).get("disc"):
            continue
        if abs(_points(s) - s["composite"]) >= 0.05:
            bad.append((t, s["composite"], round(_points(s), 3)))
    assert not bad, bad[:5]


def test_a_discounted_stocks_residual_is_its_coverage_discount(live):
    n = 0
    for t, s in live["stock_detail"].items():
        disc = (s.get("cov") or {}).get("disc")
        if not disc or s.get("composite") is None:
            continue
        expected = _points(s) * (1 - disc)
        assert abs(expected - s["composite"]) < 0.05, (t, s["composite"], expected)
        n += 1
    if n == 0:
        pytest.skip("no discounted stock in this payload")


def test_the_gap_lines_reconcile_for_every_pair_with_the_top_stock(live):
    """What the page prints: per-category point differences, plus a discount line where one of
    the two is discounted, sum to the composite gap."""
    stocks = {t: s for t, s in live["stock_detail"].items() if s.get("composite") is not None}
    top = min(stocks, key=lambda t: stocks[t]["rank"])
    a = stocks[top]
    for t, b in stocks.items():
        gap = a["composite"] - b["composite"]
        lines = sum((a["contrib"].get(c) or 0) - (b["contrib"].get(c) or 0) for c in _GAP_CATS)
        resid = gap - lines
        discounted = bool((a.get("cov") or {}).get("disc") or (b.get("cov") or {}).get("disc"))
        if not discounted:
            assert abs(resid) < 0.1, (t, gap, lines)
