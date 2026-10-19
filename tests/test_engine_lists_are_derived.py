"""Lists of metrics that used to be typed out by hand are now read from the engine.

Found 2026-10-16 (METHODOLOGY_CHANGELOG.md 2026-10-16). Four places carried their own copy of
"which metrics count", and every copy had gone stale against ``config.yaml``:

* the page's **Metric Coverage** figure averaged ``_metric_count / _metric_total``, a hard-coded
  18-metric list that counted unscored D/E, left out ten weighted metrics and counted EV and ROIC
  against banks - **95.5%** published against the engine's own **98.9%** (banks 69% against
  99.5%). It is now the mean of the engine's ``cov.n / cov.of`` (claim
  ``drilldown.avg_metric_coverage``);
* the NaN **auto-reduce** checked a dict that omitted seven weighted metrics, so an outage in any
  of them (``fy1_revision_3m``, the heaviest Revisions metric, among them) went uncaught;
* the **Revisions auto-disable** counted the weight-0 ``earnings_acceleration`` and omitted
  ``fy1_revision_3m``;
* the **universe-filter** log compared the post-filter ticker set with itself, so it never named
  an excluded stock.

None of them changed a score on the 2026-10-16 run (no weighted metric is over 70% missing;
Revisions coverage is 99.2% on the weighted set against 99.0% on the old list). One source now:
``factor_engine.weighted_metric_sets``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import factor_engine as fe
import generate_dashboard as gd
import run_screener as rs

ROOT = Path(__file__).resolve().parent.parent
PAYLOAD = ROOT / "dashboard_data.js"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def payload() -> dict:
    if not PAYLOAD.exists():
        pytest.skip("dashboard_data.js not present")
    text = PAYLOAD.read_text(encoding="utf-8", errors="replace")
    return json.loads(text[text.find("{"):text.rfind("}") + 1])


# ---------------------------------------------------------------------------
# 1. the page's Metric Coverage figure is the engine's coverage, averaged
# ---------------------------------------------------------------------------

def test_metric_coverage_is_the_mean_of_the_published_per_stock_coverage(payload):
    dq = payload["data_quality"]
    if dq.get("coverage_basis") != "weighted":
        pytest.skip("published payload predates 2026-10-16 (legacy 18-metric average); "
                    "the rehearsal and the next data run check it")
    covs = [s["cov"] for s in payload["stock_detail"].values() if s.get("cov") and s["cov"]["of"]]
    assert covs
    expected = round(sum(c["n"] / c["of"] for c in covs) / len(covs), 3)
    assert dq["avg_metric_coverage"] == expected


def test_the_figure_is_built_from_the_engines_columns_not_a_list():
    df = pd.DataFrame({"_cov_present": [27, 25, 20], "_cov_applicable": [28, 25, 25],
                       # a stale legacy count must not be read
                       "_metric_count": [1, 1, 1], "_metric_total": [18, 18, 18]})
    assert gd.average_weighted_coverage(df) == round((27 / 28 + 1 + 20 / 25) / 3, 3)
    assert gd.average_weighted_coverage(pd.DataFrame({"x": [1]})) is None


def test_the_provenance_count_is_the_discount_coverage(cfg):
    """`_metric_count` / `_metric_total` (Excel's Metrics Available / Total, and what the
    methodology page calls the provenance fields) are the same numbers the discount reads."""
    n = 6
    df = pd.DataFrame({"Ticker": [f"T{i}" for i in range(n)], "_is_bank_like": [False] * n})
    gen, _ = fe.weighted_metric_sets(cfg)
    for m in fe.METRIC_COLS:
        df[m] = 1.0
    df.loc[0, sorted(gen)[0]] = np.nan
    for c in fe.CAT_METRICS:
        df[f"{c}_score"] = 50.0
    out = fe.compute_composite(df, cfg)
    assert (out["_metric_count"] == out["_cov_present"]).all()
    assert (out["_metric_total"] == out["_cov_applicable"]).all()
    assert out.loc[0, "_metric_count"] == len(gen) - 1


# ---------------------------------------------------------------------------
# 2. the auto-reduce watches exactly the metrics that carry weight
# ---------------------------------------------------------------------------

def test_auto_reduce_candidates_are_exactly_the_weighted_set(cfg):
    cands = rs.auto_reduce_candidates(cfg)
    flat = [m for ms in cands.values() for m in ms]
    assert len(flat) == len(set(flat))
    assert set(flat) == fe.weighted_metric_sets(cfg)[0]
    for cat, ms in cands.items():
        assert set(ms) <= set(fe.CAT_METRICS[cat])


@pytest.mark.parametrize("metric", ["fy1_revision_3m", "jensens_alpha", "max_drawdown_1y",
                                    "net_debt_to_ebitda", "beneish_m_score", "revenue_cagr_3yr",
                                    "short_interest_ratio"])
def test_an_outage_in_a_previously_unwatched_metric_is_caught(cfg, metric):
    import copy
    c = copy.deepcopy(cfg)
    cat = next(k for k, ms in fe.CAT_METRICS.items() if metric in ms)
    if c["metric_weights"][cat].get(metric, 0) <= 0:
        pytest.skip(f"{metric} carries no weight today")
    n = 100
    df = pd.DataFrame({"Ticker": [f"T{i}" for i in range(n)]})
    for m in fe.CAT_METRICS[cat]:
        df[m] = 1.0
    df[metric] = [np.nan] * 90 + [1.0] * 10
    before = sum(c["metric_weights"][cat].values())
    rs._auto_reduce_high_nan_metrics(df, c)
    assert c["metric_weights"][cat][metric] == 0
    assert sum(c["metric_weights"][cat].values()) == pytest.approx(before, abs=0.5)


# ---------------------------------------------------------------------------
# 3. the Revisions auto-disable reads weighted coverage
# ---------------------------------------------------------------------------

def test_revisions_disable_reads_the_weighted_metrics(cfg):
    weighted = rs.weighted_category_metrics(cfg, "revisions")
    assert set(weighted) == {m for m in fe.CAT_METRICS["revisions"]
                             if cfg["metric_weights"]["revisions"].get(m, 0) > 0}
    assert "fy1_revision_3m" in weighted
    assert "earnings_acceleration" not in weighted       # weight 0 since 2026-10-09


def test_a_weight_zero_metric_cannot_hold_the_category_up(cfg):
    """Every weighted Revisions metric empty, a weight-0 one full: coverage is 0, not 1/N."""
    n = 50
    df = pd.DataFrame({m: [np.nan] * n for m in fe.CAT_METRICS["revisions"]})
    df["earnings_acceleration"] = 1.0
    df["analyst_rating"] = 1.0
    assert rs.weighted_coverage_pct(df, cfg, "revisions") == 0.0


def test_the_heaviest_revisions_metric_counts(cfg):
    """fy1_revision_3m alone missing: coverage falls by its share of the weighted set."""
    n = 50
    weighted = rs.weighted_category_metrics(cfg, "revisions")
    df = pd.DataFrame({m: [1.0] * n for m in fe.CAT_METRICS["revisions"]})
    df["fy1_revision_3m"] = np.nan
    assert rs.weighted_coverage_pct(df, cfg, "revisions") == pytest.approx(
        100 * (len(weighted) - 1) / len(weighted))


def test_the_pipeline_calls_the_derived_helpers():
    import inspect
    src = inspect.getsource(rs)
    # Since 2026-10-19 both auto-disables go through one function (claim overview.auto_disable),
    # which reads the derived coverage.
    assert 'auto_disable_sparse_category(df, cfg, "revisions")' in src
    assert 'auto_disable_sparse_category(df, cfg, "investment")' in src
    assert "weighted_coverage_pct(df, cfg, cat)" in inspect.getsource(rs.auto_disable_sparse_category)
    assert '"earnings_acceleration", "consecutive_beat_streak"]' not in src, (
        "a hand-typed Revisions list is back in run_screener.py")


# ---------------------------------------------------------------------------
# 4. the universe-filter log names the excluded stocks
# ---------------------------------------------------------------------------

def test_the_universe_filter_log_compares_before_with_after():
    import inspect
    src = inspect.getsource(rs)
    assert 'set(df["Ticker"].tolist()) ^ set(df["Ticker"].tolist())' not in src
    assert "pre_filter_tickers - set(df[\"Ticker\"].tolist())" in src
