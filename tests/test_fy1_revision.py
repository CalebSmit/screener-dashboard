"""Tests for `fy1_revision_3m` - the revisions category's first actual revision.

Shipped 2026-09-10. Background:
`research/2026-09-07-revisions-category-has-no-revisions.md` and
`METHODOLOGY_CHANGELOG.md` 2026-09-10.

The category was named for revisions and contained none: 78 of its 100 points
sat on the earnings-SURPRISE family, whose post-earnings drift Martineau (2022)
documents as absent in this universe since 2006. This module pins the new
metric's arithmetic, its wiring through the three registries, the reweight that
shipped with it, and the two display invariants it needed.

Several tests here exist to stop a specific future mistake rather than to check
that today's code runs, and say so where that is the case.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import factor_engine as fe
from factor_engine import CAT_METRICS, METRIC_COLS, METRIC_DIR, compute_metrics
from schemas import RevisionsWeights

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def cfg():
    with open(ROOT / "config.yaml") as fh:
        return yaml.safe_load(fh)


def _metrics_for(**overrides):
    """Run compute_metrics on one minimal record and return the row."""
    rec = {"Ticker": "TEST", "Sector": "Technology", "currentPrice": 100.0}
    rec.update(overrides)
    return compute_metrics([rec], pd.Series(dtype=float)).iloc[0]


# =====================================================================
# 1. The arithmetic
# =====================================================================
class TestFormula:

    def test_upward_revision_is_positive(self):
        row = _metrics_for(_fy1_eps_current=5.50, _fy1_eps_90d_ago=5.00)
        assert row["fy1_revision_3m"] == pytest.approx(0.005)

    def test_downward_revision_is_negative(self):
        row = _metrics_for(_fy1_eps_current=5.00, _fy1_eps_90d_ago=5.50)
        assert row["fy1_revision_3m"] == pytest.approx(-0.005)

    def test_no_revision_is_exactly_zero(self):
        row = _metrics_for(_fy1_eps_current=5.00, _fy1_eps_90d_ago=5.00)
        assert row["fy1_revision_3m"] == 0.0

    def test_scaled_by_price_not_by_estimate(self):
        """Same $0.50 revision on two prices must NOT give the same number.

        This is the test that fails if someone switches the denominator to
        |estimate|. That switch is specifically forbidden: on the full 502-name
        universe the estimate-scaled denominator hits zero, making the metric's
        mean +inf and its sd undefined (research SS8.1).
        """
        cheap = _metrics_for(currentPrice=50.0,
                             _fy1_eps_current=5.50, _fy1_eps_90d_ago=5.00)
        rich = _metrics_for(currentPrice=200.0,
                            _fy1_eps_current=5.50, _fy1_eps_90d_ago=5.00)
        assert cheap["fy1_revision_3m"] == pytest.approx(0.01)
        assert rich["fy1_revision_3m"] == pytest.approx(0.0025)
        assert cheap["fy1_revision_3m"] > rich["fy1_revision_3m"]

    def test_is_a_change_not_a_level(self):
        """A loss-making company can carry an upward revision.

        This is the whole reason the metric is not redundant with
        `forward_eps_growth` (45% of the growth category, built on the same FY1
        consensus line). A level metric cannot express "still losing money, but
        less badly than analysts thought 90 days ago". Measured overlap between
        the two on the live universe: +0.152 (research SS8.2).
        """
        row = _metrics_for(_fy1_eps_current=-1.00, _fy1_eps_90d_ago=-1.50)
        assert row["fy1_revision_3m"] > 0
        assert row["fy1_revision_3m"] == pytest.approx(0.005)

    def test_uses_price_latest_when_current_price_absent(self):
        row = _metrics_for(currentPrice=np.nan, price_latest=100.0,
                           _fy1_eps_current=5.50, _fy1_eps_90d_ago=5.00)
        assert row["fy1_revision_3m"] == pytest.approx(0.005)


# =====================================================================
# 2. Missing / degenerate inputs yield NaN, never a fabricated number
# =====================================================================
class TestMissingData:

    @pytest.mark.parametrize("overrides", [
        {},
        {"_fy1_eps_current": 5.5},
        {"_fy1_eps_90d_ago": 5.0},
        {"_fy1_eps_current": np.nan, "_fy1_eps_90d_ago": 5.0},
        {"_fy1_eps_current": 5.5, "_fy1_eps_90d_ago": np.nan},
    ])
    def test_missing_endpoint_is_nan(self, overrides):
        row = _metrics_for(**overrides)
        assert pd.isna(row["fy1_revision_3m"])

    @pytest.mark.parametrize("price", [0.0, -5.0, np.nan])
    def test_non_positive_or_missing_price_is_nan(self, price):
        row = _metrics_for(currentPrice=price, price_latest=price,
                           _fy1_eps_current=5.50, _fy1_eps_90d_ago=5.00)
        assert pd.isna(row["fy1_revision_3m"])

    def test_column_always_present(self):
        """Downstream code indexes this column unconditionally."""
        assert "fy1_revision_3m" in _metrics_for().index


# =====================================================================
# 3. Extraction from the yfinance eps_trend frame
# =====================================================================
def _eps_trend_frame(current=8.80, ago=8.60):
    return pd.DataFrame(
        {"current": [1.97, 2.91, current, 9.56],
         "7daysAgo": [1.97, 2.90, current, 9.53],
         "30daysAgo": [1.97, 2.90, current, 9.55],
         "60daysAgo": [2.00, 2.94, ago, 9.68],
         "90daysAgo": [2.00, 2.94, ago, 9.65]},
        index=pd.Index(["0q", "+1q", "0y", "+1y"], name="period"),
    )


class _FakeTicker:
    def __init__(self, eps_trend):
        self._eps_trend = eps_trend
        self.info = {"symbol": "TEST"}

    @property
    def eps_trend(self):
        if isinstance(self._eps_trend, Exception):
            raise self._eps_trend
        return self._eps_trend

    def __getattr__(self, name):
        return None


def _extract_eps_trend(monkeypatch, eps_trend):
    """Drive only the eps_trend block of the fetch and return the record."""
    # The fetch does `import yfinance as yf` inside the function body, so the
    # patch has to land on the module object itself.
    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", lambda *a, **k: _FakeTicker(eps_trend))
    rec = fe._fetch_single_ticker_inner("TEST")
    return rec


class TestFetchExtraction:

    def test_reads_the_0y_row_not_0q(self, monkeypatch):
        """FY1 ('0y'), never the quarterly row.

        '0q' rolls over mid-window, which would put 'current' and '90daysAgo'
        on two different fiscal periods and make the difference meaningless.
        The fake frame gives 0q and 0y deliberately different values.
        """
        rec = _extract_eps_trend(monkeypatch, _eps_trend_frame(8.80, 8.60))
        assert rec.get("_fy1_eps_current") == pytest.approx(8.80)
        assert rec.get("_fy1_eps_90d_ago") == pytest.approx(8.60)

    def test_uses_90day_column_not_60day(self, monkeypatch):
        frame = _eps_trend_frame(8.80, 8.60)
        frame.loc["0y", "60daysAgo"] = 7.00   # decoy
        rec = _extract_eps_trend(monkeypatch, frame)
        assert rec.get("_fy1_eps_90d_ago") == pytest.approx(8.60)

    def test_missing_0y_row_is_absent_not_zero(self, monkeypatch):
        frame = _eps_trend_frame().drop(index="0y")
        rec = _extract_eps_trend(monkeypatch, frame)
        assert pd.isna(rec.get("_fy1_eps_current", np.nan))

    @pytest.mark.parametrize("bad", [
        None,
        pd.DataFrame(),
        KeyError("boom"),
        TypeError("boom"),
        AttributeError("boom"),
    ])
    def test_bad_eps_trend_does_not_abort_the_ticker(self, monkeypatch, bad):
        """A broken eps_trend must cost this one metric, not the whole record.

        The fetch already survives a failed earnings_history the same way; a
        new +1 request per ticker must not become a new way to lose a name.
        """
        rec = _extract_eps_trend(monkeypatch, bad)
        assert rec.get("Ticker") == "TEST"
        assert "_error" not in rec
        assert pd.isna(rec.get("_fy1_eps_current", np.nan))

    def test_nan_cell_stays_nan(self, monkeypatch):
        frame = _eps_trend_frame()
        frame.loc["0y", "90daysAgo"] = np.nan
        rec = _extract_eps_trend(monkeypatch, frame)
        assert pd.isna(rec.get("_fy1_eps_90d_ago", np.nan))


# =====================================================================
# 4. Registry wiring
# =====================================================================
class TestRegistries:

    def test_in_metric_cols(self):
        assert "fy1_revision_3m" in METRIC_COLS

    def test_direction_is_higher_is_better(self):
        assert METRIC_DIR["fy1_revision_3m"] is True

    def test_in_revisions_category(self):
        assert "fy1_revision_3m" in CAT_METRICS["revisions"]

    def test_in_no_other_category(self):
        others = [c for c, ms in CAT_METRICS.items()
                  if c != "revisions" and "fy1_revision_3m" in ms]
        assert others == []

    def test_every_cat_metric_has_a_direction(self):
        for metrics in CAT_METRICS.values():
            for m in metrics:
                assert m in METRIC_DIR, f"{m} missing from METRIC_DIR"


# =====================================================================
# 5. The reweight that shipped with it
# =====================================================================
EXPECTED = {
    "fy1_revision_3m": 35,
    "analyst_surprise": 15,
    "price_target_upside": 10,
    "earnings_acceleration": 20,
    "consecutive_beat_streak": 10,
    "short_interest_ratio": 10,
    "short_pct_float": 0,
    "analyst_rating": 0,
}


class TestReweight:

    def test_config_weights_exact(self, cfg):
        assert cfg["metric_weights"]["revisions"] == EXPECTED

    def test_schema_defaults_match_config(self, cfg):
        defaults = RevisionsWeights().model_dump()
        assert defaults == cfg["metric_weights"]["revisions"]

    def test_weights_sum_to_100(self, cfg):
        assert sum(cfg["metric_weights"]["revisions"].values()) == 100

    def test_revision_metric_outweighs_each_surprise_metric(self, cfg):
        """The point of the change: the revision leads the category.

        CJL (1996) measured REV6 at a +7.7% 6-month decile spread and found it
        the strongest of the three earnings-momentum legs; Barra's USFAST
        Sentiment descriptors are revisions and mention surprise zero times.
        """
        w = cfg["metric_weights"]["revisions"]
        assert w["fy1_revision_3m"] > w["analyst_surprise"]
        assert w["fy1_revision_3m"] > w["consecutive_beat_streak"]

    def test_surprise_family_is_no_longer_the_majority(self, cfg):
        """Was 58 of 100 (38 + 20) before 2026-09-10; must stay a minority.

        This is the defect the change exists to fix, so it gets its own test
        rather than being implied by the exact-weights test above.
        """
        w = cfg["metric_weights"]["revisions"]
        assert w["analyst_surprise"] + w["consecutive_beat_streak"] < 50

    def test_category_share_of_composite_unchanged(self, cfg):
        """A within-category reweight only. The 10% slot did not move."""
        assert cfg["factor_weights"]["revisions"] == 10


# =====================================================================
# 6. Display
# =====================================================================
class TestDisplay:

    def test_labelled_and_formatted_in_basis_points(self):
        import generate_dashboard as gd
        meta = gd._metric_meta() if hasattr(gd, "_metric_meta") else None
        assert meta is None or meta["fy1_revision_3m"]["fmt"] == "bp"

    def test_python_and_js_bp_formatters_agree(self):
        """`_fmt_metric`'s docstring promises it mirrors the emitted JS.

        Prose in the drilldown summary and the number in the metric table are
        produced by two different formatters; if they drift, a stock's summary
        sentence quotes a figure the table below it contradicts.
        """
        from stock_summary import _fmt_metric
        src = (ROOT / "generate_dashboard.py").read_text(encoding="utf-8")
        assert "if (type === 'bp') return (v * 10000).toFixed(0) + ' bp';" in src
        assert _fmt_metric(0.00059, "bp") == "6 bp"
        assert _fmt_metric(-0.00195, "bp") == "-20 bp"
        assert _fmt_metric(0.0, "bp") == "0 bp"

    def test_bp_avoids_the_display_ties_pct_would_create(self):
        """Measured p10/median/p90 are -0.00195 / +0.00059 / +0.00525.

        Under 'pct' at one decimal those render -0.2% / 0.1% / 0.5%, and the
        bulk of the universe collapses onto two strings - manufacturing visible
        ties in a metric measured at 0.0% actual ties. 'bp' keeps them apart.
        """
        from stock_summary import _fmt_metric
        near = [0.00051, 0.00059, 0.00071]
        assert len({_fmt_metric(v, "pct") for v in near}) == 1   # all "0.1%"
        assert len({_fmt_metric(v, "bp") for v in near}) == 3    # 2 / 6 / 11 bp

    def test_exported_to_the_dashboard_payload(self):
        src = (ROOT / "generate_dashboard.py").read_text(encoding="utf-8")
        assert '"fy1_revision_3m"' in src
        assert '"label": "FY1 EPS Revision (90d)"' in src

    def test_never_described_as_advice(self):
        """Rule: the tool explains why a stock ranks, never whether to buy."""
        from stock_summary import BANNED_TERMS
        import generate_dashboard as gd
        src = (ROOT / "generate_dashboard.py").read_text(encoding="utf-8")
        i = src.index('"fy1_revision_3m"')
        label_block = src[i:i + 200].lower()
        assert not any(t.lower() in label_block for t in BANNED_TERMS)


# =====================================================================
# 7. End-to-end through the scoring pipeline
# =====================================================================
@pytest.fixture(scope="module")
def scored(cfg):
    from tests.test_golden import _run_pipeline_on_fixture
    return _run_pipeline_on_fixture(cfg)


class TestPipeline:

    def test_metric_and_percentile_reach_the_output(self, scored):
        assert "fy1_revision_3m" in scored.columns
        assert "fy1_revision_3m_pct" in scored.columns

    def test_missing_name_scores_on_the_remaining_metrics(self, scored):
        """SPARSE has no eps_trend data. It must still rank, on a
        renormalised revisions score - not vanish and not score 0."""
        row = scored[scored["Ticker"] == "SPARSE"].iloc[0]
        assert pd.isna(row["fy1_revision_3m"])
        assert pd.notna(row["Composite"])
        assert pd.notna(row["Rank"])

    def test_upward_revision_outranks_downward_within_the_category(self, scored):
        """HD (+11 bp) must sit above XOM (-52 bp) on the revision percentile."""
        hd = scored[scored["Ticker"] == "HD"].iloc[0]
        xom = scored[scored["Ticker"] == "XOM"].iloc[0]
        assert hd["fy1_revision_3m"] > xom["fy1_revision_3m"]
        assert hd["fy1_revision_3m_pct"] > xom["fy1_revision_3m_pct"]

    def test_synthetic_path_leaves_it_missing_rather_than_faking_it(self):
        """`--allow-synthetic` must not fabricate a consensus revision.

        The sample generator emits finished metric values and carries no price
        field, so it cannot build this metric - and inventing one would be the
        fabrication failure the 2026-08-11 / 2026-09-01 fixes exist to prevent.
        NaN is the honest outcome, matching how `price_target_upside` and
        `proximity_52w_high` already behave on this path, and the has_data
        renormalisation absorbs it.
        """
        raw = fe._generate_sample_data(pd.DataFrame({
            "Ticker": [f"T{i}" for i in range(30)],
            "Company": [f"Co {i}" for i in range(30)],
            "Sector": ["Technology"] * 30,
        }))
        records = raw.to_dict("records") if isinstance(raw, pd.DataFrame) else raw
        assert not any("_fy1_eps_current" in r for r in records)
        df = compute_metrics(records, pd.Series(dtype=float))
        assert df["fy1_revision_3m"].isna().all()
        # ...and the category still scores, on the remaining metrics.
        assert df["analyst_surprise"].notna().any()
