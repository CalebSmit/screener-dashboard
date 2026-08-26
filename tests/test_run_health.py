"""Tests for scripts/check_run_health.py - the guard against publishing a
degraded run.

Each scenario here is a real incident from 2026-08-06..10, when the screener
reported "0 issues logged, 0 fetch failures" while publishing data that had no
prices, no analyst targets, and 25-36% less dispersion than normal.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import check_run_health as h  # noqa: E402


def _payload(n=100, price=True, target=True, spread=1.0):
    """Build a dashboard payload. `spread` scales category dispersion."""
    detail, table = {}, []
    for i in range(n):
        t = f"T{i:03d}"
        score = 50 + (i - n / 2) * spread * (100.0 / n)
        detail[t] = {
            "price": 10.0 + i if price else None,
            "pt_mean": 12.0 + i if target else None,
        }
        table.append({
            "Ticker": t,
            **{f"{c}_score": score for c in h.CATEGORIES},
        })
    return {"stock_detail": detail, "table_data": table}


def _history(tmp_path, monkeypatch, value=25.0, rows=8):
    p = tmp_path / "dispersion_history.csv"
    lines = [f"2026-07-{d:02d}," + ",".join([str(value)] * len(h.CATEGORIES))
             for d in range(1, rows + 1)]
    p.write_text("\n".join(lines), encoding="utf-8")
    monkeypatch.setattr(h, "DISPERSION_HISTORY", p)
    return p


class TestFetchEvidence:
    def test_missing_raw_fetch_fails(self, tmp_path):
        """The 08-07 / 08-10 failure: warm-started from cache, never fetched."""
        res = h.Result()
        h.check_fetch_happened(tmp_path, res)
        assert not res.healthy
        assert "no evidence of a live fetch" in res.failures[0]

    def test_present_raw_fetch_passes(self, tmp_path):
        for f in h.FETCH_EVIDENCE:
            (tmp_path / f).write_bytes(b"x")
        res = h.Result()
        h.check_fetch_happened(tmp_path, res)
        assert res.healthy


class TestCoverage:
    def test_zero_prices_fails(self):
        """0 of 503 stocks had a price on 08-10."""
        res = h.Result()
        h.check_coverage(_payload(price=False), res)
        assert not res.healthy
        assert any("have a price" in f for f in res.failures)

    def test_zero_targets_fails(self):
        res = h.Result()
        h.check_coverage(_payload(target=False), res)
        assert not res.healthy
        assert any("analyst target" in f for f in res.failures)

    def test_full_coverage_passes(self):
        res = h.Result()
        h.check_coverage(_payload(), res)
        assert res.healthy

    def test_partial_analyst_coverage_tolerated(self):
        """Analyst coverage is genuinely patchy; only near-total absence is a fault."""
        p = _payload(n=100)
        for i, v in enumerate(p["stock_detail"].values()):
            if i >= 60:
                v["pt_mean"] = None
        res = h.Result()
        h.check_coverage(p, res)
        assert res.healthy

    def test_empty_detail_fails(self):
        res = h.Result()
        h.check_coverage({"stock_detail": {}}, res)
        assert not res.healthy


class TestDispersionRegression:
    def test_collapse_fails(self, tmp_path, monkeypatch):
        """The check that would have caught 08-07 automatically."""
        _history(tmp_path, monkeypatch, value=25.0)
        res = h.Result()
        h.check_dispersion(_payload(spread=0.5), res)  # ~50% of normal spread
        assert not res.healthy
        assert any("dispersion collapsed" in f for f in res.failures)

    def test_normal_dispersion_passes(self, tmp_path, monkeypatch):
        p = _payload(spread=1.0)
        import statistics
        actual = statistics.pstdev([r["valuation_score"] for r in p["table_data"]])
        _history(tmp_path, monkeypatch, value=actual)
        res = h.Result()
        h.check_dispersion(p, res)
        assert res.healthy

    def test_mild_drift_tolerated(self, tmp_path, monkeypatch):
        """Ordinary market drift must not trip the guard."""
        p = _payload(spread=1.0)
        import statistics
        actual = statistics.pstdev([r["valuation_score"] for r in p["table_data"]])
        _history(tmp_path, monkeypatch, value=actual * 1.10)  # 10% drop
        res = h.Result()
        h.check_dispersion(p, res)
        assert res.healthy

    def test_short_history_skips_check(self, tmp_path, monkeypatch):
        _history(tmp_path, monkeypatch, value=25.0, rows=1)
        res = h.Result()
        h.check_dispersion(_payload(spread=0.1), res)
        assert res.healthy  # cannot judge without a baseline

    def test_missing_history_skips_check(self, tmp_path, monkeypatch):
        monkeypatch.setattr(h, "DISPERSION_HISTORY", tmp_path / "nope.csv")
        res = h.Result()
        h.check_dispersion(_payload(spread=0.1), res)
        assert res.healthy


class TestPriceDerivedCoverage:
    """Momentum and risk are 23% of composite weight and every metric in both
    comes from one `Ticker.history()` call per stock.

    Since 2026-08-26 a series that mixes two price scales is rejected and
    those metrics are withheld. One name (MNST, 1 of 502) is the mechanism
    working; the whole universe means the feed changed shape. Dispersion
    cannot catch that - with most stocks NaN it is computed over whatever
    survives.
    """

    def _blank(self, payload, cat, k):
        for row in payload["table_data"][:k]:
            row[f"{cat}_score"] = None
        return payload

    def test_full_coverage_passes(self):
        res = h.Result()
        h.check_price_derived_coverage(_payload(n=100), res)
        assert res.healthy

    def test_one_rejected_name_is_tolerated(self):
        """The MNST case must not fail the run."""
        p = _payload(n=502)
        for row in p["table_data"][:1]:
            row["momentum_score"] = None
            row["risk_score"] = None
        res = h.Result()
        h.check_price_derived_coverage(p, res)
        assert res.healthy

    @pytest.mark.parametrize("cat", ["momentum", "risk"])
    def test_mass_rejection_fails(self, cat):
        res = h.Result()
        h.check_price_derived_coverage(self._blank(_payload(n=100), cat, 50), res)
        assert not res.healthy
        assert cat in res.failures[0]

    @pytest.mark.parametrize("cat", ["momentum", "risk"])
    def test_threshold_boundary(self, cat):
        """9% missing passes, 11% fails - the gate is at 90%."""
        res = h.Result()
        h.check_price_derived_coverage(self._blank(_payload(n=100), cat, 9), res)
        assert res.healthy

        res = h.Result()
        h.check_price_derived_coverage(self._blank(_payload(n=100), cat, 11), res)
        assert not res.healthy

    def test_empty_table_is_not_an_error(self):
        """check_coverage already fails on an empty payload; this must not
        raise a second, confusing failure on the way there."""
        res = h.Result()
        h.check_price_derived_coverage({"table_data": []}, res)
        assert res.healthy

    def test_runs_as_part_of_check_coverage(self):
        """Wired in, not merely present."""
        res = h.Result()
        h.check_coverage(self._blank(_payload(n=100), "risk", 50), res)
        assert not res.healthy


class TestPayloadLoading:
    def test_malformed_payload_raises(self, tmp_path):
        (tmp_path / "dashboard_data.js").write_text(
            "window.SCREENER_DATA = {not valid json", encoding="utf-8")
        with pytest.raises(ValueError):
            h.load_payload(tmp_path)

    def test_valid_payload_loads(self, tmp_path):
        (tmp_path / "dashboard_data.js").write_text(
            "window.SCREENER_DATA = " + json.dumps(_payload(n=3)) + ";",
            encoding="utf-8")
        assert len(h.load_payload(tmp_path)["table_data"]) == 3
