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


class TestMetricAvailability:
    """2026-10-14: the source stopped serving one `.info` field, `dividend_yield`
    fell from 402 of 501 stocks to 6, and this script said PASS - nothing had
    ever compared one run's per-metric availability with the last one's. The
    code loop's input-churn tripwire found it four hours after publication.

    "Most of the universe" has one definition, shared with that tripwire
    (`history.universe_wide_changes`), so the gate and the record cannot drift.
    """

    @staticmethod
    def _snapshots(tmp_path, monkeypatch, missing_now, n=20, cols=None):
        """Two snapshot files: yesterday complete, today missing what is named."""
        import pandas as pd

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import history as H

        cols = cols or ["roic_pct", "dividend_yield_pct"]
        tickers = [f"T{i:03d}" for i in range(n)]

        def frame(absent: set[str]) -> pd.DataFrame:
            rows = []
            for i, t in enumerate(tickers):
                row = {"Ticker": t, "Sector": "Tech", "Rank": i + 1,
                       "Composite": 50.0}
                for cat in H.CATEGORIES:
                    row[f"{cat}_score"] = 50.0
                for c in cols:
                    row[c] = None if c in absent else 50.0
                rows.append(row)
            return pd.DataFrame(rows)

        d = tmp_path / "snapshots"
        d.mkdir()
        frame(set()).to_parquet(d / "2026-10-13_aaa.parquet")
        frame(set(missing_now)).to_parquet(d / "2026-10-14_bbb.parquet")
        monkeypatch.setattr(H, "SNAPSHOTS_DIR", d)
        return d

    def test_a_weighted_metric_lost_by_the_universe_discards_the_run(
            self, tmp_path, monkeypatch):
        """Every composite would rest on a different metric set from yesterday's,
        and dispersion does not notice because renormalisation keeps the spread."""
        self._snapshots(tmp_path, monkeypatch, {"roic_pct"})
        res = h.Result()
        h.check_metric_availability(res)
        assert not res.healthy
        assert any("roic" in f for f in res.failures)
        assert any("0/20" in f and "20/20" in f for f in res.failures), res.failures

    def test_an_unweighted_metric_is_reported_and_published(
            self, tmp_path, monkeypatch):
        """The 2026-10-14 case itself. Display-only: jamming the evidence loop
        over a row in the Company Snapshot is the wrong trade."""
        self._snapshots(tmp_path, monkeypatch, {"dividend_yield_pct"})
        res = h.Result()
        h.check_metric_availability(res)
        assert res.healthy
        assert any("dividend_yield" in n and "no weight" in n for n in res.notes)

    def test_one_stock_losing_a_metric_is_not_a_source_event(
            self, tmp_path, monkeypatch):
        """The mechanism it must not become: a per-stock gap is normal."""
        import pandas as pd

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import history as H

        d = self._snapshots(tmp_path, monkeypatch, set())
        f = pd.read_parquet(d / "2026-10-14_bbb.parquet")
        f.loc[0, "roic_pct"] = None
        f.to_parquet(d / "2026-10-14_bbb.parquet")
        monkeypatch.setattr(H, "SNAPSHOTS_DIR", d)
        res = h.Result()
        h.check_metric_availability(res)
        assert res.healthy
        assert any("no metric changed availability" in n for n in res.notes)

    def test_fewer_than_two_snapshots_cannot_tell_and_says_so(
            self, tmp_path, monkeypatch):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import history as H
        d = tmp_path / "empty"
        d.mkdir()
        monkeypatch.setattr(H, "SNAPSHOTS_DIR", d)
        res = h.Result()
        h.check_metric_availability(res)
        assert res.healthy
        assert any("fewer than two snapshots" in n for n in res.notes)

    def test_a_metric_regained_by_the_universe_does_not_discard_the_run(
            self, tmp_path, monkeypatch):
        """Recovery day. The run is better than the last one, not worse - a
        weighted metric coming *back* must not refuse the publish."""
        import pandas as pd

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import history as H

        d = self._snapshots(tmp_path, monkeypatch, set())
        f = pd.read_parquet(d / "2026-10-13_aaa.parquet")
        f["roic_pct"] = None
        f.to_parquet(d / "2026-10-13_aaa.parquet")
        monkeypatch.setattr(H, "SNAPSHOTS_DIR", d)
        res = h.Result()
        h.check_metric_availability(res)
        assert res.healthy, res.failures
        assert any("roic" in n and "regained" in n for n in res.notes), res.notes

    def test_the_weights_come_from_the_one_place_they_are_resolved(self):
        """Row 0.8: never a second list of which metrics carry weight."""
        names = h._weighted_metric_names()
        assert names is not None and names
        import factor_engine as F
        gen, bank = F.weighted_metric_sets(F.load_config())
        assert names == set(gen) | set(bank)
        assert "dividend_yield" not in names, \
            "dividend_yield is a weight-0 candidate; if that changed, the " \
            "2026-10-14 outage would have been a run-discarding failure"

    def test_cannot_read_the_weights_is_a_note_not_a_failure(self, tmp_path,
                                                             monkeypatch):
        """A gate that can only fail closed jams an unattended loop."""
        self._snapshots(tmp_path, monkeypatch, {"roic_pct"})
        monkeypatch.setattr(h, "_weighted_metric_names", lambda: None)
        res = h.Result()
        h.check_metric_availability(res)
        assert res.healthy
        assert any("cannot tell whether it carries weight" in n for n in res.notes)

    def test_it_is_wired_into_main(self):
        src = (Path(__file__).resolve().parent.parent / "scripts"
               / "check_run_health.py").read_text(encoding="utf-8")
        # with the run, so the newest snapshot is checked to be this run's (2026-10-16)
        assert "check_metric_availability(res, run_dir)" in src.split("def main(")[1]


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


# --------------------------------------------------------------------------- 2026-10-16
def _run_dir(tmp_path, run_id="bbb", start="2026-10-14T02:00:03.815047",
             end="2026-10-14T02:18:18.810369", payload=True, stamp=None, cfg="",
             tickers=None):
    """A run folder as run_screener.py leaves it: fetch evidence, meta, config, payload."""
    d = tmp_path / "runs" / run_id
    d.mkdir(parents=True)
    for f in h.FETCH_EVIDENCE:
        (d / f).write_bytes(b"x")
    (d / "meta.json").write_text(json.dumps({
        "run_id": run_id, "start_time": start, "end_time": end,
        "cli_flags": {"refresh": False, "tickers": tickers}}), encoding="utf-8")
    (d / "config.yaml").write_text(cfg or "improvement:\n  enabled: true\n", encoding="utf-8")
    # A full universe, every price dated on the session the run expects, and the universe.json
    # every live run writes - what the 2026-10-22 input gates read (tests/test_input_gates.py).
    (d / "universe.json").write_text(json.dumps(
        {"scored": [], "failed": [], "scored_count": 500, "failed_count": 0}), encoding="utf-8")
    if payload:
        p = _payload(n=500)
        session = h.expected_session(start).isoformat()
        for v in p["stock_detail"].values():
            v["price_date"] = session
        p["kpis"] = {"run_timestamp": start if stamp is None else stamp}
        (d / "dashboard_data.js").write_text(
            "window.SCREENER_DATA = " + json.dumps(p) + ";", encoding="utf-8")
    return d


def _main(monkeypatch, run_dir, capsys):
    # No published ranking to compare with: main() would otherwise read the real repo's
    # committed dashboard_data.js through git.
    monkeypatch.setattr(h, "published_payload", lambda ref="HEAD": None)
    monkeypatch.setattr(sys, "argv", ["check_run_health.py", "--run-dir", str(run_dir)])
    code = h.main()
    return code, capsys.readouterr().out


class TestThePayloadIsThisRuns:
    """The root `dashboard_data.js` is the last *published* run. The run folder lacks a
    payload only when run_screener.py's step 12 failed (a WARNING there), and the fallback
    to the root then had this gate approving yesterday's page."""

    def test_no_fallback_to_the_published_payload(self, tmp_path, monkeypatch):
        (tmp_path / "dashboard_data.js").write_text(
            "window.SCREENER_DATA = " + json.dumps(_payload(n=3)) + ";", encoding="utf-8")
        monkeypatch.setattr(h, "ROOT", tmp_path)
        run = _run_dir(tmp_path, payload=False)
        assert h.load_payload(run) is None

    def test_a_run_without_its_own_payload_cannot_be_evaluated(self, tmp_path, monkeypatch,
                                                                capsys):
        run = _run_dir(tmp_path, payload=False)
        code, out = _main(monkeypatch, run, capsys)
        assert code == 2
        assert "no dashboard_data.js of its own" in out
        assert "PASS" not in out

    def test_a_payload_stamped_by_another_run_cannot_be_evaluated(self, tmp_path, monkeypatch,
                                                                  capsys):
        run = _run_dir(tmp_path, stamp="2026-10-13T02:00:01.000001")
        code, out = _main(monkeypatch, run, capsys)
        assert code == 2
        assert "not this run's" in out and "2026-10-13T02:00:01" in out

    def test_a_payload_with_no_stamp_is_not_this_runs(self, tmp_path):
        run = _run_dir(tmp_path, stamp="")
        assert "not this run's" in h.payload_run_mismatch(h.load_payload(run), run)

    def test_the_runs_own_stamp_matches(self, tmp_path):
        run = _run_dir(tmp_path)
        assert h.payload_run_mismatch(h.load_payload(run), run) is None

    def test_the_stamp_is_the_one_the_generator_writes(self):
        """kpis.run_timestamp is meta.json's start_time, copied verbatim."""
        src = (Path(__file__).resolve().parent.parent / "generate_dashboard.py").read_text(
            encoding="utf-8")
        assert '"run_timestamp": meta.get("start_time", "")' in src

    def test_data_run_discards_on_exit_2(self):
        """Exit 2 is only a refusal because the data loop treats it as one."""
        ps = (Path(__file__).resolve().parent.parent / "scripts" / "data-run.ps1").read_text(
            encoding="utf-8")
        block = ps.split("scripts/check_run_health.py')")[1].split("# --- Regenerate")[0]
        assert "$health.ExitCode -eq 1" in block
        tail = block.split("elseif ($health.ExitCode -ne 0)")[1]
        assert "Stop-Run" in tail and "discarded" in tail


class TestTheSnapshotIsThisRuns:
    """record_run_snapshot failing is a WARNING in run_screener.py. The availability check
    then compared yesterday with the day before and said PASS about a run it never saw."""

    _snapshots = staticmethod(TestMetricAvailability._snapshots)

    def test_this_runs_snapshot_is_compared(self, tmp_path, monkeypatch):
        self._snapshots(tmp_path, monkeypatch, set())
        run = _run_dir(tmp_path, run_id="bbb")
        res = h.Result()
        h.check_metric_availability(res, run)
        assert res.healthy and not res.unevaluable
        assert any("no metric changed availability" in n for n in res.notes)

    def test_a_newest_snapshot_from_an_earlier_day_cannot_be_evaluated(self, tmp_path,
                                                                       monkeypatch):
        self._snapshots(tmp_path, monkeypatch, {"roic_pct"})
        run = _run_dir(tmp_path, run_id="ccc", start="2026-10-15T02:00:00",
                       end="2026-10-15T02:18:00")
        res = h.Result()
        h.check_metric_availability(res, run)
        assert res.unevaluable and "2026-10-14_bbb.parquet" in res.unevaluable[0]
        assert not res.failures, "an old pair must not be judged as if it were this run"

    def test_a_same_day_snapshot_from_another_run_cannot_be_evaluated(self, tmp_path,
                                                                      monkeypatch):
        self._snapshots(tmp_path, monkeypatch, set())
        run = _run_dir(tmp_path, run_id="zzz")
        res = h.Result()
        h.check_metric_availability(res, run)
        assert res.unevaluable and "not this run's" in res.unevaluable[0]

    def test_no_snapshots_at_all_cannot_be_evaluated(self, tmp_path, monkeypatch):
        import history as H
        d = tmp_path / "empty"
        d.mkdir()
        monkeypatch.setattr(H, "SNAPSHOTS_DIR", d)
        res = h.Result()
        h.check_metric_availability(res, _run_dir(tmp_path))
        assert res.unevaluable

    def test_the_first_ever_snapshot_is_this_runs_and_passes(self, tmp_path, monkeypatch):
        d = self._snapshots(tmp_path, monkeypatch, set())
        (d / "2026-10-13_aaa.parquet").unlink()
        res = h.Result()
        h.check_metric_availability(res, _run_dir(tmp_path, run_id="bbb"))
        assert not res.unevaluable and res.healthy
        assert any("fewer than two snapshots" in n for n in res.notes)

    def test_a_run_that_records_no_snapshot_by_design_is_a_note(self, tmp_path, monkeypatch):
        self._snapshots(tmp_path, monkeypatch, set())
        off = _run_dir(tmp_path, run_id="off", cfg="improvement:\n  enabled: false\n")
        res = h.Result()
        h.check_metric_availability(res, off)
        assert not res.unevaluable and res.healthy
        assert any("improvement.enabled: false" in n for n in res.notes)

        sub = _run_dir(tmp_path, run_id="sub", tickers="AAPL,MSFT")
        res = h.Result()
        h.check_metric_availability(res, sub)
        assert not res.unevaluable and res.healthy
        assert any("--tickers" in n for n in res.notes)

    def test_main_exits_2_when_the_snapshot_is_missing(self, tmp_path, monkeypatch, capsys):
        self._snapshots(tmp_path, monkeypatch, set())
        _history(tmp_path, monkeypatch)
        run = _run_dir(tmp_path, run_id="ccc", start="2026-10-15T02:00:00",
                       end="2026-10-15T02:18:00")
        code, out = _main(monkeypatch, run, capsys)
        assert code == 2, out
        assert "MISSING" in out and "HEALTH: cannot evaluate" in out

    def test_main_passes_on_a_complete_run(self, tmp_path, monkeypatch, capsys):
        self._snapshots(tmp_path, monkeypatch, set())
        _history(tmp_path, monkeypatch)
        code, out = _main(monkeypatch, _run_dir(tmp_path, run_id="bbb"), capsys)
        assert code == 0, out
        assert "HEALTH: PASS" in out

    def test_a_real_failure_still_exits_1(self, tmp_path, monkeypatch, capsys):
        """Exit 1 also cleans improvement/snapshots in data-run.ps1, so it takes precedence."""
        self._snapshots(tmp_path, monkeypatch, set())
        _history(tmp_path, monkeypatch)
        run = _run_dir(tmp_path, run_id="ccc", start="2026-10-15T02:00:00",
                       end="2026-10-15T02:18:00")
        p = _payload(price=False)
        p["kpis"] = {"run_timestamp": "2026-10-15T02:00:00"}
        (run / "dashboard_data.js").write_text(
            "window.SCREENER_DATA = " + json.dumps(p) + ";", encoding="utf-8")
        code, out = _main(monkeypatch, run, capsys)
        assert code == 1, out
