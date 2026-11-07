"""The per-band gates (universe-expansion phase 1, 2026-11-07).

`plan/universe-expansion.md` phase 1 asks for two thresholds to stop being universe-wide,
"because a universe-wide average hides a thin bottom band":

* `check_run_health.MIN_TARGET_COVERAGE` (0.50) - **shipped**, and the subject of most of this
  module. The action is "discard the run", so applying the floor inside each band changes no
  score and is strictly stronger than the universe-wide check it sits beside.
* `run_screener.CATEGORY_AUTO_DISABLE_PCT` (30) - **measured per band, still triggered
  universe-wide**, because its action is to change the weights and there are three incompatible
  per-band rules (`run_screener.weighted_coverage_pct_by_band` lists them). Which is right needs
  phase 2's numbers. The tests at the bottom are the tripwire: they fail if the trigger becomes
  per-band without the `METHODOLOGY_CHANGELOG.md` entry that makes it a declared methodology
  change.

**What makes these tests worth something rather than plumbing.** With one universe there is one
band, so every per-band assertion that uses production data passes whatever the code does - the
same trap `tests/test_size_bands.py` met with the percentile identity. So the band path is
exercised here against fixtures built so that *the universe-wide check passes and a band fails*;
`test_a_thin_band_is_invisible_to_the_universe_wide_check` is the one that would catch the whole
feature being a no-op, and `TestDominance` pins the ordering property the design rests on.

Not in the 02:00 publish gate, deliberately: these check the gate's own logic against synthetic
frames, not a sentence or figure on a published artifact, and `scripts/check_published_claims.py`
speaks for what a run is about to publish. The gate itself runs on every data run regardless.

No network, no clock, no published artifact is read or written.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import check_run_health as h  # noqa: E402
import run_screener as rs  # noqa: E402
import universe  # noqa: E402

PLAN = ROOT / "plan" / "universe-expansion.md"


def payload(n: int = 100, targeted: int | None = None) -> dict:
    """`n` stocks, the first `targeted` of them carrying an analyst target (all by default)."""
    want = n if targeted is None else targeted
    detail = {f"T{i:03d}": {"price": 10.0 + i,
                            "pt_mean": (12.0 + i) if i < want else None}
              for i in range(n)}
    table = [{"Ticker": t, **{f"{c}_score": 50.0 + i * 0.1 for c in h.CATEGORIES}}
             for i, t in enumerate(detail)]
    return {"stock_detail": detail, "table_data": table}


def bands_of(p: dict, sizes: dict[str, int]) -> dict[str, str]:
    """Hand the first `sizes[band]` tickers to each band, in the order given."""
    it = iter(p["stock_detail"])
    out = {}
    for band, k in sizes.items():
        for _ in range(k):
            out[next(it)] = band
    return out


def run_bands(res: h.Result) -> tuple[list[str], list[str]]:
    return res.failures, res.notes


# --- reading the band off the run's own artifact ------------------------------------------

class TestRunSizeBands:
    def test_absent_artifact_reads_as_unknown(self, tmp_path):
        assert h.run_size_bands(tmp_path) is None

    def test_a_run_without_the_column_reads_as_unknown(self, tmp_path):
        """Every run before 2026-11-07. Must not raise and must not fail the gate."""
        pd.DataFrame({"Ticker": ["A", "B"], "Sector": ["X", "Y"]}).to_parquet(
            tmp_path / h.RAW_METRICS)
        assert h.run_size_bands(tmp_path) is None

    def test_an_all_blank_column_reads_as_unknown(self, tmp_path):
        pd.DataFrame({"Ticker": ["A", "B"],
                      universe.SIZE_BAND_COL: [None, "  "]}).to_parquet(
            tmp_path / h.RAW_METRICS)
        assert h.run_size_bands(tmp_path) is None

    def test_a_corrupt_artifact_reads_as_unknown(self, tmp_path):
        (tmp_path / h.RAW_METRICS).write_bytes(b"not a parquet file")
        assert h.run_size_bands(tmp_path) is None

    def test_the_bands_are_returned_per_ticker(self, tmp_path):
        pd.DataFrame({"Ticker": ["A", "B", "C"],
                      universe.SIZE_BAND_COL: ["big", "big", "small"]}).to_parquet(
            tmp_path / h.RAW_METRICS)
        assert h.run_size_bands(tmp_path) == {"A": "big", "B": "big", "C": "small"}

    def test_a_partly_blank_column_keeps_the_rows_that_have_one(self, tmp_path):
        pd.DataFrame({"Ticker": ["A", "B"],
                      universe.SIZE_BAND_COL: ["big", None]}).to_parquet(
            tmp_path / h.RAW_METRICS)
        assert h.run_size_bands(tmp_path) == {"A": "big"}

    def test_it_reads_the_artifact_the_gate_already_requires(self):
        """No new input: `01_raw_metrics.parquet` is already fetch evidence, so the per-band
        check cannot be reading another run's file or a file a healthy run may lack."""
        assert h.RAW_METRICS in h.FETCH_EVIDENCE

    def test_the_band_is_not_read_from_the_payload(self):
        """Phase 1's exit test is a byte-identical S&P 500 payload, so the band is not
        published. A future session that publishes it must change this test on purpose."""
        src = (ROOT / "scripts" / "check_run_health.py").read_text(encoding="utf-8")
        body = src[src.index("def run_size_bands"):src.index("def check_target_coverage_by_band")]
        assert "read_parquet" in body
        assert "payload" not in body.split('"""')[2]


# --- the gate itself ----------------------------------------------------------------------

class TestPerBandTargetCoverage:
    def test_unknown_bands_are_a_note_and_never_a_failure(self):
        """A gate that can only fail closed jams an unattended loop."""
        res = h.Result()
        h.check_target_coverage_by_band(payload(), None, res)
        assert res.healthy
        assert any("not checked per size band" in n for n in res.notes)

    def test_one_band_says_so_and_adds_no_second_verdict(self):
        p = payload(60)
        res = h.Result()
        h.check_target_coverage_by_band(p, bands_of(p, {"S&P 500": 60}), res)
        assert res.healthy
        assert any("one size band (S&P 500)" in n for n in res.notes)
        assert not any("in the S&P 500 band" in n for n in res.notes)

    def test_a_thin_band_is_invisible_to_the_universe_wide_check(self):
        """**The test this feature exists for.** 90 covered large stocks and 10 uncovered
        small ones: the universe is 90% covered and clears the 50% floor, and the small band
        is at 0%. Before today nothing in either publish path could see that."""
        p = payload(100, targeted=90)
        bands = bands_of(p, {"large": 90, "small": 10})
        res = h.Result()
        h.check_coverage(p, res, bands=bands)
        universe_wide = [f for f in res.failures if "band" not in f]
        per_band = [f for f in res.failures if "band" in f]
        assert universe_wide == []
        assert len(per_band) == 1
        assert "0/10 stocks in the small band" in per_band[0]

    def test_the_failure_names_the_band_the_count_and_the_floor(self):
        p = payload(100, targeted=90)
        res = h.Result()
        h.check_target_coverage_by_band(p, bands_of(p, {"large": 90, "small": 10}), res)
        msg = res.failures[0]
        assert "small" in msg and "0/10" in msg and "need >=50%" in msg

    def test_a_band_exactly_at_the_floor_passes(self):
        """`>=` not `>`: the floor is the lowest acceptable value, and probing either side of
        it without probing it is how `tests/test_momentum_panic_rule.py`'s volatility bar
        stayed unchecked (DECISIONS.md, the 2026-10-25 mutation pass)."""
        p = payload(100, targeted=95)
        # last 10 are the small band; 5 of them carry a target -> exactly 50%
        bands = bands_of(p, {"large": 90, "small": 10})
        for i, t in enumerate(p["stock_detail"]):
            if i >= 95:
                p["stock_detail"][t]["pt_mean"] = None
        res = h.Result()
        h.check_target_coverage_by_band(p, bands, res)
        assert res.healthy, res.failures
        assert any("5/10 (50%) in the small band" in n for n in res.notes)

    def test_a_band_one_stock_below_the_floor_fails(self):
        p = payload(100, targeted=94)
        bands = bands_of(p, {"large": 90, "small": 10})
        res = h.Result()
        h.check_target_coverage_by_band(p, bands, res)
        assert not res.healthy
        assert "4/10" in res.failures[0]

    def test_every_band_is_judged_not_just_the_smallest(self):
        p = payload(90, targeted=0)
        for i, t in enumerate(p["stock_detail"]):
            if 30 <= i < 60:
                p["stock_detail"][t]["pt_mean"] = 5.0
        res = h.Result()
        h.check_target_coverage_by_band(p, bands_of(p, {"a": 30, "b": 30, "c": 30}), res)
        assert len(res.failures) == 2
        assert {"a", "c"} == {f.split(" in the ")[1].split(" band")[0] for f in res.failures}

    def test_unmatched_tickers_are_noted_and_left_in_the_universe_wide_check(self):
        p = payload(100)
        bands = bands_of(p, {"large": 60, "small": 30})   # 10 tickers have no band
        res = h.Result()
        h.check_target_coverage_by_band(p, bands, res)
        assert res.healthy
        assert any("10 scored stock(s) carry no size band" in n for n in res.notes)

    def test_no_matchable_band_is_a_note_not_a_vacuous_pass(self):
        """A band map for a different run would otherwise pass the check having judged
        nothing, which is the shape of a gate that reads as coverage and is not."""
        res = h.Result()
        h.check_target_coverage_by_band(payload(10), {"ZZZZ": "big"}, res)
        assert res.healthy
        assert any("no scored stock's band could be matched" in n for n in res.notes)

    def test_an_empty_payload_is_left_to_the_universe_wide_check(self):
        res = h.Result()
        h.check_target_coverage_by_band({"stock_detail": {}}, {"A": "big"}, res)
        assert res.failures == [] and res.notes == []
        res2 = h.Result()
        h.check_coverage({"stock_detail": {}}, res2)
        assert any("stock_detail is empty" in f for f in res2.failures)

    def test_check_coverage_still_takes_two_positional_arguments(self):
        """Called positionally from `tests/test_run_health.py`, `rehearse_publish` and the
        data loop. The band argument is optional, so an older caller keeps working."""
        res = h.Result()
        h.check_coverage(payload(), res)
        assert res.healthy

    def test_one_floor_serves_both_checks(self):
        """Two literals for one number are free to drift, and the page would then state a
        floor no run applied (DECISIONS.md, the share-class margin, 2026-10-31)."""
        src = (ROOT / "scripts" / "check_run_health.py").read_text(encoding="utf-8")
        assert src.count("MIN_TARGET_COVERAGE = ") == 1
        assert src.count("MIN_TARGET_COVERAGE") >= 5   # the constant plus both checks' uses
        # Neither check may carry its own copy of the number.
        for fn in (h.check_coverage, h.check_target_coverage_by_band):
            import inspect
            assert "0.5" not in inspect.getsource(fn), (
                f"{fn.__name__} holds a literal floor as well as the constant")


class TestDominance:
    """The per-band check is strictly stronger than the universe-wide one, never weaker.

    The universe rate is a weighted mean of the band rates, so a universe-wide failure
    implies at least one band below the same floor. That is why keeping the universe-wide
    line costs nothing, and why this change cannot make the gate more permissive on any
    input - which is the only property that lets it ship without a shadow run behind it.
    """

    SPLITS = [
        {"a": 50, "b": 50},
        {"a": 10, "b": 90},
        {"a": 33, "b": 33, "c": 34},
        {"a": 1, "b": 99},
        {"a": 25, "b": 25, "c": 25, "d": 25},
    ]

    @pytest.mark.parametrize("split", SPLITS)
    @pytest.mark.parametrize("targeted", [0, 20, 49, 50, 51, 80, 100])
    def test_a_universe_wide_failure_always_implies_a_band_failure(self, split, targeted):
        p = payload(100, targeted=targeted)
        bands = bands_of(p, split)
        res = h.Result()
        h.check_coverage(p, res, bands=bands)
        uni = [f for f in res.failures if "analyst target" in f and "band" not in f]
        band = [f for f in res.failures if "band" in f]
        if uni:
            assert band, (f"universe-wide failed at {targeted}% but no band did, on {split} - "
                          f"the per-band check is weaker than the mean it replaces")

    def test_price_coverage_stays_universe_wide(self):
        """Deliberate scope, not an omission: a price is served for every listed stock, so
        price coverage guards a feed failure, which shows across the whole universe or not at
        all. Banding it would only add ways to fail."""
        p = payload(100)
        for t in p["stock_detail"]:
            p["stock_detail"][t]["price"] = None
        res = h.Result()
        h.check_coverage(p, res, bands=bands_of(p, {"a": 50, "b": 50}))
        price = [f for f in res.failures if "have a price" in f]
        assert len(price) == 1 and "band" not in price[0]

    def test_the_momentum_and_risk_floor_stays_universe_wide(self):
        """`MIN_CATEGORY_COVERAGE` guards the mass rejection of price series
        (`factor_engine.check_price_series_integrity`), not a thin band."""
        src = (ROOT / "scripts" / "check_run_health.py").read_text(encoding="utf-8")
        body = src[src.index("def check_price_derived_coverage"):src.index("def _snapshot_expected")]
        assert universe.SIZE_BAND_COL not in body and "band" not in body


# --- the auto-disable: measured per band, still triggered universe-wide -------------------

class TestAutoDisableIsMeasuredNotMoved:
    def _frame(self, bands: list[str], rev_values: list[float]) -> pd.DataFrame:
        cfg = rs.load_config_safe()
        metrics = rs.weighted_category_metrics(cfg, "revisions")
        data = {universe.SIZE_BAND_COL: bands}
        for m in metrics:
            data[m] = rev_values
        data["Ticker"] = [f"T{i}" for i in range(len(bands))]
        return pd.DataFrame(data), cfg

    def test_coverage_is_reported_per_band(self):
        df, cfg = self._frame(["big"] * 4 + ["small"] * 4,
                              [1.0, 1.0, 1.0, 1.0, float("nan"), float("nan"),
                               float("nan"), float("nan")])
        by = rs.weighted_coverage_pct_by_band(df, cfg, "revisions")
        assert set(by) == {"big", "small"}
        assert by["big"] == pytest.approx(100.0)
        assert by["small"] == pytest.approx(0.0)

    def test_a_frame_without_the_column_reports_nothing(self):
        df, cfg = self._frame(["big"] * 2, [1.0, 1.0])
        assert rs.weighted_coverage_pct_by_band(df.drop(columns=[universe.SIZE_BAND_COL]),
                                                cfg, "revisions") == {}

    def test_the_band_figures_agree_with_the_universe_wide_one(self):
        """One band, so the two must be the same number - the arithmetic check that stops the
        per-band figure being computed over a different metric set."""
        df, cfg = self._frame(["S&P 500"] * 6, [1.0, 1.0, 1.0, float("nan"), 1.0, 1.0])
        by = rs.weighted_coverage_pct_by_band(df, cfg, "revisions")
        assert list(by) == ["S&P 500"]
        assert by["S&P 500"] == pytest.approx(rs.weighted_coverage_pct(df, cfg, "revisions"))

    def test_the_trigger_takes_no_band_and_so_cannot_have_become_per_band(self):
        """The tripwire. Making the auto-disable per band changes the **weights**, so it is a
        methodology change with its own changelog entry, not a phase-1 side effect. The three
        candidate rules are in `weighted_coverage_pct_by_band`'s docstring and in the plan."""
        import inspect
        sig = inspect.signature(rs.auto_disable_sparse_category)
        assert list(sig.parameters) == ["df", "cfg", "cat", "threshold"]
        body = inspect.getsource(rs.auto_disable_sparse_category)
        assert universe.SIZE_BAND_COL not in body
        assert "by_band" not in body

    def test_the_universe_wide_trigger_still_fires_on_the_universe_mean(self):
        """85% in a big band and 0% in a small one averages 42.5%, which is above the 30%
        floor - so today's rule keeps the category, and that is the recorded choice (a), not
        an accident. A session that changes it must change this test and the changelog."""
        vals = [1.0] * 17 + [float("nan")] * 3 + [float("nan")] * 20
        df, cfg = self._frame(["big"] * 20 + ["small"] * 20, vals)
        disabled, pct = rs.auto_disable_sparse_category(df, cfg, "revisions")
        assert pct == pytest.approx(42.5)
        assert disabled is False
        by = rs.weighted_coverage_pct_by_band(df, cfg, "revisions")
        assert by["small"] == pytest.approx(0.0)

    def test_the_plan_states_all_three_candidate_rules(self):
        """A decision left open has to be left open *in writing*, or the next session reads
        the measurement as the unfinished half of a build and picks one."""
        text = PLAN.read_text(encoding="utf-8")
        assert "**(a)**" in text and "**(b)**" in text and "**(c)**" in text
        assert "methodology change with its own `METHODOLOGY_CHANGELOG.md` entry" in text
        assert "phase 2" in text

    def test_the_docstring_states_the_same_three_rules(self):
        doc = rs.weighted_coverage_pct_by_band.__doc__
        assert "(a)" in doc and "(b)" in doc and "(c)" in doc
        assert "METHODOLOGY_CHANGELOG.md" in doc
