"""Regression tests for the removal of pre-rank winsorization (2026-09-01).

The defect this locks down, in the words of the run that shipped it:

    On 2026-09-01 the live public site published AAPL, NVDA, MSFT, GOOG,
    GOOGL and AMZN with an identical market capitalisation of $2,802.0B.
    Nvidia's true figure was $5,331.2B — understated by 47%.

The cause was `winsorize_metrics()`, called four lines before
`compute_sector_percentiles()`. That clipped the top and bottom 1% of every
metric onto a single boundary value, and the clipped value was what the
dashboard published as the stock's `raw` figure.

Two independent things were wrong with it, and both are covered here:

1.  It could never have helped. `compute_sector_percentiles()` is
    `Series.rank(pct=True)`, which is invariant under any monotone transform
    of its input, so clipping the tails cannot change an ordering. All it can
    do is collapse distinct values onto one, which `rank` then resolves to a
    shared average rank — turning real differences into artificial ties.
2.  It corrupted the published value, and it hid data errors. See
    METHODOLOGY_CHANGELOG.md 2026-08-26: MNST's corrupt `volatility_1y` of
    1.77 was clipped to 0.845, which made a broken series look merely
    volatile instead of impossible.

See METHODOLOGY_CHANGELOG.md 2026-09-01.
"""

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import factor_engine  # noqa: E402
from factor_engine import (  # noqa: E402
    METRIC_COLS,
    compute_sector_percentiles,
    flag_metric_outliers,
)

ROOT = Path(__file__).resolve().parent.parent

# Every module that scores metrics into ranks. run_screener is the live
# pipeline; factor_engine carries a second end-to-end path; backtest and
# run_audit are replicas that have to stay faithful to it.
SCORING_MODULES = ["run_screener.py", "factor_engine.py",
                   "backtest.py", "run_audit.py"]


def _parse(module):
    return ast.parse((ROOT / module).read_text(encoding="utf-8"))


def _docstring_nodes(tree):
    """Every string node that is a module/class/function docstring."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            out.append(body[0].value)
    return out


def _frame(values, metric="size_log_mcap", sector="Information Technology"):
    """Build a minimal scoring frame carrying one populated metric."""
    df = pd.DataFrame({
        "Ticker": [f"T{i:03d}" for i in range(len(values))],
        "Sector": [sector] * len(values),
        metric: values,
    })
    for col in METRIC_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df


# =====================================================================
# 1. The function is gone and nothing in the scoring path calls it
# =====================================================================

class TestWinsorizationIsGone:
    def test_factor_engine_exposes_no_winsorize_metrics(self):
        assert not hasattr(factor_engine, "winsorize_metrics"), (
            "winsorize_metrics is back. It must not be reintroduced ahead of "
            "compute_sector_percentiles — see this module's docstring."
        )

    @pytest.mark.parametrize("module", SCORING_MODULES)
    def test_scoring_path_makes_no_winsorizing_call(self, module):
        """backtest.py and run_audit.py mirror the pipeline; they must not drift.

        factor_engine.py has its own end-to-end path as well, which is how the
        original removal missed a fourth call site.
        """
        offenders = []
        for node in ast.walk(_parse(module)):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = (fn.attr if isinstance(fn, ast.Attribute)
                    else fn.id if isinstance(fn, ast.Name) else "")
            if "winsorize" in name.lower():
                offenders.append(f"line {node.lineno}: {name}()")
        assert not offenders, f"{module} still winsorizes: {offenders}"

    @pytest.mark.parametrize("module", SCORING_MODULES)
    def test_no_user_facing_text_claims_winsorization(self, module):
        """The methodology page and the audit report are read by students.

        Docstrings are excluded: they are allowed — and needed — to explain why
        winsorization was removed.
        """
        tree = _parse(module)
        docstring_ids = {id(d) for d in _docstring_nodes(tree)}
        offenders = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node) not in docstring_ids
                    and "winsoriz" in node.value.lower()
                    # The config fallback keeps an older config.yaml loadable.
                    and "winsorize_percentiles" not in node.value):
                offenders.append(f"line {node.lineno}: {node.value[:70]!r}")
        assert not offenders, (
            f"{module} still tells the reader it winsorizes: {offenders}")

    def test_scipy_winsorize_is_not_imported(self):
        src = (ROOT / "factor_engine.py").read_text(encoding="utf-8")
        assert "mstats" not in src


# =====================================================================
# 2. The property that made winsorization pointless
# =====================================================================

class TestRankIsInvariantToMonotoneTransforms:
    def test_percentiles_unchanged_by_a_strictly_monotone_transform(self):
        """rank(pct=True) sees order only, so a monotone rescale changes nothing."""
        rng = np.random.default_rng(7)
        vals = rng.normal(20, 6, 60)
        base = compute_sector_percentiles(_frame(vals.copy(), metric="ev_ebitda"))
        # Strictly increasing transform: distances change wildly, order does not.
        shifted = compute_sector_percentiles(
            _frame(np.exp(vals / 10.0), metric="ev_ebitda"))
        pd.testing.assert_series_equal(
            base["ev_ebitda_pct"], shifted["ev_ebitda_pct"], check_exact=False)

    def test_clipping_changes_percentiles_only_by_creating_ties(self):
        """Clipping is monotone but not injective — that is the entire harm."""
        vals = np.arange(1.0, 61.0)
        clipped = vals.copy()
        clipped[-3:] = clipped[-4]  # what winsorizing the top tail did

        base = compute_sector_percentiles(_frame(vals.copy(), metric="roic"))
        tied = compute_sector_percentiles(_frame(clipped, metric="roic"))

        # The untouched middle is identical: no ordering moved.
        pd.testing.assert_series_equal(
            base["roic_pct"].iloc[:-4], tied["roic_pct"].iloc[:-4],
            check_exact=False)
        # The clipped tail collapsed four distinct ranks onto one.
        assert base["roic_pct"].iloc[-4:].nunique() == 4
        assert tied["roic_pct"].iloc[-4:].nunique() == 1


# =====================================================================
# 3. The user-facing failure: distinct inputs keep distinct published values
# =====================================================================

class TestPublishedValuesAreNotCollapsed:
    # True market caps on 2026-09-01, the run that exposed the defect.
    MEGACAPS = {
        "NVDA": 5331.2e9, "AAPL": 4624.2e9, "GOOGL": 4150.2e9,
        "GOOG": 4102.0e9, "MSFT": 3766.9e9, "AMZN": 2802.0e9,
    }

    def _universe_with_megacaps(self):
        """60 ordinary caps plus the six largest US companies, as -ln(mcap)."""
        rng = np.random.default_rng(11)
        ordinary = list(rng.uniform(20e9, 900e9, 60))
        caps = ordinary + list(self.MEGACAPS.values())
        df = _frame([-np.log(c) for c in caps])
        df.loc[df.index[-6:], "Ticker"] = list(self.MEGACAPS)
        return df

    def test_six_largest_companies_keep_six_distinct_market_caps(self):
        """The exact defect: they were published as one identical figure."""
        df = self._universe_with_megacaps()
        flag_metric_outliers(df, 0.01, 0.01)  # the pipeline's only tail step
        mega = df[df["Ticker"].isin(self.MEGACAPS)]["size_log_mcap"]
        assert mega.nunique() == 6, (
            f"market caps collapsed to {mega.nunique()} distinct values; "
            "six companies would again be published with the same figure"
        )

    def test_six_largest_companies_keep_six_distinct_size_ranks(self):
        """AAPL, NVDA and MSFT shared one size percentile on the live site."""
        df = compute_sector_percentiles(self._universe_with_megacaps())
        pcts = df[df["Ticker"].isin(self.MEGACAPS)]["size_log_mcap_pct"]
        assert pcts.nunique() == 6

    def test_largest_company_is_not_understated(self):
        """NVDA was published at $2,802B against a true $5,331B."""
        df = self._universe_with_megacaps()
        flag_metric_outliers(df, 0.01, 0.01)
        nvda = df.loc[df["Ticker"] == "NVDA", "size_log_mcap"].iloc[0]
        assert np.exp(-nvda) == pytest.approx(self.MEGACAPS["NVDA"], rel=1e-9)


# =====================================================================
# 4. The reporter itself
# =====================================================================

class TestFlagMetricOutliers:
    def test_reports_both_tails_with_cutoffs(self):
        df = _frame(np.arange(1.0, 101.0), metric="ev_ebitda")
        report = flag_metric_outliers(df, 0.01, 0.01)
        assert "ev_ebitda" in report
        info = report["ev_ebitda"]
        assert info["n_valid"] == 100
        assert info["n_low"] >= 1 and info["n_high"] >= 1
        assert info["lo_cut"] < info["hi_cut"]

    def test_omits_metrics_that_are_entirely_missing(self):
        df = _frame([np.nan] * 40, metric="ev_ebitda")
        assert "ev_ebitda" not in flag_metric_outliers(df, 0.01, 0.01)

    def test_a_corrupt_value_is_surfaced_rather_than_hidden(self):
        """The MNST case: clipping 1.77 to 0.845 hid a broken price series."""
        vals = list(np.linspace(0.15, 0.55, 59)) + [1.77]
        df = _frame(vals, metric="volatility")
        report = flag_metric_outliers(df, 0.01, 0.01)
        assert df["volatility"].max() == pytest.approx(1.77)
        assert report["volatility"]["n_high"] >= 1
