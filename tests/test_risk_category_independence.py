"""The risk category scores dispersion only, not risk-adjusted return.

Why this file exists
--------------------
Until 2026-09-02 the ``risk`` category scored five metrics::

    volatility 30 | beta 20 | sharpe_ratio 15 | sortino_ratio 15 | max_drawdown_1y 20

and ``SCREENER_OVERVIEW.md`` justified that with "Five metrics give a more
complete risk picture than two."  Measured on the published payload, it was
five metrics in name and three in substance.

Both ratios are built in ``factor_engine.py`` from the *same* numerator --
``sharpe_ratio`` at :1923 is ``(return_12m - rf) / volatility`` and
``sortino_ratio`` at :1946 is ``(return_12m - rf) / downside_deviation``.
Across the S&P 500 the cross-sectional spread in trailing returns is far
wider than the spread in dispersion, so the shared numerator dominates both.
Spearman correlations on the 2026-09-02 published payload (N=498-499), and
essentially identical on the 08-31 and 09-01 runs:

    sharpe_ratio ~ sortino_ratio     +0.993
    sharpe_ratio ~ return_12_1       +0.944
    sortino_ratio ~ return_12_1      +0.940
    sharpe_ratio ~ volatility        +0.025   <- the risk content

So the two "risk-adjusted efficiency" metrics carried almost no volatility
information and almost all of the momentum signal.  At the category level
that made momentum and risk correlate **+0.516**, the highest pair of the 28
in the 8x8 category matrix.  Dropping them from scoring takes it to **+0.150**.

The user-facing consequence, which is what makes this a defect rather than a
preference: on the 2026-09-02 run SNDK published a **risk score of 31.1**
alongside a **momentum score of 94.2**.  Scored on dispersion alone its risk
score is **1.6**.  The site was telling a student that a violently volatile
stock was mid-pack on risk, because it had gone up.

This matches how institutional risk models are actually built.  The Barra US
Equity Model's volatility style factors are composed of dispersion
descriptors -- daily standard deviation, cumulative range, residual sigma --
and MSCI's Minimum Volatility indexes optimise on those Barra BETA/RESVOL
exposures.  No index provider selects for low risk with a Sharpe ratio.
Frazzini & Pedersen (2014) report a Sharpe ratio of 0.78 *for the BAB factor
portfolio* while selecting stocks on **beta** -- which is the correct use of
the statistic: evaluating a realised portfolio, not ranking a cross-section.

What must stay true
-------------------
1. The scored risk metrics are dispersion measures only.
2. Sharpe and Sortino stay *computed and displayed* (weight-0 candidates),
   because they are genuinely informative to a reader -- they just are not
   risk.  This mirrors the existing treatment of ``proximity_52w_high``.
3. The three surviving weights keep their old 30/20/20 relative emphasis.
   Rebalancing *among* the dispersion metrics would be a separate claim that
   this change does not make and did not research.

See METHODOLOGY_CHANGELOG.md 2026-09-02 and
research/2026-09-02-category-independence-synthesis.md.
"""

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.stats import spearmanr

import factor_engine
from schemas import RiskWeights

# The metrics that legitimately belong in a risk category: each measures how
# much a stock moves, and none has a trailing return in its numerator.
DISPERSION_METRICS = {"volatility", "beta", "max_drawdown_1y"}

# Ratios of the form (return - rf) / dispersion.  Informative, but they are
# risk-adjusted *return*, and the return term dominates cross-sectionally.
RETURN_OVER_RISK_RATIOS = {"sharpe_ratio", "sortino_ratio"}


@pytest.fixture(scope="module")
def config_risk_weights():
    with open("config.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg["metric_weights"]["risk"]


# ---------------------------------------------------------------------------
# 1. The configured weights
# ---------------------------------------------------------------------------

def test_return_over_risk_ratios_carry_no_scoring_weight(config_risk_weights):
    """The core assertion. Fails against the pre-2026-09-02 config."""
    for metric in RETURN_OVER_RISK_RATIOS:
        assert config_risk_weights[metric] == 0, (
            f"{metric} is (return - rf) / dispersion; it shares a numerator "
            "with the momentum signal and must not be scored as risk"
        )


def test_only_dispersion_metrics_are_scored(config_risk_weights):
    scored = {m for m, w in config_risk_weights.items() if w > 0}
    assert scored == DISPERSION_METRICS


def test_risk_weights_still_sum_to_100(config_risk_weights):
    assert sum(config_risk_weights.values()) == pytest.approx(100, abs=0.5)


def test_relative_emphasis_of_dispersion_metrics_is_unchanged(config_risk_weights):
    """volatility : beta : max_drawdown stays 30 : 20 : 20, i.e. 3 : 2 : 2.

    This change removes two metrics; it deliberately does not also re-rank the
    three that remain.
    """
    vol = config_risk_weights["volatility"]
    beta = config_risk_weights["beta"]
    dd = config_risk_weights["max_drawdown_1y"]
    assert vol / beta == pytest.approx(30 / 20, rel=1e-2)
    assert beta / dd == pytest.approx(1.0, rel=1e-2)


def test_schema_defaults_match_config(config_risk_weights):
    """A drift between schemas.py and config.yaml would silently reintroduce
    the old behaviour for any caller relying on defaults."""
    defaults = RiskWeights().model_dump()
    assert defaults == pytest.approx(config_risk_weights)


def test_schema_rejects_reintroducing_the_old_weights_without_rebalancing():
    """The old 30/20/15/15/20 no longer sums to 100 once the ratios are 0,
    so the sum guard catches a careless partial revert."""
    with pytest.raises(ValueError):
        RiskWeights(volatility=30, beta=20, max_drawdown_1y=20,
                    sharpe_ratio=0, sortino_ratio=0)


# ---------------------------------------------------------------------------
# 2. The metrics remain computed and displayed
# ---------------------------------------------------------------------------

def test_ratios_are_still_computed_and_shown():
    """Weight 0 means 'not scored', not 'deleted'. They stay on the drilldown."""
    for metric in RETURN_OVER_RISK_RATIOS:
        assert metric in factor_engine.METRIC_COLS
        assert metric in factor_engine.CAT_METRICS["risk"]


def test_ratios_keep_their_higher_is_better_direction():
    for metric in RETURN_OVER_RISK_RATIOS:
        assert factor_engine.METRIC_DIR[metric] is True


# ---------------------------------------------------------------------------
# 3. The mechanism, demonstrated deterministically
# ---------------------------------------------------------------------------

def _risk_score(percentiles: pd.DataFrame, weights: dict) -> pd.Series:
    """Weighted mean of percentiles, renormalised over available weight --
    the same rule compute_category_scores applies."""
    cols = [m for m, w in weights.items() if w > 0]
    w = np.array([weights[m] for m in cols], dtype=float)
    return (percentiles[cols] * w).sum(axis=1) / w.sum()


def _synthetic_cross_section(n=400, seed=11):
    """A universe where volatility is *deliberately independent* of return.

    If Sharpe were measuring risk, its percentile rank would track volatility.
    It does not: with a common denominator scale, Sharpe is a monotone
    function of the return, so it ranks the cross-section by momentum.
    """
    rng = np.random.default_rng(seed)
    ret_12m = rng.normal(0.10, 0.35, n)          # wide spread, as in reality
    vol = rng.uniform(0.18, 0.55, n)             # independent by construction
    downside = vol * rng.uniform(0.6, 0.8, n)    # tracks vol, as it does live
    rf = 0.04
    sharpe = (ret_12m - rf) / vol
    sortino = (ret_12m - rf) / downside
    drawdown = -np.abs(rng.normal(0.25, 0.10, n))

    def pct(series, higher_is_better):
        r = pd.Series(series).rank(pct=True) * 100
        return r if higher_is_better else 100 - r

    return pd.DataFrame({
        "volatility": pct(vol, False),
        "beta": pct(vol * rng.uniform(0.8, 1.2, n), False),
        "max_drawdown_1y": pct(drawdown, True),
        "sharpe_ratio": pct(sharpe, True),
        "sortino_ratio": pct(sortino, True),
        "_momentum": pct(ret_12m, True),
    })


def test_sharpe_and_sortino_are_near_duplicates_of_each_other():
    df = _synthetic_cross_section()
    rho = spearmanr(df["sharpe_ratio"], df["sortino_ratio"]).statistic
    assert rho > 0.95, (
        "Sharpe and Sortino share a numerator and near-proportional "
        f"denominators; got rho={rho:.3f}. Live payload measures +0.993."
    )


def test_the_ratios_track_return_not_volatility():
    df = _synthetic_cross_section()
    with_momentum = spearmanr(df["sharpe_ratio"], df["_momentum"]).statistic
    with_volatility = spearmanr(df["sharpe_ratio"], df["volatility"]).statistic
    assert with_momentum > 0.7
    assert abs(with_volatility) < 0.4
    assert with_momentum > abs(with_volatility)


def test_dropping_the_ratios_decouples_risk_from_momentum():
    """The change, end to end: the risk score stops ranking on return."""
    df = _synthetic_cross_section()
    old = _risk_score(df, {"volatility": 30, "beta": 20, "sharpe_ratio": 15,
                           "sortino_ratio": 15, "max_drawdown_1y": 20})
    new = _risk_score(df, {"volatility": 42.86, "beta": 28.57,
                           "max_drawdown_1y": 28.57,
                           "sharpe_ratio": 0, "sortino_ratio": 0})
    old_rho = abs(spearmanr(old, df["_momentum"]).statistic)
    new_rho = abs(spearmanr(new, df["_momentum"]).statistic)
    assert new_rho < old_rho / 2, (
        f"momentum leakage into risk should fall sharply: {old_rho:.3f} -> "
        f"{new_rho:.3f}. Live category correlation went +0.516 -> +0.150."
    )


def test_new_risk_score_tracks_dispersion_more_closely_than_the_old_one():
    df = _synthetic_cross_section()
    old = _risk_score(df, {"volatility": 30, "beta": 20, "sharpe_ratio": 15,
                           "sortino_ratio": 15, "max_drawdown_1y": 20})
    new = _risk_score(df, {"volatility": 42.86, "beta": 28.57,
                           "max_drawdown_1y": 28.57,
                           "sharpe_ratio": 0, "sortino_ratio": 0})
    # 'volatility' here is already direction-adjusted (higher pct = safer),
    # so a genuine risk score should correlate positively and strongly.
    assert (spearmanr(new, df["volatility"]).statistic
            > spearmanr(old, df["volatility"]).statistic)


# ---------------------------------------------------------------------------
# 4. The published documentation must not re-assert the refuted claim
# ---------------------------------------------------------------------------

def test_overview_no_longer_claims_five_risk_metrics():
    """SCREENER_OVERVIEW.md is generated from run_screener.py. The old text
    asserted 'Five metrics give a more complete risk picture than two', which
    is exactly what the +0.993 correlation refutes."""
    with open("run_screener.py", encoding="utf-8") as fh:
        source = fh.read()
    assert "Five metrics give a more complete risk picture" not in source
    assert "5-metric risk category" not in source
