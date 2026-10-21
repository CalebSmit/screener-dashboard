"""The research engine's positive control, synthetic half (plan/research-engine.md item 4(c), 2026-10-20).

The engine reports "not distinguishable from the current weighting" for every alternative it has
tested. That sentence is only worth something if the same arithmetic *does* find a difference when
one is there. These tests plant a known IC difference in synthetic months shaped like the real ones
(176 month-ends, 2012-01 to 2026-08; about 440 stocks a month; a monthly IC that swings by about
0.10 around its mean, as the baseline's does) and run them through the engine's own ``per_month``
and ``summarise`` - the paired, common-stock comparison every ledger row is built with - never a
re-implementation.

What is pinned:

- **Unbiased:** the reported difference lands within four standard errors of the planted one, at
  0.03, 0.05 and 0.10, in every period.
- **Sees what it says it can see:** a planted 0.10 clears the engine's own multiple-testing bar
  (``pass_bar``) on the 38 held-out months; and no planted difference larger than the engine's own
  ``detectable`` figure for that comparison is reported below the bar more than chance allows.
- **No false alarms:** two configurations with the same true IC are not reported different at the bar.
- **Pairing does its job:** a screen that can rank only 70% of the stocks is still measured
  against the baseline on the same 70% - its difference is recovered, not diluted.

Synthetic months: each month's returns are standard normal; a configuration's composite is built to
have a chosen Spearman correlation with them (for a bivariate normal, Spearman rho_s = (6 / pi)
arcsin(rho / 2), so rho = 2 sin(pi rho_s / 6)). The seed is fixed; the tolerances are set from the
sampling arithmetic above, not from a run.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import v2_experiments as X  # noqa: E402

N_STOCKS = 440
IC_SWING = 0.10          # month-to-month sd of a configuration's true IC (the baseline's is ~0.12)
MONTHS = pd.date_range("2012-01-31", "2026-08-31", freq="ME").strftime("%Y-%m-%d").tolist()
TICKERS = [f"S{i:03d}" for i in range(N_STOCKS)]


def _pearson_for(spearman: float) -> float:
    return float(np.clip(2 * math.sin(math.pi * spearman / 6), -0.999, 0.999))


def _world(seed: int):
    rng = np.random.default_rng(seed)
    ret = rng.standard_normal((len(MONTHS), N_STOCKS))
    fwd = pd.DataFrame(ret, index=MONTHS, columns=TICKERS)
    return rng, ret, fwd


def _scores(rng, ret, mean_ic: float, coverage: float = 1.0) -> pd.DataFrame:
    """A configuration whose true monthly IC is ``mean_ic`` plus month-to-month noise of IC_SWING,
    ranking a random ``coverage`` share of the stocks each month."""
    frames = []
    for i, m in enumerate(MONTHS):
        a = _pearson_for(mean_ic + IC_SWING * rng.standard_normal())
        comp = a * ret[i] + math.sqrt(1 - a * a) * rng.standard_normal(N_STOCKS)
        keep = rng.random(N_STOCKS) < coverage if coverage < 1 else np.ones(N_STOCKS, bool)
        frames.append(pd.DataFrame({"month": m, "Ticker": np.array(TICKERS)[keep], "Composite": comp[keep]}))
    return pd.concat(frames, ignore_index=True)


def _compare(seed: int, planted: float, coverage: float = 1.0) -> dict:
    rng, ret, fwd = _world(seed)
    base = _scores(rng, ret, 0.0)
    alt = _scores(rng, ret, planted, coverage)
    return X.summarise(X.per_month(alt, fwd, baseline_scores=base))


@pytest.mark.parametrize("planted", [0.03, 0.05, 0.10])
def test_the_paired_comparison_recovers_a_planted_difference(planted):
    s = _compare(20261020, planted)
    for period in ("presample", "discovery", "holdout", "all"):
        vs = s[period]["ic_vs_baseline"]
        assert abs(vs["mean"] - planted) < 4 * vs["se"], (period, vs, planted)


def test_a_planted_0_10_clears_the_engine_s_own_bar_on_the_held_out_months():
    s = _compare(7, 0.10)
    vs = s["holdout"]["ic_vs_baseline"]
    n_tests = len([e for e in X.load_registry() if e["transform"]["kind"] != "baseline"])
    assert vs["n"] == 38
    assert abs(vs["t"]) >= X.pass_bar(n_tests, vs["n"]), vs


def test_no_false_alarm_when_the_two_are_the_same():
    """Ten pairs with no true difference: none may clear the bar (each would by chance well under
    1% of the time), and the share reaching p < 0.05 stays near 5%."""
    n_tests = len([e for e in X.load_registry() if e["transform"]["kind"] != "baseline"])
    cleared, nominal = 0, 0
    for seed in range(10):
        vs = _compare(1000 + seed, 0.0)["all"]["ic_vs_baseline"]
        cleared += abs(vs["t"]) >= X.pass_bar(n_tests, vs["n"])
        nominal += X.p_value(vs) < 0.05
    assert cleared == 0
    assert nominal <= 3                       # 10 x 5% = 0.5 expected; 4+ has probability < 0.2%


def test_the_detectable_figure_is_honest():
    """``detectable`` says the smallest difference caught 80% of the time at the bar. Plant twice that
    on the held-out months and it must be caught every time across five seeds (power ~ 99.9%)."""
    n_tests = len([e for e in X.load_registry() if e["transform"]["kind"] != "baseline"])
    probe = _compare(99, 0.05)["holdout"]["ic_vs_baseline"]
    d = X.detectable(probe["se"], n_tests, probe["n"])
    assert d is not None and 0.02 < d < 0.2
    for seed in range(5):
        vs = _compare(500 + seed, 2 * d)["holdout"]["ic_vs_baseline"]
        assert abs(vs["t"]) >= X.pass_bar(n_tests, vs["n"]), (seed, vs, d)


def test_a_narrow_screen_is_measured_on_the_shared_stocks():
    """Method version 2's pairing: a screen ranking 70% of the stocks is compared with the baseline
    on those same stocks, so its planted difference comes back whole."""
    s = _compare(42, 0.05, coverage=0.7)
    vs = s["all"]["ic_vs_baseline"]
    assert 0.6 * N_STOCKS < s["all"]["n_common"] < 0.8 * N_STOCKS
    assert abs(vs["mean"] - 0.05) < 4 * vs["se"], vs
