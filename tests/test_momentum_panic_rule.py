"""The momentum regime rule: a cut in the market's panic state, and nothing else.

Built 2026-10-12 from the design pre-registered in ``research/2026-10-09-momentum-regime.md``.
The rule it replaces scaled momentum by the cross-stock spread of ``momentum_score``, which
does not measure market volatility: it called 30 of 33 runs "LOW VOL" and raised momentum's
weight on most days for a reason it did not measure. So these tests pin the three properties
that made the replacement worth building, each of which the old rule lacked:

1. **the input is the market** - the S&P 500 total-return index's own return and volatility;
2. **both conditions are required** (Daniel & Moskowitz 2016's panic state), and the
   thresholds are fixed in advance rather than fitted to this screener's runs;
3. **it only ever cuts** - no state raises momentum's weight, and a missing reading changes
   nothing.

No test here touches the network or a published file.
"""
import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import factor_engine as fe  # noqa: E402
import run_screener  # noqa: E402
from presets import PRESETS  # noqa: E402
from run_context import RunContext  # noqa: E402


def closes(annual_vol: float, daily_drift: float, days: int = 760, end: str = "2026-10-09") -> pd.Series:
    """A business-day close series with an exact shape: log returns alternate +s / -s around
    a drift, so realised volatility is s x 252^0.5 and the trend is the drift alone."""
    s = annual_vol / np.sqrt(252)
    lr = np.array([daily_drift + (s if i % 2 else -s) for i in range(days)])
    idx = pd.bdate_range(end=end, periods=days)
    return pd.Series(100 * np.exp(np.cumsum(lr)), index=idx)


FALLING, RISING = -0.0005, 0.0005        # about -22% and +29% over 24 months
WILD, CALM = 0.30, 0.10                  # either side of the 20.6% threshold


# --------------------------------------------------------------------------- the state
def test_panic_needs_both_conditions():
    assert fe.market_panic_state(closes(WILD, FALLING))["state"] == "PANIC"
    assert fe.market_panic_state(closes(CALM, FALLING))["state"] == "NORMAL"     # fell, but calm
    assert fe.market_panic_state(closes(WILD, RISING))["state"] == "NORMAL"      # wild, but up: 2020, 2022
    assert fe.market_panic_state(closes(CALM, RISING))["state"] == "NORMAL"


def test_the_readings_are_the_replays_definitions():
    """Same arithmetic as research/measurements/2026-10-09-momentum-panic-replay.py, so the
    7.5% the methodology quotes is a count of this rule and not of a cousin."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range(end="2026-10-09", periods=800)
    px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, 800))), index=idx)
    got = fe.market_panic_state(px)
    lr = np.log(px).diff().dropna()
    assert got["vol"] == pytest.approx(float((lr.rolling(126).std() * np.sqrt(252)).iloc[-1]))
    then = px[: px.index[-1] - pd.DateOffset(months=24)].iloc[-1]
    assert got["ret"] == pytest.approx(float(px.iloc[-1] / then - 1))
    assert got["as_of"] == "2026-10-09"
    assert (got["vol_threshold"], got["vol_window_days"], got["return_window_months"]) == (0.206, 126, 24)


def test_the_threshold_is_strict_and_configurable():
    s = closes(WILD, FALLING)
    vol = fe.market_panic_state(s)["vol"]
    assert fe.market_panic_state(s, {"vol_threshold": vol + 0.001})["state"] == "NORMAL"
    assert fe.market_panic_state(s, {"vol_threshold": vol - 0.001})["state"] == "PANIC"
    # a 12-month window on a series that fell only in its first year reads the recovery
    lr = np.r_[np.full(500, -0.001), np.full(260, 0.001)]
    v = pd.Series(100 * np.exp(np.cumsum(lr)), index=pd.bdate_range(end="2026-10-09", periods=760))
    assert fe.market_panic_state(v)["ret"] < 0 < fe.market_panic_state(v, {"return_window_months": 12})["ret"]


@pytest.mark.parametrize("series", [
    None,
    pd.Series(dtype=float),
    closes(WILD, FALLING, days=100),          # shorter than the volatility window
    closes(WILD, FALLING, days=400),          # does not reach back 24 months
])
def test_a_series_that_cannot_answer_gives_no_state(series):
    got = fe.market_panic_state(series)
    assert got["state"] == "UNKNOWN" and got["vol"] is None and got["ret"] is None


def test_a_timezone_aware_index_is_read_the_same():
    s = closes(WILD, FALLING)
    aware = s.copy()
    aware.index = aware.index.tz_localize("America/New_York")
    assert fe.market_panic_state(aware) == fe.market_panic_state(s)


def test_a_failed_fetch_is_an_empty_series_not_an_error(monkeypatch):
    import yfinance

    def boom(*a, **k):
        raise RuntimeError("offline")
    monkeypatch.setattr(yfinance, "Ticker", boom)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = fe.fetch_market_closes(max_retries=1)
    assert got.empty and fe.market_panic_state(got)["state"] == "UNKNOWN"


# --------------------------------------------------------------------------- the weights
def test_the_rule_only_ever_cuts_momentum():
    base = PRESETS["balanced"]["factor_weights"]
    cut = fe.apply_momentum_regime(base, "PANIC")
    freed = base["momentum"] * 0.30
    assert cut["momentum"] == round(base["momentum"] * 0.70, 2)
    assert cut["quality"] == round(base["quality"] + freed / 2, 2)
    assert cut["valuation"] == round(base["valuation"] + freed / 2, 2)
    assert sum(cut.values()) == pytest.approx(sum(base.values()), abs=0.02)
    assert {k: v for k, v in cut.items() if k not in ("momentum", "quality", "valuation")} == \
           {k: v for k, v in base.items() if k not in ("momentum", "quality", "valuation")}
    # nothing else is a state the rule acts on - including the retired rule's two names
    for other in ("NORMAL", "UNKNOWN", "LOW VOL", "HIGH VOL", "", None):
        assert fe.apply_momentum_regime(base, other) == base
    assert list(fe.MOMENTUM_REGIME_SCALE) == ["PANIC"] and fe.MOMENTUM_REGIME_SCALE["PANIC"] < 1
    for p in PRESETS.values():
        w = p["factor_weights"]
        assert fe.apply_momentum_regime(w, "PANIC")["momentum"] <= w["momentum"]


def _scores():
    return pd.DataFrame({"momentum_score": np.linspace(5, 95, 40)})


def _cfg(enabled=True):
    return {"factor_weights": dict(PRESETS["balanced"]["factor_weights"]),
            "momentum_regime": {"enabled": enabled}}


def test_the_run_is_reweighted_in_the_panic_state_only(tmp_path):
    panic, normal = fe.market_panic_state(closes(WILD, FALLING)), fe.market_panic_state(closes(CALM, RISING))
    cfg = _cfg()
    out = fe.adjust_momentum_weight(_scores(), cfg, str(tmp_path), market_state=panic)
    assert out["factor_weights"] == fe.apply_momentum_regime(cfg["factor_weights"], "PANIC")
    assert cfg["factor_weights"] == PRESETS["balanced"]["factor_weights"]        # the caller's copy is untouched
    assert fe.adjust_momentum_weight(_scores(), cfg, str(tmp_path), market_state=normal)["factor_weights"] \
        == cfg["factor_weights"]
    # switched off, even a panic changes nothing
    off = _cfg(enabled=False)
    assert fe.adjust_momentum_weight(_scores(), off, str(tmp_path), market_state=panic)["factor_weights"] \
        == off["factor_weights"]


@pytest.mark.parametrize("state", [None, {}, {"state": "UNKNOWN", "vol": None, "ret": None}])
def test_no_reading_changes_nothing_and_says_so(tmp_path, state):
    cfg = _cfg()
    with pytest.warns(UserWarning, match="No market-state reading"):
        out = fe.adjust_momentum_weight(_scores(), cfg, str(tmp_path), market_state=state)
    assert out["factor_weights"] == cfg["factor_weights"]


def test_the_score_spread_no_longer_decides_anything(tmp_path):
    """The retired rule ranked this run's spread against factor_vol_history.csv. Give it a
    history that would have called today both ways: the weights do not move."""
    for history in (np.full(30, 1.0), np.full(30, 99.0)):
        (tmp_path / "factor_vol_history.csv").write_text(
            "date,momentum_vol\n" + "".join(f"2026-01-{i + 1:02d},{v}\n" for i, v in enumerate(history)))
        cfg = _cfg()
        normal = fe.market_panic_state(closes(CALM, RISING))
        assert fe.adjust_momentum_weight(_scores(), cfg, str(tmp_path), market_state=normal)["factor_weights"] \
            == cfg["factor_weights"]


# --------------------------------------------------------------------------- what is published
@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def test_the_configured_thresholds_are_the_pre_registered_ones(cfg):
    """Fixed in advance from the index's own history. A session that moves one of these to
    fit recent data or a result has turned the rule back into what it replaced; change the
    research note and the changelog first, then this test."""
    rule = cfg["momentum_regime"]
    assert rule["enabled"] is True
    assert {k: rule[k] for k in fe.MOMENTUM_REGIME_DEFAULTS} == fe.MOMENTUM_REGIME_DEFAULTS \
        == {"vol_window_days": 126, "vol_threshold": 0.206, "return_window_months": 24}
    replay = (ROOT / "research/measurements/2026-10-09-momentum-panic-replay.txt").read_text()
    assert "126-day vol 80th pct = 20.6%" in replay and "442; panic state: 33 (7.5%)" in replay


def test_the_overview_states_the_rule_the_engine_applies(cfg):
    text = run_screener.build_screener_overview(cfg)
    para = next(l for l in text.splitlines() if l.startswith("**Momentum regime rule - currently on.**"))
    for must in ("past 24 months is negative", "past 126 trading days is above 20.6% a year",
                 "multiplied by 0.70", "split equally between Quality and Valuation",
                 "never raises Momentum's weight", "33 of 442 month-ends (7.5%)",
                 "not a performance test"):
        assert must in para, must
    # the numbers are read from the config and the engine, not typed into the page
    other = {**cfg, "momentum_regime": {"enabled": False, "vol_window_days": 63,
                                        "vol_threshold": 0.25, "return_window_months": 12}}
    alt = run_screener.build_screener_overview(other)
    assert "currently off" in alt and "past 12 months is negative" in alt
    assert "past 63 trading days is above 25.0% a year" in alt
    assert "volatile markets, but the input" not in text and "Momentum regime rule: off" not in text


def test_the_runs_reading_is_published_with_its_weights(tmp_path):
    reading = fe.market_panic_state(closes(CALM, RISING))
    stub = SimpleNamespace(run_dir=tmp_path)
    base = {"factor_weights": dict(PRESETS["balanced"]["factor_weights"]), "metric_weights": {}}
    path = RunContext.save_effective_weights(stub, base, momentum_regime=reading)
    data = json.loads(Path(path).read_text())
    assert data["momentum_regime"] == reading and data["factor_weights_adjusted"] is False
    # a run with the rule off, or a warm start, records no reading rather than an empty one
    path = RunContext.save_effective_weights(stub, base)
    assert "momentum_regime" not in json.loads(Path(path).read_text())


def test_the_run_passes_its_reading_to_the_rule():
    src = (ROOT / "run_screener.py").read_text(encoding="utf-8")
    assert 'stats["momentum_regime"] = market_panic_state(fetch_market_closes(), _rule)' in src
    assert 'cfg = adjust_momentum_weight(df, cfg, str(ROOT), market_state=stats.get("momentum_regime"),' in src
    assert 'momentum_regime=fe_stats.get("momentum_regime")' in src


def test_a_cache_scored_under_another_setting_is_not_served(cfg):
    stub = SimpleNamespace()
    on = RunContext.config_hash(stub, cfg)
    off = RunContext.config_hash(stub, {**cfg, "momentum_regime": {"enabled": False}})
    assert on != off


def test_the_page_prints_the_reading_from_the_engine():
    import generate_dashboard as gd
    html = gd.generate_html()
    assert "function methodologyRunNote()" in html and "methodologyRunNote();" in html
    assert "(D.weights || {}).momentum_regime" in html
    assert "LOW VOL" not in html and "HIGH VOL" not in html
