"""``config.yaml``'s ``universe:`` block is read, and its floor can fire (2026-11-19).

Two controls in that block did nothing from February 2026 to this date, and nothing noticed:

- ``index: 'SP500'  # Options: 'SP500', 'R1000', 'R2000'`` was read by no code. Asking for the
  Russell 2000 ran the S&P 500 and said nothing.
- ``min_market_cap: 2e9  # $2B minimum`` was handed to ``factor_engine.apply_universe_filters``,
  which looked for a ``marketCap`` column. Both of its callers pass the frame ``compute_metrics``
  returns, which carries ``size_log_mcap`` and no ``marketCap``, so the floor never excluded a
  stock (``research/measurements/2026-11-19-market-cap-floor-never-fires.py``).

The second survived because ``tests/test_edge_cases.py::TestUniverseFilters`` builds its frame by
hand *with* the column - the test and production disagreed about the shape of the input, and the
test won. So the tests here take the frame from ``compute_metrics`` itself (CLAUDE.md, settled
row "gates": assert a check against the producer's own output, not a hand-built stand-in).

Nothing published moves: the S&P 500's floor is 0 by a decision made on 2026-11-03
(``plan/universe-expansion.md``), and the filter was a no-op at every floor before today.
"""
from __future__ import annotations

import copy
import io
import contextlib
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import factor_engine as fe  # noqa: E402
import universe  # noqa: E402


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def metrics_frame(cfg) -> pd.DataFrame:
    """The frame both callers hand the filter: ``compute_metrics``'s own output, built from the
    raw records - not a hand-written DataFrame. Twenty companies from $0.5B to $10B, and one
    whose market cap the source did not return."""
    from tests.test_metrics import _make_rec
    caps = [0.5e9 * (i + 1) for i in range(20)] + [None]
    raw = [_make_rec(Ticker=f"U{i:02d}", marketCap=mc, sector="Industrials")
           for i, mc in enumerate(caps)]
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        df = fe.compute_metrics(raw, pd.Series(dtype=float), cfg)
    return df


def _with_floor(cfg: dict, floor: float) -> dict:
    c = copy.deepcopy(cfg)
    c["universe"]["min_market_cap"] = floor
    return c


# --- the floor ---------------------------------------------------------------------------

def test_the_frame_the_filter_is_handed_has_no_marketcap_column(metrics_frame):
    """The premise of this module. If ``compute_metrics`` ever starts carrying ``marketCap``,
    the fallback below is no longer what production exercises and this file should say so."""
    assert "marketCap" not in metrics_frame.columns
    assert "size_log_mcap" in metrics_frame.columns
    assert metrics_frame["size_log_mcap"].notna().sum() >= 10


def test_a_positive_floor_excludes_on_the_engines_own_frame(metrics_frame, cfg):
    """The defect, as a test: at a floor of the median market cap about half the rows must go.
    Before 2026-11-19 every row came back at any floor."""
    mc = np.exp(-metrics_frame["size_log_mcap"])
    floor = float(mc.median())
    out = fe.apply_universe_filters(metrics_frame.copy(), _with_floor(cfg, floor))
    expected = metrics_frame[~(mc.notna() & (mc < floor))]
    assert 0 < len(out) < len(metrics_frame)
    assert list(out["Ticker"]) == list(expected["Ticker"])
    # and a floor nothing can pass removes every stock whose market cap is known
    none_left = fe.apply_universe_filters(metrics_frame.copy(), _with_floor(cfg, 1e15))
    assert len(none_left) == int(mc.isna().sum())


def test_an_unknown_market_cap_is_kept(metrics_frame, cfg):
    unknown = list(metrics_frame.loc[metrics_frame["size_log_mcap"].isna(), "Ticker"])
    assert unknown == ["U20"], "the fixture's one record with no market cap"
    out = fe.apply_universe_filters(metrics_frame.copy(), _with_floor(cfg, 1e15))
    assert list(out["Ticker"]) == unknown


def test_a_floor_that_cannot_be_applied_stops_the_run(cfg):
    """Neither column: the old code returned the frame untouched. Now it refuses."""
    df = pd.DataFrame({"Ticker": ["A", "B"], "roic": [0.1, 0.2]})
    with pytest.raises(ValueError, match="cannot be applied"):
        fe.apply_universe_filters(df, _with_floor(cfg, 2e9))
    # a floor of zero asks for nothing, so it needs nothing
    assert len(fe.apply_universe_filters(df, _with_floor(cfg, 0))) == 2


def test_the_two_routes_to_a_market_cap_agree():
    """``marketCap`` where a caller has it, ``exp(-size_log_mcap)`` where it does not: the same
    stocks go either way."""
    caps = [10e9, 1.9e9, 2.1e9, 5e8, np.nan]
    a = pd.DataFrame({"Ticker": list("VWXYZ"), "marketCap": caps})
    b = pd.DataFrame({"Ticker": list("VWXYZ"),
                      "size_log_mcap": [-math.log(c) if c == c else np.nan for c in caps]})
    c = {"universe": {"min_market_cap": 2e9}}
    assert list(fe.apply_universe_filters(a, c)["Ticker"]) == ["V", "X", "Z"]
    assert list(fe.apply_universe_filters(b, c)["Ticker"]) == ["V", "X", "Z"]


def test_the_sp500_floor_is_zero_and_the_config_says_why(cfg):
    """Pre-registration 3's resolution (plan/universe-expansion.md, 2026-11-03): one rule for
    every S&P index band. A session that sets a positive floor here is now changing which index
    members are ranked - a methodology change - where before today it changed nothing."""
    assert float(cfg["universe"]["min_market_cap"]) == 0
    text = (ROOT / "config.yaml").read_text(encoding="utf-8")
    assert "$2B minimum" not in text
    assert "R1000" not in text.split("universe:")[1].split("\n\n")[0]


# --- the index name ----------------------------------------------------------------------

def test_the_configured_index_is_the_universe_every_constant_reads(cfg):
    assert universe.from_config(cfg) is universe.SP500
    assert universe.CURRENT is universe.from_config(cfg)
    assert universe.UNIVERSES[universe.SP500.key] is universe.SP500


@pytest.mark.parametrize("name", ["SP500", "sp500", "S&P 500", "sp-500"])
def test_the_name_is_read_without_case_or_punctuation(name):
    assert universe.from_name(name) is universe.SP500


@pytest.mark.parametrize("name", ["R1000", "R2000", "SP400", "", None])
def test_a_universe_with_no_definition_is_refused(name):
    """'R1000' and 'R2000' are the two values the old config comment offered."""
    with pytest.raises(ValueError, match="no definition"):
        universe.from_name(name)


def test_a_config_without_the_key_still_means_the_sp500():
    """Old run snapshots and tests' partial dicts carry no ``universe.index``."""
    assert universe.from_config({}) is universe.SP500
    assert universe.from_config(None) is universe.SP500
    assert universe.from_config({"universe": {"min_market_cap": 0}}) is universe.SP500


def test_the_schema_refuses_an_undefined_index_and_a_negative_floor(cfg):
    import schemas
    schemas.RunConfig(**cfg)                      # the committed config validates
    bad = copy.deepcopy(cfg)
    bad["universe"]["index"] = "R2000"
    with pytest.raises(Exception, match="no definition"):
        schemas.RunConfig(**bad)
    neg = copy.deepcopy(cfg)
    neg["universe"]["min_market_cap"] = -1
    with pytest.raises(Exception):
        schemas.RunConfig(**neg)
    assert schemas.RunConfig.UniverseConfig().min_market_cap == 0


def test_the_run_resolves_its_universe_from_its_own_config():
    """``run_factor_engine`` must ask the run's cfg, not the import-time constant, and must hand
    the answer to ``assign_size_bands`` - otherwise a second universe's run is banded as the
    first's."""
    src = (ROOT / "run_screener.py").read_text(encoding="utf-8")
    assert "run_universe = universe.from_config(cfg)" in src
    assert "universe.assign_size_bands(df, run_universe)" in src
    assert src.index("run_universe = universe.from_config(cfg)") < src.index("get_sp500_tickers(cfg)\n")
