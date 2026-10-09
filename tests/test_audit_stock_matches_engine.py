"""scripts/audit_stock.py reproduces the published scores independently, and the public
methodology page tells readers to run it. Its percentile must be the engine's: on 2026-10-09 the
engine moved to the midpoint rank and the script did not, so it reported 0 of 501 reproducing."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from factor_engine import _directed_pct  # noqa: E402


def _audit():
    spec = importlib.util.spec_from_file_location("audit_stock", ROOT / "scripts" / "audit_stock.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_scripts_percentile_is_the_engines(seed):
    rng = np.random.default_rng(seed)
    vals = list(np.round(rng.normal(size=37), 1))          # rounding makes ties
    eng = _directed_pct(pd.Series(vals), True)
    a = _audit()
    for v, e in zip(vals, eng):
        assert a._avg_rank_pct(vals, v) == pytest.approx(e)
