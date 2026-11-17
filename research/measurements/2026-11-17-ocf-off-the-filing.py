"""Why does the run's operating cash flow sit over 10% from the filing for ten stocks?

The pre-registered re-run of the filing cross-check's range table
(`2026-11-17-filing-crosscheck-range.py`) found operating cash flow over 10% from the filing for
1-3 stocks through 2026-11-02 and for 8-10 on every run since 2026-11-03. This prints, for each
of today's ten, the source's quarterly and annual operating-cash-flow rows as the fetch sees
them, the figure the engine's own functions make of them, and which branch produced it.

    python research/measurements/2026-11-17-ocf-off-the-filing.py [TICKER ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

import factor_engine as fe  # noqa: E402

DEFAULT = ["BA", "BAC", "BNY", "AMZN", "APH", "WELL", "ILMN", "CPT", "DXCM", "CHRW"]
LABEL = "Operating Cash Flow"


def show(t: str) -> None:
    tk = yf.Ticker(t)
    q, a = tk.quarterly_cashflow, tk.cashflow
    print(f"\n=== {t}")
    for name, stmt in (("quarterly", q), ("annual", a)):
        idx = fe._find_stmt_label(stmt, LABEL) if stmt is not None and not stmt.empty else None
        if idx is None:
            print(f"  {name}: no row")
            continue
        row = fe._period_row(stmt, idx)
        print(f"  {name}: " + "  ".join(
            f"{pd.Timestamp(c).date()}={'nan' if pd.isna(v) else format(float(v) / 1e9, '.3f')}"
            for c, v in row.items()))
    partial, rolled = [], []
    v = fe._stmt_val_ltm(q, LABEL, partial_labels=partial, annual=a, rolled_labels=rolled)
    how = "rolled forward" if rolled else "3 quarters x 4/3" if partial else "four quarters summed"
    r, r_how = fe._ltm_rollforward(q, a, LABEL)
    print(f"  engine: {v / 1e9:.3f}bn by {how}"
          + (f"   [roll-forward alone: {r / 1e9:.3f}bn - {r_how}]" if not np.isnan(r) else "   [roll-forward: not available]"))


if __name__ == "__main__":
    for t in (sys.argv[1:] or DEFAULT):
        try:
            show(t)
        except Exception as exc:  # noqa: BLE001 - a diagnostic: report and carry on
            print(f"\n=== {t}: {type(exc).__name__}: {exc}")
