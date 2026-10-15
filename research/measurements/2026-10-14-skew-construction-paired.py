"""How far apart are the old and new put-skew constructions, on the same chains?

The page cites Xing, Zhang & Zhao (2010) and, until 2026-10-14, printed a different measure
beside their effect size:

    old = IV(put nearest 0.90 * spot)  -  mean(IV(ATM call), IV(ATM put))
    new = IV(OTM put nearest 0.95, within (0.80, 0.95))  -  IV(ATM call nearest 1, in [0.95, 1.05])

The switch is justified by the definition, not by this number - the page quotes the paper's
figure, so the measure under it has to be the paper's measure. This script says *how much* the
displayed number moves, which is what a reader of the changelog entry wants to know, and
whether the new construction is available for more or fewer stocks.

Both measures are computed from the **same fetched chain** per ticker, so the comparison is
paired and nothing is attributable to quotes moving between two runs.

Deliberately a sample, not the universe: a full pass is ~500 option-chain requests, which is
what the after-close cache exists to avoid (plan/context-layer.md item 2).

Usage: python -I -X utf8 research/measurements/2026-10-14-skew-construction-paired.py [N]
Reads the network. Writes nothing.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import context_signals as cs  # noqa: E402
from options_cache import market_session_date  # noqa: E402

IV_MIN, IV_MAX = cs.IV_MIN, cs.IV_MAX


def old_skew(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> float | None:
    """The construction in force until 2026-10-14, reproduced verbatim from git history."""
    common = sorted(set(calls["strike"]).intersection(set(puts["strike"])))
    if not common:
        return None
    atm = min(common, key=lambda k: abs(k - spot))
    if abs(atm - spot) / spot > 0.05:
        return None
    c_row = calls.loc[calls["strike"] == atm].iloc[0].to_dict()
    p_row = puts.loc[puts["strike"] == atm].iloc[0].to_dict()
    ivs = [r.get("impliedVolatility") for r in (c_row, p_row)]
    ivs = [float(v) for v in ivs if v is not None and np.isfinite(v) and IV_MIN <= v <= IV_MAX]
    if len(ivs) != 2:
        return None
    atm_iv = sum(ivs) / 2
    p90 = puts.iloc[(puts["strike"] - 0.9 * spot).abs().argsort()[:1]]
    if not len(p90):
        return None
    k = float(p90["strike"].iloc[0])
    iv90 = p90["impliedVolatility"].iloc[0]
    if abs(k / spot - 0.9) > 0.04 or iv90 is None or not np.isfinite(iv90):
        return None
    if not (IV_MIN <= iv90 <= IV_MAX):
        return None
    return float(iv90 - atm_iv)


def main(argv=None) -> int:
    n = int((argv or sys.argv[1:] or ["25"])[0])
    import yfinance as yf

    universe = json.loads((ROOT / "sp500_tickers.json").read_text(encoding="utf-8"))
    # Every 20th name, so the sample spans the alphabet and the sectors rather than the A's.
    step = max(1, len(universe) // n)
    tickers = [r["Ticker"] for r in universe[::step]][:n]
    today = market_session_date()

    rows = []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            spot = float(tk.fast_info["lastPrice"])
            exp, spans = cs.choose_expiry(tk.options, today, None)
            if not exp:
                rows.append({"Ticker": t, "note": "no expiry"})
                continue
            ch = tk.option_chain(exp)
            new = cs._xzz_skew(ch.calls, ch.puts, spot, (date.fromisoformat(exp) - today).days)
            rows.append({
                "Ticker": t, "exp": exp, "days": (date.fromisoformat(exp) - today).days,
                "old": old_skew(ch.calls, ch.puts, spot),
                "new": new.get("_ctx_opt_skew"),
                "new_put_k": new.get("_ctx_opt_skew_strike"),
                "new_call_k": new.get("_ctx_opt_skew_call_strike"),
                "in_window": new.get("_ctx_opt_skew_in_window"),
                "spot": spot,
            })
        except Exception as exc:  # noqa: BLE001 - a measurement reports what it could not read
            rows.append({"Ticker": t, "note": f"{type(exc).__name__}: {exc}"[:70]})

    df = pd.DataFrame(rows)
    print(f"session date {today}, sample of {len(df)} tickers\n")
    print(df.to_string(index=False, max_colwidth=28))

    both = df.dropna(subset=["old", "new"]) if {"old", "new"} <= set(df.columns) else df.iloc[0:0]
    print(f"\nold available: {df['old'].notna().sum() if 'old' in df else 0}"
          f"  new available: {df['new'].notna().sum() if 'new' in df else 0}"
          f"  both: {len(both)}")
    if len(both):
        d = (both["new"] - both["old"]) * 100
        print(f"new - old, in IV points: mean {d.mean():+.2f}  median {d.median():+.2f}  "
              f"min {d.min():+.2f}  max {d.max():+.2f}")
        print(f"spearman(old, new) = {both['old'].corr(both['new'], method='spearman'):.3f}  "
              f"pearson = {both['old'].corr(both['new']):.3f}")
        print(f"sign disagreements (one steep, one flat): "
              f"{int(((both['old'] > 0) != (both['new'] > 0)).sum())} of {len(both)}")
        print(f"expiry inside the paper's 10-60 day window: "
              f"{int(both['in_window'].sum())} of {len(both)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
