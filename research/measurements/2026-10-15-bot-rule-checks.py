"""Checks on the bot-rule evaluation (research/2026-10-15-research-engine-first-batch.md section 9).

Written after the evaluation ran, to test whether its result is the ranking or the simulation:
(1) a plain equal-weighted top 25, rebalanced in full every month, from the same rankings;
(2) how many of the simulated sells were for a name missing from the month's ranking rather than
past the band; (3) the worst calendar years for the rule against the universe. Diagnostic only - it
registers nothing and changes no verdict. Not decision-grade (rule 5).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import v2_experiments as X  # noqa: E402


def main() -> None:
    months = X.load_months()
    fwd = X.forward_returns(sorted(months))
    inputs = X.bot_month_inputs(months, X.base_config())
    rule = X.load_evaluations()[0]["rule"]
    pm = X.simulate_bot(inputs, fwd, rule)

    rows, missing_sells, band_sells = [], 0, 0
    prev = None
    for m in sorted(inputs):
        if m not in fwd.index:
            continue
        g = inputs[m]
        top = g.nsmallest(25, "Rank")["Ticker"]
        r = fwd.loc[m]
        rows.append({"month": m, "top25": float(r.reindex(top).fillna(0).mean()),
                     "universe": float(r.reindex(g["Ticker"]).mean())})
        if prev is not None:
            ranks = dict(zip(g["Ticker"], g["Rank"]))
            for t in prev:
                if t not in ranks:
                    missing_sells += 1
                elif ranks[t] > rule["sell_rank"]:
                    band_sells += 1
        prev = set(g.nsmallest(40, "Rank")["Ticker"])
    t25 = pd.DataFrame(rows).set_index("month")
    t25["excess"] = t25["top25"] - t25["universe"]
    for name, (a, b) in (("discovery", X.DISCOVERY), ("holdout", X.HOLDOUT), ("presample", X.PRESAMPLE)):
        s = t25.loc[(t25.index >= a) & (t25.index <= b), "excess"]
        q = pm.loc[(pm.index >= a) & (pm.index <= b), "excess"]
        print(f"{name:10} plain top 25: {s.mean() * 100:+.2f}%/mo (t {s.mean() / s.std(ddof=1) * np.sqrt(len(s)):+.2f})"
              f" | bot rule: {q.mean() * 100:+.2f}%/mo (t {q.mean() / q.std(ddof=1) * np.sqrt(len(q)):+.2f})")
    print(f"\nnames leaving the top 40 by the next month: {band_sells} ranked past it, "
          f"{missing_sells} missing from the next ranking")
    pm["year"] = pm.index.str[:4]
    by = pm.groupby("year").agg(rule=("ret", lambda x: (1 + x).prod() - 1),
                                universe=("universe", lambda x: (1 + x).prod() - 1))
    by["gap"] = by["rule"] - by["universe"]
    print("\n" + (by * 100).round(1).to_string())


if __name__ == "__main__":
    main()
