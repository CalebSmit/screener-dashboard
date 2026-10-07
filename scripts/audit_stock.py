#!/usr/bin/env python3
"""Audit one stock's score, or every stock's, from the published payload alone.

``python scripts/audit_stock.py JPM``       print the full workings and check them
``python scripts/audit_stock.py --all``     recompute every stock; exit 1 on any mismatch
``python scripts/audit_stock.py --sample 25``  a date-seeded sample (same set all day)

WHY THIS EXISTS - 2026-10-07 (``plan/calculation-transparency.md``, stage T5).

The page now shows the numbers behind every score. Two implementations agreeing is the
evidence that they are right, so this reads **only** ``dashboard_data.js`` and shares no
code with ``factor_engine``'s scoring: category scores and the composite come from
``calc_trace`` (the weight tables and table choices the payload publishes), the metric
equations from ``metric_lineage.RECOMPUTE``, percentiles from the raw values of the peers
they were ranked against, and Piotroski / Beneish from their components.

It is the thing to run when someone asks "how do I know this 63 is right?": it prints
every step, then says whether it reproduced.
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import calc_trace  # noqa: E402
import metric_lineage as ml  # noqa: E402

MIN_PEERS_DEFAULT = 10


def load(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    return json.loads(text[text.find("{"):text.rfind("}") + 1])


def _avg_rank_pct(values, mine):
    below = sum(1 for v in values if v < mine)
    equal = sum(1 for v in values if v == mine)
    return ((below + (equal + 1) / 2.0) / len(values)) * 100.0


def percentile_from_peers(payload: dict, ticker: str, metric: str, lower_is_better: bool):
    """Recompute a stock's sector percentile for one metric from its peers' raw values."""
    stocks = payload["stock_detail"]
    mine = stocks[ticker]["raw"].get(metric)
    if mine is None:
        return None
    sector = stocks[ticker]["sector"]
    universe = [s["raw"][metric] for s in stocks.values() if s["raw"].get(metric) is not None]
    peers = [s["raw"][metric] for s in stocks.values()
             if s["sector"] == sector and s["raw"].get(metric) is not None]
    min_peers = payload.get("sector_min_peers", MIN_PEERS_DEFAULT)
    pool = peers if len(peers) >= min_peers else universe
    pct = _avg_rank_pct(pool, mine)
    return 100.0 - pct if lower_is_better else pct


def audit(payload: dict, ticker: str, verbose: bool = True) -> list[str]:
    """Print the workings for ``ticker`` and return a list of problems (empty = reproduces)."""
    s = payload["stock_detail"][ticker]
    weights = payload["weights"]
    meta = payload.get("metric_meta", {})
    problems: list[str] = []
    out = print if verbose else (lambda *a, **k: None)

    out(f"\n{ticker} - {s['company']} ({s['sector']})   rank {s['rank']} of {len(payload['stock_detail'])}")
    out("=" * 78)

    for cat in calc_trace.CATEGORIES:
        t = calc_trace.category_trace(weights, cat, s)
        pub = t["published_score"]
        out(f"\n{cat.upper()}  weighted with the '{t['profile']}' table; "
            f"{sum(1 for r in t['metrics'] if r['pct'] is not None)} of {len(t['metrics'])} metrics have data")
        for r in t["metrics"]:
            lower = meta.get(r["metric"], {}).get("dir") == "lower"
            m = r["metric"]
            if r["pct"] is None:
                out(f"  {m:24s} no data")
                continue
            re_pct = percentile_from_peers(payload, ticker, m, lower)
            pct_ok = re_pct is not None and abs(re_pct - r["pct"]) <= 0.6
            fn = ml.RECOMPUTE.get(m)
            eq = ""
            if fn is not None:
                calc = fn(s.get("inp") or {})
                if r["raw"] is not None and calc is not None:
                    ok = abs(calc - r["raw"]) <= 1e-4 * max(1.0, abs(calc)) + 1e-4
                    eq = f"  equation {'ok' if ok else 'MISMATCH'} ({calc:.4f})"
                    if not ok:
                        problems.append(f"{cat}/{m}: inputs give {calc:.6f}, published {r['raw']}")
                else:
                    eq = "  equation: not rebuilt"
            if not pct_ok:
                problems.append(f"{cat}/{m}: percentile {r['pct']:.2f} vs {re_pct if re_pct is None else round(re_pct, 2)} from peers")
            raw = "none" if r["raw"] is None else f"{r['raw']:.4f}"
            out(f"  {m:24s} raw {raw:>12s}  pct {r['pct']:6.2f} (peers {re_pct:6.2f} {'ok' if pct_ok else 'MISMATCH'})"
                f"  weight {r['share'] * 100:5.1f}%  points {r['points']:6.2f}{eq}")
        if t["score"] is None and pub is None:
            out("  (category not scored)")
            continue
        ok = pub is not None and t["score"] is not None and abs(t["score"] - pub) <= calc_trace.SCORE_TOL
        out(f"  -> recomputed {t['score']}  published {pub}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            problems.append(f"{cat}: recomputed {t['score']} vs published {pub}")

    c = calc_trace.composite_trace(weights, s)
    out("\nCOMPOSITE")
    fw = weights.get("factor_weights", {})
    for cat in calc_trace.CATEGORIES:
        sc = (s.get("cat_scores") or {}).get(cat)
        if sc is None:
            out(f"  {cat:11s} not scored")
        else:
            out(f"  {cat:11s} score {sc:7.3f} x weight {fw.get(cat, 0):5.2f}  published points {(s.get('contrib') or {}).get(cat)}")
    out(f"  points add up to {c['pre_discount']:.4f}  (published sum {c['published_contrib_sum']:.2f})")
    if c["discount"]:
        out(f"  coverage discount {c['discount'] * 100:.2f}%  -> {c['composite']:.4f}")
    ok = c["composite"] is not None and abs(c["composite"] - c["published_composite"]) <= calc_trace.COMPOSITE_TOL
    out(f"  -> recomputed {c['composite']}  published {c['published_composite']}  {'OK' if ok else 'MISMATCH'}")
    if not ok:
        problems.append(f"composite: {c['composite']} vs {c['published_composite']}")

    pio = s.get("pio")
    if pio and s["raw"].get("piotroski_f_score") is not None:
        if pio.count("1") != round(s["raw"]["piotroski_f_score"]):
            problems.append(f"piotroski: signals {pio} vs score {s['raw']['piotroski_f_score']}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ticker", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--payload", default=str(ROOT / "dashboard_data.js"))
    args = ap.parse_args(argv)

    payload = load(Path(args.payload))
    stocks = sorted(payload["stock_detail"])

    if args.ticker:
        t = args.ticker.upper()
        if t not in payload["stock_detail"]:
            print(f"{t} is not in the payload")
            return 2
        problems = audit(payload, t)
        print("\nREPRODUCES" if not problems else "\nDOES NOT REPRODUCE:\n  " + "\n  ".join(problems))
        return 1 if problems else 0

    if args.all or args.sample:
        picked = stocks
        if args.sample:
            rng = random.Random(datetime.date.today().isoformat())
            picked = rng.sample(stocks, min(args.sample, len(stocks)))
        bad = {}
        for t in picked:
            p = audit(payload, t, verbose=False)
            if p:
                bad[t] = p
        print(f"audited {len(picked)} stocks: {len(picked) - len(bad)} reproduce, {len(bad)} do not")
        for t, p in list(bad.items())[:10]:
            print(f"  {t}: {p[0]}")
        return 1 if bad else 0

    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
