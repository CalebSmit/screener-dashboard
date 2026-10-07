"""Recompute a stock's score from the published payload alone.

WHY THIS EXISTS - 2026-10-07 (``plan/calculation-transparency.md``, stage T0b).

The drilldown prints its own arithmetic: metric percentile x weight = points, points
sum to a category score, category points sum to the composite. For 275 of 502 stocks
that arithmetic was false - the page printed the generic metric weight where the
engine had used bank, Piotroski-conditional or renormalised weights - and nothing
could see it, because nothing ever recomputed a score from what the page publishes.

This module does that recomputation. It reads **only the payload** (``stock_detail``,
``weights``) and deliberately does **not** import ``factor_engine``: two separate
implementations agreeing is the evidence, one implementation agreeing with itself is
not. It is used by the build (a payload that cannot reproduce its own scores is not
published), by ``tests/test_calculation_reproducibility.py`` and by
``scripts/audit_stock.py``.

Everything here is arithmetic on published numbers. No weight is resolved here: the
weight tables (``weights.profiles``) and each stock's table choice (``wp``) are emitted
by the engine's own ``metric_weight_profiles`` - this module only applies them.
"""

from __future__ import annotations

CATEGORIES = ["valuation", "quality", "growth", "momentum", "risk",
              "revisions", "size", "investment"]

# The payload stores floats to 4 decimal places, so a recomputed category score can
# differ from the published one by rounding only.
SCORE_TOL = 0.002
# Category contributions are rounded to 2dp by the engine and the composite is rounded
# to 2dp, so the composite chain is checked to a cent.
COMPOSITE_TOL = 0.011
CONTRIB_SUM_TOL = 0.05


def profile_table(weights: dict, cat: str, stock: dict) -> tuple[str, dict]:
    """(profile id, {metric: weight %}) the stock's ``cat`` score was built with."""
    profiles = (weights.get("profiles") or {}).get(cat) or {}
    pid = (stock.get("wp") or {}).get(cat, "generic")
    return pid, profiles.get(pid) or profiles.get("generic") or {}


def category_trace(weights: dict, cat: str, stock: dict) -> dict:
    """The full metric-level workings behind one category score.

    Each metric with a positive weight and a published percentile contributes
    ``percentile x share`` points, where ``share`` is its weight divided by the total
    weight of the metrics that *have data* (a metric with no data drops out and the
    rest are renormalised - the engine's rule, stated on the page).
    """
    pid, table = profile_table(weights, cat, stock)
    pct = stock.get("pct") or {}
    raw = stock.get("raw") or {}
    present = {m: w for m, w in table.items() if w > 0 and pct.get(m) is not None}
    total = sum(present.values())
    rows = []
    for m, w in table.items():
        if w <= 0:
            continue
        p = pct.get(m)
        has = p is not None
        share = (w / total) if (has and total > 0) else None
        rows.append({
            "metric": m,
            "raw": raw.get(m),
            "pct": p,
            "configured_weight": w,            # % within the profile
            "share": share,                    # fraction of this score, after renormalising
            "points": (p * share) if has and share is not None else None,
        })
    score = sum(r["points"] for r in rows if r["points"] is not None) if total > 0 else None
    return {
        "category": cat,
        "profile": pid,
        "metrics": rows,
        "weight_in_play": total,               # % of the profile that had data
        "score": score,
        "published_score": (stock.get("cat_scores") or {}).get(cat),
    }


def composite_trace(weights: dict, stock: dict) -> dict:
    """Category points -> composite, including the coverage discount."""
    fw = weights.get("factor_weights") or {}
    scores = stock.get("cat_scores") or {}
    scored = {c: scores[c] for c in CATEGORIES if scores.get(c) is not None and fw.get(c, 0) > 0}
    wsum = sum(fw[c] for c in scored)
    pre = sum(scored[c] * fw[c] / wsum for c in scored) if wsum > 0 else None
    cov = stock.get("cov") or {}
    disc = cov.get("disc") or 0.0
    composite = pre * (1 - disc) if pre is not None else None
    return {
        "pre_discount": pre,
        "discount": disc,
        "composite": composite,
        "published_composite": stock.get("composite"),
        "published_contrib_sum": sum((stock.get("contrib") or {}).get(c) or 0 for c in CATEGORIES),
    }


def verify_stock(weights: dict, stock: dict, check_composite: bool = True) -> list[str]:
    """Human-readable mismatches for one stock; empty list means it reproduces."""
    problems = []
    for cat in CATEGORIES:
        pub = (stock.get("cat_scores") or {}).get(cat)
        t = category_trace(weights, cat, stock)
        if pub is None and t["score"] is None:
            continue
        if pub is None or t["score"] is None:
            problems.append(f"{cat}: published {pub!r} but recomputed {t['score']!r}")
        elif abs(pub - t["score"]) > SCORE_TOL:
            problems.append(f"{cat}: published {pub:.4f}, recomputed {t['score']:.4f} "
                            f"(profile {t['profile']})")
    if check_composite:
        c = composite_trace(weights, stock)
        pub = c["published_composite"]
        if pub is not None and c["composite"] is not None:
            if abs(pub - c["composite"]) > COMPOSITE_TOL:
                problems.append(f"composite: published {pub:.2f}, recomputed "
                                f"{c['composite']:.2f} (discount {c['discount']:.4f})")
            if abs(c["published_contrib_sum"] - c["pre_discount"]) > CONTRIB_SUM_TOL:
                problems.append(f"points: sum {c['published_contrib_sum']:.2f} vs "
                                f"pre-discount composite {c['pre_discount']:.2f}")
    return problems


def verify_payload(payload: dict, check_composite: bool = True) -> dict:
    """Recompute every stock. Returns counts and the first few failures."""
    weights = payload.get("weights") or {}
    stocks = payload.get("stock_detail") or {}
    failures = {}
    cat_pairs = 0
    for t, s in stocks.items():
        for cat in CATEGORIES:
            if (s.get("cat_scores") or {}).get(cat) is not None:
                cat_pairs += 1
        probs = verify_stock(weights, s, check_composite)
        if probs:
            failures[t] = probs
    return {
        "stocks": len(stocks),
        "category_pairs": cat_pairs,
        "failing_stocks": len(failures),
        "failures": failures,
    }
