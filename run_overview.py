"""What moved across the whole run - two or three sentences per comparison baseline.

WHY THIS EXISTS - CLAUDE.md priority 4's residual, the last open piece of the 2026-08-10 owner
directive: the What Changed panel lists the stocks that moved, but nothing said what the run
*as a whole* did. A reader opening the page should be able to tell in one glance whether this
was a quiet day or a reshuffle, and whether it came from prices (momentum, risk, revisions)
or from new filings (valuation, quality, growth).

Built at build time from the published ``history`` block and the run's own factor weights,
like the per-stock summaries, so the page shows a sentence the build produced and tested
rather than one assembled in the browser. Every number in it is in the payload beside it.

Three sentences, each registered in ``claims.py`` (``history.run_overview``):

1. How many stocks moved materially, using the panel's own measured threshold.
2. Which categories the change in scores came from: each category's score change for each
   stock, multiplied by that category's weight in the composite, summed over all stocks
   **without regard to direction** - so it says where the movement was, not which way it went.
3. How many of the current top 25 were also in the top 25 at the baseline.

Never advice: the text passes ``stock_summary.advice_terms_in`` (test).
"""
from __future__ import annotations

CATEGORY_LABELS = {
    "valuation": "Valuation", "quality": "Quality", "growth": "Growth", "momentum": "Momentum",
    "risk": "Risk", "revisions": "Revisions", "size": "Size", "investment": "Investment",
}
FUNDAMENTAL = ("valuation", "quality", "growth")
TOP_N = 25


def category_shares(history: dict, base: str, factor_weights: dict) -> dict:
    """Each category's share of the summed |weight x score change| across all stocks."""
    tot = {c: 0.0 for c in CATEGORY_LABELS}
    for entry in (history.get("delta") or {}).values():
        cats = ((entry or {}).get(base) or {}).get("cat") or {}
        for c, v in cats.items():
            if c in tot and v is not None:
                tot[c] += abs(float(factor_weights.get(c, 0)) * float(v))
    s = sum(tot.values())
    return {c: v / s for c, v in tot.items()} if s > 0 else {}


def top_kept(history: dict, base_date: str, n: int = TOP_N) -> int | None:
    """How many of the current top ``n`` were also in the top ``n`` on ``base_date``."""
    dates = history.get("dates") or []
    cur = history.get("current_date")
    if base_date not in dates or cur not in dates:
        return None
    bi, ci = dates.index(base_date), dates.index(cur)
    kept = 0
    for s in (history.get("series") or {}).values():
        r = s.get("r") or []
        if len(r) > max(bi, ci) and r[ci] is not None and r[ci] <= n and r[bi] is not None and r[bi] <= n:
            kept += 1
    return kept


def _pct(x: float) -> int:
    return int(round(100 * x))


def _date(d: str) -> str:
    import datetime as _dt
    try:
        t = _dt.date.fromisoformat(d)
    except (TypeError, ValueError):
        return str(d)
    return f"{t.strftime('%b')} {t.day}"


def overview(history: dict, factor_weights: dict) -> dict:
    """``{base: {"text": [...], numbers...}}`` for each baseline the history carries."""
    out: dict = {}
    if not history or not history.get("available"):
        return out
    noise = history.get("noise") or {}
    thr = noise.get("material_threshold")
    for base in ("prev", "m1"):
        cmp = (history.get("compare") or {}).get(base)
        mv = (history.get("movers") or {}).get(base)
        if not cmp or not mv or thr is None:
            continue
        gap = cmp.get("gap_days")
        when = f"Since {_date(cmp['date'])} ({gap} day{'s' if gap != 1 else ''})"
        n_up, n_down = int(mv.get("n_up", 0)), int(mv.get("n_down", 0))
        text = []
        if n_up or n_down:
            text.append(f"{when}, {n_up} stock{'s' if n_up != 1 else ''} moved up the ranking and "
                        f"{n_down} moved down by {thr} places or more.")
        else:
            text.append(f"{when}, no stock moved {thr} places or more in either direction.")
        shares = category_shares(history, base, factor_weights)
        if shares:
            top = sorted(shares.items(), key=lambda kv: -kv[1])[:3]
            parts = [f"{CATEGORY_LABELS[c]} {_pct(v)}%" for c, v in top]
            fund = sum(shares.get(c, 0.0) for c in FUNDAMENTAL)
            first, rest = top[0], parts[1:]
            text.append("Of the movement in scores, weighted by each category's share of the composite, "
                        f"{CATEGORY_LABELS[first[0]]} accounted for {_pct(first[1])}%"
                        + ((", " + " and ".join(rest)) if rest else "")
                        + f"; Valuation, Quality and Growth together {_pct(fund)}%.")
        kept = top_kept(history, cmp["date"])
        if kept is not None:
            text.append(f"{kept} of today's top {TOP_N} {'was' if kept == 1 else 'were'} also in the top {TOP_N} "
                        f"on {_date(cmp['date'])}.")
        out[base] = {"text": text, "n_up": n_up, "n_down": n_down, "threshold": thr,
                     "shares": {c: round(v, 4) for c, v in shares.items()}, "top_kept": kept}
    return out
