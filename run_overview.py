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

And one more since 2026-11-05 (``history.market_move``), the day the S&P 500 fell 4.4%: how far the
market moved between the two runs' prices, and how far the ranking's *order* moved with it - the
Spearman rank correlation of the two rankings over the stocks ranked in both. A reader watching
the index fall wants to know whether the list fell apart; this says so in two numbers, both in
the payload beside the sentence. The market figure is the S&P 500 total-return index (^SP500TR)
the run already fetches for the momentum rule, saved with the run (``market_closes.json``), from
the close the comparison run was priced at to the close this one was: a run is priced at the
last session whose 16:00 New York close came before it started.

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


CLOSE_HOUR_ET = 16          # NYSE regular-session close, America/New_York


def session_of(started_at, sessions):
    """The market session a run started at ``started_at`` was priced at: the latest of
    ``sessions`` (``YYYY-MM-DD``, the index's own trading days, so holidays need no calendar)
    whose 16:00 New York close is at or before the start. A naive timestamp is read in this
    machine's zone, which is how ``meta.json`` writes it. None when nothing precedes it, and
    None for a run that started during a session (09:30-16:00 New York), which was priced at
    no close."""
    import datetime as _dt
    if not started_at or not sessions:
        return None
    try:
        t = _dt.datetime.fromisoformat(str(started_at))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.astimezone()
    try:
        from zoneinfo import ZoneInfo
        ny = ZoneInfo("America/New_York")
    except Exception:  # noqa: BLE001 - no tz database: no claim about which close
        return None
    # A run started while the market was open was priced mid-session, at no close at all (the
    # 2026-10-20 run started 13:21): say nothing rather than name the wrong one.
    t_ny = t.astimezone(ny)
    if t_ny.strftime("%Y-%m-%d") in sessions and (9, 30) <= (t_ny.hour, t_ny.minute) < (CLOSE_HOUR_ET, 0):
        return None
    best = None
    for d in sorted(sessions):
        close = _dt.datetime.fromisoformat(d).replace(hour=CLOSE_HOUR_ET, tzinfo=ny)
        if close <= t:
            best = d
        else:
            break
    return best


def market_move(closes, from_started, to_started):
    """The index's return between the closes two runs were priced at, or None when either
    session is not in the series or both runs were priced at the same close."""
    if not closes:
        return None
    sessions = sorted(closes)
    a, b = session_of(from_started, sessions), session_of(to_started, sessions)
    if not a or not b or a >= b:
        return None
    pa, pb = closes.get(a), closes.get(b)
    if not pa or not pb or pa <= 0:
        return None
    return {"ret": round(pb / pa - 1, 5), "from": a, "to": b}


def _spearman(x, y):
    import pandas as pd
    if len(x) < 3:
        return None
    rho = pd.Series(x, dtype=float).corr(pd.Series(y, dtype=float), method="spearman")
    return None if rho != rho else float(rho)


def _pair(history, i, j):
    x, y = [], []
    for s in (history.get("series") or {}).values():
        r = s.get("r") or []
        if len(r) > max(i, j) and r[i] is not None and r[j] is not None:
            x.append(r[i])
            y.append(r[j])
    return _spearman(x, y)


def rank_correlation(history, base_date):
    """Spearman correlation of the published ranks on ``base_date`` and today, over the stocks
    the history ranks on both - the statistic the publish gate reads (check_run_health
    ``rank_continuity``) and the history's own acceptance test (``history.rank_continuity``)."""
    dates = history.get("dates") or []
    cur = history.get("current_date")
    if base_date not in dates or cur not in dates:
        return None
    return _pair(history, dates.index(base_date), dates.index(cur))


def typical_run_to_run(history):
    """The median rank correlation between consecutive comparable runs across the history, so a
    reader can tell an ordinary day's figure from an unusual one. None under three pairs."""
    dates = history.get("dates") or []
    vals = sorted(v for v in (_pair(history, i - 1, i) for i in range(1, len(dates))) if v is not None)
    if len(vals) < 3:
        return None
    m = len(vals) // 2
    return vals[m] if len(vals) % 2 else (vals[m - 1] + vals[m]) / 2


def with_market(history, market):
    """Attach ``compare[base]["market"]`` - the index's move between the two runs' closes - from
    the run's own ``market_closes.json``. Returns ``history`` unchanged without one."""
    closes = (market or {}).get("closes") or {}
    if not history or not history.get("available") or not closes:
        return history
    cmp = dict(history.get("compare") or {})
    for base, entry in list(cmp.items()):
        if not entry:
            continue
        mv = market_move(closes, entry.get("started_at"), history.get("current_started_at"))
        if mv:
            cmp[base] = dict(entry, market=dict(mv, src=str((market or {}).get("source") or "^SP500TR")))
    return dict(history, compare=cmp)


def move_words(ret):
    """'fell 4.4%' / 'rose 0.6%' / 'was flat' - one decimal, the figure the page prints."""
    figure = f"{abs(ret) * 100:.1f}%"
    if figure == "0.0%":
        return "was flat"
    return ("rose " if ret > 0 else "fell ") + figure


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
        mkt = cmp.get("market")
        rho = rank_correlation(history, cmp["date"])
        if rho is not None:
            lead = (f"Over the same span the S&P 500 {move_words(mkt['ret'])} with dividends included "
                    f"({_date(mkt['from'])} close to {_date(mkt['to'])} close), and the "
                    if mkt else "The ")
            text.append(f"{lead}two rankings have a rank correlation of {rho:.3f}, where 1.000 "
                        f"is the same order.")
        kept = top_kept(history, cmp["date"])
        if kept is not None:
            text.append(f"{kept} of today's top {TOP_N} {'was' if kept == 1 else 'were'} also in the top {TOP_N} "
                        f"on {_date(cmp['date'])}.")
        out[base] = {"text": text, "n_up": n_up, "n_down": n_down, "threshold": thr,
                     "shares": {c: round(v, 4) for c, v in shares.items()}, "top_kept": kept,
                     "rho": None if rho is None else round(rho, 3)}
        if base == "prev":
            typ = typical_run_to_run(history)
            out[base]["rho_typical"] = None if typ is None else round(typ, 3)
    return out
