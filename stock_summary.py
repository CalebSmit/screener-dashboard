"""Deterministic per-stock summaries.

Priority 4 / owner directive 2026-08-10, specified in
``plan/dashboard-north-star.md``: *replace the "Screener AI" chat with
generated per-stock summaries.*

The chat asked every visitor to paste their own Anthropic API key into
``localStorage`` and then called ``api.anthropic.com`` from the browser. For
the investment-club audience that meant a paid API account per student, a
lesson in pasting credentials into web pages, a per-question cost, and - the
part that actually mattered - **two students asking the same question got
different answers.** The tool's whole claim is auditability, so an
un-reproducible explainer works against it.

What replaces it is a template over fields the payload already carries. Every
sentence below is arithmetic on ``contrib``, ``cat_scores``, ``pct``, ``raw``,
``peers``, ``flags``, the analyst targets and the history spine. So a summary
is exact, identical for every viewer, free, instant, and cannot hallucinate.

Two rules govern the wording, and both are enforced by tests:

1. **A summary explains why a stock ranks where it does. It never says whether
   to buy it.** ``BANNED_TERMS`` is the machine-checkable form of the line in
   ``plan/dashboard-north-star.md``; ``advice_terms_in()`` finds any breach.
   "Ranks 1st, driven by valuation and momentum" is the good kind. "Attractive
   entry point" is the kind that would make this a liability for a student
   club rather than a teaching tool.
2. **Metric percentiles are sector-relative** (``factor_engine.compute_sector_percentiles``),
   so the text says "sector percentile" and never implies a universe ranking.
   The composite *is* a universe percentile, and the text says so once,
   because that is a thing a student needs told.

Summaries are built here, at build time, and baked into the payload - not
computed in the browser - so what shipped is what a reader can diff.
"""

from __future__ import annotations

import re

CATEGORIES = [
    "valuation", "quality", "growth", "momentum",
    "risk", "revisions", "size", "investment",
]

CAT_LABELS = {
    "valuation": "Valuation", "quality": "Quality", "growth": "Growth",
    "momentum": "Momentum", "risk": "Risk", "revisions": "Revisions",
    "size": "Size", "investment": "Investment",
}

# Below this many analysts, the mean target is one or two opinions wide and the
# reader should be told. 5 is the threshold `check_run_health` already treats
# as thin coverage territory; nothing here depends on the exact number.
THIN_ANALYST_COUNT = 5

# Words that turn an explanation into a recommendation. Matched case-insensitively
# on word boundaries by `advice_terms_in()`. This list is deliberately blunt: a
# false positive costs one rephrased sentence, a false negative ships investment
# advice from a public site.
BANNED_TERMS = [
    "buy", "sell", "hold onto", "undervalued", "overvalued", "cheap",
    "expensive", "attractive", "unattractive", "bargain", "recommend",
    "recommendation", "outperform", "underperform", "should own",
    "worth owning", "avoid", "opportunity", "poised", "we think",
    "we like", "compelling", "must-own", "screaming", "no-brainer",
]

_BANNED_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in BANNED_TERMS) + r")\b",
    re.IGNORECASE,
)


def advice_terms_in(text: str) -> list[str]:
    """Return every banned recommendation term appearing in ``text``."""
    return [m.group(0) for m in _BANNED_RE.finditer(text or "")]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _ordinal(n: int) -> str:
    """1 -> '1st', 2 -> '2nd', 11 -> '11th', 23 -> '23rd'."""
    n = int(n)
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _pctile_phrase(p: float) -> str:
    """"the 97th", "the lowest", "the highest".

    A sector percentile is ``rank(pct=True) * 100``, so its extremes really do
    land on 0 and 100 - and "the 0th sector percentile" reads like a bug rather
    than like "worst in its sector", which is what it means.
    """
    n = round(p)
    if n <= 0:
        return "the lowest"
    if n >= 100:
        return "the highest"
    return f"the {_ordinal(n)}"


def _fmt_metric(value, fmt: str) -> str:
    """Mirror of ``fmtMetric`` in the emitted JS, so prose and table agree."""
    if value is None:
        return "n/a"
    if fmt == "pct":
        return f"{value * 100:.1f}%"
    if fmt == "int":
        return f"{round(value):.0f}"
    return f"{value:.2f}"


def _fmt_money(value) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.2f}"


def _num(value):
    """Coerce to float, or None for missing/non-numeric."""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # NaN


def _join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"


# ---------------------------------------------------------------------------
# Sentence builders. Each returns a string or None; None means "no honest
# sentence can be made from the data present", and the fact is simply omitted
# rather than guessed at.
# ---------------------------------------------------------------------------

def _scored_categories(detail: dict) -> list[str]:
    cats = detail.get("cat_scores") or {}
    return [c for c in CATEGORIES if _num(cats.get(c)) is not None]


def _sentence_rank(detail: dict, universe_size: int) -> str | None:
    rank = _num(detail.get("rank"))
    composite = _num(detail.get("composite"))
    if rank is None or composite is None or not universe_size:
        return None
    return (
        f"Ranks {_ordinal(rank)} of {int(universe_size)}. Its composite of "
        f"{composite:.1f} is a percentile: it scores above {composite:.0f}% of "
        f"the universe."
    )


def _sentence_drivers(detail: dict) -> str | None:
    """The two categories supplying the most composite points."""
    composite = _num(detail.get("composite"))
    contrib = detail.get("contrib") or {}
    scores = detail.get("cat_scores") or {}
    live = [
        (c, _num(contrib.get(c)), _num(scores.get(c)))
        for c in _scored_categories(detail)
        if _num(contrib.get(c)) is not None
    ]
    if not live or composite is None:
        return None
    live.sort(key=lambda t: t[1], reverse=True)
    top = live[:2]
    parts = [
        f"{CAT_LABELS[c]} (category score {s:.0f}, {k:.1f} points)"
        for c, k, s in top
    ]
    total = sum(k for _, k, _ in top)
    return (
        f"Most of that composite comes from {_join(parts)} - "
        f"{total:.1f} of its {composite:.1f} points."
    )


def _sentence_weakest(detail: dict) -> str | None:
    """The lowest-scoring category it actually has a score for."""
    contrib = detail.get("contrib") or {}
    scores = detail.get("cat_scores") or {}
    live = [(c, _num(scores.get(c)), _num(contrib.get(c)))
            for c in _scored_categories(detail)]
    if len(live) < 2:
        return None
    live.sort(key=lambda t: t[1])
    cat, score, points = live[0]
    tail = f", contributing {points:.1f} points" if points is not None else ""
    return (
        f"Its weakest scored category is {CAT_LABELS[cat]} at {score:.0f} "
        f"out of 100{tail}. A category score near 50 is the sector median."
    )


def _weighted_metrics(detail: dict, metric_weights: dict, category: str
                      ) -> list[tuple[str, float, float]]:
    """(metric, percentile, raw) for metrics that carry weight and have a value."""
    pct = detail.get("pct") or {}
    raw = detail.get("raw") or {}
    out = []
    for metric, weight in (metric_weights.get(category) or {}).items():
        if not _num(weight):
            continue
        p = _num(pct.get(metric))
        if p is None:
            continue
        out.append((metric, p, _num(raw.get(metric))))
    return out


def _label_and_value(metric: str, raw, metric_meta: dict) -> str:
    meta = metric_meta.get(metric) or {}
    label = meta.get("label", metric)
    if raw is None:
        return label
    return f"{label} ({_fmt_metric(raw, meta.get('fmt', 'ratio'))})"


def _sentence_best_inputs(detail: dict, metric_meta: dict,
                          metric_weights: dict) -> str | None:
    """Name the strongest inputs inside the strongest category."""
    contrib = detail.get("contrib") or {}
    live = [(c, _num(contrib.get(c))) for c in _scored_categories(detail)
            if _num(contrib.get(c)) is not None]
    if not live:
        return None
    lead = max(live, key=lambda t: t[1])[0]
    metrics = _weighted_metrics(detail, metric_weights, lead)
    if not metrics:
        return None
    metrics.sort(key=lambda t: t[1], reverse=True)
    best = metrics[:2]
    # "the 97th sector percentile on X and the 91st on Y" - the qualifier is
    # stated once and carried, so the sentence does not read as boilerplate.
    parts = [
        _pctile_phrase(p)
        + (" sector percentile" if i == 0 else "")
        + f" on {_label_and_value(m, r, metric_meta)}"
        for i, (m, p, r) in enumerate(best)
    ]
    return f"Inside {CAT_LABELS[lead]} it sits in {_join(parts)}."


def _sentence_worst_input(detail: dict, metric_meta: dict,
                          metric_weights: dict) -> str | None:
    """The single weakest weighted input anywhere in the score."""
    metrics: list[tuple[str, float, float]] = []
    for cat in _scored_categories(detail):
        metrics.extend(_weighted_metrics(detail, metric_weights, cat))
    if not metrics:
        return None
    metric, p, raw = min(metrics, key=lambda t: t[1])
    return (
        f"Its lowest-ranked weighted input is "
        f"{_label_and_value(metric, raw, metric_meta)}, in "
        f"{_pctile_phrase(p)} sector percentile."
    )


def _sentence_change(delta: dict | None, compare: dict | None) -> str | None:
    """Rank movement against the ~1-month baseline, falling back to the last run.

    The one-month window is the default for the same reason the movers panel
    uses it (``plan/dashboard-inventory.md``): measured on this repo's
    snapshots, every material one-day mover on 2026-08-25 was a round-trip,
    while 169 of 193 one-month moves were genuine trends.
    """
    if not delta:
        return None
    for key in ("m1", "prev"):
        entry = delta.get(key)
        if not entry:
            continue
        base = (compare or {}).get(key) or {}
        date, gap = base.get("date"), base.get("gap_days")
        when = f"the run of {date}" if date else "the last comparable run"
        if gap:
            when += f" ({int(gap)} day{'s' if int(gap) != 1 else ''} ago)"
        if entry.get("new"):
            return f"It was not in {when}, so no rank change is available."
        dr = _num(entry.get("dr"))
        dc = _num(entry.get("dc"))
        if dr is None:
            continue
        move = f"moved up {int(dr)} places" if dr > 0 else (
            f"moved down {int(abs(dr))} places" if dr < 0 else "held its rank")
        tail = ""
        if dc is not None and abs(dc) >= 0.05:
            direction = "up" if dc > 0 else "down"
            tail = f", with the composite {direction} {abs(dc):.1f}"
        return f"Since {when} it has {move}{tail}."
    return None


def _sentence_target(detail: dict) -> str | None:
    price = _num(detail.get("price"))
    target = _num(detail.get("pt_mean"))
    analysts = _num(detail.get("num_analysts"))
    if price is None or target is None or price <= 0:
        return "No mean analyst price target was available for this run."
    gap = (target / price - 1) * 100
    direction = "above" if gap >= 0 else "below"
    who = ""
    if analysts:
        who = f", the mean of {int(analysts)} analyst estimate"
        who += "s" if int(analysts) != 1 else ""
        if analysts < THIN_ANALYST_COUNT:
            who += " - a thin basis"
    return (
        f"It last traded at {_fmt_money(price)} against an analyst price "
        f"target of {_fmt_money(target)}, {abs(gap):.1f}% {direction} the "
        f"current price{who}."
    )


def _sentence_peers(detail: dict) -> str | None:
    """Where it sits among the sector names closest to it by market cap."""
    peers = detail.get("peers") or []
    composite = _num(detail.get("composite"))
    sector = detail.get("sector") or "sector"
    scored = [(p.get("ticker"), _num(p.get("composite"))) for p in peers]
    scored = [(t, c) for t, c in scored if c is not None and t]
    if not scored or composite is None:
        return None
    better = sum(1 for _, c in scored if c > composite)
    group = len(scored) + 1
    best_ticker, best_score = max(scored, key=lambda t: t[1])
    return (
        f"Among the {len(scored)} closest {sector} names by market cap plus "
        f"itself, it ranks {_ordinal(better + 1)} of {group} on composite "
        f"(best peer: {best_ticker} at {best_score:.1f})."
    )


def _sentence_flags(detail: dict) -> str | None:
    flags = detail.get("flags") or {}
    parts = []
    if detail.get("vt"):
        sev = _num(flags.get("vt_severity"))
        parts.append("a value trap" + (f" (severity {sev:.0f}/100)" if sev else ""))
    if detail.get("gt"):
        sev = _num(flags.get("gt_severity"))
        parts.append("a growth trap" + (f" (severity {sev:.0f}/100)" if sev else ""))
    if flags.get("beneish_flag"):
        parts.append("an earnings-manipulation risk on the Beneish M-score")
    if flags.get("channel_stuffing"):
        parts.append("a channel-stuffing risk (receivables growing faster than revenue)")
    if not parts:
        return "It carries no value-trap, growth-trap or accounting flag."
    return f"Flagged as {_join(parts)}."


def _sentence_confidence(detail: dict) -> str | None:
    """How much of the score is actually supported by data."""
    count = _num(detail.get("metric_count"))
    total = _num(detail.get("metric_total"))
    flags = detail.get("flags") or {}
    missing = [CAT_LABELS[c] for c in CATEGORIES
               if _num((detail.get("cat_scores") or {}).get(c)) is None]
    # Each caveat is its own sentence. `_join` is wrong here: the category-list
    # caveat already contains commas and an "and", so comma-joining these
    # produced one unreadable run-on for the stocks that need the caveats most
    # (FDXF, 12 of 18 metrics with three categories withheld and stale filings).
    bits = []
    if count is not None and total:
        bits.append(f"The score rests on {int(count)} of {int(total)} metrics.")
    if missing:
        bits.append(
            f"{_join(missing)} could not be scored for this stock, so the "
            f"remaining categories were reweighted to fill the gap."
        )
    if flags.get("stale_data"):
        age = _num(flags.get("stmt_age_days"))
        bits.append("Its filings are flagged stale"
                    + (f" ({int(age)} days old)" if age else "") + ".")
    if detail.get("eps_mismatch"):
        bits.append("Its reported and normalised EPS disagree, so earnings-based "
                    "metrics carry more uncertainty.")
    if not bits:
        return None
    return " ".join(bits)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_summary(detail: dict, *, universe_size: int, metric_meta: dict,
                  metric_weights: dict, history_delta: dict | None = None,
                  history_compare: dict | None = None) -> list[dict]:
    """Build the ordered fact list for one stock.

    Returns ``[{"k": kind, "t": sentence}, ...]``. ``kind`` lets the front end
    style a fact without parsing its text; the text is the whole payload.

    Every element is optional: a fact that cannot be stated exactly from the
    data present is dropped, never approximated. A stock with no history, no
    analyst coverage and one scored category still gets a valid - shorter -
    summary.
    """
    facts: list[tuple[str, str | None]] = [
        ("rank", _sentence_rank(detail, universe_size)),
        ("drivers", _sentence_drivers(detail)),
        ("weakest", _sentence_weakest(detail)),
        ("best_inputs", _sentence_best_inputs(detail, metric_meta, metric_weights)),
        ("worst_input", _sentence_worst_input(detail, metric_meta, metric_weights)),
        ("change", _sentence_change(history_delta, history_compare)),
        ("target", _sentence_target(detail)),
        ("peers", _sentence_peers(detail)),
        ("flags", _sentence_flags(detail)),
        ("confidence", _sentence_confidence(detail)),
    ]
    return [{"k": k, "t": t} for k, t in facts if t]


def summary_text(summary: list[dict]) -> str:
    """Flatten a summary back to one string - used by tests and the tear sheet."""
    return " ".join(f.get("t", "") for f in (summary or []))
