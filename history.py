"""Historical spine for the dashboard: comparable run-to-run score history.

The dashboard has never had a time dimension. Every view is a snapshot, so it
cannot answer the questions that actually drive a sell decision - "what moved
since last week?", "what broke?", "has this been deteriorating for three runs?"
This module builds that spine from the snapshots the data loop already writes
into ``improvement/snapshots/``.

The hard part is not the arithmetic, it is deciding **which runs are comparable
to each other**. Two failures in the existing snapshot directory make a naive
"just diff the last two files" approach actively misleading:

1. **Degraded runs.** ``2026-07-28`` scored a universe whose valuation
   dispersion was 17.2 against a trailing median of 23.9 - a 28% collapse, the
   exact failure ``scripts/check_run_health.py`` was written to catch (that
   gate postdates the run, so the bad snapshot was never blocked). Diffed
   naively it reports **411 of 501 stocks (82%) moving more than 50 ranks** - a
   screen full of fictional movers.

2. **Warm-start duplicates.** ``2026-03-01`` is identical in rank to
   ``2026-02-28``: a re-run off cache, not a new observation. This is the
   display-side face of ``CLAUDE.md`` priority 0.6. Left in, it pads the
   history with points that show no change because nothing was measured, not
   because nothing moved.

Runs are therefore quality-gated before they enter the series, and every
exclusion is reported in the payload rather than silently dropped.

**Why rank continuity rather than the publish-time dispersion rule.** The first
attempt reused ``check_run_health``'s "dispersion >20% below the trailing
median" threshold. Measured against the real snapshot directory it excluded 16
of 20 runs. Two reasons, both instructive: risk-score dispersion has drifted
legitimately from 26.7 (February) to 19.5 (August), and a history that only
records *kept* runs freezes its own baseline, so a single exclusion cascades
into excluding everything after it. Dispersion is the right gate at publish
time, where the pipeline maintains a baseline over every run; it is the wrong
one here.

The property this module actually needs is that a run's ranking is comparable
to its neighbours', so it gates on exactly that: Spearman rank correlation
against the last accepted run. Measured over the 19 consecutive pairs in the
snapshot directory, the 17 clean pairs span **0.882 to 1.000** (the lowest is a
12-day gap; a 29-day gap still scores 0.951), and the only two breaks are
**0.016** and **-0.020**, both involving ``2026-07-28``. Any threshold between
0.05 and 0.87 classifies every observed run identically, so
``MIN_RANK_CONTINUITY = 0.50`` sits in the middle of an empty region rather
than being tuned to a particular run.

Comparability of the scores themselves is established separately: no factor
weight, metric definition or threshold has ever changed
(``METHODOLOGY_CHANGELOG.md`` - the improvement engine has never fired and
``allow_auto_apply`` is false), so composites from February and from August are
on the same scale. If a weight ever does change, this assumption breaks and
this module needs a scale break at that date.

"""

from __future__ import annotations


from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SNAPSHOTS_DIR = ROOT / "improvement" / "snapshots"

CATEGORIES = ["valuation", "quality", "growth", "momentum", "risk",
              "revisions", "size", "investment"]
CATEGORY_SCORE_COLS = [f"{c}_score" for c in CATEGORIES]

# Minimum Spearman rank correlation with the previously accepted run for a
# snapshot to join the history. See the module docstring: measured over the
# real snapshot directory, clean pairs span 0.882-1.000 and the two degraded
# pairs are 0.016 and -0.020, so this threshold sits in an empty region and any
# value in [0.05, 0.87] would classify every observed run the same way.
MIN_RANK_CONTINUITY = 0.50

# Below this many shared tickers, a correlation is not a meaningful check and
# the run is accepted rather than rejected on thin evidence.
MIN_OVERLAP_FOR_CONTINUITY = 30

# A "material" rank move is one beyond the 95th percentile of ordinary
# run-to-run variation, measured from the history itself (see
# `rank_change_noise`). The fallback is used only when there are too few paired
# runs to measure; it is the value measured on 2026-08-25 from 12 consecutive
# clean pairs (n=6,011 ticker-pairs): p50 7, p90 36, p95 55.
FALLBACK_MATERIAL_RANK_MOVE = 55
MIN_PAIRS_FOR_NOISE = 3
MAX_NOISE_PAIR_GAP_DAYS = 7

# How far back the secondary comparison looks, and the tolerance around it.
LOOKBACK_TARGET_DAYS = 30
LOOKBACK_MIN_DAYS = 14

# Cap on how many movers are listed per direction. The full material count is
# reported alongside so truncation is visible rather than implied-complete.
MAX_MOVERS_LISTED = 15

# How many recent runs the round-trip check looks across. Five keeps it to
# roughly a trading week of daily runs, long enough to catch a metric that
# drops out and returns a run or two later (the observed failure spanned two
# runs) without reaching so far back that a genuine reversal counts as noise.
ROUND_TRIP_WINDOW = 5


@dataclass
class RunSnapshot:
    """One quality-accepted run, reduced to what the history needs."""
    date: str
    ranks: dict[str, int]
    composites: dict[str, float]
    cat_scores: dict[str, dict[str, float]]

    @property
    def tickers(self) -> set[str]:
        return set(self.ranks)


def snapshot_index(snapshots_dir: Path | None = None) -> dict[str, Path]:
    """Map run_date -> snapshot path, one file per date.

    Several runs on one day each write their own file. Matches
    ``improvement_engine.compute_forward_returns``: sorted order, last file for
    a date wins.
    """
    snapshots_dir = snapshots_dir or SNAPSHOTS_DIR
    if not snapshots_dir.exists():
        return {}
    index: dict[str, Path] = {}
    for path in sorted(snapshots_dir.glob("*.parquet")):
        index[path.stem.split("_")[0]] = path
    return index


def rank_continuity(a: RunSnapshot, b: RunSnapshot) -> float | None:
    """Spearman rank correlation between two runs over their shared tickers.

    Returns None when too few tickers overlap for the number to mean anything.
    """
    shared = a.tickers & b.tickers
    if len(shared) < MIN_OVERLAP_FOR_CONTINUITY:
        return None
    ordered = sorted(shared)
    left = pd.Series([a.ranks[t] for t in ordered], dtype=float)
    right = pd.Series([b.ranks[t] for t in ordered], dtype=float)
    rho = left.corr(right, method="spearman")
    return None if pd.isna(rho) else float(rho)


def _to_run_snapshot(date: str, df: pd.DataFrame) -> RunSnapshot:
    df = df.dropna(subset=["Ticker"])
    ranks, composites, cats = {}, {}, {}
    for _, row in df.iterrows():
        ticker = str(row["Ticker"])
        try:
            ranks[ticker] = int(row["Rank"])
        except (TypeError, ValueError):
            continue
        composite = pd.to_numeric(row.get("Composite"), errors="coerce")
        composites[ticker] = round(float(composite), 2) if pd.notna(composite) else None
        per_cat = {}
        for cat in CATEGORIES:
            value = pd.to_numeric(row.get(f"{cat}_score"), errors="coerce")
            if pd.notna(value):
                per_cat[cat] = round(float(value), 1)
        cats[ticker] = per_cat
    return RunSnapshot(date=date, ranks=ranks, composites=composites, cat_scores=cats)


def select_comparable_runs(
    snapshots_dir: Path | None = None,
    up_to_date: str | None = None,
) -> tuple[list[RunSnapshot], list[dict]]:
    """Load snapshots in date order, keeping only mutually comparable runs.

    Returns ``(kept, excluded)``. Every excluded run carries a machine-readable
    ``reason`` and a human-readable ``detail`` so the dashboard can show what
    was left out and why - a silently shortened history is indistinguishable
    from a stable one.
    """
    index = snapshot_index(snapshots_dir)
    excluded: list[dict] = []

    # Pass 1: load everything readable and structurally complete.
    candidates: list[RunSnapshot] = []
    for date in sorted(index):
        if up_to_date is not None and date > up_to_date:
            continue
        try:
            df = pd.read_parquet(index[date])
        except Exception as exc:  # noqa: BLE001 - a bad file must not kill the build
            excluded.append({"date": date, "reason": "unreadable",
                             "detail": f"{type(exc).__name__}: {exc}"})
            continue

        missing = [c for c in ("Ticker", "Rank", "Composite") if c not in df.columns]
        if missing:
            excluded.append({"date": date, "reason": "missing_columns",
                             "detail": f"no {', '.join(missing)} column"})
            continue

        snap = _to_run_snapshot(date, df)
        if not snap.ranks:
            excluded.append({"date": date, "reason": "empty",
                             "detail": "no ranked tickers"})
            continue
        candidates.append(snap)

    if not candidates:
        return [], excluded

    # Pass 2: seed the chain. The first run has no predecessor to be checked
    # against, so it is validated against its successor instead - otherwise a
    # degraded run in first position would reject every good run after it.
    seed = 0
    while seed + 1 < len(candidates):
        rho = rank_continuity(candidates[seed], candidates[seed + 1])
        if rho is None or rho >= MIN_RANK_CONTINUITY:
            break
        excluded.append({
            "date": candidates[seed].date, "reason": "rank_discontinuity",
            "detail": (f"rank correlation {rho:.2f} with the next run "
                       f"({candidates[seed + 1].date}), below "
                       f"{MIN_RANK_CONTINUITY:.2f}"),
        })
        seed += 1

    kept: list[RunSnapshot] = [candidates[seed]]

    # Pass 3: walk forward, always comparing against the last *accepted* run so
    # that one bad snapshot cannot drag the chain with it.
    for snap in candidates[seed + 1:]:
        last = kept[-1]
        if snap.ranks == last.ranks:
            excluded.append({"date": snap.date, "reason": "duplicate_of_previous",
                             "detail": f"ranks identical to {last.date} "
                                       "(re-run off cache, not a new observation)"})
            continue
        rho = rank_continuity(snap, last)
        if rho is not None and rho < MIN_RANK_CONTINUITY:
            excluded.append({
                "date": snap.date, "reason": "rank_discontinuity",
                "detail": (f"rank correlation {rho:.2f} with {last.date}, "
                           f"below {MIN_RANK_CONTINUITY:.2f} - the ranking is "
                           "not comparable to the rest of the series"),
            })
            continue
        kept.append(snap)

    return kept, excluded


def rank_change_noise(kept: list[RunSnapshot]) -> dict:
    """Measure the distribution of |rank change| between consecutive runs.

    This is what makes the "material mover" threshold a measurement rather than
    a guess: a stock is only worth surfacing if it moved further than ordinary
    run-to-run variation. Only pairs no more than MAX_NOISE_PAIR_GAP_DAYS apart
    are used, so a multi-month gap in the data does not inflate the noise floor
    and hide real moves.
    """
    deltas: list[int] = []
    pairs = 0
    for prev, curr in zip(kept, kept[1:]):
        gap = (pd.Timestamp(curr.date) - pd.Timestamp(prev.date)).days
        if gap > MAX_NOISE_PAIR_GAP_DAYS:
            continue
        pairs += 1
        for ticker, rank in curr.ranks.items():
            if ticker in prev.ranks:
                deltas.append(abs(rank - prev.ranks[ticker]))

    if pairs < MIN_PAIRS_FOR_NOISE or not deltas:
        return {"p50": None, "p90": None, "p95": None, "n_pairs": pairs,
                "n_observations": len(deltas), "source": "fallback",
                "material_threshold": FALLBACK_MATERIAL_RANK_MOVE}

    series = pd.Series(deltas)
    p95 = int(round(float(series.quantile(0.95))))
    return {
        "p50": int(round(float(series.quantile(0.50)))),
        "p90": int(round(float(series.quantile(0.90)))),
        "p95": p95,
        "n_pairs": pairs,
        "n_observations": len(deltas),
        "source": "measured",
        "material_threshold": max(p95, 1),
    }


def _pick_lookback(kept: list[RunSnapshot]) -> RunSnapshot | None:
    """The kept run closest to LOOKBACK_TARGET_DAYS before the latest one."""
    if len(kept) < 2:
        return None
    current = pd.Timestamp(kept[-1].date)
    candidates = [
        (abs((current - pd.Timestamp(s.date)).days - LOOKBACK_TARGET_DAYS), s)
        for s in kept[:-1]
        if (current - pd.Timestamp(s.date)).days >= LOOKBACK_MIN_DAYS
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda t: t[0])[1]


def _compare(current: RunSnapshot, baseline: RunSnapshot) -> dict[str, dict]:
    """Per-ticker deltas from baseline to current."""
    out: dict[str, dict] = {}
    for ticker, rank in current.ranks.items():
        if ticker not in baseline.ranks:
            out[ticker] = {"new": True}
            continue
        # Rank 1 is best, so a *fall* in rank number is an improvement. Store
        # the improvement-positive value: dr > 0 means "moved up the table".
        entry = {"dr": baseline.ranks[ticker] - rank}
        cur_c, base_c = current.composites.get(ticker), baseline.composites.get(ticker)
        if cur_c is not None and base_c is not None:
            entry["dc"] = round(cur_c - base_c, 2)
        cat_deltas = {}
        for cat in CATEGORIES:
            cur_v = current.cat_scores.get(ticker, {}).get(cat)
            base_v = baseline.cat_scores.get(ticker, {}).get(cat)
            if cur_v is not None and base_v is not None:
                delta = round(cur_v - base_v, 1)
                if delta:
                    cat_deltas[cat] = delta
        if cat_deltas:
            entry["cat"] = cat_deltas
        out[ticker] = entry
    return out


def round_trip_tickers(kept: list[RunSnapshot], threshold: int) -> set[str]:
    """Tickers whose recent rank path is an excursion that returned to base.

    Some of the largest apparent movers are not moves at all. MNST is the
    worked example: its ``return_12_1`` percentile read 97.1 on 2026-08-20,
    **2.9** on 08-21 and 08-24, then 97.1 again on 08-25, while its price went
    47.5 -> 48.9. A momentum score cannot travel from the 97th percentile to
    the 3rd and back on a 3% price move; the underlying twelve-month return was
    being computed from a corrupted price series. The metric is not NaN -
    missing metrics are correctly excluded by ``factor_engine``'s
    ``na_option="keep"`` and ``has_data`` mask - so it is a *computed* value
    from bad price history, and nothing downstream can tell it apart from a
    real collapse.

    **Correction, 2026-08-26.** This docstring originally read "failed to
    compute for two runs", i.e. that 2.9 was the artifact and 97.1 the truth.
    It was the other way round. Yahoo's 13-month series for MNST alternates
    between pre- and post-split prices across its 2026-08-11 2:1 split, so
    ``return_12_1`` was computed as (unadjusted July price 93.49 - adjusted
    2025 price 62.30) / 62.30 = **+0.50**, the 97th percentile. The true
    split-adjusted figure is about **-0.25**, the 3rd percentile - so 08-21
    and 08-24 were the two runs that got it *right*. Fixed at source by
    ``factor_engine.check_price_series_integrity``; see
    METHODOLOGY_CHANGELOG.md 2026-08-26.

    None of this changes the detector below, which was correct to flag MNST
    and correct about why: a round trip in the ranking is evidence of a data
    artifact somewhere, whichever end of it is wrong.

    The signature is detectable without knowing the cause: over a short window
    the rank made a material excursion and then came back to within noise of
    where it started. Such a stock is flagged rather than hidden, because the
    honest claim is "this looks like a data artifact, here is the path" and not
    "this was the biggest mover today".
    """
    flagged: set[str] = set()
    if len(kept) < 3:
        return flagged
    window = kept[-ROUND_TRIP_WINDOW:]
    if len(window) < 3:
        return flagged
    current, start = window[-1], window[0]
    for ticker, rank in current.ranks.items():
        path = [s.ranks[ticker] for s in window if ticker in s.ranks]
        if len(path) < 3 or ticker not in start.ranks:
            continue
        excursion = max(path) - min(path)
        if excursion >= threshold and abs(rank - start.ranks[ticker]) < threshold:
            flagged.add(ticker)
    return flagged


def _movers(deltas: dict[str, dict], threshold: int,
            round_trips: set[str] | None = None) -> dict:
    """Material movers in each direction, largest first."""
    round_trips = round_trips or set()
    material = [
        (ticker, entry) for ticker, entry in deltas.items()
        if not entry.get("new") and abs(entry.get("dr", 0)) >= threshold
    ]
    ups = sorted([m for m in material if m[1]["dr"] > 0],
                 key=lambda t: -t[1]["dr"])
    downs = sorted([m for m in material if m[1]["dr"] < 0],
                   key=lambda t: t[1]["dr"])

    def _pack(items):
        packed = []
        for ticker, entry in items[:MAX_MOVERS_LISTED]:
            row = {"t": ticker, "dr": entry["dr"]}
            if "dc" in entry:
                row["dc"] = entry["dc"]
            cats = entry.get("cat") or {}
            if cats:
                # The category that moved furthest in the same direction as the
                # rank move - a first answer to "what changed?".
                driver = max(cats.items(), key=lambda kv: abs(kv[1]))
                row["drv"] = [driver[0], driver[1]]
            if ticker in round_trips:
                row["rt"] = True
            packed.append(row)
        return packed

    return {"up": _pack(ups), "down": _pack(downs),
            "n_up": len(ups), "n_down": len(downs),
            "n_round_trip": sum(1 for t, _ in material if t in round_trips),
            "listed": MAX_MOVERS_LISTED}


def build_history(
    current_df: pd.DataFrame | None = None,
    current_date: str | None = None,
    snapshots_dir: Path | None = None,
) -> dict:
    """Build the dashboard's ``history`` payload block.

    ``current_df`` is the run being published. If its date is not already in
    the snapshot directory it is appended, so the dashboard is never a run
    behind its own history (and so regenerating from an older run directory
    truncates rather than mixes eras).
    """
    kept, excluded = select_comparable_runs(snapshots_dir, up_to_date=current_date)

    if current_df is not None and current_date is not None:
        if not kept or kept[-1].date != current_date:
            live = _to_run_snapshot(current_date, current_df)
            if live.ranks:
                kept = [s for s in kept if s.date != current_date] + [live]

    empty = {
        "dates": [], "series": {}, "excluded": excluded, "noise": None,
        "compare": {"prev": None, "m1": None}, "movers": {}, "delta": {},
        "available": False,
    }
    if len(kept) < 2:
        empty["dates"] = [s.date for s in kept]
        return empty

    dates = [s.date for s in kept]
    current = kept[-1]

    # Aligned per-ticker series, restricted to the current universe - the
    # dashboard has no drill-down for a name that is no longer scored, so a
    # series for one would be unreachable payload. Runs where a current ticker
    # was absent get null rather than being dropped, so index additions stay
    # visible as a gap in the line.
    series: dict[str, dict] = {}
    for ticker in sorted(current.tickers):
        series[ticker] = {"r": [s.ranks.get(ticker) for s in kept],
                          "c": [s.composites.get(ticker) for s in kept]}

    noise = rank_change_noise(kept)
    threshold = noise["material_threshold"]
    round_trips = round_trip_tickers(kept, threshold)

    prev = kept[-2]
    lookback = _pick_lookback(kept)

    prev_deltas = _compare(current, prev)
    delta: dict[str, dict] = {t: {"prev": e} for t, e in prev_deltas.items()}
    compare = {
        "prev": {"date": prev.date,
                 "gap_days": (pd.Timestamp(current.date) - pd.Timestamp(prev.date)).days},
        "m1": None,
    }
    movers = {"prev": _movers(prev_deltas, threshold, round_trips)}

    if lookback is not None:
        lb_deltas = _compare(current, lookback)
        for ticker, entry in lb_deltas.items():
            delta.setdefault(ticker, {})["m1"] = entry
        compare["m1"] = {
            "date": lookback.date,
            "gap_days": (pd.Timestamp(current.date) - pd.Timestamp(lookback.date)).days,
        }
        movers["m1"] = _movers(lb_deltas, threshold, round_trips)

    return {
        "dates": dates,
        "current_date": current.date,
        "series": series,
        "excluded": excluded,
        "noise": noise,
        "compare": compare,
        "movers": movers,
        "delta": delta,
        "available": True,
    }
