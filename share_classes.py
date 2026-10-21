"""Score each company once, on its designated share class line.

The S&P 500 holds 503 lines for 500 companies: Alphabet, Fox and News Corp each list two
classes of common stock. Every company-statement metric is identical across a pair, as it
must be; the rows differ through the line's own price, the line's own return series, and -
with no economic content - the data source's analyst record for that line. So two lines of
one company are two tickets on one idea, and ranking both tells a reader something false
about the company and counts it twice in its sector's percentile distribution.

**What the providers do** (`research/2026-10-19-share-classes.md` section 4). Fundamentals
and forecasts are company facts: MSCI "calculates the data on a company level in which all
classes (listed and unlisted) of equity are aggregated" (*Fundamental Data Methodology*,
March 2021, section 1, p. 10). Prices and returns stay per line. A list that selects *names*
represents each company once by its **Designated Listing** - "the share class with both the
highest one-year trading liquidity ... and largest float-adjusted market capitalization"
(S&P DJI, *Equity Indices Policies & Practices*, August 2026, pp. 17-18, approach B) - or
copies the **pricing vehicle**'s scores to every class (FTSE Russell, *Russell US Equity
Indexes* v7.2, rules 6.5.1-6.5.4).

**Why the designated line is declared here and not recomputed every run.** S&P DJI's rule is
*sticky*: the Designated Listing changes only when the other line leads on both criteria by
more than 20%. That matters, and it is the one thing the 2026-10-19 proposal did not measure.
Over 89 month-ends (2019-06 to 2026-10, 63-day average daily dollar volume) Alphabet's two
lines sat **within 20% of each other in 31 of them (34.8%)**, ratio median 1.28 and minimum
1.00, and the higher-volume line actually changed **four times** (2020-09, 2020-11, 2020-12,
2021-01). Fox and News Corp never came close (minimum ratios 2.08 and 2.19). Rescoring the
2026-10-21 run both ways, **Alphabet ranks 179 on GOOGL and 101 on GOOG** - a 78-place swing,
and a different ticker in the ranking - because the two lines carry different analyst records
(`research/2026-10-19-share-classes.md` section 7). A rule that re-picks the line from each
run's own volume would have moved the company 78 places on a liquidity tiebreak that is a
coin flip a third of the time. So the choice is declared in `config.yaml`, changed only with
a changelog entry, and the run *reports* when a non-designated line leads by more than the
margin instead of acting on it.

Measurements: `.scratch`-free reproductions in `research/measurements/2026-10-21-share-class-stability.py`.
"""
from __future__ import annotations

import json
import pathlib
from typing import Iterable

# S&P DJI's switching margin: the Designated Listing changes only when the other line leads
# by more than this. Used here to *report*, never to switch - the switch is a config change.
CHALLENGE_MARGIN = 0.20

SEC_TICKER_FILE = pathlib.Path("data/sec/company_tickers.json")


def _norm(ticker: str) -> str:
    return str(ticker).strip().upper()


def cik_map(path: pathlib.Path | None = None) -> dict[str, int]:
    """{TICKER: CIK} from the SEC's own company_tickers.json, or {} if it is not cached.

    The file lists tickers with dots where the universe uses dashes (BRK.B / BRK-B), so both
    spellings are indexed. Measured 2026-10-21 on the 503-line universe: **503 of 503 lines
    resolve exactly**, with no format change needed, and the three CIKs holding more than one
    universe line are exactly Alphabet, Fox and News Corp - no false merge.
    """
    p = pathlib.Path(path) if path is not None else SEC_TICKER_FILE
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    records = raw.values() if isinstance(raw, dict) else raw
    out: dict[str, int] = {}
    for rec in records:
        try:
            t, c = _norm(rec["ticker"]), int(rec["cik_str"])
        except (KeyError, TypeError, ValueError):
            continue
        out[t] = c
        out[t.replace(".", "-")] = c
        out[t.replace("-", ".")] = c
    return out


def group_companies(
    tickers: Iterable[str],
    names: dict[str, str] | None = None,
    ciks: dict[str, int] | None = None,
) -> tuple[list[list[str]], str]:
    """Return ([[lines of a multi-line company], ...], basis_used).

    Grouped by SEC CIK where the cached file is available, else by identical ``longName``.
    Measured 2026-10-21: on the live universe the two bases give the *same three groups with
    the same membership*, so the fallback is corroborated rather than assumed. Only groups
    with more than one line are returned; the order inside a group is the input order.
    """
    tickers = [_norm(t) for t in tickers]
    ciks = cik_map() if ciks is None else ciks
    keyed: dict[object, list[str]] = {}
    basis = "none"
    if ciks and any(t in ciks for t in tickers):
        basis = "sec_cik"
        for t in tickers:
            k = ciks.get(t)
            if k is not None:
                keyed.setdefault(k, []).append(t)
    elif names:
        basis = "long_name"
        for t in tickers:
            k = (names.get(t) or "").strip().casefold()
            if k:
                keyed.setdefault(k, []).append(t)
    groups = [v for v in keyed.values() if len(v) > 1]
    return groups, basis


def designate(
    groups: list[list[str]],
    volumes: dict[str, float],
    declared: Iterable[str] = (),
    margin: float = CHALLENGE_MARGIN,
) -> tuple[dict[str, str], list[str]]:
    """Choose one line per multi-line company.

    Returns ``(secondary_to_designated, notes)``. ``secondary_to_designated`` maps each line
    that leaves the ranking to the line that represents its company; ``notes`` are lines for
    ``run.log`` and the health note.

    A group is settled by the declaration when exactly one of its lines is declared. A group
    with no declared line is a company that has newly listed a second class: it falls back to
    the higher average daily dollar volume and says so loudly, because an undeclared choice is
    the per-run recomputation this module exists to avoid. A group with two declared lines is
    a configuration error and raises.
    """
    declared_set = {_norm(t) for t in declared}
    secondary: dict[str, str] = {}
    notes: list[str] = []

    def vol(t: str) -> float:
        try:
            v = float(volumes.get(t, float("nan")))
        except (TypeError, ValueError):
            return float("nan")
        return v

    for group in groups:
        picked = [t for t in group if t in declared_set]
        if len(picked) > 1:
            raise ValueError(
                "share_classes.designated names more than one line of the same company: "
                f"{sorted(picked)} - a company is represented once (config.yaml)"
            )
        if picked:
            keep = picked[0]
        else:
            ranked = sorted(group, key=lambda t: (-(vol(t) if vol(t) == vol(t) else -1.0), t))
            keep = ranked[0]
            notes.append(
                f"ATTENTION - {'/'.join(sorted(group))} is a multi-class company with no "
                f"designated listing in config.yaml; scored on {keep} by dollar volume this "
                "run only. Declare it (share_classes.designated) with a changelog entry."
            )
        for t in group:
            if t != keep:
                secondary[t] = keep

        # S&P DJI's 20% switching margin, reported and never acted on.
        kv = vol(keep)
        for t in group:
            if t == keep:
                continue
            tv = vol(t)
            if kv == kv and tv == tv and kv > 0 and tv > kv * (1.0 + margin):
                notes.append(
                    f"ATTENTION - {t} led the designated listing {keep} on average daily "
                    f"dollar volume by {tv / kv - 1:.0%} (>{margin:.0%}, S&P DJI's switching "
                    f"margin). Review share_classes.designated in config.yaml."
                )
            elif kv == kv and tv == tv and kv > 0 and tv > kv / (1.0 + margin):
                notes.append(
                    f"NOTE - {t} is within {margin:.0%} of the designated listing {keep} on "
                    f"dollar volume ({tv / kv:.2f}x); the choice is declared, so the ranking "
                    "does not move with it."
                )
    return secondary, notes


def price_gaps(secondary: dict[str, str], prices: dict[str, float]) -> dict[str, dict]:
    """Per dropped line: the company's ranked line and what that line's price gap is.

    Every valuation ratio differs between two lines of one company only through the price
    (`research/2026-10-19-share-classes.md` section 3: earnings yield differs by exactly the
    ratio of the prices), so one number states the whole valuation difference.
    """
    out: dict[str, dict] = {}
    for sec, keep in sorted(secondary.items()):
        ps, pk = prices.get(sec), prices.get(keep)
        gap = None
        try:
            if ps and pk and float(pk) > 0:
                gap = round((float(ps) / float(pk) - 1.0) * 100.0, 2)
        except (TypeError, ValueError):
            gap = None
        out[sec] = {"scored_as": keep, "price": ps, "price_gap_pct": gap}
    return out
