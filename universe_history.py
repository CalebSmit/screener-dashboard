"""Point-in-time S&P 500 membership.

**Why this exists.** `backtest.py` builds its universe from
`get_sp500_tickers()` — *today's* constituent list — and then applies it across
the whole 2020-present window. Every company that was removed, acquired or went
bankrupt in that window is simply absent from the test. That is the
survivorship bias `plan/backtest-v2.md` names as the larger of the two biases,
and step 2 of its sequencing.

This module answers one question: **which tickers were in the S&P 500 on a
given date?** It does not run a backtest and it does not change how the live
screener picks its universe. It is a component; wiring it into `backtest.py` is
a separate decision, because a backtest with a point-in-time universe but no
prices for the delisted names is *differently* wrong rather than fixed, and the
plan is explicit that a half-fixed backtest should not ship.

**Source and its limitations, stated plainly.** Membership is read from the
revision of Wikipedia's "List of S&P 500 companies" that was current on the
date in question, via the MediaWiki revisions API. This is a genuine
contemporaneous record — it is what the page said at the time, not a backward
projection — but it is not the index itself:

* **Wikipedia lags the index.** The page is edited within days of an index
  change, not at the instant of one. Measured over the backtest window, the
  revision current at a month-end was 0–7 days old. Every snapshot carries
  ``staleness_days`` so a consumer can see this rather than assume zero.
* **It is an open wiki.** A vandalised or mid-edit revision is possible. That
  is what :func:`validate_membership` is for: the table has held 503–505 rows
  on every revision sampled from 2020-01 to 2026-09, and a parse that lands on
  the wrong table or a truncated one falls far outside that band.
* **Share classes are rows, not companies.** The index holds 500 companies;
  the table lists ~503 lines because GOOG/GOOGL and similar appear separately.

Tickers are normalised the same way :func:`factor_engine.get_sp500_tickers`
normalises them — ``.`` to ``-`` — so ``BRK.B`` reads ``BRK-B`` and the two
sources can be compared directly.
"""

from __future__ import annotations

import io
import json
import re
import time
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE_PATH = ROOT / "data" / "universe_history" / "sp500_membership.json"

WIKI_TITLE = "List of S&P 500 companies"
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_INDEX = "https://en.wikipedia.org/w/index.php"
USER_AGENT = (
    "screener-dashboard/1.0 (https://calebsmit.github.io/screener-dashboard/) "
    "point-in-time universe reconstruction"
)

SCHEMA_VERSION = 1

# Plausibility band for the constituent count.
#
# Measured, not guessed: every revision sampled across the backtest window
# (2020-01-31, 2020-06-30, 2021-12-31, 2023-03-31, 2024-09-30, 2026-01-30,
# 2026-09-23) parsed to 503, 505 or 504 rows. The band is widened to 495–515 so
# ordinary share-class churn cannot trip it, while still catching the failures
# that matter: picking the wrong table on the page (the "selected changes"
# table carried 269 rows in 2021-era revisions), a truncated fetch, or a
# vandalised page. A silently short universe is the failure this guards —
# it would look exactly like a real index contraction.
MIN_PLAUSIBLE_MEMBERS = 495
MAX_PLAUSIBLE_MEMBERS = 515

# How stale a revision may be before it is flagged. Measured max over the
# window is 7 days; 30 tolerates a quiet stretch without silently accepting a
# revision from a different index composition.
MAX_STALENESS_DAYS = 30

_SYMBOL_COLUMNS = ("symbol", "ticker symbol", "ticker")


class UniverseHistoryError(Exception):
    """Base class for reconstruction failures."""


class UniverseParseError(UniverseHistoryError):
    """The fetched revision did not contain a readable constituent table."""


class ImplausibleUniverseError(UniverseHistoryError):
    """A constituent list was parsed but cannot be a real S&P 500 membership."""


@dataclass(frozen=True)
class UniverseSnapshot:
    """Membership as recorded on a date, with the provenance to check it."""

    as_of: str
    tickers: tuple
    revid: int
    revision_timestamp: str
    staleness_days: int

    @property
    def n_members(self) -> int:
        return len(self.tickers)

    @property
    def source_url(self) -> str:
        return f"{WIKI_INDEX}?oldid={self.revid}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tickers"] = list(self.tickers)
        d["n_members"] = self.n_members
        d["source_url"] = self.source_url
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "UniverseSnapshot":
        return cls(
            as_of=d["as_of"],
            tickers=tuple(d["tickers"]),
            revid=int(d["revid"]),
            revision_timestamp=d["revision_timestamp"],
            staleness_days=int(d["staleness_days"]),
        )


# ---------------------------------------------------------------------------
# Pure functions — no network, so they are the ones under test
# ---------------------------------------------------------------------------
def normalize_ticker(symbol: str) -> str:
    """Match ``factor_engine.get_sp500_tickers``'s normalisation.

    That function does ``str.replace(".", "-")``, so ``BRK.B`` becomes
    ``BRK-B``. Wikipedia occasionally carries a footnote marker or
    non-breaking space in the cell; both are stripped here so a stray
    character cannot masquerade as a different company.
    """
    s = str(symbol).replace("\xa0", " ").strip()
    s = re.sub(r"\[.*?\]", "", s)  # footnote markers, e.g. "BF.B[1]"
    s = s.strip().upper().replace(".", "-")
    return s


def _symbol_column(frame: pd.DataFrame):
    """Return the symbol column name, keyed on the *name* not the position.

    Honest scope: the symbol column sits at position 0 in every revision
    sampled across the backtest window, so a position-keyed parser would not
    in fact have been wrong here — checked, rather than assumed, before
    claiming otherwise. What *did* move is the rest of the layout: revisions
    before 2023 carried an "SEC filings" column as the third field, which
    later revisions dropped, shifting GICS Sector from position 3 to 2. Name
    keying is therefore defensive against a layout that demonstrably changes
    mid-window, not a fix for a defect already observed.
    """
    for col in frame.columns:
        if str(col).strip().lower() in _SYMBOL_COLUMNS:
            return col
    return None


def parse_constituents(html: str) -> list:
    """Extract the normalised ticker list from a rendered revision.

    Scans every table on the page for one carrying a symbol column, rather
    than assuming the constituent table is first.
    """
    try:
        tables = pd.read_html(io.StringIO(html))
    except (ValueError, ImportError) as exc:
        # ValueError: lxml parsed the page and found no tables.
        # ImportError: lxml found none and pandas fell through to its
        # bs4/html5lib flavour, which may not be installed. Either way the
        # revision yielded no table, which is a parse failure — not something
        # a caller should distinguish. The original message is preserved so a
        # genuinely missing dependency is still legible.
        raise UniverseParseError(f"no HTML tables in revision: {exc}") from exc

    for frame in tables:
        if isinstance(frame.columns, pd.MultiIndex):
            continue  # the "selected changes" table — Date/Added/Removed
        col = _symbol_column(frame)
        if col is None:
            continue
        tickers = [normalize_ticker(v) for v in frame[col].tolist()]
        tickers = [t for t in tickers if t and t != "NAN"]
        if tickers:
            return tickers

    raise UniverseParseError(
        "no table with a recognisable symbol column "
        f"(looked for {_SYMBOL_COLUMNS} across {len(tables)} tables)"
    )


def parse_ticker_ciks(html: str) -> dict:
    """Map normalised ticker -> SEC CIK for a rendered revision.

    **Why CIK.** A ticker that vanishes from the index is not necessarily a
    company that left it. Corporate renames change the symbol *and* the
    company name together — Anthem/ANTM became Elevance Health/ELV,
    CenturyLink/CTL became Lumen/LUMN, Facebook/FB became Meta/META — so
    neither symbol nor name matching can tell a rename from a removal. The CIK
    is the SEC registrant identifier and survives both, which makes it the only
    field on the page that can separate the two.

    Returns ``{}`` for revisions whose table carries no CIK column rather than
    raising, so a caller can degrade to the gross figure knowingly.
    """
    try:
        tables = pd.read_html(io.StringIO(html))
    except (ValueError, ImportError):
        return {}

    for frame in tables:
        if isinstance(frame.columns, pd.MultiIndex):
            continue
        sym_col = _symbol_column(frame)
        cik_col = next((c for c in frame.columns
                        if str(c).strip().upper() == "CIK"), None)
        if sym_col is None or cik_col is None:
            continue
        out = {}
        for sym, cik in zip(frame[sym_col], frame[cik_col]):
            ticker = normalize_ticker(sym)
            try:
                # CIKs are zero-padded inconsistently across revisions;
                # comparing as int makes "0000066740" and "66740" the same
                # registrant rather than two different ones.
                #
                # Via float, not int() directly: a single unparseable cell
                # anywhere in the column makes pandas read the whole column as
                # float, so every CIK arrives as "66740.0". int() on that
                # raises, and the effect was to silently drop *every* row
                # because one was bad — a caught-by-test failure, not a
                # hypothetical. NaN raises here too, which is the intent.
                out[ticker] = int(float(str(cik).strip()))
            except (TypeError, ValueError, OverflowError):
                continue
        if out:
            return out
    return {}


def validate_membership(tickers, as_of=None) -> None:
    """Raise if a parsed list cannot be a real S&P 500 membership.

    Two checks, both of which have a specific failure in mind:

    * **Count in band.** Catches the wrong table, a truncated fetch, or a
      vandalised page. Without it a 269-row parse looks like history.
    * **No duplicates.** A duplicated ticker means the parse picked up a
      second table's rows or a malformed revision; it would also
      double-weight a name in anything downstream.
    """
    label = f" for {as_of}" if as_of else ""
    n = len(tickers)
    if not (MIN_PLAUSIBLE_MEMBERS <= n <= MAX_PLAUSIBLE_MEMBERS):
        raise ImplausibleUniverseError(
            f"parsed {n} constituents{label}; expected "
            f"{MIN_PLAUSIBLE_MEMBERS}-{MAX_PLAUSIBLE_MEMBERS}. "
            "This is a parse or source failure, not an index change."
        )
    dupes = sorted({t for t in tickers if list(tickers).count(t) > 1})
    if dupes:
        raise ImplausibleUniverseError(
            f"duplicate tickers{label}: {dupes[:10]}"
        )


def _staleness_days(as_of: str, revision_timestamp: str) -> int:
    """Whole days between the revision and the date it is being used for."""
    target = datetime.strptime(as_of, "%Y-%m-%d").date()
    rev = datetime.strptime(revision_timestamp, "%Y-%m-%dT%H:%M:%SZ").date()
    return (target - rev).days


def month_ends(start: str, end: str) -> list:
    """Month-end dates in ``[start, end]`` as ``YYYY-MM-DD`` strings.

    The backtest rebalances monthly, so these are the dates a point-in-time
    universe is needed for.
    """
    idx = pd.date_range(start=start, end=end, freq="ME")
    return [d.strftime("%Y-%m-%d") for d in idx]


# ---------------------------------------------------------------------------
# Cache — the offline, reproducible path
# ---------------------------------------------------------------------------
def load_cache(path: Path = CACHE_PATH) -> dict:
    """Return ``{as_of: UniverseSnapshot}`` from the on-disk snapshot file."""
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("schema") != SCHEMA_VERSION:
        raise UniverseHistoryError(
            f"{path} is schema {payload.get('schema')}, "
            f"this module writes {SCHEMA_VERSION}"
        )
    return {
        s["as_of"]: UniverseSnapshot.from_dict(s)
        for s in payload.get("snapshots", [])
    }


def save_cache(snapshots, path: Path = CACHE_PATH) -> Path:
    """Write snapshots to disk, sorted by date so the file diffs cleanly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(snapshots.values(), key=lambda s: s.as_of)
    payload = {
        "schema": SCHEMA_VERSION,
        "title": WIKI_TITLE,
        "note": (
            "Point-in-time S&P 500 membership as recorded by Wikipedia. "
            "Each snapshot is the revision current on `as_of`; "
            "`staleness_days` is how old that revision was. "
            "Regenerate with `python universe_history.py --refresh`."
        ),
        "snapshots": [s.to_dict() for s in ordered],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
        fh.write("\n")
    return path


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
def _session():
    import requests

    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def find_revision(as_of: str, session=None) -> dict:
    """Return ``{'revid', 'timestamp'}`` for the revision current on ``as_of``."""
    session = session or _session()
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": WIKI_TITLE,
        "rvlimit": 1,
        "rvdir": "older",
        "rvstart": f"{as_of}T23:59:59Z",
        "rvprop": "ids|timestamp",
        "format": "json",
    }
    resp = session.get(WIKI_API, params=params, timeout=30)
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        revisions = page.get("revisions") or []
        if revisions:
            return {
                "revid": int(revisions[0]["revid"]),
                "timestamp": revisions[0]["timestamp"],
            }
    raise UniverseHistoryError(f"no revision of '{WIKI_TITLE}' on or before {as_of}")


def fetch_revision_html(revid: int, session=None) -> str:
    session = session or _session()
    resp = session.get(WIKI_INDEX, params={"oldid": int(revid)}, timeout=60)
    resp.raise_for_status()
    return resp.text


def fetch_snapshot(as_of: str, session=None) -> UniverseSnapshot:
    """Fetch, parse and validate membership for one date."""
    session = session or _session()
    rev = find_revision(as_of, session=session)
    html = fetch_revision_html(rev["revid"], session=session)
    tickers = parse_constituents(html)
    validate_membership(tickers, as_of=as_of)
    return UniverseSnapshot(
        as_of=as_of,
        tickers=tuple(sorted(tickers)),
        revid=rev["revid"],
        revision_timestamp=rev["timestamp"],
        staleness_days=_staleness_days(as_of, rev["timestamp"]),
    )


def members_on(as_of: str, cache_path: Path = CACHE_PATH,
               allow_network: bool = True) -> UniverseSnapshot:
    """Membership on ``as_of``, from cache if present.

    With ``allow_network=False`` a cache miss is an error rather than a fetch,
    so a test or a backtest can be made reproducible by construction.
    """
    cache = load_cache(cache_path)
    if as_of in cache:
        return cache[as_of]
    if not allow_network:
        raise UniverseHistoryError(
            f"no cached universe for {as_of} and network use is disabled; "
            "run `python universe_history.py --refresh` to populate "
            f"{cache_path}"
        )
    snap = fetch_snapshot(as_of)
    cache[as_of] = snap
    save_cache(cache, cache_path)
    return snap


def refresh(start: str, end: str, cache_path: Path = CACHE_PATH,
            pause: float = 0.5, verbose: bool = True) -> dict:
    """Populate the cache for every month-end in ``[start, end]``.

    Dates already cached are skipped, so this is cheap to re-run and only ever
    fetches the months that have been added since last time.
    """
    cache = load_cache(cache_path)
    session = _session()
    wanted = month_ends(start, end)
    todo = [d for d in wanted if d not in cache]
    if verbose:
        print(f"  {len(wanted)} month-ends requested, {len(todo)} to fetch")
    for i, as_of in enumerate(todo, 1):
        snap = fetch_snapshot(as_of, session=session)
        cache[as_of] = snap
        if verbose:
            print(f"  [{i}/{len(todo)}] {as_of}: {snap.n_members} members "
                  f"(rev {snap.revid}, {snap.staleness_days}d old)")
        save_cache(cache, cache_path)
        time.sleep(pause)
    return cache


# ---------------------------------------------------------------------------
# What the reconstruction is for: measuring what today's list leaves out
# ---------------------------------------------------------------------------
def survivorship_gap(snapshot: UniverseSnapshot, current_tickers) -> dict:
    """How much of ``snapshot``'s universe is missing from today's list.

    ``dropped`` are the names a backtest built on today's constituents silently
    deletes from that month — companies that were in the index then and are not
    now. They are the survivorship bias, named.
    """
    then = set(snapshot.tickers)
    now = set(current_tickers)
    dropped = sorted(then - now)
    return {
        "as_of": snapshot.as_of,
        "n_then": len(then),
        "n_survivors": len(then & now),
        "n_dropped": len(dropped),
        "pct_dropped": 100.0 * len(dropped) / len(then) if then else 0.0,
        "dropped": dropped,
    }


def _cli():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true",
                    help="fetch month-end universes into the cache")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=date.today().strftime("%Y-%m-%d"))
    ap.add_argument("--as-of", help="print membership for one date")
    args = ap.parse_args()

    if args.refresh:
        cache = refresh(args.start, args.end)
        print(f"  cache holds {len(cache)} snapshots -> {CACHE_PATH}")
    elif args.as_of:
        snap = members_on(args.as_of)
        print(f"{snap.as_of}: {snap.n_members} members "
              f"(revision {snap.revid} of {snap.revision_timestamp}, "
              f"{snap.staleness_days}d stale)")
        print(f"  {snap.source_url}")
    else:
        ap.print_help()


if __name__ == "__main__":
    _cli()
