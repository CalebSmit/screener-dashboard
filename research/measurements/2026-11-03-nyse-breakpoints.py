"""Verify the phase-0 proposal's floor: the NYSE 20th-percentile market cap.

`research/2026-11-02-universe-expansion-phase0.md` pre-registers a phase-4 floor of
**$1,224.13M** -- "the NYSE 20th-percentile breakpoint in Ken French's ME_Breakpoints file
for the most recent June (June 2026, n = 1,079 NYSE stocks)" -- and lists an unresolved
discrepancy: Fama & French (2008) quote $2.3B as the end-June 2005 NYSE *median* where the
current file reads $1.79B, which the note guesses is "probably a CRSP vintage difference".

This script downloads the file and reads both numbers itself. Run:

    python research/measurements/2026-11-03-nyse-breakpoints.py

It writes nothing outside data/survivorship/ (gitignored cache) and prints a table.
"""

from __future__ import annotations

import io
import pathlib
import sys
import urllib.request
import zipfile

URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "ME_Breakpoints_CSV.zip"
)
CACHE = pathlib.Path("data/survivorship/ME_Breakpoints_CSV.zip")
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def fetch() -> bytes:
    if CACHE.exists():
        print(f"using cached {CACHE} ({CACHE.stat().st_size:,} bytes)")
        return CACHE.read_bytes()
    req = urllib.request.Request(URL, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        blob = r.read()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(blob)
    print(f"downloaded {URL} -> {CACHE} ({len(blob):,} bytes)")
    return blob


def parse(blob: bytes) -> dict[str, list[float]]:
    """Return {YYYYMM: [count, bp5, bp10, ..., bp95]} from the breakpoint CSV.

    Format (per French's own header): the first lines are prose, then one row per month:
    date, number of firms, then the 5th through 95th percentiles in 5-point steps.
    """
    zf = zipfile.ZipFile(io.BytesIO(blob))
    name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
    text = zf.read(name).decode("latin-1")
    rows: dict[str, list[float]] = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 20 or not parts[0].isdigit() or len(parts[0]) != 6:
            continue
        try:
            rows[parts[0]] = [float(p) for p in parts[1:]]
        except ValueError:
            continue
    print(f"parsed {name}: {len(rows)} monthly rows, "
          f"{min(rows)} to {max(rows)}, {len(next(iter(rows.values())))} columns after the date")
    return rows


def pct_col(values: list[float], percentile: int) -> float:
    """values = [count, bp5, bp10, ..., bp95]; percentile must be a multiple of 5."""
    assert percentile % 5 == 0 and 5 <= percentile <= 95, percentile
    return values[1 + (percentile // 5) - 1]


def main() -> int:
    rows = parse(fetch())

    # --- the pre-registered floor -------------------------------------------------
    print("\n=== Pre-registration 2: the NYSE 20th-percentile breakpoint, June rows ===")
    print(f"{'June':<8}{'n NYSE':>9}{'20th pct ($M)':>16}{'50th pct ($M)':>16}")
    for year in range(2020, 2027):
        key = f"{year}06"
        if key not in rows:
            print(f"{year:<8}{'(absent)':>9}")
            continue
        v = rows[key]
        print(f"{year:<8}{int(v[0]):>9,}{pct_col(v, 20):>16,.2f}{pct_col(v, 50):>16,.2f}")

    june26 = rows.get("202606")
    if june26:
        claimed_n, claimed_bp = 1079, 1224.13
        n, bp = int(june26[0]), pct_col(june26, 20)
        print(f"\nnote claims: n = {claimed_n:,}, 20th percentile = ${claimed_bp:,.2f}M")
        print(f"file reads:  n = {n:,}, 20th percentile = ${bp:,.2f}M")
        print(f"VERDICT: count {'MATCHES' if n == claimed_n else 'DIFFERS'}; "
              f"breakpoint {'MATCHES' if abs(bp - claimed_bp) < 0.01 else 'DIFFERS'}")

    # --- the unresolved 2005 discrepancy ------------------------------------------
    print("\n=== Section 6's unresolved figure: end-June 2005 ===")
    print("FF 2008 (p. 1655) quotes the 20th and 50th NYSE percentiles as $610M and $2.3B.")
    v = rows["200506"]
    bp20, bp50 = pct_col(v, 20), pct_col(v, 50)
    print(f"June 2005 file: n = {int(v[0]):,}, 20th = ${bp20:,.2f}M, 50th = ${bp50:,.2f}M")
    print(f"  20th vs FF's $610M: {bp20 / 610 - 1:+.1%}")
    print(f"  50th vs FF's $2,300M: {bp50 / 2300 - 1:+.1%}")
    print("\nIf a CRSP vintage change explained the gap it would move BOTH percentiles.")
    print("Candidate explanation to test: FF's 'big' line is the NYSE median by *market value*,")
    print("i.e. the cap at which cumulative NYSE market value reaches 50%, not the 50th")
    print("percentile of the firm count. Breakpoints cannot settle that; what CAN be checked")
    print("is whether the 50th-percentile firm-count line is stable across nearby months,")
    print("which tells us whether $1.79B is a data artefact or the file's actual answer:")
    for key in ("200503", "200506", "200509", "200512", "200606"):
        if key in rows:
            w = rows[key]
            print(f"  {key}: n = {int(w[0]):,}, 20th = ${pct_col(w, 20):>9,.2f}M, "
                  f"50th = ${pct_col(w, 50):>9,.2f}M")

    # --- how far the floor moves year to year -------------------------------------
    print("\n=== Does the floor need re-tuning? (the note's claim: 'it moves with the market') ===")
    prev = None
    for year in range(2015, 2027):
        key = f"{year}06"
        if key not in rows:
            continue
        bp = pct_col(rows[key], 20)
        chg = f"{bp / prev - 1:+.1%}" if prev else "-"
        print(f"  June {year}: ${bp:>9,.2f}M  {chg:>8}")
        prev = bp
    return 0


if __name__ == "__main__":
    sys.exit(main())
