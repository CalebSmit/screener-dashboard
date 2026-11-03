"""Pre-registration 3, measured: does ranking within (Sector, size_band) still rank within sector?

`research/2026-11-02-universe-expansion-phase0.md` section 3 proposes grouping percentiles by
`(Sector, size_band)` instead of `Sector`, with the sub-ten-peer fallback changed from
universe-wide to **band-wide**. It calls the repo change "small", says the S&P 500 comes out
byte-identical, and names exactly one case where the ten-peer minimum would bite: "the S&P 400
has 6 Communication Services members".

That example counts *members*. `factor_engine.compute_sector_percentiles` counts
**valid (non-NaN) values for that one metric**:

    valid = grp[col].dropna()
    if len(valid) < SECTOR_MIN_PEERS:   # 10

So the question is not how many sector-band groups have under ten members -- it is how many
(sector, band, metric) cells have under ten *populated* values. This script measures:

  1. How often the fallback fires **today**, on the published S&P 500 run, per metric and sector,
     and what share of composite weight sits in a fallen-back cell. (The note implies this is an
     edge case; if it is already common at 11 sectors x ~500 names, the band split multiplies it.)
  2. Per-metric availability on that same run, which is what decides cell 1 above.
  3. Sector member counts for the S&P 400 and S&P 600 from the free constituent lists, giving a
     hard **lower bound** on the number of sector-band groups where every metric must fall back
     (a cell cannot have more valid values than the group has members).
  4. The projection for the three-band S&P 1500, combining 2 and 3.

Run:  python research/measurements/2026-11-03-sector-band-peer-counts.py
Reads the published run via run_context; fetches two Wikipedia lists (cached under
data/survivorship/). Writes nothing else.
"""

from __future__ import annotations

import io
import json
import pathlib
import sys
import urllib.request

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import factor_engine as fe  # noqa: E402
import run_context  # noqa: E402

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
CACHE_DIR = pathlib.Path("data/survivorship")
WIKI = {
    "S&P 400": ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "sp400.html"),
    "S&P 600": ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "sp600.html"),
}


def wiki_sectors(label: str) -> pd.Series:
    """GICS Sector counts for a free constituent list, cached on disk."""
    url, fname = WIKI[label]
    path = CACHE_DIR / fname
    if not path.exists():
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(r.read())
        print(f"  fetched {url}")
    tables = pd.read_html(io.StringIO(path.read_text(encoding="utf-8", errors="replace")))
    for t in tables:
        cols = {str(c).strip() for c in t.columns}
        if "GICS Sector" in cols and len(t) > 100:
            s = t["GICS Sector"].astype(str).str.strip()
            print(f"  {label}: {len(t)} members from {path.name}")
            return s.value_counts()
    raise SystemExit(f"no constituent table with a GICS Sector column found for {label}")


def main() -> int:
    cfg = fe.load_config()
    generic, bank = fe.weighted_metric_sets(cfg)
    weighted = sorted(set(generic) | set(bank))
    # Composite weight of each metric under the generic profile (category points x within-
    # category fraction), for the weight-share figure.
    gen_w: dict[str, float] = {}
    for cat, pts in cfg["factor_weights"].items():
        prof = fe.metric_weight_profiles(cfg, cat).get("generic", {})
        for m, frac in prof.items():
            gen_w[m] = gen_w.get(m, 0.0) + pts * frac

    run_dir = pathlib.Path(run_context.latest_run_dir())
    df = pd.read_parquet(run_dir / "02_outliers_flagged.parquet")
    print(f"run {run_dir.name}: {len(df)} rows, {df['Sector'].nunique()} sectors")
    print(f"weighted metrics: {len(generic)} generic, {len(bank)} bank, {len(weighted)} union")
    print(f"SECTOR_MIN_PEERS = {fe.SECTOR_MIN_PEERS}")

    # A bank-only metric carries no weight for a non-bank stock, so its percentile falling
    # back outside Financials changes no score. Count a cell only where the metric can
    # actually reach a composite. (Financials is the proxy for the bank-like cohort; the real
    # rule is factor_engine's bank detection, which the note says will fire far more often in
    # the S&P 600's 118 Financials.)
    bank_only = set(bank) - set(generic)

    def scoped(sector: str) -> list[str]:
        if sector == "Financials":
            return weighted
        return [c for c in weighted if c not in bank_only]

    # --- 1. How often the fallback fires today ------------------------------------
    print("\n=== 1. Today, S&P 500: (sector, metric) cells below the ten-peer minimum ===")
    print(f"bank-only metrics (counted in Financials alone): {sorted(bank_only)}")
    cells = 0
    fallback = 0
    per_metric: dict[str, int] = {}
    per_sector: dict[str, int] = {}
    for sector, grp in df.groupby("Sector"):
        for col in scoped(sector):
            if col not in df.columns:
                continue
            cells += 1
            n_valid = int(grp[col].notna().sum())
            if n_valid < fe.SECTOR_MIN_PEERS:
                fallback += 1
                per_metric[col] = per_metric.get(col, 0) + 1
                per_sector[sector] = per_sector.get(sector, 0) + 1
    print(f"weight-bearing cells: {cells}   falling back to universe-wide: "
          f"{fallback} ({fallback / cells:.1%})")
    if per_metric:
        print("\n  by metric (cells that fall back, of 11 sectors):")
        for col, k in sorted(per_metric.items(), key=lambda kv: -kv[1]):
            print(f"    {col:<30}{k:>3}   generic weight {gen_w.get(col, 0.0):>6.2f}")
    if per_sector:
        print("\n  by sector:")
        for s, k in sorted(per_sector.items(), key=lambda kv: -kv[1]):
            n = int((df["Sector"] == s).sum())
            print(f"    {s:<26} {k:>3} of {len(scoped(s))} metrics   ({n} members)")

    # --- 2. Per-metric availability on the S&P 500 --------------------------------
    print("\n=== 2. Per-metric availability, S&P 500 (what decides cell 1 after the split) ===")
    avail = {}
    for col in weighted:
        if col in df.columns:
            avail[col] = float(df[col].notna().mean())
    lo = sorted(avail.items(), key=lambda kv: kv[1])
    print("  ten least-populated weighted metrics:")
    for col, a in lo[:10]:
        print(f"    {col:<30}{a:>8.1%}")
    import statistics
    print(f"  median availability across {len(avail)} weighted metrics: "
          f"{statistics.median(avail.values()):.1%}")

    # --- 3. Sector member counts per band -----------------------------------------
    print("\n=== 3. Sector member counts per band (free constituent lists) ===")
    bands: dict[str, pd.Series] = {}
    sp500 = df["Sector"].value_counts()
    bands["S&P 500"] = sp500
    for label in WIKI:
        bands[label] = wiki_sectors(label)

    sectors = sorted(set().union(*[set(s.index) for s in bands.values()]))
    print(f"\n  {'Sector':<26}{'S&P 500':>9}{'S&P 400':>9}{'S&P 600':>9}")
    under10 = {k: 0 for k in bands}
    for sec in sectors:
        row = [int(bands[k].get(sec, 0)) for k in ("S&P 500", "S&P 400", "S&P 600")]
        marks = []
        for k, n in zip(("S&P 500", "S&P 400", "S&P 600"), row):
            if n < fe.SECTOR_MIN_PEERS:
                under10[k] += 1
                marks.append("*")
            else:
                marks.append(" ")
        print(f"  {sec:<26}" + "".join(f"{n:>8}{m}" for n, m in zip(row, marks)))
    print("  (* = fewer than ten members, so EVERY metric in that cell must fall back)")
    print("\n  sector-band groups under ten members:")
    for k in ("S&P 500", "S&P 400", "S&P 600"):
        print(f"    {k}: {under10[k]} of {len(bands[k])} sectors   "
              f"({int(bands[k].sum())} members)")

    # --- 4. The projection --------------------------------------------------------
    print("\n=== 4. Projected (sector, band, metric) cells falling back, S&P 1500 ===")
    print("Lower bound: a cell cannot have more valid values than its group has members, so")
    print("every metric in an under-ten-member group falls back regardless of availability.")
    total_cells = 0
    bound = 0
    for k in ("S&P 500", "S&P 400", "S&P 600"):
        for sec, n in bands[k].items():
            total_cells += len(scoped(sec))
            if n < fe.SECTOR_MIN_PEERS:
                bound += len(scoped(sec))
                print(f"    {k} / {sec}: {n} members -> all {len(scoped(sec))} "
                      f"weighted metrics fall back to band-wide")
    print(f"  weight-bearing cells after the split: {total_cells}")
    print(f"  guaranteed fallbacks from member count alone: {bound} ({bound / total_cells:.1%})")
    print("\nAvailability-aware estimate: a cell falls back when (members x availability) < 10.")
    print("Availability below the large caps is NOT measured here; the note's own section 4")
    print("probe gives median analyst counts of 20 / 11 / 6 for the 500 / 400 / 600, so the")
    print("five analyst metrics are the ones to watch. Applying the S&P 500's OWN availability")
    print("rates (an upper bound on small-cap availability, hence a lower bound on fallbacks):")
    est = 0
    marginal: list[str] = []
    for k in ("S&P 500", "S&P 400", "S&P 600"):
        c = 0
        n_cells = 0
        for sec, n in bands[k].items():
            for col in scoped(sec):
                n_cells += 1
                need = fe.SECTOR_MIN_PEERS / max(n, 1)
                if n * avail.get(col, 1.0) < fe.SECTOR_MIN_PEERS:
                    c += 1
                elif need > 0.60:
                    marginal.append(
                        f"{k} / {sec} ({n} members): {col} needs "
                        f"{need:.0%} availability to stay sector-ranked "
                        f"(S&P 500 rate {avail.get(col, 1.0):.0%})")
        est += c
        print(f"    {k}: {c} of {n_cells} cells ({c / n_cells:.1%})")
    print(f"  total: {est} of {total_cells} ({est / total_cells:.1%})")
    print("\n  Marginal cells -- sector-ranked today only because availability is high, so a")
    print("  small-cap availability drop pushes them to band-wide:")
    seen = set()
    for m in marginal:
        key = m.split(":")[0]
        if key in seen:
            continue
        seen.add(key)
        print(f"    {m}")
    if not marginal:
        print("    (none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
