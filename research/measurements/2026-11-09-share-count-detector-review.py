"""Review of the open proposal in `research/2026-11-09-share-counts-and-corporate-actions.md`.

The note asks a later session to choose between three ways of handling a stock whose implied
share count (`marketCap / currentPrice`) has moved while the earnings behind its multiples
have not. Its own preference is option 2: detect it and print a note in `check_run_health`,
on the `sec_crosscheck` precedent.

This script measures the two things the note did not:

1. **The detector it asks for already exists**, anchored to the companies' filings rather
   than to the previous run - `sec_crosscheck.market_value_check`, called every run by
   `check_run_health.check_inputs_against_filings`. And it has been comparing **nothing**
   since 2026-11-08, because `financials` moved into `dashboard_detail.js` that day and the
   check reads the run's `dashboard_data.js` only. Measured here on the run's own artifacts.

2. **Event versus state.** A run-to-run delta fires on the one run date the share count
   moves and is silent for the months the distortion lasts; the filings-anchored check holds
   for the whole window. Measured on KMB, whose share count moved on 2026-11-04 - five runs
   before the newest one on file.

Run with no arguments. Reads `runs/*/` and the SEC companyfacts cache; writes nothing.

    python -I -X utf8 research/measurements/2026-11-09-share-count-detector-review.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import sec_crosscheck as X  # noqa: E402
import check_run_health as H  # noqa: E402


def runs_in_order():
    out = []
    for d in sorted((ROOT / "runs").glob("*")):
        meta = d / "meta.json"
        if not (meta.exists() and (d / "dashboard_data.js").exists()):
            continue
        try:
            start = json.loads(meta.read_text(encoding="utf-8")).get("start_time", "")
        except Exception:  # noqa: BLE001
            continue
        out.append((start, d))
    return [(s, d) for s, d in sorted(out)]


def read_shard(run_dir: Path) -> dict:
    """``dashboard_detail.js``'s payload, or {} where the run predates the shard."""
    p = run_dir / "dashboard_detail.js"
    if not p.exists():
        return {}
    t = p.read_text(encoding="utf-8")
    body = t[t.index("=") + 1:].strip().rstrip(";")
    return (json.loads(body) or {}).get("detail") or {}


def merged(payload: dict, shard: dict) -> dict:
    """The payload as a browser sees it once the shard has loaded."""
    out = dict(payload)
    sd = {t: dict(v) for t, v in (payload.get("stock_detail") or {}).items()}
    for t, part in shard.items():
        if t in sd:
            sd[t].update(part)
    out["stock_detail"] = sd
    return out


def far_set(payload: dict, facts) -> tuple[pd.DataFrame, set]:
    mv = X.market_value_check(payload, facts)
    if mv.empty:
        return mv, set()
    return mv, set(mv.loc[(mv["ratio"] - 1).abs() > X.FAR, "ticker"])


def main() -> None:
    from sec_fundamentals import FACTS_PATH
    facts = pd.read_parquet(FACTS_PATH)
    order = runs_in_order()
    print("runs with a payload of their own: %d" % len(order))
    print("sec_crosscheck.FAR = %.2f (the 'over 10%%' line)" % X.FAR)

    print("\n== 1. The check that was already there, on the run it was needed for")
    start, newest = order[-1]
    print("   newest run: %s  %s" % (start[:19], newest.name))
    raw_payload = H.load_payload(newest)
    shard = read_shard(newest)
    print("   dashboard_data.js stock_detail entries carrying 'financials': %d of %d"
          % (sum(1 for v in (raw_payload.get("stock_detail") or {}).values() if v.get("financials")),
             len(raw_payload.get("stock_detail") or {})))
    print("   dashboard_detail.js entries carrying 'financials':            %d of %d"
          % (sum(1 for v in shard.values() if v.get("financials")), len(shard)))

    mv_live, far_live = far_set(raw_payload, facts)
    print("\n   AS THE GATE RUNS IT TODAY (dashboard_data.js alone):")
    print("      compared %d stocks, %d over 10%%  <-- this is the '0/0' in the run log"
          % (len(mv_live), len(far_live)))

    mv_fix, far_fix = far_set(merged(raw_payload, shard), facts)
    print("   WITH THE SHARD MERGED BACK IN:")
    print("      compared %d stocks, %d over 10%%" % (len(mv_fix), len(far_fix)))
    if not mv_fix.empty:
        within5 = int(((mv_fix["ratio"] - 1).abs() <= 0.05).sum())
        print("      %d of %d within 5%%" % (within5, len(mv_fix)))

    print("\n   The run before the shard existed, for comparison:")
    for s, d in order[-4:-1]:
        mv, far = far_set(H.load_payload(d), facts)
        print("      %s  %s  compared %4d, over 10%%: %d  (dashboard_detail.js: %s)"
              % (s[:19], d.name, len(mv), len(far), (d / "dashboard_detail.js").exists()))

    print("\n== 2. Does the existing check see FOXA?")
    if not mv_fix.empty:
        row = mv_fix[mv_fix["ticker"] == "FOXA"]
        if row.empty:
            print("   FOXA not compared (no filed share count in window)")
        else:
            r = row.iloc[0]
            print("   FOXA implied %.0f vs filed %.0f -> ratio %.4f (%+.1f%%), over the 10%% line: %s"
                  % (r["implied"], r["filed"], r["ratio"], 100 * (r["ratio"] - 1),
                     abs(r["ratio"] - 1) > X.FAR))
        print("   the 10 furthest from their filed count on this run:")
        top = mv_fix.assign(off=(mv_fix["ratio"] - 1).abs()).sort_values("off", ascending=False)
        for _, r in top.head(10).iterrows():
            print("      %-6s ratio %7.4f  (%+8.1f%%)" % (r["ticker"], r["ratio"], 100 * (r["ratio"] - 1)))

    print("\n== 3. Event or state: what each detector says about KMB and FOXA, run by run")
    print("   KMB's implied share count moved +84.5%% between the 11-03 and 11-04 runs")
    print("   (8-K of 2026-11-04, completion of the Exchange Offers).")
    print("   %-20s %-10s  %-28s %s" % ("run", "dir", "run-to-run >5% (event)", "filings >10% (state)"))
    watch = ("KMB", "FOXA", "CHTR")
    prev_shares = None
    for s, d in order:
        try:
            raw = pd.read_parquet(d / "00_raw_fetch.parquet").set_index("Ticker")
            shares = pd.to_numeric(raw["marketCap"], errors="coerce") / \
                pd.to_numeric(raw["currentPrice"], errors="coerce")
        except Exception:  # noqa: BLE001
            continue
        event = []
        if prev_shares is not None:
            common = shares.index.intersection(prev_shares.index)
            move = (shares.loc[common] / prev_shares.loc[common] - 1).dropna()
            event = [t for t in watch if t in move.index and abs(move[t]) > 0.05]
        pay = H.load_payload(d)
        pay = merged(pay, read_shard(d))
        _, far = far_set(pay, facts)
        state = [t for t in watch if t in far]
        print("   %-20s %-10s  %-28s %s"
              % (s[:19], d.name, ", ".join(event) or "-", ", ".join(state) or "-"))
        prev_shares = shares

    print("\n== 4. How many stocks each detector would put in front of a session")
    print("   (the void condition in the note's section 5 is >3 stocks a run date)")
    prev_shares = None
    per_date = {}
    for s, d in order:
        try:
            raw = pd.read_parquet(d / "00_raw_fetch.parquet").set_index("Ticker")
            shares = pd.to_numeric(raw["marketCap"], errors="coerce") / \
                pd.to_numeric(raw["currentPrice"], errors="coerce")
        except Exception:  # noqa: BLE001
            continue
        if prev_shares is not None:
            common = shares.index.intersection(prev_shares.index)
            move = (shares.loc[common] / prev_shares.loc[common] - 1).dropna()
            n = int((move.abs() > 0.05).sum())
            per_date.setdefault(s[:10], 0)
            per_date[s[:10]] += n
        prev_shares = shares
    print("   run-to-run, per run date:")
    for k in sorted(per_date):
        print("      %s  %d" % (k, per_date[k]))
    print("      dates with any firing: %d of %d; mean per date %.2f"
          % (sum(1 for v in per_date.values() if v), len(per_date),
             sum(per_date.values()) / max(1, len(per_date))))


def section5(facts) -> None:
    """How should the note name them? By magnitude, or by what is new?

    The reporting half of the note's option 2 needs a short list a session can act on. The
    obvious construction - the stocks furthest from their filed count - is measured here against
    the one that compares the run's set with the page the site is serving.
    """
    import subprocess

    order = runs_in_order()
    start, newest = order[-1]
    cur = merged(H.load_payload(newest), read_shard(newest))
    mv, far = far_set(cur, facts)

    print("\n== 5. Naming them: by magnitude, or by what is new since the published page?")
    by_mag = mv.assign(off=(mv["ratio"] - 1).abs()).sort_values("off", ascending=False)
    top8 = list(by_mag.head(8)["ticker"])
    print("   the 8 furthest from their filed count: %s" % ", ".join(top8))
    print("   FOXA's position in that order: %d of %d"
          % (list(by_mag["ticker"]).index("FOXA") + 1 if "FOXA" in set(by_mag["ticker"]) else -1, len(by_mag)))
    print("   a magnitude list of 8 would name FOXA: %s" % ("FOXA" in top8))

    base = {}
    for name in ("dashboard_data.js", "dashboard_detail.js"):
        r = subprocess.run(["git", "-C", str(ROOT), "show", "origin/main:" + name],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("   the published %s could not be read - no baseline" % name)
            return
        t = r.stdout
        base[name] = json.loads(t[t.index("{"):].rstrip().rstrip(";"))
    pub = base["dashboard_data.js"]
    for t_, part in (base["dashboard_detail.js"].get("detail") or {}).items():
        if t_ in (pub.get("stock_detail") or {}):
            pub["stock_detail"][t_].update(part)
    _, far_pub = far_set(pub, facts)
    print("   published page: %s, %d over 10%%"
          % ((pub.get("kpis") or {}).get("run_timestamp", "?")[:19], len(far_pub)))
    print("   this run:       %s, %d over 10%%" % (start[:19], len(far)))
    print("   NEW since the published page: %s" % (", ".join(sorted(far - far_pub)) or "-"))
    print("   gone since it:                %s" % (", ".join(sorted(far_pub - far)) or "-"))
    print("   standing in both:             %d" % len(far & far_pub))


if __name__ == "__main__":
    main()
    from sec_fundamentals import FACTS_PATH as _FP
    section5(pd.read_parquet(_FP))
