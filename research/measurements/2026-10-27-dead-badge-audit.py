"""The eleven badges in the dead `renderFlagsWarnings`: how many stocks each fires for, and
whether its information reaches the reader by any other route.

The audit behind `tests/test_dead_badges.py` and NIGHTLY_LOG.md 2026-10-27. Reads the committed
`dashboard_data.js` (the page the public site is serving) and the newest run's scored frame, so
it is reproducible from the repository at any commit.

    python research/measurements/2026-10-27-dead-badge-audit.py

Run it against a pre-fix commit and the four gaps are the four rows printed with a bare "no";
run it after and three of them name the sentence that now carries them, and the fourth is gone
from the payload entirely.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def payload() -> dict:
    txt = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    return json.loads(txt[txt.index("{"):txt.rstrip().rstrip(";").rindex("}") + 1])


# badge -> (predicate on (detail, flags), where its information reaches the reader)
BADGES = [
    ("value trap", lambda s, f: s.get("vt") and (f.get("vt_severity") or 0) > 0,
     "table flag column; stock_summary; 'The case against'"),
    ("growth trap", lambda s, f: s.get("gt") and (f.get("gt_severity") or 0) > 0,
     "table flag column; stock_summary; 'The case against'"),
    ("Beneish", lambda s, f: f.get("beneish_flag"),
     "stock_summary; 'The case against'"),
    ("channel stuffing", lambda s, f: f.get("channel_stuffing"),
     "stock_summary; 'The case against'"),
    ("stale financials", lambda s, f: f.get("stale_data"),
     "Filing Age badge; stock_summary; 'The case against'"),
    ("EPS basis mismatch", lambda s, f: s.get("eps_mismatch"),
     "provenance badge; data-quality KPIs"),
    ("bank-like", lambda s, f: f.get("is_bank"),
     "ROE/ROIC column label; methodology page"),
    ("LTM 3-of-4 annualised", lambda s, f: f.get("ltm_annualized"),
     "#provenance-ltm-annualized (2026-10-27)"),
    ("enterprise value substituted", lambda s, f: f.get("ev_flag"),
     "#provenance-ev-substituted (2026-10-27)"),
    ("beta overlap under 80%",
     lambda s, f: f.get("beta_overlap_pct") is not None and f["beta_overlap_pct"] < 80,
     "#provenance-beta-overlap (2026-10-27)"),
    ("financial-sector caveat", lambda s, f: f.get("fin_caveat"),
     "deleted 2026-10-27: never set on this pipeline"),
]


def main() -> int:
    d = payload()
    sd = d["stock_detail"]
    gen = (ROOT / "generate_dashboard.py").read_text(encoding="utf-8")
    dead = "function renderFlagsWarnings" in gen

    print("published run: %s, %d stocks" % (d.get("kpis", {}).get("run_timestamp"), len(sd)))
    print("renderFlagsWarnings present: %s" % dead)
    print()
    print("%-30s %6s  %s" % ("badge", "stocks", "where its information reaches the reader"))
    print("-" * 100)
    for name, pred, where in BADGES:
        hits = sorted(t for t, s in sd.items() if pred(s, s.get("flags") or {}))
        shown = re.sub(r"#provenance-\S+ \(2026-10-27\)", lambda m: m.group(0), where)
        print("%-30s %6d  %s" % (name, len(hits), shown))
        if 0 < len(hits) <= 20:
            print("%-30s         %s" % ("", ", ".join(hits)))

    # the defect, stated as the arithmetic of the lead sentence's own condition
    print()
    print("the lead sentence picks on (ltm_rolled || q_rejected || ltm_annualized);")
    print("before 2026-10-27 the third term was absent, so these stocks were told their figures")
    print("were 'the sum of the last four reported quarters':")
    for t, s in sorted(sd.items()):
        f = s.get("flags") or {}
        if f.get("ltm_annualized") and not (f.get("ltm_rolled") or f.get("q_rejected")):
            print("  %-6s rank %-4s data_source=%s" % (t, s.get("rank"), s.get("data_source")))

    # and the one that was dead at source
    print()
    try:
        import pandas as pd

        import run_context
        rd = run_context.latest_run_dir()
        f = pathlib.Path(rd) / "05_final_scored.parquet"
        if f.exists():
            cols = pd.read_parquet(f).columns
            print("newest scored frame %s: Financial_Sector_Caveat present = %s, _is_bank_like = %s"
                  % (pathlib.Path(rd).name, "Financial_Sector_Caveat" in cols,
                     "_is_bank_like" in cols))
    except Exception as exc:  # noqa: BLE001
        print("scored frame not read (%s: %s)" % (type(exc).__name__, exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
