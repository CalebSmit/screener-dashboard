"""Did rebuilding the v2 months after the 2026-10-19 ``pit_record`` fixes change any verdict? Measured.

Pairs each registered id's row the report read before the rebuild (the newest row carrying the
pre-rebuild ``content_hash``, ``--before-content``) with its newest row now, and compares the two
verdicts the report prints (held-out and 2012-2019, ``v2_experiments.verdict`` /
``verdict_presample`` - the engine's own functions, with the bar's test count as the report uses
it), plus each period's IC difference against the baseline and its t, and the bot evaluation's
excess returns. Writes the JSON beside this file and prints every verdict that changed.

Usage: python research/measurements/2026-10-19-pit-fix-rerun-verdicts.py [--before-content 318ed0cfc297]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import v2_experiments as X  # noqa: E402

OUT = Path(__file__).with_suffix(".json")
PERIODS = ("discovery", "holdout", "all", "presample")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before-content", default="318ed0cfc297")
    a = ap.parse_args()
    rows = [json.loads(x) for x in X.LEDGER.read_text(encoding="utf-8").splitlines() if x.strip()]
    before, after = {}, {}
    for r in rows:
        if r.get("content_hash") == a.before_content:
            before[r["id"]] = r
        after[r["id"]] = r
    reg = X.load_registry()
    n = len(reg)
    out, changed, worst = {}, [], 0.0
    for e in reg:
        i = e["id"]
        b, c = before.get(i), after.get(i)
        if not b or not c:
            out[i] = {"compared": False}
            continue
        vb, va = X.verdict(e, b, n), X.verdict(e, c, n)
        pb, pa = X.verdict_presample(e, b, n), X.verdict_presample(e, c, n)
        rec = {"verdict": [vb, va], "verdict_presample": [pb, pa], "periods": {}}
        for p in PERIODS:
            sb, sa = b["summary"].get(p, {}), c["summary"].get(p, {})
            rec["periods"][p] = {"ic": [sb.get("ic", {}).get("mean"), sa.get("ic", {}).get("mean")]}
            if "ic_vs_baseline" in sa:
                vsb, vsa = sb.get("ic_vs_baseline", {}), sa["ic_vs_baseline"]
                rec["periods"][p]["vs_baseline"] = [vsb.get("mean"), vsa.get("mean")]
                rec["periods"][p]["t"] = [vsb.get("t"), vsa.get("t")]
                if vsb.get("mean") is not None and vsa.get("mean") is not None:
                    worst = max(worst, abs(vsa["mean"] - vsb["mean"]))
        if vb != va or pb != pa:
            changed.append(i)
        out[i] = rec
    evals = {}
    for e in X.load_evaluations():
        b, c = before.get(e["id"]), after.get(e["id"])
        if b and c:
            evals[e["id"]] = {p: {k: [b["summary"].get(p, {}).get(k), c["summary"].get(p, {}).get(k)]
                                  for k in ("excess", "excess_index", "turnover_a_year", "max_drawdown")}
                              for p in PERIODS if p in c["summary"]}
    res = {"before_content": a.before_content, "after_content": next(iter(after.values()))["content_hash"],
           "n_registered": n, "verdicts_changed": changed,
           "largest_change_in_an_ic_difference": worst, "experiments": out, "evaluations": evals}
    OUT.write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"compared {sum(1 for v in out.values() if v.get('compared', True))} of {n}; "
          f"verdicts changed: {changed or 'none'}; largest change in an IC difference {worst:.5f}")
    for i in changed:
        print(f"  {i}: {out[i]['verdict']} | {out[i]['verdict_presample']}")
    for i, v in out.items():
        if v.get("compared", True):
            t = {p: v["periods"][p].get("t") for p in PERIODS if "t" in v["periods"][p]}
            print(f"  {i}: t {t}")
    print(json.dumps(evals, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
