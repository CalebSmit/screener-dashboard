"""What the pre-registered 10% stop did, month by month, from the ledger's own rows.

Reads ``research/experiments/ledger.jsonl`` only - no re-simulation, so the figures in the note are
the published ones. Writes ``2026-10-22-stop-loss-arm.json`` beside this file.

    python research/measurements/2026-10-22-stop-loss-arm.py

Not decision-grade for anything (CLAUDE.md rule 5, until 2027-02-11).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LEDGER = ROOT / "research" / "experiments" / "ledger.jsonl"
OUT = Path(__file__).with_suffix(".json")

BASE, ARM = "bot-rule-v1.1", "bot-rule-v1.1-stop10"
PERIODS = {"presample": ("2012-01-31", "2019-12-31"), "discovery": ("2020-01-31", "2023-06-30"),
           "holdout": ("2023-07-31", "2026-08-31"), "all": ("2020-01-31", "2026-08-31")}


def latest(ident: str) -> dict:
    row = None
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["id"] == ident:
                row = r
    if row is None:
        raise SystemExit(f"{ident} has no ledger row - run `python v2_experiments.py evaluate`")
    return row


def main() -> int:
    b, a = latest(BASE), latest(ARM)
    same_code = b.get("code_hash") == a.get("code_hash") and b.get("data_hash") == a.get("data_hash")
    bm_b, bm_a = b["by_month"], a["by_month"]
    months = sorted(set(bm_b) & set(bm_a))
    out = {"base": BASE, "arm": ARM, "months": len(months),
           "comparable_provenance": same_code,
           "code_version": a.get("code_version"), "data_hash": a.get("data_hash"),
           "stops": {}, "periods": {}, "worst_months": []}

    stopped = {m: bm_a[m]["stopped"] for m in months}
    held = {m: 0 for m in months}                      # positions held is not in by_month; use the
    mean_held = {p: a["summary"][p]["mean_held"] for p in PERIODS if p in a["summary"]}  # summary's
    total = sum(stopped.values())
    out["stops"] = {
        "total": total, "months_with_none": sum(1 for v in stopped.values() if v == 0),
        "per_month_mean": round(total / len(months), 3),
        "per_month_median": sorted(stopped.values())[len(months) // 2],
        "busiest": sorted(stopped.items(), key=lambda kv: -kv[1])[:6],
        "mean_held_by_period": mean_held,
        # the paper's comparable figure: about 10.05% of the winners' positions closed in an
        # average month (Han, Zhou & Zhu 2016, p. 8)
        "share_of_positions_per_month": {
            p: round(sum(stopped[m] for m in months if PERIODS[p][0] <= m <= PERIODS[p][1])
                     / max(sum(1 for m in months if PERIODS[p][0] <= m <= PERIODS[p][1]), 1)
                     / mean_held[p], 4)
            for p in mean_held},
    }
    del held

    for p, (lo, hi) in PERIODS.items():
        if p not in a["summary"] or p not in b["summary"]:
            continue
        sb, sa = b["summary"][p], a["summary"][p]
        out["periods"][p] = {
            k: {"base": sb.get(k), "arm": sa.get(k)}
            for k in ("worst_month", "max_drawdown", "universe_worst_month",
                      "universe_max_drawdown", "turnover_a_year", "positions_stopped")
        } | {
            "ret_mean": {"base": sb["ret"].get("mean"), "arm": sa["ret"].get("mean")},
            "excess_mean": {"base": sb["excess"].get("mean"), "arm": sa["excess"].get("mean")},
            "excess_t": {"base": sb["excess"].get("t"), "arm": sa["excess"].get("t")},
            "both_figures_better": bool(sa["worst_month"] > sb["worst_month"]
                                        and sa["max_drawdown"] > sb["max_drawdown"]),
            "months": sum(1 for m in months if lo <= m <= hi),
        }

    # the ten months the unstopped rule lost most, and what the stop did in each
    worst = sorted(months, key=lambda m: bm_b[m]["ret"])[:10]
    out["worst_months"] = [{"month": m, "base_ret": bm_b[m]["ret"], "arm_ret": bm_a[m]["ret"],
                            "universe": bm_b[m]["universe"], "stopped": bm_a[m]["stopped"],
                            "stop_helped": bm_a[m]["ret"] > bm_b[m]["ret"]} for m in worst]
    out["stop_helped_in_worst_ten"] = sum(1 for r in out["worst_months"] if r["stop_helped"])
    # and across every month: how often the stop helped at all, and how often it helped in a
    # month the rule was already losing
    out["stop_helped_months"] = sum(1 for m in months if bm_a[m]["ret"] > bm_b[m]["ret"])
    out["stop_changed_months"] = sum(1 for m in months if bm_a[m]["ret"] != bm_b[m]["ret"])
    down = [m for m in months if bm_b[m]["ret"] < 0]
    out["down_months"] = len(down)
    out["stop_helped_down_months"] = sum(1 for m in down if bm_a[m]["ret"] > bm_b[m]["ret"])

    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
