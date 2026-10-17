"""What pairing on each month's common stocks changed - plan/research-engine.md queue item 8a.

Reads `research/experiments/ledger.jsonl` only; runs nothing. For every registered experiment it
takes the newest method-version-1 row and the newest method-version-2 row **on the same data
version** and compares what each reported, so the only difference between the two readings is how
the comparison was measured.

Under version 1 an experiment's IC was computed over the stocks *it* could rank and differenced
against the baseline's IC over the stocks the *baseline* could rank, so part of every difference
was a difference in who was ranked. Version 2 recomputes both sides on each month's intersection.

    python research/measurements/2026-10-17-method-version-2-effect.py

Writes `2026-10-17-method-version-2-effect.json` beside itself and prints the table.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "research" / "experiments" / "ledger.jsonl"
OUT = Path(__file__).with_suffix(".json")
PERIODS = ("discovery", "holdout", "presample")


def newest_per_method(rows: list[dict]) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    for r in rows:                       # the ledger is append-only and in order
        out[(r["id"], r.get("method_version", 1))] = r
    return out


def main() -> int:
    rows = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    newest = newest_per_method(rows)
    v2_data = {r.get("data_hash") for (i, m), r in newest.items() if m == 2}
    if len(v2_data) != 1:
        raise SystemExit(f"method-2 rows span more than one data version: {v2_data}")
    data_hash = v2_data.pop()

    compared, skipped, verdict_moves = [], [], []
    for (exp_id, method), new in sorted(newest.items()):
        if method != 2 or exp_id == "baseline" or new.get("kind") == "evaluation":
            continue
        old = newest.get((exp_id, 1))
        if not old or old.get("data_hash") != data_hash:
            skipped.append(exp_id)       # no version-1 row on this data: nothing fair to compare
            continue
        row: dict = {"id": exp_id}
        for p in PERIODS:
            o = (old["summary"].get(p) or {}).get("ic_vs_baseline")
            n = (new["summary"].get(p) or {}).get("ic_vs_baseline")
            if not o or not n:
                continue
            row[p] = {"v1_mean": o.get("mean"), "v2_mean": n.get("mean"),
                      "shift": round(n.get("mean", 0) - o.get("mean", 0), 5),
                      "v1_t": o.get("t"), "v2_t": n.get("t"),
                      "n_common": (new["summary"].get(p) or {}).get("n_common")}
        compared.append(row)

    shifts = [row[p]["shift"] for row in compared for p in PERIODS if p in row]
    towards_zero = sum(1 for row in compared for p in PERIODS
                       if p in row and abs(row[p]["v2_mean"]) < abs(row[p]["v1_mean"]))
    out = {
        "what": "method version 1 vs 2 (pairing on each month's common stocks), same data version",
        "data_hash": data_hash,
        "experiments_compared": len(compared),
        "comparisons": len(shifts),
        "skipped_no_v1_row_on_this_data": skipped,
        "abs_shift_mean": round(statistics.fmean(abs(s) for s in shifts), 5) if shifts else None,
        "abs_shift_max": round(max((abs(s) for s in shifts), default=0), 5),
        "shift_moved_the_difference_towards_zero": towards_zero,
        "rows": compared,
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print(f"data version {data_hash}; {len(compared)} experiments, {len(shifts)} period comparisons")
    print(f"mean |shift| {out['abs_shift_mean']}, max {out['abs_shift_max']}, "
          f"{towards_zero}/{len(shifts)} moved towards zero")
    print(f"{'experiment':30} {'period':10} {'v1':>8} {'v2':>8} {'shift':>8} {'v1 t':>6} {'v2 t':>6} {'common':>7}")
    for row in compared:
        for p in PERIODS:
            if p in row:
                c = row[p]
                print(f"{row['id']:30} {p:10} {c['v1_mean']:+8.4f} {c['v2_mean']:+8.4f} "
                      f"{c['shift']:+8.4f} {c['v1_t']:6} {c['v2_t']:6} {c['n_common'] or 0:7}")
    if skipped:
        print("no comparable version-1 row:", ", ".join(skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
