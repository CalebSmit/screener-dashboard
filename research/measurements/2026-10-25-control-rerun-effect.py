"""Did adding the positive control move any existing result? (2026-10-25)

`plan/research-engine.md` item 4(c). Registering the control changed `registry.yaml` and
`v2_experiments.py`, both in `RESULT_PATHS`, so every ledger row went stale and all 31 were rerun.
The diff adds a parallel code path (`run_controls`, `control_verdict`, a report section) and touches
nothing on the experiment path, so **no number should have moved** - but "it should be identical" is
how a silent change gets published (the 2026-10-18 rerun made the same check for the same reason).

So: for every id, compare the newest ledger row's stored statistics with the previous row's, at full
precision, and report the largest absolute difference found anywhere.

    python research/measurements/2026-10-25-control-rerun-effect.py

Writes the JSON beside this file. Read-only; imports nothing from production.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "research" / "experiments" / "ledger.jsonl"
OUT = Path(__file__).with_suffix(".json")


def flatten(obj, prefix: str = "") -> dict[str, float]:
    """Every numeric leaf of a row's statistics, by path."""
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out |= flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out |= flatten(v, f"{prefix}[{i}]")
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out[prefix] = float(obj)
    return out


def main() -> None:
    rows: dict[str, list[dict]] = {}
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows.setdefault(r["id"], []).append(r)

    report, worst = {}, {"id": None, "key": None, "diff": 0.0}
    for rid, hist in sorted(rows.items()):
        if len(hist) < 2:
            report[rid] = {"compared": False, "why": "only one run in the ledger"}
            continue
        new, old = hist[-1], hist[-2]
        keys = ("summary", "ic_by_month", "tmu_by_month", "base_ic_by_month", "n_common_by_month")
        a = flatten({k: new.get(k) for k in keys})
        b = flatten({k: old.get(k) for k in keys})
        shared = sorted(set(a) & set(b))
        diffs = {k: abs(a[k] - b[k]) for k in shared if a[k] != b[k]}
        biggest = max(diffs.items(), key=lambda kv: kv[1], default=(None, 0.0))
        report[rid] = {"compared": True, "statistics": len(shared),
                       "only_in_new": sorted(set(a) - set(b)), "only_in_old": sorted(set(b) - set(a)),
                       "differing": len(diffs), "largest": biggest[1], "largest_key": biggest[0],
                       "new_code_hash": new.get("code_hash"), "old_code_hash": old.get("code_hash"),
                       "new_run_at": new.get("run_at"), "old_run_at": old.get("run_at")}
        if biggest[1] > worst["diff"]:
            worst = {"id": rid, "key": biggest[0], "diff": biggest[1]}

    compared = [r for r in report.values() if r.get("compared")]
    payload = {
        "what": "newest ledger row vs the previous one, per id, after the control was registered",
        "why": "registering a control stales every row; the diff cannot move an experiment's number,"
               " and that has to be checked rather than asserted",
        "ids_compared": len(compared),
        "ids_not_compared": [k for k, v in report.items() if not v.get("compared")],
        "statistics_compared": sum(r["statistics"] for r in compared),
        "ids_with_any_difference": [k for k, v in report.items() if v.get("differing")],
        "largest_difference_anywhere": worst,
        "per_id": report,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"{payload['ids_compared']} ids, {payload['statistics_compared']:,} statistics compared")
    print(f"ids with any difference: {payload['ids_with_any_difference'] or 'none'}")
    print(f"largest difference anywhere: {worst}")
    print(f"not compared: {payload['ids_not_compared'] or 'none'}")
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
