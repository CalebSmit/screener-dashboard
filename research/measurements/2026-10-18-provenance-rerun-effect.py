"""Did item 8b's rerun change any number? Measured, not assumed.

Queue item 8(b) added provenance fields to every ledger row and reran all 28 experiments and the
one evaluation to populate them. Neither the data nor the method changed, so every figure should be
identical to the method-version-2 run of 2026-10-17 - but "should be" is how a silent regression
gets published, and the research page quotes these numbers.

So this pairs each id's newest row (today's, carrying ``code_version``) with the newest row that
**predates 8b** - the last one written before any provenance field existed - and compares every
statistic in the summary, to the full stored precision. It writes
``2026-10-18-provenance-rerun-effect.json`` and prints the largest absolute difference it found.

Pairing against the pre-8b row rather than simply the previous one is deliberate: the day took two
reruns, the second after the rerun itself exposed a defect in the dirty marker, so "the previous
row" would compare one 8b rerun against another and answer a question nobody asked. The comparison
that matters is against the last figures published without provenance, and it stays that comparison
however many times the engine is rerun afterwards.

Run: python research/measurements/2026-10-18-provenance-rerun-effect.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LEDGER = ROOT / "research" / "experiments" / "ledger.jsonl"
OUT = Path(__file__).with_suffix(".json")

PROVENANCE = {"code_version", "code_hash", "code_hashes", "content_hash", "content_parts",
              "base_config_hash", "run_at", "commit"}


def walk(node, path=""):
    """Every leaf of a nested dict/list, as (dotted path, value)."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f"{path}[{i}]")
    else:
        yield path, node


def main() -> int:
    rows = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id: dict[str, list[dict]] = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(r)

    report, worst = {}, (0.0, None)
    for i, runs in sorted(by_id.items()):
        if len(runs) < 2:
            report[i] = {"compared": False, "why": "only one run on file"}
            continue
        new = runs[-1]
        if not new.get("code_version"):
            report[i] = {"compared": False, "why": "newest row predates item 8b"}
            continue
        pre = [r for r in runs if not r.get("code_version")]
        if not pre:
            report[i] = {"compared": False, "why": "no pre-8b run on file to compare against"}
            continue
        old = pre[-1]
        a = {k: v for k, v in walk({k: v for k, v in old.items() if k not in PROVENANCE})}
        b = {k: v for k, v in walk({k: v for k, v in new.items() if k not in PROVENANCE})}
        diffs, numeric = {}, []
        for k in sorted(set(a) | set(b)):
            x, y = a.get(k), b.get(k)
            if x == y:
                continue
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
                d = abs(float(y) - float(x))
                numeric.append(d)
                if d > worst[0]:
                    worst = (d, f"{i}:{k}")
                diffs[k] = {"before": x, "after": y, "abs_diff": d}
            else:
                diffs[k] = {"before": x, "after": y}
        report[i] = {"compared": True, "against": old["run_at"], "leaves": len(b),
                     "differing": len(diffs),
                     "max_abs_numeric_diff": max(numeric) if numeric else 0.0,
                     "method_version": [old.get("method_version", 1), new.get("method_version", 1)],
                     "data_hash_same": old.get("data_hash") == new.get("data_hash"),
                     "diffs": diffs}

    compared = [v for v in report.values() if v.get("compared")]
    out = {"ledger_rows": len(rows), "ids": len(by_id), "ids_compared": len(compared),
           "ids_with_any_difference": sum(1 for v in compared if v["differing"]),
           "largest_numeric_difference": worst[0], "where": worst[1], "by_id": report}
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"{len(compared)} ids compared against their newest pre-8b run; "
          f"{out['ids_with_any_difference']} differ anywhere; "
          f"largest numeric difference {worst[0]!r} ({worst[1]})")
    for i, v in sorted(report.items()):
        if v.get("compared") and v["differing"]:
            print(f"  {i}: {v['differing']} differing leaves, max {v['max_abs_numeric_diff']}")
        elif not v.get("compared"):
            print(f"  {i}: not compared - {v['why']}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
