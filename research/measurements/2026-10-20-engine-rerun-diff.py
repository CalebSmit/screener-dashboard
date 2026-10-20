"""Did the 2026-10-20 bank P/B change move any research-engine number?

The change touches `factor_engine.py` and `pit_record.py`, both in `v2_experiments.RESULT_PATHS`,
so every ledger row went stale on code and all 30 were rerun. The prediction, written before the
rerun (`METHODOLOGY_CHANGELOG.md` 2026-10-20, "Point-in-time record"): **nothing moves**, because
`pit_record.statement_fields` sets `commonEquity` to the filed stockholders' equity and
`market_fields` already built `priceToBook` as price x shares / equity - the same ratio.

This compares each experiment's newest ledger row with the one before it, at full precision, over
every numeric statistic either row carries. "It should be identical" is how a silent change gets
published (the 2026-10-18 rerun's lesson), so it is checked rather than assumed.

Usage:  python research/measurements/2026-10-20-engine-rerun-diff.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "research" / "experiments" / "ledger.jsonl"
OUT = Path(__file__).resolve().with_suffix(".json")


def flat(obj, prefix=""):
    """Every numeric leaf, keyed by its path."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out |= flat(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out |= flat(v, f"{prefix}[{i}]")
    elif isinstance(obj, bool) or obj is None:
        pass
    elif isinstance(obj, (int, float)):
        out[prefix] = float(obj)
    return out


rows = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
by_id = {}
for r in rows:
    by_id.setdefault(r.get("id") or r.get("experiment") or "?", []).append(r)

report = {"ledger_rows": len(rows), "ids": len(by_id), "compared": 0, "identical": 0,
          "differences": {}, "largest": 0.0, "no_previous_run": []}

# provenance moves by design: the code version, its hashes, and when the row was written
SKIP = ("code_version", "code_hash", "code_hashes", "ran_at", "run_at", "timestamp", "duration",
        "seconds", "elapsed")

for name, rs in sorted(by_id.items()):
    if len(rs) < 2:
        report["no_previous_run"].append(name)
        continue
    new, old = flat(rs[-1]), flat(rs[-2])
    keys = sorted(set(new) & set(old))
    keys = [k for k in keys if not any(s in k for s in SKIP)]
    diffs = {k: [old[k], new[k]] for k in keys
             if not (old[k] == new[k] or (old[k] != old[k] and new[k] != new[k]))}
    report["compared"] += 1
    if diffs:
        report["differences"][name] = diffs
        report["largest"] = max(report["largest"],
                                max(abs(b - a) for a, b in diffs.values()))
    else:
        report["identical"] += 1
    only_new = sorted(set(new) - set(old) - {k for k in set(new) if any(s in k for s in SKIP)})
    if only_new:
        report["differences"].setdefault(name, {})["_keys_only_in_the_new_row"] = only_new

OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k != "differences"}, indent=1))
print("differences:", json.dumps(report["differences"], indent=1)[:4000])
print("\nwrote", OUT)
