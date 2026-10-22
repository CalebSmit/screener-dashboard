"""Did today's rerun change any published number? Compare each id's newest ledger row with the one
before it, at full precision.

The 2026-10-18 session's rule: a rerun for provenance must be *checked* not to have moved a number,
because the public research page quotes these figures and "it should be identical" is how a silent
change gets published. Today's edit added a stop-loss arm and new summary fields; it touched no
transform and no measurement method, so every experiment's statistics should be identical and the
expected answer is 0 differences.

    python research/measurements/2026-10-22-rerun-moved-nothing.py

Writes ``2026-10-22-rerun-moved-nothing.json`` beside this file. Exit 1 if anything moved.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LEDGER = ROOT / "research" / "experiments" / "ledger.jsonl"
OUT = Path(__file__).with_suffix(".json")

# fields that are *expected* to differ between two runs of the same thing
SKIP = {"run_at", "commit", "code_version", "code_hash", "code_hashes", "months"}


def flatten(obj, prefix: str = "") -> dict:
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not prefix and k in SKIP:
                continue
            out |= flatten(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out |= flatten(v, f"{prefix}[{i}]")
    else:
        out[prefix] = obj
    return out


def main() -> int:
    rows: dict[str, list[dict]] = {}
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows.setdefault(r["id"], []).append(r)

    out = {"ids": 0, "compared": 0, "no_previous_row": [], "differences": {},
           "largest_numeric_difference": 0.0, "new_fields": {}}
    for ident, hist in sorted(rows.items()):
        out["ids"] += 1
        if len(hist) < 2:
            out["no_previous_row"].append(ident)
            continue
        out["compared"] += 1
        new, old = flatten(hist[-1]), flatten(hist[-2])
        added = sorted(set(new) - set(old))
        if added:
            out["new_fields"][ident] = added
        for k in sorted(set(new) & set(old)):
            a, b = new[k], old[k]
            if a == b:
                continue
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
                    and not isinstance(a, bool) and not isinstance(b, bool):
                d = abs(float(a) - float(b))
                out["largest_numeric_difference"] = max(out["largest_numeric_difference"], d)
            out["differences"].setdefault(ident, {})[k] = [b, a]

    out["n_differences"] = sum(len(v) for v in out["differences"].values())
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"{out['compared']} of {out['ids']} ids compared with their previous run; "
          f"{out['n_differences']} field(s) differ, largest numeric difference "
          f"{out['largest_numeric_difference']}")
    if out["no_previous_row"]:
        print("no previous row (new this run):", ", ".join(out["no_previous_row"]))
    for ident, diffs in out["differences"].items():
        for k, (b, a) in list(diffs.items())[:6]:
            print(f"  {ident} {k}: {b} -> {a}")
    return 1 if out["n_differences"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
