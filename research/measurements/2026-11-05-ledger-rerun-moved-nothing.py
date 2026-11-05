"""Did bringing the ledger level with the live code move any number?

The 32 rows went stale on 2026-11-03, when `universe_history.py` was edited (commit f02cf1b,
"Universe phase 1, first slice") - a file in `v2_experiments.RESULT_PATHS` that v2 reads when it
*builds* the monthly records, not when it ranks them. The records themselves were untouched
(`content_hash` identical), so the prediction before the rerun was: every statistic identical,
only the provenance moves.

This compares each id's newest row with the row before it, ignoring the fields a rerun is expected
to change, and reports every difference it finds.

Run: ``python research/measurements/2026-11-05-ledger-rerun-moved-nothing.py``
Writes ``research/measurements/2026-11-05-ledger-rerun-moved-nothing.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import v2_experiments as X  # noqa: E402

OUT = Path(__file__).with_suffix(".json")

#: What a rerun is allowed to change: when it ran, what it ran on, and the identifiers of that.
PROVENANCE = {"run_at", "commit", "code_version", "code_hash", "code_hashes",
              "content_hash", "content_parts", "data_fingerprint", "arm_sha256"}


def _flatten(obj, prefix: str = "") -> dict:
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out |= _flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out |= _flatten(v, f"{prefix}[{i}]")
    else:
        out[prefix] = obj
    return out


def main() -> dict:
    rows: dict[str, list[dict]] = {}
    for line in X.LEDGER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows.setdefault(r["id"], []).append(r)

    live = X.code_fingerprint()
    compared, stats, diffs = 0, 0, {}
    for i, series in rows.items():
        if len(series) < 2:
            continue
        new, old = series[-1], series[-2]
        if new.get("code_hash") != live["code_hash"]:
            continue                           # not a row from this rerun
        compared += 1
        a = {k: v for k, v in _flatten(new).items()
             if k.split(".")[0] not in PROVENANCE}
        b = {k: v for k, v in _flatten(old).items()
             if k.split(".")[0] not in PROVENANCE}
        stats += len(a)
        moved = {k: [b.get(k), a[k]] for k in a if k in b and a[k] != b[k]}
        gained = sorted(set(a) - set(b))
        lost = sorted(set(b) - set(a))
        if moved or gained or lost:
            diffs[i] = {"moved": moved, "only_in_new": gained, "only_in_old": lost}

    result = {
        "ids_compared": compared,
        "statistics_compared": stats,
        "ids_with_a_difference": len(diffs),
        "differences": diffs,
        "live_code_hash": live["code_hash"],
        "which_file_had_moved": "universe_history.py",
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "differences"},
                     indent=2, sort_keys=True))
    if diffs:
        print("\nDIFFERENCES:")
        print(json.dumps(diffs, indent=2, sort_keys=True)[:4000])
    return result


if __name__ == "__main__":
    main()
