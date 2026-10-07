#!/usr/bin/env python3
"""Diff two ``dashboard_data.js`` payloads and bucket every difference.

Why this exists
---------------
``plan/calculation-transparency.md`` verification step 2 requires, at every
stage, a payload diff "classifying keys as unchanged / added / changed", because
the governing constraint of both open owner items is that **presentation and
explanation work must leave every number, rank and sentence byte-identical**.
Asserting that by eye over a 5 MB payload is not possible, and "the tests pass"
does not show it - no test enumerates all 502 x ~30 keys.

The plan assigns this script to stage T0b. It was written during **T0a** instead
because T0a changes a *sentence* that is baked into all 502 ``stock_detail``
entries, and the only honest way to show that nothing else moved with it is to
walk both payloads. Every later stage needs it too.

What it reports
---------------
Leaf-level buckets, by JSON path:

* ``added`` / ``removed`` - keys present in only one payload
* ``changed`` - keys present in both with different values
* ``unchanged`` - counted, not listed

Paths are generalised for the summary: ``stock_detail.AAPL.composite`` is
reported individually, but when the same leaf differs for many tickers the
``--group`` view collapses them to ``stock_detail.*.composite`` with a count, so
"the rank sentence changed for 502 stocks and nothing else did" is one line
rather than 502.

Numeric leaves compare with a tolerance (default 0.0 - exact), so a stage that
is *meant* to be byte-identical can assert exactly that, while a stage that
legitimately re-derives a float can allow rounding.

Usage
-----
    python scripts/diff_payload.py BEFORE.js AFTER.js
    python scripts/diff_payload.py BEFORE.js AFTER.js --group
    python scripts/diff_payload.py BEFORE.js AFTER.js --tolerance 0.01
    python scripts/diff_payload.py BEFORE.js AFTER.js --only stock_detail
    python scripts/diff_payload.py BEFORE.js AFTER.js --max-show 40

Exit codes:
    0  no differences (within tolerance)
    1  differences found
    2  a payload could not be read or parsed
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

#: ``window.SCREENER_DATA = {...};`` - the payload is JSON inside an assignment.
_ASSIGN = re.compile(r"^\s*(?:window\.)?[A-Za-z_$][\w$.]*\s*=\s*", re.MULTILINE)


def load_payload(path: Path) -> dict:
    """Parse the JSON object out of a ``dashboard_data.js`` file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"{path}: no JSON object found")
    return json.loads(text[start:end + 1])


def _is_num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def walk(before, after, tolerance: float, path: str = ""):
    """Yield ``(bucket, path, before, after)`` for every differing leaf."""
    if isinstance(before, dict) and isinstance(after, dict):
        for key in before.keys() | after.keys():
            sub = f"{path}.{key}" if path else str(key)
            if key not in after:
                yield ("removed", sub, before[key], None)
            elif key not in before:
                yield ("added", sub, None, after[key])
            else:
                yield from walk(before[key], after[key], tolerance, sub)
        return

    if isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            yield ("changed", f"{path}[len]", len(before), len(after))
        for i in range(min(len(before), len(after))):
            yield from walk(before[i], after[i], tolerance, f"{path}[{i}]")
        return

    if _is_num(before) and _is_num(after):
        if abs(float(before) - float(after)) > tolerance:
            yield ("changed", path, before, after)
        return

    if before != after:
        yield ("changed", path, before, after)


def _generalise(path: str) -> str:
    """``stock_detail.AAPL.summary[0].t`` -> ``stock_detail.*.summary[*].t``.

    Collapses the ticker level under the known per-ticker maps and every list
    index, so a difference affecting many stocks reads as one row.
    """
    path = re.sub(r"\[\d+\]", "[*]", path)
    for container in ("stock_detail", "table_data", "history", "compare"):
        path = re.sub(rf"^{container}\.[^.\[]+", f"{container}.*", path)
    return path


def count_leaves(obj) -> int:
    if isinstance(obj, dict):
        return sum(count_leaves(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(count_leaves(v) for v in obj)
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="max absolute numeric difference treated as equal (default 0.0)")
    ap.add_argument("--group", action="store_true",
                    help="collapse per-ticker paths to one row with a count")
    ap.add_argument("--only", default=None,
                    help="restrict to paths starting with this prefix")
    ap.add_argument("--max-show", type=int, default=25,
                    help="how many individual differences to print (default 25)")
    args = ap.parse_args(argv)

    try:
        before = load_payload(args.before)
        after = load_payload(args.after)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2

    diffs = list(walk(before, after, args.tolerance))
    if args.only:
        diffs = [d for d in diffs if d[1].startswith(args.only)]

    total_leaves = count_leaves(after)
    buckets = Counter(b for b, _, _, _ in diffs)

    print(f"before: {args.before}  ({args.before.stat().st_size:,} bytes)")
    print(f"after:  {args.after}  ({args.after.stat().st_size:,} bytes)")
    print(f"tolerance: {args.tolerance}")
    print()
    print(f"leaves in after:  {total_leaves:,}")
    print(f"unchanged:        {total_leaves - buckets['changed'] - buckets['added']:,}")
    print(f"changed:          {buckets['changed']:,}")
    print(f"added:            {buckets['added']:,}")
    print(f"removed:          {buckets['removed']:,}")

    if not diffs:
        print("\nIdentical.")
        return 0

    if args.group:
        print("\nBy generalised path:")
        grouped = Counter((b, _generalise(p)) for b, p, _, _ in diffs)
        for (bucket, path), n in sorted(grouped.items(), key=lambda kv: -kv[1]):
            print(f"  {bucket:9s} {n:6,d}  {path}")

    shown = diffs[:args.max_show]
    print(f"\nFirst {len(shown)} of {len(diffs)} differences:")
    for bucket, path, b, a in shown:
        if bucket == "changed":
            print(f"  {bucket:9s} {path}")
            print(f"      - {str(b)[:300]}")
            print(f"      + {str(a)[:300]}")
        else:
            value = a if bucket == "added" else b
            print(f"  {bucket:9s} {path}  {str(value)[:200]}")

    return 1


if __name__ == "__main__":
    sys.exit(main())
