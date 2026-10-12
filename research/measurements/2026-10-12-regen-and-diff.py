"""Regenerate the dashboard from the newest run into a scratch dir and diff it against what is live.

Used 2026-10-12 to check that repairing the option-quote session date changed the option fields
and nothing else. Leaves the committed artifacts alone - it only reports.

    python research/measurements/2026-10-12-regen-and-diff.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SCRATCH = Path("logs") / "regen-2026-10-12"


def load_js(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))


def leaves(obj, prefix="", out=None):
    out = {} if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            leaves(v, f"{prefix}.{k}", out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            leaves(v, f"{prefix}[{i}]", out)
    else:
        out[prefix] = obj
    return out


def main() -> int:
    import generate_dashboard as gd

    run = Path(max(glob.glob("runs/*/00_raw_fetch.parquet"), key=os.path.getmtime)).parent
    SCRATCH.mkdir(parents=True, exist_ok=True)
    print(f"Regenerating from {run} into {SCRATCH}\n")
    gd.generate_dashboard(run, SCRATCH / "dashboard.html")

    for name in ("dashboard_data.js", "dashboard_context.js"):
        old, new = load_js(Path(name)), load_js(SCRATCH / name)
        lo, ln = leaves(old), leaves(new)
        changed = [k for k in lo if k in ln and lo[k] != ln[k]]
        added = [k for k in ln if k not in lo]
        removed = [k for k in lo if k not in ln]
        print(f"\n== {name}: {len(lo)} leaves live, {len(ln)} regenerated")
        print(f"   changed {len(changed)}, added {len(added)}, removed {len(removed)}")

        def shape(keys):
            # group by the last path element, which is the payload field
            return Counter(k.rsplit(".", 1)[-1].split("[")[0] for k in keys).most_common(12)

        for label, keys in (("changed", changed), ("added", added), ("removed", removed)):
            if keys:
                print(f"   {label} by field: {shape(keys)}")
        for k in changed[:5]:
            print(f"     e.g. {k}: {lo[k]!r} -> {ln[k]!r}")

    newctx = load_js(SCRATCH / "dashboard_context.js")
    qd = Counter(v.get("oqd") for v in (newctx.get("ctx") or {}).values())
    st = Counter(v.get("os") for v in (newctx.get("ctx") or {}).values())
    print(f"\nregenerated quote dates: {qd.most_common()}")
    print(f"regenerated statuses:    {st.most_common()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
