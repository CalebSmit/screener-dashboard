"""What first paint actually pays for `stock_detail`, field by field.

Phase 1's last bullet (`plan/universe-expansion.md`) is to shard `stock_detail` out of the
first-paint payload so that first paint does not grow with the universe. Before choosing a
shape for the shard, measure what is actually there: `stock_detail` is ~90% of
`dashboard_data.js` by the plan's own table, but a shard has to be drawn around *fields*,
because ~85 call sites in the generated JS read `D.stock_detail[t]` synchronously for an
arbitrary ticker (search, peers, holdings, movers, compare), and making every one of them
await a per-stock fetch is a different and much larger change.

Run:  python research/measurements/2026-11-08-stock-detail-field-weights.py

Reads the committed `dashboard_data.js` (read-only; writes nothing).
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def gz(payload: str) -> int:
    return len(gzip.compress(payload.encode("utf-8"), 6))


def load(path: Path) -> dict:
    txt = path.read_text(encoding="utf-8")
    m = re.match(r"^\s*window\.SCREENER_DATA\s*=\s*", txt)
    if not m:
        raise SystemExit(f"{path.name} does not open with the expected assignment")
    return json.loads(txt[m.end():].rstrip().rstrip(";"))


def main() -> int:
    path = ROOT / "dashboard_data.js"
    if not path.exists():
        print("MISSING DATA: dashboard_data.js")
        return 3
    data = load(path)
    sd = data.get("stock_detail") or {}
    full_raw = len(json.dumps(data))
    full_gz = gz(json.dumps(data))
    print(f"dashboard_data.js   {path.stat().st_size / 1e6:7.3f} MB on disk")
    print(f"payload             {full_raw / 1e6:7.3f} MB raw   {full_gz / 1e6:7.3f} MB gzipped")
    print(f"stock_detail        {len(sd)} stocks")

    without = {k: v for k, v in data.items() if k != "stock_detail"}
    wo_raw, wo_gz = len(json.dumps(without)), gz(json.dumps(without))
    print(f"without stock_detail{wo_raw / 1e6:7.3f} MB raw   {wo_gz / 1e6:7.3f} MB gzipped"
          f"   ({100 * (1 - wo_gz / full_gz):.1f}% of gzipped weight is stock_detail)")

    # Per-field cost, measured the way the shard would pay it: remove the field from every
    # stock and re-gzip the whole payload. Gzip is not additive, so each field is priced
    # against the same whole rather than on its own.
    keys: dict[str, int] = {}
    for s in sd.values():
        for k in s:
            keys[k] = keys.get(k, 0) + 1
    rows = []
    for k in sorted(keys):
        trimmed = dict(data)
        trimmed["stock_detail"] = {t: {a: b for a, b in s.items() if a != k} for t, s in sd.items()}
        rows.append((k, keys[k], full_gz - gz(json.dumps(trimmed))))
    rows.sort(key=lambda r: -r[2])
    print(f"\n{'field':<14}{'stocks':>8}{'gzipped cost':>16}{'share of payload':>18}")
    for k, n, cost in rows:
        print(f"{k:<14}{n:>8}{cost / 1e3:>13.1f} KB{100 * cost / full_gz:>17.1f}%")

    top = [r for r in rows if r[2] >= 20_000]
    print(f"\nfields costing >=20 KB gzipped: {len(top)} of {len(rows)}, "
          f"{sum(r[2] for r in top) / 1e3:.1f} KB = {100 * sum(r[2] for r in top) / full_gz:.1f}% "
          f"of the whole first-paint payload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
