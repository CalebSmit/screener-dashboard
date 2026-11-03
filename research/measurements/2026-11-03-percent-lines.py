"""Every line carrying a percentage, on every stock's drilldown, from the generator's own page.

Used to size the signed-zero defect and to verify the fix changed nothing else: run it before the
change and after, and diff the two JSON files. A line only ever moves where a percentage that is
not zero was printing as a zero.

    python research/measurements/2026-11-03-percent-lines.py before.json
    python research/measurements/2026-11-03-percent-lines.py after.json
    python research/measurements/2026-11-03-percent-lines.py --diff before.json after.json
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

SWEEP = r"""() => {
  const root = () => { let e = document.getElementById('modal-ticker');
    while (e && e.parentElement && e.innerText.length < 5000) e = e.parentElement; return e; };
  const out = {};
  for (const r of D.table_data) {
    openStockDetail(r.Ticker);
    out[r.Ticker] = root().innerText.split('\n').filter(l => l.includes('%')).map(l => l.trim());
  }
  closeModal();
  return out;
}"""


def collect() -> dict:
    import generate_dashboard as gd
    from playwright import sync_api

    d = Path(tempfile.mkdtemp(prefix="pctlines"))
    (d / "index.html").write_text(
        gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t", data_version="v"),
        encoding="utf-8")
    for f in ("dashboard_data.js", "dashboard_context.js"):
        shutil.copy(ROOT / f, d / f)
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1440, "height": 900})
        errs: list[str] = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto((d / "index.html").as_uri(), wait_until="load", timeout=60000)
        pg.wait_for_function(
            "typeof openStockDetail === 'function' && typeof D !== 'undefined' && "
            "Object.values(D.stock_detail).some(s => s.ctx)", timeout=60000)
        out = pg.evaluate(SWEEP)
        b.close()
    if errs:
        raise SystemExit(f"page errors: {errs[:3]}")
    shutil.rmtree(d, ignore_errors=True)
    return out


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--diff":
        a = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
        b = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        assert set(a) == set(b), (len(a), len(b))
        moved = 0
        for t in a:
            for x, y in zip(a[t], b[t]):
                if x != y:
                    moved += 1
                    print(f"{t}: {x!r} -> {y!r}")
            if len(a[t]) != len(b[t]):
                print(f"{t}: LINE COUNT {len(a[t])} -> {len(b[t])}")
        print(f"\n{len(a)} stocks, {sum(len(v) for v in a.values())} percentage lines, {moved} moved")
        return 0
    out = collect()
    Path(argv[0]).write_text(json.dumps(out, indent=0, ensure_ascii=False), encoding="utf-8")
    print(f"{len(out)} stocks, {sum(len(v) for v in out.values())} percentage lines -> {argv[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
