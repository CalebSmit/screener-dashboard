"""Look at the reworded staleness sentence and the "Statement age" badge on real stocks.

The committed payload was built before the rewording (the summary sentences are baked at run
time), so this regenerates every published file from the newest full run with the working
tree's code - `scripts/rehearse_publish.regenerate`, into a temp folder, nothing installed -
counts each wording across the universe, then opens two stocks at 1440 and 375 px: one read from
its annual statement and one flagged for another reason. Screenshots go to the gitignored
logs/design/.

    python research/measurements/2026-11-18-stale-sentence-on-the-page.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import rehearse_publish as rp  # noqa: E402

OUT = ROOT / "logs" / "design"

if __name__ == "__main__":
    from playwright.sync_api import sync_playwright
    OUT.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="stale-scratch"))
    d = Path(tempfile.mkdtemp(prefix="stale-site"))
    run = rp.newest_full_run(ROOT / "runs")
    files = rp.regenerate(run, scratch)
    for name, data in files.items():
        (d / name).write_bytes(data)
    text = files["dashboard_data.js"].decode("utf-8")
    payload = json.loads(text[text.index("{"): text.rindex("}") + 1])
    sd = payload["stock_detail"]

    kinds: Counter = Counter()
    example: dict[str, str] = {}
    for t, s in sd.items():
        if not (s.get("flags") or {}).get("stale_data"):
            continue
        conf = next((f["t"] for f in s.get("summary") or [] if f["k"] == "confidence"), "")
        kind = ("OLD WORDING" if "filings are flagged stale" in conf
                else "annual" if "are the last fiscal year's" in conf
                else "generic" if "The income statement this score read is flagged stale" in conf
                else "NOTHING SAID")
        kinds[kind] += 1
        example.setdefault(kind, t)
        if kind in ("annual", "generic") and t in ("AMZN", "DD"):
            example[kind] = t
    print(f"run {run.name}: flagged stale {sum(kinds.values())}; wording: {dict(kinds)}")
    for kind, t in example.items():
        conf = next(f["t"] for f in sd[t]["summary"] if f["k"] == "confidence")
        print(f"  {kind:8} {t}: {conf}")

    with sync_playwright() as p:
        b = p.chromium.launch()
        for label, vp in (("desktop", {"width": 1440, "height": 900}),
                          ("mobile", {"width": 375, "height": 800})):
            page = b.new_page(viewport=vp)
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto((d / "index.html").as_uri(), wait_until="load", timeout=60000)
            page.wait_for_function(
                "typeof openStockDetail === 'function' && typeof D !== 'undefined'", timeout=60000)
            for t in sorted(set(example.values())):
                page.evaluate("t => openStockDetail(t)", t)
                page.wait_for_timeout(1500)
                page.evaluate("() => document.getElementById('section-provenance').classList.remove('collapsed')")
                for sel, name in ((".summary-confidence", "summary"),
                                  ("#modal-provenance .provenance-badges", "badges")):
                    info = page.evaluate("""sel => { const e = [...document.querySelectorAll(sel)]
                            .find(x => x.getBoundingClientRect().height > 0);
                        if (!e) return null; e.scrollIntoView({block: 'center'});
                        const r = e.getBoundingClientRect();
                        return {text: e.textContent.replace(/\\s+/g, ' ').trim(), w: Math.round(r.width),
                                h: Math.round(r.height), left: Math.round(r.left), right: Math.round(r.right),
                                vw: window.innerWidth, clipped: e.scrollWidth > e.clientWidth,
                                doc_wider: document.documentElement.scrollWidth > window.innerWidth}; }""", sel)
                    print(f"\n[{label}] {t} {name}: {info}")
                    page.wait_for_timeout(300)
                    page.screenshot(path=str(OUT / f"stale-{label}-{t}-{name}.png"))
            print(f"[{label}] page errors: {errors}")
            page.close()
        b.close()
    shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(scratch, ignore_errors=True)
