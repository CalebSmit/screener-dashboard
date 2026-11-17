"""Look at the year-to-date note on a real stock's sheet before the 02:00 run publishes it.

The committed payload was built before the note existed, so no published stock carries it. This
builds a throwaway copy of the page in a temp folder - the generator's own HTML, the published
payload with `generate_dashboard._attach_ytd_column_note` run over it from the SEC cache, nothing
else changed - opens each stock the note names, expands its free-cash-flow-yield and accruals
rows and its Data Provenance panel, and writes screenshots at 1440 and 375 px to the gitignored
logs/design/. It also prints the rendered text, and whether anything overflows its box.

    python research/measurements/2026-11-17-ytd-note-on-the-page.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import generate_dashboard as g  # noqa: E402

OUT = ROOT / "logs" / "design"

if __name__ == "__main__":
    from playwright.sync_api import sync_playwright
    OUT.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="ytdnote"))
    text = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    head, body, tail = text[: text.index("{")], text[text.index("{"): text.rindex("}") + 1], text[text.rindex("}") + 1:]
    payload = json.loads(body)
    n = g._attach_ytd_column_note(payload["stock_detail"])
    named = [t for t, s in payload["stock_detail"].items() if s.get("srcn")]
    print(f"note attached to {n} stocks: {named}")
    (d / "dashboard_data.js").write_text(head + json.dumps(payload) + tail, encoding="utf-8")
    for f in ("dashboard_context.js", "dashboard_detail.js"):
        shutil.copy(ROOT / f, d / f)
    (d / "index.html").write_text(g.generate_html(), encoding="utf-8")
    with sync_playwright() as p:
        b = p.chromium.launch()
        for label, vp in (("desktop", {"width": 1440, "height": 900}), ("mobile", {"width": 375, "height": 800})):
            page = b.new_page(viewport=vp)
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto((d / "index.html").as_uri(), wait_until="load", timeout=60000)
            page.wait_for_function("typeof openStockDetail === 'function' && typeof D !== 'undefined'", timeout=60000)
            for t in named:
                page.evaluate("t => openStockDetail(t)", t)
                page.wait_for_timeout(1500)
                for m in ("fcf_yield", "accruals"):
                    page.evaluate("""m => { const r = document.querySelector('#modal-categories tr.metric-row[data-metric="' + m + '"]');
                        if (!r) return;
                        r.closest('.cat-detail-section').classList.remove('collapsed');
                        if (!r.classList.contains('wk-open')) r.click(); }""", m)
                    page.wait_for_timeout(400)
                    info = page.evaluate("""() => { const e = [...document.querySelectorAll('.wk-caveat-source')].find(x => x.getBoundingClientRect().height > 0);
                        if (!e) return null; e.scrollIntoView({block: 'center'});
                        const r = e.getBoundingClientRect();
                        return {text: e.textContent, w: Math.round(r.width), h: Math.round(r.height), left: Math.round(r.left),
                                right: Math.round(r.right), vw: window.innerWidth, clipped: e.scrollWidth > e.clientWidth,
                                doc_wider: document.documentElement.scrollWidth > window.innerWidth}; }""")
                    print(f"\n[{label}] {t} {m}: {info}")
                    page.wait_for_timeout(300)
                    page.screenshot(path=str(OUT / f"ytdnote-{label}-{t}-{m}.png"))
                    page.evaluate("""m => { const r = document.querySelector('#modal-categories tr.metric-row[data-metric="' + m + '"]');
                        if (r && r.classList.contains('wk-open')) r.click(); }""", m)
                prov = page.evaluate("""() => { const e = document.getElementById('provenance-ytd-column');
                    if (!e) return null; e.scrollIntoView({block: 'center'});
                    return {text: e.textContent, panel: document.getElementById('modal-provenance').textContent.replace(/\\s+/g, ' ').slice(-900)}; }""")
                print(f"\n[{label}] {t} provenance: {prov}")
                page.wait_for_timeout(300)
                page.screenshot(path=str(OUT / f"ytdnote-{label}-{t}-provenance.png"))
            print(f"[{label}] page errors: {errors}")
            page.close()
        b.close()
    shutil.rmtree(d, ignore_errors=True)
