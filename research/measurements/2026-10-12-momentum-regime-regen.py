"""Does carrying the momentum rule's market reading change anything else on the page?

Copies the newest full run into a scratch folder, adds the reading the rebuilt rule takes
(``factor_engine.market_panic_state``, live, one request) to that copy's
``effective_weights.json`` - which is what a run made with the rule on writes - regenerates the
dashboard from it, and diffs both payload files against what is committed. Expected: one added
block, ``weights.momentum_regime``, and nothing else. The committed artifacts and the real run
folder are only read.

    python research/measurements/2026-10-12-momentum-regime-regen.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SCRATCH = ROOT / "logs" / "regen-momentum-regime"


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


def newest_full_run() -> Path:
    runs = []
    for d in (ROOT / "runs").iterdir():
        scored = d / "05_final_scored.parquet"
        if scored.exists() and (d / "effective_weights.json").exists() and len(pd.read_parquet(scored)) >= 495:
            runs.append(d)
    return max(runs, key=lambda d: (d / "05_final_scored.parquet").stat().st_mtime)


def main() -> int:
    import factor_engine as fe
    import generate_dashboard as gd

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    reading = fe.market_panic_state(fe.fetch_market_closes(), cfg["momentum_regime"])
    print("reading:", reading)

    src = newest_full_run()
    run = SCRATCH / "run"
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH, ignore_errors=True)
    shutil.copytree(src, run)
    w = json.loads((run / "effective_weights.json").read_text())
    w["momentum_regime"] = reading
    (run / "effective_weights.json").write_text(json.dumps(w, indent=2))
    print(f"regenerating from a copy of {src.name} into {SCRATCH / 'site'}")
    (SCRATCH / "site").mkdir(parents=True, exist_ok=True)
    gd.generate_dashboard(run, SCRATCH / "site" / "dashboard.html")

    for name in ("dashboard_data.js", "dashboard_context.js"):
        lo, ln = leaves(load_js(ROOT / name)), leaves(load_js(SCRATCH / "site" / name))
        changed = [k for k in lo if k in ln and lo[k] != ln[k]]
        added = [k for k in ln if k not in lo]
        removed = [k for k in lo if k not in ln]
        print(f"\n== {name}: {len(lo)} leaves committed, {len(ln)} regenerated; "
              f"changed {len(changed)}, added {len(added)}, removed {len(removed)}")
        for label, keys in (("changed", changed), ("added", added), ("removed", removed)):
            for k in keys[:12]:
                print(f"   {label}: {k}" + (f" = {ln[k]!r}" if label == "added" else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
