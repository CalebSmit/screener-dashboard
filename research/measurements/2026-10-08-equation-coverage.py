"""How much of the composite sits on metrics the page cannot show the arithmetic for?

Re-measures the gap CLAUDE.md priority 0.10 describes, from the live payload and the
engine's own weight tables. Run from the repo root:

    python research/measurements/2026-10-08-equation-coverage.py
"""
import json
import re
import sys

import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")
import factor_engine as fe  # noqa: E402
import metric_lineage as ml  # noqa: E402

with open("config.yaml", encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)

cat_w = {k: v / 100.0 for k, v in cfg["factor_weights"].items()}
print("category weights:", cat_w)

# Per-category metric weights, generic profile.
rows = []
for cat in cat_w:
    try:
        prof = fe.metric_weight_profiles(cfg, cat)
    except Exception as exc:  # pragma: no cover - diagnostic
        print(f"  !! {cat}: {type(exc).__name__}: {exc}")
        continue
    rows.append((cat, prof))

total_by_metric = {}
for cat, prof in rows:
    generic = prof.get("generic", prof) if isinstance(prof, dict) else prof
    if not isinstance(generic, dict):
        print(f"  !! {cat}: unexpected shape {type(generic)}")
        continue
    for m, w in generic.items():
        if not isinstance(w, (int, float)) or w <= 0:
            continue
        total_by_metric[m] = total_by_metric.get(m, 0.0) + w * cat_w.get(cat, 0.0)

print(f"\n{len(total_by_metric)} weighted metrics, composite weight sums to "
      f"{sum(total_by_metric.values()):.4f}")

has_eq = {m for m in total_by_metric if m in ml.EQUATIONS}
exact = {m for m in has_eq if ml.EQUATIONS[m][0]}
no_eq = {m for m in total_by_metric if m not in ml.EQUATIONS}

def share(ms):
    return sum(total_by_metric[m] for m in ms)

print(f"  exact equation      : {len(exact):2d} metrics, {share(exact)*100:5.2f}% of composite")
print(f"  inexact equation    : {len(has_eq - exact):2d} metrics, {share(has_eq - exact)*100:5.2f}%")
print(f"  NO equation (SOURCES only): {len(no_eq):2d} metrics, {share(no_eq)*100:5.2f}%")
for m in sorted(no_eq, key=lambda x: -total_by_metric[x]):
    print(f"      {m:26s} {total_by_metric[m]*100:5.2f}%   src={ml.SOURCES.get(m, '(none)')!r}")

# Coverage: for how many stocks is each no-equation metric actually scored?
with open("dashboard_data.js", encoding="utf-8") as fh:
    txt = fh.read()
payload = json.loads(re.sub(r"^[^=]*=\s*", "", txt.strip().rstrip(";"), count=1))
detail = payload["stock_detail"]
print(f"\nlive payload: {len(detail)} stocks")
for m in sorted(no_eq, key=lambda x: -total_by_metric[x]):
    n = sum(1 for d in detail.values() if d.get("raw", {}).get(m) is not None)
    print(f"      {m:26s} scored for {n:3d} stocks")
