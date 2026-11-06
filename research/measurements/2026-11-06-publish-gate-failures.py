"""Why the 2026-11-06 02:00 data run was discarded: the two publish-gate failures, measured.

The run passed every health check and then failed `scripts/check_published_claims.py` on

    tests/test_metric_lineage.py::test_the_equation_on_the_row_gives_the_value_that_was_scored[price_target_upside]
    tests/test_dashboard_printed_figures.py::test_the_holdings_strip_prints_the_saved_names_ranks_of_the_number_ranked

while the same two tests pass against the committed (2026-11-05) payload. Run it on both
payloads and the difference is the day's data, not the code.

    python research/measurements/2026-11-06-publish-gate-failures.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import metric_lineage as ml  # noqa: E402

PAYLOADS = [
    ("2026-11-06 (discarded)", ROOT / "runs" / "e1174e2f4363" / "dashboard_data.js"),
    ("committed (2026-11-05)", ROOT / "dashboard_data.js"),
]


def load(path):
    txt = path.read_text(encoding="utf-8")
    body = txt[txt.index("=") + 1:].strip()
    if body.endswith(";"):
        body = body[:-1]
    return json.loads(body)


def _tol(v):
    """The tolerance test_metric_lineage uses."""
    return max(abs(v) * 0.005, 1e-6)


def equation_agreement(payload, metric="price_target_upside"):
    ok = total = 0
    bad = []
    for ticker, s in payload["stock_detail"].items():
        pub = s["raw"].get(metric)
        if pub is None:
            continue
        tpl = ml.choose_template(metric, s.get("inp") or {})
        if tpl is None:
            continue
        total += 1
        v = ml.evaluate_template(tpl, s["inp"])
        if v is not None and abs(v - pub) <= _tol(v):
            ok += 1
        else:
            bad.append((ticker, pub, v))
    return ok, total, bad


def holdings_premise(payload):
    rows = sorted((r for r in payload["table_data"] if r.get("Rank")), key=lambda r: r["Rank"])
    n = payload["kpis"]["universe_size"]
    return n, len(rows), rows[-1]["Rank"], rows[0]["Rank"]


def main():
    clamps = {}
    try:
        import yaml
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        clamps = (cfg.get("metric_clamps") or {})
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"(could not read clamps from config.yaml: {exc})")
    lo, hi = (clamps.get("price_target_upside") or [-0.50, 1.0])[:2]
    print(f"price_target_upside clamp from config.yaml: [{lo}, {hi}]\n")

    for label, path in PAYLOADS:
        if not path.exists():
            print(f"{label}: MISSING DATA: {path}")
            continue
        payload = load(path)
        ok, total, bad = equation_agreement(payload)
        at_clamp = [b for b in bad if b[1] in (lo, hi)]
        print(f"{label}")
        print(f"  equation on the row reproduces the scored value: {ok}/{total} "
              f"= {ok / total:.4f}   (test bar 0.99)")
        print(f"  failures: {len(bad)}, of which sitting exactly on a clamp: {len(at_clamp)}")
        for ticker, pub, v in sorted(bad, key=lambda r: -abs((r[2] or 0) - r[1]))[:6]:
            print(f"    {ticker:6s} scored {pub!r:>8}  equation gives {v!r}")
        n, nrows, last, first = holdings_premise(payload)
        print(f"  holdings premise: kpis.universe_size={n}, ranked rows={nrows}, "
              f"last rank={last}, first rank={first}  "
              f"-> {'HOLDS' if n == nrows == last and first == 1 else 'FAILS'}")
        print()


if __name__ == "__main__":
    main()
