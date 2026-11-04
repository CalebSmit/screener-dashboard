"""Mutation check for the expected-move divisor fix (2026-11-03, owner-run evening).

Each mutation reverts one part of the fix in a scratch copy of a source file; the checks
registered on claim ``context.expected_move`` must fail on every one. Run from the repo root:

    python research/measurements/2026-11-03-expected-move-divisor-mutations.py

Restores every file it touches, whatever happens.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ["tests/test_context_layer_claims.py", "tests/test_option_quote_hours.py",
         "tests/test_context_layer.py", "-k", "divisor or straddle or option_paths or falls_back "
         "or record_counts or with_its_divisor or real_session or harness"]

MUTATIONS = [
    ("refresh divides by the handed-in close", "options_cache.py",
     "spot, src = chain_spot(getattr(ch, \"underlying\", None), price)",
     "spot, src = chain_spot(None, price)"),
    ("live pass divides by the run close", "context_fetch.py",
     "spot, src = chain_spot(getattr(ch, \"underlying\", None), price)",
     "spot, src = float(price), \"chain\""),
    ("divisor not published", "context_signals.py",
     "out[\"_ctx_opt_spot\"] = float(spot)", "pass"),
    ("page names this run's close again", "generate_dashboard.py",
     "divided by ' + cPrice(c.osp) + ', the stock", "divided by ' + cPrice(c.px) + ', the stock"),
    ("generator publishes a move without its divisor", "generate_dashboard.py",
     "if ({\"om\", \"oi\", \"osk\"} & out.keys()) and not", "if False and not"),
    ("generator keeps an unanchored volatility", "generate_dashboard.py",
     "anchored_keys = (\"om\", \"ost\", \"ok\", \"osp\", \"oi\",", "anchored_keys = (\"om\", \"ost\", \"ok\", \"osp\","),
    ("the page shows an unanchored volatility", "generate_dashboard.py",
     "if (an && c.oi !== undefined) o += '<tr><td>Implied volatility (at the money)",
     "if (c.oi !== undefined) o += '<tr><td>Implied volatility (at the money)"),
    ("the record counts unanchored volatilities", "context_eval.py",
     "if key in (\"_ctx_opt_move\", \"_ctx_opt_iv\", \"_ctx_opt_skew\"):", "if key in (\"_ctx_opt_move\", \"_ctx_opt_skew\"):"),
    ("record counts pre-fix moves", "context_eval.py",
     "            v = v.where(move_has_its_divisor(log))", "            pass"),
    ("withheld card prints a figure", "generate_dashboard.py",
     "const mv = moveShown(c), an = optAnchored(c);", "const mv = c.om !== undefined, an = optAnchored(c);"),
    ("the page's one predicate ignores the divisor", "generate_dashboard.py",
     "function optAnchored(c) { return !!c && c.osp !== undefined; }",
     "function optAnchored(c) { return !!c; }"),
    ("the summary chip bypasses the predicate", "generate_dashboard.py",
     "if (moveShown(c) && c.os === 'ok') bits.push(", "if (c.om !== undefined && c.os === 'ok') bits.push("),
]


def main() -> int:
    caught = 0
    for name, rel, old, new in MUTATIONS:
        p = ROOT / rel
        orig = p.read_bytes()
        text = orig.decode("utf-8")
        assert text.count(old) == 1, f"{name}: anchor not unique in {rel}"
        try:
            p.write_bytes(text.replace(old, new).encode("utf-8"))
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS],
                               cwd=ROOT, capture_output=True, text=True)
        finally:
            p.write_bytes(orig)
        hit = r.returncode != 0
        caught += hit
        print(f"{'CAUGHT' if hit else 'MISSED'}  {name}")
    print(f"{caught} of {len(MUTATIONS)} caught")
    return 0 if caught == len(MUTATIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
