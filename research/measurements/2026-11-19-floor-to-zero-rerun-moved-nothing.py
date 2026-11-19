"""Did the 2026-11-19 rerun - forced by setting ``universe.min_market_cap`` to 0 and wiring the
floor so it can fire - move any research-engine number?

Three ``v2_experiments.RESULT_PATHS`` files changed that morning (``config.yaml``,
``factor_engine.py``, ``universe.py``), so all 32 ledger rows went stale and were rerun.

The prediction, written before the comparison was run and the whole reason the change was called
a no-op: **every statistic identical, only the provenance moves.** The old ``2e9`` never excluded
a row (``2026-11-19-market-cap-floor-never-fires.py``: the filter looked for a ``marketCap``
column the metrics frame does not carry), and a floor of 0 asks for nothing - so
``backtest_v2.rank_month`` ranks exactly the rows it ranked before.

Had the prediction failed, the claim "no-op by construction" in ``METHODOLOGY_CHANGELOG.md``
2026-11-19 would be false and the entry would need rewriting, not the comparison.

Same comparison as 2026-11-05 and 2026-11-10, reused rather than restated: each id's newest row
against the row before it, ignoring the fields a rerun is expected to change.

Run: ``python research/measurements/2026-11-19-floor-to-zero-rerun-moved-nothing.py``
Writes ``research/measurements/2026-11-19-floor-to-zero-rerun-moved-nothing.json``.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = HERE / "2026-11-05-ledger-rerun-moved-nothing.py"
OUT = Path(__file__).with_suffix(".json")


def _prior_module():
    spec = importlib.util.spec_from_file_location("ledger_rerun_compare", PRIOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> dict:
    m = _prior_module()
    m.OUT = OUT                      # write this date's file, never overwrite an earlier one
    with contextlib.redirect_stdout(io.StringIO()) as quiet:
        result = m.main()
    # Unlike the two earlier reruns, this one *did* change the config (min_market_cap 2e9 -> 0),
    # so every row's config hashes are expected to move and the prior script counts them as
    # differences. They are provenance: what matters is whether anything else moved.
    expected = {"base_config_hash", "config_hash"}
    other = {}
    for rid, diff in (result.get("differences") or {}).items():
        moved = set(diff.get("moved", {})) - expected
        if moved or diff.get("only_in_new") or diff.get("only_in_old"):
            other[rid] = sorted(moved) + diff.get("only_in_new", []) + diff.get("only_in_old", [])
    result["ids_whose_only_difference_is_the_config_hash"] = len(result.get("differences") or {}) - len(other)
    result["ids_with_any_other_difference"] = len(other)
    result["other_differences"] = other
    result.pop("which_file_had_moved", None)      # the 2026-11-05 run's own answer, not this one's
    if other:
        print(quiet.getvalue())
    result["which_files_had_moved"] = "config.yaml, factor_engine.py, universe.py (code and config, not data)"
    result["why_it_could_not_have_mattered"] = (
        "apply_universe_filters returned its input unchanged at min_market_cap 2e9 (no marketCap "
        "column on the metrics frame) and returns it unchanged at 0; universe.CURRENT is the same "
        "S&P 500 definition, now resolved from config.yaml.")
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "differences"},
                     indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
