"""Did the 2026-11-10 rerun - forced by a weekly filings-cache refresh - move any number?

`main` arrived red on 2026-11-10: all 32 newest ledger rows carried a `content_hash` the live
tree disagreed with, while `code_hash`, `base_config_hash` and `method_version` were level and no
commit had touched any of the nine `v2_experiments.RESULT_PATHS` files since the rows ran on
2026-11-07. The single part that had moved was `sec_facts` - `data/sec/pit/facts.parquet`, which
the 02:00 data run rewrote at 02:16:52 and which `sec_fundamentals.FACTS_MAX_AGE_DAYS = 7` makes
a weekly event. `month_files` was identical, so the monthly records the experiments read had not
been rebuilt.

The prediction before the rerun, written here because it is the whole argument for narrowing what
`content_hash` is taken over: **every statistic identical, only the provenance moves.** The
filings reach a row only through the monthly records, and those did not change.

This is the 2026-11-05 measurement repeated for a different cause, and it reuses that script's
comparison rather than restating it: each id's newest row against the row before it, ignoring the
fields a rerun is expected to change.

Run: ``python research/measurements/2026-11-10-filings-cache-rerun-moved-nothing.py``
Writes ``research/measurements/2026-11-10-filings-cache-rerun-moved-nothing.json``.
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
    m.OUT = OUT                      # write this date's file, never overwrite 2026-11-05's
    # its own summary names the file that moved in November's *first* case; print ours only
    with contextlib.redirect_stdout(io.StringIO()) as quiet:
        result = m.main()
    if result.get("differences"):
        print(quiet.getvalue())
    result["which_file_had_moved"] = "data/sec/pit/facts.parquet (sec_facts), refreshed weekly"
    result["why_it_could_not_have_mattered"] = (
        "backtest_v2._load_facts reads the filings when it builds the monthly records; the only "
        "DATA_SOURCES file read at experiment time is cache/v2_daily.parquet. month_files was "
        "identical, so the records were not rebuilt.")
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "differences"},
                     indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
