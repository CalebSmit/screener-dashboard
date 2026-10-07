"""Prove the 2026-10-07 weight-resolution refactor changed no score.

``factor_engine.metric_weight_profiles`` replaced inline weight handling in
``compute_category_scores``, and ``applicable_coverage`` replaced a loop in
``compute_composite``. The claim in METHODOLOGY_CHANGELOG.md 2026-10-07 (owner-run) is that
scores are **bit-identical**. This script is the check: it loads the engine as committed at a
reference revision (default: the tag ``good/2026-10-07``, the nightly merge immediately before
the change), runs both engines over the same percentile table from the latest run, and compares
every score column, ``Composite``, ``Composite_Pct`` and ``Composite_Confidence`` for exact
equality.

Run:  python research/measurements/2026-10-07-scoring-refactor-identity.py [git-revision]

Needs a ``runs/<id>/03_percentiles.parquet`` (the data loop leaves several). Exit 0 if identical.
"""

from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_old_engine(rev: str):
    src = subprocess.run(["git", "show", f"{rev}:factor_engine.py"], cwd=ROOT,
                         capture_output=True, text=True, encoding="utf-8", check=True).stdout
    tmp = Path(tempfile.mkdtemp()) / "factor_engine_old.py"
    tmp.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("factor_engine_old", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def latest_run_with_percentiles() -> Path:
    runs = sorted((p for p in (ROOT / "runs").iterdir() if (p / "03_percentiles.parquet").exists()),
                  key=lambda p: (p / "03_percentiles.parquet").stat().st_mtime)
    if not runs:
        raise SystemExit("no run with 03_percentiles.parquet - run the screener first")
    return runs[-1]


def main() -> int:
    rev = sys.argv[1] if len(sys.argv) > 1 else "good/2026-10-07"
    import factor_engine as new  # noqa: PLC0415

    old = load_old_engine(rev)
    run = latest_run_with_percentiles()
    with open(run / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = pd.read_parquet(run / "03_percentiles.parquet")

    a = old.compute_category_scores(base.copy(), copy.deepcopy(cfg))
    b = new.compute_category_scores(base.copy(), copy.deepcopy(cfg))
    score_cols = [c for c in a.columns if c.endswith("_score")]
    same_scores = a[score_cols].equals(b[score_cols])

    a2 = old.compute_composite(a.copy(), copy.deepcopy(cfg))
    b2 = new.compute_composite(b.copy(), copy.deepcopy(cfg))
    same = {c: bool(a2[c].equals(b2[c])) for c in ("Composite", "Composite_Pct", "Composite_Confidence")}

    print(f"reference revision: {rev}   run: {run.name}   rows: {len(base)}")
    print(f"category scores identical ({len(score_cols)} columns): {same_scores}")
    for k, v in same.items():
        print(f"{k} identical: {v}")
    print(f"coverage discount applied to {int((b2['_cov_discount'] > 0).sum())} rows")
    return 0 if same_scores and all(same.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
