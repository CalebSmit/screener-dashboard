"""What did the 2026-10-19 ``pit_record`` fixes change in backtest v2's monthly records? Measured.

Two changes to ``pit_record.Filed`` on 2026-10-19, and the 176 cached months rebuilt for both:

* **The fiscal-year rule** (research/2026-10-19-top-of-ranking-data-audit.md, section 4): of
  several versions of a fiscal-year context, the latest **10-K** wins over a 10-Q filed after it.
* **The tie rule**, found by the first rebuild: rows filed the same day were told apart by an
  unstable sort's row order, so the fiscal-year fix - which changed the sort - also moved hundreds
  of name-months it had nothing to do with. Ties now go to the end nearest the period asked for,
  then end and value; sorts are stable (research/2026-10-19-pit-fix-rebuild-effect.md).

This measures, before anyone reads the rerun, what the rebuild moved and why:

1. **The fiscal-year rule's footprint, exactly.** For every scored name-month whose company has a
   fiscal-year-length 10-Q fact that disagrees with a 10-K fact for the same context (the only place
   that rule can bite), the record is rebuilt with today's ``Filed`` and with today's ``Filed``
   minus the 10-K preference (``NoFiscalYearRule``) by the build's own ``backtest_v2.record_at``,
   and every numeric field compared. Same tie rule on both sides, so the difference is that rule's.
2. **Metrics** - the months before (``--before``, the 2026-10-15 build the ledger rows ran on)
   against after (``cache/v2_metrics``): which name-months and metric columns moved, split into the
   fiscal-year footprint and the rest. The rest is the tie rule: the old build's ties were settled
   by row order, so those moves replace an arbitrary pick with a defined one.
   ``--intermediate`` (the first rebuild: fiscal-year rule, old ties) isolates it - between that
   build and the final one only the tie rule differs.
3. **Ranks** - the baseline weighting ranked on both (``v2_experiments.rank_all``, the engine's own
   ranking): per month the Spearman correlation of composites, how far ranks moved, how many names
   changed decile, and top-decile turnover - for the footprint's names and for everyone else.

Usage: python research/measurements/2026-10-19-pit-fix-rebuild-effect.py --before DIR
       [--intermediate DIR]   (copies of earlier cache/v2_metrics). Writes the JSON beside this file.
Research tooling only: nothing here is imported by production, and under CLAUDE.md rule 5 nothing it
reports decides anything before 2027-02-11.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import backtest_v2 as v2  # noqa: E402
import pit_record as pr  # noqa: E402
import v2_experiments as X  # noqa: E402

OUT = Path(__file__).with_suffix(".json")
NEW_FILED = pr.Filed
TOL = 1e-9


class NoFiscalYearRule(NEW_FILED):
    """Today's ``pit_record.Filed`` without the 2026-10-19 fiscal-year rule: of each (concept,
    start, end) the version filed last wins, a 10-Q's fiscal-year context included. The tie rule
    (stable sorts, nearest end) is today's, so comparing the two isolates the fiscal-year rule."""

    def __init__(self, facts, ticker, as_of):
        self.as_of = pd.Timestamp(as_of)
        f = facts[(facts["ticker"] == ticker)]
        f = f[f["form"].isin(pr.FORMS) & (pd.to_datetime(f["filed"]) <= self.as_of)].copy()
        for c in ("start", "end", "filed"):
            f[c] = pd.to_datetime(f[c], errors="coerce")
        f = f.dropna(subset=["end", "val"])
        f["days"] = (f["end"] - f["start"]).dt.days
        f["_amendment"] = f["form"].str.endswith("/A").astype(int)
        f = f.sort_values(["filed", "_amendment"], kind="mergesort")
        f = f.drop_duplicates(["concept", "start", "end"], keep="last").sort_values("filed", kind="mergesort")
        self._by = {c: g for c, g in f.groupby("concept")}
        self._fam = {}


def candidates(facts: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """ticker -> the earliest filing date at which the fiscal-year rule can matter: a
    fiscal-year-length 10-Q fact whose context a 10-K also reports with a different value."""
    f = facts[facts["form"].isin(pr.FORMS)].copy()
    for c in ("start", "end", "filed"):
        f[c] = pd.to_datetime(f[c], errors="coerce")
    f = f.dropna(subset=["start", "end", "val"])
    f = f[(f["end"] - f["start"]).dt.days.between(340, 380)]
    k = f[f["form"].str.startswith("10-K")]
    q = f[f["form"].str.startswith("10-Q")]
    m = q.merge(k, on=["ticker", "concept", "start", "end"], suffixes=("_q", "_k"))
    m = m[(m["val_q"] - m["val_k"]).abs() > TOL * m["val_k"].abs().clip(lower=1)]
    first = m.assign(when=m[["filed_q", "filed_k"]].max(axis=1)).groupby("ticker")["when"].min()
    return {t: d for t, d in first.items()}


def numeric_diff(a: dict, b: dict) -> dict:
    out = {}
    for key in sorted(set(a) | set(b)):
        x, y = a.get(key), b.get(key)
        if not isinstance(x, (int, float, np.floating, np.integer)) or isinstance(x, bool):
            continue
        if not isinstance(y, (int, float, np.floating, np.integer)) or isinstance(y, bool):
            continue
        x, y = float(x), float(y)
        if np.isnan(x) and np.isnan(y):
            continue
        if np.isnan(x) != np.isnan(y) or abs(x - y) > TOL * max(abs(x), abs(y), 1.0):
            out[key] = (None if np.isnan(x) else x, None if np.isnan(y) else y)
    return out


def _selection(F) -> pd.DataFrame:
    """The facts a ``Filed`` chose, one row per (concept, start, end)."""
    if not F._by:
        return pd.DataFrame(columns=["concept", "start", "end", "val"])
    s = pd.concat(F._by.values())[["concept", "start", "end", "val"]]
    return s.sort_values(["concept", "start", "end"]).reset_index(drop=True)


def selections_differ(facts: pd.DataFrame, tk: str, t: pd.Timestamp) -> bool:
    a, b = _selection(NoFiscalYearRule(facts, tk, t)), _selection(NEW_FILED(facts, tk, t))
    return len(a) != len(b) or not a.equals(b)


def record_footprint(months_after: dict, inputs: v2.Inputs, cands: dict) -> dict:
    """{(month, ticker): {field: (without the rule, with it)}} over every scored name-month the
    fiscal-year rule can touch. A name-month whose two selections are identical cannot differ and
    is not rebuilt; the rest are rebuilt both ways by the build's own ``record_at``."""
    out = {}
    for m, (df, _state) in sorted(months_after.items()):
        t = pd.Timestamp(m)
        for tk in df["Ticker"]:
            since = cands.get(tk)
            if since is None or since > t:
                continue
            if not selections_differ(inputs.facts.get(tk, inputs.empty_facts), tk, t):
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                pr.Filed = NoFiscalYearRule
                try:
                    old, _ = v2.record_at(tk, inputs, t)
                finally:
                    pr.Filed = NEW_FILED
                new, _ = v2.record_at(tk, inputs, t)
            if old is None or new is None:
                continue
            d = numeric_diff(old, new)
            if d:
                out[(m, tk)] = d
        if m.endswith("-12-31"):
            print(f"  {m}: {len(out)} name-months differ so far", flush=True)
    return out


def metric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not c.startswith("_") and c not in ("Ticker", "Company", "Sector")
            and pd.api.types.is_numeric_dtype(df[c])]


def metric_moves(before: dict, after: dict) -> tuple[dict, dict, dict]:
    """({(month, ticker): [metric, ...]}, membership changes per month, static-field changes)."""
    moved, members, static = {}, {}, {}
    for m in sorted(after):
        a = after[m][0].set_index("Ticker")
        b = before[m][0].set_index("Ticker") if m in before else a.iloc[:0]
        added, removed = sorted(set(a.index) - set(b.index)), sorted(set(b.index) - set(a.index))
        if added or removed:
            members[m] = {"added": added, "removed": removed}
        common = a.index.intersection(b.index)
        for col in ("Sector", "_is_bank_like"):
            if col in a and col in b:
                ch = [tk for tk in common if str(a.at[tk, col]) != str(b.at[tk, col])]
                if ch:
                    static.setdefault(m, {})[col] = ch
        for col in metric_columns(a):
            if col not in b:
                continue
            x, y = b.loc[common, col].astype(float), a.loc[common, col].astype(float)
            diff = (x.isna() != y.isna()) | ((x - y).abs() > TOL * np.maximum(np.maximum(x.abs(), y.abs()), 1.0))
            for tk in common[diff.fillna(False).values]:
                moved.setdefault((m, tk), []).append(col)
    return moved, members, static


def by_metric(d: dict) -> dict:
    c: dict[str, int] = {}
    for cols in d.values():
        for col in cols:
            c[col] = c.get(col, 0) + 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


def rank_moves(before: dict, after: dict, touched: set) -> dict:
    cfg = X.base_config()
    rb, ra = X.rank_all(before, cfg), X.rank_all(after, cfg)
    per_month, shifts_fix, shifts_other = {}, [], []
    top_changes = top_slots = decile_fix = decile_other = 0
    for m in sorted(ra["month"].unique()):
        a = ra[ra["month"] == m].set_index("Ticker")
        b = rb[rb["month"] == m].set_index("Ticker")
        common = a.index.intersection(b.index)
        if len(common) < 10:
            continue
        rho = float(a.loc[common, "Composite"].rank().corr(b.loc[common, "Composite"].rank()))
        # ranks among the common names, so a name entering or leaving does not shift everyone
        ar = a.loc[common, "Composite"].rank(ascending=False)
        br_ = b.loc[common, "Composite"].rank(ascending=False)
        shift = (ar - br_).abs()
        n = len(common)
        dec_a, dec_b = np.ceil(ar * 10 / n), np.ceil(br_ * 10 / n)
        k = max(1, n // 10)
        top_a, top_b = set(ar.nsmallest(k).index), set(br_.nsmallest(k).index)
        top_changes += len(top_a - top_b)
        top_slots += k
        fix = [tk for tk in common if (m, tk) in touched]
        for tk in common:
            (shifts_fix if tk in fix else shifts_other).append(float(shift[tk]))
            if dec_a[tk] != dec_b[tk]:
                if tk in fix:
                    decile_fix += 1
                else:
                    decile_other += 1
        per_month[m] = {"n_common": int(n), "spearman": round(rho, 6),
                        "max_rank_shift": float(shift.max()), "names_shifted": int((shift > 0).sum()),
                        "footprint_names": fix, "top_decile_changes": len(top_a - top_b),
                        "footprint_shifts": {tk: float(shift[tk]) for tk in fix}}
    sf, so = np.array(shifts_fix), np.array(shifts_other)
    rhos = np.array([v["spearman"] for v in per_month.values()])
    return {"months": len(per_month),
            "spearman_min": float(rhos.min()), "spearman_median": float(np.median(rhos)),
            "months_identical": int((rhos >= 1 - 1e-12).sum()),
            "footprint_name_months": int(len(sf)),
            "footprint_shift_mean": float(sf.mean()) if len(sf) else None,
            "footprint_shift_max": float(sf.max()) if len(sf) else None,
            "footprint_changed_decile": int(decile_fix),
            "other_name_months": int(len(so)),
            "other_shift_mean": float(so.mean()) if len(so) else None,
            "other_shift_p99": float(np.percentile(so, 99)) if len(so) else None,
            "other_shift_max": float(so.max()) if len(so) else None,
            "other_changed_decile": int(decile_other),
            "top_decile_slots": int(top_slots), "top_decile_changes": int(top_changes),
            "per_month": per_month}


def load(dirpath: Path) -> dict:
    keep = v2.METRICS_DIR
    try:
        v2.METRICS_DIR = dirpath
        return X.load_months()
    finally:
        v2.METRICS_DIR = keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True, help="a copy of the pre-fix cache/v2_metrics")
    ap.add_argument("--intermediate", default="",
                    help="a copy of the first rebuild (fiscal-year rule, old ties)")
    a = ap.parse_args()
    before, after = load(Path(a.before)), load(v2.METRICS_DIR)
    print(f"months: before {len(before)}, after {len(after)}", flush=True)
    inputs = v2.Inputs.load()
    cands = candidates(pd.concat(inputs.facts.values()))
    print(f"companies the fiscal-year rule can touch: {len(cands)}", flush=True)
    foot = record_footprint(after, inputs, cands)
    touched = set(foot)
    fields: dict[str, int] = {}
    for d in foot.values():
        for k in d:
            fields[k] = fields.get(k, 0) + 1
    moved, members, static = metric_moves(before, after)
    in_fix = {k: v for k, v in moved.items() if k in touched}
    outside = {k: v for k, v in moved.items() if k not in touched}
    weighted = sorted({mm for t in ("metric_weights", "bank_metric_weights")
                       for cat in (X.base_config().get(t) or {}).values() for mm, w in cat.items() if float(w) > 0})
    ranks = rank_moves(before, after, touched)
    tie_alone = None
    if a.intermediate:
        mid = load(Path(a.intermediate))
        mv, _, _ = metric_moves(mid, after)
        mr = rank_moves(mid, after, set())
        tie_alone = {"what": "first rebuild (fiscal-year rule, old ties) against the final one: "
                             "only the tie rule differs",
                     "name_months_moved": len(mv), "in_fiscal_year_footprint": len(set(mv) & touched),
                     "companies": len({t for _, t in mv}), "by_metric": by_metric(mv),
                     "ranks": {k: v for k, v in mr.items() if k != "per_month"}}
    total_nm = sum(len(df) for df, _ in after.values())
    out = {
        "before_content": X.content_fingerprint(metrics_dir=Path(a.before)),
        "after_content": X.content_fingerprint(),
        "name_months_after": total_nm,
        "companies_fiscal_year_rule_can_touch": len(cands),
        "fiscal_year_footprint": {"name_months": len(foot), "companies": sorted({t for _, t in foot}),
                                  "months": len({m for m, _ in foot}),
                                  "raw_fields": dict(sorted(fields.items(), key=lambda kv: -kv[1])),
                                  "rows": {f"{m} {t}": d for (m, t), d in sorted(foot.items())}},
        "metric_moves": {"name_months_moved": len(moved),
                         "companies_moved": len({t for _, t in moved}),
                         "in_fiscal_year_footprint": len(in_fix), "outside_it": len(outside),
                         "companies_outside": len({t for _, t in outside}),
                         "by_metric_in_footprint": by_metric(in_fix),
                         "weighted_metrics_moved_in_footprint": [m for m in by_metric(in_fix) if m in weighted],
                         "by_metric_outside": by_metric(outside),
                         "weighted_metrics_moved_outside": [m for m in by_metric(outside) if m in weighted],
                         "outside_rows": {f"{m} {t}": c for (m, t), c in sorted(outside.items())}},
        "tie_rule_alone": tie_alone,
        "membership_changes": members,
        "static_field_changes": static,
        "baseline_ranks": ranks,
    }
    OUT.write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    fy = out["fiscal_year_footprint"]
    mm = out["metric_moves"]
    print(f"fiscal-year footprint: {len(foot)} name-months, {len(fy['companies'])} companies, "
          f"{fy['months']} months of {len(after)}; {total_nm} name-months in all")
    print(f"raw fields: {fy['raw_fields']}")
    print(f"metric moves: {len(moved)} name-months, {mm['companies_moved']} companies "
          f"({len(in_fix)} in the footprint, {len(outside)} outside, {mm['companies_outside']} companies)")
    print(f"  in: {mm['by_metric_in_footprint']}")
    print(f"  outside: {mm['by_metric_outside']}")
    if tie_alone:
        print(f"tie rule alone: {tie_alone['name_months_moved']} name-months, {tie_alone['by_metric']}")
        print(f"  ranks: {tie_alone['ranks']}")
    print(f"membership changes: {members}")
    print(f"static field changes: { {m: {k: len(v) for k, v in d.items()} for m, d in static.items()} }")
    print(f"baseline ranks: { {k: v for k, v in ranks.items() if k != 'per_month'} }")
    print(f"wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
