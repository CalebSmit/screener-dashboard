"""Replay the 2026-10-22 input gates against every published ranking.

OWNER_FOCUS.md 2026-10-21, items 1-3 (screener-bot ROADMAP 1.3-1.5) add three FAIL rules to
``scripts/check_run_health.py``. A gate on a measurement nobody has replayed jams the loop on its
first odd morning, so before shipping, every rule is run against history with the gate's own
functions:

* every version of ``dashboard_data.js`` on ``main``'s first-parent line since 2026-07-28 - each
  one a page the site actually served - judged against the version before it (the "last
  published" ranking the gate reads from git), and
* the complete run folders in the main checkout's gitignored ``runs/`` (their ``universe.json`` is
  the real input of the unscored-share rule; the payload history does not carry it).

What cannot be replayed is said so in the output, not approximated silently:

* **price dates** did not exist before 2026-10-22, so the bar-date rule is replayed through its
  necessary condition - a stock with no price cannot be dated on any session - and the share of
  stocks with a price is reported as the proxy;
* the **unscored share** before the run folders exists only where the committed
  ``validation/data_quality_log.csv`` holds the run's own rows ("Excluded from scoring").

The rank-continuity rule's methodology version is replayed as the gate will read it: the current
``RunContext.config_hash`` applied to each commit's own ``config.yaml`` and that commit's
``scoring_schema`` counter, and the changelog as it stood at that commit.

It also measures how the published ranking drifts with distance (Spearman against the version
``k`` back), which bounds how long a refused run can leave the baseline standing before ordinary
drift alone would also fail the gate.

Run from the repo root:  python research/measurements/2026-10-22-input-gates-replay.py
Writes the JSON beside this file.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import check_run_health as h  # noqa: E402
from run_context import RunContext  # noqa: E402

MAIN_RUNS = Path(r"C:\Users\smitc\OneDrive\Documents\Screener\runs")
SINCE = "2026-07-28"
SCOPE_FROM = "2026-08-21"          # the owner's replay window; earlier rows are the known-bad runs
OUT = Path(__file__).with_suffix(".json")


def git(*args: str) -> bytes:
    r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True)
    return r.stdout if r.returncode == 0 else b""


def show(sha: str, path: str) -> str:
    return git("show", f"{sha}:{path}").decode("utf-8", errors="replace")


def versions() -> list[dict]:
    out = git("log", "--first-parent", "main", "--reverse", f"--since={SINCE}T00:00:00",
              "--format=%H|%ad|%s", "--date=iso-strict", "--", "dashboard_data.js").decode()
    rows = []
    for line in out.splitlines():
        sha, when, subj = line.split("|", 2)
        rows.append({"sha": sha, "committed": when, "subject": subj})
    # the version published before the window opens, as the first baseline
    prior = git("log", "--first-parent", "main", "-1", f"--before={SINCE}T00:00:00",
                "--format=%H|%ad|%s", "--date=iso-strict", "--", "dashboard_data.js").decode().strip()
    if prior:
        sha, when, subj = prior.split("|", 2)
        rows.insert(0, {"sha": sha, "committed": when, "subject": subj, "baseline_only": True})
    return rows


_version_cache: dict[str, str | None] = {}


def methodology_version_at(sha: str) -> str | None:
    if sha not in _version_cache:
        try:
            cfg = yaml.safe_load(show(sha, "config.yaml")) or {}
            m = re.search(r'"scoring_schema":\s*(\d+)', show(sha, "run_context.py"))
            _version_cache[sha] = RunContext.config_hash(None, cfg, int(m.group(1)) if m else 0)
        except Exception:  # noqa: BLE001
            _version_cache[sha] = None
    return _version_cache[sha]


def unscored_at(sha: str, run_day: str) -> dict | None:
    """The run's own 'Excluded from scoring' rows, if the committed DQ log is this run's."""
    text = show(sha, "validation/data_quality_log.csv")
    if not text.strip():
        return None
    try:
        d = pd.read_csv(io.StringIO(text))
    except Exception:  # noqa: BLE001
        return None
    d = d[d["Timestamp"].astype(str).str[:10] == run_day]
    if d.empty:
        return None
    ex = d[d["Action_Taken"].astype(str).str.strip() == "Excluded from scoring"]
    try:
        universe = len(json.loads(show(sha, "sp500_tickers.json")))
    except ValueError:
        universe = None
    return {"unscored": sorted(set(ex["Ticker"].astype(str))), "universe": universe,
            "fetch_failures": int((d["Issue_Type"] == "fetch_failure").sum())}


def judge(cur: dict, base: dict, sha: str, base_sha: str) -> dict:
    """Every new rule on one published version, through check_run_health's own functions."""
    cur = dict(cur, kpis=dict(cur.get("kpis") or {}, methodology_version=methodology_version_at(sha)))
    base = dict(base, kpis=dict(base.get("kpis") or {},
                                methodology_version=methodology_version_at(base_sha)))
    run_ts = cur["kpis"].get("run_timestamp") or ""
    run_day = str(run_ts)[:10]
    row: dict = {"run_timestamp": run_ts, "baseline_run_timestamp": base["kpis"].get("run_timestamp")}

    # rule 1
    c = h.rank_continuity(cur, base)
    res = h.Result()
    changelog = show(sha, "METHODOLOGY_CHANGELOG.md")
    h.check_rank_continuity(cur, base, res, changelog_text=changelog)
    row["rule1"] = {
        "spearman": None if c["spearman"] is None else round(c["spearman"], 4),
        "n_common": c["n_common"], "new_in_top": len(c["new_in_top"]), "fallen": len(c["fallen"]),
        "fallen_names": c["fallen"],
        "version_changed": cur["kpis"]["methodology_version"] != base["kpis"]["methodology_version"],
        "override": next((n.split("allowed because ")[1] for n in res.notes if "allowed because" in n), None),
        # whether the waiver would have been available had a limit been crossed
        "override_available": h.continuity_override(cur, base, changelog),
        # the owner's literal wording, "an entry dated today": the window collapsed to the run day
        "override_if_dated_today_only": h.continuity_override(
            cur, dict(base, kpis=dict(base["kpis"], run_timestamp=run_ts)), changelog),
        "verdict": "FAIL" if res.failures else ("PASS (override)" if any("allowed because" in n for n in res.notes) else "PASS"),
    }

    # rule 2 - the floor from the payload; the unscored share from the run's own DQ rows
    scored = len(cur.get("table_data") or [])
    u = unscored_at(sha, run_day)
    share = (len(u["unscored"]) / u["universe"]) if (u and u["universe"]) else None
    fails = []
    if scored < h.MIN_SCORED:
        fails.append(f"{scored} ranked < {h.MIN_SCORED}")
    if share is not None and share > h.MAX_EXCLUDED_SHARE:
        fails.append(f"{len(u['unscored'])} of {u['universe']} unscored")
    row["rule2"] = {"scored": scored, "unscored": None if u is None else u["unscored"],
                    "universe": None if u is None else u["universe"],
                    "unscored_share": None if share is None else round(share, 4),
                    "verdict": "FAIL" if fails else "PASS", "why": fails,
                    "unscored_source": "committed DQ log" if u else "not in history"}

    # rule 3 - the dated half through its necessary condition (a price), the unchanged half as is
    detail = cur.get("stock_detail") or {}
    priced = sum(1 for v in detail.values() if v.get("price"))
    session = h.expected_session(run_ts).isoformat() if run_ts else None
    probe = {**cur, "stock_detail": {t: dict(v, price_date=session if v.get("price") else None)
                                     for t, v in detail.items()}}
    res3 = h.Result()
    h.check_price_freshness(probe, base, res3, None)
    unchanged = next((x for x in res3.notes + res3.failures if "unchanged from the published" in x), None)
    m = re.match(r"(\d+)/(\d+) prices unchanged", unchanged or "")
    row["rule3"] = {
        "expected_session": session,
        "priced_share": round(priced / len(detail), 4) if detail else None,
        "dated_proxy_verdict": "FAIL" if detail and priced / len(detail) < h.MIN_PRICE_ON_SESSION else "PASS",
        "unchanged": None if not m else [int(m.group(1)), int(m.group(2))],
        "unchanged_checked": bool(m),
        "verdict": "FAIL" if res3.failures else "PASS",
        "why": res3.failures,
    }
    row["any_fail"] = any(row[k]["verdict"].startswith("FAIL") for k in ("rule1", "rule2", "rule3"))
    return row


def run_folders(published: list[tuple[str, dict]]) -> list[dict]:
    """The complete run folders: rule 2 from the real universe.json, and rule 1 against the version
    that was published when each run started."""
    out = []
    if not MAIN_RUNS.exists():
        return out
    for d in sorted(MAIN_RUNS.iterdir()):
        if not (d / "meta.json").exists() or not (d / "dashboard_data.js").exists():
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        payload = h.load_payload(d)
        start = meta.get("start_time", "")
        before = [(s, p) for s, p in published if (p.get("kpis") or {}).get("run_timestamp", "") < start]
        res = h.Result()
        h.check_scored_universe(payload, d, res)
        row = {"run": d.name, "start_time": start, "scored": len(payload.get("table_data") or []),
               "universe_json_failed": json.loads((d / "universe.json").read_text())["failed"]
               if (d / "universe.json").exists() else None,
               "rule2": "FAIL" if res.failures else ("MISSING" if res.unevaluable else "PASS"),
               "rule2_lines": res.failures + res.notes + res.unevaluable}
        if before:
            bsha, base = before[-1]
            c = h.rank_continuity(payload, base)
            row["rule1_vs"] = (base.get("kpis") or {}).get("run_timestamp")
            row["rule1"] = {"spearman": round(c["spearman"], 4), "new_in_top": len(c["new_in_top"]),
                            "fallen": len(c["fallen"])}
        out.append(row)
    return out


def main() -> None:
    vs = versions()
    loaded: list[tuple[str, dict, dict]] = []
    for v in vs:
        text = show(v["sha"], "dashboard_data.js")
        try:
            loaded.append((v["sha"], v, h._parse_payload_text(text)))
        except ValueError:
            print("unparseable", v["sha"])
    rows = []
    for (bsha, _bv, base), (sha, v, cur) in zip(loaded, loaded[1:]):
        r = judge(cur, base, sha, bsha)
        r.update({"sha": sha[:7], "subject": v["subject"], "committed": v["committed"],
                  "baseline_sha": bsha[:7],
                  "data_loop": v["subject"].startswith("data: screener run"),
                  "in_scope": v["committed"][:10] >= SCOPE_FROM})
        rows.append(r)
        print(f"{v['committed'][:16]} {sha[:7]} {'D' if r['data_loop'] else '-'} "
              f"r1 {r['rule1']['verdict']:<15} rho {r['rule1']['spearman']} new {r['rule1']['new_in_top']} "
              f"fallen {r['rule1']['fallen']} vchg {r['rule1']['version_changed']} | "
              f"r2 {r['rule2']['verdict']} scored {r['rule2']['scored']} unscored "
              f"{None if r['rule2']['unscored'] is None else len(r['rule2']['unscored'])} | "
              f"r3 {r['rule3']['verdict']} priced {r['rule3']['priced_share']} "
              f"unchanged {r['rule3']['unchanged']}  {v['subject'][:50]}")

    # drift with distance, over the in-scope data-loop versions (a refused run leaves the baseline
    # standing; how many runs later would ordinary drift alone fail the gate?)
    dl = [(s, p) for s, v, p in loaded if v["subject"].startswith("data: screener run")
          and v["committed"][:10] >= SCOPE_FROM]
    drift = {}
    for k in (1, 2, 3, 5, 10, 15, 20, 30):
        rhos, news, falls = [], [], []
        for i in range(k, len(dl)):
            c = h.rank_continuity(dl[i][1], dl[i - k][1])
            if c["spearman"] is not None:
                rhos.append(c["spearman"]); news.append(len(c["new_in_top"])); falls.append(len(c["fallen"]))
        if rhos:
            drift[k] = {"pairs": len(rhos), "min_spearman": round(min(rhos), 4),
                        "median_spearman": round(sorted(rhos)[len(rhos) // 2], 4),
                        "max_new_in_top": max(news), "max_fallen": max(falls)}

    folders = run_folders([(s, p) for s, _v, p in loaded])
    in_scope = [r for r in rows if r["in_scope"]]
    ordinary = [r for r in in_scope if r["data_loop"]]
    summary = {
        "versions_judged": len(rows),
        "in_scope_versions": len(in_scope),
        "in_scope_data_loop_runs": len(ordinary),
        "fails_by_rule": {k: [f"{r['committed'][:10]} {r['sha']} {r['subject'][:60]}" for r in rows
                              if r[k]["verdict"] == "FAIL"] for k in ("rule1", "rule2", "rule3")},
        "overrides": [f"{r['committed'][:10]} {r['sha']} {r['subject'][:60]}: {r['rule1']['override']}"
                      for r in rows if r["rule1"]["verdict"] == "PASS (override)"],
        "override_available_on": [f"{r['committed'][:10]} {r['sha']} {r['subject'][:50]}: "
                                  f"{r['rule1']['override_available']}"
                                  for r in rows if r["rule1"]["override_available"]],
        "data_loop_runs_first_scored_under_a_change_documented_the_day_before": [
            f"{r['committed'][:10]} {r['sha']}" for r in rows
            if r["data_loop"] and r["rule1"]["override_available"]
            and not r["rule1"]["override_if_dated_today_only"]],
        "ordinary_data_loop_min_spearman": min((r["rule1"]["spearman"] for r in ordinary
                                                if r["rule1"]["spearman"] is not None), default=None),
        "ordinary_data_loop_max_new_in_top": max((r["rule1"]["new_in_top"] for r in ordinary), default=None),
        "ordinary_data_loop_max_fallen": max((r["rule1"]["fallen"] for r in ordinary), default=None),
        "ordinary_data_loop_min_scored": min((r["rule2"]["scored"] for r in ordinary), default=None),
        "ordinary_data_loop_max_unscored_share": max((r["rule2"]["unscored_share"] for r in ordinary
                                                      if r["rule2"]["unscored_share"] is not None), default=None),
        "ordinary_data_loop_max_unchanged_share": max((r["rule3"]["unchanged"][0] / r["rule3"]["unchanged"][1]
                                                       for r in ordinary if r["rule3"]["unchanged"]), default=None),
        "drift_by_distance": drift,
        "thresholds": {"RANK_MIN_SPEARMAN": h.RANK_MIN_SPEARMAN, "RANK_MAX_NEW_IN_TOP": h.RANK_MAX_NEW_IN_TOP,
                       "RANK_MAX_FALLEN": h.RANK_MAX_FALLEN, "RANK_FALL_LINE": h.RANK_FALL_LINE,
                       "MIN_SCORED": h.MIN_SCORED, "MAX_EXCLUDED_SHARE": h.MAX_EXCLUDED_SHARE,
                       "MIN_PRICE_ON_SESSION": h.MIN_PRICE_ON_SESSION,
                       "MAX_UNCHANGED_ACROSS_SESSION": h.MAX_UNCHANGED_ACROSS_SESSION},
    }
    OUT.write_text(json.dumps({"summary": summary, "versions": rows, "run_folders": folders},
                              indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=str))
    for f in folders:
        print(f)


if __name__ == "__main__":
    main()
