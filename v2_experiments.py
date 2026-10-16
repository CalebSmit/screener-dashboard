"""The research engine: pre-registered experiments on backtest v2, judged as a sceptic would.

WHY THIS EXISTS - 2026-10-15, owner: build the screener's methodology the way automated research
loops are being built - many hypotheses from the literature, tested against each other - so the
result can be shown to other people and trusted. The danger of such a loop is documented: test
enough ideas on one history and some look excellent by chance (Harvey, Liu & Zhu 2016; McLean &
Pontiff 2016; the Factor Lab's own after-the-paper table). So the loop is built around four rules,
and they are what make a result from it worth showing:

1. **Pre-registration.** Every experiment - its hypothesis, its source, and its predicted outcome -
   is written in ``research/experiments/registry.yaml`` and committed *before* it is run. The git
   history is the proof that nothing was chosen after seeing results.
2. **A held-out period.** Ideas are compared on the discovery months (2020-01 .. 2023-06) and must
   hold on months they were never looked at (2023-07 .. 2026-08). The split is fixed here, in code.
3. **Every test counts.** ``research/experiments/ledger.jsonl`` records every run of every
   experiment. Significance is judged against the number of experiments ever registered
   (Bonferroni), and against Harvey, Liu & Zhu's t > 3 bar for a new factor.
4. **Paired with the baseline.** Each idea is compared month by month with the current screener's
   weighting on the same stocks and the same months; the interval is on the difference.

What it is not: a way to set weights. Under CLAUDE.md rules 4 and 5 nothing it reports changes a
weight before 2027-02-11 - and its universe still lacks the stocks that left the index and every
analyst input (``backtest_v2_returns.CAVEATS``). A result here becomes a research question; a
research note and a changelog entry are still the only route to a weight.

Usage::

    python v2_experiments.py run [ID ...]     # run registered experiments (all if none named)
    python v2_experiments.py evaluate [ID ...] # run registered evaluations (a trading rule's record)
    python v2_experiments.py report           # rewrite research/experiments/RESULTS.md

Reads ``cache/v2_metrics/<month>.parquet`` (written by ``backtest_v2.py``) and
``cache/v2_daily.parquet``. Never imported by production code.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import math
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import backtest_v2 as v2
import backtest_v2_returns as br

ROOT = Path(__file__).resolve().parent
EXP_DIR = ROOT / "research" / "experiments"
REGISTRY = EXP_DIR / "registry.yaml"
LEDGER = EXP_DIR / "ledger.jsonl"
RESULTS = EXP_DIR / "RESULTS.md"

DISCOVERY = ("2020-01-31", "2023-06-30")
HOLDOUT = ("2023-07-31", "2026-08-31")
HLZ_T = 3.0                       # Harvey, Liu & Zhu (2016): the bar for a new factor's t-statistic
# A third period, registered 2026-10-15 AFTER the first batch had been read on 2020-2026 and BEFORE
# a single month of it was built: the eight years before the sample, which nobody here has looked
# at. Testing a result on the history that precedes its sample is Linnainmaa & Roberts (2018), "The
# history of the cross-section of stock returns", RFS 31(7) - most anomalies they tested that way
# weakened or vanished. 96 months against the 38 held out, so it can see a difference about 0.6x
# the size. Its limit is worse survivorship: the further back, the more members have since left
# the index, and v2 only has the ones still in it (the census counts them per month).
PRESAMPLE = ("2012-01-31", "2019-12-31")
SAMPLE = (DISCOVERY[0], HOLDOUT[1])


# --------------------------------------------------------------------------- configurations
def base_config() -> dict:
    return v2.v2_config(yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")))


def apply_transform(cfg: dict, tr: dict) -> dict:
    """A registered experiment's change to the baseline configuration."""
    c = copy.deepcopy(cfg)
    fw, mw = c["factor_weights"], c["metric_weights"]
    kind = tr["kind"]
    if kind == "baseline":
        pass
    elif kind == "only_category":
        for k in fw:
            fw[k] = 1 if k == tr["category"] else 0
    elif kind == "drop_category":
        fw[tr["category"]] = 0
    elif kind == "equal_categories":
        for k in fw:
            if fw[k]:
                fw[k] = 1
    elif kind == "metrics":
        # a screen on named metrics only: {category: {metric: weight}}; everything else zero
        keep = tr["metrics"]
        for cat in fw:
            fw[cat] = sum(keep.get(cat, {}).values()) if cat in keep else 0
        for table in ("metric_weights", "bank_metric_weights"):
            for cat, metrics in (c.get(table) or {}).items():
                for m in metrics:
                    metrics[m] = keep.get(cat, {}).get(m, 0)
    elif kind == "factor_weights":
        for k, v in tr["weights"].items():
            fw[k] = v
    elif kind == "config":
        # a switch the screener already has (e.g. factor_neutralization, sector_relative_composite)
        for path, value in tr["set"].items():
            node = c
            keys = path.split(".")
            for k in keys[:-1]:
                node = node.setdefault(k, {})
            node[keys[-1]] = value
    elif kind == "whole_market":
        pass                                          # handled in ranking, not in the config
    else:
        raise ValueError(f"unknown transform {kind!r}")
    return c


def config_hash(cfg: dict, tr: dict) -> str:
    return hashlib.sha256(json.dumps([cfg, tr], sort_keys=True, default=str).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- data
def load_months() -> dict[str, tuple[pd.DataFrame, dict]]:
    out = {}
    for p in sorted(v2.METRICS_DIR.glob("*.parquet")):
        try:
            if not pd.Timestamp(p.stem).is_month_end:
                continue                                # only month-ends are months
        except ValueError:
            continue
        df = pd.read_parquet(p)
        state = {"state": df["_month_state"].iloc[0] if "_month_state" in df and len(df) else "UNKNOWN"}
        out[p.stem] = (df.drop(columns=["_month_state"], errors="ignore"), state)
    return out


def data_fingerprint(months: dict) -> dict:
    """Which data an experiment ran on: a hash of every month's member list, and the count of
    stock-months. Two runs compare only when it matches - a rebuild of the monthly records (a
    fix to who is in them) is a new vintage, and the ledger says so."""
    h = hashlib.sha256()
    n = 0
    for m in sorted(months):
        tickers = sorted(map(str, months[m][0]["Ticker"]))
        n += len(tickers)
        h.update((m + ":" + ",".join(tickers) + ";").encode("utf-8"))
    return {"data_hash": h.hexdigest()[:12], "stock_months": n}


def forward_returns(months: list[str]) -> pd.DataFrame:
    daily = pd.read_parquet(ROOT / "cache" / "v2_daily.parquet")
    last = months[-1]
    nxt = (pd.Timestamp(last) + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")
    return br.forward_returns(br.month_end_prices(daily, months + [nxt]))


# --------------------------------------------------------------------------- one experiment
def rank_all(months: dict, cfg: dict, whole_market: bool = False) -> pd.DataFrame:
    frames = []
    for m, (metrics, state) in months.items():
        df = metrics.copy()
        if whole_market:
            sectors = df["Sector"].copy()
            df["Sector"] = "All"
            r = v2.rank_month(df, cfg, state)
            r["Sector"] = r["Ticker"].map(dict(zip(metrics["Ticker"], sectors)))
        else:
            r = v2.rank_month(df, cfg, state)
        cats = [c for c in r.columns if c.endswith("_score")]
        out = r[["Ticker", "Composite", "Rank"] + cats].copy()
        out.insert(0, "month", m)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def per_month(scores: pd.DataFrame, fwd: pd.DataFrame) -> pd.DataFrame:
    """IC, top-tenth-minus-universe and top-tenth volatility inputs, month by month."""
    rows = []
    for m, g in scores.groupby("month"):
        if m not in fwd.index:
            continue
        g = g.assign(ret=fwd.loc[m].reindex(g["Ticker"].values).values).dropna(subset=["ret", "Composite"])
        if len(g) < 50:
            continue
        dec = np.ceil(g["Composite"].rank(pct=True, method="first") * 10).clip(1, 10)
        top = g.loc[dec == 10, "ret"]
        rows.append({"month": m, "n": len(g), "ic": float(g["Composite"].rank().corr(g["ret"].rank())),
                     "top": float(top.mean()), "universe": float(g["ret"].mean()),
                     "top_minus_universe": float(top.mean() - g["ret"].mean())})
    return pd.DataFrame(rows).set_index("month")


def _stats(x: pd.Series) -> dict:
    x = x.dropna()
    n = len(x)
    if n < 3:
        return {"n": n}
    m, sd = float(x.mean()), float(x.std(ddof=1))
    se = sd / math.sqrt(n)
    t = m / se if se > 0 else 0.0
    # two-sided p from the t-statistic with a normal approximation (n >= 30 here)
    p = math.erfc(abs(t) / math.sqrt(2))
    return {"n": n, "mean": round(m, 5), "se": round(se, 5), "t": round(t, 2), "p": round(p, 4),
            "ci95": [round(m - 1.96 * se, 5), round(m + 1.96 * se, 5)]}


def summarise(pm: pd.DataFrame, base: pd.DataFrame | None) -> dict:
    out = {}
    for name, (a, b) in (("discovery", DISCOVERY), ("holdout", HOLDOUT), ("all", SAMPLE),
                         ("presample", PRESAMPLE)):
        sl = pm[(pm.index >= a) & (pm.index <= b)]
        if name == "presample" and sl["ic"].notna().sum() < 3:
            continue                                    # not built yet: no row, not an empty one
        r = {"ic": _stats(sl["ic"]), "top_minus_universe": _stats(sl["top_minus_universe"]),
             "top_vol": round(float(sl["top"].std(ddof=1)), 5) if len(sl) > 2 else None}
        if base is not None:
            bs = base.reindex(sl.index)
            r["ic_vs_baseline"] = _stats(sl["ic"] - bs["ic"])
            r["tmu_vs_baseline"] = _stats(sl["top_minus_universe"] - bs["top_minus_universe"])
        out[name] = r
    return out


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=ROOT).stdout.strip()
    except OSError:
        return "?"


def load_registry() -> list[dict]:
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["experiments"]


def run(ids: list[str] | None = None) -> list[dict]:
    reg = load_registry()
    by_id = {e["id"]: e for e in reg}
    if ids:
        unknown = [i for i in ids if i not in by_id]
        if unknown:
            raise SystemExit(f"not registered (register before running): {unknown}")
    months = load_months()
    if not months:
        raise SystemExit("no cached monthly metrics - run backtest_v2.py first")
    fwd = forward_returns(sorted(months))
    vintage = data_fingerprint(months)
    cfg0 = base_config()
    base_pm = per_month(rank_all(months, cfg0), fwd)
    results = []
    for e in reg:
        if ids and e["id"] not in ids:
            continue
        tr = e["transform"]
        cfg = apply_transform(cfg0, tr)
        pm = base_pm if tr["kind"] == "baseline" else per_month(
            rank_all(months, cfg, whole_market=(tr["kind"] == "whole_market")), fwd)
        res = {"id": e["id"], "run_at": datetime.now().isoformat(timespec="seconds"),
               "commit": git_commit(), "config_hash": config_hash(cfg, tr), "months": int(len(pm)),
               **vintage,
               "summary": summarise(pm, None if tr["kind"] == "baseline" else base_pm)}
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(res) + "\n")
        results.append(res)
        s = res["summary"]
        print(f"{e['id']:28} IC disc {s['discovery']['ic'].get('mean')}  hold {s['holdout']['ic'].get('mean')}"
              + (f"  vs base hold t {s['holdout']['ic_vs_baseline'].get('t')}" if "ic_vs_baseline" in s['holdout'] else ""),
              flush=True)
    return results


# --------------------------------------------------------------------------- the report
def _z_two_sided(alpha: float) -> float:
    """The normal quantile for a two-sided test at ``alpha`` (bisection on erfc; no scipy)."""
    lo, hi = 0.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if math.erfc(mid / math.sqrt(2)) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def detectable(se: float | None, n_tests: int) -> float | None:
    """The smallest true difference a comparison with standard error ``se`` detects 80% of the time
    at the engine's bar (Bonferroni over ``n_tests`` and |t| >= 3). "Not distinguishable" means
    "smaller than this", never "equal" - plan/research-engine.md queue item 2 (2026-10-15)."""
    if not se:
        return None
    return round((max(_z_two_sided(0.05 / max(n_tests, 1)), HLZ_T) + 0.84) * se, 4)


def _vs_cell(vs: dict, n_tests: int) -> str:
    if not vs:
        return "-"
    d = detectable(vs.get("se"), n_tests)
    return f"{vs.get('mean', 0):+.3f} ({vs.get('t')}" + (f"; sees {d:.3f})" if d is not None else ")")


def _presample_cells(s: dict, n_tests: int = 1) -> str:
    p = s.get("presample")
    if not p:
        return " | "
    vs = p.get("ic_vs_baseline", {})
    return (f"{p['ic'].get('mean', 0):+.3f} ({p['ic'].get('t')})"
            + " | " + _vs_cell(vs, n_tests))



def verdict(e: dict, res: dict, n_tests: int) -> str:
    """The pre-registered prediction against what happened, in words, judged on the held-out months."""
    if e.get("void"):
        return "VOID - " + e["void"]
    s = res["summary"]
    if "ic_vs_baseline" not in s["holdout"]:
        return "baseline"
    d, h = s["discovery"]["ic_vs_baseline"], s["holdout"]["ic_vs_baseline"]
    bonf = 0.05 / max(n_tests, 1)
    sig_d = d.get("p", 1) < 0.05
    sig_h = h.get("p", 1) < 0.05
    if sig_d and sig_h and np.sign(d.get("mean", 0)) == np.sign(h.get("mean", 0)):
        word = "better" if h["mean"] > 0 else "worse"
        strong = " and clears the multiple-testing bar" if h.get("p", 1) < bonf and abs(h.get("t", 0)) >= HLZ_T else \
            ", but not the multiple-testing bar"
        return f"{word} than the baseline in both periods{strong}"
    if sig_d and not sig_h:
        return "looked different in discovery; did not hold out"
    if sig_h and not sig_d:
        return "different only in the held-out months - not confirmed"
    return "not distinguishable from the baseline"


def verdict_presample(e: dict, res: dict, n_tests: int) -> str:
    """The pre-registered rule for 2012-2019 (registry.yaml ``presample``), in words."""
    if e.get("void"):
        return "void"
    s = res["summary"]
    if "presample" not in s:
        return "not run on 2012-2019"
    if "ic_vs_baseline" not in s["presample"]:
        return "baseline"
    d, ref = s["presample"]["ic_vs_baseline"], s["all"]["ic_vs_baseline"]
    bonf = 0.05 / max(n_tests, 1)
    if d.get("p", 1) < bonf and abs(d.get("t", 0)) >= HLZ_T:
        same = np.sign(d.get("mean", 0)) == np.sign(ref.get("mean", 0))
        word = "better" if d["mean"] > 0 else "worse"
        return (f"{word} in 2012-2019 beyond the multiple-testing bar, "
                + ("the same direction as 2020-2026" if same else "the opposite direction to 2020-2026"))
    if d.get("p", 1) < 0.05:
        return "p < 0.05 in 2012-2019 only - about 1 in 20 would by chance"
    return "not distinguishable in 2012-2019"


# --------------------------------------------------------------------------- evaluations
# An *evaluation* asks a different question from an experiment: not "does this weighting rank
# better" but "what would this construction rule have done" - the trading bot's own rule, on the
# baseline's rankings. Registered the same way (``registry.yaml`` ``evaluations:``), kept in the same
# ledger, judged against a prediction written before the run. Added 2026-10-15
# (plan/research-engine.md queue item 4).
def load_evaluations() -> list[dict]:
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8")).get("evaluations") or []


def bot_month_inputs(months: dict, cfg: dict) -> dict:
    """Each month's ranking with what the bot reads: rank, sector and the two trap flags, set by
    the engine's own functions in the live run's order (composite, flags, rank)."""
    out = {}
    for m, (metrics, state) in months.items():
        r = v2.rank_month(metrics.copy(), cfg, state)
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            r = v2.fe.apply_value_trap_flags(r, cfg)
            r = v2.fe.apply_growth_trap_flags(r, cfg)
        out[m] = r[["Ticker", "Rank", "Sector", "Value_Trap_Flag", "Growth_Trap_Flag"]].sort_values("Rank")
    return out


def simulate_bot(inputs: dict, fwd: pd.DataFrame, rule: dict, start_equity: float = 100_000.0) -> pd.DataFrame:
    """The bot's ``plan_rebalance`` (screener-bot ``bot/strategy.py``, v1.1) applied once a month.

    Exits: a holding ranked worse than ``sell_rank``, or missing from the month's ranking, is sold.
    Entries: empty slots up to ``target_count`` are filled from the best-ranked names not held before
    the month's sells, skipping trap-flagged names and any that would put more than
    ``max_per_sector`` in one sector, each sized at the smaller of an equal share of equity after the
    cash reserve and an equal share of the cash available. No new buys while equity is
    ``drawdown_halt_pct`` below its peak. Every traded dollar costs ``cost_bps``. Positions drift with
    their returns between months; cash earns nothing.

    Differences from the live bot, all stated in the report: monthly, not weekly; a name missing from
    a month's ranking is sold at once (the bot waits a few runs); no order-size cap or minimum."""
    target, sell_rank = int(rule["target_count"]), int(rule["sell_rank"])
    cap, reserve = int(rule["max_per_sector"]), float(rule["cash_reserve_pct"]) / 100
    cost, halt = float(rule["cost_bps"]) / 10_000, float(rule["drawdown_halt_pct"]) / 100
    pos: dict[str, float] = {}
    cash, peak = start_equity, start_equity
    rows = []
    for m in sorted(inputs):
        if m not in fwd.index:
            continue
        g = inputs[m]
        ranks = dict(zip(g["Ticker"], g["Rank"]))
        sector = dict(zip(g["Ticker"], g["Sector"]))
        equity = cash + sum(pos.values())
        peak = max(peak, equity)
        held_before = set(pos)
        traded = 0.0
        for t in sorted(pos):
            if t not in ranks or ranks[t] > sell_rank:
                v = pos.pop(t)
                cash += v * (1 - cost)
                traded += v
        halted = equity < peak * (1 - halt)
        bought = 0
        if not halted and target - len(pos) > 0:
            slots = target - len(pos)
            counts: dict = {}
            for t in pos:
                counts[sector.get(t)] = counts.get(sector.get(t), 0) + 1
            cands = []
            for t, s, vt, gt in zip(g["Ticker"], g["Sector"], g["Value_Trap_Flag"], g["Growth_Trap_Flag"]):
                if len(cands) == slots:
                    break
                if t in held_before or bool(vt) or bool(gt):
                    continue
                if cap and counts.get(s, 0) >= cap:
                    continue
                counts[s] = counts.get(s, 0) + 1
                cands.append(t)
            if cands:
                each = min(equity * (1 - reserve) / target, max(cash - equity * reserve, 0) / len(cands))
                if each > 0:
                    for t in cands:
                        pos[t] = each * (1 - cost)
                        cash -= each
                        traded += each
                    bought = len(cands)
        r = fwd.loc[m]
        for t in pos:
            x = r.get(t)
            pos[t] *= 1 + (float(x) if pd.notna(x) else 0.0)
        after = cash + sum(pos.values())
        uni = r.reindex(g["Ticker"]).astype(float)
        rows.append({"month": m, "ret": after / equity - 1, "universe": float(uni.mean()),
                     "index": float(r.get("^SP500TR")) if pd.notna(r.get("^SP500TR")) else np.nan,
                     "turnover": traded / 2 / equity, "held": len(pos), "bought": bought,
                     "halted": bool(halted)})
    out = pd.DataFrame(rows).set_index("month")
    out["excess"] = out["ret"] - out["universe"]
    out["excess_index"] = out["ret"] - out["index"]
    return out


def _drawdown(rets: pd.Series) -> float | None:
    rets = rets.dropna()
    if len(rets) < 2:
        return None
    w = (1 + rets).cumprod()
    return round(float((w / w.cummax() - 1).min()), 4)


def summarise_evaluation(pm: pd.DataFrame) -> dict:
    out = {}
    for name, (a, b) in (("discovery", DISCOVERY), ("holdout", HOLDOUT), ("all", SAMPLE),
                         ("presample", PRESAMPLE)):
        sl = pm[(pm.index >= a) & (pm.index <= b)]
        if len(sl) < 3:
            continue
        out[name] = {"ret": _stats(sl["ret"]), "excess": _stats(sl["excess"]),
                     "excess_index": _stats(sl["excess_index"]),
                     "turnover_a_year": round(float(sl["turnover"].mean() * 12), 3),
                     "max_drawdown": _drawdown(sl["ret"]), "universe_max_drawdown": _drawdown(sl["universe"]),
                     "months_halted": int(sl["halted"].sum()), "mean_held": round(float(sl["held"].mean()), 1)}
    return out


def run_evaluations(ids: list[str] | None = None) -> list[dict]:
    evs = load_evaluations()
    if ids:
        unknown = [i for i in ids if i not in {e["id"] for e in evs}]
        if unknown:
            raise SystemExit(f"not registered (register before running): {unknown}")
    months = load_months()
    fwd = forward_returns(sorted(months))
    vintage = data_fingerprint(months)
    cfg0 = base_config()
    inputs = bot_month_inputs(months, cfg0)
    results = []
    for e in evs:
        if ids and e["id"] not in ids:
            continue
        pm = simulate_bot(inputs, fwd, e["rule"])
        res = {"id": e["id"], "kind": "evaluation", "run_at": datetime.now().isoformat(timespec="seconds"),
               "commit": git_commit(), "months": int(len(pm)), **vintage, "summary": summarise_evaluation(pm)}
        with LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(res) + "\n")
        results.append(res)
        for p, s in res["summary"].items():
            print(f"{e['id']} {p}: excess vs universe {s['excess'].get('mean')} (t {s['excess'].get('t')}), "
                  f"turnover {s['turnover_a_year']}/yr, max dd {s['max_drawdown']}", flush=True)
    return results


def _evaluation_lines(latest: dict) -> list[str]:
    evs = load_evaluations()
    if not evs:
        return []
    lines = ["", "## Evaluations - what a construction rule would have done", "",
             "The baseline's rankings, run through a trading rule month by month. Excess is against the "
             "equal-weighted average of every stock ranked that month; *vs index* against the S&P 500 "
             "total-return index. Costs included; no taxes.", "",
             "| Evaluation | Period | Return a month | Excess vs universe (t) | vs index (t) | Turnover a year | Max drawdown (universe) | Months halted |",
             "|---|---|---|---|---|---|---|---|"]
    for e in evs:
        r = latest.get(e["id"])
        if not r:
            lines.append(f"| {e['id']} | not run | | | | | | |")
            continue
        for p in ("discovery", "holdout", "presample"):
            s = r["summary"].get(p)
            if not s:
                continue
            lines.append(f"| {e['id']} | {p} | {s['ret'].get('mean', 0) * 100:+.2f}% | "
                         f"{s['excess'].get('mean', 0) * 100:+.2f}% ({s['excess'].get('t')}) | "
                         f"{s['excess_index'].get('mean', 0) * 100:+.2f}% ({s['excess_index'].get('t')}) | "
                         f"{s['turnover_a_year'] * 100:.0f}% | {s['max_drawdown']} ({s['universe_max_drawdown']}) | "
                         f"{s['months_halted']} |")
    for e in evs:
        lines += ["", f"**{e['id']}** - predicted: {e.get('prediction', '')} Differences from the live rule: "
                  f"{e.get('differences', '')}"]
    return lines


def report() -> Path:
    reg = load_registry()
    latest = {}
    if LEDGER.exists():
        for line in LEDGER.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                latest[r["id"]] = r
    runs = sum(1 for _ in LEDGER.read_text(encoding="utf-8").splitlines()) if LEDGER.exists() else 0
    n = len([e for e in reg if e["transform"]["kind"] != "baseline"])
    vintages = sorted({r.get("data_hash", "before-fingerprint") for r in latest.values()})
    lines = ["# Research engine results", "",
             "Generated by `python v2_experiments.py report` from `ledger.jsonl` - do not edit by hand.",
             "Every experiment was registered, with its prediction, in `registry.yaml` before it ran; the git",
             "history shows when. Judged on the held-out months (2023-07 .. 2026-08) against the current",
             f"weighting on the same months. {n} experiments registered, {runs} runs in the ledger; the",
             f"multiple-testing bar is p < 0.05 / {n} = {0.05 / max(n, 1):.4f} and |t| >= {HLZ_T}.", "",
             "**Not decision-grade** (CLAUDE.md rule 5): " + " ".join(br.CAVEATS[:2]), "",
             "**2012-2019 carries heavier survivorship** - 58% of members present in 2012, 78% in 2019 "
             "(`cache/v2_census_presample.json`); read `research/2026-10-15-research-engine-first-batch.md` "
             "section 4 before any number in the last three columns.", "",
             ("Data: every row below ran on the same monthly records (" + vintages[0] + ")." if len(vintages) == 1
              else "**Data: the rows below ran on more than one version of the monthly records ("
                   + ", ".join(vintages) + ")** - each compares with the baseline of its own run, but "
                   "rows of different versions do not compare with each other. Rerun to bring them level."), "",
             "*sees* is the smallest true IC difference that comparison would detect 80% of the time at the "
             "bar above. A row that is \"not distinguishable\" would usually have caught a difference that "
             "large; it says nothing about smaller ones.", "",
             "| Experiment | Predicted | IC, discovery | IC, held out | Held out vs baseline (t; sees) | Top tenth vs universe, held out | Verdict | IC, 2012-19 | 2012-19 vs baseline (t; sees) | 2012-19 verdict |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for e in reg:
        r = latest.get(e["id"])
        if not r:
            lines.append(f"| {e['id']} | {e.get('prediction', '')} | not run | | | | | | | |")
            continue
        s = r["summary"]
        vs = s["holdout"].get("ic_vs_baseline", {})
        lines.append(f"| {e['id']} | {e.get('prediction', '')} | {s['discovery']['ic'].get('mean', 0):+.3f} | "
                     f"{s['holdout']['ic'].get('mean', 0):+.3f} | "
                     + _vs_cell(vs, n)
                     + f" | {s['holdout']['top_minus_universe'].get('mean', 0) * 100:+.2f}% a month | {verdict(e, r, n)} | "
                     + _presample_cells(s, n) + f" | {verdict_presample(e, r, n)} |")
    lines += ["", "Each row's hypothesis and source are in `registry.yaml`.", ""]
    lines += _evaluation_lines(latest)
    lines += [""]
    RESULTS.write_text("\n".join(lines), encoding="utf-8")
    return RESULTS


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("run", "evaluate", "report"):
        print(__doc__)
        return 2
    if argv[0] == "run":
        run(argv[1:] or None)
    elif argv[0] == "evaluate":
        run_evaluations(argv[1:] or None)
    report()
    print(f"wrote {RESULTS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
