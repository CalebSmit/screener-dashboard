"""If you had picked the best backtest, would it have held up? The research engine's overfitting check.

WHY THIS EXISTS - 2026-10-20, owner: "should we be running more backtests, like all the backtests we
possibly can to decide what is best moving forward for the methodology, weightings, etc?" The honest
answer is no, and this module is the measurement behind it. Choosing the best of many backtests on
one history is the textbook way to fit noise: the winner is partly the best idea and partly the
luckiest. Bailey, Borwein, Lopez de Prado & Zhu (2017), "The probability of backtest overfitting",
Journal of Computational Finance 20(4), give a direct test of how much of a winner is luck -
combinatorially symmetric cross-validation (CSCV):

1. Split the months into ``S`` contiguous blocks (16 here, 11 months each over 2012-2026).
2. For every way of choosing half of the blocks as the "in-sample" history (C(16, 8) = 12,870),
   pick the configuration that did best on that half, then see where it ranks on the other half.
3. The **probability of backtest overfitting (PBO)** is the share of splits in which the in-sample
   winner ranks in the *bottom half* out of sample. If the differences between configurations are
   pure noise, the winner's out-of-sample rank is a coin toss and PBO is about 0.5; if one is
   genuinely better, it keeps winning and PBO falls toward 0.

The engine already guards each single comparison (pre-registration, a held-out period, Bonferroni and
t >= 3). This asks the question the owner asked, of the whole set at once: would "run everything and
keep the winner" have chosen well? It reads only the committed ledger - each experiment's monthly IC
and the current weighting's IC on the same stocks that month (method version 2) - so it reruns
nothing, and it is deliberately **not** in ``v2_experiments.RESULT_PATHS``: it cannot change a ledger
row, and editing it must not mark 30 rows stale for a 47-minute rerun that would reproduce them.

What each configuration is scored on, month by month: its IC minus the current weighting's IC over
the stocks both ranked (``ic_by_month - base_ic_by_month``); the current weighting itself scores 0.
That is the engine's own currency - a difference in ranking, not in which companies were ranked.

Never imported by production code; never sets a weight (CLAUDE.md rules 4 and 5).
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BLOCKS = 16                 # Bailey et al. (2017) use an even S; 16 gives 12,870 splits of 11 months a block
CURRENT = "baseline"


def candidates(reg: list, latest: dict) -> tuple[pd.DataFrame, list[str]]:
    """Months x configurations: each one's monthly IC lead over the current weighting on the stocks
    both ranked (the current weighting is the zero column). Only rows the engine would compare with
    its baseline - same records, same method, same provenance, not void - and only months every
    candidate covers. Returns the frame and the ids left out, with the reason."""
    import v2_experiments as X        # the engine's own comparability rule; research tooling only
    base = latest.get(CURRENT)
    if not base:
        return pd.DataFrame(), ["no baseline row"]
    cols, skipped = {}, []
    for e in reg:
        i = e["id"]
        r = latest.get(i)
        if i == CURRENT:
            continue
        if e.get("void"):
            skipped.append(f"{i}: void")
            continue
        if not r:
            skipped.append(f"{i}: not run")
            continue
        if r.get("data_hash") != base.get("data_hash") \
                or r.get("method_version", 1) != base.get("method_version", 1) \
                or not X.comparable_provenance(r, base):
            skipped.append(f"{i}: different records, method or provenance")
            continue
        ic, b = pd.Series(r.get("ic_by_month") or {}, dtype=float), \
            pd.Series(r.get("base_ic_by_month") or {}, dtype=float)
        d = (ic - b).dropna()
        if d.empty:
            skipped.append(f"{i}: no paired months")
            continue
        cols[i] = d
    if not cols:
        return pd.DataFrame(), skipped
    frame = pd.DataFrame(cols).dropna()                 # months every candidate covers
    frame.insert(0, CURRENT, 0.0)
    return frame.sort_index(), skipped


def cscv(perf: pd.DataFrame, blocks: int = BLOCKS) -> dict:
    """Combinatorially symmetric cross-validation (Bailey et al. 2017, section 2) on a months x
    configurations frame of a per-month score. The selection rule is the mean over the in-sample
    months - the same statistic the engine reports."""
    t, n = perf.shape
    if n < 2 or t < 2 * blocks:
        raise ValueError(f"need at least 2 configurations and {2 * blocks} months; got {n} and {t}")
    x = perf.to_numpy(float)
    edges = np.linspace(0, t, blocks + 1).round().astype(int)
    sums = np.stack([x[a:b].sum(axis=0) for a, b in zip(edges[:-1], edges[1:])])    # blocks x n
    counts = np.diff(edges).astype(float)
    total_sum, total_n = sums.sum(axis=0), counts.sum()
    names = list(perf.columns)
    wins = np.zeros(n, int)
    logits, oos_pct, w_is, w_oos, w_lead = [], [], [], [], []
    for combo in itertools.combinations(range(blocks), blocks // 2):
        idx = list(combo)
        s_in, n_in = sums[idx].sum(axis=0), counts[idx].sum()
        is_mean = s_in / n_in
        oos_mean = (total_sum - s_in) / (total_n - n_in)
        k = int(np.argmax(is_mean))
        wins[k] += 1
        # the winner's out-of-sample rank among all n, 1 = worst, ties averaged; omega in (0, 1)
        rank = pd.Series(oos_mean).rank(method="average").iloc[k]
        omega = rank / (n + 1)
        logits.append(np.log(omega / (1 - omega)))
        oos_pct.append(omega)
        w_is.append(is_mean[k])
        w_oos.append(oos_mean[k])
        w_lead.append(oos_mean[k] - oos_mean[names.index(CURRENT)] if CURRENT in names else np.nan)
    logits, w_lead = np.array(logits), np.array(w_lead)
    order = np.argsort(-wins)
    splits = len(logits)
    return {
        "configurations": n, "months": t, "blocks": blocks, "splits": splits,
        "first_month": str(perf.index[0]), "last_month": str(perf.index[-1]),
        "pbo": round(float((logits <= 0).mean()), 4),
        "winner_median_oos_percentile": round(float(np.median(oos_pct)), 4),
        "winner_mean_in_sample": round(float(np.mean(w_is)), 5),
        "winner_mean_out_of_sample": round(float(np.mean(w_oos)), 5),
        "winner_beat_current_out_of_sample": round(float((w_lead > 0).mean()), 4),
        "winner_was_current": round(float(wins[names.index(CURRENT)] / splits), 4) if CURRENT in names else None,
        "top_winners": [{"id": names[i], "share": round(float(wins[i] / splits), 4)}
                        for i in order[:3] if wins[i]],
    }


def check(reg: list, latest: dict) -> dict | None:
    """The check on the ledger's latest rows, three ways; ``None`` when there is too little to run it.

    ``all``      every month the ledger has (2012-2026), 16 blocks.
    ``recent``   the engine's own sample, 2020-2026, 8 blocks - the era with the smaller survivorship
                 gap (``cache/v2_census*.json``), on its own.
    ``without``  all months, with the configuration that most often won in-sample left out.

    Why three: **a low PBO measures consistency, not cause.** A screen that leans on a bias in the
    history - here, which companies the record is missing - wins consistently and reads as a
    genuine difference. Measured on 2026-10-20: PBO 0.21 over 2012-2026, almost all of it Size
    alone (68% of in-sample wins, its lead concentrated in the heavy-survivorship 2012-2019 and
    negative in the held-out months); 0.73 on 2020-2026; 0.63 without Size
    (``research/2026-10-20-backtest-overfitting.md``). Reporting only the first would have told the
    owner the opposite of what the ledger shows."""
    import v2_experiments as X
    perf, skipped = candidates(reg, latest)
    if perf.empty or perf.shape[1] < 3 or len(perf) < 2 * BLOCKS:
        return None
    out = {"all": cscv(perf), "left_out": skipped}
    recent = perf.loc[X.SAMPLE[0]:X.SAMPLE[1]]
    if len(recent) >= 16:
        out["recent"] = cscv(recent, blocks=8)
    top = out["all"]["top_winners"][0]["id"] if out["all"]["top_winners"] else None
    if top and top != CURRENT and perf.shape[1] > 3:
        out["without"] = cscv(perf.drop(columns=[top])) | {"dropped": top}
    return out


def _latest() -> tuple[list, dict]:
    import v2_experiments as X
    latest = {}
    for line in X.LEDGER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[r["id"]] = r
    return X.load_registry(), latest


def main() -> int:
    reg, latest = _latest()
    print(json.dumps(check(reg, latest), indent=1))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
