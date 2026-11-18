"""Every way the published page knows an input is impaired, counted on one run - 2026-11-18

Read-only, offline. Reads the payload the site is serving (HEAD, shards folded back) and counts,
for each caveat the page can attach to a stock's *inputs*, how many stocks carry it - in the
universe, in the top 40 (what `signals.json` hands the trading bot) and in the top 25 - and how
the caveats overlap. Trap and manipulation flags are listed separately: they are statements
about the company, not about the data.

Usage:  python research/measurements/2026-11-18-what-the-page-knows-about-its-inputs.py [git-ref]
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import check_run_health as h  # noqa: E402

CATS = ("valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment")


def _num(v):
    return isinstance(v, (int, float)) and v == v


def input_caveats(s: dict) -> list[str]:
    """The names of every input caveat this stock's sheet carries."""
    fl = s.get("flags") or {}
    cov = s.get("cov") or {}
    xk = {k: v for k, v in (s.get("xchk") or {}).items() if _num(v)}
    kinds = {f.get("k") for f in s.get("summary") or []}
    out = []
    if cov.get("of") and cov.get("n", 0) < cov["of"]:
        out.append("metric missing")
    if any(not _num((s.get("cat_scores") or {}).get(c)) for c in CATS):
        out.append("category unscored")
    if fl.get("px_rejected"):
        out.append("price series refused")
    if fl.get("q_rejected"):
        out.append("quarterly statement refused")
    if fl.get("annual_instead"):
        out.append("annual statement instead")
    if fl.get("ltm_rolled"):
        out.append("twelve months rolled forward")
    if fl.get("ltm_annualized"):
        out.append("three quarters scaled up")
    if fl.get("ev_flag"):
        out.append("enterprise value substituted")
    if fl.get("stale_data"):
        out.append("statement flagged stale")
    if s.get("eps_mismatch"):
        out.append("EPS bases disagree")
    if any(abs(v) > 0.02 for v in xk.values()):
        out.append("input over 2% from the filing")
    if any(abs(v) > 0.10 for v in xk.values()):
        out.append("input over 10% from the filing")
    if s.get("newer_filing"):
        out.append("a period behind the filings")
    if s.get("srcn"):
        out.append("source error named (year-to-date column)")
    for m in sorted(s.get("cavs") or {}):
        out.append(f"filing note on {m}")
    if "input_churn" in kinds:
        out.append("rank move is input churn")
    return out


def company_flags(s: dict) -> list[str]:
    fl = s.get("flags") or {}
    return [name for name, on in (("value trap", s.get("vt")), ("growth trap", s.get("gt")),
                                  ("Beneish", fl.get("beneish_flag")),
                                  ("channel stuffing", fl.get("channel_stuffing"))) if on]


def main(ref: str = "HEAD") -> int:
    D = h.published_payload_whole(ref)
    if D is None:
        print(f"no payload at {ref}")
        return 2
    sd = D["stock_detail"]
    rank = {r["Ticker"]: r["Rank"] for r in D["table_data"]}
    n = len(sd)
    print(f"run {D['kpis']['run_timestamp'][:10]} ({ref}): {n} scored, "
          f"{len(D.get('unscored') or [])} index members unscored\n")

    per = {t: input_caveats(s) for t, s in sd.items()}
    for title, fn in (("INPUT CAVEATS", lambda t: per[t]),
                      ("FLAGS ABOUT THE COMPANY", lambda t: company_flags(sd[t]))):
        count, top40, top25 = Counter(), Counter(), Counter()
        for t in sd:
            for k in fn(t):
                count[k] += 1
                top40[k] += rank.get(t, 999) <= 40
                top25[k] += rank.get(t, 999) <= 25
        print(f"{title:44} universe  top 40  top 25")
        for k, c in count.most_common():
            print(f"  {k:42} {c:8}  {top40[k]:6}  {top25[k]:6}")
        print()

    # "over 10%" is a subset of "over 2%"; count a stock once.
    def distinct(t):
        return {k for k in per[t] if k != "input over 10% from the filing"}
    for label, pool in (("universe", list(sd)),
                        ("top 40", [t for t in sd if rank.get(t, 999) <= 40]),
                        ("top 25", [t for t in sd if rank.get(t, 999) <= 25])):
        dist = Counter(min(len(distinct(t)), 4) for t in pool)
        any_ = sum(v for k, v in dist.items() if k)
        print(f"{label:9} {len(pool):3} stocks: {any_} carry at least one input caveat "
              f"(0: {dist[0]}, 1: {dist[1]}, 2: {dist[2]}, 3: {dist[3]}, 4+: {dist[4]})")

    stale = {t for t in sd if "statement flagged stale" in per[t]}
    fallback = {t for t in sd if {"annual statement instead", "quarterly statement refused"} & set(per[t])}
    behind = {t for t in sd if "a period behind the filings" in per[t]}
    print(f"\nstale {len(stale)}; annual fallback (either kind) {len(fallback)}; "
          f"stale and fallback {len(stale & fallback)}; stale without a fallback {len(stale - fallback)}")
    print(f"a period behind the filings {len(behind)}; of the stale, also 'a period behind' "
          f"{len(stale & behind)}")
    print("\ntop 40 with an input over 10% from the filing, a named source error, or a fallback:")
    for t in sorted(sd, key=lambda x: rank.get(x, 999)):
        if rank.get(t, 999) > 40:
            continue
        hit = [k for k in per[t] if k in ("input over 10% from the filing",
                                           "source error named (year-to-date column)",
                                           "annual statement instead", "quarterly statement refused",
                                           "twelve months rolled forward", "three quarters scaled up")]
        if hit:
            xk = {k: round(v, 3) for k, v in (sd[t].get("xchk") or {}).items() if _num(v) and abs(v) > 0.10}
            print(f"  #{rank[t]:<3} {t:6} {hit} {xk or ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:2]))
