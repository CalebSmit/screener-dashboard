"""Claims-mutation audit, 2026-10-31 (13:00): the methodology page's four remaining prose rows.

`plan/claims-mutation-audit.md`'s next list, after `_pbo_words`: "the three bank P/B-ROE prose
rows, whose residual clause also chooses a **word** on `bank_net_pb < 0` with nothing probing
that boundary, and `coverage-discount-worked-example`, five figures deep and read by none of
them."

Each mutation of `run_screener.py` is run against the **claim's own `checked_by`**, which is
what `scripts/check_published_claims.py` derives and therefore the only coverage that can stop
a publish (audit method step 3). `--before` drops the checks this session added, which is how
the gap is recorded rather than asserted.

    python research/measurements/2026-10-31-overview-prose-mutations.py
    python research/measurements/2026-10-31-overview-prose-mutations.py --before

`run_screener.py` is restored in a ``finally`` and asserted byte-for-byte.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import claims  # noqa: E402

SRC = pathlib.Path("run_screener.py")

ADDED_TODAY = (
    "test_the_five_metric_total_in_the_prose_is_the_sum_of_those_five",
    "test_the_offset_sentence_prints_each_metrics_own_points_and_the_tables_net",
    "test_the_residual_clause_states_the_net_it_measures_and_which_side_it_is_on",
    "test_the_discount_examples_own_stock_is_the_one_the_generator_chooses",
)

BANK = "overview.bank_pb_roe_identity"
COV = "overview.coverage_discount"

# (id, what it breaks, find, replace, claim)
MUTATIONS = [
    ("M", "the five-metric total silently drops the equity ratio",
     "f\"So the {_bp('pb_ratio') + _bp('earnings_yield') + _bp('roe') + _bp('roa') + _bp('equity_ratio'):.1f} \"",
     "f\"So the {_bp('pb_ratio') + _bp('earnings_yield') + _bp('roe') + _bp('roa'):.1f} \"",
     BANK),
    ("N", "the offset sentence prints ROA's points where ROE's belong",
     "f\"{_bp('pb_ratio'):.1f} points are offset by ROE's {_bp('roe'):.1f} and ROA's \"",
     "f\"{_bp('pb_ratio'):.1f} points are offset by ROE's {_bp('roa'):.1f} and ROA's \"",
     BANK),
    ("O", "the residual clause's word turns on the wrong side of zero",
     "{'against' if bank_net_pb < 0 else 'on'}",
     "{'on' if bank_net_pb < 0 else 'against'}",
     BANK),
    ("P", "the residual clause states the net on earnings yield instead of on price-to-book",
     "**{abs(bank_net_pb):.1f} points",
     "**{abs(bank_net_ep):.1f} points",
     BANK),
    ("Q", "the neutrality bar widens from a quarter of the earnings-yield loading to a third",
     "_bank_pb_is_neutral = abs(bank_net_pb) <= 0.25 * bank_net_ep if bank_net_ep else False",
     "_bank_pb_is_neutral = abs(bank_net_pb) <= 0.33 * bank_net_ep if bank_net_ep else False",
     BANK),
    ("R", "the worked example is quoted at the coverage filter's floor instead",
     "_ex_k = next((k for k in range(_n_gen, 0, -1) if k / _n_gen < _thr_frac - 0.05), _bars[_n_gen])",
     "_ex_k = _bars[_n_gen]",
     COV),
    ("S", "the worked example's coverage percentage is taken out of the bank denominator",
     "{_cov_pct(_ex_k, _n_gen)}% coverage",
     "{_cov_pct(_ex_k, _n_bank)}% coverage",
     COV),
    ("T", "the example's margin below the threshold doubles, moving which stock it quotes",
     "if k / _n_gen < _thr_frac - 0.05)",
     "if k / _n_gen < _thr_frac - 0.10)",
     COV),
    # An equivalent mutant, recorded rather than dropped (`plan/claims-mutation-audit.md`,
    # "Equivalent mutants"). Halving the margin cannot move the example at n = 27: the rule
    # takes the largest k with k/27 below the bar, and 0.775 x 27 = 20.9 selects the same k = 20
    # as 0.75 x 27 = 20.3. The page is byte-identical, so nothing can see it and nothing should.
    # It is here because the next session would otherwise pick the same mutation and read the
    # pass as a gap. T above is the version that does change the page.
    ("T2", "[equivalent mutant] the margin halves, which selects the same k at n = 27",
     "if k / _n_gen < _thr_frac - 0.05)",
     "if k / _n_gen < _thr_frac - 0.025)",
     COV),
]


EQUIVALENT = {"T2"}

PRINT_PAGE = (
    "import hashlib, yaml, run_screener as rs;"
    "cfg = yaml.safe_load(open('config.yaml', encoding='utf-8').read());"
    "p = rs.build_screener_overview(cfg);"
    # sha256, not `hash()` - str hashing is salted per interpreter, so `hash()` differs between
    # two runs of the same page and would report every equivalent mutant as a real change
    "print(len(p), hashlib.sha256(p.encode('utf-8')).hexdigest()[:16])"
)


def page_fingerprint() -> str:
    """Length and hash of the generated methodology page, from a fresh interpreter."""
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", PRINT_PAGE],
                       capture_output=True, text=True)
    return r.stdout.strip() or f"(exit {r.returncode}) {r.stderr.strip()[:200]}"


def nodes(claim_id: str, before: bool) -> list[str]:
    claim = next(c for c in claims.CLAIMS if c.id == claim_id)
    out = list(claim.checked_by)
    return [n for n in out if not any(a in n for a in ADDED_TODAY)] if before else out


def run(node_ids: list[str]) -> tuple[bool, str]:
    r = subprocess.run([sys.executable, "-m", "pytest", *node_ids, "-q", "--no-header",
                        "--tb=no", "-rf"], capture_output=True, text=True)
    lines = [ln for ln in r.stdout.splitlines() if " passed" in ln or " failed" in ln]
    out = lines[-1].strip() if lines else f"(no summary, exit {r.returncode})"
    failed = [ln.split()[1] for ln in r.stdout.splitlines() if ln.startswith("FAILED ")]
    if failed:
        out += "\n           failed: " + ", ".join(sorted(set(failed)))
    return bool(failed), out


def main(argv: list[str]) -> int:
    before = "--before" in argv
    only = [a for a in argv if a in {m[0] for m in MUTATIONS}]
    original = SRC.read_text(encoding="utf-8")
    worst = 0
    try:
        print("register state:", "before this session's registration" if before
              else "as committed")
        for mid, what, find, replace, claim_id in MUTATIONS:
            if only and mid not in only:
                continue
            if original.count(find) != 1:
                print(f"SKIPPED {mid}: anchor appears {original.count(find)} times - {what}")
                worst = 1
                continue
            SRC.write_text(original.replace(find, replace), encoding="utf-8")
            if mid in EQUIVALENT:
                got = page_fingerprint()
                SRC.write_text(original, encoding="utf-8")
                clean = page_fingerprint()
                print(f"EQUIV   {mid}: {what}\n           claim {claim_id}\n"
                      f"           page {'unchanged' if got == clean else 'CHANGED - not an '
                                         'equivalent mutant after all'}: {got} vs {clean}")
                continue
            caught, summary = run(nodes(claim_id, before))
            print(f"{'CAUGHT ' if caught else 'GAP    '} {mid}: {what}\n"
                  f"           claim {claim_id}\n           {summary}")
            if not caught and not before:
                worst = 1
            SRC.write_text(original, encoding="utf-8")
    finally:
        SRC.write_text(original, encoding="utf-8")
        assert SRC.read_text(encoding="utf-8") == original, "failed to restore run_screener.py"
        print("run_screener.py restored byte-for-byte")
    return worst


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
