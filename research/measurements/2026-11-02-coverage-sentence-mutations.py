"""Audit rows 174-177 (plan/claims-mutation-audit.md, 2026-11-02 evening): mutate the methodology
page's coverage-count sentence (run_screener.py) and the engine rule it describes (factor_engine.py),
run the claim's test, restore."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = "{len(_w_gen)} for every other ({len(_w_gen) - 1} for other financials"
MUTS = {
    "other financials printed as the generic count": (
        "run_screener.py", GEN, "{len(_w_gen)} for every other ({len(_w_gen)} for other financials"),
    "bank and generic counts swapped": (
        "run_screener.py", "{len(_w_bank)} for a bank-like stock and \"\n        f\"{len(_w_gen)} for every other",
        "{len(_w_gen)} for a bank-like stock and \"\n        f\"{len(_w_bank)} for every other"),
    "financials' count off the bank table": (
        "run_screener.py", GEN, "{len(_w_gen)} for every other ({len(_w_bank) - 1} for other financials"),
    "engine stops excusing financials from Beneish": (
        "factor_engine.py", 'if m == "beneish_m_score" and "_beneish_na" in df.columns:',
        'if False and m == "beneish_m_score" and "_beneish_na" in df.columns:'),
}
TESTS = ["tests/test_claims_register.py::test_the_coverage_sentence_counts_what_the_engine_counts"]
for name, (fname, a, b) in MUTS.items():
    src = ROOT / fname
    orig = src.read_text(encoding="utf-8")
    try:
        assert orig.count(a) == 1, (name, orig.count(a))
        src.write_text(orig.replace(a, b), encoding="utf-8")
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS],
                           cwd=ROOT, capture_output=True, text=True)
        last = r.stdout.strip().splitlines()[-1]
        print(f"{name:<46} {'CAUGHT' if r.returncode else 'MISSED'}  ({last})")
    finally:
        src.write_text(orig, encoding="utf-8")
        assert src.read_text(encoding="utf-8") == orig
print("restored")
