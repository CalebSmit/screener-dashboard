"""Audit rows 148-153 (plan/claims-mutation-audit.md, 2026-11-02 evening): apply each pass-bar
mutation to research_page.py, run the claim's tests, restore. Prints a verdict per mutation."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "research_page.py"
orig = SRC.read_text(encoding="utf-8")
LINE = 'bar_p, bar_h = X.pass_bar(n, mid_p.get("n")), X.pass_bar(n, mid_h.get("n"))'
MUTS = {
    "caption prints bar_h": ("{bar_p:.2f}</span> over these", "{bar_h:.2f}</span> over these"),
    "caption months from held-out": ("over these {mid_p.get('n')} months", "over these {mid_h.get('n')} months"),
    "eras swapped": (LINE, 'bar_p, bar_h = X.pass_bar(n, mid_h.get("n")), X.pass_bar(n, mid_p.get("n"))'),
    "normal curve, not t": (LINE, 'bar_p, bar_h = X.pass_bar(n, None), X.pass_bar(n, None)'),
    "no HLZ floor": (LINE, 'bar_p, bar_h = X.t_critical(0.05 / n, mid_p.get("n") - 1), X.t_critical(0.05 / n, mid_h.get("n") - 1)'),
    "Bonferroni over one test": (LINE, 'bar_p, bar_h = X.pass_bar(1, mid_p.get("n")), X.pass_bar(1, mid_h.get("n"))'),
}
TESTS = ["tests/test_research_page.py::test_the_pass_bar_is_the_engines_bar_for_each_eras_own_months",
         "tests/test_research_page.py::test_the_pass_bar_never_prints_below_three",
         "tests/test_research_page.py::test_the_two_printings_of_the_bar_agree_on_the_real_ledger"]
try:
    for name, (a, b) in MUTS.items():
        assert orig.count(a) == 1, (name, orig.count(a))
        SRC.write_text(orig.replace(a, b), encoding="utf-8")
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS],
                           cwd=ROOT, capture_output=True, text=True)
        last = r.stdout.strip().splitlines()[-1]
        print(f"{name:<30} {'CAUGHT' if r.returncode else 'MISSED'}  ({last})")
finally:
    SRC.write_text(orig, encoding="utf-8")
print("restored:", SRC.read_text(encoding="utf-8") == orig)
