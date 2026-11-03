"""Audit rows 154-162 (plan/claims-mutation-audit.md, 2026-11-02 evening): apply each mutation of the
Size sentence to research_page.py, run the claim's tests, restore. Prints a verdict per mutation."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "research_page.py"
orig = SRC.read_text(encoding="utf-8")
YC = 'year_cov = lambda y: float(np.mean([months_cov[m]["coverage"] for m in ms if m.startswith(y)]))'
MUTS = {
    "word: < becomes <=": ('reach = "short of" if abs(p["t"]) < bar', 'reach = "short of" if abs(p["t"]) <= bar'),
    "word: abs dropped": ('reach = "short of" if abs(p["t"]) < bar', 'reach = "short of" if p["t"] < bar'),
    "word: held-out bar": ("size_coverage_relation(latest, coverage), bar_p)", "size_coverage_relation(latest, coverage), bar_h)"),
    "second row's reading": ("size_html = size_paragraph(rows[0],", "size_html = size_paragraph(rows[1],"),
    "t printed from the held-out": ('(t&nbsp;{num(p["t"], 2)}), {reach}', '(t&nbsp;{num(row["h"]["t"], 2)}), {reach}'),
    "year's first month": (YC, 'year_cov = lambda y: float([months_cov[m]["coverage"] for m in ms if m.startswith(y)][0])'),
    "year's median": (YC, YC.replace("np.mean", "np.median")),
    "every month in the file": (YC, YC.replace("for m in ms if", "for m in months_cov if")),
    "second-last year": ('"last_year": years[-1], "last_cov": year_cov(years[-1])', '"last_year": years[-2], "last_cov": year_cov(years[-2])'),
}
TESTS = ["tests/test_research_page.py::test_the_size_sentence_is_the_top_rows_own_reading_and_its_word_turns_at_the_bar",
         "tests/test_research_page.py::test_the_coverage_percentages_are_the_first_and_last_years_means",
         "tests/test_research_page.py::test_the_page_prints_the_coverage_relation_of_the_committed_files"]
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
