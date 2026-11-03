"""Audit rows 163-172 (plan/claims-mutation-audit.md, 2026-11-02 evening): apply each mutation of the
research page's chart geometry to research_page.py, run the claim's tests, restore."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "research_page.py"
orig = SRC.read_text(encoding="utf-8")
POS = "return max(0.0, min(100.0, (x + LIM) / (2 * LIM) * 100))"
AX = ("<span>{num(-LIM, 2)}</span><span>{num(-LIM / 2, 2)}</span><span>0</span>"
      "<span>{num(LIM / 2, 2)}</span><span>{num(LIM, 2)}</span>")
MUTS = {
    "_pos mirrored (sign error)": (POS, "return max(0.0, min(100.0, (LIM - x) / (2 * LIM) * 100))"),
    "_pos at twice the scale": (POS, "return max(0.0, min(100.0, (x + LIM) / LIM * 50 + 25))"),
    "_pos off-centre": (POS, "return max(0.0, min(100.0, (x + LIM) / (2 * LIM) * 100 + 3))"),
    "_pos unclamped": (POS, "return (x + LIM) / (2 * LIM) * 100"),
    "axis labels at a different L": (AX, AX.replace("LIM", "(LIM * 1.5)")),
    "axis labels reversed": (AX, "<span>{num(LIM, 2)}</span><span>{num(LIM / 2, 2)}</span><span>0</span>"
                                 "<span>{num(-LIM / 2, 2)}</span><span>{num(-LIM, 2)}</span>"),
    "dot at the held-out mean": ('<span class="dot" style="left:{_pos(p["mean"]):.2f}%">',
                                 '<span class="dot" style="left:{_pos(r["h"]["mean"]):.2f}%">'),
    "ticks at bar x mean, not x se": ('left:{_pos(bar * p["se"]):.2f}%', 'left:{_pos(bar * abs(p["mean"])):.2f}%'),
    "band one-sided": ('<span class="see" style="left:{_pos(-d):.2f}%;width:{_pos(d) - _pos(-d):.2f}%">',
                       '<span class="see" style="left:{_pos(0):.2f}%;width:{_pos(d) - _pos(0):.2f}%">'),
    "interval on the normal curve's 1.0": ('q = X.t_critical(0.05, p["n"] - 1) if p.get("n") else 1.96', 'q = 1.0'),
    "word against the 0.05 bar": ('where = "inside the pass bar" if abs(p["t"]) < bar else "beyond the pass bar"',
                                  'where = "inside the pass bar" if abs(p["t"]) < q else "beyond the pass bar"'),
}
TESTS = ["tests/test_research_page.py::test_the_chart_puts_every_mark_where_its_printed_numbers_and_the_axis_say",
         "tests/test_research_page.py::test_the_word_and_the_picture_agree_between_the_two_bars",
         "tests/test_research_page.py::test_a_mark_past_the_axis_is_drawn_at_its_end_not_off_the_track"]
try:
    for name, (a, b) in MUTS.items():
        n = orig.count(a)
        assert n >= 1, (name, n)
        SRC.write_text(orig.replace(a, b), encoding="utf-8")
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS],
                           cwd=ROOT, capture_output=True, text=True)
        last = r.stdout.strip().splitlines()[-1]
        print(f"{name:<36} {'CAUGHT' if r.returncode else 'MISSED'}  ({last})")
finally:
    SRC.write_text(orig, encoding="utf-8")
print("restored:", SRC.read_text(encoding="utf-8") == orig)
