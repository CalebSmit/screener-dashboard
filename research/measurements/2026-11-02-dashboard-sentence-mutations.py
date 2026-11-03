"""Audit rows 178-188 (plan/claims-mutation-audit.md, 2026-11-02 owner-run night): the dashboard
sweep's own teeth, and the Concentration block's sentences (`dashboard.holdings_concentration`).

Each mutation is one exact single-occurrence replacement in a source file. The concentration
mutations run **the claim's own `checked_by`** (what `scripts/check_published_claims.py` derives),
and with `--before` the same list minus the check added that night - which is how a row is
recorded as a gap rather than asserted to be one. The sweep mutations run the sweep's test.
The source is restored in `finally` and compared byte for byte.

    python research/measurements/2026-11-02-dashboard-sentence-mutations.py [--before]
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import claims  # noqa: E402

NEW = "tests/test_holdings_concentration.py::test_the_concentration_sentences_print_their_own_figures"
CLAIM = [c for c in claims.CLAIMS if c.id == "dashboard.holdings_concentration"][0].checked_by
if "--before" in sys.argv:
    CLAIM = tuple(t for t in CLAIM if t != NEW)
SWEEP = ("tests/test_claims_register.py::test_every_computed_sentence_on_the_dashboard_has_a_verdict",)
GD = "generate_dashboard.py"

MUTS = [
    # --- the Concentration block, against the claim's checked_by --------------------------
    ("178 the count beside the split is n + 1", GD,
     "'An equal split across ' + n + ' names", "'An equal split across ' + (n + 1) + ' names", CLAIM),
    ("179 the split is over n + 1 names", GD,
     "(100 / n).toFixed(1)", "(100 / (n + 1)).toFixed(1)", CLAIM),
    ("180 the ratio's second printing doubled", GD,
     "ratio.toFixed(1) + '&times; as much as '", "(ratio * 2).toFixed(1) + '&times; as much as '", CLAIM),
    ("181 the ratio's second printing to a whole number", GD,
     "ratio.toFixed(1) + '&times; as much as '", "ratio.toFixed(0) + '&times; as much as '", CLAIM),
    ("182 the second sentence names the calmer stock as the mover", GD,
     "'equal dollar amounts, ' + escapeHtml(hi.t)", "'equal dollar amounts, ' + escapeHtml(lo.t)", CLAIM),
    ("183 the sector percentile instead of raw volatility", GD,
     "return {{t: r.t, v: (r.s.raw || {{}}).volatility}}", "return {{t: r.t, v: (r.s.pct || {{}}).volatility}}", CLAIM),
    ("184 the list sorted the wrong way (extremes swapped)", GD,
     "{{ return b.v - a.v; }}", "{{ return a.v - b.v; }}", CLAIM),
    # --- the sweep's own teeth --------------------------------------------------------------
    ("185 a NEW computed sentence in the emitted JavaScript", GD,
     "+ '. This is the past year measured, not a forecast.</p>'",
     "+ '. This is the past year measured, not a forecast.</p>' + '<p>Held for ' + (n * 21) + ' trading days on this list.</p>'",
     SWEEP),
    ("186 a registered sentence reworded", GD,
     "'%</strong> annualised volatility against <strong>'", "'%</strong> annual volatility against <strong>'", SWEEP),
    ("187 the sweep stops seeing toFixed (would pass vacuously)", "tests/js_sweep.py",
     '_NUMBER_WORDS = frozenset({"toFixed", ', '_NUMBER_WORDS = frozenset({', SWEEP),
    ("188 a ternary's condition counted as a printed figure", "tests/js_sweep.py",
     "    branches = _split_ternary(toks)", "    branches = None", SWEEP),
]

print(f"checked_by: {len(CLAIM)} nodes{' (--before: the new check dropped)' if '--before' in sys.argv else ''}")
for name, fname, a, b, tests in MUTS:
    src = ROOT / fname
    orig = src.read_bytes()
    text = orig.decode("utf-8")
    try:
        assert text.count(a) == 1, (name, text.count(a))
        src.write_bytes(text.replace(a, b).encode("utf-8"))
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests],
                           cwd=ROOT, capture_output=True, text=True)
        last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
        print(f"{name:<62} {'CAUGHT' if r.returncode else 'MISSED'}  ({last})")
    finally:
        src.write_bytes(orig)
        assert src.read_bytes() == orig
print("restored")
