"""Claims-mutation audit, 2026-10-31 (13:00): the five `pbo-*` rows of the research page.

`plan/claims-mutation-audit.md`'s next list names `_pbo_words` at its own bars, 0.45 and
0.55, as the cheapest remaining `open:` row. Closing it means closing the sentences it sits
in, so this covers all five rows of "If you had kept the best backtest": the three prose
PBOs and their words, the two out-of-sample shares, the count of alternatives, and the top
winner's two era leads with the clause that depends on their signs.

Each mutation is run two ways, which answer two different questions:

* against the **claim's own `checked_by`** (`--scope claim`, the default) - what the data
  loop's publish gate derives, which is the only coverage that stops a publish;
* against the **whole suite** with the new checks deselected (`--scope rest`) - what
  anything that existed before today would have seen. The 2026-10-30 correction applies:
  a drift tripwire ("the committed page is what the generator makes") and the sentence
  sweep ("this sentence has no verdict") detect *staleness* and *an edit*, not a wrong
  number, so they are named rather than counted.

    python research/measurements/2026-10-31-pbo-prose-mutations.py
    python research/measurements/2026-10-31-pbo-prose-mutations.py --scope rest E F

`research_page.py` is restored in a ``finally`` and asserted byte-for-byte.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import claims  # noqa: E402

SRC = pathlib.Path("research_page.py")
CLAIM = "research.overfitting_reading"

NEW_CHECKS = (
    "tests/test_overfitting.py::test_each_prose_pbo_carries_the_rows_own_reading_and_its_word",
    "tests/test_overfitting.py::test_the_out_of_sample_shares_and_the_configuration_count_are_the_rows_own",
    "tests/test_overfitting.py::test_the_top_winners_two_era_leads_and_the_clause_that_depends_on_their_signs",
    "tests/test_overfitting.py::test_the_word_beside_each_pbo_turns_at_its_own_bars",
)

# (id, what it breaks, find, replace)
MUTATIONS = [
    ("E", "the 'worse than a coin toss' bar moves 0.55 -> 0.65",
     "    if pbo >= 0.55:", "    if pbo >= 0.65:"),
    ("F", "the 'better than luck' bar moves 0.45 -> 0.25",
     "    if pbo > 0.45:", "    if pbo > 0.25:"),
    ("G", "the recent-era sentence prints the all-years PBO",
     'f\'<span class="num">{rec["pbo"]:.2f}</span>, {_pbo_words(rec["pbo"])}, and the winner beat the \'',
     'f\'<span class="num">{a["pbo"]:.2f}</span>, {_pbo_words(rec["pbo"])}, and the winner beat the \''),
    ("H", "the all-years share becomes the winner's median out-of-sample percentile",
     'in <span class="num">{a["winner_beat_current_out_of_sample"] * 100:.0f}%</span> \'',
     'in <span class="num">{a["winner_median_oos_percentile"] * 100:.0f}%</span> \''),
    ("I", "the count of alternatives stops excluding the current weighting",
     "the {a['configurations'] - 1} alternatives above",
     "the {a['configurations']} alternatives above"),
    ("J", "the 'fell behind' clause fires on a held-out lead of exactly zero",
     'if hold["mean"] < 0 < pre["mean"] else "."',
     'if hold["mean"] <= 0 <= pre["mean"] else "."'),
    ("K", "the top winner's two era leads are printed the wrong way round",
     'was <span class=\\"num\\">{num(pre[\'mean\'])}</span> over "\n                     f"2012&ndash;2019 and <span class=\\"num\\">{num(hold[\'mean\'])}</span>',
     'was <span class=\\"num\\">{num(hold[\'mean\'])}</span> over "\n                     f"2012&ndash;2019 and <span class=\\"num\\">{num(pre[\'mean\'])}</span>'),
    ("L", "the without-winner sentence drops two decimals to one",
     '{wo["pbo"]:.2f}</span>, {_pbo_words(wo["pbo"])}.',
     '{wo["pbo"]:.1f}</span>, {_pbo_words(wo["pbo"])}.'),
]


def nodes() -> list[str]:
    claim = next(c for c in claims.CLAIMS if c.id == CLAIM)
    return list(claim.checked_by)


def run(scope: str) -> tuple[bool, str]:
    if scope == "claim":
        cmd = [sys.executable, "-m", "pytest", *nodes(), "-q", "--no-header", "--tb=no", "-rf"]
    else:
        cmd = [sys.executable, "scripts/run_tests.py", "tests/", "test_screener.py", "-q",
               "--tb=no", "-rf"]
        for n in NEW_CHECKS:
            cmd += ["--deselect", n]
    r = subprocess.run(cmd, capture_output=True, text=True)
    lines = [ln for ln in r.stdout.splitlines() if " passed" in ln or " failed" in ln]
    out = lines[-1].strip() if lines else f"(no summary, exit {r.returncode})"
    failed = [ln.split()[1] for ln in r.stdout.splitlines() if ln.startswith("FAILED ")]
    if failed:
        out += "\n           failed: " + ", ".join(sorted(set(failed)))
    return bool(failed), out


def main(argv: list[str]) -> int:
    scope = "rest" if "rest" in argv else "claim"
    only = [a for a in argv if a in {m[0] for m in MUTATIONS}]
    original = SRC.read_text(encoding="utf-8")
    worst = 0
    try:
        print("scope:", "the claim's own checked_by (what the publish gate runs)"
              if scope == "claim" else
              "the whole suite, today's four new checks deselected")
        for mid, what, find, replace in MUTATIONS:
            if only and mid not in only:
                continue
            if original.count(find) != 1:
                print(f"SKIPPED {mid}: anchor appears {original.count(find)} times - {what}")
                worst = 1
                continue
            mutated = original.replace(find, replace)
            assert mutated != original
            SRC.write_text(mutated, encoding="utf-8")
            caught, summary = run(scope)
            tag = ("CAUGHT " if caught else "GAP    ") if scope == "claim" else \
                  ("CAUGHT " if caught else "BLIND  ")
            print(f"{tag} {mid}: {what}\n           {summary}")
            if not caught and scope == "claim":
                worst = 1
            SRC.write_text(original, encoding="utf-8")
    finally:
        SRC.write_text(original, encoding="utf-8")
        assert SRC.read_text(encoding="utf-8") == original, "failed to restore research_page.py"
        print("research_page.py restored byte-for-byte")
    return worst


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
