"""Second reading of the proposal in ``research/2026-10-22-free-data-that-adds-evidence.md``.

The proposal says: *"register tests A, B and C (section 4) in
``research/experiments/registry.yaml``, each in its own commit before any of its data is
built."* Its three YAML blocks are written "in the registry's style".

This asks whether the registry can actually hold them, by executing the engine's own code
paths against the note's entries rather than reading the YAML and judging by eye. And it
computes, for each of the two entries that state a power figure, the probability that the
entry's own pre-registered verdict bands fire under the null the entry predicts - i.e.
whether the pre-registration is falsifiable as written.

Read-only: it monkeypatches a registry in memory and never writes ``registry.yaml``.
Decision-grade for nothing about any weight (CLAUDE.md rule 5); it decides only whether the
proposed registrations are executable and well-formed.

    python -I -X utf8 research/measurements/2026-10-23-can-the-registry-hold-these-tests.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import v2_experiments as X                                            # noqa: E402

OUT = Path(__file__).with_suffix(".json")

# ---------------------------------------------------------------------------
# The note's three entries, reduced to the fields that decide whether the engine
# can execute them. Verbatim ids; no transform, because none of the three states one.
# ---------------------------------------------------------------------------
NOTE_ENTRIES = [
    {"id": "size-survivorship-13f-quarterly", "section": "experiments",
     "fields": ["id", "hypothesis", "source", "data", "statistic", "prediction",
                "refuted_if", "void_if", "differences"]},
    {"id": "short-interest-dtc-finra", "section": "experiments",
     "fields": ["id", "hypothesis", "source", "data", "statistic", "prediction",
                "refuted_if", "power"]},
    {"id": "phasein", "section": "(a top-level block, not a list entry)",
     "fields": ["months", "hypothesis", "source", "data",
                "outcome_known_at_registration", "prediction", "refuted_if", "power"]},
]


def what_the_registry_holds() -> dict:
    """The shape of each section the registry actually has, from the committed file."""
    reg = X.load_registry()
    ev = X.load_evaluations()
    return {
        "experiments": {
            "count": len(reg),
            "every_entry_has_transform": all("transform" in e for e in reg),
            "transform_kinds": sorted({e["transform"]["kind"] for e in reg}),
            "keys_of_first": sorted(reg[0].keys()),
        },
        "evaluations": {
            "count": len(ev),
            "every_entry_has_rule": all("rule" in e for e in ev),
            "every_entry_has_transform": all("transform" in e for e in ev),
            "keys_of_first": sorted(ev[0].keys()),
        },
    }


def does_the_engine_accept(entry: dict) -> dict:
    """Try the engine's own code paths on an entry with no ``transform``, as the note's are.

    Each probe is the real call site, not a reimplementation of it.
    """
    probes: dict[str, str] = {}
    note_entry = {k: f"<{k} from the note>" for k in entry["fields"]}
    note_entry["id"] = entry["id"]

    # 1. apply_transform - what run() calls for every entry it executes.
    try:
        X.apply_transform({}, note_entry.get("transform") or {})
        probes["apply_transform"] = "accepted (no error)"
    except Exception as e:                                            # noqa: BLE001
        probes["apply_transform"] = f"{type(e).__name__}: {e}"

    # 2. statistic_keys - how a result is matched to a measurement.
    try:
        X.statistic_keys(note_entry.get("transform") or {})
        probes["statistic_keys"] = "accepted (no error)"
    except Exception as e:                                            # noqa: BLE001
        probes["statistic_keys"] = f"{type(e).__name__}: {e}"

    # 3. The unguarded reads in report() / summary_for_page() and the calibration test:
    #    they count entries with  e["transform"]["kind"] != "baseline".
    try:
        _ = note_entry["transform"]["kind"] != "baseline"
        probes["transform_kind_read"] = "accepted (no error)"
    except Exception as e:                                            # noqa: BLE001
        probes["transform_kind_read"] = f"{type(e).__name__}: {e}"

    # 4. Would v2 even carry the input? short_interest_ratio is zeroed by v2_config and
    #    NaN-ed by the record builder, so a transform could not weight it.
    import backtest_v2 as b2
    import pit_record as pr
    probes["short_interest_in_UNMEASURABLE_METRICS"] = str(
        "short_interest_ratio" in b2.UNMEASURABLE_METRICS)
    probes["short_interest_fields_in_no_free_history"] = str(
        [f for f in pr.FIELD_CLASSES["no_free_history"] if "short" in f.lower()])
    return {"id": entry["id"], "section_the_note_implies": entry["section"], "probes": probes}


# ---------------------------------------------------------------------------
# Are the entries' own verdict bands falsifiable?
# ---------------------------------------------------------------------------
def _p_two_sided_beyond(z: float) -> float:
    """P(|Z| >= z) for a standard normal."""
    return math.erfc(z / math.sqrt(2.0))


def verdict_bands() -> dict:
    """For each entry that states a power figure, the probability its own bands fire
    when the true IC is zero - the null each entry predicts."""
    out = {}

    # Test B: SE ~ 0.010-0.012 over ~104 months (the entry's own 'power' field).
    # prediction : mean IC positive but < 0.02 and |t| < 2
    # refuted_if : "mean IC <= 0 at the point estimate"
    for se in (0.010, 0.011, 0.012):
        # Under a true IC of 0 the point estimate is <= 0 with probability 1/2, by symmetry.
        p_refuted = 0.5
        # P(0 < IC < 0.02 and |t| < 2) = P(0 < z < min(0.02/se, 2))
        z_hi = min(0.02 / se, 2.0)
        p_confirm = 0.5 * (1.0 - _p_two_sided_beyond(z_hi))
        out.setdefault("B_short_interest_dtc_finra", []).append({
            "se": se,
            "p_prediction_holds_under_null": round(p_confirm, 4),
            "p_refuted_if_fires_under_null": round(p_refuted, 4),
            "note": "refuted_if is a condition on the SIGN of the point estimate, so under the "
                    "entry's own predicted null it fires on a coin flip",
        })

    # Test C: SE ~ 0.023 over 22 months (the entry's own 'power' field).
    # prediction : |IC| < 0.03 and |t| < 2   -> binding: |t| < 0.03/se
    # refuted_if : |IC| >= 0.05 and |t| >= 2 -> binding: |t| >= 0.05/se
    for se in (0.021, 0.023):
        z_conf = min(0.03 / se, 2.0)
        z_ref = max(0.05 / se, 2.0)
        p_confirm = 1.0 - _p_two_sided_beyond(z_conf)
        p_refute = _p_two_sided_beyond(z_ref)
        out.setdefault("C_phasein", []).append({
            "se": se,
            "confirm_band_t": round(z_conf, 3),
            "refute_band_t": round(z_ref, 3),
            "p_prediction_holds_under_null": round(p_confirm, 4),
            "p_refuted_if_fires_under_null": round(p_refute, 4),
            "p_neither_under_null": round(1.0 - p_confirm - p_refute, 4),
        })

    # What 22 extra months buy the 2012-2019 era, at the engine's own IC spread.
    spread = 0.11
    for n0, extra in ((96, 22),):
        se0 = spread / math.sqrt(n0)
        se1 = spread / math.sqrt(n0 + extra)
        out["C_precision_gain"] = {
            "months_now": n0, "months_added": extra, "ic_spread_assumed": spread,
            "se_now": round(se0, 5), "se_with_phasein": round(se1, 5),
            "precision_gain_pct": round(100.0 * (1.0 - se1 / se0), 1),
            "smallest_ic_at_t3_now": round(3 * se0, 4),
            "smallest_ic_at_t3_with_phasein": round(3 * se1, 4),
        }
    return out


def main() -> int:
    result = {
        "registry_as_committed": what_the_registry_holds(),
        "engine_probes": [does_the_engine_accept(e) for e in NOTE_ENTRIES],
        "verdict_bands_under_the_null": verdict_bands(),
    }
    print(json.dumps(result, indent=2))
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
