"""`SCREENER_OVERVIEW.md` is a generated file, and its claims must hold in the
*generator*.

The 2026-09-25 session found four false statements on the public methodology
page and corrected all four by editing `SCREENER_OVERVIEW.md`. It shipped with
14 new tests, every one of them green, and the live site was correct for three
days.

Then the 2026-09-28 02:00 data run reverted all four and published them.

`run_screener.py` step 11 calls `generate_screener_overview()`, which
**overwrites the file from a template on every full run**. `CLAUDE.md` rule 10
listed `dashboard.html`, `index.html` and `dashboard_data.js` as generated and
did not list this one; "Where things live" filed it under hand-maintained
public docs. So the correction went into the output, the tests read the output,
and the generator kept its original text - including an empty "What It
Measures" cell for `fy1_revision_3m`, the heaviest metric in its category,
which the generator had been emitting since the 2026-09-10 reweight because no
label or description was ever added for it.

These tests assert three things the artifact-reading tests structurally cannot:

1. **The generator's metric dictionaries cover the configured metrics.** This
   fails on the 2026-09-10 tree - the day the defect was created - not 15 days
   later once someone read the page.
2. **The claims hold in freshly generated text**, so a regeneration cannot
   reintroduce a corrected falsehood.
3. **The committed file equals the generator's output.** This is the tripwire
   that fires on a hand-edit, pointing the next session at the generator
   instead of letting it write a correction that a data run will silently undo.

Nothing here writes to `SCREENER_OVERVIEW.md`; the build is called directly.
"""

import re
from pathlib import Path

import pytest
import yaml

import run_screener

ROOT = Path(__file__).resolve().parent.parent
OVERVIEW = ROOT / "SCREENER_OVERVIEW.md"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def generated(cfg) -> str:
    """The document the next data run will publish."""
    return run_screener.build_screener_overview(cfg)


def _configured_metrics(cfg: dict) -> dict:
    """Every metric carrying non-zero weight, mapped to its category.

    Zero-weight metrics are excluded because `_metric_table` skips them, so an
    unlabelled one cannot reach the page. `debt_equity` is the live example: it
    sits in the config at weight 0 for schema compatibility.
    """
    out = {}
    for category, weights in (cfg.get("metric_weights") or {}).items():
        if not isinstance(weights, dict):
            continue
        for metric, weight in weights.items():
            if weight:
                out[metric] = category
    return out


def _all_descriptions() -> dict:
    merged = {}
    for name in dir(run_screener):
        if name.endswith("_DESCRIPTIONS"):
            value = getattr(run_screener, name)
            if isinstance(value, dict):
                merged.update(value)
    return merged


def _table_rows(text: str):
    for n, line in enumerate(text.splitlines(), 1):
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or set("".join(cells)) <= set("-: "):
            continue
        yield n, cells


def _section(text: str, heading: str) -> str:
    match = re.search(
        rf"^#{{1,3}}\s.*{re.escape(heading)}.*$", text, re.M | re.I)
    assert match, f"no section heading matching {heading!r}"
    rest = text[match.end():]
    nxt = re.search(r"^#{1,3}\s", rest, re.M)
    return rest[: nxt.start()] if nxt else rest


class TestGeneratorCoversEveryConfiguredMetric:
    """The root cause. A metric given weight but no prose is invisible to the
    author and blank to the reader."""

    def test_every_weighted_metric_has_a_label(self, cfg):
        missing = sorted(
            m for m in _configured_metrics(cfg)
            if m not in run_screener._METRIC_LABELS
        )
        assert not missing, (
            f"metrics carry weight but have no human label, so the table will "
            f"print the raw config key: {missing}"
        )

    def test_every_weighted_metric_has_a_description(self, cfg):
        descriptions = _all_descriptions()
        missing = sorted(
            m for m in _configured_metrics(cfg) if not descriptions.get(m)
        )
        assert not missing, (
            f"metrics carry weight but have no description, so the "
            f"'What It Measures' cell will be empty: {missing}"
        )

    def test_fy1_revision_is_covered(self, cfg):
        """The specific metric this defect was made of - heaviest in Revisions
        since 2026-09-10, blank on the page until 2026-09-28."""
        metrics = _configured_metrics(cfg)
        if "fy1_revision_3m" not in metrics:
            pytest.skip("fy1_revision_3m is not weighted in this config")
        assert run_screener._METRIC_LABELS.get("fy1_revision_3m")
        assert _all_descriptions().get("fy1_revision_3m")


class TestGeneratedTablesAreComplete:
    def test_no_empty_cell_in_any_generated_table(self, generated):
        blank = [
            (n, cells) for n, cells in _table_rows(generated) if any(
                c == "" for c in cells)
        ]
        assert not blank, (
            f"generated tables contain empty cells: "
            f"{[(n, cells[0]) for n, cells in blank][:5]}"
        )

    def test_no_generated_row_is_labelled_with_a_raw_config_key(
            self, generated, cfg):
        keys = set(_configured_metrics(cfg))
        offenders = []
        for n, cells in _table_rows(generated):
            label = cells[0].strip("* ")
            if label in keys:
                offenders.append((n, label))
        assert not offenders, (
            f"rows labelled with raw snake_case config keys: {offenders}"
        )


class TestGeneratedRevisionsProseMatchesConfig:
    """All three Revisions falsehoods, asserted where they are written."""

    def test_the_metric_named_heaviest_actually_is(self, generated, cfg):
        rev = (cfg.get("metric_weights") or {}).get("revisions") or {}
        live = {m: w for m, w in rev.items() if w}
        if not live:
            pytest.skip("no weighted revisions metrics")
        heaviest = max(live, key=lambda m: live[m])
        label = run_screener._METRIC_LABELS.get(heaviest, heaviest)
        section = _section(generated, "Analyst Revisions")
        # `[^*]` so the opening `**` cannot be borrowed from the preceding
        # `**Why these?**`, which swallows the whole sentence.
        match = re.search(
            r"\*\*([^*]+?) gets the highest weight\*\*", section)
        assert match, "the category no longer names its heaviest metric"
        assert match.group(1) == label, (
            f"prose names {match.group(1)!r} as heaviest; config says "
            f"{label!r} at {live[heaviest]}%"
        )

    def test_does_not_call_a_live_metric_infeasible(self, generated, cfg):
        """Limitation 5 and the category note both proposed FY1 revisions as a
        future FactSet/Refinitiv enhancement for 15 days after it shipped."""
        rev = (cfg.get("metric_weights") or {}).get("revisions") or {}
        if not rev.get("fy1_revision_3m"):
            pytest.skip("fy1_revision_3m is not weighted in this config")
        for phrase in (
            "not feasible with yfinance",
            "No EPS revision data",
            "cannot include the single most powerful revisions signal",
        ):
            assert phrase not in generated, (
                f"generated page calls a live, top-weighted metric "
                f"unavailable: {phrase!r}"
            )

    def test_the_real_residual_limit_is_still_stated(self, generated):
        """The honest limit is depth, not absence - 90 days of estimate
        history against the six-month window CJL measured."""
        assert "90 days" in generated
        assert "Chan, Jegadeesh & Lakonishok" in generated


class TestGeneratedFetchReliabilityClaim:
    def test_does_not_claim_a_double_digit_fetch_failure_rate(self, generated):
        assert not re.search(
            r"\d{1,2}\s*-\s*\d{1,2}%\s+of\s+tickers\s+may\s+fail", generated), (
            "the stale 10-25% fetch-failure claim is back in the generator"
        )

    def test_states_the_measured_rate_with_its_sample(self, generated):
        section = _section(generated, "Limitations")
        assert "9,036" in section and "0 fetch failures" in section

    def test_points_at_the_gate_that_makes_the_claim_safe(self, generated):
        assert "check_run_health" in _section(generated, "Limitations")


class TestCommittedFileMatchesTheGenerator:
    """The tripwire. A hand-edit to the markdown is silently reverted by the
    next data run, so it must fail here instead of on the live site."""

    def test_committed_overview_is_what_the_generator_produces(self, generated):
        assert OVERVIEW.exists(), "SCREENER_OVERVIEW.md is missing"
        on_disk = OVERVIEW.read_text(encoding="utf-8")
        assert on_disk == generated, (
            "SCREENER_OVERVIEW.md differs from build_screener_overview(cfg). "
            "This file is GENERATED - run_screener.py step 11 overwrites it "
            "every run. Edit build_screener_overview() and regenerate; a "
            "hand-edit here survives only until the next 02:00 data run."
        )
