"""One company, one ranked line - and the choice of line does not move with the volume.

The proposal reviewed on 2026-10-21 (`research/2026-10-19-share-classes.md`) was accepted with
one change: the designated listing is **declared** in `config.yaml` rather than recomputed from
each run's dollar volume. The measurement behind that change is in
`research/measurements/2026-10-21-share-class-stability.py`: Alphabet's two lines sat within
S&P DJI's 20% switching margin in 31 of 89 month-ends and the higher-volume line changed four
times, while the same run scored on GOOG instead of GOOGL puts Alphabet 78 places higher.

So the tests that matter here are about **stability and blast radius**, not only about grouping:
a run's own volumes must not be able to change which line represents a company.
"""
from __future__ import annotations

import json

import pytest
import yaml

import share_classes as sc

with open("config.yaml", encoding="utf-8") as _fh:
    CFG = yaml.safe_load(_fh)
PAIRS = [["GOOGL", "GOOG"], ["FOXA", "FOX"], ["NWSA", "NWS"]]


# --------------------------------------------------------------------- the declaration
class TestDeclaration:
    def test_config_declares_one_line_per_known_pair(self):
        declared = set(CFG["share_classes"]["designated"])
        for pair in PAIRS:
            picked = [t for t in pair if t in declared]
            assert len(picked) == 1, f"{pair} must have exactly one declared line, got {picked}"

    def test_score_once_is_on_and_the_margin_is_spdji_s(self):
        assert CFG["share_classes"]["score_once_per_company"] is True
        assert CFG["share_classes"]["challenge_margin"] == pytest.approx(0.20)
        assert sc.CHALLENGE_MARGIN == pytest.approx(0.20)

    def test_the_config_comment_names_the_note_and_the_module(self):
        with open("config.yaml", encoding="utf-8") as fh:
            text = fh.read()
        i = text.index("share_classes:")
        block = text[max(0, i - 1400):i]
        assert "research/2026-10-19-share-classes.md" in block
        assert "share_classes.py" in block
        # the reason the choice is declared rather than recomputed
        assert "switching margin" in block


# ------------------------------------------------------------------------ the grouping
class TestGrouping:
    def test_groups_the_three_known_pairs_by_cik(self):
        ciks = {"GOOGL": 1652044, "GOOG": 1652044, "FOXA": 1754301, "FOX": 1754301,
                "NWSA": 1564708, "NWS": 1564708, "AAPL": 320193, "MSFT": 789019}
        groups, basis = sc.group_companies(list(ciks), ciks=ciks)
        assert basis == "sec_cik"
        assert sorted(sorted(g) for g in groups) == [["FOX", "FOXA"], ["GOOG", "GOOGL"],
                                                     ["NWS", "NWSA"]]

    def test_a_single_line_company_is_never_a_group(self):
        ciks = {"AAPL": 320193, "MSFT": 789019}
        groups, _ = sc.group_companies(list(ciks), ciks=ciks)
        assert groups == []

    def test_falls_back_to_longname_when_no_cik_file_is_cached(self):
        """`data/sec/` is gitignored, so a fresh checkout has no CIK file. Measured
        2026-10-21 on the live universe: the two bases give the same three groups with the
        same membership, so the fallback is corroborated and not assumed."""
        names = {"GOOGL": "Alphabet Inc.", "GOOG": "Alphabet Inc.",
                 "FOXA": "Fox Corporation", "FOX": "Fox Corporation",
                 "AAPL": "Apple Inc."}
        groups, basis = sc.group_companies(list(names), names=names, ciks={})
        assert basis == "long_name"
        assert sorted(sorted(g) for g in groups) == [["FOX", "FOXA"], ["GOOG", "GOOGL"]]

    def test_no_cik_and_no_names_groups_nothing_rather_than_guessing(self):
        groups, basis = sc.group_companies(["AAPL", "MSFT"], names=None, ciks={})
        assert (groups, basis) == ([], "none")

    def test_cik_map_indexes_both_ticker_spellings(self, tmp_path):
        p = tmp_path / "company_tickers.json"
        p.write_text(json.dumps({"0": {"cik_str": 1067983, "ticker": "BRK-B",
                                       "title": "BERKSHIRE HATHAWAY INC"}}))
        m = sc.cik_map(p)
        assert m["BRK-B"] == 1067983 and m["BRK.B"] == 1067983

    def test_a_missing_cik_file_is_an_empty_map_not_a_crash(self, tmp_path):
        assert sc.cik_map(tmp_path / "nope.json") == {}

    def test_an_unreadable_cik_file_is_an_empty_map_not_a_crash(self, tmp_path):
        p = tmp_path / "company_tickers.json"
        p.write_text("{not json")
        assert sc.cik_map(p) == {}


# ------------------------------------------------------------- the choice, and its stickiness
class TestDesignation:
    def test_the_declared_line_is_kept_and_the_other_leaves(self):
        sec, notes = sc.designate([["GOOGL", "GOOG"]], {"GOOGL": 9.0, "GOOG": 8.0},
                                  declared=["GOOGL"])
        assert sec == {"GOOG": "GOOGL"}
        assert all(n.startswith(("NOTE", "ATTENTION")) for n in notes)

    def test_the_declaration_beats_this_run_s_volume(self):
        """The whole point of the review's change: GOOG trading 3x GOOGL does not move
        Alphabet's ranked line, because that is a 78-place swing on a liquidity tiebreak."""
        sec, notes = sc.designate([["GOOGL", "GOOG"]], {"GOOGL": 1.0, "GOOG": 3.0},
                                  declared=["GOOGL"])
        assert sec == {"GOOG": "GOOGL"}, "a run's own volume must not re-pick the line"
        assert any("ATTENTION" in n and "GOOG" in n for n in notes)

    def test_a_crossing_over_the_margin_is_reported(self):
        _, notes = sc.designate([["GOOGL", "GOOG"]], {"GOOGL": 1.0, "GOOG": 1.25},
                                declared=["GOOGL"])
        assert any(n.startswith("ATTENTION") and "25%" in n for n in notes)

    def test_inside_the_margin_is_a_note_not_an_attention(self):
        notes = sc.designate([["GOOGL", "GOOG"]], {"GOOGL": 1.0, "GOOG": 1.10},
                             declared=["GOOGL"])[1]
        assert notes and all(n.startswith("NOTE") for n in notes)

    def test_well_clear_of_the_margin_says_nothing(self):
        """Fox and News Corp never came within 20% in 89 month-ends; they must stay quiet."""
        assert sc.designate([["FOXA", "FOX"]], {"FOXA": 5.0, "FOX": 1.0},
                            declared=["FOXA"])[1] == []

    def test_an_undeclared_pair_falls_back_to_volume_and_says_so_loudly(self):
        sec, notes = sc.designate([["AAA", "AAAB"]], {"AAA": 2.0, "AAAB": 1.0}, declared=[])
        assert sec == {"AAAB": "AAA"}
        assert any(n.startswith("ATTENTION") and "no " in n and "designated" in n for n in notes)

    def test_two_declared_lines_of_one_company_is_a_configuration_error(self):
        with pytest.raises(ValueError, match="more than one line"):
            sc.designate([["GOOGL", "GOOG"]], {"GOOGL": 2.0, "GOOG": 1.0},
                         declared=["GOOGL", "GOOG"])

    def test_a_missing_volume_never_wins_the_fallback(self):
        sec, _ = sc.designate([["AAA", "AAAB"]], {"AAA": float("nan"), "AAAB": 1.0}, declared=[])
        assert sec == {"AAA": "AAAB"}

    def test_all_volumes_missing_is_deterministic_rather_than_arbitrary(self):
        sec, _ = sc.designate([["BBB", "AAA"]], {}, declared=[])
        assert sec == {"BBB": "AAA"}, "ties break alphabetically so two runs agree"

    def test_the_declaration_in_config_applied_to_the_live_pairs(self):
        groups = [list(p) for p in PAIRS]
        vols = {"GOOGL": 1.52, "GOOG": 1.0, "FOXA": 4.9, "FOX": 1.0, "NWSA": 3.2, "NWS": 1.0}
        sec, _ = sc.designate(groups, vols, declared=CFG["share_classes"]["designated"])
        assert sec == {"GOOG": "GOOGL", "FOX": "FOXA", "NWS": "NWSA"}


# ---------------------------------------------------------------------- what the page gets
class TestPriceGaps:
    def test_the_gap_is_the_second_line_against_the_ranked_one(self):
        out = sc.price_gaps({"FOX": "FOXA"}, {"FOX": 57.22, "FOXA": 64.15})
        assert out["FOX"]["scored_as"] == "FOXA"
        assert out["FOX"]["price"] == pytest.approx(57.22)
        assert out["FOX"]["price_gap_pct"] == pytest.approx(-10.80, abs=0.01)

    def test_a_missing_price_is_none_rather_than_a_wrong_number(self):
        out = sc.price_gaps({"FOX": "FOXA"}, {"FOX": 57.22})
        assert out["FOX"]["price_gap_pct"] is None

    def test_a_zero_price_does_not_divide_by_zero(self):
        out = sc.price_gaps({"FOX": "FOXA"}, {"FOX": 1.0, "FOXA": 0.0})
        assert out["FOX"]["price_gap_pct"] is None


# ------------------------------------------------------------------ the pipeline is wired
class TestPipelineWiring:
    def test_run_screener_drops_the_secondary_lines_before_the_percentiles(self):
        """A two-line company otherwise counts twice in its sector's distribution: measured
        2026-10-21, the three second lines held the other 16 Communication Services stocks'
        composites down by 1.565 points, and nothing outside that sector moved (0.0)."""
        with open("run_screener.py", encoding="utf-8") as fh:
            src = fh.read()
        drop = src.index('df = df[~df["Ticker"].isin(list(secondary_lines))]')
        pct = src.index("df = compute_sector_percentiles(df)")
        assert drop < pct, "the drop must happen before the sector percentiles"

    def test_the_run_records_what_it_dropped(self):
        with open("run_screener.py", encoding="utf-8") as fh:
            src = fh.read()
        assert 'ctx.run_dir / "share_classes.json"' in src

    def test_no_scoring_module_imports_share_classes_for_a_score(self):
        """The module decides *which line represents a company*, never what a metric is
        worth. It must not become an input to a score."""
        import factor_engine
        import stock_summary
        for mod in (factor_engine, stock_summary):
            with open(mod.__file__, encoding="utf-8") as fh:
                src = fh.read()
            assert "share_classes" not in src, f"{mod.__name__} must not read share_classes"
