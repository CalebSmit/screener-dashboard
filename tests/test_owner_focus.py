"""The owner's queue must stay wired into the nightly routine.

`OWNER_FOCUS.md` is the only channel the owner has for directing this routine:
he does not review PRs, does not write tickets, and by design is not reading the
diff every morning. If the nightly prompt stops pointing at that file, requests
land in it and are never seen - and nothing else in the repo would notice,
because the sessions would keep running the rotation and reporting success.

That is the same failure shape as the evidence base sitting at 3 rows for 183
days while every data run passed (CLAUDE.md rule 8): a silent channel looks
exactly like an empty one.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO))

FOCUS = REPO / "OWNER_FOCUS.md"
NIGHTLY = REPO / "prompts" / "nightly.md"


def test_owner_focus_file_exists():
    assert FOCUS.exists(), "OWNER_FOCUS.md is the owner's only input channel"


@pytest.mark.parametrize("heading", ["## Open", "## Done"])
def test_owner_focus_has_the_headings_the_prompt_relies_on(heading):
    """The prompt tells the session to read **Open** and move finished items to
    **Done**. Renaming either heading breaks that instruction silently."""
    assert heading in FOCUS.read_text(encoding="utf-8")


def test_nightly_prompt_reads_the_queue_before_the_rotation():
    text = NIGHTLY.read_text(encoding="utf-8")
    assert "OWNER_FOCUS.md" in text, "nightly prompt no longer points at the owner queue"

    orient = text.index("## 1. Orient")
    baseline = text.index("## 2. Baseline")
    assert orient < text.index("OWNER_FOCUS.md") < baseline, (
        "the owner queue must be read during Orient, before any work is chosen")


def test_nightly_prompt_states_what_outranks_the_queue():
    """Two things legitimately outrank an owner item - a stalled data loop and
    the ship gates. Both must stay written down, or a session could justify
    pushing a broken build because the owner asked for a feature."""
    text = NIGHTLY.read_text(encoding="utf-8")
    section = text[text.index("OWNER_FOCUS.md"):text.index("## 2. Baseline")]
    assert "data loop" in section
    assert "ship gate" in section.lower()


def test_claude_md_documents_the_channel():
    text = (REPO / "CLAUDE.md").read_text(encoding="utf-8")
    assert "OWNER_FOCUS.md" in text


def test_deferred_items_must_be_reported_not_silently_skipped():
    """An unmentioned deferral is indistinguishable from an ignored request.
    The prompt has to say so explicitly."""
    text = NIGHTLY.read_text(encoding="utf-8")
    section = text[text.index("OWNER_FOCUS.md"):text.index("## 2. Baseline")]
    assert "defer" in section.lower()
    assert "log" in section.lower()
