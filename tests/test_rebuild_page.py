"""`scripts/rebuild_page.py` - the page rebuilt from the published payload, payload untouched."""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("rebuild_page", ROOT / "scripts" / "rebuild_page.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)


def test_committed_page_is_what_the_generator_builds():
    """The committed page equals the generator's output on the committed payload, so a design
    change to the generator ships with the page it produces (run scripts/rebuild_page.py)."""
    assert rp.main(["--check"]) == 0


def test_building_never_touches_the_payload():
    before = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in rp.PAYLOADS}
    rp.build()
    assert before == {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in rp.PAYLOADS}
