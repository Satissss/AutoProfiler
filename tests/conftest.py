"""Test-wide safety rails."""
import pytest


@pytest.fixture(autouse=True)
def _no_shared_cache(tmp_path, monkeypatch):
    """Keep anything a test caches inside tmp_path.

    An Env built by hand has cache_dir=None and writes nothing, so this only matters
    for the few tests that build one through build_env. Real runs cache under
    output/_cache; tests that wrote there left `class ModelNew: pass` seeds behind
    that a later real run loaded as if they were ports.
    """
    monkeypatch.setenv("AUTOPROFILER_CACHE_DIR", str(tmp_path / "cache"))
