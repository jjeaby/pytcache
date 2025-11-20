from pathlib import Path

import pytest

pytest_plugins = ("pytcache.pytest_llm_cache",)


_CACHE_FILE = Path(".pytest-llm-cache/llm_responses.json")


@pytest.fixture(scope="session", autouse=True)
def reset_llm_cache_session(llm_cache):
    """Ensure the shared cache file starts clean once per test session."""

    llm_cache._store.clear()
    llm_cache._dirty = False
    _CACHE_FILE.unlink(missing_ok=True)
    yield
