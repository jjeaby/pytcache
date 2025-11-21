import json
from pathlib import Path

from pytest_llm_cache.pytest_llm_cache import LLMCache


def _read_ndjson_lines(p: Path):
    if not p.exists():
        return []
    lines = []
    with p.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                lines.append(json.loads(line))
            except Exception:
                continue
    return lines


def test_stats_json_written_on_save(tmp_path):
    """
    When LLMCache.save() is called after changes, a separate stats.json should be written
    next to the cache file for quick inspection.
    """
    cache_path = tmp_path / "llm_cache.json"
    cache = LLMCache(cache_path)

    # create first entry and persist
    cache.get_or_create("providerX", "hello", lambda: "resp-1")
    cache.save()

    stats_path = tmp_path / "stats.json"
    assert stats_path.exists(), "stats.json must be written on save for inspection"

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    # minimal sanity checks on keys written
    assert "total_entries" in stats
    assert "cache_hit_count" in stats
    assert "cache_miss_count" in stats
    assert "providers" in stats


def test_entries_ndjson_appends_new_entries(tmp_path):
    """
    Newly created entries should be appended into entries.ndjson with a grep-friendly line
    including an outcome marker and the cache key.
    """
    cache_path = tmp_path / "llm_cache.json"
    cache = LLMCache(cache_path)

    # first entry (new)
    cache.get_or_create("openai", "prompt-1", lambda: "resp-1", metadata={"model": "gpt-4o-mini"})
    cache.save()

    ndjson_path = tmp_path / "entries.ndjson"
    assert ndjson_path.exists(), "entries.ndjson must exist after saving new entries"

    lines = _read_ndjson_lines(ndjson_path)
    assert len(lines) >= 1, "entries.ndjson should contain at least one line for the new entry"

    # validate structure of the latest line
    last = lines[-1]
    assert last.get("approved") is True
    assert last.get("outcome") == "stored"
    assert "key" in last  # SHA-256 of provider/prompt/metadata payload
    assert last.get("provider") == "openai"
    # human-readable summary is best-effort; for string responses it equals the response
    assert last.get("response") == "resp-1"
    assert last.get("response_text") == "resp-1"
