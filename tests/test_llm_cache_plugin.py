import json

import pytest

from pytcache.pytest_llm_cache import LLMCache


def _load_payload(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_entries(path):
    return _load_payload(path)["entries"]


def _first_entry(entries):
    return next(iter(entries.values()))


def test_llm_cache_persists_responses(tmp_path):
    cache_path = tmp_path / "llm_cache.json"
    cache = LLMCache(cache_path)
    calls = {"count": 0}

    def factory():
        calls["count"] += 1
        return f"response-{calls['count']}"

    first = cache.get_or_create("gemini", "프롬프트", factory)
    second = cache.get_or_create("gemini", "프롬프트", factory)

    assert first == second
    assert calls["count"] == 1

    cache.save()
    assert cache_path.exists()

    payload = _load_payload(cache_path)
    assert payload["stats"]["total_entries"] == 1
    assert payload["stats"]["providers"] == ["gemini"]
    assert payload["stats"]["total_usage_count"] == 2
    assert payload["stats"]["average_usage_count"] == 2
    assert payload["stats"]["provider_usage"]["gemini"] == 2
    assert payload["stats"]["provider_entry_counts"]["gemini"] == 1
    assert payload["stats"]["cache_hit_count"] == 1
    assert payload["stats"]["cache_miss_count"] == 1
    assert payload["stats"]["reused_entries"] == 1
    assert payload["stats"]["provider_reused_entry_counts"]["gemini"] == 1
    assert payload["stats"]["hit_rate"] == pytest.approx(0.5)
    assert payload["stats"]["cache_request_count"] == 2
    assert payload["stats"]["cache_miss_rate"] == pytest.approx(0.5)
    assert payload["stats"]["entries_with_hits"] == 1
    assert payload["stats"]["provider_hit_counts"]["gemini"] == 1
    assert payload["stats"]["provider_request_counts"]["gemini"] == 2
    assert payload["stats"]["provider_hit_rates"]["gemini"] == pytest.approx(0.5)
    assert payload["stats"]["provider_average_usage_counts"]["gemini"] == 2
    provider_stats = payload["stats"]["provider_stats"]["gemini"]
    assert provider_stats["entry_count"] == 1
    assert provider_stats["usage_count"] == 2
    assert provider_stats["request_count"] == 2
    assert provider_stats["cache_hit_count"] == 1
    assert provider_stats["cache_miss_count"] == 1
    assert provider_stats["hit_rate"] == pytest.approx(0.5)
    assert provider_stats["average_usage_count"] == 2
    assert provider_stats["reused_entry_count"] == 1

    entries = payload["entries"]
    entry = _first_entry(entries)
    assert entry["response"] == "response-1"
    assert entry["request"]["prompt"] == "프롬프트"
    assert entry["usage_count"] == 2
    assert entry["first_created_at"] <= entry["last_used_at"]

    reloaded = LLMCache(cache_path)
    third = reloaded.get_or_create("gemini", "프롬프트", factory)

    assert third == "response-1"
    assert calls["count"] == 1  # no additional API calls


def test_llm_cache_disable_option(tmp_path):
    cache_path = tmp_path / "llm_cache.json"
    cache = LLMCache(cache_path, disabled=True)
    calls = {"count": 0}

    def factory():
        calls["count"] += 1
        return f"response-{calls['count']}"

    value_1 = cache.get_or_create("openai", "prompt", factory)
    value_2 = cache.get_or_create("openai", "prompt", factory)

    assert value_1 != value_2  # 새로운 값이 생성됨
    assert calls["count"] == 2
    cache.save()
    assert not cache_path.exists()


def test_llm_cache_refresh_option(tmp_path):
    cache_path = tmp_path / "llm_cache.json"
    cache = LLMCache(cache_path)

    cache.get_or_create("provider", "a", lambda: "first")
    cache.save()

    entries = _load_entries(cache_path)
    entry = _first_entry(entries)
    assert entry["response"] == "first"
    assert entry["usage_count"] == 1

    refreshed = LLMCache(cache_path, refresh=True)

    calls = {"count": 0}

    def factory():
        calls["count"] += 1
        return "second"

    assert refreshed.get_or_create("provider", "a", factory) == "second"
    assert calls["count"] == 1
    refreshed.save()

    entries = _load_entries(cache_path)
    entry = _first_entry(entries)
    assert entry["response"] == "second"
    assert entry["usage_count"] == 1


def test_llm_cache_stores_metadata(tmp_path):
    cache_path = tmp_path / "llm_cache.json"
    cache = LLMCache(cache_path)

    cache.get_or_create(
        "gemini",
        "Q1",
        lambda: "A1",
        metadata={"model": "gemini-test", "temperature": 0.0},
    )
    cache.save()

    entries = _load_entries(cache_path)
    entry = _first_entry(entries)
    assert entry["request"]["metadata"]["model"] == "gemini-test"
    assert entry["request"]["metadata"]["temperature"] == 0.0
    assert entry["usage_count"] == 1


def test_llm_cache_distinguishes_metadata(tmp_path):
    cache_path = tmp_path / "llm_cache.json"
    cache = LLMCache(cache_path)
    calls = {"count": 0}

    def factory():
        calls["count"] += 1
        return f"resp-{calls['count']}"

    first = cache.get_or_create(
        "gemini", "prompt", factory, metadata={"model": "A", "temperature": 0.5}
    )
    second = cache.get_or_create(
        "gemini", "prompt", factory, metadata={"model": "B", "temperature": 0.5}
    )

    assert first == "resp-1"
    assert second == "resp-2"
    assert calls["count"] == 2

    cache.save()
    entries = _load_entries(cache_path)
    assert len(entries) == 2


def test_llm_cache_usage_count_increments(tmp_path):
    cache_path = tmp_path / "llm_cache.json"
    cache = LLMCache(cache_path)

    cache.get_or_create("gemini", "P", lambda: "A")
    cache.get_or_create("gemini", "P", lambda: "A")
    cache.save()

    entries = _load_entries(cache_path)
    entry = _first_entry(entries)
    assert entry["usage_count"] == 2


def test_llm_cache_skips_entries_when_should_cache_returns_false(tmp_path):
    cache_path = tmp_path / "llm_cache.json"
    cache = LLMCache(cache_path)

    responses = [
        "에러 발생: quota",  # should not be cached
        "정상 응답",  # should be cached
    ]

    def factory():
        return responses.pop(0)

    def should_cache(response):
        return not response.startswith("에러 발생:")

    first = cache.get_or_create(
        "openai", "prompt", factory, should_cache=should_cache
    )

    assert first.startswith("에러 발생:")
    cache.save()
    assert not cache_path.exists()

    second = cache.get_or_create(
        "openai", "prompt", factory, should_cache=should_cache
    )

    assert second == "정상 응답"
    cache.save()
    assert cache_path.exists()

    payload = _load_payload(cache_path)
    entry = _first_entry(payload["entries"])
    assert entry["response"] == "정상 응답"
    assert entry["usage_count"] == 1
