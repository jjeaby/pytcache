import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_REAL_LLM_TESTS") != "1" or not os.getenv("OPENAI_API_KEY"),
    reason=(
        "Set RUN_REAL_LLM_TESTS=1 and configure OPENAI_API_KEY to run integration cache tests."
    ),
)

CACHE_FILE = Path(".pytest-llm-cache/llm_responses.json")


def _read_cache():
    if not CACHE_FILE.exists():
        return {}
    return json.loads(CACHE_FILE.read_text(encoding="utf-8")).get("entries", {})


def _assert_cache_exists():
    assert CACHE_FILE.exists(), "캐시 파일이 생성되지 않았습니다."


def _skip_if_error(response):
    if isinstance(response, str) and response.startswith("에러 발생:"):
        pytest.skip(f"OpenAI API 호출 실패: {response}")


def _has_prompt(entries, prompt):
    return any(entry["request"]["prompt"] == prompt for entry in entries.values())


def _get_prompt_entry(entries, prompt):
    for entry in entries.values():
        if entry["request"]["prompt"] == prompt:
            return entry
    raise AssertionError(f"{prompt} not found in cache")


def test_openai_cached_response_reuses_cached_value(
    openai_cached_response, llm_cache
):
    prompt = "OpenAI 캐시 재사용 확인"

    first = openai_cached_response(prompt)
    llm_cache.save()
    assert isinstance(first, str)
    assert first
    _skip_if_error(first)

    _assert_cache_exists()
    cache_after_first = _get_prompt_entry(_read_cache(), prompt)

    second = openai_cached_response(prompt)
    llm_cache.save()
    _assert_cache_exists()
    cache_after_second = _get_prompt_entry(_read_cache(), prompt)

    assert second == first
    assert cache_after_first["usage_count"] == 1
    assert cache_after_second["usage_count"] == 2


def test_openai_cached_response_adds_new_prompt(
    openai_cached_response, llm_cache
):
    prompt1 = "첫 번째 OpenAI 프롬프트"
    prompt2 = "두 번째 OpenAI 프롬프트"

    first = openai_cached_response(prompt1)
    llm_cache.save()
    _skip_if_error(first)
    _assert_cache_exists()
    data = _read_cache()
    assert _has_prompt(data, prompt1)

    second = openai_cached_response(prompt2)
    llm_cache.save()
    _skip_if_error(second)
    _assert_cache_exists()
    updated = _read_cache()
    assert _has_prompt(updated, prompt1)
    assert _has_prompt(updated, prompt2)
