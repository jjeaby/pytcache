"""
Pytest plugin that caches Gemini/OpenAI LLM responses to reduce redundant API calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict
import hashlib
import time

import pytest

from .clients import gemini_client, openai_client
from .session_cache import (
    set_current_test_id as set_session_test_id,
    configure_session_cache,
    lookup_approved_response_if_enabled as session_lookup,
    record_pending_call_for_current_test as session_record,
    approve_passed_test_and_persist as session_approve,
    persist_failed_test_entries as session_persist_failed,
    write_session_stats_summary as session_write_summary,
)
import os
from datetime import datetime, timezone
import importlib.util

HAS_XDIST = importlib.util.find_spec("xdist") is not None

DEFAULT_CACHE_FILE = Path(".pytest-llm-cache/llm_responses.json")


def _should_cache_llm_response(response: Any) -> bool:
    """Return False when the response looks like a transient LLM error."""

    if isinstance(response, str):
        return not response.strip().startswith("에러 발생:")
    return True


class LLMCache:
    """
    Simple JSON-backed cache for storing LLM responses.
    """

    def __init__(
        self,
        cache_file: Path,
        *,
        disabled: bool = False,
        refresh: bool = False,
    ):
        self.cache_file = Path(cache_file)
        self.disabled = disabled
        self.refresh = refresh
        self._store: Dict[str, Dict[str, Any]] = {}
        self._dirty = False

        if not self.disabled and self.cache_file.exists() and not self.refresh:
            try:
                loaded = json.loads(self.cache_file.read_text(encoding="utf-8"))
                self._store = self._normalize_store(loaded)
            except json.JSONDecodeError:
                self._store = {}

    def get_or_create(
        self,
        provider: str,
        prompt: str,
        factory: Callable[[], Any],
        *,
        metadata: Dict[str, Any] | None = None,
        should_cache: Callable[[Any], bool] | None = None,
    ) -> Any:
        """
        Return cached response for provider/prompt or execute factory and store result.
        """
        if self.disabled:
            return factory()

        metadata = metadata or {}
        cache_key = self._build_key(provider, prompt, metadata)

        if cache_key not in self._store or self.refresh:
            response = factory()
            if should_cache is not None and not should_cache(response):
                return response

            entry = {
                "provider": provider,
                "request": {"prompt": prompt, "metadata": metadata},
                "response": response,
                "usage_count": 1,
                "first_created_at": time.time(),
                "last_used_at": time.time(),
            }
            self._store[cache_key] = entry
            self._dirty = True
        else:
            entry = self._store[cache_key]
            entry["usage_count"] += 1
            entry["last_used_at"] = time.time()
            self._dirty = True

        return self._store[cache_key]["response"]

    @staticmethod
    def _build_key(provider: str, prompt: str, metadata: Dict[str, Any]) -> str:
        payload = {"provider": provider, "prompt": prompt, "metadata": metadata}
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_store(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Normalize legacy provider-nested structures into a flat entry dictionary.
        """
        entries = raw.get("entries")
        if entries is None:
            entries = raw

        normalized: Dict[str, Dict[str, Any]] = {}
        for cache_key, entry in entries.items():
            entry = dict(entry)
            entry.setdefault("provider", "unknown")
            entry.setdefault("usage_count", 1)
            entry.setdefault("first_created_at", entry.get("timestamp", 0))
            entry.setdefault("last_used_at", entry["first_created_at"])
            normalized[cache_key] = entry
        return normalized

    def save(self) -> None:
        """
        Persist cache to disk if it has been modified.
        """
        if self.disabled or not self._dirty:
            return

        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        stats = self._build_stats()
        payload = {"entries": self._store, "stats": stats}

        self.cache_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._dirty = False

    def _build_stats(self) -> Dict[str, Any]:
        total_entries = len(self._store)
        provider_usage: Dict[str, int] = {}
        provider_entries: Dict[str, int] = {}
        provider_reused_entries: Dict[str, int] = {}
        provider_hit_counts: Dict[str, int] = {}
        total_usage_count = 0
        total_cache_hits = 0
        reused_entries = 0

        for entry in self._store.values():
            provider = entry["provider"]
            usage = entry.get("usage_count", 1)
            provider_usage[provider] = provider_usage.get(provider, 0) + usage
            provider_entries[provider] = provider_entries.get(provider, 0) + 1
            total_usage_count += usage

            hits = max(usage - 1, 0)
            total_cache_hits += hits
            provider_hit_counts[provider] = (
                provider_hit_counts.get(provider, 0) + hits
            )
            if hits:
                reused_entries += 1
                provider_reused_entries[provider] = (
                    provider_reused_entries.get(provider, 0) + 1
                )

        average_usage = total_usage_count / total_entries if total_entries else 0
        total_requests = total_cache_hits + total_entries
        hit_rate = total_cache_hits / total_requests if total_requests else 0
        miss_rate = 1 - hit_rate if total_requests else 0

        provider_stats: Dict[str, Dict[str, Any]] = {}
        provider_request_counts: Dict[str, int] = {}
        provider_hit_rates: Dict[str, float] = {}
        provider_average_usage: Dict[str, float] = {}

        for provider, entry_count in provider_entries.items():
            hits = provider_hit_counts.get(provider, 0)
            requests = hits + entry_count
            provider_request_counts[provider] = requests
            provider_hit_rates[provider] = hits / requests if requests else 0
            avg_usage = (
                provider_usage[provider] / entry_count if entry_count else 0
            )
            provider_average_usage[provider] = avg_usage
            provider_stats[provider] = {
                "entry_count": entry_count,
                "usage_count": provider_usage[provider],
                "request_count": requests,
                "cache_hit_count": hits,
                "cache_miss_count": entry_count,
                "hit_rate": provider_hit_rates[provider],
                "average_usage_count": avg_usage,
                "reused_entry_count": provider_reused_entries.get(provider, 0),
            }

        return {
            "total_entries": total_entries,
            "providers": sorted(provider_entries.keys()),
            "generated_at": time.time(),
            "total_usage_count": total_usage_count,
            "average_usage_count": average_usage,
            "provider_usage": provider_usage,
            "provider_entry_counts": provider_entries,
            "cache_hit_count": total_cache_hits,
            "cache_miss_count": total_entries,
            "hit_rate": hit_rate,
            "cache_request_count": total_requests,
            "cache_miss_rate": miss_rate,
            "reused_entries": reused_entries,
            "entries_with_hits": reused_entries,
            "provider_reused_entry_counts": provider_reused_entries,
            "provider_hit_counts": provider_hit_counts,
            "provider_request_counts": provider_request_counts,
            "provider_hit_rates": provider_hit_rates,
            "provider_average_usage_counts": provider_average_usage,
            "provider_stats": provider_stats,
        }


def pytest_addoption(parser):
    group = parser.getgroup("llm-cache")
    group.addoption(
        "--llm-cache-file",
        action="store",
        dest="llm_cache_file",
        default=str(DEFAULT_CACHE_FILE),
        help="JSON 파일 경로 (기본값: .pytest-llm-cache/llm_responses.json)",
    )
    group.addoption(
        "--llm-cache-refresh",
        action="store_true",
        dest="llm_cache_refresh",
        default=False,
        help="캐시 파일을 무시하고 LLM 응답을 다시 요청합니다.",
    )
    group.addoption(
        "--llm-cache-disable",
        action="store_true",
        dest="llm_cache_disable",
        default=False,
        help="캐시를 완전히 비활성화합니다.",
    )
    group.addoption(
        "--llm-cache-pass-only",
        action="store_true",
        dest="llm_cache_pass_only",
        default=False,
        help="PASS된 테스트에서 승인된 응답만 세션 동안 재사용합니다.",
    )
    group.addoption(
        "--llm-cache-session-dir",
        action="store",
        dest="llm_cache_session_dir",
        default="",
        help="세션 캐시 아티팩트 디렉터리 경로(기본값: .pytest-llm-cache/session/<UTC_TS>).",
    )


@pytest.fixture(scope="session")
def llm_cache(request):
    """
    Session-scoped fixture providing access to the LLM cache helper.
    """
    cache = LLMCache(
        Path(request.config.getoption("llm_cache_file")),
        disabled=bool(request.config.getoption("llm_cache_disable")),
        refresh=bool(request.config.getoption("llm_cache_refresh")),
    )

    yield cache
    cache.save()


@pytest.fixture(scope="session")
def llm_cached_call(llm_cache):
    """
    Helper fixture that wraps arbitrary callables with caching behavior.
    PASS-only session cache (optional) is consulted first; then JSON cache is used.
    """

    def _call(
        provider: str,
        prompt: str,
        factory: Callable[[], Any],
        *,
        metadata: Dict[str, Any] | None = None,
        should_cache: Callable[[Any], bool] | None = None,
    ):
        # Try PASS-only session cache first (no-op if disabled)
        cached = session_lookup(provider, prompt, metadata or {})
        if cached is not None:
            return cached

        # Execute and store via primary JSON cache
        result = llm_cache.get_or_create(
            provider,
            prompt,
            factory,
            metadata=metadata,
            should_cache=should_cache,
        )

        # Record pending call for PASS-only session cache (respect should_cache if provided)
        try:
            if should_cache is None or should_cache(result):
                session_record(result, provider, prompt, metadata or {})
        except Exception:
            # Keep tests resilient; ignore session cache recording errors
            pass

        return result

    return _call


@pytest.fixture(scope="session")
def gemini_cached_response(llm_cache):
    """
    Cached Gemini response helper. 사용 예) gemini_cached_response(prompt)
    PASS-only session cache (optional) is consulted first; then JSON cache is used.
    """

    def _call(prompt: str) -> Any:
        metadata = {"model": "gemini-2.0-flash-lite-preview-02-05"}

        # Try PASS-only session cache first (no-op if disabled)
        cached = session_lookup("gemini", prompt, metadata)
        if cached is not None:
            return cached

        # Fallback to primary JSON cache
        result = llm_cache.get_or_create(
            "gemini",
            prompt,
            lambda: gemini_client.call_llm(prompt),
            metadata=metadata,
            should_cache=_should_cache_llm_response,
        )

        # Record pending call for PASS-only session cache, respecting error-filter
        try:
            if _should_cache_llm_response(result):
                session_record(result, "gemini", prompt, metadata)
        except Exception:
            pass

        return result

    return _call


@pytest.fixture(scope="session")
def openai_cached_response(llm_cache):
    """
    Cached OpenAI response helper. 사용 예) openai_cached_response(prompt)
    PASS-only session cache (optional) is consulted first; then JSON cache is used.
    """

    def _call(prompt: str) -> Any:
        metadata = {
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "system_prompt": "You are a helpful assistant.",
        }

        # Try PASS-only session cache first (no-op if disabled)
        cached = session_lookup("openai", prompt, metadata)
        if cached is not None:
            return cached

        # Fallback to primary JSON cache
        result = llm_cache.get_or_create(
            "openai",
            prompt,
            lambda: openai_client.call_openai(prompt),
            metadata=metadata,
            should_cache=_should_cache_llm_response,
        )

        # Record pending call for PASS-only session cache, respecting error-filter
        try:
            if _should_cache_llm_response(result):
                session_record(result, "openai", prompt, metadata)
        except Exception:
            pass

        return result

    return _call


def pytest_configure(config) -> None:
    """
    Initialize per-session timestamp and configure PASS-only session cache.
    """
    try:
        # Establish a stable UTC timestamp across the whole session (controller and workers)
        if hasattr(config, "workerinput"):
            # Worker process: receive timestamp from controller
            ts = config.workerinput.get("pytest_llm_cache_session_ts")
            if ts:
                os.environ["PYTEST_LLM_CACHE_SESSION_TS"] = ts
        else:
            # Controller or non-xdist: compute (or reuse) a UTC timestamp and export it
            ts = os.environ.get("PYTEST_LLM_CACHE_SESSION_TS") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            os.environ["PYTEST_LLM_CACHE_SESSION_TS"] = ts
    except Exception:
        # Never fail test config due to timestamp wiring
        pass

    # Configure PASS-only session cache from options
    try:
        enabled = bool(config.getoption("llm_cache_pass_only"))
        session_dir_opt = config.getoption("llm_cache_session_dir") or ""
        configure_session_cache(session_dir=session_dir_opt if session_dir_opt else None, enabled=enabled)
    except Exception:
        # Keep configuration resilient
        pass


if HAS_XDIST:
    def pytest_configure_node(node) -> None:
        """
        Controller hook to propagate a stable UTC timestamp to all xdist workers.
        """
        try:
            ts = os.environ.get("PYTEST_LLM_CACHE_SESSION_TS") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            node.workerinput["pytest_llm_cache_session_ts"] = ts
        except Exception:
            pass


def pytest_runtest_setup(item) -> None:
    """
    Set the current test id for PASS-only session cache attribution before each test.
    """
    try:
        set_session_test_id(item.name)
    except Exception:
        # Keep test run resilient
        pass


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Approve/persist PASS-only cache entries on success/failure of test call phase.
    """
    outcome = yield
    report = outcome.get_result()

    # Approve and persist cache entries only on successful test calls
    if report.when == "call" and report.passed:
        try:
            session_approve()
        except Exception:
            # Do not let cache persistence impact tests
            pass

    # Persist pending cache entries as failed for debugging/inspection
    if report.when == "call" and report.failed and not getattr(report, "wasxfail", False):
        try:
            err_val = getattr(call, "excinfo", None)
            err_value = getattr(err_val, "value", None) if err_val is not None else None
            err_str = str(err_value) if err_value is not None else (getattr(err_val, "typename", "TestFailed") if err_val else "TestFailed")
            session_persist_failed(error=err_str)
        except Exception:
            pass


def pytest_sessionfinish(session, exitstatus) -> None:
    """
    Write a summary stats JSON file for PASS-only session cache at session end.
    """
    try:
        session_write_summary()
    except Exception:
        pass
