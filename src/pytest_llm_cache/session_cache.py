from __future__ import annotations

"""
Session-scoped PASS-only cache for pytest runs (optional mode for pytest-llm-cache).

Features:
- Context-aware tracking of the current test id (pytest item.name) via ContextVar
- Pending cache entries recorded per test during execution
- Approval of a test's cache entries only when the pytest test PASSES
- Reuse of approved cache entries across other tests in the same pytest session
- Human-readable per-test cache persistence (NDJSON) under a per-session folder
- Simple stats logging (hits, misses, total calls, approved items) persisted at snapshot or session end

Default behavior:
- This cache is DISABLED unless the plugin enables it via configuration (CLI flags in pytest-llm-cache).
- When enabled, only PASS-approved caches are considered for lookup.
- Only caches from the current pytest session are used (no preload from previous runs).

Notes:
- This module is standalone to avoid cycles with the primary JSON cache (LLMCache).
- It provides generic keying compatible with LLMCache: provider + prompt + metadata (sorted JSON) -> SHA256.
- Response objects are stored as JSON strings with best-effort serialization. If not serializable, minimal string form is kept.
"""

import json
import os
import time
import hashlib
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from contextvars import ContextVar


# Context: current test id (e.g., pytest item.name) for attribution of cache records
_CURRENT_TEST_ID: ContextVar[Optional[str]] = ContextVar("_CURRENT_TEST_ID", default=None)

# Session directory (configured by plugin); if None, lazy-resolve to default
_SESSION_DIR: Optional[Path] = None

# Toggle: enabled/disabled; controlled by plugin options
_PASS_ONLY_ENABLED: bool = False


def _get_session_id() -> str:
    """
    Session identifier accessor (UTC timestamp propagated via env or computed lazily).
    Environment variable name chosen to avoid collisions with other tools.
    """
    return os.getenv("PYTEST_LLM_CACHE_SESSION_TS") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_session_dir() -> Path:
    """
    Default artifact root for session cache data.
    """
    return Path(".pytest-llm-cache") / "session" / _get_session_id()


@dataclass
class CacheRecord:
    """
    A single cache record for one LLM call (provider/prompt/metadata key).
    """

    key: str
    test_id: str
    created_at: str
    provider: str
    request: Dict[str, Any]  # {"prompt": str, "metadata": Dict[str, Any]}
    response_json: str
    # Optional human-readable summary
    response_text: Optional[str] = None

    # In-memory only: native response object to return on reuse
    _response_obj: Any = None


class SessionPassOnlyCache:
    """
    Manages PASS-only cache entries for pytest sessions.
    """

    def __init__(self) -> None:
        # Pending per-test (not yet approved)
        self._pending_by_test: Dict[str, Dict[str, CacheRecord]] = {}
        # Approved (reusable across tests in current session)
        self._approved_global: Dict[str, CacheRecord] = {}
        # Per-key hit counts for inspection
        self._hit_counts_per_key: Dict[str, int] = {}
        # Failed items per test for inspection/debugging
        self._failed_per_test: Dict[str, int] = {}
        # Stats
        self._stats: Dict[str, int] = {
            "cache_hits": 0,
            "cache_misses": 0,
            "llm_calls_total": 0,  # includes both cached and real calls
            "approved_items_total": 0,
            "failed_items_total": 0,
        }

    # Configuration
    def configure(self, *, session_dir: Optional[Path] = None, enabled: bool = False) -> None:
        """
        Configure the cache manager: enable/disable and set session artifact directory.
        """
        global _SESSION_DIR, _PASS_ONLY_ENABLED
        _PASS_ONLY_ENABLED = bool(enabled)
        _SESSION_DIR = Path(session_dir) if session_dir else None

    # Enable/active toggles
    def _is_enabled(self) -> bool:
        return _PASS_ONLY_ENABLED

    def _is_active(self) -> bool:
        # Active when enabled and a current test id is known
        return self._is_enabled() and self.get_current_test_id() is not None

    # Context management
    def set_current_test_id(self, test_id: str) -> None:
        """
        Set the current test id for attribution.
        """
        _CURRENT_TEST_ID.set(test_id)

    def get_current_test_id(self) -> Optional[str]:
        """
        Get the current test id from context.
        """
        return _CURRENT_TEST_ID.get()

    # Artifact directory helpers
    def _resolve_session_dir(self) -> Path:
        """
        Resolve and (optionally) create the current session directory.
        Only creates directories when cache is active.
        """
        session_dir = _SESSION_DIR or _default_session_dir()
        if self._is_active():
            session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    @staticmethod
    def _safe_filename(name: str) -> str:
        """
        Make a filename from a test id with safe characters.
        """
        name = name.replace("/", "_").replace("\\", "_")
        return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:200]

    # Keying compatible with LLMCache
    @staticmethod
    def make_key(provider: str, prompt: str, metadata: Dict[str, Any]) -> str:
        """
        Create a stable key from provider/prompt/metadata affecting completion output.
        Mirrors LLMCache._build_key to remain consistent.
        """
        payload: Dict[str, Any] = {
            "provider": provider,
            "prompt": prompt,
            "metadata": metadata or {},
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # Lookup, record, approve/persist, fail/persist
    def lookup_if_enabled(self, provider: str, prompt: str, metadata: Dict[str, Any]) -> Optional[Any]:
        """
        Return a cached response if available and enabled; otherwise None.
        """
        if not self._is_enabled():
            return None
        # Enforce active reuse: require a current test id
        if self.get_current_test_id() is None:
            return None

        key = self.make_key(provider, prompt, metadata)
        record = self._approved_global.get(key)

        if record and record._response_obj is not None:
            self._stats["cache_hits"] += 1
            self._stats["llm_calls_total"] += 1
            self._hit_counts_per_key[key] = self._hit_counts_per_key.get(key, 0) + 1
            self._write_session_stats_snapshot_if_enabled()
            return record._response_obj

        self._stats["cache_misses"] += 1
        self._stats["llm_calls_total"] += 1
        self._write_session_stats_snapshot_if_enabled()
        return None

    def record_call(self, response_obj: Any, provider: str, prompt: str, metadata: Dict[str, Any]) -> None:
        """
        Record a call under the current test id (pending until PASS). Does not record when cache is not active.
        """
        # When cache is enabled, llm_calls_total is incremented at lookup time (hit or miss).
        # When cache is disabled or non-active, increment here to keep total calls accurate.
        if not self._is_active():
            self._stats["llm_calls_total"] += 1
        if not self._is_active():
            # Skip caching and avoid any file output outside tests or when disabled
            return

        key = self.make_key(provider, prompt, metadata)

        # Serialize response to JSON best-effort
        try:
            # Common pattern for pydantic-like objects
            response_json = response_obj.model_dump_json()  # type: ignore[attr-defined]
        except Exception:
            try:
                response_json = json.dumps(response_obj, ensure_ascii=False)
            except Exception:
                response_json = json.dumps({"_repr": str(response_obj)}, ensure_ascii=False)

        # Extract a human-readable summary if possible
        response_text: Optional[str] = None
        try:
            # Certain responses may be simple strings
            if isinstance(response_obj, str):
                response_text = response_obj
            else:
                # Best-effort: OpenAI-style choices[0].message.content
                choices = getattr(response_obj, "choices", [])
                if choices and getattr(choices[0], "message", None):
                    response_text = choices[0].message.content
        except Exception:
            response_text = None

        # Guaranteed non-None due to _is_active gate above
        test_id = self.get_current_test_id() or "unknown"

        rec = CacheRecord(
            key=key,
            test_id=test_id,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
            provider=provider,
            request={"prompt": prompt, "metadata": metadata},
            response_json=response_json,
            response_text=response_text,
            _response_obj=response_obj,
        )

        if test_id not in self._pending_by_test:
            self._pending_by_test[test_id] = {}
        # Record/overwrite latest
        self._pending_by_test[test_id][key] = rec

        self._write_session_stats_snapshot_if_enabled()

    def approve_current_test_and_persist(self) -> None:
        """
        Approve pending records for current test (on PASS) and persist NDJSON file.
        """
        if not self._is_active():
            return

        test_id = self.get_current_test_id()
        if not test_id:
            return

        pending = self._pending_by_test.get(test_id, {})
        if not pending:
            return

        # Move to approved
        for key, rec in pending.items():
            self._approved_global[key] = rec
        self._stats["approved_items_total"] += len(pending)

        # Persist per-test NDJSON file
        session_dir = self._resolve_session_dir()
        safe_test_id = self._safe_filename(test_id)
        test_file = session_dir / f"{safe_test_id}.ndjson"

        with test_file.open("a", encoding="utf-8") as f:
            for _, rec in pending.items():
                # Persist without in-memory-only fields
                d = asdict(rec)
                d.pop("_response_obj", None)
                # Explicit markers for inspection/debugging:
                # - approved: true → this entry is from a PASSed test and reusable within this session
                # - session_id: current pytest session identifier (folder name)
                # - outcome: "passed" to mirror failed entries and simplify downstream parsing
                d["approved"] = True
                d["session_id"] = _get_session_id()
                d["outcome"] = "passed"
                line = json.dumps(d, ensure_ascii=False)
                f.write(line + "\n")

        # Clear pending for this test
        self._pending_by_test[test_id] = {}

        self._write_session_stats_snapshot_if_enabled()

    def persist_current_test_as_failed(self, error: Optional[str] = None) -> None:
        """
        Persist pending records for current test as FAILED (for debugging/inspection only).

        - Does NOT approve or reuse these entries.
        - Writes NDJSON lines with approved=False and outcome='failed'.
        - Clears pending for this test afterward.
        """
        if not self._is_active():
            # Respect the same flag; if disabled or non-active, skip persistence.
            return

        test_id = self.get_current_test_id()
        if not test_id:
            return

        pending = self._pending_by_test.get(test_id, {})
        if not pending:
            return

        session_dir = self._resolve_session_dir()
        safe_test_id = self._safe_filename(test_id)
        test_file = session_dir / f"{safe_test_id}.ndjson"

        failed_count = 0
        with test_file.open("a", encoding="utf-8") as f:
            for _, rec in pending.items():
                d = asdict(rec)
                d.pop("_response_obj", None)
                d["approved"] = False
                d["session_id"] = _get_session_id()
                d["outcome"] = "failed"
                if error:
                    d["error"] = str(error)[:2000]
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                failed_count += 1

        # Update stats and per-test counters
        self._stats["failed_items_total"] = self._stats.get("failed_items_total", 0) + failed_count
        self._failed_per_test[test_id] = self._failed_per_test.get(test_id, 0) + failed_count

        # Clear pending for this test
        self._pending_by_test[test_id] = {}

        self._write_session_stats_snapshot_if_enabled()

    def write_session_stats_summary(self) -> None:
        """
        Write a consolidated summary stats JSON file at session end.
        """
        if not self._is_enabled():
            return
        session_dir = self._resolve_session_dir()
        session_dir.mkdir(parents=True, exist_ok=True)

        stats_public_keys = [
            "cache_hits",
            "cache_misses",
            "llm_calls_total",
            "approved_items_total",
            "failed_items_total",
        ]
        # Ensure llm_calls_total matches cache_hits + cache_misses at time of summary
        snapshot = {k: int(self._stats.get(k, 0)) for k in stats_public_keys}
        snapshot["llm_calls_total"] = snapshot.get("cache_hits", 0) + snapshot.get("cache_misses", 0)

        summary = {
            "session_id": _get_session_id(),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "worker": "single",  # simplified (no xdist consolidation here)
            "is_consolidated": True,
            "stats": snapshot,
            "approved_keys_count": len(self._approved_global),
            "hit_counts_per_key": self._hit_counts_per_key,
            "session_dir": str(session_dir),
            "approved_per_test": {
                tid: sum(1 for rec in self._approved_global.values() if rec.test_id == tid)
                for tid in {rec.test_id for rec in self._approved_global.values()}
            },
            "failed_per_test": self._failed_per_test,
            "failed_items_total": int(self._stats.get("failed_items_total", 0)),
        }
        with (session_dir / "_stats.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    # Internal: write a live snapshot to _stats.single.json (optional convenience)
    def _write_session_stats_snapshot_if_enabled(self) -> None:
        """
        Persist stats snapshot updated live during the session when enabled & active.
        Avoids IO when disabled or not in active test context.
        """
        if not self._is_enabled():
            return
        if self.get_current_test_id() is None and not self._approved_global and not self._pending_by_test:
            return

        session_dir = self._resolve_session_dir()
        session_dir.mkdir(parents=True, exist_ok=True)
        stats_file = session_dir / "_stats.single.json"

        stats_public_keys = [
            "cache_hits",
            "cache_misses",
            "llm_calls_total",
            "approved_items_total",
            "failed_items_total",
        ]
        snapshot = {k: int(self._stats.get(k, 0)) for k in stats_public_keys}
        snapshot["llm_calls_total"] = snapshot.get("cache_hits", 0) + snapshot.get("cache_misses", 0)

        local_count = sum(1 for rec in self._approved_global.values())
        summary = {
            "session_id": _get_session_id(),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "worker": "single",
            "is_consolidated": False,
            "stats": snapshot,
            "approved_keys_count_worker": local_count,
            "approved_keys_worker": [
                {
                    "key": key,
                    "test_id": rec.test_id,
                    "provider": rec.provider,
                    "created_at": rec.created_at,
                    "hits": self._hit_counts_per_key.get(key, 0),
                }
                for key, rec in self._approved_global.items()
            ],
            "hit_counts_per_key": self._hit_counts_per_key,
            "session_dir": str(session_dir),
            "approved_per_test": {
                test_id: sum(1 for rec in self._approved_global.values() if rec.test_id == test_id)
                for test_id in {rec.test_id for rec in self._approved_global.values()}
            },
            "failed_per_test": self._failed_per_test,
            "failed_items_total": int(self._stats.get("failed_items_total", 0)),
        }
        with stats_file.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


# Singleton manager
_session_pass_only_cache = SessionPassOnlyCache()


# Public API for plugin integration (pytest-llm-cache)
def set_current_test_id(test_id: str) -> None:
    _session_pass_only_cache.set_current_test_id(test_id)


def configure_session_cache(*, session_dir: Optional[str] = None, enabled: bool = False) -> None:
    _session_pass_only_cache.configure(
        session_dir=Path(session_dir) if session_dir else None,
        enabled=enabled,
    )


def lookup_approved_response_if_enabled(provider: str, prompt: str, metadata: Dict[str, Any]) -> Optional[Any]:
    return _session_pass_only_cache.lookup_if_enabled(provider, prompt, metadata or {})


def record_pending_call_for_current_test(response_obj: Any, provider: str, prompt: str, metadata: Dict[str, Any]) -> None:
    _session_pass_only_cache.record_call(response_obj, provider, prompt, metadata or {})


def approve_passed_test_and_persist() -> None:
    _session_pass_only_cache.approve_current_test_and_persist()


def persist_failed_test_entries(error: Optional[str] = None) -> None:
    _session_pass_only_cache.persist_current_test_as_failed(error)


def write_session_stats_summary() -> None:
    _session_pass_only_cache.write_session_stats_summary()
