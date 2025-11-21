import json
from pathlib import Path

import pytest

from pytest_llm_cache.session_cache import (
    configure_session_cache,
    set_current_test_id,
    lookup_approved_response_if_enabled,
    record_pending_call_for_current_test,
    approve_passed_test_and_persist,
    persist_failed_test_entries,
    write_session_stats_summary,
)


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


def test_session_pass_only_cache_basic(tmp_path):
    """
    PASS-only session cache should:
    - Miss before approval
    - Record pending entries
    - Approve on PASS and reuse approved entries
    - Persist per-test NDJSON
    """
    session_dir = tmp_path / "session"
    configure_session_cache(session_dir=str(session_dir), enabled=True)

    # Test context
    set_current_test_id("case-1")
    provider = "openai"
    prompt = "hello"
    metadata = {"model": "gpt-4o-mini", "temperature": 0.0}

    # Lookup miss before any record/approval
    assert lookup_approved_response_if_enabled(provider, prompt, metadata) is None

    # Record pending response under current test (not yet reusable)
    record_pending_call_for_current_test("R1", provider, prompt, metadata)
    assert lookup_approved_response_if_enabled(provider, prompt, metadata) is None

    # Approve on PASS and persist
    approve_passed_test_and_persist()

    # Now lookup should hit and return native response object
    cached = lookup_approved_response_if_enabled(provider, prompt, metadata)
    assert cached == "R1"

    # Per-test NDJSON exists
    ndjson_path = session_dir / "case-1.ndjson"
    assert ndjson_path.exists()
    lines = _read_ndjson_lines(ndjson_path)
    assert any(l.get("approved") is True and l.get("outcome") == "passed" for l in lines)


def test_session_pass_only_cache_failed_entries_not_reused(tmp_path):
    """
    Failed entries should be persisted with approved=False and never reused.
    """
    session_dir = tmp_path / "session2"
    configure_session_cache(session_dir=str(session_dir), enabled=True)

    # Test id contains unsafe chars; filename should be sanitized
    set_current_test_id("bad/test")
    provider = "gemini"
    prompt = "안녕"
    metadata = {"model": "gemini-2.0-flash-lite-preview-02-05"}

    # Record and persist as failed
    record_pending_call_for_current_test("ERR", provider, prompt, metadata)
    persist_failed_test_entries(error="Boom")

    # Lookup should miss for failed entries
    assert lookup_approved_response_if_enabled(provider, prompt, metadata) is None

    # NDJSON exists with sanitized filename and approved=False lines
    ndjson_path = session_dir / "bad_test.ndjson"
    assert ndjson_path.exists()
    lines = _read_ndjson_lines(ndjson_path)
    assert any(l.get("approved") is False and l.get("outcome") == "failed" for l in lines)


def test_session_stats_summary(tmp_path):
    """
    Stats summary should be written and include public keys.
    """
    session_dir = tmp_path / "session3"
    configure_session_cache(session_dir=str(session_dir), enabled=True)

    set_current_test_id("stats-case")
    provider = "openai"
    prompt = "world"
    metadata = {"model": "gpt-4o-mini"}

    # Record, approve, then write summary
    record_pending_call_for_current_test("R2", provider, prompt, metadata)
    approve_passed_test_and_persist()
    write_session_stats_summary()

    stats_path = session_dir / "_stats.json"
    assert stats_path.exists()
    data = json.loads(stats_path.read_text(encoding="utf-8"))

    assert data.get("is_consolidated") is True
    stats = data.get("stats", {})
    # Public keys should be present
    for k in ["cache_hits", "cache_misses", "llm_calls_total", "approved_items_total", "failed_items_total"]:
        assert k in stats
