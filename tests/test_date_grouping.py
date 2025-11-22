import json
from pathlib import Path

import pytest

from pytest_llm_cache.session_cache import (
    configure_session_cache,
    set_current_test_id,
    record_pending_call_for_current_test,
    approve_passed_test_and_persist,
)
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


def test_default_session_dir_uses_date_grouping(tmp_path, monkeypatch):
    """
    When PASS-only session cache is enabled without an explicit session_dir,
    the default directory should group by UTC date (YYYY-MM-DD) and then by session timestamp.
    """
    # Use a stable, known session timestamp and run in an isolated CWD
    monkeypatch.setenv("PYTEST_LLM_CACHE_SESSION_TS", "20251122T123456Z")
    monkeypatch.chdir(tmp_path)

    # Enable PASS-only session cache with default dir (None -> use _default_session_dir)
    configure_session_cache(session_dir=None, enabled=True)

    # Simulate a passing test flow
    set_current_test_id("grouping-case")
    provider = "openai"
    prompt = "date-grouping-check"
    metadata = {"model": "gpt-4o-mini", "temperature": 0.0}

    record_pending_call_for_current_test("RESP", provider, prompt, metadata)
    approve_passed_test_and_persist()

    # Validate default path structure: .pytest-llm-cache/session/YYYY-MM-DD/UTC_TS/<test>.ndjson
    base_dir = tmp_path / ".pytest-llm-cache" / "session" / "2025-11-22" / "20251122T123456Z"
    ndjson_path = base_dir / "grouping-case.ndjson"
    assert ndjson_path.exists(), f"Expected NDJSON at {ndjson_path}"
    lines = _read_ndjson_lines(ndjson_path)
    assert any(l.get("approved") is True and l.get("outcome") == "passed" for l in lines), "Approved passed entry must exist"
