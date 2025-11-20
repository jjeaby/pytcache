import os

import pytest

from pytcache import gemini_client, openai_client


def _require_real_env(var_name: str):
    if os.getenv("RUN_REAL_LLM_TESTS") != "1" or not os.getenv(var_name):
        pytest.skip(
            f"Set RUN_REAL_LLM_TESTS=1 and configure {var_name} to run this test."
        )


def test_gemini_api_call():
    """
    실제 Gemini API 호출이 성공적으로 수행되는지 통합 테스트한다.
    """

    _require_real_env("GOOGLE_API_KEY")

    prompt = "테스트: 안녕, 짧게 대답해줘."
    response = gemini_client.call_llm(prompt)

    assert isinstance(response, str)
    assert len(response) > 0
    # 실제 API 오류 메시지는 허용한다 (예: quota 초과).


def test_gemini_api_call_without_key(monkeypatch):
    """
    GOOGLE_API_KEY 가 없을 때 예외가 발생하는지 확인한다.
    """

    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setattr(gemini_client, "_configured_api_key", None)

    response = gemini_client.call_llm("임시 프롬프트")
    assert "환경 변수가 설정되지 않았습니다" in response


def test_openai_api_call():
    """
    실제 OpenAI API 호출이 성공적으로 수행되는지 통합 테스트한다.
    """

    _require_real_env("OPENAI_API_KEY")

    prompt = "테스트: 안녕, 짧게 대답해줘."
    response = openai_client.call_openai(prompt)

    assert isinstance(response, str)
    assert len(response) > 0
    # 실제 API 오류 메시지는 허용한다 (예: quota 초과).


def test_openai_api_call_without_key(monkeypatch):
    """
    OPENAI_API_KEY 가 없을 때 예외가 발생하는지 확인한다.
    """

    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(openai_client, "_configured_api_key", None)
    monkeypatch.setattr(openai_client, "_client", None)

    response = openai_client.call_openai("임시 프롬프트")
    assert "환경 변수가 설정되지 않았습니다" in response
