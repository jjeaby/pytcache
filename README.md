# PYTEST LLM Cache Example (Gemini & ChatGPT)

이 프로젝트는 Python을 사용하여 Google Gemini와 OpenAI ChatGPT API를 호출하는 예제

## 사전 준비

1. Python이 설치되어 있어야 합니다.
2. API 키가 필요합니다.
   - **Google Gemini**: [Google AI Studio](https://aistudio.google.com/app/apikey)
   - **OpenAI ChatGPT**: [OpenAI Platform](https://platform.openai.com/api-keys)

## 설치 방법

권장 가상환경을 생성한 뒤 패키지를 설치합니다. Poetry를 사용하면 pyproject.toml 기반으로 동일하게 동작합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # 내부적으로 -e .[dev] 를 설치합니다.
```

또는 `pip install -e .[dev]` 를 직접 실행해도 동일합니다. Poetry 사용 시에는 `poetry install` 로 기본 의존성만 설치하고, 테스트/플러그인 개발까지 진행하려면 `poetry install --with dev` 로 옵션 의존성(`pytest`)을 함께 설치하세요.

## 설정

1. `.env.example` 파일을 복사하여 `.env` 파일을 만듭니다.
2. `.env` 파일을 열고 필요한 API 키를 입력합니다.

```
GOOGLE_API_KEY=AIzaSy...
OPENAI_API_KEY=sk-...
```

## 폴더 구조

```
├── pyproject.toml          # 패키지/빌드 설정
├── requirements.txt        # 개발 편의를 위한 메타 설치 파일 (-e .[dev])
├── src
│   └── pytcache            # 라이브러리 및 pytest 플러그인 구현
│       ├── __init__.py
│       ├── gemini_client.py
│       ├── openai_client.py
│       └── pytest_llm_cache.py
├── tests                   # 단위/통합 테스트
│   ├── conftest.py
│   ├── test_llm_cache_plugin.py
│   ├── test_gemini_cache_usage.py
│   ├── test_openai_cache_usage.py
│   └── test_llm_apis.py
└── ...
```

`pyproject.toml` 의 `[tool.pytest.ini_options]` 에서 `pythonpath = "src"` 를 지정하므로, 별도의 환경변수 수정 없이 pytest 가 패키지를 찾을 수 있습니다.

## 실행 방법

### Google Gemini 실행
```bash
python -m pytcache.gemini_client
```

### OpenAI ChatGPT 실행
```bash
python -m pytcache.openai_client
```

## 테스트 실행

pytest를 사용하여 API 호출이 정상적으로 작동하는지 테스트할 수 있습니다.

```bash
pytest
```

일부 통합 테스트(`tests/test_llm_apis.py`, `tests/test_gemini_cache_usage.py`, `tests/test_openai_cache_usage.py`)는 실제 LLM API를 호출합니다.
이 테스트들을 실행하려면 다음 환경 변수를 설정한 뒤 API 키도 함께 지정해야 합니다.

```bash
export RUN_REAL_LLM_TESTS=1
export GOOGLE_API_KEY=...
export OPENAI_API_KEY=...
pytest
```

## Pytest LLM 캐시 플러그인

`pytcache/pytest_llm_cache.py` 는 pytest 실행 시 실제 LLM 호출 결과를 JSON 캐시에 저장하고,
동일한 프롬프트가 다시 요청되면 저장된 응답을 재사용합니다. `tests/conftest.py` 에서 기본적으로
플러그인을 로드하므로 바로 사용할 수 있습니다.

주요 옵션:

- `--llm-cache-file`: 캐시 파일 경로 (기본값 `.pytest-llm-cache/llm_responses.json`)
- `--llm-cache-refresh`: 기존 캐시를 무시하고 이번 실행에서만 새 응답을 저장
- `--llm-cache-disable`: 캐시 기능 비활성화

예시:

```bash
pytest --llm-cache-file .cache/llm.json
pytest --llm-cache-refresh   # 강제로 새 응답 요청
pytest --llm-cache-disable   # 캐시 사용하지 않음
```

플러그인은 다음과 같은 세션 스코프 픽스처를 제공합니다.

- `llm_cache`: 저수준 캐시 객체 (`get_or_create` 메서드 사용)
- `llm_cached_call(provider, prompt, factory, *, metadata=None, should_cache=None)`: 임의의 호출 래핑용 헬퍼 (`should_cache(response)` 콜백으로 캐시 여부 제어 가능)
- `gemini_cached_response(prompt)`, `openai_cached_response(prompt)`: 각 LLM을 편하게 호출하는 헬퍼

빌트인 LLM 헬퍼들은 `"에러 발생:"` 으로 시작하는 문자열(실패 응답)을 자동으로 캐시에 저장하지 않습니다.
따라서 API 쿼터 부족 등 일시적인 오류 메시지가 캐시를 오염시키지 않으며, 필요하다면 사용자 정의
`should_cache` 콜백으로 다른 기준을 지정할 수 있습니다.

캐시 파일은 JSON 형식으로 저장되며, `entries` 객체 아래에 해시된 요청 ID별 레코드가 쌓입니다. 각 레코드는
`provider`, `request`(프롬프트와 호출 시 사용한 모든 파라미터), `response` 문자열, `usage_count`
(해당 응답을 몇 번 재사용했는지), `first_created_at`, `last_used_at` 타임스탬프를 포함합니다.
최상위에는 `stats` 블록이 있어 전체 캐시 현황을 한눈에 파악할 수 있습니다. 주요 키:

- `total_entries`, `provider_entry_counts`: 캐시에 저장된 고유 요청 수(전체/Provider별)
- `total_usage_count`, `average_usage_count`, `provider_usage`: 캐시된 응답이 사용된 누적 횟수와 평균값
- `cache_hit_count`, `cache_miss_count`, `cache_request_count`, `hit_rate`, `cache_miss_rate`: 캐시 적중/미스와 전체 요청 대비 비율
- `entries_with_hits`, `provider_reused_entry_counts`: 한 번 이상 재사용된 엔트리 수(전체/Provider별)
- `provider_hit_counts`, `provider_request_counts`, `provider_hit_rates`, `provider_average_usage_counts`: Provider별 세부지표
- `provider_stats`: 위 모든 정보를 Provider별 객체 단위로 모아둔 종합 뷰

이 정보를 기반으로 어떤 Provider가 캐시 효과를 잘 보고 있는지, 적중률이 떨어지는 프롬프트가 있는지 등을 쉽게 파악할 수 있습니다.
내부적으로 provider + 프롬프트 + 메타데이터 조합을 SHA-256으로 해시하여 키로 사용하므로, 프롬프트뿐 아니라
모델, temperature 등 메타데이터까지 동일할 때만 캐시를 재사용하며, 요청 파라미터가 다르면 자동으로 신규
응답을 받아 저장합니다. 이런 구조 덕분에 `entries.<hash>.request` 를 확인하면 어떤 조건으로 캐시되었는지 쉽게
디버깅하고, `stats` 로 전반적인 상태를 살필 수 있습니다.

### 캐시 파일 확인 및 초기화

- 기본 저장 위치는 `.pytest-llm-cache/llm_responses.json` 입니다. 실행 후 아래처럼 내용을 바로 확인할 수 있습니다.

  ```bash
  cat .pytest-llm-cache/llm_responses.json | jq .stats
  ```

- 특정 테스트/세션에서만 새 응답을 받고 싶다면 `pytest --llm-cache-refresh` 를 사용하거나 파일을 직접 삭제하면 됩니다.
- 캐시에 실패 응답이 남지 않도록 기본 헬퍼가 "에러 발생:" 문자열을 무시하므로, 오류가 반복될 경우 로그와 환경 변수를 먼저 확인하세요.
