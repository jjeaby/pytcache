# pytest-llm-cache

`pytest-llm-cache` 는 pytest 플러그인으로, LLM 호출 결과를 디스크에 저장하여 반복되는 API 호출을 자동으로 캐싱합니다.
Gemini/OpenAI 예제 클라이언트와 테스트 스위트를 함께 제공하므로, 실전 플러그인 개발 흐름에 맞춰
구조화된 레포지토리를 참고할 수 있습니다.

- 동일 provider+프롬프트+메타데이터 조합을 SHA-256 키로 관리, 디스크 캐시(JSON)로 일관된 결과 재사용
- `"에러 발생:"` 로 시작하는 실패 응답은 자동으로 캐시하지 않아, 임시 오류가 캐시를 오염시키지 않음
- 캐시 통계(`stats`)에 provider별 히트율/재사용 횟수 등을 포함하여 상태를 쉽게 파악

## 사전 준비

1. Python이 설치되어 있어야 합니다.
2. API 키가 필요합니다.
   - **Google Gemini**: [Google AI Studio](https://aistudio.google.com/app/apikey)
   - **OpenAI ChatGPT**: [OpenAI Platform](https://platform.openai.com/api-keys)

## 설치 방법

Poetry 기반 설치를 지원합니다. 아래 명령으로 의존성을 설치하세요.

```bash
# (권장) Poetry 설치 후 기본 의존성 설치
poetry install

# 테스트/플러그인 개발까지 포함(옵션 dev 그룹 설치)
poetry install --with dev

# Poetry 환경에서 테스트 실행
poetry run pytest -q
```

pip 기반 대안도 제공됩니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # 또는
pip install -e .[dev]
```

Poetry 사용 시 `pyproject.toml`의 [tool.poetry.dependencies]와 dev 그룹 의존성으로 설치가 이루어집니다. pytest 플러그인 엔트리포인트는 `[tool.poetry.plugins."pytest11"]`에 등록되어 있어 `poetry run pytest` 실행 시 자동으로 인식됩니다.

## 설정

1. `.env.example` 파일을 복사하여 `.env` 파일을 만듭니다.
2. `.env` 파일을 열고 필요한 API 키를 입력합니다.

```
GOOGLE_API_KEY=AIzaSy...
OPENAI_API_KEY=sk-...
```

## 폴더 구조

```python
├── pyproject.toml          # 패키지/빌드 설정 및 pytest 플러그인 엔트리포인트
├── requirements.txt        # 개발 편의를 위한 메타 설치 파일 (-e .[dev])
├── src
│   └── pytest_llm_cache
│       ├── pytest_llm_cache.py     # pytest 플러그인 본체
│       ├── clients/                # 플러그인 데모/테스트용 LLM 클라이언트
│       │   ├── gemini_client.py
│       │   └── openai_client.py
│       └── __init__.py
├── tests                   # 단위/통합 테스트
│   ├── conftest.py
│   ├── test_llm_cache_plugin.py
│   ├── test_gemini_cache_usage.py
│   ├── test_openai_cache_usage.py
│   └── test_llm_apis.py
└── ...
```

`pyproject.toml` 의 `[tool.pytest.ini_options]` 에서 `pythonpath = "src"` 를 지정하므로, 별도의 환경변수 수정 없이 pytest 가 패키지를 찾을 수 있습니다. `src/pytest_llm_cache/clients` 디렉터리는 플러그인 테스트 및 문서용 샘플 코드에만 사용되며, 실제 애플리케이션에서는 자신만의 클라이언트를 연결하면 됩니다.

## 실행 방법

### Google Gemini 실행

```bash
python -m pytest_llm_cache.clients.gemini_client
```

### OpenAI ChatGPT 실행

```bash
python -m pytest_llm_cache.clients.openai_client
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

`pytest_llm_cache/pytest_llm_cache.py` 는 pytest 실행 시 실제 LLM 호출 결과를 JSON 캐시에 저장하고,
동일한 프롬프트가 다시 요청되면 저장된 응답을 재사용합니다. `tests/conftest.py` 에서 기본적으로
플러그인을 로드하므로 바로 사용할 수 있습니다. 패키지를 설치하면 `pyproject.toml` 의 `pytest11`
엔트리포인트 덕분에 pytest 가 자동으로 플러그인을 발견합니다.

주요 옵션(간소화):
- `--llm-cache-file`: 캐시 파일 경로 (기본값 `.pytest-llm-cache/llm_responses.json`)
- `--llm-cache-disable`: 캐시 기능 비활성화
- `--llm-cache-pass-only`: PASS된 테스트에서 승인된 응답만 세션 동안 재사용

예시:
```bash
pytest --llm-cache-file .cache/llm.json
pytest --llm-cache-disable   # 캐시 사용하지 않음
pytest --llm-cache-pass-only # PASS-only 세션 캐시 활성화
```

플러그인은 다음과 같은 세션 스코프 픽스처를 제공합니다(테스트나 conftest에서 바로 주입 가능).

- `llm_cache`: 저수준 캐시 객체 (`get_or_create` 메서드 사용)
- `llm_cached_call(provider, prompt, factory, *, metadata=None, should_cache=None)`: 임의 호출 래핑용 헬퍼 (`should_cache(response)` 콜백으로 캐시 여부 제어)
- `gemini_cached_response(prompt)`, `openai_cached_response(prompt)`: 각 LLM 클라이언트 호출을 손쉽게 래핑한 유틸

빌트인 LLM 헬퍼들은 `"에러 발생:"` 으로 시작하는 문자열(실패 응답)을 자동으로 캐시에 저장하지 않습니다.
따라서 API 쿼터 부족 등 일시적인 오류 메시지가 캐시를 오염시키지 않으며, 필요하다면 사용자 정의
`should_cache` 콜백으로 다른 기준을 지정할 수 있습니다.

캐시 파일은 JSON 형식으로 저장되며, `entries` 객체 아래에 해시된 요청 ID별 레코드가 쌓입니다. 각 레코드는
`provider`, `request`(프롬프트와 호출 시 사용한 모든 파라미터), `response` 문자열, `usage_count`
(해당 응답을 몇 번 재사용했는지), `first_created_at`, `last_used_at` 타임스탬프를 포함합니다. 최상위에는
`stats` 블록이 있어 전체 캐시 현황을 한눈에 파악할 수 있습니다. 주요 키:

- `total_entries`, `provider_entry_counts`: 캐시에 저장된 고유 요청 수(전체/Provider별)
- `total_usage_count`, `average_usage_count`, `provider_usage`: 캐시된 응답이 사용된 누적 횟수와 평균값
- `cache_hit_count`, `cache_miss_count`, `cache_request_count`, `hit_rate`, `cache_miss_rate`: 캐시 적중/미스와 전체 요청 대비 비율
- `entries_with_hits`, `provider_reused_entry_counts`: 한 번 이상 재사용된 엔트리 수(전체/Provider별)
- `provider_hit_counts`, `provider_request_counts`, `provider_hit_rates`, `provider_average_usage_counts`: Provider별 세부지표
- `provider_stats`: 위 모든 정보를 Provider별 객체 단위로 모아둔 종합 뷰

### 캐시 파일 확인 및 초기화

- 기본 저장 위치는 `.pytest-llm-cache/llm_responses.json` 입니다. 실행 후 아래처럼 내용을 바로 확인할 수 있습니다.
  ```bash
  cat .pytest-llm-cache/llm_responses.json | jq .stats
  ```

- 특정 테스트/세션에서 새 응답을 받고 싶으면 파일을 삭제하거나, 프롬프트/메타데이터(모델/온도 등)를 변경해 새로운 키를 생성하세요.

## 검사/디버깅 아티팩트 (Inspection & Debugging Artifacts)

본 플러그인은 테스트 중 캐시 동작을 빠르게 점검할 수 있도록 별도의 검사 파일을 함께 저장합니다.

- stats.json (요약 통계)
  - 경로: 캐시 파일과 동일한 디렉터리 (기본: .pytest-llm-cache/stats.json)
  - 저장 시점: LLMCache.save()가 호출되고 내부 상태가 변경되었을 때(=dirty일 때) 항상 쓰여집니다.
  - 주요 키: total_entries, cache_hit_count, cache_miss_count, providers, cache_request_count, hit_rate, 등.
  - 용도: jq 등으로 빠르게 캐시 현황을 요약 확인.
  - 예:
    ```bash
    cat .pytest-llm-cache/stats.json | jq .
    ```

- entries.ndjson (신규 엔트리 추적)
  - 경로: 캐시 파일과 동일한 디렉터리 (기본: .pytest-llm-cache/entries.ndjson)
  - 저장 시점: save() 시점에 “이번 실행에서 새로 생성된” 캐시 엔트리만 NDJSON 라인으로 append
  - 라인 스키마(예): 
    ```json
    {
      "key": "<SHA-256 키>",
      "approved": true,
      "outcome": "stored",
      "provider": "openai",
      "request": {"prompt": "...", "metadata": {...}},
      "response": "resp-1",
      "response_text": "resp-1",
      "usage_count": 1,
      "first_created_at": 1732200000.123,
      "last_used_at": 1732200000.123
    }
    ```
  - 용도: grep/awk 등 텍스트 툴로 신규 캐시 엔트리를 빠르게 추적/검사.

- PASS-only 세션 캐시 (선택 기능)
  - 활성화 플래그: `--llm-cache-pass-only`
  - 세션 아티팩트 경로(기본): `.pytest-llm-cache/session/<YYYY-MM-DD>/<UTC_TS>/`
    - `<테스트명>.ndjson`: 각 테스트의 승인/실패 엔트리 (approved=true/false, outcome=passed/failed)
    - `_stats.single.json`: 세션 중간 스냅샷
    - `_stats.json`: 세션 종료 시 집계 요약
  - 저장 기준:
    - 테스트 통과 시: pending 엔트리를 approved=true, outcome=passed 로 기록
    - 테스트 실패 시: approved=false, outcome=failed 로 기록(재사용되지 않음)
  - 용도: e2e 스타일 시나리오에서 PASS된 호출만 세션 내에서 재사용하고, 문제 상황을 per-test 파일로 바로 추적
