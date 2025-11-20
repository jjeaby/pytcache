import os

import google.generativeai as genai
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

_configured_api_key = None


def _ensure_configured():
    """
    Gemini SDK가 API 키로 초기화되었는지 확인한다.
    """
    global _configured_api_key

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다. "
            ".env 파일에 GOOGLE_API_KEY=... 를 추가해주세요."
        )

    if api_key != _configured_api_key:
        genai.configure(api_key=api_key)
        _configured_api_key = api_key

def call_llm(prompt):
    """
    Google Gemini API를 호출하여 응답을 받는 함수
    """
    try:
        _ensure_configured()
        print(f"질문: {prompt}")
        print("답변 생성 중...")

        # 모델 선택 (gemini-2.0-flash-lite-preview-02-05)
        model = genai.GenerativeModel("gemini-2.0-flash-lite-preview-02-05")

        # 콘텐츠 생성 요청
        response = model.generate_content(prompt)

        # 응답 텍스트 반환
        return response.text

    except Exception as e:
        return f"에러 발생: {str(e)}"

if __name__ == "__main__":
    # 테스트 질문
    user_prompt = "파이썬으로 Google Gemini API 호출하는 방법을 간단히 설명해줘."
    
    result = call_llm(user_prompt)
    
    print("-" * 50)
    print(f"답변:\n{result}")
    print("-" * 50)
