import os

from dotenv import load_dotenv
from openai import OpenAI

# .env 파일에서 환경 변수 로드
load_dotenv()

_client = None
_configured_api_key = None


def _get_client():
    """
    OPENAI_API_KEY를 기반으로 한 OpenAI 클라이언트를 생성/재사용한다.
    """
    global _client, _configured_api_key

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. "
            ".env 파일에 OPENAI_API_KEY=sk-... 형식으로 입력해주세요."
        )

    if _client is None or api_key != _configured_api_key:
        _client = OpenAI(api_key=api_key)
        _configured_api_key = api_key

    return _client

def call_openai(prompt):
    """
    OpenAI (ChatGPT) API를 호출하여 응답을 받는 함수
    """
    try:
        client = _get_client()
        print(f"질문: {prompt}")
        print("답변 생성 중...")

        # API 호출 (gpt-4o-mini 모델 사용)
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )

        # 응답 텍스트 추출
        answer = response.choices[0].message.content
        return answer

    except Exception as e:
        return f"에러 발생: {str(e)}"

if __name__ == "__main__":
    # 테스트 질문
    user_prompt = "파이썬으로 OpenAI API 호출하는 방법을 간단히 설명해줘."
    
    result = call_openai(user_prompt)
    
    print("-" * 50)
    print(f"답변:\n{result}")
    print("-" * 50)
