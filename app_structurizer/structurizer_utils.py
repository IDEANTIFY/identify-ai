import json
import os
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# --- OpenAI API 클라이언트 초기화 ---
# load_dotenv()가 실행된 후이므로 os.environ.get이 .env 파일의 값을 읽을 수 있습니다.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def extract_structured_idea_info(raw_text: str) -> dict:
    """
    자유 서술형 아이디어 설명에서 주요 구성 요소를 추출하여 구조화된 딕셔너리로 반환합니다.

    Input:
        - raw_text (str): 사용자가 입력한 아이디어에 대한 상세 설명

    Output:
        - dict: 아이디어의 핵심 요소들이 키-값 형태로 정리된 딕셔너리
          성공 시 예:
          {
              "주요 내용": "AI 기반 식단 분석 및 맞춤형 레시피 추천 모바일 앱",
              "도메인": "건강 및 피트니스, 푸드테크",
              "목적": "개인 맞춤형 건강 관리 및 식습관 개선",
              "차별성": "AI를 활용한 자동 식단 분석 및 정밀한 레시피 추천",
              "핵심 기술": "인공지능(AI), 머신러닝, 이미지 인식(음식 사진 분석)",
              "서비스 대상": "건강에 관심이 많은 사용자, 특정 식단이 필요한 환자"
          }
          실패 시에는 모든 값에 "정보 없음" 또는 원본 텍스트가 채워집니다.
    """
    prompt = f"""
다음 아이디어 설명에서 핵심 정보를 항목별로 명사구 형태로 간결하게 생성해줘.

- 설명: "{raw_text}"

- 출력 형식 (JSON):
{{
  "주요 내용": "...",
  "도메인": "...",
  "목적": "...",
  "차별성": "...",
  "핵심 기술": "...",
  "서비스 대상": "..."
}}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content
        return json.loads(content)
    
    except Exception as e:
        print(f"[오류] 구조화된 정보 추출에 실패했습니다: {e}")
        return {
            "주요 내용": raw_text,
            "도메인": "정보 없음",
            "목적": "정보 없음",
            "차별성": "정보 없음",
            "핵심 기술": "정보 없음",
            "서비스 대상": "정보 없음"
        }