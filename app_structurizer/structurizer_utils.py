import json
import os
from openai import OpenAI

# --- OpenAI API 클라이언트 초기화 ---
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

def extract_structured_idea_info(raw_text: str) -> dict:
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