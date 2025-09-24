# -*- coding: utf-8 -*-

import json
import os
from openai import OpenAI

# --- OpenAI API 클라이언트 초기화 ---
# 환경 변수에서 API 키를 가져옵니다. 보안을 위해 코드에 직접 키를 입력하는 것보다 이 방법이 권장됩니다.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

def generate_search_query(structured_info: dict) -> str:
    """
    구조화된 아이디어 정보를 바탕으로 웹 검색에 적합한 단일 검색 쿼리를 생성합니다.

    Input:
        - structured_info (dict): `extract_structured_idea_info` 함수로부터 반환된,
          아이디어의 핵심 요소가 정리된 딕셔너리.

    Output:
        - str: 웹 검색 엔진에 바로 사용할 수 있는 자연어 형태의 검색 쿼리.
          성공 시 예: "AI 식단 분석 기반 맞춤형 레시피 추천 앱 시장 동향"
          실패 시에는 입력된 정보들을 조합한 기본 쿼리가 생성됩니다.
    """

    # 입력 딕셔너리의 각 필드가 비어있지 않도록 '정보 없음'으로 기본값 처리
    fields = ["주요 내용", "도메인", "목적", "차별성", "핵심 기술", "서비스 대상"]
    cleaned_input = {k: (structured_info.get(k) or "정보 없음") for k in fields}

    prompt = (
        f"다음 핵심 정보를 조합하여, 시장 조사, 기술 동향, 경쟁 서비스 분석을 위한 "
        f"웹 검색용 명사구 형태의 자연스러운 검색어 1개를 생성해줘.\n\n"
        f"- 주요 내용: {cleaned_input['주요 내용']}\n"
        f"- 도메인: {cleaned_input['도메인']}\n"
        f"- 핵심 기술: {cleaned_input['핵심 기술']}\n"
        f"- 서비스 대상: {cleaned_input['서비스 대상']}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}]
        )
        query = response.choices[0].message.content.strip().replace('"', '')
        return query
    
    except Exception as e:
        print(f"[경고] API 호출 실패로 기본 쿼리를 생성합니다: {e}")

        # API 실패 시, 주요 키워드를 조합하여 대체 쿼리 생성
        fallback_query = f"{cleaned_input['도메인']} {cleaned_input['핵심 기술']} {cleaned_input['주요 내용']}"
        return fallback_query.replace("정보 없음", "").strip()
