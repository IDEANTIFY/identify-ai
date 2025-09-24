# -*- coding: utf-8 -*-

import json
import os
from openai import OpenAI

# --- OpenAI API 클라이언트 초기화 ---
# 환경 변수에서 API 키를 가져옵니다. 보안을 위해 코드에 직접 키를 입력하는 것보다 이 방법이 권장됩니다.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

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

# if __name__ == '__main__':
    # 1. 분석하고 싶은 아이디어를 자유롭게 서술합니다.
    # user_idea = "블록체인 기술을 활용하여 중고거래 시 발생할 수 있는 사기를 방지하고, 거래 과정을 투명하게 기록하는 플랫폼"
    # print(f"입력 아이디어: {user_idea}", flush = True)

    # 2. 아이디어 설명으로부터 구조화된 정보를 추출합니다.
    # print("\n[1단계] 아이디어에서 구조화된 정보 추출을 시작합니다...", flush = True)
    # structured_data = extract_structured_idea_info(user_idea)

    # 추출된 정보를 보기 좋게 출력합니다.
    # print(json.dumps(structured_data, indent=2, ensure_ascii=False), flush = True)

    # 3. 추출된 정보를 바탕으로 웹 검색 쿼리를 생성합니다.
    # search_query = generate_search_query(structured_data)
    # print(f"\n[최종 생성 쿼리]: {search_query}\n", flush = True)
