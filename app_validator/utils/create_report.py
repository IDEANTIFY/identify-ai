import os
import json
from openai import OpenAI
from typing import List, Dict, Any

# --- ⚙️ 1. OpenAI 클라이언트 초기화 ---
# 보안을 위해 API 키는 환경 변수에서 불러옵니다.
# Colab의 경우 Secrets에 OPENAI_API_KEY를 설정하세요.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

# --- 📝 2. [보고서 1] 요약 보고서 생성 함수 ---

def generate_summary_report(
    query: str, 
    web_docs: List[Dict], 
    db_docs: List[Dict], 
    approx_similar_count: int
) -> Dict[str, Any]:
    """
    아이디어를 정량적으로 평가하는 요약 보고서(JSON)를 생성합니다.
    
    Args:
        query (str): 사용자 아이디어 쿼리.
        web_docs (List[Dict]): 웹 검색 결과 (e.g., [{'title': ..., 'snippet': ...}]).
        db_docs (List[Dict]): 내부 DB 검색 결과 (e.g., [{'title': ..., 'content': ...}]).
        approx_similar_count (int): 전체 유사 사례 건수.

    Returns:
        Dict[str, Any]: GPT가 생성한 요약 보고서 JSON 객체.
    """
    # 웹 및 DB 문서 섹션 구성
    web_section = "\n\n".join([f"[WEB{i+1}] {doc.get('title', '')}\n{doc.get('snippet', '')}" for i, doc in enumerate(web_docs)])
    db_section = "\n\n".join([f"[DB{i+1}] {doc.get('title', '')}\n{doc.get('content', '')}" for i, doc in enumerate(db_docs)])
    
    # 프롬프트 구성 (사용자 요청에 따라 원본 유지)
    prompt = f"""
당신은 대학생 스타업 아이디어를 정량적 기준에 따라 평가하고, 결과를 반드시 JSON 형식으로 반환하는 전문가입니다.

다음은 사용자의 아이디어입니다:
"{query}"

유사한 웹 문서:
{web_section}

유사한 공모전 수상작 및 내부 DB 사례:
{db_section}

다음 기준에 따라 아이디어를 평가하고, 결과를 아래에 명시된 JSON 형식 예시에 정확히 맞춰 출력해 주세요.

[평가 기준]
1.  total_similar_cases: 웹 검색 결과({approx_similar_count}건)와 DB 문서 수를 종합하여 전체 유사 건수를 정수로 판단.
2.  similarity: DB + 웹 문서 기준, 기술적 컨셉의 중복성을 판단하여 0~100 사이 정수 값으로 평가.
3.  creativity: 기존과 다른 조합, 새로움, 다른 목적/대상 설정 여부를 판단하여 0~100 사이 정수 값으로 평가.
4.  feasibility: 기술 성숙도, 상용화 여부, 구현 난이도 등을 고려하여 0~100 사이 정수 값으로 평가.
5.  analysis_narrative: 위 평가를 종합하여 600자 내외의 분석 요약을 문자열로 작성.

[출력 JSON 형식 예시]
```json
{{
  "report_summary": {{
    "total_similar_cases": 19,
    "evaluation_scores": {{
      "similarity": 80,
      "creativity": 50,
      "feasibility": 50
    }},
    "analysis_narrative": "공중 모빌리티 도메인의 도심 통근용 수직 이착륙 자율 비행 제어 하늘 비행 자동차 아이디어는, UAM/eVTOL 관련 다수 문헌과 상용화 흐름과 높은 중복성을 보입니다. 핵심 컨셉(도심 통근, 수직 이착륙, 자율 제어)은 기존 연구와 상당히 겹치나, 대학생 스타트업의 차별화 포인트가 모호합니다. 실현 가능성은 기술 성숙도와 규제·인프라 이슈로 인해 보수적으로 평가됩니다."
  }}
}}
```

다른 설명 없이 JSON 객체만 생성하세요. 이제 평가를 시작하세요.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt.strip()}]
        )
        report_str = response.choices[0].message.content.strip()
        return json.loads(report_str)
    except Exception as e:
        print(f"⚠️ [요약 보고서] GPT 호출 또는 JSON 파싱 오류: {e}")
        return {"error": str(e)}


# --- 📑 3. [보고서 2] 상세 소스 분석 보고서 생성 함수 ---

def generate_detailed_sources_report(
    query: str, 
    web_docs: List[Dict], 
    db_docs: List[Dict]
) -> Dict[str, Any]:
    """
    각 유사 사례(소스)를 상세히 분석하는 보고서(JSON)를 생성합니다.
    
    Args:
        query (str): 사용자 아이디어 쿼리.
        web_docs (List[Dict]): 웹 검색 결과 리스트.
        db_docs (List[Dict]): 내부 DB 검색 결과 리스트.

    Returns:
        Dict[str, Any]: 상세 소스 분석 결과가 포함된 JSON 객체.
    """
    # 소스 문서 블록 구성
    def make_doc_block(docs: List[Dict], prefix: str) -> str:
        return "\n\n".join([
            f"[{prefix}{i+1}] Title: {doc.get('title', '')}\nContent: {doc.get('snippet') or doc.get('content', '')}\nLink: {doc.get('link', '')}\nScore: {doc.get('score', '')}"
            for i, doc in enumerate(docs)
        ])

    web_block = make_doc_block(web_docs, "WEB")
    db_block = make_doc_block(db_docs, "DB")

    # 프롬프트 구성 (사용자 요청에 따라 원본 유지)
    prompt = f"""
당신은 창업 아이디어 분석 전문가입니다.

다음은 사용자의 아이디어입니다:
"{query}"

아래는 위 아이디어와 관련된 유사 자료 목록입니다. 각 자료를 개별적으로 분석하고 지정된 JSON 형식으로 결과를 출력해주세요.
---
[유사한 웹 문서]
{web_block}

[유사한 공모전 수상작 및 내부 DB 사례]
{db_block}
---

각 문서에 대해 다음 항목을 JSON 리스트 형식으로 평가해 주세요. 웹 문서는 "web", DB 사례는 "internal_db"로 `source_type`을 지정하세요.

[출력 JSON 형식]
```json
[
  {{
    "source_type": "web",
    "link": "Link 필드 참고",
    "thumbnail": null,
    "summary": "이 서비스는 XYZ 기술을 활용해 ABC 문제를 해결하는 내용.",
    "score": "Score 필드 참고하여 정수 또는 실수로 변환",
    "insight": "이 서비스의 사용자 대상 설정 방식이나 비즈니스 모델은 현재 아이디어에 참고할 만함."
  }},
  ...
]
```
다른 설명 없이 JSON 객체만 생성하세요. 이제 평가를 시작하세요.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-5-nano", # 최신 모델 사용 권장
            messages=[{"role": "user", "content": prompt.strip()}]
        )
        report_str = response.choices[0].message.content.strip()
        # GPT가 리스트를 감싸는 객체를 만들 경우를 대비하여 유연하게 파싱
        parsed_json = json.loads(report_str)
        if isinstance(parsed_json, dict) and len(parsed_json.keys()) == 1:
             # e.g., {"results": [...]} 형태일 경우
            results_list = next(iter(parsed_json.values()))
        else:
            results_list = parsed_json
            
        return {"query": query, "detailed_results": results_list}

    except Exception as e:
        print(f"⚠️ [상세 보고서] GPT 호출 또는 JSON 파싱 오류: {e}")
        return {"query": query, "detailed_results": [], "error": str(e)}