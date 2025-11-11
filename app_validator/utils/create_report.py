import os
import json
from openai import OpenAI
from typing import List, Dict, Any
from dotenv import load_dotenv
import re

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# [보고서 1] 요약 보고서 생성 함수
def generate_summary_report(
    query: Dict[str, Any], 
    web_docs: List[Dict], 
    crawling_docs: List[Dict],   # 외부 크롤링
    user_docs: List[Dict],   # user_docs
    approx_similar_count: int
) -> Dict[str, Any]:
    """
    아이디어를 정량적으로 평가하는 요약 보고서(JSON)를 생성
    """
    query_str = json.dumps(query, ensure_ascii=False, indent=2) 

    # 웹 및 DB 문서 섹션 구성
    web_section = "\n\n".join([
        f"[WEB{i+1}] Title: {doc.get('title', '')}\n"
        f"Snippet: {doc.get('snippet', '')}\n"
        f"Source: {doc.get('source', 'N/A')} (Link: {doc.get('link', 'N/A')})\n"
        for i, doc in enumerate(web_docs)
    ])
    
    # 정적 DB 문서 섹션 구성 (크롤링 DB)
    crawling_section = "\n\n".join([ 
        f"[CRAWLING_DB{i+1}] Title: {doc.get('title', 'N/A')}\n" 
        f"Content: {doc.get('content', '')}\n"
        f"Keyword: {doc.get('keyword', 'N/A')}\n"
        f"Team Members: {', '.join(doc.get('team_members', [])) or 'N/A'}\n"
        f"image: {doc.get('thumbnail_url', '')}"
        f"Source: {doc.get('source', 'N/A')} (Link: {doc.get('link', 'N/A')})\n"
        f"Score: {doc.get('score', '')}"
        for i, doc in enumerate(crawling_docs) 
    ])
    # 유저 DB 구성
    user_section = "\n\n".join([ 
        f"[USER_DB{i+1}] Title: {doc.get('title', 'N/A')}\n"
        f"Content: {doc.get('content', '')}\n"
        f"Keyword: {doc.get('keyword', 'N/A')}\n"
        f"Team Members: {', '.join(doc.get('team_members', [])) or 'N/A'}\n"
        f"image: {doc.get('avatar', '')}"
        f"Source: {doc.get('source', 'N/A')} (Link: {doc.get('link', 'N/A')})\n"
        f"Score: {doc.get('score', '')}"
        for i, doc in enumerate(user_docs) 
    ])


    prompt = f"""
    당신은 대학생 스타트업 아이디어를 정량적 기준에 따라 평가하고, 결과를 반드시 JSON 형식으로 반환하는 전문가입니다.

    [사용자 아이디어]
    다음은 사용자의 아이디어입니다:
    {query_str}

    [유사 사례 데이터]
    1. 웹 검색 상위 {len(web_docs)}개 결과:
    {web_section}

    2. 크롤링 DB (공모전 등 정적 데이터):
    {crawling_section}

    3. 유저 DB (outer_project 실시간 데이터):
    {user_section}

    다음 기준에 따라 아이디어를 평가하고, 결과를 아래에 명시된 JSON 형식 예시에 정확히 맞춰 존댓말로 출력해 주세요.

    [평가 기준]
    1.  total_similar_cases: 웹 검색 결과({approx_similar_count}건)와 크롤링 DB, 유저 DB 결과 문서 수를 종합하여 전체 유사 건수를 정수로 판단.
    2.  similarity: 크롤링 DB + 유저 DB + 웹 문서 기준, 기술적 컨셉의 중복성을 판단하여 0~100 사이 정수 값으로 평가.
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
        match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", report_str)
        if match:
            report_str = match.group(1)
        
        return json.loads(report_str)
    except Exception as e:
        print(f"⚠️ [요약 보고서] GPT 호출 또는 JSON 파싱 오류: {e}")
        return {"error": str(e)}


# [보고서 2] 상세 소스 분석 보고서 생성 함수
def generate_detailed_sources_report(
    query: Dict[str, Any],
    web_docs: List[Dict], 
    crawling_docs: List[Dict], 
    user_docs: List[Dict]
) -> Dict[str, Any]:
    """
    각 유사 사례(소스)를 상세히 분석하는 보고서(JSON)를 생성
    """

    query_str = json.dumps(query, ensure_ascii=False, indent=2)
    
    def make_doc_block(docs: List[Dict], prefix: str) -> str:
        blocks = []
        for i, doc in enumerate(docs):
            blocks.append(
                f"[{prefix}{i+1}] Title: {doc.get('title', 'N/A')}\n"
                f"Content: {doc.get('content') or doc.get('snippet', 'N/A')}\n"
                f"Source: {doc.get('source', 'N/A')} (Link: {doc.get('link', 'N/A')})\n"
                f"Keyword: {doc.get('keyword', 'N/A')}\n"
                f"Team Members: {', '.join(doc.get('team_members', [])) or 'N/A'}\n"
                f"image: {doc.get('thumbnail_url') or doc.get('avatar', '')}\n"
                f"Date: {doc.get('date') or doc.get('updatedAt', 'N/A')}\n"
                f"Score: {doc.get('score', 'N/A')}"
            )
        return "\n\n".join(blocks)

    all_db_docs = crawling_docs + user_docs
    web_block = make_doc_block(web_docs, "WEB")
    db_block = make_doc_block(all_db_docs, "DB")
    
    prompt = f"""
    당신은 창업 아이디어 분석 전문가입니다.

    다음은 사용자의 아이디어입니다:
    "{query}"

    아래는 위 아이디어와 관련된 유사 자료 목록입니다. 각 자료를 개별적으로 분석하고 지정된 JSON 형식으로 결과를 존댓말로 출력해주세요.
    [유사한 웹 문서]
    {web_block}

    [유사한 공모전 수상작 및 내부 DB 사례]
    {db_block}

    각 문서에 대해 다음 항목을 JSON 리스트 형식으로 평가해 주세요.

    [평가 기준]
    1. 'source_type': "web" 또는 "internal_db"로 지정
    2. 'title': 원본 문서 제목을 기반으로 생성
    3. 'summary': 문서 요약
    4. 'score': 유사도 점수 (정수 또는 실수)
    5. 'insight': 이 유사 사례의 'Keyword' 필드, 'Team Members' 필드, 'Link' 필드, 'Source' 필드를 활용하여, 현재 아이디어와 이 사례의 차이점 또는 참고할 만한 비즈니스 모델을 중점적으로 분석하여 한국어 존댓말로 서술하세요. Insight는 아래 [Insight 출력 구조 예시]에 따라 작성되어야 합니다.
    6. 'image': 입력으로 제공된 이미지 URL을 **절대 변경 없이 그대로 복사하여 출력**하세요.  
    [Insight 출력 구조 예시] <기존 아이디어> 개별 일정 관리 중심 단일 사용자 위주 설계 <내 아이디어> 팀플 일정 자동 추천: 캘린더 데이터 기반으로 팀 전체 일정 최적 시간대 자동 제안 프라이버시 존중 일정 공유: 업무명은 숨기고 상태(바쁨/여유)만 표시 → 부담 최소화 집중 모드 연계: 일정 시작 시 자동으로 ‘방해금지 모드 + 모각작 집중방 참여’ 연동 크로스 툴 연결: Notion/Trello와 양방향 싱크 → 프로젝트 관리 + 일정관리 통합

    [출력 JSON 형식]

    ```json

    {{
    "detailed_results": [
        {{
        "source_type": "internal_db",
        "title": "내부 공모전: 스마트 물류 최적화 시스템",
        "summary": "AI 기반 물류 창고 관리 및 경로 최적화 솔루션을 통해 배송 시간을 단축하고 운영 비용을 절감하는 프로젝트. 주요 기술은 딥러닝 예측 모델입니다.",
        "score": 85.5,
        "insight": "<기존 아이디어>\n**물류 창고** 대상 B2B 솔루션\n재고 및 경로 예측 최적화 중심\n<내 아이디어>\n**도심 라스트 마일** 배송 효율화: 소규모/다중 배송지 최적 경로 실시간 제안\n자율 드론 배송 통합 모델: 특정 지역(캠퍼스/신도시) 내 **드론 연계** 파일럿 구축\n수익 모델 참고: 초기 설치비 없는 **구독형 SaaS** 및 성능 개선 시 성과 공유 모델 도입\n(Keyword: AI, 물류 최적화, 딥러닝, Team Members: 5명, Source: 내부DB)"
        "image": "입력으로 주어진 URL을 변경 없이 그대로 출력해야 합니다. 생성 금지."
        }},
        {{
        "source_type": "web",
        "title": "2024년 전국 대학생 아이디어 경진대회 최우수상: 지속가능한 폐기물 관리 플랫폼",
        "summary": "블록체인 기술을 활용하여 폐기물 배출부터 처리까지 전 과정을 투명하게 기록하고, 인센티브를 제공하여 시민 참여를 유도하는 플랫폼입니다.",
        "score": 72,
        "insight": "<기존 아이디어>\n**블록체인** 기반의 **환경** 문제 해결 플랫폼\n폐기물 투명성 및 시민 인센티브 제공 중심\n<내 아이디어>\n**대학 생활** 환경 특화: 교내 공용 물품/폐기물 순환 및 중고 거래 통합 관리\n**인센티브 모델** 확장: 토큰을 교내 카페, 도서 대여 할인 등 **실질적 보상**과 연계\n기술 참고: 블록체인 대신 AI 기반 **탄소 발자국** 측정으로 기술 차별화 모색\n(Keyword: 블록체인, 환경, 인센티브, Team Members: 4명, Source: 경진대회 공식 홈페이지)"
        }}
    ]
    }}
    ```
    다른 설명 없이 JSON 객체만 생성하세요. 이제 평가를 시작하세요.
    """
    
    results_list = web_docs + all_db_docs 

    try:
        response = client.chat.completions.create(
            model="gpt-5-nano", 
            messages=[{"role": "user", "content": prompt.strip()}]
        )
        report_str = response.choices[0].message.content.strip()
        match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", report_str)
        if match:
            report_str = match.group(1)
        parsed_json = json.loads(report_str)
        return {"query": query, "detailed_results": parsed_json.get("detailed_results", []), "raw_report": parsed_json}
    except Exception as e:
        print(f"⚠️ [상세 보고서] GPT 호출 또는 JSON 파싱 오류: {e}")
        return {"query": query, "detailed_results": [], "error": str(e)}



if __name__ == "__main__":    
    # 1. 사용자 아이디어 쿼리
    test_query = {
        "title": "내 마음의 친구, AI 멘탈 케어 챗봇 'MindMate'",
        "content": "개인의 감정 상태, 과거 대화 이력 및 성향을 딥러닝으로 분석하여, 맞춤형 상담 스크립트와 정서적 지원을 제공하는 24시간 멘탈 케어 챗봇 서비스. 사용자에게 심리 전문가 연결 옵션도 제공."
    }

    # 2. 웹 검색 결과 (테스트용 더미)
    test_web_docs = [
        {
            "title": "AI Mental Health Chatbot Trends 2024",
            "snippet": "AI 기반 정신 건강 챗봇 서비스 확장 중",
            "source": "TechCrunch",
            "link": "https://techcrunch.com/test",
            "keyword": "AI, Mental Health",
            "team_members": [],
            "date": "2024-01-10",
            "score": 0.72
        }
    ]

    # 3. 크롤링 DB (공모전 등 정적 DB)
    test_crawling_docs = [
        {
            "title": "공모전: AI 심리케어 플랫폼",
            "content": "대학생 대상 스트레스 관리 AI 플랫폼",
            "source": "K-Startup",
            "link": "https://kstartup.test",
            "keyword": ["AI", "심리"],
            "team_members": ["김철수", "이영희"],
            "thumbnail_url": "dkdididi",
            "date": "2023-11-21",
            "score": 0.81
        }
    ]

    # 4. 유저 DB (라이브 데이터)
    test_user_docs = [
        {
            "title": "감정 분석 기반 상담 챗봇",
            "content": "사용자 감정 로그 분석 기반 조언 제공",
            "source": "UserDB",
            "link": "https://outerproject.test",
            "keyword": ["감정", "상담"],
            "team_members": ["홍길동"],
            "avatar": "dkdididi",
            "updatedAt": "2025-01-05",
            "score": 0.67
        }
    ]

    # print("\n 1. 요약 분석 시작 \n")

    # summary_report = generate_summary_report(
    #     query=test_query,
    #     web_docs=test_web_docs,
    #     crawling_docs=test_crawling_docs,
    #     user_docs=test_user_docs,
    #     approx_similar_count=5
    # )

    # print(json.dumps(summary_report, ensure_ascii=False, indent=2))

    print("\n 2. 상세 소스 분석 시작 \n")

    detailed_report = generate_detailed_sources_report(
        query=test_query,
        web_docs=test_web_docs,
        crawling_docs=test_crawling_docs,
        user_docs=test_user_docs
    )

    print(json.dumps(detailed_report, ensure_ascii=False, indent=2))
