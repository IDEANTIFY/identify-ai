# app_issue/issue_web_search.py

import os
import re
import requests
import json
from fastapi import APIRouter
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
SERPER_URL = "https://google.serper.dev/news"

router = APIRouter()


def search_news(keywords, country="kr", num=10):
    """
    Serper (Google News) API를 사용해 뉴스 검색 (모든 키워드 포함)
    - keywords: 콤마로 구분된 문자열 또는 리스트
    - 공백으로 키워드를 연결하여 AND 검색 수행
    - 매칭된 키워드를 별도 컬럼에 표시
    """
    if isinstance(keywords, str):
        keywords = [kw.strip() for kw in keywords.split(",") if kw.strip()]

    # AND 쿼리 생성 (모든 키워드가 포함된 결과)
    query = " ".join(keywords)
    headers = {"X-API-KEY": SERPER_API_KEY,
               "Content-Type": "application/json"}
    payload = {"q": query,
               "gl": country,
               "hl": "ko",
               "num": num}

    results_list = []
    seen_links = set() # 중복 링크 제거
    try:
        res = requests.post(SERPER_URL, headers=headers, json=payload)
        res.raise_for_status()
        data = res.json()

        for item in data.get("news", []):
            link = item.get("link")
            # 링크가 없거나, 이미 추가된 링크라면 건너뛰기
            if not link or link in seen_links:
                continue
            seen_links.add(link)
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            text = f"{title} {snippet}"
            matched = [kw for kw in keywords if re.search(re.escape(kw), text, re.IGNORECASE)]
            results_list.append({
                "title": title,
                "link": link,
                "source": item.get("source"),
                "date": item.get("date"),
                "image": item.get("imageUrl"),
                "snippet": snippet,
                "matched_keywords": ", ".join(matched)
            })
    except Exception as e:
        print(f"뉴스 검색 API 오류: {e}")

    # 파일 저장 로직 
    save_dir = "../dataset/issue"
    file_name = "news.json"
    save_path = os.path.join(save_dir, file_name)
    try:
        # 디렉토리 생성 (없는 경우)
        os.makedirs(save_dir, exist_ok=True)
        # results_list를 JSON으로 저장
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(
                results_list, 
                f, 
                ensure_ascii=False,
                indent=4
            )
        print(f"검색 결과 {len(results_list)}건을 {save_path}에 성공적으로 저장했습니다.")
    except Exception as e:
        print(f"파일 저장 중 오류 발생 ({save_path}): {e}")
    return results_list

# ===============================
# FastAPI 엔드포인트
# ===============================

class IssueRequest(BaseModel):
    keywords: list[str]

@router.post("/search/news")
async def search_news_api(req: IssueRequest):
    """
    백엔드에서 AI 서버로 POST 요청 시,
    키워드 기반 뉴스 검색 결과를 JSON으로 반환
    """
    results = search_news(req.keywords)
    return {"count": len(results), "results": results}
