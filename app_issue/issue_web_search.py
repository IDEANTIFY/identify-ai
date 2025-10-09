import os
import re
import requests
import pandas as pd


SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
SERPER_URL = "https://google.serper.dev/news"


def search_news(keywords, country="kr", num=10):
    """
    Serper (Google News) API를 사용해 뉴스 검색 (모든 키워드 포함)
    - keywords: 콤마로 구분된 문자열 또는 리스트
    - 공백으로 키워드를 연결하여 AND 검색 수행
    - 매칭된 키워드를 별도 컬럼에 표시
    """
    # 입력이 문자열이면 리스트로 변환
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

    rows = []
    try:
        res = requests.post(SERPER_URL, headers=headers, json=payload)
        res.raise_for_status()
        data = res.json()

        for item in data.get("news", []):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            # 기사 제목과 스니펫을 합쳐 키워드 매칭
            text = f"{title} {snippet}"
            matched = [kw for kw in keywords if re.search(re.escape(kw), text, re.IGNORECASE)]
            rows.append({"title": title,
                         "link": item.get("link"),
                         "source": item.get("source"),
                         "date": item.get("date"),
                         "image": item.get("imageUrl"),
                         "snippet": snippet,
                         "matched_keywords": ", ".join(matched)  # 실제 포함된 키워드 표시 -> 실제로 다 포함되어서 나온거지만 .. 
                         })
    except Exception as e:
        print(f"검색 실패: {e}")

    if not rows:
        return pd.DataFrame(columns=["title", "link", "source", "date", "image", "snippet", "matched_keywords"])
    return pd.DataFrame(rows).drop_duplicates(subset=["link"]).reset_index(drop=True)


## 사용 예시
## AND 검색: 웹서비스 AND 대학생 AND 아이디어
## df = search_news("웹서비스, 대학생, 아이디어", num=20)
## print(df.head())
