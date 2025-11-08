import os
import re
import requests
import json
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
SERPER_URL = "https://google.serper.dev/news"


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
        df = pd.DataFrame(columns=["title", "link", "source", "date", "image", "snippet", "matched_keywords"])
        return df.to_json(orient="records", force_ascii=False, indent=2)
    df = pd.DataFrame(rows).drop_duplicates(subset=["link"]).reset_index(drop=True)
    return df.to_json(orient="records", force_ascii=False, indent=2)



def load_user_keywords(user_id="A000"):
    """
    user_info JSON 파일에서 keyword 필드를 읽어옴
    - user_id: 사용자 ID (기본값: A000)
    - returns: 키워드 리스트 또는 None
    """
    # 현재 실행 파일 기준으로 프로젝트 루트의 dataset 폴더 경로 설정
    project_root = Path(__file__).parent.parent
    user_info_path = project_root / "dataset" / "user_info" / f"{user_id}.json"
    
    try:
        with open(user_info_path, "r", encoding="utf-8") as f:
            user_data = json.load(f)
        
        # keyword 필드가 있으면 사용, 없으면 interests 사용
        keywords = user_data.get("keyword")
        if keywords is None:
            keywords = user_data.get("interests", [])
        
        # 리스트가 아닌 경우 문자열로 변환하여 처리
        if isinstance(keywords, list):
            keywords = ", ".join(keywords)
        elif isinstance(keywords, str):
            pass
        else:
            raise ValueError(f"키워드 형식이 올바르지 않습니다: {type(keywords)}")
        
        return keywords
    except FileNotFoundError:
        print(f"오류: {user_info_path} 파일을 찾을 수 없습니다.")
        return None
    except json.JSONDecodeError:
        print(f"오류: {user_info_path} 파일의 JSON 형식이 올바르지 않습니다.")
        return None
    except Exception as e:
        print(f"오류: {e}")
        return None


if __name__ == "__main__":
    # user_info에서 키워드 로드
    user_id = "A000"  # 필요시 변경 가능
    keywords = load_user_keywords(user_id)
    
    if keywords is None:
        print("키워드를 불러올 수 없습니다. 프로그램을 종료합니다.")
        exit(1)
    
    # 뉴스 검색 수행
    issue_news_search = search_news(keywords)
    
    # 현재 실행 파일 기준으로 프로젝트 루트의 dataset/user_info 폴더에 저장
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "dataset" / "user_info"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"issue_news_search_{user_id}.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(issue_news_search)

    print(f"JSON 파일 생성 완료: {output_path}")
