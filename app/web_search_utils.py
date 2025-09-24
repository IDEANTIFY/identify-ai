import os
import re
import json
import torch
import requests
import pandas as pd
from tavily import TavilyClient
from urllib.parse import quote
from typing import Dict, List, Any
import concurrent.futures
from sentence_transformers import SentenceTransformer, util


# --- ⚙️ API 설정 및 클라이언트 초기화 ---
# 보안을 위해 API 키는 코드에 직접 작성하는 대신 환경 변수에서 불러옵니다.
# 셸에서 export TAVILY_API_KEY="당신의키" 와 같이 설정할 수 있습니다.
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
NAVER_CLIENT_ID = os.environ.get("NAVER_CLIENT_ID", "")
NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET", "")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
HTTP_SESSION = requests.Session()  # 세션 객체 재사용

SERPER_HEADERS = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
NAVER_HEADERS = {"X-Naver-Client-Id": NAVER_CLIENT_ID, "X-Naver-Client-Secret": NAVER_CLIENT_SECRET}

# --- 헬퍼 함수 ---
def _clean_html(raw_html: str) -> str:
    """HTML 태그를 제거하는 간단한 정규식 함수"""
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def _contains_korean(text: str) -> bool:
    """텍스트에 한국어가 포함되어 있는지 확인하는 함수"""
    return bool(re.search(r'[가-힣]', text))

# --- 개별 API 호출 함수 ---
def _search_tavily(query: str) -> List[Dict[str, Any]]:
    """Tavily API를 호출하여 검색 결과를 반환합니다."""
    try:
        resp = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=10,
            country="south korea"
        )
        return resp.get("results", [])
    except Exception as e:
        print(f"[오류] Tavily 검색 실패: {e}")
        return []

def _search_serper(query: str) -> List[Dict[str, Any]]:
    """Serper API를 호출하여 검색 결과를 반환합니다."""
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    try:
        response = requests.post(url, headers=SERPER_HEADERS, data=payload)
        response.raise_for_status()
        return response.json().get("organic", [])
    except Exception as e:
        print(f"[오류] Serper 검색 실패: {e}")
        return []

def _search_naver(query: str) -> List[Dict[str, Any]]:
    """Naver Search API를 호출하여 검색 결과를 반환합니다."""
    url = f"https://openapi.naver.com/v1/search/webkr.json?query={quote(query)}&display=10&sort=sim"
    try:
        response = requests.get(url, headers=NAVER_HEADERS)
        response.raise_for_status()
        return response.json().get("items", [])
    except Exception as e:
        print(f"[오류] Naver 검색 실패: {e}")
        return []


def fetch_all_search_results(query: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Tavily, Serper, Naver API를 병렬로 호출하여 모든 검색 결과를 가져옵니다.
    
    Input:
        - query (str): 검색할 쿼리 문자열

    Output:
        - Dict[str, List[Dict[str, Any]]]: API 소스별로 정리된 결과 딕셔너리
          예: {'tavily': [...], 'serper': [...], 'naver': [...]}
    """
    # ThreadPoolExecutor를 사용하여 각 API 호출을 병렬로 실행합니다.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_api = {
            executor.submit(_search_tavily, query): "tavily",
            executor.submit(_search_serper, query): "serper",
            executor.submit(_search_naver, query): "naver"
        }
        
        results = {"tavily": [], "serper": [], "naver": []}
        for future in concurrent.futures.as_completed(future_to_api):
            api_name = future_to_api[future]
            try:
                results[api_name] = future.result()
            except Exception as exc:
                print(f"[오류] {api_name} API 호출 중 예외 발생: {exc}")
    
    return results

# --- 결과 처리 함수 (Pandas 연산 최적화) ---
def process_and_merge_results(raw_results: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    all_dfs = []
    # 1. Tavily 결과 처리 (벡터화 연산으로 변경)
    if raw_results.get('tavily'):
        df = pd.DataFrame(raw_results['tavily']).rename(columns={'content': 'snippet', 'url': 'link'})
        df["score"] = pd.to_numeric(df.get("score"), errors="coerce").fillna(0)
        if not df.empty and df["score"].max() > 0:
            adaptive_thr = max(0.3, df["score"].max() * 0.6)
            df = df[df["score"] >= adaptive_thr]
        
        # [최적화] title 또는 snippet에 한국어가 포함된 행을 필터링
        korean_pattern = re.compile(r'[가-힣]')
        df = df[
            df['title'].astype(str).str.contains(korean_pattern) |
            df['snippet'].astype(str).str.contains(korean_pattern)
        ]
        df['api_source'] = 'tavily'
        all_dfs.append(df[['title', 'snippet', 'link', 'api_source']])
    
    # (Serper, Naver 처리는 기존 로직이 효율적이므로 유지)
    if raw_results.get('serper'):
        df = pd.DataFrame(raw_results['serper'])
        df['snippet'] = df.get('snippet', '') + " " + df.get('description', '')
        df['snippet'] = df['snippet'].str.strip()
        df = df[~df['snippet'].str.contains('Missing', na=False)]
        df['api_source'] = 'serper'
        all_dfs.append(df[['title', 'snippet', 'link', 'api_source']])
    if raw_results.get('naver'):
        df = pd.DataFrame(raw_results['naver']).rename(columns={'description': 'snippet'})
        df['title'] = df['title'].apply(_clean_html)
        df['snippet'] = df['snippet'].apply(_clean_html)
        df['api_source'] = 'naver'
        all_dfs.append(df[['title', 'snippet', 'link', 'api_source']])

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset='link', keep='first').reset_index(drop=True)

# --- 유사도 재정렬 함수 (GPU 사용 최적화) ---
def rerank_results_by_similarity(
    results_df: pd.DataFrame,
    reference_text: str,
    model: SentenceTransformer
) -> pd.DataFrame:
    if results_df.empty:
        return results_df

    # [최적화] batch_size를 지정하여 GPU 메모리를 효율적으로 사용
    snippet_list = results_df['snippet'].fillna("").tolist()
    snippet_embeddings = model.encode(
        snippet_list, 
        batch_size=128,  # GPU 성능에 따라 조절 가능
        convert_to_tensor=True, 
        normalize_embeddings=True,
        show_progress_bar=True # 진행 상황 확인
    )
    
    ref_embedding = model.encode(
        reference_text, 
        convert_to_tensor=True, 
        normalize_embeddings=True
    )
    
    cos_scores = util.cos_sim(ref_embedding, snippet_embeddings)[0]
    
    results_df['similarity_score'] = cos_scores.cpu().tolist()
    return results_df.sort_values(by='similarity_score', ascending=False).reset_index(drop=True)


# --- (fetch_all_search_results, get_top_n_results, run_web_search_pipeline, __main__ 등 나머지 코드는 이전과 동일) ---
def fetch_all_search_results(query: str) -> Dict[str, List[Dict[str, Any]]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_api = {
            executor.submit(_search_tavily, query): "tavily",
            executor.submit(_search_serper, query): "serper",
            executor.submit(_search_naver, query): "naver"
        }
        results = {"tavily": [], "serper": [], "naver": []}
        for future in concurrent.futures.as_completed(future_to_api):
            api_name = future_to_api[future]
            try:
                results[api_name] = future.result()
            except Exception as exc:
                print(f"⚠️ [{api_name} API 호출 중 예외 발생] {exc}", file=sys.stderr)
    return results

def get_top_n_results(df: pd.DataFrame, n: int = 3) -> List[Dict[str, Any]]:
    return df.head(n).to_dict('records')

def run_web_search_pipeline(query: str, model: SentenceTransformer) -> pd.DataFrame:
    print(f"🔍 1. '{query}'에 대한 웹 검색을 시작합니다...", flush = True)
    raw_results = fetch_all_search_results(query)
    print(f"📊 2. Tavily: {len(raw_results.get('tavily',[]))}, Serper: {len(raw_results.get('serper',[]))}, Naver: {len(raw_results.get('naver',[]))} 건의 결과를 가져왔습니다.", flush = True)
    merged_df = process_and_merge_results(raw_results)
    print(f"✅ 3. 총 {len(merged_df)} 건의 고유한 검색 결과로 통합되었습니다.", flush = True)
    print("✨ 4. 원본 쿼리와의 의미적 유사도를 기준으로 결과를 재정렬합니다...", flush = True)
    reranked_df = rerank_results_by_similarity(merged_df, query, model)
    return reranked_df

# if __name__ == "__main__":
    # search_query = "LLM을 활용한 개인화 추천 시스템 구축 사례"
    # final_results_df = run_web_search_pipeline(search_query)
    
    # if not final_results_df.empty:
        # pd.set_option("display.max_colwidth", 70)
        # print("\n--- 🏆 최종 재정렬된 검색 결과 (상위 10개) ---", flush = True)
        # print(final_results_df.head(10), flush = True)
        # top_3_list = get_top_n_results(final_results_df, n=3)
        # print("\n--- 🎯 상위 3개 결과 (활용 예시) ---", flush = True)
        # for i, item in enumerate(top_3_list):
            # print(f"[{i+1}]\n  - Title: {item['title']}\n  - Link: {item['link']}\n  - Score: {item['similarity_score']:.4f}\n", flush = True)
    # else:
        # print("\n--- 최종 검색 결과가 없습니다 ---", flush = True)