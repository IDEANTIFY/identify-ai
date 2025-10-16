import os
import re
import html
import copy
import pandas as pd
import torch
from html.parser import HTMLParser
from typing import Dict, List, Any
import concurrent.futures
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from langchain_tavily import TavilySearch
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_naver_community.tool import NaverSearchResults, NaverNewsSearch, NaverBlogSearch
from langchain_naver_community.utils import NaverSearchAPIWrapper

# 공통 설정값
from dotenv import load_dotenv
load_dotenv()

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
NAVER_CLIENT_ID = os.environ.get("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET")

# 검색 결과 개수 설정
TAVILY_MAX_RESULTS = 10         # Tavily에서 가져올 최대 결과 수
SERPER_MAX_RESULTS = 10         # Serper에서 가져올 최대 결과 수
NAVER_DISPLAY = 10              # Naver 검색에서 한 번에 가져올 결과 수
DUPLICATE_THR = 0.9             # 문서 간 중복 삭제 임계값

# Tavily 검색 상세 옵션
TAVILY_TOPIC = "general"        # "general" | "news" | "finance" 등 검색 주제
TAVILY_INCLUDE_ANSWER = True    # AI 요약 답변 포함 여부
TAVILY_INCLUDE_RAW = True       # 원문(raw_content) 포함 여부
TAVILY_INCLUDE_IMAGES = True    # 이미지 포함 여부
TAVILY_INCLUDE_IMG_DESC = False # 이미지 캡션 포함 여부
TAVILY_SEARCH_DEPTH = "advanced" # "basic" | "advanced" (심층 검색)
TAVILY_TIME_RANGE = None        # "day" | "week" | "month" 등 시간 범위 필터
TAVILY_INCLUDE_DOMAINS = None   # 특정 도메인만 포함 (리스트)
TAVILY_EXCLUDE_DOMAINS = None   # 특정 도메인 제외 (리스트)

# 임베딩 모델 설정
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"   # 한국어/다국어 지원 모델
EMBEDDING_BATCH_SIZE = 128                             # 배치 크기
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Cross-Encoder 모델 -> 리랭크를 위한..

# 전처리
class _Strip(HTMLParser):
    def __init__(self):
        super().__init__()
        self.buf = []
    def handle_data(self, d):
        if d and not d.isspace():
            self.buf.append(d)
    def text(self):
        return " ".join(self.buf)

def _strip_html(s: str) -> str:
    if not s:
        return ""
    p = _Strip()
    try:
        p.feed(s)
        p.close()
    except:
        pass
    return p.text()

_MD_IMG = re.compile(r"!\[[^\]]*\]\([^)]+?\)")
_MD_LNK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_WS = re.compile(r"\s+")

def _basic_clean(s: str) -> str:
    if not s:
        return ""
    s = html.unescape(s)
    s = _MD_IMG.sub("", s)
    s = _MD_LNK.sub(r"\1", s)
    s = _WS.sub(" ", s).strip()
    return s

def clean_tavily_results(data: dict, *, inplace: bool = False) -> dict:
    '''TavilySearch 결과를 받아 content/raw_content를 정제'''
    if not isinstance(data, dict):
        return {"error": "dict required"}
    out = data if inplace else copy.deepcopy(data)
    results = out.get("results")
    if isinstance(results, list):
        for i, it in enumerate(results):
            if not isinstance(it, dict):
                continue
            c = it.get("content")
            rc = it.get("raw_content")
            if isinstance(c, str):
                out["results"][i]["content"] = _basic_clean(c)
            if isinstance(rc, str):
                rc_txt = html.unescape(rc)
                rc_txt = _strip_html(rc_txt)
                out["results"][i]["raw_content"] = _basic_clean(rc_txt)
    return out

def tavily_to_df(data: dict) -> pd.DataFrame:
    j = clean_tavily_results(data, inplace=False)
    return pd.json_normalize(j.get("results") or [])

def clean_serper_results(results: dict) -> List[Dict[str, Any]]:
    '''Serper 결과에서 불필요한 문구를 제거'''
    organic = results.get("organic", []) or results.get("results", [])
    cleaned = []
    for r in organic:
        title = r.get("title", "").strip()
        link = r.get("link", "").strip()
        snippet = r.get("snippet", "") or r.get("description", "")
        snippet = html.unescape(snippet).strip()
        snippet = re.sub(r"\s+", " ", snippet).strip()
        cleaned.append({"title": title, "link": link, "content": snippet})
    return cleaned

# web search
tavily_tool = TavilySearch(
    max_results=TAVILY_MAX_RESULTS,
    topic=TAVILY_TOPIC,
    include_answer=TAVILY_INCLUDE_ANSWER,
    include_raw_content=TAVILY_INCLUDE_RAW,
    include_images=TAVILY_INCLUDE_IMAGES,
    include_image_descriptions=TAVILY_INCLUDE_IMG_DESC,
    search_depth=TAVILY_SEARCH_DEPTH,
    time_range=TAVILY_TIME_RANGE,
    include_domains=TAVILY_INCLUDE_DOMAINS,
    exclude_domains=TAVILY_EXCLUDE_DOMAINS)

serper_tool = GoogleSerperAPIWrapper(
    type="search",
    k=SERPER_MAX_RESULTS,
    gl="kr",
    hl="ko")

search = NaverSearchAPIWrapper()
naver_tool = NaverSearchResults(api_wrapper=search)
naver_news_tool = NaverNewsSearch(api_wrapper=search)
naver_blog_tool = NaverBlogSearch(api_wrapper=search)


# 검색 실행 및 병합
def _search_tavily(q: str) -> pd.DataFrame:
    res = tavily_tool.invoke({"query": q})
    return tavily_to_df(res)

def _search_serper(q: str) -> pd.DataFrame:
    res = serper_tool.results(q)
    return pd.DataFrame(clean_serper_results(res))

def _search_naver(q: str) -> pd.DataFrame:
    res = pd.DataFrame(naver_blog_tool.invoke({"query": q, "display": NAVER_DISPLAY, "start": 1, "sort": "sim"}))
    return pd.DataFrame(res)

def fetch_all_search_results(query: str) -> pd.DataFrame:
    '''
    Tavily / Serper / Naver를 병렬 호출 후 통합된 DataFrame을 반환
    -> 후처리: score 컬럼 제거, url 값이 있으면 link 컬럼에 통합, link 기준 중복 제거
    '''
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(_search_tavily, query): "tavily",
                   ex.submit(_search_serper, query): "serper",
                   ex.submit(_search_naver, query): "naver"}
        collected = []
        for f in concurrent.futures.as_completed(futures):
            try:
                df = f.result()
                if not df.empty:
                    df["api_source"] = futures[f]
                    collected.append(df)
            except Exception as e:
                print(f"{futures[f]} 호출 실패: {e}")
    if not collected:
        return pd.DataFrame()
    df = pd.concat(collected, ignore_index=True)
    # score 컬럼 제거
    if "score" in df.columns:
        df = df.drop(columns=["score"])
    # url → link 통합
    if "url" in df.columns:
        if "link" not in df.columns:
            df["link"] = df["url"]
        else:
            df["link"] = df["link"].fillna(df["url"])
        df = df.drop(columns=["url"])
    # 칼럼 순서 정리
    ordered_cols = [c for c in ["title", "link", "description", "content", "api_source"] if c in df.columns]
    ordered_cols += [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols]
    # 링크 기준 중복 제거
    df = df.drop_duplicates(subset=["link"], keep="first").reset_index(drop=True)
    return df


# 임베딩 기반 유사도 재정렬 + Cross-Encoder
bi_encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)

def rerank_results(df: pd.DataFrame, query: str, duplicate_threshold: float = 0.9,
                   cross_top_k: int = 50) -> pd.DataFrame:
    '''
    Bi-Encoder 기반 1차 정렬 + 중복 제거 후
    Cross-Encoder로 상위 cross_top_k개만 재정렬
    '''
    if df.empty:
        return df

    # 텍스트 컬럼 선택
    if "content" in df.columns:
        texts = df["content"].fillna("").tolist()
    elif "snippet" in df.columns:
        texts = df["snippet"].fillna("").tolist()
    else:
        texts = df.iloc[:, 0].astype(str).fillna("").tolist()

    # --- Bi-Encoder ---
    q_emb = bi_encoder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    doc_embs = bi_encoder.encode(texts, batch_size=EMBEDDING_BATCH_SIZE, convert_to_tensor=True, normalize_embeddings=True)
    cos_sim = util.cos_sim(q_emb, doc_embs)[0]
    df = df.copy()
    df["bi_score"] = cos_sim.cpu().tolist()
    df = df.sort_values("bi_score", ascending=False).reset_index(drop=True)

    # --- 문서 간 중복 제거 ---
    keep_idx = []
    seen = []
    for i, emb in enumerate(doc_embs):
        if not seen:
            keep_idx.append(i)
            seen.append(emb)
            continue
        sims = util.cos_sim(emb.unsqueeze(0), torch.stack(seen))[0]
        if sims.max().item() < duplicate_threshold:
            keep_idx.append(i)
            seen.append(emb)
    df = df.iloc[keep_idx].reset_index(drop=True)

    # --- Cross-Encoder (상위만 재정렬) ---
    top_k = min(len(df), cross_top_k)
    pairs = [[query, t] for t in df.loc[:top_k - 1, "content"].fillna("").tolist()]
    if pairs:
        cross_scores = cross_encoder.predict(pairs)
        df.loc[:top_k - 1, "cross_score"] = cross_scores
        df_top = df.iloc[:top_k].sort_values("cross_score", ascending=False)
        df_rest = df.iloc[top_k:]
        df = pd.concat([df_top, df_rest], ignore_index=True)

    return df


# 파이프라인
def run_web_search_pipeline(query: str, top_k: int = None) -> pd.DataFrame:
    """
    통합 검색 → Bi-Encoder 유사도 → Cross-Encoder 재정렬 → 상위 k개 반환
    top_k가 None이면 전체 결과 반환
    """
    raw_df = fetch_all_search_results(query)
    ranked_df = rerank_results(raw_df, query, DUPLICATE_THR)
    if top_k is None:
        return ranked_df
    return ranked_df.head(top_k)

## 사용법
## run_web_search_pipeline(query, 15)
