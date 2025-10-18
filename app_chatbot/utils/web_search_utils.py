# app_chatbot/utils/web_search_utils.py

import os
from typing import List
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_community.utilities import GoogleSerperAPIWrapper
load_dotenv()

class WebSearch:
    '''
    Tavily와 Google Serper를 활용한 웹 검색 기능 클래스
    '''
    def __init__(self):
        '''
        웹 검색 도구 초기화 함수
        '''
        # 사용 가능 여부 확인
        self.use_tavily = bool(os.getenv("TAVILY_API_KEY")) and TavilySearch is not None
        self.use_serper = bool(os.getenv("SERPER_API_KEY")) and GoogleSerperAPIWrapper is not None
        # Tavily 초기화
        if self.use_tavily:
            self.tavily = TavilySearch(max_results=5)
        # Serper 초기화
        if self.use_serper:
            self.serper = GoogleSerperAPIWrapper(gl=os.getenv("SERPER_API_GEO", "kr"), hl="ko")

    def search(self, query: str, top_k: int = 5) -> List[str]:
        '''
        검색 쿼리에 대한 웹 검색 결과를 문자열 리스트로 반환하는 함수
        '''
        results = []
        # Tavily 우선 검색
        if self.use_tavily:
            try:
                t_res = self.tavily.invoke({"query": query})
                for item in t_res.get("results", [])[:top_k]:
                    results.append(f"{item.get('title', '')} - {item.get('content', '')}")
            except Exception:
                pass

        # Serper 백업 검색
        if not results and self.use_serper:
            try:
                s_res = self.serper.results(query).get("organic", [])
                for item in s_res[:top_k]:
                    results.append(f"{item.get('title', '')} - {item.get('snippet', '')}")
            except Exception:
                pass

        return results[:top_k]
