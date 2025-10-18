# app_chatbot/utils/db_search_utils.py

import os
import faiss
import pickle
from typing import List, Dict, Any
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


class FaissSearchEngine:
    '''
    SentenceTransformer와 FAISS를 활용한 벡터 유사도 검색 엔진 클래스
    '''
    def __init__(self, model_path: str):
        '''
        검색 엔진 초기화 함수
        '''
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"DBSearch 엔진 초기화 - device={self.device}")
        self.model = SentenceTransformer(model_path, device=self.device)
        self.index = None
        self.metadata = []

    def load_index(self, index_path: str, meta_path: str):
        '''
        FAISS 인덱스와 메타데이터 로드 함수
        '''
        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            raise FileNotFoundError("DB 인덱스 파일 또는 metadata.pkl 없음")
        self.index = faiss.read_index(index_path)
        with open(meta_path, 'rb') as f:
            self.metadata = pickle.load(f)
        print("DB 인덱스 로드 완료")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        '''
        벡터 기반 유사도 검색 함수
        '''
        if self.index is None:
            raise RuntimeError("검색 전에 load_index()를 먼저 호출해야 함")
        # 쿼리 임베딩 생성
        query_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        # 유사도 검색 수행
        distances, indices = self.index.search(query_vec, top_k)
        # 검색 결과 구성
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx != -1 and idx < len(self.metadata):
                item = self.metadata[idx].copy()
                item["score"] = float(score)
                results.append(item)
        return results


class DBSearch:
    '''
    일반 챗봇에서 사용하는 DB 검색 wrapper 클래스
    '''
    def __init__(self, model_path: str, index_path: str, meta_path: str):
        '''
        DBSearch wrapper 초기화 함수
        '''
        self.engine = FaissSearchEngine(model_path)
        self.engine.load_index(index_path, meta_path)

    def search(self, query: str, top_k: int = 5) -> List[str]:
        '''
        검색 결과를 문자열 리스트 형태로 반환하는 함수
        '''
        try:
            hits = self.engine.search(query, top_k=top_k)
            return [
                f"[{h.get('title', '제목 없음')}] {h.get('content', '')}"
                for h in hits
            ]
        except Exception as e:
            print(f"DB 검색 실패: {e}")
            return []
