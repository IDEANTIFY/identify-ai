## crawling_db_search_utils.py

import os
import faiss
import pickle
import pandas as pd
import torch
import uuid
import numpy as np
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any


class CrawlingdbSearchEngine:
    """
    Hybrid Retrieval Engine:
    BM25(Sparse) + FAISS(Dense) + (Optional) CrossEncoder Re-ranking
    각 단계별 검색 속도 측정
    """

    def __init__(self, model_path: str, use_reranker: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"'{self.device}' 장치를 사용하여 모델을 로딩합니다.", flush=True)
        self.model = SentenceTransformer(model_path, device=self.device)

        self.index = None
        self.metadata = []
        self.embeddings = None
        self.bm25 = None
        self.bm25_corpus = None
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") if use_reranker else None

    # 슬라이딩 윈도우 청킹 함수
    def _chunk_text(self, text: str, chunk_size: int = 150, overlap: int = 30):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.split()) >= 30:
                chunks.append(chunk)
        return chunks

    # 인덱스 구축
    def build_index(self, csv_path: str, index_path: str, meta_path: str):
        # Path 객체를 문자열로 변환
        csv_path_str = str(csv_path)
        index_path_str = str(index_path)
        meta_path_str = str(meta_path)
        
        print(f"'{csv_path_str}' 파일로부터 인덱스 구축을 시작합니다...", flush=True)
        start_total = time.time()

        # 데이터 로드 및 전처리
        t0 = time.time()
        df = pd.read_csv(csv_path_str)
        # detail_url은 선택 필드이므로 필수 필드에서 제외 (비어있어도 허용)
        df.dropna(
            subset=["title", "content", "upper_keyword", "lower_keyword", "producer"],
            inplace=True,
        )
        # detail_url이 없거나 비어있는 경우 빈 문자열로 채움
        df["detail_url"] = df["detail_url"].fillna("")
        df["content_length"] = df["content"].apply(lambda x: len(str(x).split()))
        df = df[df["content_length"] > 10].reset_index(drop=True)
        print(f"전처리 완료: {len(df)}개 문서 (소요: {time.time() - t0:.2f}s)", flush=True)

        # 텍스트 청킹 및 메타데이터 확장
        t1 = time.time()
        texts_to_embed = []
        expanded_metadata = []
        metadata_cols = ["title", "content", "date", "detail_url", "upper_keyword", "lower_keyword", "producer"]

        for _, row in df.iterrows():
            base_meta = row[metadata_cols].fillna("").to_dict()

            # 제목 + 키워드 조합
            chunk_title_keyword = f"{base_meta['title']} {base_meta['upper_keyword']} {base_meta['lower_keyword']}".strip()
            if chunk_title_keyword:
                texts_to_embed.append(chunk_title_keyword)
                meta_item = base_meta.copy()
                meta_item["document_id"] = str(uuid.uuid4())
                meta_item["chunk_type"] = "TITLE_KEYWORD"
                meta_item["text"] = chunk_title_keyword
                expanded_metadata.append(meta_item)

            # 본문 청킹 (슬라이딩 윈도우)
            chunks = self._chunk_text(base_meta["content"], chunk_size=150, overlap=30)
            for i, chunk in enumerate(chunks):
                texts_to_embed.append(chunk)
                meta_item = base_meta.copy()
                meta_item["document_id"] = str(uuid.uuid4())
                meta_item["chunk_type"] = f"CONTENT_CHUNK_{i + 1}"
                meta_item["text"] = chunk
                expanded_metadata.append(meta_item)

        print(f"청킹 완료: {len(texts_to_embed)}개 청크 (소요: {time.time() - t1:.2f}s)", flush=True)

        # Dense Embedding 생성
        t2 = time.time()
        embeddings = self.model.encode(
            texts_to_embed,
            batch_size=128,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        print(f"임베딩 완료 (소요: {time.time() - t2:.2f}s)", flush=True)

        # FAISS 인덱스 구축
        t3 = time.time()
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        # 인덱스 저장 전에 디렉토리 생성
        Path(index_path_str).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, index_path_str)
        print(f"FAISS 인덱스 저장 완료 (소요: {time.time() - t3:.2f}s)", flush=True)

        # BM25 인덱스 구축
        t4 = time.time()
        tokenized_corpus = [text.split() for text in texts_to_embed]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.bm25_corpus = texts_to_embed
        print(f"BM25 인덱스 구축 완료 (소요: {time.time() - t4:.2f}s)", flush=True)

        # 메타데이터 저장
        t5 = time.time()
        self.metadata = expanded_metadata
        # 메타데이터 저장 전에 디렉토리 생성
        Path(meta_path_str).parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path_str, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"메타데이터 저장 완료 (소요: {time.time() - t5:.2f}s)", flush=True)

        print(f"전체 인덱스 구축 완료 (총 소요: {time.time() - start_total:.2f}s)", flush=True)
        self.index = index
        self.embeddings = embeddings
        return True

    # 인덱스 로드
    def load_index(self, index_path: str, meta_path: str):
        # Path 객체를 문자열로 변환
        index_path_str = str(index_path)
        meta_path_str = str(meta_path)
        
        if not (os.path.exists(index_path_str) and os.path.exists(meta_path_str)):
            return False
        t0 = time.time()
        print("인덱스 및 메타데이터 로드 중...", flush=True)
        self.index = faiss.read_index(index_path_str)
        with open(meta_path_str, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"로드 완료 (소요: {time.time() - t0:.2f}s)", flush=True)
        print("임베딩 복원 중...", flush=True)
        self.embeddings = self.index.reconstruct_n(0, self.index.ntotal)
        print(f"임베딩 복원 완료: {self.embeddings.shape}", flush=True)
        
        # 메타데이터에 'text' 필드가 없으면 재생성 (기존 인덱스 호환성)
        print("메타데이터 검증 및 보완 중...", flush=True)
        texts_for_bm25 = []
        for m in self.metadata:
            if "text" not in m:
                # text 필드가 없으면 기존 필드로부터 재생성
                title = m.get("title", "") or ""
                upper_keyword = m.get("upper_keyword", "") or ""
                lower_keyword = m.get("lower_keyword", "") or ""
                content = m.get("content", "") or ""
                
                # chunk_type에 따라 text 재생성
                chunk_type = m.get("chunk_type", "")
                if chunk_type and "TITLE_KEYWORD" in chunk_type:
                    m["text"] = f"{title} {upper_keyword} {lower_keyword}".strip()
                elif content:
                    # CONTENT_CHUNK인 경우 content 사용
                    m["text"] = content
                else:
                    # 기본값: title과 keyword 조합
                    m["text"] = f"{title} {upper_keyword} {lower_keyword}".strip()
            
            # text 필드가 비어있지 않은지 확인
            text_value = m.get("text", "")
            if not text_value:
                # 비어있으면 기본값으로 재생성
                title = m.get("title", "") or ""
                content = m.get("content", "") or ""
                m["text"] = content if content else title
            
            texts_for_bm25.append(m.get("text", ""))
        
        # BM25 인덱스 재생성
        print("BM25 인덱스 재생성 중...", flush=True)
        tokenized_corpus = [t.split() if t else [] for t in texts_for_bm25]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.bm25_corpus = texts_for_bm25
        print("BM25 인덱스 재생성 완료.", flush=True)
        return True

    # 검색 (BM25 → Dense → Rerank)
    def search(self, query: str, top_k: int = 5, bm25_top_n: int = 100) -> List[Dict[str, Any]]:
        if self.index is None or self.bm25 is None:
            raise RuntimeError("검색 전 인덱스를 로드하거나 build_index를 먼저 호출해야 합니다.")

        print(f"\n[검색 시작] Query: {query}", flush=True)
        total_start = time.time()

        # BM25 단계
        t0 = time.time()
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_top_idx = np.argsort(bm25_scores)[::-1][:bm25_top_n]
        print(f"BM25 검색 완료 (top-{bm25_top_n}, 소요: {time.time() - t0:.4f}s)", flush=True)

        # Dense 검색 단계 (FAISS)
        t1 = time.time()
        query_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        dense_vectors = self.embeddings[bm25_top_idx]
        dense_scores = np.dot(dense_vectors, query_vec.T).flatten()
        dense_top_idx = bm25_top_idx[np.argsort(dense_scores)[::-1][:top_k]]
        print(f"Dense 검색 완료 (소요: {time.time() - t1:.4f}s)", flush=True)

        # 결과 생성
        results = []
        for i in dense_top_idx:
            item = self.metadata[i].copy()
            # dense_scores에서 해당 인덱스의 점수 찾기
            score_idx = np.where(bm25_top_idx == i)[0]
            if len(score_idx) > 0:
                item["score"] = float(dense_scores[score_idx[0]])
            else:
                item["score"] = 0.0
            results.append(item)

        # Reranker -> 현재 사용 X
        if self.reranker:
            t2 = time.time()
            pairs = [[query, r["text"]] for r in results]
            rerank_scores = self.reranker.predict(pairs)
            results = [r for _, r in sorted(zip(rerank_scores, results), key=lambda x: x[0], reverse=True)]
            print(f"Rerank 완료 (소요: {time.time() - t2:.4f}s)", flush=True)

        print(f"전체 검색 완료 (총 소요: {time.time() - total_start:.4f}s)", flush=True)

        # 중복 문서 제거 (같은 detail_url 또는 같은 제목 기준)
        seen_urls = set()
        seen_titles = set()
        unique_results = []
        for r in results:
            url = (r.get("detail_url", "") or "").strip().lower()
            title = (r.get("title", "") or "").strip().lower()
            
            # URL이 있으면 URL로, 없으면 제목으로 중복 체크
            if url:
                if url in seen_urls:
                    continue
                seen_urls.add(url)
            else:
                if title and title in seen_titles:
                    continue
                if title:
                    seen_titles.add(title)

            unique_results.append(r)

        results = unique_results

        formatted_results = []
        for res in results:
            producer_data = res.get("producer", "") or ""
            member_names = [name.strip() for name in producer_data.split(",")] if producer_data else []
            
            # detail_url에서 link 추출 (비어있거나 None이면 빈 문자열)
            link = (res.get("detail_url") or "").strip()
            # 빈 문자열이나 None인 경우 빈 문자열로 통일
            if not link:
                link = ""
            
            formatted_results.append({
                "source": "외부",
                "score": float(res.get("score", 0.0)),  # 점수 (0~1 범위로 정규화된 값)
                "title": res.get("title", "") or "",
                "content": res.get("text", "") or "",
                "chunk_type": res.get("chunk_type", "") or "",
                "keyword": res.get("lower_keyword", "") or "",
                "date": res.get("date", "") or "",
                "link": link,  # detail_url에서 가져온 link (없으면 빈 문자열)
                "team_members": member_names,
            })
        return formatted_results


if __name__ == "__main__":
    # 현재 파일 기준으로 프로젝트 루트 경로 설정
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    base_path = PROJECT_ROOT / "dataset" / "crawling"
    csv_path = base_path / "crawling_db.csv"
    index_path = base_path / "crawling_total.index"
    meta_path = base_path / "crawling_total.pkl"

    model_path = "jhgan/ko-sroberta-multitask"
    engine = CrawlingdbSearchEngine(model_path=model_path, use_reranker=False)

    if not engine.load_index(index_path, meta_path):
        engine.build_index(csv_path, index_path, meta_path)

    query = "유기 동물, 유기견 관련 서비스"
    results = engine.search(query, top_k=10, bm25_top_n=1000)

    print(f"\n검색 결과 ({len(results)}건):")
    for i, res in enumerate(results):
        print(f"{i + 1}. Score: {res['score']:.4f}, Title: {res['title']} | Keyword: {res['keyword']}")
