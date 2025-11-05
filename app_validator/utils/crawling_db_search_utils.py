## crawling_db_search_utils.py

import os
import faiss
import pickle
import pandas as pd
import torch
import uuid
import numpy as np
import time
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
        print(f"'{csv_path}' 파일로부터 인덱스 구축을 시작합니다...", flush=True)
        start_total = time.time()

        # 데이터 로드 및 전처리
        t0 = time.time()
        df = pd.read_csv(csv_path)
        df.dropna(
            subset=["title", "content", "detail_url", "upper_keyword", "lower_keyword", "producer"],
            inplace=True,
        )
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
        faiss.write_index(index, index_path)
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
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"메타데이터 저장 완료 (소요: {time.time() - t5:.2f}s)", flush=True)

        print(f"전체 인덱스 구축 완료 (총 소요: {time.time() - start_total:.2f}s)", flush=True)
        self.index = index
        self.embeddings = embeddings
        return True

    # 인덱스 로드
    def load_index(self, index_path: str, meta_path: str):
        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            return False
        t0 = time.time()
        print("인덱스 및 메타데이터 로드 중...", flush=True)
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"로드 완료 (소요: {time.time() - t0:.2f}s)", flush=True)
        print("임베딩 복원 중...", flush=True)
        self.embeddings = self.index.reconstruct_n(0, self.index.ntotal)
        print(f"임베딩 복원 완료: {self.embeddings.shape}", flush=True)
        # BM25 인덱스 재생성 추가
        print("BM25 인덱스 재생성 중...", flush=True)
        texts = [m["text"] for m in self.metadata]
        tokenized_corpus = [t.split() for t in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.bm25_corpus = texts
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
            item["score"] = float(dense_scores[np.where(bm25_top_idx == i)[0][0]])  # 원래 방식 유지
            results.append(item)

        # Reranker -> 현재 사용 X
        if self.reranker:
            t2 = time.time()
            pairs = [[query, r["text"]] for r in results]
            rerank_scores = self.reranker.predict(pairs)
            results = [r for _, r in sorted(zip(rerank_scores, results), key=lambda x: x[0], reverse=True)]
            print(f"Rerank 완료 (소요: {time.time() - t2:.4f}s)", flush=True)

        print(f"전체 검색 완료 (총 소요: {time.time() - total_start:.4f}s)", flush=True)

        # 중복 문서 제거
        seen_urls = set()
        unique_results = []
        for r in results:
            url = r.get("detail_url", "").strip().lower()
            title = r.get("title", "").strip().lower()

            if url in seen_urls or any(title == u["title"].strip().lower() for u in unique_results):
                continue

            seen_urls.add(url)
            unique_results.append(r)

        results = unique_results

        formatted_results = []
        for res in results:
            producer_data = res.get("producer", "")
            member_names = [name.strip() for name in producer_data.split(",")] if producer_data else []
            formatted_results.append({
                "source": "외부",
                "score": res["score"],  # 기존처럼 그대로 유지
                "title": res["title"],
                "content": res["text"],
                "chunk_type": res["chunk_type"],
                "keyword": res["lower_keyword"],
                "date": res["date"],
                "link": res["detail_url"],
                "team_members": member_names,
            })
        return formatted_results


if __name__ == "__main__":
    base_path = os.path.join(os.path.dirname(__file__), "../../dataset/crawling")
    csv_path = os.path.join(base_path, "crawling_db.csv")
    index_path = os.path.join(base_path, "crawling_total.index")
    meta_path = os.path.join(base_path, "crawling_total.pkl")

    model_path = "jhgan/ko-sroberta-multitask"
    engine = CrawlingdbSearchEngine(model_path=model_path, use_reranker=False)

    if not engine.load_index(index_path, meta_path):
        engine.build_index(csv_path, index_path, meta_path)

    query = "유기 동물, 유기견 관련 서비스"
    results = engine.search(query, top_k=10, bm25_top_n=1000)

    print(f"\n검색 결과 ({len(results)}건):")
    for i, res in enumerate(results):
        print(f"{i + 1}. Score: {res['score']:.4f}, Title: {res['title']} | Keyword: {res['keyword']}")
