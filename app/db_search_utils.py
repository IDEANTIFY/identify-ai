import os
import faiss
import pickle
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class FaissSearchEngine:
    """
    SentenceTransformer와 FAISS를 사용하여 CSV 데이터에 대한 의미 검색을 수행하는 엔진.
    인덱스 구축, 로딩, 검색 기능을 클래스로 캡슐화하여 재사용성과 효율성을 높입니다.
    """
    def __init__(self, model_path: str):
        """
        엔진 초기화 시 SentenceTransformer 모델을 로드합니다.
        GPU 사용이 가능하면 자동으로 GPU를 사용하도록 설정합니다.
        """
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"'{self.device}' 장치를 사용하여 모델을 로딩합니다.", flush = True)
        self.model = SentenceTransformer(model_path, device=self.device)
        self.index = None
        self.metadata = []

    def build_index(self, csv_path: str, index_path: str, meta_path: str):
        """
        CSV 파일로부터 FAISS 인덱스와 메타데이터를 구축하고 파일로 저장합니다.

        Args:
            csv_path (str): 원본 데이터 CSV 파일 경로.
            index_path (str): 생성된 FAISS 인덱스를 저장할 경로.
            meta_path (str): 추출된 메타데이터(pickle)를 저장할 경로.
        """
        print(f"'{csv_path}' 파일로부터 인덱스 구축을 시작합니다...", flush = True)
        
        # 1. 데이터 로드 및 전처리
        df = pd.read_csv(csv_path)
        df.dropna(subset=['title', 'content'], inplace=True)
        df['content_length'] = df['content'].apply(lambda x: len(str(x).split()))
        df = df[df['content_length'] > 10].reset_index(drop=True)
        print(f"전처리 후 {len(df)}개의 유효한 문서를 찾았습니다.", flush = True)

        # 2. 텍스트 임베딩 생성
        print("텍스트 임베딩을 생성합니다... (시간이 걸릴 수 있습니다)", flush = True)
        texts = (df['title'].fillna('') + " " + df['content'].fillna('')).tolist()
        embeddings = self.model.encode(
            texts, 
            batch_size=128, # GPU 사용 시 배치 크기 조절
            convert_to_numpy=True, 
            normalize_embeddings=True,
            show_progress_bar=True
        )

        # 3. FAISS 인덱스 생성 및 저장 (IndexFlatIP: 내적 기반 거리 계산)
        print("FAISS 인덱스를 생성하고 저장합니다...", flush = True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, index_path)

        # 4. 메타데이터 저장
        print("메타데이터를 생성하고 저장합니다...", flush = True)
        metadata = df[['title', 'content', 'date', 'detail_url']].fillna('').to_dict(orient='records')
        with open(meta_path, 'wb') as f:
            pickle.dump(metadata, f)
            
        print(f"✅ 인덱스 및 메타데이터 저장이 완료되었습니다. ('{index_path}', '{meta_path}')", flush = True)
        
        # 생성된 인덱스와 메타데이터를 바로 사용할 수 있도록 내부에 로드
        self.index = index
        self.metadata = metadata

    def load_index(self, index_path: str, meta_path: str):
        """
        미리 구축된 FAISS 인덱스와 메타데이터 파일을 로드합니다.

        Args:
            index_path (str): FAISS 인덱스 파일(.index) 경로.
            meta_path (str): 메타데이터 파일(.pkl) 경로.
        """
        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            raise FileNotFoundError(f"인덱스 파일 '{index_path}' 또는 메타데이터 파일 '{meta_path}'을 찾을 수 없습니다.")
            
        print(f"'{index_path}' 및 '{meta_path}'에서 인덱스와 메타데이터를 로드합니다...", flush = True)
        self.index = faiss.read_index(index_path)
        with open(meta_path, 'rb') as f:
            self.metadata = pickle.load(f)
        print("✅ 로딩이 완료되었습니다.", flush = True)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        로드된 인덱스에서 주어진 쿼리와 가장 유사한 문서를 검색합니다.

        Args:
            query (str): 사용자 검색 쿼리.
            top_k (int): 반환할 상위 결과의 수.

        Returns:
            List[Dict[str, Any]]: score가 포함된 검색 결과 딕셔너리의 리스트.
        """
        if self.index is None:
            raise RuntimeError("검색을 수행하기 전에 'build_index' 또는 'load_index'를 먼저 호출해야 합니다.")
            
        # 1. 쿼리 임베딩 및 정규화
        query_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

        # 2. FAISS 검색 (D: 거리, I: 인덱스)
        distances, indices = self.index.search(query_vec, top_k)

        # 3. 메타데이터와 결합하여 결과 포맷팅
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:  # 유효한 인덱스인 경우
                item = self.metadata[idx].copy()
                item['score'] = float(dist)
                results.append(item)
                
        return results

# --- ✅ 예시 실행 ---
# if __name__ == "__main__":
    # 0. 파일 경로 설정
    # CSV_FILE = "/content/drive/MyDrive/Conference_2025/Dataset/crawling_total.csv"
    # INDEX_FILE = "/content/drive/MyDrive/Conference_2025/Dataset/crawling_total.index"
    # META_FILE = "/content/drive/MyDrive/Conference_2025/Dataset/crawling_total.pkl"

    # 1. 검색 엔진 인스턴스 생성
    # MODEL_FOLDER_PATH = "/content/drive/MyDrive/Conference_2025/ko-sroberta-multitask-local"
    # search_engine = FaissSearchEngine(model_path=MODEL_FOLDER_PATH) # 경로를 직접 전달

    # 2. 인덱스 파일이 없으면 새로 구축
    # if not os.path.exists(INDEX_FILE):
        # print("기존 인덱스 파일이 없습니다. 새로 구축합니다.")
        # search_engine.build_index(
            # csv_path=CSV_FILE,
            # index_path=INDEX_FILE,
            # meta_path=META_FILE
        # )
    # 인덱스 파일이 있으면 로드
    # else:
        # search_engine.load_index(
            # index_path=INDEX_FILE,
            # meta_path=META_FILE
        # )

    # 3. 검색 실행
    # search_query = "인공지능을 활용한 신약 개발 동향"
    # print(f"\n--- 쿼리: '{search_query}' ---", flush = True)
    
    # search_results = search_engine.search(query=search_query, top_k=5)

    # 4. 결과 출력
    # if search_results:
        # for i, result in enumerate(search_results):
            # print(f"[{i+1}] Score: {result['score']:.4f}", flush = True)
            # print(f"  - Title: {result['title']}", flush = True)
            # print(f"  - URL: {result['detail_url']}\n", flush = True)
    # else:
        # print("검색 결과가 없습니다.", flush = True)