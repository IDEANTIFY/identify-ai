import os
import faiss
import pickle
import pandas as pd
import torch
import sys
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any


class CrawlingdbFaissSearchEngine:# 💡 클래스명 변경: FaissSearchEngine -> CrawlingdbFaissSearchEngine
    """
    SentenceTransformer와 FAISS를 사용하여 CSV 데이터에 대한 의미 검색을 수행하는 엔진.
    인덱스 구축, 로딩, 검색 기능을 클래스로 캡슐화하여 재사용성과 효율성을 높입니다.
    """
    def __init__(self, model_path: str):
        """
        엔진 초기화 시 SentenceTransformer 모델을 로드합니다.
        GPU 사용이 가능하면 자동으로 GPU를 사용하도록 설정합니다.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
        ### 💡 수정 : 컬럼 추가
        df.dropna(subset=['title', 'content', 'detail_url', 'upper_keyword', 'lower_keyword', 'producer'], inplace=True)        
        df['content_length'] = df['content'].apply(lambda x: len(str(x).split()))
        df = df[df['content_length'] > 10].reset_index(drop=True)
        print(f"전처리 후 {len(df)}개의 유효한 문서를 찾았습니다.", flush = True)

        # 2. 텍스트 임베딩 생성
        print("텍스트 임베딩을 생성합니다... (시간이 걸릴 수 있습니다)", flush = True)
        ### 💡 수정 : 새로운 텍스트 조합: title + content + upper_keyword + lower_keyword
        # texts = (df['title'].fillna('') + " " + df['content'].fillna('')).tolist()
        texts = (
            df['title'].fillna('') + " " + 
            df['content'].fillna('') + " " + 
            df['etc'].fillna('') + " " +   ### 💡 추가
            df['upper_keyword'].fillna('') + " " +  
            df['lower_keyword'].fillna('')  
        ).tolist()

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
        ### 💡 수정 : 메타데이터 필드 목록 수정 -> 상하위 키워드(), 출처(detail_url, 멤버(producer) 추가
        # metadata = df[['title', 'content', 'date', 'detail_url']].fillna('').to_dict(orient='records')
        metadata_cols = ['title', 'content', 'date', 'detail_url', 'upper_keyword', 'lower_keyword', 'producer']        
        metadata = df[metadata_cols].fillna('').to_dict(orient='records')

        with open(meta_path, 'wb') as f:
            pickle.dump(metadata, f)
            
        print(f"✅ 인덱스 및 메타데이터 저장이 완료되었습니다. ('{index_path}', '{meta_path}')", flush = True)
        
        # 생성된 인덱스와 메타데이터를 바로 사용할 수 있도록 내부에 로드
        self.index = index
        self.metadata = metadata
        
        # 💡 수정: 인덱스 구축 성공 시 True 반환 추가
        return True

    def load_index(self, index_path: str, meta_path: str):
        """
        미리 구축된 FAISS 인덱스와 메타데이터 파일을 로드합니다.

        Args:
            index_path (str): FAISS 인덱스 파일(.index) 경로.
            meta_path (str): 메타데이터 파일(.pkl) 경로.
        """
        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            # raise FileNotFoundError(f"인덱스 파일 '{index_path}' 또는 메타데이터 파일 '{meta_path}'을 찾을 수 없습니다.")
            return False

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
        # results = []
        # for idx, dist in zip(indices[0], distances[0]):
        #     if idx != -1:  # 유효한 인덱스인 경우
        #         item = self.metadata[idx].copy()
        #         item['score'] = float(dist)
        #         results.append(item)
                
        # return results

        ### 💡 수정 : 최종 출력 포맷에 키워드 추가
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:
                item = self.metadata[idx].copy()
                link_url = item.get('detail_url', f"/archive/{idx}") 
                
                # link: CSV의 detail_url 필드를 사용합니다.
                link_url = item.get('detail_url', f"/archive/{idx}") 
                
                # team_members : producer 필드를 team_members로 매핑 및 포맷팅
                # producer가 문자열이므로 최종적으로는 문자열 리스트로 변환하여 Live Engine과 통일시킵니다.
                producer_data = item.get('producer', '정보 없음')
                if isinstance(producer_data, str) and ',' in producer_data:
                    member_names = [name.strip() for name in producer_data.split(',')]
                elif isinstance(producer_data, str) and producer_data:
                    member_names = [producer_data]
                else:
                    member_names = []

                results.append({
                    "score": float(dist),
                    "source": "외부",
                    "title": item['title'], 
                    "content": item['content'],
                    "keyword": item['lower_keyword'], # 👈 하위_키워드 추가
                    "date": item['date'],
                    "link": link_url,
                    "team_members": member_names
                })
                
        return results




if __name__ == '__main__':
    # -----------------------
    # --- 테스트 실행 함수 ---
    # -----------------------

    # 현재 파일 위치 기준: /identify-ai/app_validator/utils/
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    APP_VALIDATOR_DIR = os.path.dirname(CURRENT_DIR)
    PROJECT_ROOT_PATH = os.path.dirname(APP_VALIDATOR_DIR) # ROOT는 /identify-ai여야 합니다.
    
    DB_CSV_FILE = os.path.join(PROJECT_ROOT_PATH, "dataset", "crawling_db.csv")
    INDEX_FILE = os.path.join(PROJECT_ROOT_PATH, "dataset", "crawling_total.index")
    META_FILE = os.path.join(PROJECT_ROOT_PATH, "dataset", "crawling_total.pkl")

    MODEL_PATH = "jhgan/ko-sroberta-multitask"
    
    # 1. 검색 엔진 초기화
    try:
        engine = CrawlingdbFaissSearchEngine(model_path=MODEL_PATH) 
    except Exception as e:
        print(f"❌ 엔진 초기화 중 오류 발생: {e}")
        sys.exit(1)

    # 2. 인덱스 로드 또는 구축 시도 (단순화된 플로우)
    index_loaded = engine.load_index(index_path=INDEX_FILE, meta_path=META_FILE)
    
    if not index_loaded:
        print("\n[INFO] 사전 구축된 인덱스가 없습니다. CSV로부터 인덱스 구축을 시도합니다.")
        build_success = engine.build_index(csv_path=DB_CSV_FILE, index_path=INDEX_FILE, meta_path=META_FILE)
        
        if not build_success:
             print("❌ 인덱스 구축에 실패했습니다. 테스트를 종료합니다.")
             sys.exit(1)
        else:
             print("✅ 인덱스 구축이 완료되어 검색을 진행합니다.") # 성공 메시지 출력 후 다음 단계 진행
             # build_index 성공 시 engine.index와 engine.metadata가 이미 설정되어 있습니다.

    # 2. 검색 쿼리 실행
    if engine.index is not None:
        # print("\n[TEST] 검색 쿼리 실행: '가족 실종 입양아를 찾아주는 인공지능 서비스'")
        print("\n[TEST] 검색 쿼리 실행: '인공지능을 활용한 실종자 및 해외 입양자 찾기 플랫폼'")
        # query = "가족 실종 입양아를 찾아주는 인공지능 서비스"
        query = "인공지능을 활용한 실종자 및 해외 입양자 찾기 플랫폼"
        results = engine.search(query, top_k=10)
        
        # 3. 결과 출력
        if results:
            print(f"✅ 검색 결과 ({len(results)}건):")
            for i, res in enumerate(results):
                print(f"  {i+1}. Score: {res['score']:.4f}, Title: {res['title']}, Keyword: {res['keyword']}")
            
        else:
            print("❌ 검색 결과가 없습니다.")