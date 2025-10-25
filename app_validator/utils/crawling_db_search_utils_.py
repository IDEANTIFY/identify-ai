import os
import faiss
import pickle
import pandas as pd
import torch
import sys
import uuid
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any


class CrawlingdbFaissSearchEngine:# 💡 클래스명 변경: FaissSearchEngine -> CrawlingdbFaissSearchEngine
    """
    SentenceTransformer와 FAISS를 사용하여 CSV 데이터에 대한 의미 검색을 수행하는 엔진.
    인덱스 구축, 로딩, 검색 기능을 클래스로 캡슐화하여 재사용성과 효율성을 높입니다.
    """
    def __init__(self, model_path: str): # 💡 reranker_path 제거
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
        ⭐ 문서 청킹(제목+키워드, 내용)을 통해 정확도를 높입니다.
        Args:
            csv_path (str): 원본 데이터 CSV 파일 경로.
            index_path (str): 생성된 FAISS 인덱스를 저장할 경로.
            meta_path (str): 추출된 메타데이터(pickle)를 저장할 경로.
        """
        print(f"'{csv_path}' 파일로부터 인덱스 구축을 시작합니다...", flush = True)
        
        # 1. 데이터 로드 및 전처리
        df = pd.read_csv(csv_path)
        df.dropna(subset=['title', 'content', 'detail_url', 'upper_keyword', 'lower_keyword', 'producer'], inplace=True)        
        df['content_length'] = df['content'].apply(lambda x: len(str(x).split()))
        df = df[df['content_length'] > 10].reset_index(drop=True)
        print(f"전처리 후 {len(df)}개의 유효한 문서를 찾았습니다.", flush = True)

        # 2. 텍스트 청킹 및 메타데이터 확장
        texts_to_embed = []
        expanded_metadata = []
        metadata_cols = ['title', 'content', 'date', 'detail_url', 'upper_keyword', 'lower_keyword', 'producer']
        
        # 원본 문서를 순회하며 여러 개의 '청크'를 생성합니다.
        for original_idx, row in df.iterrows():
            base_meta = row[metadata_cols].fillna('').to_dict()
            
            # --- 청크 전략 1: 제목과 키워드 조합 (쿼리가 짧고 핵심일 때 유리) ---
            chunk_title_keyword = (
                base_meta['title'] + " " +
                base_meta['upper_keyword'] + " " +
                base_meta['lower_keyword']
            ).strip()
            
            if chunk_title_keyword:
                texts_to_embed.append(chunk_title_keyword)
                meta_item = base_meta.copy()
                meta_item['document_id'] = str(uuid.uuid4())
                meta_item['chunk_type'] = 'TITLE_KEYWORD'
                meta_item['text'] = chunk_title_keyword
                expanded_metadata.append(meta_item)
                
            # --- 청크 전략 2: 내용을 문장 단위로 분할 및 인덱싱 (쿼리가 Content와 일치할 때 가장 중요) ---
            chunk_content_full = base_meta['content'].strip()
            
            if chunk_content_full:
                # 🌟 수정: 마침표 기준으로 문장 분할 (한국어 텍스트 분할을 위한 간단한 방법)
                sentences = [s.strip() for s in chunk_content_full.split('.') if s.strip()] 
                
                # 문장을 1개 또는 2개씩 묶어 짧은 단락 청크 생성 (문맥 유지 및 적절한 길이 확보)
                # 1개씩 묶는 것이 쿼리와 정확히 일치할 때 가장 유리합니다.
                chunk_size = 1 # 문장 하나를 하나의 청크로!
                sentence_chunks = [' '.join(sentences[i:i + chunk_size]) 
                                   for i in range(0, len(sentences), chunk_size)]

                for i, sentence_chunk in enumerate(sentence_chunks):
                    # 쿼리와 정확히 일치하는 문장이 인덱싱되도록 합니다.
                    if len(sentence_chunk.split()) >= 5: # 너무 짧은 청크(단어 5개 미만) 제외
                        texts_to_embed.append(sentence_chunk)
                        meta_item = base_meta.copy()
                        meta_item['document_id'] = str(uuid.uuid4())
                        # 이 청크는 내용의 일부임을 명시
                        meta_item['chunk_type'] = f'CONTENT_CHUNK_{i+1}' 
                        meta_item['text'] = sentence_chunk # 임베딩에 사용된 텍스트
                        expanded_metadata.append(meta_item)

        print(f"청킹 후 총 {len(texts_to_embed)}개의 임베딩을 생성합니다.", flush = True)
        
        # 3. 텍스트 임베딩 생성
        embeddings = self.model.encode(
            texts_to_embed,  # ⭐ 수정 : 'texts'를 'texts_to_embed'로 수정            
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
        ### ⭐ 수정 : 확장된 메타데이터 저장
        # metadata_cols = ['title', 'content', 'date', 'detail_url', 'upper_keyword', 'lower_keyword', 'producer']        
        # metadata = df[metadata_cols].fillna('').to_dict(orient='records')
        self.metadata = expanded_metadata # 클래스 멤버 업데이트
        with open(meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)

            
        print(f"✅ 인덱스 및 메타데이터 저장이 완료되었습니다. ('{index_path}', '{meta_path}')", flush = True)
        
        # 생성된 인덱스와 메타데이터를 바로 사용할 수 있도록 내부에 로드
        self.index = index
        
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

    # 💡 search 함수: 재순위화 로직 제거, 청킹 결과 포맷팅만 유지
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Bi-Encoder와 청킹된 인덱스를 사용하여 검색을 수행합니다.
        """
        if self.index is None:
            raise RuntimeError("검색을 수행하기 전에 'build_index' 또는 'load_index'를 먼저 호출해야 합니다.")
            
        # 1. 쿼리 임베딩 및 정규화
        query_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

        # 2. FAISS 검색
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:
                item = self.metadata[idx].copy()
                link_url = item.get('detail_url', f"/archive/{idx}") 
                
                producer_data = item.get('producer', '정보 없음')
                member_names = [name.strip() for name in producer_data.split(',')] if isinstance(producer_data, str) and producer_data else []

                results.append({
                    "score": float(dist),
                    "source": "외부",
                    "title": item['title'], 
                    "content": item['text'],  # ⭐ 청크된 내용 반환
                    "chunk_type": item['chunk_type'], # ⭐ 청크 유형 반환
                    "keyword": item['lower_keyword'],
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
    
    DB_CSV_FILE = os.path.join(PROJECT_ROOT_PATH, "dataset","crawling", "crawling_db.csv")
    INDEX_FILE = os.path.join(PROJECT_ROOT_PATH, "dataset","crawling",  "crawling_total.index")
    META_FILE = os.path.join(PROJECT_ROOT_PATH, "dataset","crawling",  "crawling_total.pkl")

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