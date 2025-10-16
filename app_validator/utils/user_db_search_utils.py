import os
import faiss
import pandas as pd
import torch
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple

# DB 접속 정보 Mocking
# 실제 환경에서는 .env 또는 설정 파일에서 불러오는 게 좋음
DB_CONFIG = {
    "DB_TYPE": "MySQL",
    "HOST": "mock_host",
    "USER": "mock_user",
    "PASSWORD": "mock_password",
    "DATABASE": "ideantify"
}

# user 데이터 JSON 파일 경로
# /home/work/Team_AI/identify-ai/app_validator/user_db.json
USER_DB_PATH = os.path.join(
os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
'user_db.json'
)


class UserdbFaissSearchEngine: # 💡 클래스명 변경: LiveFaissSearchEngine -> UserdbFaissSearchEngine
    """
    SentenceTransformer와 FAISS를 사용하여 outer_project (실시간으로 업데이트 되는 DB)의 
    아이디어 데이터에 대한 의미 검색을 수행하는 엔진 (Mocking 포함).
    """
    def __init__(self, model_path: str):
        """
        엔진 초기화 시 SentenceTransformer 모델을 로드하고, user_db 데이터를 받습니다.
        <동일>
        GPU 사용이 가능하면 자동으로 GPU를 사용하도록 설정합니다.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"'{self.device}' 장치를 사용하여 모델을 로딩합니다.", flush = True)
        self.model = SentenceTransformer(model_path, device=self.device)
        self.index = None
        self.metadata = []

    def _fetch_all_ideas_from_db(self) -> pd.DataFrame:
        """
        [MOCK] outer_project 테이블에서 모든 아이디어를 추출하는 과정을 Mocking합니다.
        실제 DB 접속 대신 JSON 파일을 읽어와 데이터를 반환합니다.
        """

        # 1. JSON 파일로부터 데이터 로드
        with open(USER_DB_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"{DB_CONFIG['DATABASE']} DB에 접속하여 user_db 데이터를 추출 중...") # "outer_project DB 접속하여 데이터를 추출 중..."

        # 2. 로드된 JSON 리스트를 DataFrame으로 변환
        df = pd.DataFrame(data)
        
        # 3. 인덱스 구축에 필요한 핵심 필드만 추출하여 반환
        # 💡 id와 updatedAt은 내부적으로 사용하기 위해 추출합니다. (모아보기 상세 모달로 이동 등)
        required_cols = ['id', 'user_id', 'createdAt', 'updatedAt', 'input_title', 'input_details', 'upper_keyword', 'lower_keyword', 'team_members_json']        
        df = df[required_cols]

        print(f"✅ DB 데이터 추출 완료. {len(df)}개의 아이디어 로드됨.")
        return df


    def update_and_load_index(self):
        """
        DB에서 최신 데이터를 추출하고, 메모리 내에 FAISS 인덱스를 새로 구축합니다.
        """
        df = self._fetch_all_ideas_from_db()
        
        if df.empty:
            print("추출된 데이터가 없어 인덱스를 구축하지 않습니다.")
            self.index = None
            self.metadata = []
            return

        print("텍스트 임베딩을 생성하고 인덱스를 메모리에 로드합니다...", flush = True)
        
        # 1. 검색에 사용할 텍스트 조합
        texts = (
            df['input_title'].fillna('') + " " + 
            df['input_details'].fillna('') + " " + 
            df['upper_keyword'].fillna('') + " " + 
            df['lower_keyword'].fillna('')
        ).tolist()


        # 2. 임베딩 생성 (db_search_utils.py 로직 재사용)
        embeddings = self.model.encode(
            texts, 
            batch_size=128, # GPU 사용 시 배치 크기 조절
            convert_to_numpy=True, 
            normalize_embeddings=True
        )

        # 3. FAISS 인덱스 생성 및 메모리 로드
        print("FAISS 인덱스를 생성하고 저장합니다...", flush = True)
        self.index = faiss.IndexFlatIP(embeddings.shape[1]) # (db_search_utils.py 로직 재사용)
        self.index.add(embeddings)  # 추가
        ### (id, createdAt, updatedAt 포함)
        metadata_cols = ['id', 'createdAt', 'updatedAt', 'input_title', 'input_details', 'upper_keyword', 'lower_keyword', 'team_members_json']        
        self.metadata = df[metadata_cols].fillna('').to_dict(orient='records')
        print(f"✅ UserDB 인덱스 구축 완료. (아이디어 수: {len(self.metadata)})", flush = True)
    

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        로드된 인덱스에서 주어진 쿼리와 가장 유사한 문서를 검색합니다.
        (db_search_utils.py의 search 코드와 동일)

        Args:
            query (str): 사용자 검색 쿼리.
            top_k (int): 반환할 상위 결과의 수.

        Returns:
            List[Dict[str, Any]]: score가 포함된 검색 결과 딕셔너리의 리스트.
        """
        if self.index is None:
            raise RuntimeError("검색을 수행하기 전에 'update_and_load_index'를 먼저 호출해야 합니다.")
            
        # 1. 쿼리 임베딩 및 정규화 (db_search_utils.py 로직 재사용)
        query_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

        # 2. FAISS 검색 (db_search_utils.py 로직 재사용)
        distances, indices = self.index.search(query_vec, top_k)

        # 3. 메타데이터와 결합하여 결과 포맷팅
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:
                item = self.metadata[idx].copy()
                
                # 💡 멤버 이름 리스트 추출 (JSON 문자열일 경우 로드 후 사용)
                members_data = item.get('team_members_json', []) 
                try:
                    # 데이터가 문자열이면 JSON으로 로드
                    if isinstance(members_data, str):
                        members = json.loads(members_data)
                    else:
                        # 이미 리스트/딕셔너리 형태 (DataFrame이 자동 변환한 경우)
                        members = members_data 
                        member_names = [m.get('name', '이름없음') for m in members]
                except (json.JSONDecodeError, AttributeError):
                    member_names = ['데이터 오류']

                # 💡 최종 결과 포맷팅: create_report.py의 입력 규격을 따름
                results.append({
                    "score": float(dist),
                    "source": "Iideantify 유저", #### 출처 명시
                    "title": item['input_title'], 
                    "content": item['input_details'], 
                    "keyword": item['lower_keyword'],
                    "date": item['createdAt'],
                    "link": f"/project/{item['id']}", # 클릭하면 모아보기 UI로 이동했음 좋겠는데 어카지? 해당 페이지 링크 어케 받지? 프론트?백엔드? 문의 필요
                    "team_members": member_names
                })
                
        return results



if __name__ == '__main__':
    # -----------------------
    # --- 테스트 실행 함수 ---
    # -----------------------

    # user 데이터 JSON 파일 경로
    # /home/work/Team_AI/identify-ai/app_validator/user_db.json
    USER_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    'user_db.json'
)
    # 1. 검색 엔진 초기화 및 인덱스 로드
    MODEL_PATH = "jhgan/ko-sroberta-multitask"
    engine = UserdbFaissSearchEngine(model_path=MODEL_PATH)    
    engine.update_and_load_index()
    
    # 2. 검색 쿼리 실행
    print("\n[TEST] 검색 쿼리 실행: '가족을 찾아주는 인공지능 서비스'")
    query = "가족을 찾아주는 인공지능 서비스"
    results = engine.search(query, top_k=3)
    
    # 3. 결과 출력
    if results:
        print(f"✅ 검색 결과 ({len(results)}건):")
        for i, res in enumerate(results):
            # 점수 (유사도)는 가장 높은 것이 1.0
            print(f"  {i+1}. Score: {res['score']:.4f}, Title: {res['title']}, Keyword: {res['keyword']}")
        print("---")
        print(f"가장 유사한 항목 (ID: {results[0]['link'].split('/')[-1]})은 실종자 관련 아이디어일 것으로 예상됩니다.")
    else:
        print("❌ 검색 결과가 없습니다.")