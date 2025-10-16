import pandas as pd
import numpy as np
import faiss
import torch
import json
import re
import os
from ast import literal_eval
from typing import Dict, List, Tuple, Any, Set, Optional
from sentence_transformers import SentenceTransformer


# --- 프로젝트 설정 상수 ---
MODEL_PATH = "jhgan/ko-sroberta-multitask"
PROJECT_ROOT_PATH = "/home/work/Team_AI/identify-ai" 

DB_CSV_FILE = os.path.join(PROJECT_ROOT_PATH, "dataset", "crawling_db.csv")
RULES_FILE = os.path.join(PROJECT_ROOT_PATH, "app_keyword", "category_rules.json") 
MAPPED_OUTPUT_FILE = DB_CSV_FILE # 원본 파일 업데이트
UNMAPPED_OUTPUT_FILE = os.path.join(PROJECT_ROOT_PATH, "dataset", "unmapped_crawling.csv")
TEMP_COL = '_is_mapped_temp'

# --- FAISS 검색 엔진 클래스 ---
class FaissSearchEngine:
    """
    SentenceTransformer와 FAISS를 사용하여 카테고리 메타데이터에 대한 의미 검색을 수행하는 엔진.
    """
    def __init__(self, model_path: str):
        """ 엔진 초기화 시 SentenceTransformer 모델을 로드합니다. (디바이스 자동 감지) """
        
        # 디바이스 자동 감지 로직
        if torch.cuda.is_available(): 
            self.device = "cuda"
        elif torch.backends.mps.is_available(): 
            self.device = "mps" # Apple Metal GPU (MPS)
        else: 
            self.device = "cpu" # CPU
        
        print(f"Sentence Transformer 모델을 로딩합니다... (Device: {self.device})", flush=True)
        
        # 모델 로드
        self.model = SentenceTransformer(model_path, device=self.device)
        self.index = None # FAISS 인덱스 객체
        self.metadata: List[Tuple[str, str]] = [] # [(upper_cat, lower_cat), ...]

    def build_index(self, categories: List[Tuple[str, str]]):
        """ 카테고리 메타데이터를 기반으로 임베딩을 생성하고 FAISS 인덱스를 구축합니다. """
        self.metadata = categories
        # 카테고리 이름 조합 (검색 대상 문구)
        texts = [f"{upper} {lower}" for upper, lower in categories]
        
        print(f"총 {len(texts)}개의 카테고리 메타데이터를 임베딩합니다.")
        
        # 임베딩 생성
        embeddings = self.model.encode(
            texts, 
            convert_to_tensor=True, 
            show_progress_bar=True
        ).cpu().numpy()
        
        # FAISS 인덱스 구축 (L2 거리 사용)
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d) 
        self.index.add(embeddings)
        
        print("FAISS 인덱스 구축 완료.")

    def search(self, query_text: str, k: int = 1) -> Optional[Tuple[str, str]]:
        """ 쿼리 텍스트와 가장 유사한 상위/하위 카테고리를 검색합니다. """
        if self.index is None or not self.metadata or not query_text.strip():
            return None

        # 쿼리 임베딩 생성
        query_embedding = self.model.encode(
            [query_text], 
            convert_to_tensor=True
        ).cpu().numpy()
        
        # FAISS 검색
        distances, indices = self.index.search(query_embedding, k)
        
        best_index = indices[0][0]
        best_distance = distances[0][0]
        
        # L2 거리 임계값 설정 (거리가 너무 멀면 의미적 유사도가 낮다고 판단)
        L2_THRESHOLD = 0.5 
        if best_distance < L2_THRESHOLD:
            return self.metadata[best_index]
        else:
            return None # 유사도 낮음


# --- 유틸리티 및 매핑 함수 ---
def flatten_rules(category_rules: Dict) -> List[Tuple[str, str, str]]:
    """ JSON 규칙을 평탄화 """
    flat_rules = []
    for upper_cat, lower_cat_dict in category_rules.items():
        for lower_cat_raw, patterns in lower_cat_dict.items():
            lower_cat_names = [lower_cat_raw]
            try:
                if lower_cat_raw.startswith("['") and lower_cat_raw.endswith("']"):
                    lower_cat_names = literal_eval(lower_cat_raw)
            except (ValueError, SyntaxError):
                pass
            
            for lower_cat in lower_cat_names:
                for pattern in patterns:
                    if pattern:
                        flat_rules.append((upper_cat, lower_cat, pattern))
    return flat_rules

def get_category_metadata(category_rules: Dict) -> List[Tuple[str, str]]:
    """ FAISS 인덱스 구축을 위해 모든 고유한 (상위, 하위) 카테고리 쌍을 추출 """
    categories: Set[Tuple[str, str]] = set()
    for upper_cat, lower_cat_dict in category_rules.items():
        for lower_cat_raw in lower_cat_dict.keys():
            lower_cat_names = [lower_cat_raw]
            try:
                if lower_cat_raw.startswith("['") and lower_cat_raw.endswith("']"):
                    lower_cat_names = literal_eval(lower_cat_raw)
            except (ValueError, SyntaxError):
                pass
            
            for lower_cat in lower_cat_names:
                categories.add((upper_cat, lower_cat))
    return sorted(list(categories))

def prepare_text(row: pd.Series) -> str:
    """ title과 content를 병합 """
    return str(row.get('title', '') if pd.notna(row.get('title')) else '') + ' ' + \
           str(row.get('content', '') if pd.notna(row.get('content')) else '')

def get_regex_matches(text: str, flat_rules: List[Tuple[str, str, str]]) -> Tuple[Set[str], Set[str]]:
    """ 정규식 기반 매칭 (A)"""
    upper_matches = set()
    lower_matches = set()
    for upper_cat, lower_cat, pattern in flat_rules:
        try:
            if re.search(pattern, text, re.IGNORECASE):
                upper_matches.add(upper_cat)
                lower_matches.add(lower_cat)
        except re.error:
            continue
    return upper_matches, lower_matches


def map_keywords_to_dataframe(
    df: pd.DataFrame, 
    rules_filepath: str,
    faiss_engine: FaissSearchEngine
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    DataFrame에 키워드를 매핑하고, 매핑된 DataFrame과 매핑 안 된 DataFrame을 분리하며, 
    원본 구조를 유지한 전체 DataFrame도 반환합니다.
    
    :param df: 원본 DataFrame
    :param rules_filepath: JSON 규칙 파일 경로
    :param faiss_engine: FaissSearchEngine 인스턴스
    :return: (전체 DataFrame, 매핑된 DataFrame, 매핑 안 된 DataFrame)
    """
    try:
        with open(rules_filepath, 'r', encoding='utf-8') as f:
            category_rules = json.load(f)
    except Exception as e:
        print(f"오류: 규칙 파일 로드 중 문제가 발생했습니다: {e}")
        return df.copy(), pd.DataFrame(), pd.DataFrame()

    flat_rules = flatten_rules(category_rules)
    
    if faiss_engine.index is None:
        print("경고: FAISS 인덱스가 구축되지 않아 정규식만 사용합니다.")
    
    # 결과를 담을 임시 리스트
    result_data = []
    
    for index, row in df.iterrows():
        text = prepare_text(row)
        upper_final, lower_final = get_regex_matches(text, flat_rules)
        
        # FAISS 의미 검색 (B)
        if not upper_final and faiss_engine.index is not None:
            best_match = faiss_engine.search(text, k=1)
            
            if best_match:
                upper_final.add(best_match[0])
                lower_final.add(best_match[1])
        
        # 매핑 성공 여부
        is_mapped = bool(upper_final)

        # 원본 데이터에 매핑 결과를 추가하고 리스트에 저장
        row_dict = row.to_dict()
        
        # ⭐️⭐️⭐️ 매핑 성공 시에만 키워드를 할당하고, 실패 시 빈 문자열 할당 ⭐️⭐️⭐️
        if is_mapped:
            row_dict['upper_keyword'] = ', '.join(sorted(list(upper_final)))
            row_dict['lower_keyword'] = ', '.join(sorted(list(lower_final)))
        else:
            # 매핑 실패 시 빈 문자열로 초기화
            row_dict['upper_keyword'] = ''
            row_dict['lower_keyword'] = ''
            
        row_dict['_is_mapped_temp'] = is_mapped
        # df가 가지는 인덱스(Pandas 내부 인덱스)를 _original_index_에 저장 
        row_dict['_original_index_'] = row['_original_index_']
        
        result_data.append(row_dict)

    # 임시 컬럼이 있는 전체 DataFrame 생성
    mapped_df_with_temp = pd.DataFrame(result_data)
    
    # 1. 전체 DataFrame: 저장 시 _is_mapped_temp와 _original_index_ 제거
    final_db_df = mapped_df_with_temp.drop(columns=['_is_mapped_temp', '_original_index_'])
    
    # 2. 매핑된 행 (출력 및 통계용)
    mapped_df = mapped_df_with_temp[mapped_df_with_temp['_is_mapped_temp']].drop(columns=['_is_mapped_temp', '_original_index_']) 
    
    # 3. 미매핑된 행 (다음 모듈 입력용, _original_index_ 유지)
    unmapped_df = mapped_df_with_temp[~mapped_df_with_temp['_is_mapped_temp']].drop(columns=['_is_mapped_temp'])
    
    return final_db_df, mapped_df, unmapped_df


# --- 메인 실행 블록 ---
if __name__ == '__main__':
    print("--- FAISS 기반 하이브리드 키워드 매핑 모듈 실행 ---")
    
    # 0. 폴더 생성 및 입력 파일 존재 확인 (Output 폴더만 생성)
    output_data_dir = os.path.dirname(UNMAPPED_OUTPUT_FILE)
    if output_data_dir and not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir, exist_ok=True)
        print(f"[{output_data_dir}] 폴더를 생성했습니다.")
    
    # DB_CSV_FILE (crawling_db.csv)이 없으면 실행 불가
    if not os.path.exists(DB_CSV_FILE):
        raise FileNotFoundError(f"원본 DB 파일이 지정된 경로에 존재하지 않습니다: {DB_CSV_FILE}")

    try:
        # 1. 규칙 파일 로드 (FAISS 인덱스 구축에 사용)
        with open(RULES_FILE, 'r', encoding='utf-8') as f:
            category_rules = json.load(f)
        
        # 2. 검색 엔진 준비 및 인덱스 구축
        category_metadata = get_category_metadata(category_rules)
        
        faiss_engine = FaissSearchEngine(model_path=MODEL_PATH) 
        faiss_engine.build_index(category_metadata)
        
        # 3. 데이터 로드 및 매핑 실행
        df = pd.read_csv(DB_CSV_FILE)
        
        # ⭐️⭐️⭐️ df가 가지는 인덱스(Pandas 내부 인덱스)를 _original_index_에 저장 ⭐️⭐️⭐️
        # 이 컬럼은 DB 파일이 업데이트된 후에도 다음 실행에서 고유 키로 재사용됩니다.
        if '_original_index_' not in df.columns:
             df['_original_index_'] = df.index 
             
        print(f"'{DB_CSV_FILE}' 파일 로드 완료. 총 {len(df)}건 매핑 시작.")

        # 매핑 함수 실행 및 결과 분리
        # final_db_df는 모든 행을 보존하고 키워드가 덮어씌워진 최종 DF입니다.
        final_db_df, mapped_df, unmapped_df = map_keywords_to_dataframe(df, RULES_FILE, faiss_engine=faiss_engine)
        
        # 4. 결과 저장
        
        # 4-1. 전체 DB를 원본 파일에 업데이트 (매핑된/안된 행 모두 보존)
        print(f"\n[성공] 전체 {len(final_db_df)}건을 '{MAPPED_OUTPUT_FILE}'(원본 DB 파일)에 업데이트합니다. (모든 행 보존)")
        final_db_df.to_csv(MAPPED_OUTPUT_FILE, index=False, encoding='utf-8-sig')
        
        # 4-2. 매핑 안 된 데이터만 별도 파일로 저장 (다음 GPT 모듈의 입력 파일)
        print(f"[실패] 매핑 안 된 행 {len(unmapped_df)}건을 '{UNMAPPED_OUTPUT_FILE}'에 저장합니다. (키워드 추가/수정 필요)")
        unmapped_df.to_csv(UNMAPPED_OUTPUT_FILE, index=False, encoding='utf-8-sig')

        print("\n--- 매핑 완료. DB 파일 업데이트 내용 샘플 ---")
        # 최종 DB 파일의 상위 5개 행을 출력
        print(final_db_df[['title', 'upper_keyword', 'lower_keyword']].head(5)) #.to_markdown(index=False, numalign="left", stralign="left"))
        
        print("\n--- 매핑 안 된 결과 샘플 (unmapped_crawling.csv 내용 확인) ---")
        if not unmapped_df.empty:
            # _original_index_ 컬럼은 다음 모듈에서 사용되므로 출력 시에는 제외합니다.
            unmapped_df.drop(columns=['_original_index_'])
            print(unmapped_df[['title', 'upper_keyword', 'lower_keyword']].head(5))
        else:
            print("매핑 안 된 행이 없습니다.")
    
    except FileNotFoundError as e:
        print(f"\n[오류] 필요한 파일이 없습니다: {e.args[0]}. 프로젝트 경로 및 파일명 확인이 필요합니다.")
    except Exception as e:
        print(f"\n[오류] 모듈 실행 중 예기치 않은 오류가 발생했습니다: {e}.")