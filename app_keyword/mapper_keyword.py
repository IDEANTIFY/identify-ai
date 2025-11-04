import pandas as pd
import numpy as np
import faiss
import torch
import json
import re
import os
import tempfile
import traceback
from ast import literal_eval
from typing import Dict, List, Tuple, Any, Set, Optional, Union
from sentence_transformers import SentenceTransformer


# --- 프로젝트 설정 상수 ---
MODEL_PATH = "jhgan/ko-sroberta-multitask"
TEMP_COL = '_is_mapped_temp'

# L2(유클리드) 거리 임계값. (낮을수록 유사함)
L2_DISTANCE_THRESHOLD = 0.6

## -----------------------------------------------------------------------------
## 1. 공통 모듈
## - API(userDB)와 배치 처리(crawlingDB) 양쪽에서 모두 사용되는 유틸리티 함수들
## - AI 모델 로딩, 텍스트 전처리, 중복된 규칙 파싱 로직
## -----------------------------------------------------------------------------
def get_device() -> str:
    """ AI 모델을 실행할 최적의 장치(GPU/MPS/CPU)를 감지합니다. """
    if torch.cuda.is_available(): 
        return "cuda"
    elif torch.backends.mps.is_available(): 
        return "mps"
    else: 
        return "cpu"

def load_model(model_path: str, device: str) -> Optional[SentenceTransformer]:
    """ SentenceTransformer AI 모델을 로드합니다. """
    try:
        print(f"Sentence Transformer 모델을 로딩합니다... (Device: {device})", flush=True)
        model = SentenceTransformer(model_path, device=device)
        return model
    except Exception as e:
        print(f"⚠️ 치명적 오류: AI 모델 로딩에 실패했습니다. {e}")
        return None
        
def prepare_text(row: Union[pd.Series, Dict], cols: List[str] = ['title', 'content']) -> str:
    """ DataFrame or 딕셔너리에서 지정된 컬럼들을 합쳐 단일 문자열로 만듭니다. """
    parts = []
    for col in cols:
        value = row.get(col) if isinstance(row, dict) else row.get(col) 
        if pd.notna(value) and str(value).strip():
            parts.append(str(value).strip())
    return ' '.join(parts).strip()

## -----------------------------------------------------------------------------
## 2. 배치 처리용 모듈 (Batch Processing Modules)
## -----------------------------------------------------------------------------
# --- FAISS 검색 엔진 클래스 ---
class FaissSearchEngine:
    """(B) SentenceTransformer와 FAISS L2 거리 기준으로 의미 검색을 수행하는 엔진"""
    def __init__(self, model: SentenceTransformer):
        self.model = model
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


    def search(self, query_text: str, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:        
        """ 쿼리 텍스트와 가장 유사한 상위/하위 카테고리를 검색하고 (거리, 인덱스)를 반환합니다. """
        # 쿼리 임베딩 생성
        query_embedding = self.model.encode(
            [query_text], 
            convert_to_tensor=True
        ).cpu().numpy()
        
        # (k=1이든 k=T이든) 요청된 k만큼 검색
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices
        

# --- 유틸리티 및 매핑 함수 ---
def flatten_rules(category_rules: Dict) -> List[Tuple[str, str, str]]:
    """ JSON 규칙을 (상위, 하위, 패턴) 튜플 리스트로 평탄화합니다. """
    flat_rules = []
    if not isinstance(category_rules, dict): return flat_rules # 입력 유효성 검사

    for upper_cat, lower_cat_dict in category_rules.items():
        if not isinstance(lower_cat_dict, dict): continue # 2단계 딕셔너리 (하위 카테고리) 확인
        
        for lower_cat_raw, patterns in lower_cat_dict.items():
            if not isinstance(patterns, list): continue # 3단계 (패턴 리스트) 확인

            lower_cat = lower_cat_raw.strip()
            if not lower_cat: continue            

            for pattern in patterns:
                if isinstance(pattern, str) and pattern.strip():
                    flat_rules.append((upper_cat, lower_cat, pattern.strip()))
                elif not isinstance(pattern, str):
                    print(f"   ⚠️ 경고: '{upper_cat} - {lower_cat}' 규칙에서 비문자열 패턴(타입: {type(pattern)}) 발견 및 건너뜀.")
                    continue 
    return flat_rules

def get_category_metadata(category_rules: Dict) -> List[Tuple[str, str]]:
    """ FAISS 인덱스 구축을 위해 모든 고유한 (상위, 하위) 쌍을 추출 """
    categories: Set[Tuple[str, str]] = set()
    for upper_cat, lower_cat_dict in category_rules.items():
        if not isinstance(lower_cat_dict, dict): continue
        for lower_cat_raw in lower_cat_dict.keys():
            lower_cat = lower_cat_raw.strip()
            if lower_cat:
                categories.add((upper_cat, lower_cat))
    return sorted(list(categories))

def get_regex_for_batch(text: str, flat_rules: List[Tuple[str, str, str]]) -> Tuple[Set[str], Set[str]]:
    """ (A) 정규식 기반 매칭 (일치하는 *모든* 상위/하위 Set을 반환) """
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
    faiss_engine: FaissSearchEngine,
    T: int = 5                          # 💡 찾을 "전체 Top-T" 개수
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    [batch 메인] DataFrame에 키워드를 매핑 (A: Regex -> B: FAISS Top-T)
    """
    try:
        with open(rules_filepath, 'r', encoding='utf-8') as f:
            category_rules = json.load(f)
    except Exception as e:
        print(f"오류: 규칙 파일 로드 중 문제가 발생했습니다: {e}")
        return df.copy(), pd.DataFrame(), pd.DataFrame()

    flat_rules = flatten_rules(category_rules)
    
    if faiss_engine.index is None:
        print("경고: FAISS 인덱스가 구축되지 않아 정규식(A)만 사용합니다.")
    
    result_data = []
    for index, row in df.iterrows():
        text = prepare_text(row)
        upper_final = set()
        lower_final = set()

        # (A) 정규식 기반 매칭 시도 (일치하는 모든 것)
        regex_uppers, regex_lowers = get_regex_for_batch(text, flat_rules)
        upper_final.update(regex_uppers) # set.union()
        lower_final.update(regex_lowers) # set.union()
        # 정규식이 아예 0개였을 때만 print ---
        if not regex_uppers:
            print(f"  [Row {index}] ℹ️ [Batch] Regex 매칭 실패. FAISS로 탐색합니다...")

        # (B) FAISS Top-T 검색
        if faiss_engine.index is not None:
            distances, indices = faiss_engine.search(text, k=T)
            
            if indices.size > 0: 
                for i in range(indices.shape[1]):
                    dist = distances[0][i]
                    if dist < L2_DISTANCE_THRESHOLD:
                        idx = indices[0][i]
                        best_match = faiss_engine.metadata[idx] 
                        # FAISS 결과가 정규식 결과와 중복되어도 set이 알아서 처리
                        upper_final.add(best_match[0])
                        lower_final.add(best_match[1])
                    else:
                        break
        
        # 최종 결과 종합 ---
        is_mapped = bool(upper_final) # 매핑 성공 여부
        row_dict = row.to_dict() # 원본 데이터에 매핑 결과를 추가하고 리스트에 저장
        
        if is_mapped:
            row_dict['upper_keyword'] = ', '.join(sorted(list(upper_final)))
            row_dict['lower_keyword'] = ', '.join(sorted(list(lower_final)))
        else:
            row_dict['upper_keyword'] = ''
            row_dict['lower_keyword'] = ''
            
        row_dict[TEMP_COL] = is_mapped
        # _original_index_는 DataFrame의 index를 사용하도록 변경 (더 안전함)
        row_dict['_original_index_'] = index 
        
        result_data.append(row_dict)

    mapped_df_with_temp = pd.DataFrame(result_data)
    
    final_db_df = mapped_df_with_temp.drop(columns=[TEMP_COL, '_original_index_'])
    mapped_df = mapped_df_with_temp[mapped_df_with_temp[TEMP_COL]].drop(columns=[TEMP_COL, '_original_index_']) 
    unmapped_df = mapped_df_with_temp[~mapped_df_with_temp[TEMP_COL]].drop(columns=[TEMP_COL])
    
    return final_db_df, mapped_df, unmapped_df
    # mapper가 **mapped_df (성공한 행)**와 **unmapped_df (실패한 행)**를 분리하여 반환합니다.

## -----------------------------------------------------------------------------
## 3. API 처리용 모듈 (API Processing Module)
## -----------------------------------------------------------------------------
def get_regex_for_api(
    text: str, 
    lower_keyword_options: Dict[str, List[str]], # 1:Few 대상
) -> List[str]:
    """1:Few (부분) 정규식 매칭을 수행하고 Top-k를 반환"""
    
    matched_lowers_by_regex = set()
    for lower_cat_raw, patterns in lower_keyword_options.items():
        if not isinstance(patterns, list): continue
        lower_cat = lower_cat_raw.strip()
        if not lower_cat: continue
        for pattern in patterns:
            if isinstance(pattern, str) and pattern:
                try:
                    if re.search(pattern, text, re.IGNORECASE):
                        matched_lowers_by_regex.add(lower_cat) 
                        break 
                except re.error: continue
    
    if matched_lowers_by_regex:
        sorted_matches = sorted(list(matched_lowers_by_regex))
        return sorted_matches
    return []

def get_ai_matches_for_api(
    text: str,
    given_upper_keyword: str,
    lower_keyword_options: Dict[str, List[str]], # 1:Few 대상
    model: SentenceTransformer,
    k: int
) -> List[str]:
    """1:Few (부분) L2(NumPy) 검색을 수행하고 Top-k를 반환"""
    
    try:
        candidate_lowers_raw = list(lower_keyword_options.keys())
        if not candidate_lowers_raw: return []

        candidate_map = {} # {"상위 하위": "하위_원본"}
        for lower_raw in candidate_lowers_raw:
            lower_name = lower_raw.strip()
            if lower_name:
                 candidate_map[f"{given_upper_keyword} {lower_name}"] = lower_name

        if not candidate_map: return []

        candidate_texts = list(candidate_map.keys())

        text_embedding = model.encode([text], convert_to_tensor=False)[0] 
        candidate_embeddings = model.encode(candidate_texts, convert_to_tensor=False) 
        distances = np.linalg.norm(candidate_embeddings - text_embedding, axis=1)
        
        sorted_dist_indices = distances.argsort()
        
        ai_matches = []
        for idx in sorted_dist_indices:
            lowest_distance = distances[idx]
            if lowest_distance < L2_DISTANCE_THRESHOLD:
                best_match_text = candidate_texts[idx]
                best_match_lower = candidate_map[best_match_text]
                ai_matches.append(best_match_lower)
                if len(ai_matches) == k:
                    break
            else:
                break 
        
        return ai_matches

    except Exception as e:
        print(f"   ⚠️ L2 거리 비교 중 오류 발생: {e}")
        return []

def find_best_lower_keywords_top_k(
    text: str,
    given_upper_keywords: List[str], # 최소1개 ~ 최대4개 입력 받을 수 있음
    rules_dict: Dict[str, Dict[str, List[str]]],
    model: SentenceTransformer,
    k: int = 3  # (하나의 상위 키워드에 대해) '하위' k(임시)개를 찾도록 인자 추가
) -> List[str]:
    """
    [API 메인] 1~4개의 상위 키워드 리스트를 받아, 각각에 대해 (1) 정규식, (2) L2(NumPy)를 호출하여
    결과를 *모두* 조합한 뒤 최종 Top-K를 반환합니다.
    """

    # API 전달 받을 때의 오류
    if not text or not given_upper_keywords:
        print(f" ⚠️ 상위 키워드가 없거나 내용이 비어있음.")
        return []

    final_matches_set = set()
    total_regex_matches_for_log = set() # 로깅용 (1) 정규식
    total_ai_matches_for_log = set()    # 로깅용 (2) L2 거리(NumPy) 
    
    # 1~4개의 상위 키워드를 모두 순회
    for upper_kw in given_upper_keywords:
        
        # 현재 상위 키워드(upper_kw)가 유효한지 확인
        if not upper_kw or upper_kw not in rules_dict or not rules_dict[upper_kw]:
            print(f" ⚠️ 규칙에 '{upper_kw}' 상위 키워드가 없거나 하위 키워드가 비어있어 건너뜁니다.")
            continue # 다음 상위 키워드로 넘어감

        lower_keyword_options = rules_dict[upper_kw]
        
        # (A) 정규식 매칭 (일치하는 모든 것 찾기)
        regex_matches = get_regex_for_api(text, lower_keyword_options)
        final_matches_set.update(regex_matches)
        total_regex_matches_for_log.update(regex_matches)

        if not regex_matches:
            print(f"   ℹ️ [API] '{upper_kw}' > Regex 매칭 실패. L2(AI)로 탐색합니다...")

        # (B) L2 거리(NumPy) 비교 (AI)
        ai_matches = get_ai_matches_for_api( 
            text, upper_kw, lower_keyword_options, model, k
        )    
        final_matches_set.update(ai_matches)
        total_ai_matches_for_log.update(ai_matches)

    
    # 최종 결과 종합 ---
    sorted_final_matches = sorted(list(final_matches_set))
    final_top_k = sorted_final_matches[:k] # k개만큼 자르기
    
    print(f"   ℹ️  Regex 결과: {regex_matches}, AI 결과: {total_ai_matches_for_log}")
    print(f"   ✅ [Combined Top-{k}] 최종 매칭: {final_top_k}")
    
    return final_top_k


################################################################################
# --- 메인 실행 블록 ---
################################################################################
if __name__ == '__main__':    
    mock_rules_content = {
        "AI": {
            "머신러닝": ["머신러닝", "인공지능", "스타트업"]
        },
        "IT": {
            "스마트폰": ["아이폰", "갤럭시"]
        },
        "경영": {
            "운영": ["경영", "전략", "재무"]
        },
        "헬스케어": {
            "정신건강": ["명상", "심리"],
            "신체건강": ["운동", "헬스", "다이어트", "피트니스"]
        }
    }
    print("✅ 샘플 규칙 생성 완료.")

    DEVICE = get_device()
    MODEL = load_model(MODEL_PATH, DEVICE)

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.json', delete=True) as f:
        json.dump(mock_rules_content, f, ensure_ascii=False, indent=4)
        f.flush()
        TEMP_RULES_FILE_PATH = f.name
        print(f"✅ 샘플 규칙 임시 저장 (자동 삭제될 예정): {TEMP_RULES_FILE_PATH}")

        try:
            # --- 1. 배치 처리 (crawlingDB) 테스트 ---
            print("\n--- 📌 [1] map_keywords_to_dataframe 함수 (배치 처리) 테스트 ---")
            # 테스트를 위한 DataFrame 생성
            test_data = [
                {"title": "AI 스타트업의 경영 전략과 머신러닝", "content": "중요한 작업"}, 
                {"title": "마음 챙김과 심리 안정, 그리고 피트니스", "content": "멘탈 케어가 필요해"},
                {"title": "최신 휴대폰과 스마트폰 소식", "content": "애플, 삼성"}, 
            ]
            mock_df = pd.DataFrame(test_data).reset_index().rename(columns={'index': '_original_index_'})
            print("✅ 샘플 데이터 생성 완료.")
            print(mock_df)

            category_metadata = get_category_metadata(mock_rules_content)
            faiss_engine = FaissSearchEngine(model=MODEL)  # MODEL = load_model(MODEL_PATH, DEVICE)
            faiss_engine.build_index(category_metadata)

            final_df, mapped_df, unmapped_df = map_keywords_to_dataframe(
                mock_df,
                TEMP_RULES_FILE_PATH,
                faiss_engine,
                T=3 
            )
            
            print("\n[1-1] 최종 DF (모든 행)")
            print(final_df[['title', 'upper_keyword', 'lower_keyword']].to_markdown(index=False, numalign="left", stralign="left"))
            print("\n[1-2] 미매핑된 DF (unmapped_df)")
            print(unmapped_df[['title', 'upper_keyword', 'lower_keyword']].to_markdown(index=False, numalign="left", stralign="left"))


            # --- 2. API 처리 (userDB) 테스트 ---
            print("\n\n--- 📌 [2] find_best_lower_keywords_top_k 함수 (API) 테스트 ---")
            
            test_text_api = "정신건강과 피트니스를 위한 루틴"
            print(f"입력 텍스트: \"{test_text_api}\"")

            # 2-1: 실패 케이스
            test_upper_api_fail = "IT"
            print(f"\n[2-1] 상위: \"{test_upper_api_fail}\", k=3")
            found_lowers_1 = find_best_lower_keywords_top_k(
                test_text_api,
                [test_upper_api_fail],
                mock_rules_content, 
                MODEL,
            )

            # 2-2: 성공 케이스 (상위 별 매칭 3개)
            test_upper_api_success = "헬스케어"
            print(f"\n[2-2] 상위: \"{test_upper_api_success}\", k=3")
            found_lowers_2 = find_best_lower_keywords_top_k(
                test_text_api,
                [test_upper_api_success],
                mock_rules_content,
                MODEL,
            )
            
            # 2-3: 상위 2개 동시 전달 테스트
            print(f"\n[2-3] 상위: \"IT\"와 \"헬스케어\" 동시 전달, k=3")
            found_lowers_3 = find_best_lower_keywords_top_k(
                test_text_api,
                ["IT", "헬스케어"], # [추가] 2개 리스트 동시 전달
                mock_rules_content,
                MODEL,
            )
            print("\n--- [ 모든 테스트 완료 ] ---")

        except Exception as e_main:
            print(f"\n--- [ 테스트 중 오류 발생 ] ---")
            print(f"오류: {e_main}")
            traceback.print_exc()