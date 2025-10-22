import pandas as pd
import json
import os
import re
from openai import OpenAI, APIError
from typing import Optional, Tuple, List, Any, Set, Dict
from ast import literal_eval
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
    
# --- 0. 환경 설정 및 API 클라이언트 초기화 ---
# API_KEY_STRING = os.environ.get("OPENAI_API_KEY_ahyun", "")
API_KEY_STRING = os.getenv("OPENAI_API_KEY")
# 파일 및 모델 경로
MODEL_NAME = "gpt-5-nano" # "gpt-4o-500k-ko-5nano"
PROJECT_ROOT_PATH = "/home/work/Team_AI/identify-ai"

DB_CSV_FILE = os.path.join(PROJECT_ROOT_PATH, "dataset", "crawling_db.csv")
ORIGINAL_CSV_FILE = DB_CSV_FILE 
JSON_RULE_FILE = os.path.join(PROJECT_ROOT_PATH, "app_keyword", "category_rules.json")
UNMAPPED_CSV_FILE = os.path.join(PROJECT_ROOT_PATH, "dataset", "unmapped_crawling.csv")
TEXT_COLUMN_NAMES = ['title', 'content', 'etc'] # GPT 입력에 사용할 컬럼 목록

# OpenAI 클라이언트 초기화 (API 키는 환경 변수에서 로드됨)
client: Optional[OpenAI] = None
if API_KEY_STRING and API_KEY_STRING != "YOUR_API_KEY_HERE":
    try:
        client = OpenAI(api_key=API_KEY_STRING)
        # print("✅ OpenAI 클라이언트 초기화 성공.")
    except Exception as e:
        # print(f"❌ OpenAI 클라이언트 초기화 실패 (API 호출 불가): {e}")
        client = None
# else:
    # print("⚠️ 경고: API_KEY가 설정되지 않아 GPT 호출을 건너뜁니다.")


# --- 1. 유틸리티 함수 정의 ---

def load_simplified_rules(file_path: str) -> dict:
    """JSON 파일을 읽어 바로 반환하며, 파일이 없으면 빈 딕셔너리를 반환합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # print(f"경고: JSON 규칙 파일 '{file_path}'을 찾을 수 없습니다. 새로운 파일로 시작합니다.")
        return {}
    except Exception as e:
        # print(f"경고: JSON 규칙 파일 로드 실패. 오류: {e}")
        return {}

def extract_existing_keywords(rules: dict) -> Tuple[str, str, Set[str]]:
    """로드된 룰 딕셔너리에서 모든 상위 및 하위 키워드를 추출하여 문자열과 Set으로 반환합니다."""
    
    existing_upper_set = set(rules.keys())
    upper_str_to_reuse = ', '.join(sorted(list(existing_upper_set)))

    existing_lower_set = set()
    for sub_categories in rules.values():
        for key in sub_categories.keys():
             try:
                if key.startswith("['") and key.endswith("']"):
                    parsed_keys = literal_eval(key)
                    existing_lower_set.update(parsed_keys)
                else:
                    existing_lower_set.add(key)
             except (ValueError, SyntaxError):
                existing_lower_set.add(key)

    lower_str_to_avoid = ', '.join(sorted(list(existing_lower_set)))
    
    return upper_str_to_reuse, lower_str_to_avoid, existing_lower_set

def combine_content_columns(row: pd.Series, column_names: List[str]) -> str:
    """title, content, etc 컬럼을 합쳐 GPT 입력용 단일 문자열을 생성합니다."""
    parts = []
    
    for col in column_names:
        if col in row and pd.notna(row[col]):
            parts.append(str(row[col]).strip())
        
    return ', '.join(parts).strip()


# --- 2. GPT API 호출 함수 ---

def generate_keywords(content: str, existing_upper_str: str, lower_str_to_avoid: str) -> Tuple[Optional[str], Optional[str], Optional[List[str]]]:
    """GPT API를 호출하여 콘텐츠에 대한 상위/하위 키워드와 정규식 패턴을 제안합니다."""
    if client is None:
        return None, None, None

    cleaned_content = str(content).strip()
    if not cleaned_content or cleaned_content.lower() == 'nan':
        return None, None, None

    # 시스템 프롬프트 (규칙 강조 및 정규식 생성 지시)
    system_prompt = (
        "당신은 기술 콘텐츠를 분석하여 새로운 키워드와 해당 키워드를 식별할 수 있는 정규식 패턴을 제안하는 전문가입니다. "
        "제공된 'Content'를 분석하여 가장 적합한 '상위 키워드', '하위 키워드', 그리고 하위 키워드와 관련된 **3~5개의 핵심 정규식 패턴 리스트**를 제안해야 합니다. "
        "\n**규칙:**\n"
        f"1. 상위 키워드 제안 시, **현재 시점의 최신 목록**('{existing_upper_str}') 중 하나를 사용하거나, 적합한 기존 항목이 없으면 새로운 상위 키워드를 생성하세요.\n"
        f"2. 하위 키워드는 반드시 새로운 것을 생성해야 하며, **방금 생성된 규칙을 포함한 최신 금지 목록**('{lower_str_to_avoid}')에 있는 것은 사용하지 마세요.\n"
        "3. 만약 적절한 하위 키워드가 여러 개라고 판단되면, ', ' (파이프 스페이스)로 구분하여 모두 제공하세요.\n"
        "4. **정규식 패턴은 반드시 리스트 형태로 반환하며, 키워드를 식별할 수 있는 핵심 단어**여야 합니다. (예: ['인공지능', 'AI', '머신러닝'])\n"
        "응답은 반드시 지정된 JSON 형식으로만 반환해야 합니다. 응답 JSON에는 **'upper_keyword'**, **'lower_keyword'**, **'regex_patterns'** 필드만 포함해야 합니다."
    )
    user_content = (
        f"상세 내용(Content - Title, Content, ETC 정보 결합됨): {cleaned_content}\n\n"
        "분석 결과를 JSON 형식으로 반환하세요."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            timeout=30.0
        )

        if not response.choices or not response.choices[0].message.content:
             return None, None, None

        response_text = response.choices[0].message.content
        result_dict = json.loads(response_text)

        upper_kw = result_dict.get('upper_keyword', None)
        lower_kw = result_dict.get('lower_keyword', None)
        regex_patterns = result_dict.get('regex_patterns', []) 

        if isinstance(upper_kw, list):
            upper_kw = ', '.join([str(k) for k in upper_kw])
        if isinstance(lower_kw, list):
            lower_kw = ', '.join([str(k) for k in lower_kw])
        
        if not isinstance(regex_patterns, list):
             regex_patterns = []

        return upper_kw, lower_kw, regex_patterns

    except APIError:
        pass
    except json.JSONDecodeError:
        pass
    except Exception:
        pass
        
    return None, None, None


# --- 3. JSON 규칙 즉시 업데이트 함수 ---

def update_rules_with_single_suggestion(
    rules: Dict[str, Dict[str, List[Any]]], 
    suggested_upper_str: str, 
    suggested_lower_str: str,
    regex_patterns: List[str],
    dynamic_lower_set: Set[str]
) -> Tuple[Dict[str, Dict[str, List[Any]]], Set[str], Optional[str]]:
    """
    단일 제안을 기존 규칙 딕셔너리에 병합하고, JSON 파일로 저장합니다.
    """
    suggested_upper = suggested_upper_str.split(', ')[0].strip()
    
    if not suggested_upper or not suggested_lower_str:
        return rules, dynamic_lower_set, None

    suggested_lower_raw = str(suggested_lower_str).replace('[', '').replace(']', '').replace("'", '').strip()
    if ', ' in suggested_lower_raw:
        lower_keywords = [kw.strip() for kw in suggested_lower_raw.split(', ') if kw.strip()]
    elif ',' in suggested_lower_raw:
        lower_keywords = [kw.strip() for kw in suggested_lower_raw.split(',') if kw.strip()]
    else:
        lower_keywords = [suggested_lower_raw] if suggested_lower_raw else []

    new_upper_generated = None
    merge_count = 0
    
    if suggested_upper not in rules:
        rules[suggested_upper] = {}
        new_upper_generated = suggested_upper
    
    for suggested_lower in lower_keywords:
        if not suggested_lower: continue

        if suggested_lower not in dynamic_lower_set: 
            if suggested_upper not in rules:
                 rules[suggested_upper] = {}

            rules[suggested_upper][suggested_lower] = regex_patterns # ⭐️ 정규식 패턴 저장 ⭐️
            merge_count += 1
            dynamic_lower_set.add(suggested_lower)

    if merge_count > 0:
        try:
            os.makedirs(os.path.dirname(JSON_RULE_FILE), exist_ok=True)
            with open(JSON_RULE_FILE, 'w', encoding='utf-8') as f:
                json.dump(rules, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"❌ 오류: JSON 파일 저장 중 문제가 발생했습니다: {e}")
            
    return rules, dynamic_lower_set, new_upper_generated


# --- 4. 메인 실행 함수 (즉시 반영 로직 및 원본 업데이트) ---
def process_unmapped_data():
    """미매핑된 CSV를 처리하고, 생성된 규칙을 즉시 반영하며 원본 CSV에 업데이트합니다."""

    print("--- GPT API 기반 키워드 생성 시작 (즉시 반영 모드) ---")

    # 0. 파일 존재 확인
    if not os.path.exists(UNMAPPED_CSV_FILE):
        print(f"❌ 오류: 미매핑 검토 파일 '{UNMAPPED_CSV_FILE}'을 찾을 수 없습니다. 'mapper_keyword.py'를 먼저 실행하여 파일을 생성하세요.")
        return
    
    # 1. 초기 규칙 로드 및 동적 변수 초기화
    current_rules = load_simplified_rules(JSON_RULE_FILE)
    existing_upper_str, lower_str_avoid_init, dynamic_lower_keywords_set = extract_existing_keywords(current_rules)
    
    current_existing_upper_str = existing_upper_str
    
    # 2. 미매핑 CSV 로드
    df_unmapped = pd.read_csv(UNMAPPED_CSV_FILE)
    initial_rows = len(df_unmapped)
    print(f"✅ 미매핑 항목 {initial_rows}개 로드 완료. GPT API 호출 시작...")

    # 결과를 저장할 리스트 초기화
    suggested_upper_kws = []
    suggested_lower_kws = []
    
    # 3. GPT API 함수 적용 (행 단위 순회 및 즉시 반영)
    for index, row in df_unmapped.iterrows():
        # 3-1. 3개 컬럼 결합
        combined_text = combine_content_columns(row, TEXT_COLUMN_NAMES) 
        
        # 현재까지 생성된 하위 키워드를 포함하여 GPT에게 전달할 금지 목록 문자열 생성
        current_lower_str_to_avoid = ', '.join(sorted(list(dynamic_lower_keywords_set)))
        
        # 3-2. GPT 호출 (현재의 최신 규칙/목록 반영)
        upper_kw, lower_kw, regex_patterns = generate_keywords(
            combined_text, 
            current_existing_upper_str, 
            current_lower_str_to_avoid  
        )

        suggested_upper_kws.append(upper_kw)
        suggested_lower_kws.append(lower_kw)

        # 3-3. ⭐️ 즉시 반영 로직 (핵심) ⭐️
        # GPT가 키워드를 생성했고, 패턴 리스트가 비어있지 않은 경우에만 업데이트
        if upper_kw and lower_kw and regex_patterns:
            
            current_rules, dynamic_lower_keywords_set, new_upper_name = \
                update_rules_with_single_suggestion(
                    current_rules,
                    upper_kw,
                    lower_kw,
                    regex_patterns, # ⭐️ 정규식 패턴 전달 ⭐️
                    dynamic_lower_keywords_set
                )
            
            # 다음 행의 GPT 프롬프트에 사용될 '기존 상위 키워드' 목록 업데이트
            if new_upper_name:
                current_existing_upper_set = set(current_rules.keys())
                current_existing_upper_str = ', '.join(sorted(list(current_existing_upper_set)))
                print(f"💡 신규 상위 키워드 '{new_upper_name}' 즉시 반영.")


    # 컬럼 이름 변경 반영
    df_unmapped['upper_keyword_new'] = suggested_upper_kws # 임시 컬럼명 사용
    df_unmapped['lower_keyword_new'] = suggested_lower_kws # 임시 컬럼명 사용

    # 4. 결과 분류 및 원본 CSV 업데이트
    
    # 4-1. GPT가 키워드를 생성한 행과 실패한 행 분리
    df_unmapped['is_fixed'] = df_unmapped['upper_keyword_new'].notna()
    df_fixed_by_gpt = df_unmapped[df_unmapped['is_fixed']].copy()
    
    # 4-2. 원본 crawling_db.csv 업데이트 (핵심: _original_index_ 기준 병합)
    if not df_fixed_by_gpt.empty:
        # DB 파일 로드
        df_original = pd.read_csv(ORIGINAL_CSV_FILE)
        
        # '_original_index_'를 찾거나 생성하여 병합 키로 사용
        if '_original_index_' not in df_original.columns:
            df_original['_original_index_'] = df_original.index
            
        # 병합에 필요한 컬럼만 추출
        df_new_kws = df_fixed_by_gpt[['_original_index_', 'upper_keyword_new', 'lower_keyword_new']]
        
        # 원본 DF와 GPT 결과 DF를 병합
        df_original = pd.merge(
            df_original, 
            df_new_kws, 
            on=['_original_index_'], 
            how='left'
        )
        
        # 2. 최종 키워드 컬럼 업데이트: GPT가 만든 새 값이 있으면 덮어쓰고, 없으면 기존 값 유지
        df_original['upper_keyword'] = df_original['upper_keyword_new'].fillna(df_original.get('upper_keyword', ''))
        df_original['lower_keyword'] = df_original['lower_keyword_new'].fillna(df_original.get('lower_keyword', ''))
        
        # 3. 임시/중복 컬럼 및 인덱스 컬럼 삭제
        cols_to_drop = [col for col in df_original.columns if col.endswith('_new') or col == '_original_index_']
        df_original = df_original.drop(columns=cols_to_drop, errors='ignore')
        
        # 4. 업데이트된 원본 DB 파일 저장
        os.makedirs(os.path.dirname(ORIGINAL_CSV_FILE), exist_ok=True)
        df_original.to_csv(ORIGINAL_CSV_FILE, index=False, encoding='utf-8-sig')
        print(f"[성공] GPT로 매핑된 항목 {len(df_fixed_by_gpt)}건을 '{ORIGINAL_CSV_FILE}'에 업데이트했습니다.")
    
    # 4-3. GPT도 매핑 못한 애들 저장 (unmapped_crawling.csv 파일 업데이트)
    # 불필요한 키워드 컬럼 제거 후 저장
    cols_to_drop_unfixed = ['upper_keyword_new', 'lower_keyword_new', 'is_fixed']
    df_unfixed_by_gpt = df_unmapped[~df_unmapped['is_fixed']].drop(columns=cols_to_drop_unfixed, errors='ignore')
    
    os.makedirs(os.path.dirname(UNMAPPED_CSV_FILE), exist_ok=True)
    df_unfixed_by_gpt.to_csv(UNMAPPED_CSV_FILE, index=False, encoding='utf-8-sig')
    print(f"[실패] GPT도 매핑하지 못한 항목 {len(df_unfixed_by_gpt)}건을 '{UNMAPPED_CSV_FILE}'에 다시 저장했습니다. (수동 검토 필요)")

    print("------------------------------------------------------------------")
    
    # ⭐️⭐️⭐️ 추가된 샘플 출력 로직 ⭐️⭐️⭐️
    if not df_fixed_by_gpt.empty:
        print("\n--- ✅ GPT 생성으로 추가 매핑된 결과 샘플 (5건) ---")
        preview_cols_fixed = ['title', 'upper_keyword_new', 'lower_keyword_new']
        # 임시 컬럼이므로 df_unmapped에서 추출 후 출력 (실제 업데이트 값 확인)
        print(df_unmapped[df_unmapped['is_fixed']][preview_cols_fixed].head(5).to_markdown(index=False, numalign="left", stralign="left"))
    else:
        print("\n--- ✅ GPT로 추가 매핑된 행이 없습니다. ---")


    if not df_unfixed_by_gpt.empty:
        print("\n--- ❌ GPT도 매핑하지 못한 최종 미해결 목록 샘플 (unmapped_crawling.csv) ---")
        # 실패한 행은 title, content만 출력
        preview_cols_unfixed = ['title', 'content']
        print(df_unfixed_by_gpt[preview_cols_unfixed].head(5).to_markdown(index=False, numalign="left", stralign="left"))
    else:
        print("매핑 안 된 행이 없습니다. 훌륭합니다!")


# --- 5. 모듈 실행 시 메인 블록 ---

if __name__ == "__main__":
    load_dotenv()
    # 0. 필수 폴더 생성 확인 (규칙 파일의 폴더 경로)
    rules_dir = os.path.dirname(JSON_RULE_FILE)
    if rules_dir and not os.path.exists(rules_dir):
        os.makedirs(rules_dir, exist_ok=True)
        print(f"[{rules_dir}] 폴더를 생성했습니다.")

    # 1. GPT 호출을 통해 미매핑된 데이터에 키워드 제안을 추가하고 CSV 저장 (즉시 반영됨)
    process_unmapped_data()