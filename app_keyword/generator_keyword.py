import os
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Optional, Tuple, Any, Union, Set
import traceback  # 테스트 블록을 위해 임포트
import json

# --- 경로 설정 ---
CURRENT_DIR = Path(__file__).parent 
PROJECT_ROOT_PATH = CURRENT_DIR.parent
load_dotenv(PROJECT_ROOT_PATH / ".env")

# --- 파일 경로 정의 (generator는 이 파일들을 직접 사용하진 않음) ---
# JSON_RULE_FILE = CURRENT_DIR / "category_rules.json"
# CRAWLING_CSV_FILE = os.path.join(PROJECT_ROOT_PATH, "dataset", "crawling_db.csv") 
# UNMAPPED_CSV_FILE = os.path.join(PROJECT_ROOT_PATH, "dataset", "unmapped_crawling.csv")

# --- OpenAI 클라이언트 초기화 ---
API_KEY_STRING = os.environ.get("OPENAI_API_KEY_ahyun") 
client: Optional[OpenAI] = None
if API_KEY_STRING and API_KEY_STRING != "YOUR_API_KEY_HERE":
    try:
        client = OpenAI(api_key=API_KEY_STRING)
        print("✅ [Generator] OpenAI 클라이언트 초기화 성공.")
    except Exception as e:
        print(f"❌ [Generator] OpenAI 클라이언트 초기화 실패: {e}")
        client = None
else:
    print("⚠️ [Generator] 경고: API_KEY가 설정되지 않아 GPT 호출이 비활성화됩니다.")


## -----------------------------------------------------------------------------
## 공통 모듈 (Mapper에서 가져와서 활용)
## -----------------------------------------------------------------------------
from mapper_keyword import prepare_text


## -----------------------------------------------------------------------------
## 1. API 처리용 함수 (by run_api.py)
## 배치 처리에서 하위만 생성할 때 재사용함!
## -----------------------------------------------------------------------------
def generate_keywords_for_api(
    text: str, 
    given_upper_keywords: List[str],  # 최소1개 ~ 최대4개 입력 받을 수 있음
    k: int = 3
) -> List[str]:
    """
    [Gen-API] 텍스트와 1~4개의 '상위 키워드 리스트'를 받아, 
    각각 GPT 호출 후 결과를 통합하여 '하위 키워드' Top-K개를 생성합니다.
    """
    if not client:
        print("   ❌ [Gen-API] OpenAI client가 없어 생성을 건너뜁니다.")
        return []
    if not text or not given_upper_keywords:
        return []

    final_new_keywords_set = set() # 모든 결과를 합칠 Set

    # [수정] 1~4개의 상위 키워드를 순회
    for upper_kw in given_upper_keywords:
        if not upper_kw: continue

        system_prompt = f"""
        당신은 텍스트를 분석하여 '{upper_kw}' 카테고리에 속하는
        새로운 하위 키워드를 생성하는 AI입니다.
        - 하위 키워드는 명사형으로, 1~3단어 이내로 간결하게 만드세요.
        - 텍스트의 핵심 내용을 잘 반영해야 합니다.
        - 요청한 K개수만큼, 가장 적절한 순서대로 콤마(,)로 구분하여 응답하세요.
        - 예시: AI, 머신러닝, 딥러닝
        """
        human_prompt = f"""
        텍스트: "{text}"
        상위 카테고리: "{upper_kw}"
        가장 적합한 하위 키워드 {k}개를 생성해주세요.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": human_prompt}
                ],
                temperature=0.3,
                max_tokens=50
            )
            result_text = response.choices[0].message.content.strip()
            new_keywords = [kw.strip() for kw in result_text.split(',') if kw.strip()]
            
            if new_keywords:
                print(f"   ✅ [Gen-API] (상위: {upper_kw}) -> 생성: {new_keywords}")
                final_new_keywords_set.update(new_keywords) # Set에 누적
            else:
                 print(f"   ⚠️ [Gen-API] (상위: {upper_kw}) -> GPT가 키워드를 생성하지 못했습니다.")
            
        except Exception as e:
            print(f"   ❌ [Gen-API] (상위: {upper_kw}) -> GPT 호출 중 오류: {e}")

    # 모든 Set 결과를 정렬하여 최종 Top-K 반환
    sorted_final_list = sorted(list(final_new_keywords_set))
    print(f"   ℹ️  [Gen-API] (종합) 최종 생성 {len(sorted_final_list)}개 -> Top-{k}: {sorted_final_list[:k]}")
    return sorted_final_list[:k]

## -----------------------------------------------------------------------------
## 2. 배치 처리용 함수 (by run_batch.py)
## -----------------------------------------------------------------------------
# def _parse_gpt_batch_response(text: str) -> Tuple[str, str]:
#     """ "상위: A, 하위: B" 형식의 텍스트를 파싱합니다. """
#     try:
#         # 정규표현식 대신 간단한 분리 사용
#         parts = text.split(',')
#         upper_raw = parts[0]
#         lower_raw = parts[1]
        
#         upper = upper_raw.split(':')[1].strip()
#         lower = lower_raw.split(':')[1].strip()
        
#         return upper, lower
#     except Exception:
#         print(f"   ⚠️ [Gen-Batch] GPT 응답 파싱 실패: {text}")
#         return "", ""
def _parse_gpt_json_list(text: str) -> Tuple[Set[str], Set[str]]:
    """ 
    (Case 2용) GPT가 반환한 JSON 리스트 문자열을 파싱합니다.
    (e.g. '[{"upper": "AI", "lower": "딥러닝"}, ...]')
    """
    uppers = set()
    lowers = set()
    try:
        # GPT가 텍스트 블록(```json ... ```)으로 감쌀 경우를 대비
        if text.startswith("```json"):
            text = text[7:-3].strip() # ```json 마커와 ``` 마커 제거

        # GPT가 응답에 불필요한 따옴표(' 또는 ")를 포함할 경우를 대비해 제거
        text = text.strip().strip("'").strip('"')
        data = json.loads(text) # JSON 문자열을 Python 객체(List[Dict])로 변환
        
        if not isinstance(data, list):
            print("   ⚠️ [Gen-Batch-Parse] GPT가 리스트(list) 형식을 반환하지 않음.")
            return uppers, lowers
        
        for item in data:
            if isinstance(item, dict):
                u = item.get("upper")
                l = item.get("lower")
                if u and l: # 상위, 하위가 모두 존재하면
                    uppers.add(u.strip())
                    lowers.add(l.strip())
            else:
                print(f"   ⚠️ [Gen-Batch-Parse] 리스트 내의 항목이 dict가 아님: {item}")

    except json.JSONDecodeError:
        print(f"   ⚠️ [Gen-Batch-Parse] JSON 파싱 실패. GPT 응답: {text}")
    except Exception as e:
        print(f"   ⚠️ [Gen-Batch-Parse] 알 수 없는 파싱 오류: {e}")
        
    return uppers, lowers

def generate_keywords_for_batch(unmapped_df: pd.DataFrame) -> pd.DataFrame:
    """
    [Gen-Batch] Mapper가 실패한 DataFrame에 대해 새 (상위, 하위) 키워드를 생성합니다.
    (반환값: 생성에 성공한 데이터로 구성된 새 DataFrame)
    """
    if not client:
        print("❌ [Gen-Batch] OpenAI client가 없어 생성을 건너뜁니다.")
        return pd.DataFrame() # 빈 DF 반환

    if unmapped_df.empty:
        print("ℹ️ [Gen-Batch] 생성할 데이터(unmapped_df)가 없습니다.")
        return pd.DataFrame()


    generated_data = []  # 성공적으로 생성된 데이터를 (딕셔너리 형태) 저장할 리스트    
    print(f"총 {len(unmapped_df)}개의 미매핑 데이터에 대해 키워드 생성을 시작합니다...")
    
    for index, row in unmapped_df.iterrows():
        text = prepare_text(row)
        if not text:
            continue

        ## 원본 데이터에서 'upper_keyword'를 가져옴
        existing_upper = row.get('upper_keyword')

        # [Case 1] 상위 키워드가 *이미* 있는 경우 (하위 키워드만 생성)
        if pd.notna(existing_upper) and str(existing_upper).strip():
            current_uppers_list = [up.strip() for up in str(existing_upper).split(',') if up.strip()]
            print(f"  [Gen-Batch] Row {row.get('_original_index_', index)}: (Case 1) 하위 생성 (상위: {current_uppers_list})")
            try:
                # [4] API용 함수를 재활용 (주의: API용 함수는 List[str]을 반환함)
                new_lowers_list = generate_keywords_for_api(
                    text=text,
                    given_upper_keywords=current_uppers_list,
                    k=3 # 배치에서도 3개 생성 (혹은 k=3)
                )

                if new_lowers_list:
                    row_dict = row.to_dict()
                    # 기존 상위 키워드는 유지, 생성된 하위 키워드(들) 저장
                    row_dict['lower_keyword'] = ', '.join(new_lowers_list) # List[str] -> "키워드1, 키워드2" (문자열)로 변환하여 저장
                    generated_data.append(row_dict)
                    
            except Exception as e_api:
                print(f"    ❌ (Case 1) 오류: {e_api} / Row {row.get('_original_index_', index)}")        

        # [Case 2] 상위 키워드가 *없는* 경우 (상/하위 키워드 모두 생성)        
        else:
            if index % 10 == 0: # 10건마다 로그 출력
                 print(f"  [Gen-Batch] Row {row.get('_original_index_', index)}: (Case 2) 상/하위 모두 생성")

            system_prompt = """
            당신은 텍스트를 분석하여 (상위 카테고리, 하위 카테고리) 쌍을
            생성하는 AI입니다.
            - 텍스트와 관련된 (상위, 하위) 쌍을 최대 3개까지 생성하세요.
            - 응답은 반드시 JSON 리스트 형식으로만 답하세요.
            - 형식: '[{"upper": "...", "lower": "..."}, ...]'
            - 예시: '[{"upper": "AI", "lower": "딥러닝"}, {"upper": "IT", "lower": "클라우드 보안"}]'
            """
            human_prompt = f"""
            텍스트: "{text}"
            이 텍스트의 핵심 주제에 대한 (상위, 하위) 키워드 쌍을
            JSON 리스트 형식으로 생성해주세요.
            """
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": human_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=200 # JSON 응답을 위해 넉넉하게
                )
                result_text = response.choices[0].message.content.strip()
                
                # 새로운 JSON 파서 호출
                new_uppers_set, new_lowers_set = _parse_gpt_json_list(result_text)
            
                if new_uppers_set and new_lowers_set:
                    row_dict = row.to_dict()
                    # Set -> 정렬된 리스트 -> 콤마(,)로 구분된 문자열
                    row_dict['upper_keyword'] = ', '.join(sorted(list(new_uppers_set)))
                    row_dict['lower_keyword'] = ', '.join(sorted(list(new_lowers_set)))
                    generated_data.append(row_dict)
                else:
                    print(f"    ℹ️ (Case 2) 파싱 결과가 비어있음 / Row {row.get('_original_index_', index)}")
                    
            except Exception as e_all:
                print(f"    ❌ (Case 2) 오류: {e_all} / Row {row.get('_original_index_', index)}")                
     
    print(f"✅ [Gen-Batch] 총 {len(generated_data)}개 키워드 생성 완료.")    
    # 생성된 데이터를 새 DataFrame으로 반환
    return pd.DataFrame(generated_data)



################################################################################
# --- 메인 실행 블록 ---
################################################################################
if __name__ == '__main__':
    
    if client is None:
        print("❌ [TEST] OpenAI 클라이언트가 초기화되지 않았습니다.")
    else:
        print("\n--- 📌 [Generator] 모듈 테스트 시작 (OpenAI client 사용) ---")
        
        try:
            # --- 1. 배치 처리 (generate_keywords_for_batch) 테스트 ---
            print("\n--- 📌 [1] 배치 테스트 (Mapper 결과 연동 및 최종 업데이트) ---")
            
            # (1) Mapper가 '성공'한 데이터 (2건)
            mapped_batch_data = [
                {"_original_index_": 0, "title": "AI 스타트업의 경영 전략과 머신러닝", "content": "...", "upper_keyword": "AI, 경영", "lower_keyword": "머신러닝, 운영"},
                {"_original_index_": 1, "title": "마음 챙김과 심리 안정, 그리고 피트니스", "content": "...", "upper_keyword": "헬스케어", "lower_keyword": "신체건강, 정신건강"}
            ]
            mock_mapped_df = pd.DataFrame(mapped_batch_data) # [수정] 2건만

            # (2) Mapper가 '실패'한 데이터 (1건) + (Case 1 테스트 1건)
            unmapped_batch_data = [
                { # (Case 2) Mapper가 실제로 실패한 데이터
                    "_original_index_": 2, "title": "최신 휴대폰과 스마트폰 소식", "content": "애플, 삼성", "upper_keyword": "", "lower_keyword": ""
                },
                { # (Case 1) 1~4개 로직 테스트용 (상위 2개)
                    "_original_index_": 3, "title": "새로운 AI 딥러닝 모델과 클라우드 보안 전략", "content": "...", "upper_keyword": "AI, IT", "lower_keyword": ""
                }
            ]
            mock_unmapped_df = pd.DataFrame(unmapped_batch_data) # [수정] 2건
            
            print("✅ [Test 1-1] Generator 입력 DF (unmapped_df):")
            print(mock_unmapped_df[['_original_index_', 'title', 'upper_keyword']].to_markdown(index=False, numalign="left", stralign="left"))
            
            # --- Generator 배치 함수 실행 ---
            generated_df = generate_keywords_for_batch(mock_unmapped_df)
            
            print("\n✅ [Test 1-2] Generator 생성 결과 (generated_df):")
            if not generated_df.empty:
                print(generated_df[['_original_index_', 'title', 'upper_keyword', 'lower_keyword']].to_markdown(index=False, numalign="left", stralign="left"))
            else: print("-> 생성된 데이터가 없습니다.")

            # (3) 'run_batch.py'의 최종 업데이트 시뮬레이션
            print("\n✅ [Test 1-3] 최종 DF 업데이트 시뮬레이션 (Mapped + Generated):")
            # [수정] Mapped(2건) + Generated(2건) = 4건 (테스트 포함)
            final_df_simulation = pd.concat([mock_mapped_df, generated_df], ignore_index=True)
            print(final_df_simulation[['_original_index_', 'title', 'upper_keyword', 'lower_keyword']].to_markdown(index=False, numalign="left", stralign="left"))


            # --- 2. API 처리 (generate_keywords_for_api) 테스트 ---
            # ... (API 테스트 로직은 동일) ...
            print("\n\n--- 📌 [2] API 테스트 (Mapper 시나리오별) ---")
            
            test_text_api = "정신건강과 피트니스를 위한 루틴"
            print(f"입력 텍스트: \"{test_text_api}\"")
            print(f"\n[Test 2-1] 상위: [\"IT\"], k=2 (Mapper 실패 케이스)")
            generated_lowers_1 = generate_keywords_for_api(test_text_api, ["IT"], k=2)
            print(f"\n[Test 2-2] 상위: [\"헬스케어\"], k=2 (Mapper 성공 케이스 - 비교용)")
            generated_lowers_2 = generate_keywords_for_api(test_text_api, ["헬스케어"], k=2)
            print(f"\n[Test 2-3] 상위: [\"IT\", \"헬스케어\"], k=3 (상위 2개 동시 전달)")
            generated_lowers_3 = generate_keywords_for_api(test_text_api, ["IT", "헬스케어"], k=3)
            
            print("\n--- [ 모든 테스트 완료 ] ---")

        except Exception as e_main:
            print(f"\n--- [ 테스트 중 오류 발생 ] ---")
            print(f"오류: {e_main}")
            traceback.print_exc()