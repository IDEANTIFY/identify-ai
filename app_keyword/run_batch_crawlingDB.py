import os
import pandas as pd
import json
from pathlib import Path
from dotenv import load_dotenv

import mapper_keyword
import generator_keyword

CURRENT_DIR = Path(__file__).parent 
PROJECT_ROOT_PATH = CURRENT_DIR.parent
load_dotenv(PROJECT_ROOT_PATH / ".env")


# 1. 원본 규칙
JSON_RULE_FILE = CURRENT_DIR / "category_rules.json"
# [추가] 검토 대기열 CSV 파일 경로
REVIEW_CSV_FILE = CURRENT_DIR / "review_queue.csv"
# 2. 원본 데이터 (매핑 전)
ORIGINAL_CSV_FILE = PROJECT_ROOT_PATH / "dataset" / "crawling" / "raw_crawling_db.csv"

# --- 출력 파일 ---
# 1. 최종 성공본 (매핑 + 생성 완료)
CRAWLING_CSV_FILE = PROJECT_ROOT_PATH / "dataset" / "crawling" / "crawling_db.csv"
# 2. 최종 실패본 (검토용)
UNMAPPED_CSV_FILE = PROJECT_ROOT_PATH / "dataset" / "crawling" / "unmapped_crawling.csv"

# --- [D. 배치 작업 전역 설정] ---
# Mapper가 FAISS 검색 시 고려할 상위 개수
TOP_T_FOR_FAISS = 5 


def main():
    """
    배치 처리 메인 파이프라인
    (1) Mapper 실행 -> (2) Generator 실행 -> (3) 결과 병합 및 저장
    """
    
    print("--- [ 1. 배치 작업 시작 ] ---")

    # --- 1-1. SBERT 모델 및 FAISS 엔진 초기화 (Mapper용) ---
    print("... [Mapper] SBERT 모델 및 FAISS 엔진 초기화 중 ...")
    DEVICE = mapper_keyword.get_device()
    SBERT_MODEL = mapper_keyword.load_model(mapper_keyword.MODEL_PATH, DEVICE)
    
    if SBERT_MODEL is None:
        print("❌ 치명적 오류: SBERT 모델 로드 실패. 배치를 중단합니다.")
        return

    try:
        with open(JSON_RULE_FILE, 'r', encoding='utf-8') as f:
            category_rules = json.load(f)
    except Exception as e:
        print(f"❌ 치명적 오류: 규칙 파일({JSON_RULE_FILE}) 로드 실패: {e}")
        return
        
    faiss_engine = mapper_keyword.FaissSearchEngine(model=SBERT_MODEL)
    category_metadata = mapper_keyword.get_category_metadata(category_rules)
    faiss_engine.build_index(category_metadata)
    print("✅ [Mapper] FAISS 인덱스 구축 완료.")

    # --- 1-2. OpenAI 클라이언트 확인 (Generator용) ---
    if generator_keyword.client is None:
        print("⚠️ [Generator] 경고: OpenAI 클라이언트가 초기화되지 않아 2단계가 비활성화될 수 있습니다.")
    else:
        print("✅ [Generator] OpenAI 클라이언트 확인 완료.")


    # --- 1-3. 원본 CSV 데이터 로드 ---
    try:
        original_df = pd.read_csv(ORIGINAL_CSV_FILE)
        # (중요) mapper/generator가 원본 인덱스를 추적할 수 있게 reset_index()
        original_df = original_df.reset_index().rename(columns={'index': '_original_index_'})
    except Exception as e:
        print(f"❌ 치명적 오류: 원본 CSV({ORIGINAL_CSV_FILE}) 로드 실패: {e}")
        return
        
    print(f"✅ 원본 데이터 로드 완료. (총 {len(original_df)}건)")


    # --- 2. [1단계: Mapper] 키워드 매핑 실행 ---
    print("\n--- [ 2. Mapper 실행 ] (Regex + FAISS) ---")
    
    # map_keywords_to_dataframe은 3개의 DF를 반환:
    # 1. final_db_df (전체 결과 - 지금은 안 씀)
    # 2. mapped_df (매핑 성공)
    # 3. unmapped_df (매핑 실패, _original_index_ 포함)
    _, mapped_df, unmapped_df = mapper_keyword.map_keywords_to_dataframe(
        original_df,
        str(JSON_RULE_FILE), # pathlib 객체를 문자열로 변환
        faiss_engine,
        T=TOP_T_FOR_FAISS
    )
    print(f"✅ [Mapper] 작업 완료.")
    print(f"  - 매핑 성공: {len(mapped_df)} 건")
    print(f"  - 매핑 실패: {len(unmapped_df)} 건")


    # --- 3. [2단계: Generator] 키워드 생성 실행 ---
    print("\n--- [ 3. Generator 실행 ] (OpenAI GPT) ---")
    
    # Mapper가 실패한 'unmapped_df'를 Generator에 전달
    generated_df = generator_keyword.generate_keywords_for_batch(unmapped_df)
    
    print(f"✅ [Generator] 작업 완료.")
    print(f"  - 신규 생성 성공: {len(generated_df)} 건")

    
    # --- 4. [3단계: 최종] 결과 병합 및 저장 ---
    print("\n--- [ 4. 최종 결과 저장 ] ---")

    # 4-1. 최종 성공본 (Mapper 성공 + Generator 성공)
    final_success_df = pd.concat([mapped_df, generated_df], ignore_index=True)
    
    # [ ✅ CSV 저장 로직 추가 ] ---
    if not generated_df.empty:
        try:
            # 1. 저장할 데이터만 추출
            # (prepare_text를 다시 쓰긴 비효율적이니, title만 저장)
            review_df = generated_df[['upper_keyword', 'lower_keyword', 'title']].copy()
            review_df['source'] = 'Batch' # 출처 추가
            # 텍스트 스니펫으로 컬럼명 변경 (API와 통일)
            review_df = review_df.rename(columns={'title': 'text_snippet'})
            
            # 2. 파일이 없으면 헤더 포함, 있으면 헤더 제외하고 이어붙이기
            needs_header = not os.path.exists(REVIEW_CSV_FILE)
            review_df.to_csv(
                REVIEW_CSV_FILE, 
                mode='a', 
                header=needs_header, 
                index=False,
                encoding='utf-8-sig',
                columns=["source", "upper_keyword", "lower_keyword", "text_snippet"] # 순서 고정
            )
            print(f"✅ [Batch-Run] 검토 목록({REVIEW_CSV_FILE.name})에 {len(review_df)}건 저장 완료.")
        except Exception as e:
            print(f"❌ [Batch-Run] 검토 목록 저장 실패: {e}")
    
    # _original_index_ 컬럼이 있다면 제거 (저장 시 불필요)
    if '_original_index_' in final_success_df.columns:
        final_success_df = final_success_df.drop(columns=['_original_index_'])

    final_success_df.to_csv(CRAWLING_CSV_FILE, index=False, encoding='utf-8-sig')
    print(f"✅ [성공] 총 {len(final_success_df)}건 저장 완료: {CRAWLING_CSV_FILE}")


    # 4-2. 최종 실패본 (Generator 마저 실패한 데이터)
    final_unmapped_df = pd.DataFrame() # 빈 DataFrame으로 시작
    
    if not unmapped_df.empty:
        if not generated_df.empty and '_original_index_' in generated_df.columns:
            # 생성에 성공한 데이터의 인덱스를 추출
            generated_indices = set(generated_df['_original_index_'])
            # Mapper 실패본(unmapped_df)에서 -> Generator 성공본(generated_indices)을 제외
            final_unmapped_df = unmapped_df[
                ~unmapped_df['_original_index_'].isin(generated_indices)
            ]
        else:
            # Generator가 아무것도 생성 못했으면, Mapper 실패본이 모두 최종 실패본
            final_unmapped_df = unmapped_df.copy()

    # _original_index_ 컬럼이 있다면 제거
    if '_original_index_' in final_unmapped_df.columns:
        final_unmapped_df = final_unmapped_df.drop(columns=['_original_index_'])
        
    final_unmapped_df.to_csv(UNMAPPED_CSV_FILE, index=False, encoding='utf-8-sig')
    print(f"✅ [실패] 총 {len(final_unmapped_df)}건 저장 완료: {UNMAPPED_CSV_FILE}")
    print("\n--- [ 🏁 모든 배치 작업 완료 ] ---")


# --- [E. 스크립트 실행] ---
# (터미널에서 'python3 run_batch_crawlingDB.py'로 실행)
if __name__ == "__main__":
    main()