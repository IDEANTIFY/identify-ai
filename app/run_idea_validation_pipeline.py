import os
import sys
import json
import torch
from sentence_transformers import SentenceTransformer
import concurrent.futures
import time

# --- ⚙️ 1. 모듈 임포트 ---
# 각 기능별로 분리된 Python 파일에서 필요한 함수와 클래스를 가져옵니다.
from .convert_idea_to_query import *
from .web_search_utils import *
from .db_search_utils import *
from .create_report import *

# --- ✅ 2. 설정 및 전역 객체 초기화 ---

# 파일 및 모델 경로 (사용자 환경에 맞게 수정)
MODEL_FOLDER_PATH = "models/ko-sroberta-multitask-local"
INDEX_FILE = "dataset/crawling_total.index"
META_FILE = "dataset/crawling_total.pkl"

def initialize_components():
    """파이프라인에 필요한 모델과 검색 엔진을 초기화합니다."""
    print("=" * 60, flush = True)
    print("🚀 파이프라인 초기화를 시작합니다...", flush = True)
    
    # Sentence Transformer 모델 로딩 (메모리에 한 번만)
    if torch.cuda.is_available():      # NVIDIA CUDA GPU 확인
        device = "cuda"
    elif torch.backends.mps.is_available():  # Apple Metal GPU (MPS) 확인
        device = "mps"
    else:                               # 둘 다 없으면 CPU 사용
        device = "cpu"
    print(f"Sentence Transformer 모델을 로딩합니다... (Device: {device})", flush = True)
    model = SentenceTransformer(MODEL_FOLDER_PATH, device=device)
    
    # 내부 DB 검색 엔진(FAISS) 로딩
    print("내부 DB 인덱스를 로딩합니다...", flush = True)
    db_search_engine = FaissSearchEngine(model_path=MODEL_FOLDER_PATH)
    if not (os.path.exists(INDEX_FILE) and os.path.exists(META_FILE)):
        print(f"❌ [오류] DB 인덱스 파일({INDEX_FILE}) 또는 메타 파일({META_FILE})이 없습니다.", file=sys.stderr, flush = True)
        sys.exit(1)
    db_search_engine.load_index(index_path=INDEX_FILE, meta_path=META_FILE)
    
    print("✅ 모든 컴포넌트 초기화 완료.", flush = True)
    print("=" * 60, flush = True)
    
    return model, db_search_engine

def execute_full_pipeline(user_idea: str) -> dict:
    """
    사용자 아이디어를 입력받아 전체 검증 파이프라인을 실행하고,
    요약 및 상세 보고서를 딕셔너리 형태로 반환합니다.

    Args:
        user_idea (str): 검증을 원하는 사용자의 아이디어 텍스트.

    Returns:
        dict: 'summary_report'와 'detailed_report'를 포함하는 딕셔너리.
    """
    
    start_time = time.time()
    
    # 서버 로그: 파이프라인 시작 알림
    print(f"🚀 파이프라인 시작: \"{user_idea}\"", flush=True)

    # 1. 컴포넌트 초기화
    # 참고: 실제 서비스에서는 API 서버가 시작될 때 한 번만 초기화하는 것이 성능에 유리합니다.
    model, db_search_engine = initialize_components()

    # 2. 아이디어 -> 검색 쿼리 변환
    print("\n[단계 1/4] 아이디어를 핵심 검색 쿼리로 변환 중...", flush=True)
    structured_data = extract_structured_idea_info(user_idea)
    search_query = generate_search_query(structured_data)
    print(f"  🔍 변환된 검색 쿼리: \"{search_query}\"", flush=True)
    
    # 3. 정보 검색 (웹 & DB 병렬 처리)
    print("\n[단계 2/4] 웹 및 내부 DB에서 유사 사례를 병렬로 검색 중...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # 웹 검색 태스크 제출
        future_web_search = executor.submit(run_web_search_pipeline, search_query, model)
        # DB 검색 태스크 제출
        future_db_search = executor.submit(db_search_engine.search, search_query, 5)
        
        # 결과 취합
        web_search_df = future_web_search.result()
        db_search_results = future_db_search.result()
    
    print(f"  - 웹 검색 완료: {len(web_search_df)}건의 관련 문서 발견", flush=True)
    print(f"  - DB 검색 완료: {len(db_search_results)}건의 관련 문서 발견", flush=True)
    
    # 4. RAG를 위한 데이터 준비
    top_web_docs = web_search_df.head(3).to_dict('records')
    approx_similar_count = len(web_search_df)

    # 5. 리포트 생성 (요약 & 상세 병렬 처리)
    print("\n[단계 3/4] 검색된 정보를 바탕으로 RAG 리포트를 병렬로 생성 중...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # 요약 리포트 생성 태스크 제출
        future_summary = executor.submit(
            generate_summary_report,
            user_idea, top_web_docs, db_search_results, approx_similar_count
        )
        # 상세 리포트 생성 태스크 제출
        future_detailed = executor.submit(
            generate_detailed_sources_report,
            user_idea, top_web_docs, db_search_results
        )
        
        # 결과 취합
        summary_report = future_summary.result()
        detailed_report = future_detailed.result()
        
    print("  - 정량적 요약 보고서 생성 완료.", flush=True)
    print("  - 상세 소스 분석 보고서 생성 완료.", flush=True)
    
    # 6. 최종 결과 취합 및 반환
    print("\n[단계 4/4] 최종 결과 취합 완료.", flush=True)
    final_result = {
        "summary_report": summary_report,
        "detailed_report": detailed_report
    }
    
    end_time = time.time()
    print(f"🎉 파이프라인 실행 완료! (총 소요 시간: {end_time - start_time:.2f}초)", flush=True)
    
    return final_result

# --- 테스트를 위한 실행 블록 (실제 서버에서는 호출되지 않음) ---
# if __name__ == '__main__':
    # test_idea = "인공지능 튜터와 실시간으로 대화하며 배우는 맞춤형 외국어 학습 애플리케이션"
    
    # 수정된 함수 호출
    # reports = execute_full_pipeline(test_idea)
    
    # 반환된 결과 확인
    # print("\n" + "="*80)
    # print("✅ 함수가 반환한 최종 결과:")
    # print("-" * 60)
    # print("📊 [요약 보고서]")
    # print(json.dumps(reports['summary_report'], indent=2, ensure_ascii=False))
    # print("-" * 60)
    # print("📑 [상세 보고서]")
    # print(json.dumps(reports['detailed_report'], indent=2, ensure_ascii=False))
    # print("="*80)
