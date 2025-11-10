import os
import sys
import json
import torch
from sentence_transformers import SentenceTransformer
import concurrent.futures
import time
from datetime import datetime
from pathlib import Path

# --- ⚙️ 1. 모듈 임포트 ---
# 각 기능별로 분리된 Python 파일에서 필요한 함수와 클래스를 가져옵니다.
from utils.convert_idea_to_query import *
from utils.web_search_utils import *
from utils.db_search_utils import *
from utils.crawling_db_search_utils import CrawlingdbSearchEngine
from utils.user_db_search_utils import UserdbFaissSearchEngine
from utils.create_report import *
from utils.sqs_server import *

# --- ✅ 2. 설정 및 전역 객체 초기화 ---

# 현재 실행 파일 기준으로 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent

# 파일 및 모델 경로 (사용자 환경에 맞게 수정)
# 환경 변수에서 모델 경로를 가져오거나 기본값 사용
ENV_MODEL_PATH = os.environ.get("MODEL_FOLDER_PATH") or os.environ.get("EMBED_MODEL")

if ENV_MODEL_PATH:
    # 환경 변수로 지정된 모델 경로 사용
    MODEL_FOLDER_PATH = ENV_MODEL_PATH
    print(f"✅ 환경 변수에서 모델 경로 로드: {MODEL_FOLDER_PATH}", flush=True)
else:
    # 로컬 모델 경로 (프로젝트 루트 기준)
    LOCAL_MODEL_PATH = PROJECT_ROOT / "models" / "ko-sroberta-multitask-local"
    # Hugging Face 모델 ID (로컬 모델이 없을 경우 사용)
    HF_MODEL_ID = "jhgan/ko-sroberta-multitask"
    
    # 로컬 모델이 있으면 사용, 없으면 Hugging Face 모델 사용
    if LOCAL_MODEL_PATH.exists() and (LOCAL_MODEL_PATH / "config.json").exists():
        MODEL_FOLDER_PATH = str(LOCAL_MODEL_PATH)
        print(f"✅ 로컬 모델 사용: {MODEL_FOLDER_PATH}", flush=True)
    else:
        MODEL_FOLDER_PATH = HF_MODEL_ID
        print(f"⚠️ 로컬 모델을 찾을 수 없어 Hugging Face 모델 사용: {MODEL_FOLDER_PATH}", flush=True)
        print(f"   로컬 모델 경로 확인: {LOCAL_MODEL_PATH}", flush=True)

# 크롤링 DB 파일 경로
CRAWLING_INDEX_FILE = PROJECT_ROOT / "dataset/crawling/crawling_total.index"
CRAWLING_META_FILE = PROJECT_ROOT / "dataset/crawling/crawling_total.pkl"
# 기존 DB 파일 경로 (사용 안 함)
INDEX_FILE = PROJECT_ROOT / "dataset/crawling/crawling_total.index"
META_FILE = PROJECT_ROOT / "dataset/crawling/crawling_total.pkl"
REPORTS_OUTPUT_DIR = PROJECT_ROOT / "dataset/reports"
STRUCTURED_IDEA_FILE = PROJECT_ROOT / "dataset/structured_idea.json"

def initialize_components():
    if torch.cuda.is_available():      # NVIDIA CUDA GPU 확인
        device = "cuda"
    elif torch.backends.mps.is_available():  # Apple Metal GPU (MPS) 확인
        device = "mps"
    else:                               # 둘 다 없으면 CPU 사용
        device = "cpu"

    # Sentence Transformer 모델 로딩 (메모리에 한 번만)
    print(f"Sentence Transformer 모델을 로딩합니다... (Device: {device})", flush = True)
    model = SentenceTransformer(MODEL_FOLDER_PATH, device=device)
    
    # 크롤링 DB 검색 엔진 초기화 및 로딩
    print("크롤링 DB 검색 엔진을 초기화합니다...", flush = True)
    crawling_db_search_engine = CrawlingdbSearchEngine(model_path=MODEL_FOLDER_PATH, use_reranker=False)
    if not (CRAWLING_INDEX_FILE.exists() and CRAWLING_META_FILE.exists()):
        print(f"❌ [오류] 크롤링 DB 인덱스 파일({CRAWLING_INDEX_FILE}) 또는 메타 파일({CRAWLING_META_FILE})이 없습니다.", file=sys.stderr, flush = True)
        sys.exit(1)
    crawling_db_search_engine.load_index(index_path=str(CRAWLING_INDEX_FILE), meta_path=str(CRAWLING_META_FILE))
    
    # User DB 검색 엔진 초기화 및 인덱스 구축
    print("User DB 검색 엔진을 초기화합니다...", flush = True)
    user_db_search_engine = UserdbFaissSearchEngine(model_path=MODEL_FOLDER_PATH)
    user_db_search_engine.update_and_load_index()
    
    print("✅ 모든 컴포넌트 초기화 완료.", flush = True)
    print("=" * 60, flush = True)
    
    return model, crawling_db_search_engine, user_db_search_engine

def execute_full_pipeline(structured_idea: dict) -> dict:
    """
    사용자 아이디어를 입력받아 전체 검증 파이프라인을 실행하고,
    요약 및 상세 보고서를 딕셔너리 형태로 반환합니다.

    Args:
        user_idea (str): 검증을 원하는 사용자의 아이디어 구조.

    Returns:
        dict: 'summary_report'와 'detailed_report'를 포함하는 딕셔너리.
    """
    
    start_time = time.time()

    # 컴포넌트 초기화
    model, crawling_db_search_engine, user_db_search_engine = initialize_components()

    # 아이디어 -> 검색 쿼리 변환
    print("\n[단계 1/4] 아이디어를 핵심 검색 쿼리로 변환 중...", flush=True)
    search_query = generate_search_query(structured_idea)
    print(f"  🔍 변환된 검색 쿼리: \"{search_query}\"", flush=True)
    
    # 정보 검색 (웹 & DB 병렬 처리)
    print("\n[단계 2/4] 웹 및 내부 DB에서 유사 사례를 병렬로 검색 중...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # 웹 검색 태스크 제출
        future_web_search = executor.submit(run_web_search_pipeline, search_query)
        # 크롤링 DB 검색 태스크 제출 (CrawlingdbSearchEngine 사용)
        future_crawling_db_search = executor.submit(crawling_db_search_engine.search, search_query, 5)
        # User DB 검색 태스크 제출
        future_user_db_search = executor.submit(user_db_search_engine.search, search_query, 5)
        
        # 결과 취합
        web_search_df = future_web_search.result()
        crawling_db_results = future_crawling_db_search.result()
        try:
            user_db_results = future_user_db_search.result()
        except RuntimeError as e:
            # User DB 인덱스가 없거나 데이터가 없는 경우
            print(f"  ⚠️ User DB 검색 실패: {e}", flush=True)
            user_db_results = []
    
    print(f"  - 웹 검색 완료: {len(web_search_df)}건의 관련 문서 발견", flush=True)
    print(f"  - 크롤링 DB 검색 완료: {len(crawling_db_results)}건의 관련 문서 발견", flush=True)
    print(f"  - User DB 검색 완료: {len(user_db_results)}건의 관련 문서 발견", flush=True)
    
    # RAG를 위한 데이터 준비
    top_web_docs = web_search_df.head(3).to_dict('records')
    approx_similar_count = len(web_search_df)

    # 리포트 생성 (요약 & 상세 병렬 처리)
    print("\n[단계 3/4] 검색된 정보를 바탕으로 RAG 리포트를 병렬로 생성 중...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # 요약 리포트 생성 태스크 제출
        # query는 structured_idea (dict)를 전달해야 함
        future_summary = executor.submit(
            generate_summary_report,
            structured_idea, top_web_docs, crawling_db_results, user_db_results, approx_similar_count
        )
        # 상세 리포트 생성 태스크 제출
        future_detailed = executor.submit(
            generate_detailed_sources_report,
            structured_idea, top_web_docs, crawling_db_results, user_db_results
        )
        
        # 결과 취합
        summary_report = future_summary.result()
        detailed_report = future_detailed.result()
        
    print("  - 정량적 요약 보고서 생성 완료.", flush=True)
    print("  - 상세 소스 분석 보고서 생성 완료.", flush=True)
    
    # 최종 결과 취합 및 반환
    print("\n[단계 4/4] 최종 결과 취합 완료.", flush=True)

    final_report = {
        "summary_report": summary_report,
        "detailed_report": detailed_report
    }
    # 리포트 저장 폴더 생성 (없을 경우)
    REPORTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 고유 파일명을 위한 타임스탬프 생성
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 요약 리포트 파일로 저장
    report_filepath = REPORTS_OUTPUT_DIR / f"report_total_{timestamp}.json"
    with open(report_filepath, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=4)
    print(f"  - ✅ 요약 리포트 저장 완료: {report_filepath}", flush=True)
    
    end_time = time.time()
    print(f"\n🎉 파이프라인 실행 완료! (총 소요 시간: {end_time - start_time:.2f}초)", flush=True)
    
    return final_report

if __name__ == '__main__':
    load_dotenv()

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("OPENAI_API_KEY 미설정")
        raise SystemExit(1)
    
    # SQS 요청큐, 응답큐, 리전 정보 불러오기
    request_queue_url = os.getenv("AWS_SQS_IDEA_REPORT_REQUEST_QUEUE")
    response_queue_url = os.getenv("AWS_SQS_IDEA_REPORT_RESPONSE_QUEUE")
    region_name = os.getenv("AWS_DEFAULT_REGION")

    print(f"✅ 환경변수 로드 완료:")
    print(f"   - Request Queue: {request_queue_url}")
    print(f"   - Response Queue: {response_queue_url}")
    print(f"   - Region: {region_name}")

    # SQS 리소스 및 큐 객체 생성
    sqs = get_sqs_resource(region_name)
    request_queue = get_queue(sqs, request_queue_url)
    response_queue = get_queue(sqs, response_queue_url)

    print("\n🔄 SQS 메시지 처리 루프 시작...\n")

    processed_count = 0

    # 메시지 처리 루프
    while has_messages_in_queue(request_queue):
        try:
            processed_count += 1
            print(f"\n{'='*60}")
            print(f"📨 메시지 #{processed_count} 처리 중...")
            print(f"{'='*60}\n")
            
            # SQS로부터 데이터 가져오기
            body, message = receive_message_from_sqs(request_queue)
            if not body:
                print("⚠️ 메시지 없음 또는 파싱 실패, 다음 메시지로 이동")
                continue

            # 필수 필드 확인
            required_fields = ["query", "summary", "purpose", "differentiation", "technology", "target"]
            missing_fields = [field for field in required_fields if field not in body]
            if missing_fields:
                raise RuntimeError(f"[오류] 메시지에 필수 필드가 없습니다: {', '.join(missing_fields)}")
            
            print(f"SQS에서 아이디어 데이터 로드 완료: {request_queue}", flush=True)
            print(f"   - Query: {body.get('query')}", flush=True)
                
            # 파이프라인을 실행하여 리포트 파일 생성
            print(f"\n🚀 파이프라인 실행 시작: {body.get('query')}")
            result = execute_full_pipeline(body)
            
            # SQS에 데이터 전송
            print(f"\n📤 결과를 SQS 응답 큐로 전송 중...")
            message_id = send_message_to_sqs(response_queue, result)
            
            if message_id:
                print(f"✅ 메시지 #{processed_count} 처리 완료!")
                print(f"   MessageId: {message_id}")
            else:
                print(f"❌ 메시지 #{processed_count} 전송 실패!")
                print(f"   결과는 로컬 파일로만 저장되었습니다.")
                # 실패한 경우에도 계속 진행할지, 중단할지 결정
                # raise Exception("SQS 전송 실패")
                
        except Exception as e:
            print(f"\n❌ 메시지 #{processed_count} 처리 중 오류 발생:")
            print(f"   {type(e).__name__}: {e}")
            import traceback
            print(traceback.format_exc())
            # 오류 발생 시 다음 메시지로 계속 진행
            continue

        # 과도한 API 호출 방지를 위한 잠시 대기
        time.sleep(1)
    
    print(f"\n{'='*60}")
    print(f"✅ 전체 처리 완료! 총 {processed_count}개 메시지 처리")
    print(f"{'='*60}\n")