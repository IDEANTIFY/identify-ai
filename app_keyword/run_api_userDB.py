import uvicorn  # API 서버 실행기 (pip install uvicorn)
from fastapi import FastAPI, HTTPException  # FastAPI 프레임워크 (pip install fastapi)
from pydantic import BaseModel  # 데이터 유효성 검사 (FastAPI에 포함됨)
from typing import List, Dict, Optional
import json
from pathlib import Path
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import csv  

import mapper_keyword
import generator_keyword

# --- [1. 경로 설정] ---
CURRENT_DIR = Path(__file__).parent 
PROJECT_ROOT_PATH = CURRENT_DIR.parent
load_dotenv(PROJECT_ROOT_PATH / ".env")
JSON_RULE_FILE = CURRENT_DIR / "category_rules.json"
# [추가] 검토 대기열 CSV 파일 경로
REVIEW_CSV_FILE = CURRENT_DIR / "review_queue.csv"

# --- [2. FastAPI 앱 및 전역 객체 로드] ---
# FastAPI 앱(app) 객체 생성
app = FastAPI(
    title="Keyword Processing API",
    description="Mapper와 Generator를 연동하여 키워드를 처리하는 API"
)

@app.on_event("startup")
def load_globals():
    """
    API 서버가 시작될 때 1회만 실행되는 함수.
    무거운 모델과 규칙 파일을 미리 메모리에 로드합니다.
    """
    print("--- [API 서버] SBERT 모델 및 규칙 파일 로딩 시작 ---")
    
    # 1. SBERT 모델 로드 (Mapper용)
    device = mapper_keyword.get_device()
    app.state.sbert_model = mapper_keyword.load_model(mapper_keyword.MODEL_PATH, device)
    if app.state.sbert_model:
        print("✅ [API 서버] SBERT 모델 로드 완료.")
    else:
        print("❌ 치명적 오류: SBERT 모델 로드 실패.")

    # 2. JSON 규칙 로드 (Mapper용)
    try:
        with open(JSON_RULE_FILE, 'r', encoding='utf-8') as f:
            app.state.rules_dict = json.load(f)
        print(f"✅ [API 서버] JSON 규칙 ({JSON_RULE_FILE}) 로드 완료.")
    except Exception as e:
        print(f"❌ 치명적 오류: JSON 규칙 파일 로드 실패: {e}")
        app.state.rules_dict = None
        
    # 3. Generator 클라이언트 확인 (Generator 모듈 임포트 시 자동 초기화됨)
    if generator_keyword.client:
        print("✅ [API 서버] Generator (OpenAI) 클라이언트 확인 완료.")
    else:
        print("⚠️ [API 서버] 경고: Generator의 OpenAI 클라이언트가 로드되지 않았습니다.")


# --- [3. API 데이터 모델 정의 (Pydantic)] ---
class KeywordRequest(BaseModel):
    text: str
    upper_keywords: List[str]  # [수정됨] 1~4개의 상위 키워드 리스트
    k: int = 3                 # 반환받을 최대 하위 키워드 개수

class KeywordResponse(BaseModel):
    status: str                # "mapped", "generated", "failed", "error"
    request_text: str          # 요청받은 원본 텍스트
    upper_keywords: List[str]  # 요청받은 상위 키워드
    lower_keywords: List[str]  # 최종적으로 찾은 하위 키워드


# --- [4. API 엔드포인트 정의] ---
@app.get("/")
def read_root():
    """ 서버가 살아있는지 확인하는 기본 엔드포인트 """
    return {"message": "Keyword API (Mapper/Generator) is running."}


@app.post("/process-keywords", response_model=KeywordResponse)
async def process_keywords_endpoint(request: KeywordRequest):
    """
    [메인 API] 텍스트와 상위 키워드를 받아 매핑 또는 생성을 수행합니다.
    """
    
    # 0. 서버 시작 시 모델 로드에 실패했는지 확인
    if not app.state.sbert_model or not app.state.rules_dict:
        raise HTTPException(status_code=503, detail="서버가 준비되지 않았습니다. (모델/규칙 로드 실패)")

    print(f"\n--- [API 요청] Uppers: {request.upper_keywords}, Text: {request.text[:20]}... ---")

    # --- 1단계: Mapper 호출 ---
    mapped_lowers = mapper_keyword.find_best_lower_keywords_top_k(
        text=request.text,
        given_upper_keywords=request.upper_keywords, 
        rules_dict=app.state.rules_dict,
        model=app.state.sbert_model,
        k=request.k
    )

    # --- 2단계: Mapper 결과 확인 ---
    if mapped_lowers:
        print("✅ [API] Mapper 매핑 성공.")
        return KeywordResponse(
            status="mapped",
            request_text=request.text,
            upper_keywords=request.upper_keywords,
            lower_keywords=mapped_lowers
        )

    # --- 3단계: Generator 호출 (Mapper가 실패했을 경우) ---
    print("ℹ️ [API] Mapper 매핑 실패. Generator 호출...")
    
    # (주의) generator의 OpenAI client가 로드 안됐을 경우
    if not generator_keyword.client:
        print("❌ [API] Generator 클라이언트가 없어 생성 실패.")
        return KeywordResponse(
            status="failed",
            request_text=request.text,
            upper_keywords=request.upper_keywords,
            lower_keywords=[]
        )
        
    generated_lowers = generator_keyword.generate_keywords_for_api(
        text=request.text,
        given_upper_keywords=request.upper_keywords, 
        k=request.k
    )

    # --- 4단계: Generator 결과 확인 ---
    if generated_lowers:
        print("✅ [API] Generator 생성 성공.")
        
        # [ ✅ CSV 저장 로직 추가 (TODO 위치) ] ---
        try:
            # 1. 파일이 없는 경우 헤더(header)를 추가
            needs_header = not os.path.exists(REVIEW_CSV_FILE)
            
            with open(REVIEW_CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if needs_header:
                    # CSV 파일의 첫 줄(헤더)을 정의
                    writer.writerow(["source", "upper_keyword", "lower_keyword", "text_snippet"])
                
                # 2. CSV 파일에 데이터 한 줄 추가
                writer.writerow([
                    "API",                                 # 출처
                    ', '.join(request.upper_keywords),     # 상위 키워드 (예: "AI, IT")
                    ', '.join(generated_lowers),           # 생성된 하위 키워드 (예: "딥러닝")
                    request.text[:100] + "..."             # 텍스트 일부
                ])
            print(f"   -> [검토 대기] {REVIEW_CSV_FILE.name}에 저장 완료.")
        except Exception as e:
            print(f"   ❌ [검토 대기] CSV 저장 실패: {e}")

        return KeywordResponse(
            status="generated",
            request_text=request.text,
            upper_keywords=request.upper_keywords,
            lower_keywords=generated_lowers
        )
    else:
        print("❌ [API] Generator 생성 실패.")
        return KeywordResponse(
            status="failed",
            request_text=request.text,
            upper_keywords=request.upper_keywords,
            lower_keywords=[]
        )


# --- [5. API 서버 실행] ---
if __name__ == "__main__":
    print("--- [API 서버] Uvicorn으로 실행합니다 (http://127.0.0.1:8000) ---")
    uvicorn.run(
        "run_api_userDB:app", # "파이썬파일명:FastAPI객체이름"
        host="127.0.0.1", 
        port=8000, 
        reload=True # 개발용 (코드 변경 시 자동 재시작)
    )