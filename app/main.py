from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

# 리팩토링된 파이프라인 함수를 임포트
from .run_idea_validation_pipeline import execute_full_pipeline

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 인스턴스 생성
app = FastAPI(
    title="Idea Validation API",
    description="사용자의 아이디어를 입력받아 웹/DB 검색을 통해 유효성을 검증하고 리포트를 생성하는 API입니다.",
    version="1.0.0"
)

# 입력 데이터 모델 정의 (Request Body)
class IdeaRequest(BaseModel):
    idea_text: str

# 루트 엔드포인트
@app.get("/")
def read_root():
    return {"message": "Idea Validation API에 오신 것을 환영합니다."}

# 아이디어 검증 메인 엔드포인트
@app.post("/validate-idea/")
async def validate_idea(request: IdeaRequest):
    """
    사용자 아이디어를 받아 전체 검증 파이프라인을 실행합니다.
    - **idea_text**: 검증하고자 하는 아이디어 (문자열)
    """
    try:
        logger.info(f"아이디어 검증 요청 수신: {request.idea_text}")
        # 파이프라인 실행
        result = execute_full_pipeline(request.idea_text)
        logger.info("파이프라인 실행 완료.")
        return result
    except Exception as e:
        logger.error(f"파이프라인 실행 중 오류 발생: {e}", exc_info=True)
        # 클라이언트에게는 상세 오류를 숨기고 일반적인 오류 메시지를 반환
        raise HTTPException(status_code=500, detail="아이디어 검증 중 서버 내부 오류가 발생했습니다.")