# app_chatbot/utils/get_user_info.py

from fastapi import APIRouter
from pydantic import BaseModel
from app_chatbot.utils.file_utils import save_user_info

router = APIRouter()

# ===============================
# FastAPI 엔드포인트
# ===============================

class UserInfoRequest(BaseModel):
    userId: str
    nickname: str
    keywords: list[str]

@router.post("/user/info")
async def receive_user_info(req: UserInfoRequest):
    """
    서버(Spring Boot)로부터 유저 정보를 수신하고 로컬에 저장
    - 저장 경로: dataset/user_info/{userId}.json
    """
    try:
        # JSON 데이터 구성
        user_data = {
            "user_id": req.userId,
            "nickname": req.nickname,
            "keywords": req.keywords
        }

        # 로컬 JSON 저장
        saved = save_user_info(req.userId, user_data)

        if saved:
            return {"message": "유저 정보 저장 성공", "user_id": req.userId}
        else:
            return {"error": "JSON 저장 실패"}

    except Exception as e:
        return {"error": f"요청 처리 중 오류 발생: {e}"}
