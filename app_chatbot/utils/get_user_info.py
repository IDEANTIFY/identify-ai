# app_chatbot/utils/get_user_info.py

import httpx
from fastapi import APIRouter
from app_chatbot.utils.file_utils import save_user_info

router = APIRouter()

BACKEND_USER_API = "http://백엔드주소/"  # <--- 수정 필요

@router.post("/fetch/{user_id}")
async def fetch_and_save_user(user_id: str):
    """
    1) 백엔드에서 GET 요청으로 user info 가져옴
    2) dataset/user_info/{user_id}.json 에 저장
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BACKEND_USER_API}/{user_id}")
    if response.status_code != 200:
        return {"error": "백엔드 요청 실패", "status_code": response.status_code}
    user_info = response.json()
  
    # user_info 저장
    saved = save_user_info(user_id, user_info)
    if saved:
        return {"message": "User info fetched & saved", "user_id": user_id}
    else:
        return {"error": "JSON 저장 실패"}

