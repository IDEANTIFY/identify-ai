# app_validator/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from .pipeline import execute_full_pipeline

app = FastAPI(title="Idea Validator API")

# Input 모델 정의 (구조화된 아이디어 형태)
class StructuredIdea(BaseModel):
    주요_내용: str
    도메인: str
    목적: str
    차별성: str
    핵심_기술: str
    서비스_대상: str

@app.post("/validate-idea/")
async def validate_idea_endpoint(request: StructuredIdea):
    """구조화된 아이디어(JSON)를 받아 검증 후 최종 리포트를 반환합니다."""
    # Pydantic 모델을 일반 딕셔너리로 변환하여 전달
    structured_idea_dict = request.model_dump()
    final_report = execute_full_pipeline(structured_idea_dict)
    return final_report