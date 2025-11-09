# app_structurizer/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from .structurizer_utils import extract_structured_idea_info

app = FastAPI(title="Idea Structurizer API")

# Input 모델 정의
class IdeaText(BaseModel):
    query: str

# Output 모델 정의 (API 문서화를 위해)
class StructuredIdea(BaseModel):
    summary: str
    purpose: str
    differentiation: str
    technology: str
    target: str

@app.post("/structure-idea/", response_model=StructuredIdea)
async def structure_idea_endpoint(request: IdeaText):
    """사용자의 아이디어 텍스트를 받아 구조화된 정보(JSON)를 반환합니다."""
    structured_data = extract_structured_idea_info(request.idea_text)
    return structured_data