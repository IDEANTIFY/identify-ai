from fastapi import FastAPI
from app_issue.issue_web_search import router as issue_router
from app_chatbot.utils.get_user_info import router as user_router

app = FastAPI(title="AI Server")

# prefix 없이 그대로 등록 (원래 경로 유지)
app.include_router(issue_router)
app.include_router(user_router)

@app.get("/")
def root():
    return {"message": "AI Server is running"}