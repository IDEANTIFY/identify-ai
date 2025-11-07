# app_chatbot/utils/title_manager.py

import os
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from .file_utils import chat_history_path, load_json, save_json


def is_first_chat(user_id: str, chat_id: str) -> bool:
    '''
    해당 대화가 첫 대화인지 확인하는 함수
    chat_histories/{user_id}/{chat_id}.json 파일이 없으면 첫 대화
    '''
    path = chat_history_path(user_id, chat_id)
    return not os.path.exists(path)


def generate_chat_title(first_user_input: str, first_ai_response: str, openai_api_key: str) -> str:
    '''
    첫 대화 내용을 기반으로 제목을 생성하는 함수
    GPT-5-nano를 사용하여 제목 생성
    '''
    llm = ChatOpenAI(model_name="gpt-5-nano", api_key=openai_api_key)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "사용자와 AI의 첫 대화를 읽고, 이 대화의 주제를 간결하게 요약한, 오글거리는 느낌의 제목을 생성하세요. 오글거리는 느낌의 제목은 20자 이내의 한글로 작성하세요. 제목만 출력하세요."),
        ("human", "사용자: {user_input}\n\nAI: {ai_response}")
    ])
    
    try:
        response = llm.invoke(prompt.format_messages(
            user_input=first_user_input,
            ai_response=first_ai_response
        ))
        title = (response.content or "").strip()
        # 따옴표나 불필요한 문자 제거
        title = title.strip('"').strip("'").strip()
        return title if title else "대화"
    except Exception as e:
        print(f"제목 생성 중 오류 발생: {e}")
        return "대화"


def create_chat_title_if_first(
    user_id: str, 
    chat_id: str, 
    first_user_input: str, 
    first_ai_response: str, 
    openai_api_key: str
) -> Optional[str]:
    '''
    첫 대화인 경우 제목을 생성하고 chat_histories에 추가하는 함수
    
    Args:
        user_id: 사용자 ID
        chat_id: 채팅 ID
        first_user_input: 첫 사용자 입력
        first_ai_response: 첫 AI 응답
        openai_api_key: OpenAI API 키
    
    Returns:
        생성된 제목 (첫 대화인 경우), None (첫 대화가 아닌 경우)
    '''
    if not is_first_chat(user_id, chat_id):
        return None
    
    # 제목 생성
    title = generate_chat_title(first_user_input, first_ai_response, openai_api_key)
    
    # 파일 경로
    path = chat_history_path(user_id, chat_id)
    
    # 첫 대화 턴 저장 (title 포함)
    data = {
        "title": title,
        "turns": [
            {
                "human": first_user_input,
                "ai": first_ai_response
            }
        ]
    }
    
    save_json(path, data)
    return title