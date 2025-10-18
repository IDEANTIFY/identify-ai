# app_chatbot/utils/history_manager.py

import os
from datetime import datetime
from typing import List, Dict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from .file_utils import save_json, find_latest_file, load_json, ensure_directory


HISTORY_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "chat_histories")


def save_history(user_name: str, messages: List[BaseMessage]) -> Optional[str]:
    '''
    대화 기록을 JSON 파일로 저장하는 함수
    '''
    ensure_directory(HISTORY_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(HISTORY_DIR, f"history_{user_name}_{timestamp}.json")
    # 대화 데이터 변환
    history_data = [
        {"role": msg.type, "content": msg.content}
        for msg in messages
    ]
    if save_json(file_path, history_data):
        return file_path
    return None


def load_history(user_name: str) -> Optional[List[BaseMessage]]:
    '''
    사용자의 가장 최신 대화 기록을 불러오는 함수
    '''
    latest_file = find_latest_file(HISTORY_DIR, prefix=f"history_{user_name}_")
    if not latest_file:
        return None
    data = load_json(latest_file)
    if not data:
        return None

    messages = []
    for item in data:
        role = item.get("role")
        content = item.get("content", "")

        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))
        elif role == "system":
            messages.append(SystemMessage(content=content))

    return messages
