# app_chatbot/utils/history_manager.py

import os
import json
from typing import List, Dict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from .file_utils import save_json, load_json, chat_history_path

MAX_TURNS = 10   # <-- message가 아니라 턴 기준 (인간 + ai 가 한 턴)


def save_history(user_id: str, chat_id: str, messages: List[BaseMessage]) -> Optional[str]:
    """
    메시지를 턴 단위(human + ai)로 저장
    저장 구조: -> 변경했습니다! 
    [
        {"human": "hi", "ai": "hello"},
        {"human": "what is idea", "ai": "here you go"}
    ]
    """
    path = chat_history_path(user_id, chat_id)

    # 기존 데이터 읽기
    existing = load_json(path) or []   # list[ {human, ai} ]
    # messages는 [system?, human, ai, human, ai ...]
    turns: List[Dict[str, str]] = []

    last_human: Optional[str] = None

    for msg in messages:
        if isinstance(msg, HumanMessage):
            last_human = msg.content
        elif isinstance(msg, AIMessage) and last_human is not None:
            turns.append({"human": last_human, "ai": msg.content})
            last_human = None

    # 기존 + 새로운 턴 병합
    existing += turns

    # 최근 MAX_TURNS 개만 유지
    if len(existing) > MAX_TURNS:
        existing = existing[-MAX_TURNS:]

    success = save_json(path, existing)
    return path if success else None



def load_history(user_id: str, chat_id: str) -> Optional[List[BaseMessage]]:
    """
    저장된 턴을 불러와 다시 BaseMessage 목록으로 풀어서 반환
    """
    path = chat_history_path(user_id, chat_id)
    data = load_json(path)

    if not data or not isinstance(data, list):
        return None

    messages: List[BaseMessage] = []

    for turn in data:
        messages.append(HumanMessage(content=turn.get("human", "")))
        messages.append(AIMessage(content=turn.get("ai", "")))

    return messages
