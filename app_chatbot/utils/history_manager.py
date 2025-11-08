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
    저장 구조: 
    - 첫 대화: {"title": "...", "turns": [{"human": "...", "ai": "..."}]}
    - 기존 대화: {"title": "...", "turns": [...]} 또는 [{"human": "...", "ai": "..."}]
    """
    path = chat_history_path(user_id, chat_id)
    
    # 기존 데이터 읽기
    existing_data = load_json(path)
    
    # 기존 데이터 구조 확인
    if existing_data is None:
        # 파일이 없으면 빈 리스트
        existing_turns = []
        existing_title = None
    elif isinstance(existing_data, dict) and "turns" in existing_data:
        # title 구조인 경우
        existing_turns = existing_data.get("turns", [])
        existing_title = existing_data.get("title")
    elif isinstance(existing_data, list):
        # 기존 리스트 구조 (하위 호환)
        existing_turns = existing_data
        existing_title = None
    else:
        existing_turns = []
        existing_title = None
    
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
    existing_turns += turns
    
    # 최근 MAX_TURNS 개만 유지
    if len(existing_turns) > MAX_TURNS:
        existing_turns = existing_turns[-MAX_TURNS:]
    
    # 저장 구조 결정
    if existing_title:
        # title이 있으면 title 구조로 저장
        data = {
            "title": existing_title,
            "turns": existing_turns
        }
    else:
        # title이 없으면 기존 리스트 구조로 저장 (하위 호환)
        data = existing_turns
    
    success = save_json(path, data)
    return path if success else None


def load_history(user_id: str, chat_id: str) -> Optional[List[BaseMessage]]:
    """
    저장된 턴을 불러와 다시 BaseMessage 목록으로 풀어서 반환
    """
    path = chat_history_path(user_id, chat_id)
    data = load_json(path)
    
    if not data:
        return None
    
    # 구조 확인
    if isinstance(data, dict) and "turns" in data:
        turns = data.get("turns", [])
    elif isinstance(data, list):
        turns = data
    else:
        return None
    
    if not isinstance(turns, list):
        return None
    
    messages: List[BaseMessage] = []
    
    for turn in turns:
        if isinstance(turn, dict):
            messages.append(HumanMessage(content=turn.get("human", "")))
            messages.append(AIMessage(content=turn.get("ai", "")))
    
    return messages
