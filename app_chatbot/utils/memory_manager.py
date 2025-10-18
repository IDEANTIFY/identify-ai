# app_chatbot/utils/memory_manager.py

from typing import Optional, List
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import BaseMessage, SystemMessage

from .history_manager import load_history


def create_memory(system_prompt: Optional[str] = None):
    '''
    새로운 대화 메모리를 생성하는 함수
    '''
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    if system_prompt:
        memory.chat_memory.add_message(SystemMessage(content=system_prompt))

    return memory


def load_memory(user_name: str, system_prompt: Optional[str] = None):
    '''
    기존 사용자 메모리를 불러오거나 새로 생성하는 함수
    '''
    history_messages: Optional[List[BaseMessage]] = load_history(user_name)

    # 대화 메모리 초기화
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)

    # 기존 기록이 있을 경우 메모리에 추가
    if history_messages:
        for msg in history_messages:
            memory.chat_memory.add_message(msg)
    else:
        # 기존 기록이 없으면 시스템 프롬프트 추가
        if system_prompt:
            memory.chat_memory.add_message(SystemMessage(content=system_prompt))

    return memory
