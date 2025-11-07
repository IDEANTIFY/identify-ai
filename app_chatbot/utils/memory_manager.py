# app_chatbot/utils/memory_manager.py

from typing import Optional, List
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import BaseMessage, SystemMessage

from .history_manager import load_history, MAX_TURNS

def create_memory(system_prompt: Optional[str] = None) -> ConversationBufferMemory:
    '''
    새로운 대화 메모리를 생성하는 함수
    '''
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    if system_prompt:
        memory.chat_memory.add_message(SystemMessage(content=system_prompt))
    return memory


def load_memory(user_id: str, chat_id: str, system_prompt: Optional[str] = None) -> ConversationBufferMemory:
    '''
    기존 사용자 채팅 기록을 불러와 메모리로 초기화
    '''
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)

    # history_manager에서 턴 단위(human+ai)로 로드됨
    history_messages: Optional[List[BaseMessage]] = load_history(user_id, chat_id)

    if history_messages:
        for msg in history_messages:
            memory.chat_memory.add_message(msg)
    else:
        if system_prompt:
            memory.chat_memory.add_message(SystemMessage(content=system_prompt))
    memory.chat_memory.messages = memory.chat_memory.messages[-(MAX_TURNS * 2 + 2):]

    return memory

