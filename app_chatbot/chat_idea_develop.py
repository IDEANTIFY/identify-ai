import os
import glob
import json
from typing import Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class IdeaDevelopChatbot:
    def __init__(self, user_info: Dict, openai_api_key: str, memory: ConversationBufferMemory):
        # 1. 시스템 프롬프트 생성
        system_prompt = self.create_system_prompt(user_info)
        
        # 2. LangChain 구성요소 초기화
        llm = ChatOpenAI(model_name="gpt-5-nano", api_key=openai_api_key)
        
        self.memory = memory
        # 새 대화 시작 시, 시스템 프롬프트를 메모리의 가장 처음에 추가
        if not self.memory.chat_memory.messages:
            self.memory.chat_memory.add_message(SystemMessage(content=system_prompt))

        # 3. 대화 프롬프트 템플릿 정의
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="history"), # 대화 기록
            ("human", "{user_input}"),                   # 사용자 입력
        ])

        # 4. LLMChain 생성
        self.chain = LLMChain(llm=llm, prompt=prompt, memory=self.memory)

        print(f"✅ {user_info.get('name', '사용자')}님을 위한 '아이디어 디벨롭 챗봇'이 준비되었습니다.")
        if len(self.memory.chat_memory.messages) > 1:
            print("💬 이전 대화 기록을 성공적으로 불러왔습니다. 하던 이야기를 계속 이어가세요.")

    def create_system_prompt(self, user_info: Dict) -> str:
        """챗봇의 역할과 정체성을 정의하는 시스템 프롬프트를 생성합니다."""
        user_info_str = "\n".join([f"- {key}: {value}" for key, value in user_info.items()])

        return f"""당신은 사용자의 머릿속에 있는 막연한 아이디어를 구체적인 컨셉으로 발전시키는 '아이디어 디벨롭 전문 멘토'입니다.
당신의 목표는 단순한 대답이 아닌, 영감을 주는 질문을 통해 사용자가 스스로 아이디어를 탐색하고 구체화하도록 돕는 것입니다.

**당신의 역할:**
1.  **탐색 (Explore):** 사용자의 첫 아이디어에 대해 "왜?", "누구를 위한 것인가요?"와 같은 근본적인 질문을 던져 핵심 동기를 파악합니다.
2.  **확장 (Expand):** 사용자의 관심사({user_info.get('interests', '미지정')})와 강점을 연결하여 아이디어를 더 넓은 관점으로 확장하도록 유도합니다.
3.  **구체화 (Solidify):** "만약 이 서비스를 한 문장으로 설명한다면?", "가장 중요한 기능 3가지는 무엇일까요?" 등 실행 가능한 질문으로 아이디어를 명확하게 만듭니다.
4.  **격려 (Encourage):** 항상 긍정적이고 격려하는 태도를 유지하며, 사용자가 자신감을 갖고 아이디어를 발전시킬 수 있는 안전한 환경을 제공합니다.

---
### 멘토링 대상 사용자 정보
{user_info_str}
---

이제, 사용자가 첫마디를 건네면 위 역할에 맞춰 대화를 시작하세요.
"""

    def chat(self, user_input: str) -> str:
        """사용자 입력에 대한 챗봇의 응답을 생성하고 반환합니다."""
        try:
            response = self.chain.invoke({"user_input": user_input})
            return response.get('text', "오류: 응답 텍스트를 찾을 수 없습니다.")
        except Exception as e:
            return f"API 호출 중 오류가 발생했습니다: {e}"

    def save_history(self, user_name: str) -> None:
        """대화 기록을 사용자의 이름과 타임스탬프를 포함한 JSON 파일로 저장합니다."""
        save_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'idea_dev_histories')
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = os.path.join(save_dir, f"history_{user_name}_{timestamp}.json")

        history_to_save = [
            {"role": msg.type, "content": msg.content}
            for msg in self.memory.chat_memory.messages
        ]

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(history_to_save, f, ensure_ascii=False, indent=4)
            print(f"✅ 대화 기록이 안전하게 저장되었습니다: {file_path}")
        except Exception as e:
            print(f"⚠️ 대화 기록 저장 중 오류 발생: {e}")

# --- 🚀 유틸리티 함수 ---
def find_latest_file(directory: str, prefix: str) -> Optional[str]:
    """디렉토리에서 특정 접두사를 가진 가장 최신 수정 파일을 찾습니다."""
    files = glob.glob(os.path.join(directory, f'{prefix}*.json'))
    return max(files, key=os.path.getctime) if files else None

def load_json_data(file_path: str) -> Optional[Dict]:
    """JSON 파일을 안전하게 로드합니다."""
    if not file_path or not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"파일 로드 중 오류 발생: {e}")
        return None
        
def load_conversation_history(user_name: str) -> ConversationBufferMemory:
    """사용자의 최신 대화 기록을 찾아 LangChain 메모리 객체로 변환합니다."""
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    history_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'idea_dev_histories')
    
    latest_history_file = find_latest_file(history_dir, f"history_{user_name}_")
    
    if latest_history_file:
        history_data = load_json_data(latest_history_file)
        if history_data:
            for message in history_data:
                role, content = message.get("role"), message.get("content")
                if role == "human":
                    memory.chat_memory.add_message(HumanMessage(content=content))
                elif role == "ai":
                    memory.chat_memory.add_message(AIMessage(content=content))
    return memory

# --- 🚀 메인 실행 블록 ---
if __name__ == '__main__':
    load_dotenv()
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        exit()

    print("="*60)
    print("🚀 아이디어 디벨롭 챗봇을 시작합니다.")
    print("="*60)

    # 필수 파일 경로 설정
    script_dir = os.path.dirname(__file__)
    dataset_dir = os.path.join(script_dir, '..', 'dataset')
    user_info_file = os.path.join(dataset_dir, 'user_info.json')

    # 사용자 정보 로드
    user_info = load_json_data(user_info_file)

    if user_info:
        try:
            # 챗봇 세션 시작
            user_name = user_info.get("name", "user")
            conversation_memory = load_conversation_history(user_name)

            bot = IdeaDevelopChatbot(
                user_info=user_info,
                openai_api_key=openai_key,
                memory=conversation_memory
            )
            
            print("\n안녕하세요! 어떤 아이디어를 발전시켜 볼까요? 편하게 말씀해주세요.")
            print("대화를 종료하려면 'exit' 또는 'quit'을 입력하세요.\n")

            # 대화 루프
            while True:
                user_question = input("You: ")
                if user_question.lower() in ['exit', 'quit']:
                    bot.save_history(user_name=user_name)
                    print("\n이용해주셔서 감사합니다. 멘토와의 대화는 언제든 다시 이어갈 수 있습니다.")
                    break
                
                print("\nMentor: 생각 중...", end="", flush=True)
                response = bot.chat(user_question)
                print(f"\rMentor: {response}\n")

        except Exception as e:
            print(f"챗봇 실행 중 예측하지 못한 오류가 발생했습니다: {e}")
    else:
        print("❌ 챗봇 실행에 필요한 사용자 정보 파일(`dataset/user_info.json`)을 찾을 수 없습니다.")