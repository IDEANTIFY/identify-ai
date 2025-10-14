import os
import glob
import json
from typing import Dict, Optional, List
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class chatbot:
    # ✨ 변경: memory를 외부에서 주입받도록 __init__ 수정
    def __init__(self, report_path: str, user_info: Dict, structured_idea: Dict, openai_api_key: str, memory: ConversationBufferMemory):
        if not os.path.exists(report_path):
            raise FileNotFoundError(f"리포트 파일 경로를 찾을 수 없습니다: {report_path}")

        # 1. 시스템 프롬프트 생성
        rag_context = self.create_context(report_path)
        system_prompt = self.create_system_prompt(rag_context, user_info, structured_idea)
        
        # 2. LangChain 구성요소 초기화
        llm = ChatOpenAI(model_name="gpt-5-nano", api_key=openai_api_key)
        
        self.memory = memory
        # 시스템 메시지는 메모리 로드 후 가장 처음에 추가
        if not self.memory.chat_memory.messages:
            self.memory.chat_memory.add_message(SystemMessage(content=system_prompt))

        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="history"),
            ("human", "{user_input}"),
        ])

        # 3. 체인 생성
        self.chain = LLMChain(llm=llm, prompt=prompt, memory=self.memory)

        print(f"✅ {user_info.get('name', '사용자')}님을 위한 챗봇이 준비되었습니다.")
        # ✨ 추가: 과거 대화 로드 여부 안내
        if len(self.memory.chat_memory.messages) > 1:
            print("💬 이전 대화 기록을 성공적으로 불러왔습니다.")

    def create_context(self, report_path: str) -> str:
        """리포트 파일에서 RAG 컨텍스트를 생성합니다."""
        with open(report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        summary_report = report_data.get('summary_report', {}).get('report_summary', {})
        scores = summary_report.get('evaluation_scores', {})
        
        summary_context = (
            "## 아이디어 종합 분석 요약\n"
            f"- 전체 유사 사례 수: {summary_report.get('total_similar_cases', 'N/A')}건\n"
            f"- 창의성 점수: {scores.get('creativity', 'N/A')}/100\n"
            f"- 기술 실현 가능성 점수: {scores.get('feasibility', 'N/A')}/100\n"
            f"\n### 종합 의견\n{summary_report.get('analysis_narrative', 'N/A')}\n"
        )
        
        detailed_context = "\n## 📑 유사 서비스 상세 분석\n"
        detailed_results = report_data.get('detailed_report', {}).get('detailed_results', [])
        for i, service in enumerate(detailed_results):
            detailed_context += (
                f"### [유사 서비스 {i+1}]\n"
                f"- 제목: {service.get('title', 'N/A')}\n"
                f"- 요약: {service.get('summary', 'N/A')}\n"
                f"- 아이디어 참고점(Insight): {service.get('insight', 'N/A')}\n\n"
            )
        return summary_context + detailed_context

    def create_system_prompt(self, rag_context: str, user_info: Dict, structured_idea: Dict) -> str:
        """시스템 프롬프트를 생성합니다."""
        user_info_str = "\n".join([f"- {key}: {value}" for key, value in user_info.items()])
        idea_str = "\n".join([f"- {key}: {value}" for key, value in structured_idea.items()])

        return f"""당신은 사용자의 아이디어를 구체화하고 발전시키는 '개인 맞춤형 스타트업 멘토'입니다.
사용자 정보, 원본 아이디어, 분석 리포트를 종합적으로 참고하여 개인화된 조언을 제공하세요.
친절하고 격려하는 전문가의 톤을 유지하고, 실행 가능한 '액션 아이템' 중심으로 제안하세요.

---
## 멘토링 대상 사용자 정보
{user_info_str}
---
## 사용자의 원본 아이디어
{idea_str}
---
## 아이디어 분석 리포트 (RAG)
{rag_context}
---
"""

    def chat(self, user_input: str) -> str:
        """사용자 입력에 대한 챗봇의 응답을 반환합니다."""
        try:
            response = self.chain.invoke({"user_input": user_input})
            return response.get('text', "오류: 응답 텍스트를 찾을 수 없습니다.")
        except Exception as e:
            return f"API 호출 중 오류가 발생했습니다: {e}"

    def save_history(self, user_name: str) -> None:
        """대화 기록을 JSON 파일로 저장합니다."""
        save_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'chat_histories')
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
            print(f"✅ 대화 기록이 저장되었습니다: {file_path}")
        except Exception as e:
            print(f"⚠️ 대화 기록 저장 중 오류 발생: {e}")

# --- 🚀 유틸리티 함수 ---
def find_latest_file(directory: str, prefix: str) -> Optional[str]:
    """주어진 디렉토리에서 특정 접두사를 가진 가장 최신 파일을 찾습니다."""
    files = glob.glob(os.path.join(directory, f'{prefix}*.json'))
    return max(files, key=os.path.getctime) if files else None

def load_json_data(file_path: str) -> Optional[Dict]:
    """JSON 파일을 로드하고 데이터를 반환합니다."""
    if not file_path or not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"❌ JSON 파싱 오류: {file_path}")
        return None
    except Exception as e:
        print(f"파일 로드 중 오류 발생: {e}")
        return None
        
# ✨ 추가: 과거 대화 기록을 로드하여 메모리 객체를 생성하는 함수
def load_conversation_history(user_name: str) -> ConversationBufferMemory:
    """사용자의 최신 대화 기록을 찾아 메모리에 로드합니다."""
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    
    script_dir = os.path.dirname(__file__)
    history_dir = os.path.join(script_dir, '..', 'dataset', 'chat_histories')
    
    latest_history_file = find_latest_file(history_dir, f"history_{user_name}_")
    
    if latest_history_file:
        with open(latest_history_file, 'r', encoding='utf-8') as f:
            history_data = json.load(f)
            
        for message in history_data:
            role = message.get("role")
            content = message.get("content")
            if role == "human":
                memory.chat_memory.add_message(HumanMessage(content=content))
            elif role == "ai":
                memory.chat_memory.add_message(AIMessage(content=content))
            # SystemMessage는 매번 새로 생성하므로 불러오지 않음
    
    return memory

# --- 🚀 메인 실행 블록 ---
if __name__ == '__main__':
    load_dotenv()
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("❌ OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
        exit()

    print("="*60)
    print("🚀 챗봇 초기화를 시작합니다.")
    print("="*60)

    script_dir = os.path.dirname(__file__)
    dataset_dir = os.path.join(script_dir, '..', 'dataset')
    
    user_info_file = os.path.join(dataset_dir, 'user_info.json')
    structured_idea_file = os.path.join(dataset_dir, 'structured_idea.json')
    report_file = find_latest_file(os.path.join(dataset_dir, 'reports'), 'report_total_')

    user_info = load_json_data(user_info_file)
    structured_idea = load_json_data(structured_idea_file)

    if user_info and structured_idea and report_file:
        try:
            user_name = user_info.get("name", "user")
            conversation_memory = load_conversation_history(user_name)

            bot = chatbot(
                report_path=report_file,
                user_info=user_info,
                structured_idea=structured_idea,
                openai_api_key=openai_key,
                memory=conversation_memory
            )
            print("\n안녕하세요! 아이디어에 대해 무엇이든 물어보세요.")
            print("종료하려면 'exit' 또는 'quit'을 입력하세요.\n")

            while True:
                user_question = input("You: ")
                if user_question.lower() in ['exit', 'quit']:
                    bot.save_history(user_name=user_name)
                    print("챗봇을 종료합니다.")
                    break
                
                print("\nMentor: 생각 중...", end="", flush=True)
                response = bot.chat(user_question)
                print(f"\rMentor: {response}\n")

        except Exception as e:
            print(f"챗봇 실행 중 오류 발생: {e}")
    else:
        print("❌ 챗봇 실행에 필요한 파일(사용자 정보, 아이디어, 리포트)을 모두 찾을 수 없습니다.")