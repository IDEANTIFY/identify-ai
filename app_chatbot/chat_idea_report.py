## chat_idea_report.py

import os
import json
from typing import Dict, Optional, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from utils.file_utils import load_user_info
from utils.memory_manager import load_memory
from utils.history_manager import save_history as save_history_json
from utils.title_manager import create_chat_title_if_first
from utils.sqs_server import *


class IdeaReportChatbot:
    def __init__(self, user_id: str, chat_id: str, report_json: Dict, openai_api_key: str):
        '''
        챗봇 초기화 함수
        '''
        self.user_id = user_id
        self.chat_id = chat_id
        self.openai_api_key = openai_api_key  # 추가
        self.user_info = load_user_info(user_id) or {}
        self.user_name = self.user_info.get("name", "user")
        self.report_json = report_json
        
        # 1. 시스템 프롬프트 생성
        rag_context = self.create_context(report_json)
        system_prompt = self.create_system_prompt(rag_context, self.user_info, report_json)
        
        # 2. LangChain 구성요소 초기화
        self.llm = ChatOpenAI(model_name="gpt-5-nano", api_key=openai_api_key)
        
        # 3. 메모리 로드 (user_id, chat_id 기반)
        self.memory = load_memory(
            user_id=self.user_id,
            chat_id=self.chat_id,
            system_prompt=system_prompt
        )

        # 4. 대화 프롬프트 템플릿 정의
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="history"),
            ("human", "{user_input}"),
        ])

        # 5. 체인 생성
        self.chain = LLMChain(llm=self.llm, prompt=prompt, memory=self.memory)

        print(f"✅ {self.user_name}님을 위한 챗봇이 준비되었습니다. ({self.user_id}, chat:{self.chat_id})")
        # ✨ 추가: 과거 대화 로드 여부 안내
        if len(self.memory.chat_memory.messages) > 1:
            print("💬 이전 대화 기록을 성공적으로 불러왔습니다.")

    def create_context(self, report_json: Dict) -> str:
        """리포트 JSON에서 RAG 컨텍스트를 생성합니다."""
        summary_report = report_json.get('summary_report', {}).get('report_summary', {})
        scores = summary_report.get('evaluation_scores', {})
        
        summary_context = (
            "## 아이디어 종합 분석 요약\n"
            f"- 전체 유사 사례 수: {summary_report.get('total_similar_cases', 'N/A')}건\n"
            f"- 창의성 점수: {scores.get('creativity', 'N/A')}/100\n"
            f"- 기술 실현 가능성 점수: {scores.get('feasibility', 'N/A')}/100\n"
            f"\n### 종합 의견\n{summary_report.get('analysis_narrative', 'N/A')}\n"
        )
        
        detailed_context = "\n## 📑 유사 서비스 상세 분석\n"
        detailed_results = report_json.get('detailed_report', {}).get('detailed_results', [])
        for i, service in enumerate(detailed_results):
            detailed_context += (
                f"### [유사 서비스 {i+1}]\n"
                f"- 제목: {service.get('title', 'N/A')}\n"
                f"- 요약: {service.get('summary', 'N/A')}\n"
                f"- 아이디어 참고점(Insight): {service.get('insight', 'N/A')}\n\n"
            )
        return summary_context + detailed_context

    def create_system_prompt(self, rag_context: str, user_info: Dict, report_json: Dict) -> str:
        """시스템 프롬프트를 생성합니다."""
        user_info_str = "\n".join([f"- {key}: {value}" for key, value in user_info.items()])
        
        # structured_idea는 report_json에서 추출하거나 별도로 전달받을 수 있음
        # 여기서는 report_json에 포함되어 있다고 가정하거나, 별도 필드로 받을 수 있음
        structured_idea = report_json.get('structured_idea', {})
        idea_str = "\n".join([f"- {key}: {value}" for key, value in structured_idea.items()]) if structured_idea else "N/A"

        return f"""당신은 사용자의 아이디어를 구체화하고 발전시키는 '개인 맞춤형 스타트업 멘토'입니다.
사용자 정보, 원본 아이디어, 분석 리포트를 종합적으로 참고하여 개인화된 조언을 제공하세요.
친절하고 격려하는 전문가의 톤을 유지하고, 실행 가능한 '액션 아이템' 중심으로 제안하세요.

**답변 스타일:**
- 답변은 간결하고 핵심만 전달하세요. 불필요하게 길게 설명하지 마세요.
- 리포트 내용을 인용할 때는 요약해서 전달하세요.

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
            ans = response.get('text', "오류: 응답 텍스트를 찾을 수 없습니다.")
        except Exception as e:
            ans = f"API 호출 중 오류가 발생했습니다: {e}"
        
        # 첫 대화인 경우 제목 생성
        from utils.title_manager import is_first_chat
        if is_first_chat(self.user_id, self.chat_id):
            create_chat_title_if_first(
                user_id=self.user_id,
                chat_id=self.chat_id,
                first_user_input=user_input,
                first_ai_response=ans,
                openai_api_key=self.openai_api_key
            )
        
        return ans

    def save_history(self):
        """대화 기록 저장 함수"""
        save_history_json(self.user_id, self.chat_id, self.memory.chat_memory.messages)


# --- 🚀 메인 실행 블록 ---
if __name__ == "__main__":
    load_dotenv()

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("OPENAI_API_KEY 미설정")
        raise SystemExit(1)
    
    # SQS 요청큐, 응답큐, 리전 정보 불러오기
    request_queue_url = os.getenv("AWS_SQS_CHAT_IDEA_REPORT_REQUEST_QUEUE")
    response_queue_url = os.getenv("AWS_SQS_CHAT_IDEA_REPORT_RESPONSE_QUEUE")
    region_name = os.getenv("AWS_DEFAULT_REGION")

    print(f"✅ 환경변수 로드 완료:")
    print(f"   - Request Queue: {request_queue_url}")
    print(f"   - Response Queue: {response_queue_url}")
    print(f"   - Region: {region_name}")

    # SQS 리소스 및 큐 객체 생성
    sqs = get_sqs_resource(region_name)
    request_queue = get_queue(sqs, request_queue_url)
    response_queue = get_queue(sqs, response_queue_url)

    print("\n🔄 SQS 메시지 처리 루프 시작...\n")

    processed_count = 0

    while has_messages_in_queue(request_queue):
        try:
            processed_count += 1
            print(f"\n{'='*60}")
            print(f"📨 메시지 #{processed_count} 처리 중...")
            print(f"{'='*60}\n")
            
            # SQS로부터 데이터 가져오기
            body, message = receive_message_from_sqs(request_queue)
            if not body:
                print("⚠️ 메시지 없음 또는 파싱 실패, 다음 메시지로 이동")
                continue

            # 봇 생성하여 채팅 시작
            bot = IdeaReportChatbot(
                user_id=body.get('user_id'),
                chat_id=body.get('chat-room-id'),
                report_json=body.get('idea-report-data'),
                openai_api_key=key
            )
            content = body.get('content')
            print(f"\n🚀 파이프라인 실행 시작: {content}")
            result = bot.chat(content)
            bot.save_history()

            # SQS에 데이터 전송
            print(f"\n📤 결과를 SQS 응답 큐로 전송 중...")
            message_id = send_message_to_sqs(response_queue, result)

            if message_id:
                print(f"✅ 메시지 #{processed_count} 처리 완료!")
                print(f"   MessageId: {message_id}")
            else:
                print(f"❌ 메시지 #{processed_count} 전송 실패!")
                print(f"   결과는 로컬 파일로만 저장되었습니다.")
                # 실패한 경우에도 계속 진행할지, 중단할지 결정
                # raise Exception("SQS 전송 실패")

            # 메시지 삭제 (중복 처리 방지)
            delete_message_from_sqs(message)
            print("🗑️ 원본 메시지 삭제 완료")
            
        except Exception as e:
            print(f"\n❌ 메시지 #{processed_count} 처리 중 오류 발생:")
            print(f"   {type(e).__name__}: {e}")
            import traceback
            print(traceback.format_exc())
            # 오류 발생 시 다음 메시지로 계속 진행
            continue

        # 과도한 API 호출 방지를 위한 잠시 대기
        time.sleep(1)

    print(f"\n{'='*60}")
    print(f"✅ 전체 처리 완료! 총 {processed_count}개 메시지 처리")
    print(f"{'='*60}\n")