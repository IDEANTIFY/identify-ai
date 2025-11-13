# app_chatbot/chat_general.py

import os
from typing import Dict, List, Optional, TypedDict, Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from utils.file_utils import load_user_info
from utils.memory_manager import load_memory
from utils.history_manager import save_history as save_history_json
from utils.relevance_utils import check_relevance
from utils.db_search_utils import DBSearch
from utils.web_search_utils import WebSearch
from utils.title_manager import create_chat_title_if_first
from utils.sqs_server import *
import concurrent.futures

Intent = Literal["IDEA_DEV", "REPORT_Q", "SITE_DATA_Q", "GENERAL", "SMALL_TALK"]


class GraphState(TypedDict):
    '''
    그래프 상태 저장용 TypedDict 구조
    '''
    messages: List[BaseMessage]
    intent: Optional[Intent]
    answer: Optional[str]
    need_db: bool
    need_web: bool


INTENT_SYS = """사용자의 입력을 다음 중 하나로만 분류한다:
- IDEA_DEV: 아이디어 발전/확장/구체화 요청
- REPORT_Q: 기존 아이디어 '검증 리포트'에 관한 질문
- SITE_DATA_Q: 우리 '사이트/DB/등록된 정보' 안에서의 검색/확인 요청
- GENERAL: 일반 정보성 질문
- SMALL_TALK: 인사/잡담
한 단어 태그만 출력해야 한다. """

BASE_SYS = """너는 사용자의 아이디어 생성 · 발전 · 검증을 도와주는 아이디어 멘토이다.
- 먼저 사용자의 의도를 이해하고 필요한 경우 질문을 통해 맥락을 확인한다.
- 아이디어 관련 요청이 명확하지 않다면 목적을 먼저 질문한다.
- 답변은 명확하고 직관적으로 작성한다.
- 아이디어 관련 대화가 아닌 경우에도 자연스럽게 응답한다.
- **중요: 답변은 간결하고 핵심만 전달하세요. 불필요하게 길게 설명하지 마세요.**"""

REPORT_GUIDE = "기존에 생성하신 아이디어 검증 리포트에 관한 질문은 아래 ‘아이디어 검증 리포트 불러오기’ 버튼을 눌러 해당 리포트를 불러오시면 더 자세한 답변을 드릴 수 있어요!"
DEV_TAIL = "\n\n더 자세한 아이디어 디벨롭을 원하신다면 아래 ‘아이디어 디벨롭’ 버튼을 눌러보세요!"


class GeneralChatbot:
    '''
    일반 대화형 챗봇 클래스
    '''
    def __init__(self, user_id: str, chat_id: str, openai_api_key: str):
        '''
        챗봇 초기화 함수
        '''
        self.user_id = user_id
        self.chat_id = chat_id
        self.openai_api_key = openai_api_key  # 추가
        self.user_info = load_user_info(user_id) or {}
        self.user_name = self.user_info.get("name", "user")
        self.llm = ChatOpenAI(model_name="gpt-5-nano", api_key=openai_api_key)
        self.memory = load_memory(
            user_id=self.user_id,
            chat_id=self.chat_id,
            system_prompt=f"당신은 아이디어 멘토입니다. 사용자 이름은 {self.user_name}입니다."
        )
        self.web = WebSearch()
        self.db = self._init_db()
        self.graph = self._build_graph()
        print(f"General Chatbot ready: {self.user_name} ({self.user_id}, chat:{self.chat_id})")


    def _init_db(self) -> Optional[DBSearch]:
        '''
        DB 검색 엔진 초기화 함수
        '''
        base = os.path.join(os.path.dirname(__file__), "..", "dataset", "faiss")
        idx, meta = os.path.join(base, "index.faiss"), os.path.join(base, "metadata.pkl")
        model = os.getenv("EMBED_MODEL", "jhgan/ko-sroberta-multitask")
        try:
            if os.path.exists(idx) and os.path.exists(meta):
                return DBSearch(model_path=model, index_path=idx, meta_path=meta)
        except Exception as e:
            print(f"DB 초기화 실패: {e}")
        return None

    def _node_intent(self, s: GraphState) -> GraphState:
        '''
        사용자 의도 분류 노드 (모호 입력 처리 강화)
        '''
        msg = next((m for m in reversed(s["messages"]) if isinstance(m, HumanMessage)), None)
        guess = "GENERAL"
        if not msg or not msg.content or len(msg.content.strip()) < 2:
            # 입력이 너무 짧거나 의미를 파악할 수 없는 경우
            s["intent"] = "GENERAL"
            s["answer"] = "조금 더 구체적으로 설명해주실 수 있을까요?"
            return s
        user_text = msg.content.strip()
        lowered = user_text.lower()
        # 1차: LLM 기반 의도 분류
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", INTENT_SYS),
                ("human", "{x}")
            ])
            res = self.llm.invoke(prompt.format_messages(x=user_text))
            tag = (res.content or "").strip().upper()
        except Exception:
            tag = "GENERAL"
        # 2차: 분류 보정 (fallback 키워드 매칭)
        if tag in {"IDEA_DEV", "REPORT_Q", "SITE_DATA_Q", "GENERAL", "SMALL_TALK"}:
            guess = tag
        else:
            if any(k in lowered for k in ["아이디어", "발전", "구체화", "확장", "디벨롭"]):
                guess = "IDEA_DEV"
            elif any(k in lowered for k in ["리포트", "검증", "지난번", "report"]):
                guess = "REPORT_Q"
            elif any(k in lowered for k in ["사이트", "db", "데이터", "유사 사례", "등록된"]):
                guess = "SITE_DATA_Q"
            elif any(k in lowered for k in ["안녕", "반가", "고마워", "수고"]):
                guess = "SMALL_TALK"
            else:
                guess = "GENERAL"
        # 모호한 입력 처리 (GENERAL인데 내용이 애매하면 추가 질문 요청)
        if guess == "GENERAL" and len(user_text) < 6:
            s["answer"] = "조금 더 구체적으로 설명해주실 수 있을까요?"
        s["intent"] = guess
        return s

    def _node_tools(self, s: GraphState) -> GraphState:
        '''
        의도에 따른 DB 및 웹 검색 사용 여부 설정 노드
        '''
        s["need_db"], s["need_web"] = False, False
        it = s.get("intent")
        qmsg = next((m for m in reversed(s["messages"]) if isinstance(m, HumanMessage)), None)
        q = (qmsg.content if qmsg else "").lower()
        if it == "SITE_DATA_Q":
            s["need_db"] = True
        if any(k in q for k in ["최신", "최근", "요즘", "올해", "업데이트", "뉴스"]):
            s["need_web"] = True
        return s

    def _node_report_shortcut(self, s: GraphState) -> GraphState:
        '''
        리포트 관련 질문 처리 노드
        '''
        s["answer"] = REPORT_GUIDE
        return s

    def _node_draft(self, s: GraphState) -> GraphState:
        '''
        초안 응답 생성 노드 (병렬 검색 최적화)
        '''
        msgs = s["messages"]
        qmsg = next((m for m in reversed(msgs) if isinstance(m, HumanMessage)), None)
        q = qmsg.content if qmsg else ""
        ctx: List[str] = []
        
        # DB 검색과 웹 검색을 병렬로 실행
        db_hits = []
        web_hits = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # DB 검색 (필요한 경우)
            if s.get("need_db") and self.db:
                db_future = executor.submit(self._search_db, q)
            else:
                db_future = None
            
            # 웹 검색 (필요한 경우)
            if s.get("need_web"):
                web_future = executor.submit(self.web.search, q, 5)
            else:
                web_future = None
            
            # 결과 수집
            if db_future:
                try:
                    db_hits = db_future.result(timeout=3.0)  # 타임아웃 3초
                    if db_hits:
                        ctx.append("[DB]\n" + "\n---\n".join(db_hits))
                except Exception as e:
                    print(f"DB 검색 실패: {e}")
            
            if web_future:
                try:
                    web_hits = web_future.result(timeout=5.0)  # 타임아웃 5초
                    if web_hits:
                        ctx.append("[WEB]\n" + "\n---\n".join(web_hits))
                except Exception as e:
                    print(f"웹 검색 실패: {e}")
        
        u = self.user_info
        if u:
            ctx.insert(
                0,
                f"[USER] name={u.get('name','')}, major={u.get('major','')}, interests={u.get('interests','')}"
            )
        ctx_text = "\n\n".join(ctx) if ctx else "(컨텍스트 없음)"
        p = ChatPromptTemplate.from_messages([("system", BASE_SYS + "\n\n" + ctx_text), ("human", "{q}")])
        res = self.llm.invoke(p.format_messages(q=q))
        text = (res.content or "").strip()
        if s.get("intent") == "IDEA_DEV":
            text += DEV_TAIL
        s["answer"] = text
        return s
    
    def _search_db(self, query: str) -> List[str]:
        '''DB 검색 헬퍼 메서드'''
        try:
            return self.db.search(query, top_k=5)
        except Exception:
            return []

    def _node_relevance(self, s: GraphState) -> GraphState:
        '''
        응답 적합성 검증 노드
        '''
        qmsg = next((m for m in reversed(s["messages"]) if isinstance(m, HumanMessage)), None)
        ans = s.get("answer") or ""
        if not qmsg or not ans:
            return s
        decision = check_relevance(self.llm, qmsg.content, ans)
        if decision == "RETRY":
            s["need_db"] = True or s.get("need_db")
            if not s.get("need_web"):
                s["need_web"] = False
            s["answer"] = None
        return s

    def _build_graph(self):
        '''
        대화 흐름 그래프 구성 함수
        '''
        g = StateGraph(GraphState)
        g.add_node("intent", self._node_intent)
        g.add_node("tools", self._node_tools)
        g.add_node("report_shortcut", self._node_report_shortcut)
        g.add_node("draft", self._node_draft)
        g.add_node("relevance", self._node_relevance)
        g.set_entry_point("intent")

        def route_after_intent(s: GraphState):
            if s.get("intent") == "REPORT_Q":
                return "report_shortcut"
            return "tools"
        g.add_conditional_edges("intent", route_after_intent, {"report_shortcut": "report_shortcut", "tools": "tools"})
        g.add_edge("report_shortcut", END)
        g.add_edge("tools", "draft")
        g.add_edge("draft", "relevance")

        def after_check(s: GraphState):
            return "draft" if s.get("answer") is None else END
        g.add_conditional_edges("relevance", after_check, {"draft": "draft", END: END})
        return g.compile()

    def chat(self, user_input: str) -> str:
        '''
        사용자 입력 처리 및 응답 반환 함수
        '''
        self.memory.chat_memory.add_message(HumanMessage(content=user_input))
        # 초기 그래프 상태
        init: GraphState = {
            "messages": self.memory.chat_memory.messages,
            "intent": None,
            "answer": None,
            "need_db": False,
            "need_web": False
        }
        out: GraphState = self.graph.invoke(init)
        # 의도를 파악하지 못한 경우 (_node_intent에서 answer 생성됨)
        if out.get("answer"):
            ans = out.get("answer")
        else:
            # 일반적인 흐름에서 생성된 최종 답변
            ans = out.get("answer") or "도움이 될 만한 답변을 찾지 못했어요."
        # 응답 저장
        self.memory.chat_memory.add_message(AIMessage(content=ans))
        
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
        '''
        대화 기록 저장 함수
        '''
        save_history_json(self.user_id, self.chat_id, self.memory.chat_memory.messages)


if __name__ == "__main__":
    load_dotenv()

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("OPENAI_API_KEY 미설정")
        raise SystemExit(1)
    
    # SQS 요청큐, 응답큐, 리전 정보 불러오기
    request_queue_url = os.getenv("AWS_SQS_CHAT_USER_REQUEST_QUEUE")
    response_queue_url = os.getenv("AWS_SQS_CHAT_USER_RESPONSE_QUEUE")
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
            body_raw, message = receive_message_from_sqs(request_queue)
            if not body_raw:
                print("⚠️ 메시지 없음 또는 파싱 실패, 다음 메시지로 이동")
                continue

            # 문자열이면 JSON 파싱
            body = json.loads(body_raw) if isinstance(body_raw, str) else body_raw

            # 봇 생성하여 채팅 시작
            bot = GeneralChatbot(
                user_id=body.get('user_id'),
                chat_id=body.get('chat-room-id'),
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