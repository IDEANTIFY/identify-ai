# app_chatbot/chat_general.py

import os
from typing import Dict, List, Optional, TypedDict, Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from utils.file_utils import load_json
from utils.memory_manager import load_memory
from utils.history_manager import save_history as save_history_json
from utils.relevance_utils import check_relevance
from utils.db_search_utils import DBSearch
from utils.web_search_utils import WebSearch

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
- 아이디어 관련 대화가 아닌 경우에도 자연스럽게 응답한다."""

REPORT_GUIDE = "기존에 생성하신 아이디어 검증 리포트에 관한 질문은 아래 ‘아이디어 검증 리포트 불러오기’ 버튼을 눌러 해당 리포트를 불러오시면 더 자세한 답변을 드릴 수 있어요!"
DEV_TAIL = "\n\n더 자세한 아이디어 디벨롭을 원하신다면 아래 ‘아이디어 디벨롭’ 버튼을 눌러보세요!"


class GeneralChatbot:
    '''
    일반 대화형 챗봇 클래스
    '''
    def __init__(self, user_info: Dict, openai_api_key: str):
        '''
        챗봇 초기화 함수
        '''
        self.user_info = user_info
        self.user_name = user_info.get("name", "user")
        self.llm = ChatOpenAI(model_name="gpt-5-nano", api_key=openai_api_key)
        self.memory = load_memory(self.user_name, system_prompt="당신은 아이디어 일반 멘토입니다.")
        self.web = WebSearch()
        self.db = self._init_db()
        self.graph = self._build_graph()
        print(f"General Chatbot ready: {self.user_name}")

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
        초안 응답 생성 노드
        '''
        msgs = s["messages"]
        qmsg = next((m for m in reversed(msgs) if isinstance(m, HumanMessage)), None)
        q = qmsg.content if qmsg else ""
        ctx: List[str] = []
        if s.get("need_db") and self.db:
            try:
                hits = self.db.search(q, top_k=5)
                if hits:
                    ctx.append("[DB]\n" + "\n---\n".join(hits))
            except Exception:
                pass
        if s.get("need_web"):
            hits = self.web.search(q, top_k=5)
            if hits:
                ctx.append("[WEB]\n" + "\n---\n".join(hits))
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
        return ans

    def save_history(self):
        '''
        대화 기록 저장 함수
        '''
        save_history_json(self.user_name, self.memory.chat_memory.messages)


if __name__ == "__main__":
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("OPENAI_API_KEY 미설정")
        raise SystemExit(1)

    user_info = load_json(os.path.join(os.path.dirname(__file__), "..", "dataset", "user_info.json"))
    if not user_info:
        print("user_info.json 없음")
        raise SystemExit(1)

    bot = GeneralChatbot(user_info=user_info, openai_api_key=key)
    print("\n안녕하세요! 아이디어에 대해 무엇이든 물어보세요.")
    print("종료하려면 'exit' 또는 'quit'을 입력하세요.\n")

    while True:
        q = input("You: ")
        if q.lower() in ["exit", "quit"]:
            bot.save_history()
            print("\n챗봇을 종료합니다.")
            break
        print("Bot: 생각 중...", end="", flush=True)
        a = bot.chat(q)
        print(f"\rBot: {a}\n")
