# app_chatbot/utils/relevance_utils.py

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage


RELEVANCE_CHECK_SYSTEM = """
당신은 답변의 품질을 평가하는 심사관입니다.
사용자 질문과 답변을 비교하고 판단하세요.

[PASS 기준]
- 질문의 핵심 의도를 충족
- 구체적인 정보 또는 아이디어 포함
- 모호하지 않고 유의미한 도움 제공

[RETRY 기준]
- 질문과 무관한 답변
- 너무 일반적이거나 공허한 표현
- 핵심적인 내용을 놓침
- 추가 정보(DB/Web 검색)가 필요한데 제공되지 않음

출력은 반드시 'PASS' 또는 'RETRY' 중 하나로만 답하세요.
"""


def check_relevance(llm, question: str, answer: str) -> str:
    '''
    LLM 답변의 질문 연관성과 품질을 검증하는 함수
    '''
    # 프롬프트 구성
    prompt = ChatPromptTemplate.from_messages([
        ("system", RELEVANCE_CHECK_SYSTEM),
        ("human", "질문: {q}\n\n답변: {a}")
    ])

    result = llm.invoke(prompt.format_messages(q=question, a=answer))
    decision = result.content.strip().upper()

    return "PASS" if "PASS" in decision else "RETRY"
