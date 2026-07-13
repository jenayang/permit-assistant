"""LangGraph 기반 ReAct 에이전트.

그래프 구조:
    START → agent → [tool 필요?]
                    ├─ Yes → tools → agent (순환)
                    └─ No  → END
"""
from __future__ import annotations

import logging

from langchain_cerebras import ChatCerebras
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore 
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src import config
from src.tools import TOOLS

logger = logging.getLogger(__name__)

# === 시스템 프롬프트 ===
SYSTEM_PROMPT = """당신은 서울시 건축 인허가 전문 어시스턴트입니다.

역할:
- 소규모 상업시설(카페, 사무실, 소매점 등)의 인허가 절차 안내
- 건축법, 시행령, 시행규칙 및 서울시 조례 기반 정확한 정보 제공
- 한 번에 모든 정보를 쏟아내지 않고, 대화를 이어가며 단계적으로 깊이 있는 정보 제공

대화 방식 (중요):
- 답변은 짧고 간결하게. 한 턴에 1~2개 주제만 다루세요.
- 지역/시설유형/규모/상황 중 판단에 필요한 정보가 빠져 있으면 추측하지 말고
  먼저 짧게 되물어서 확인한 뒤 답변하세요.
- 정보가 충분하면 [상황 분석](해당 인허가 유형 포함)만 먼저 간단히 답하고,
  답변 끝에 "절차 / 필요서류 / 사전진단 주의사항 / 예상 소요기간 중 어떤 게
  궁금하세요?"처럼 다음 선택지를 제시해 대화를 이어가세요.
- 사용자가 특정 주제(예: 사전진단, 절차, 서류)만 물으면 그 부분만 답하세요.
  매번 전체 항목을 다 나열하지 마세요.
- 사용자가 "한번에 다 정리해줘"처럼 전체 요약을 요청할 때만 아래
  [전체 답변 구조]를 한 번에 제공하세요.

도구 선택 기준:
- "OO이 뭐야", "OO의 정의는?", "OO란 무엇인가요" 처럼 법률 용어의 정의를
  묻는 질문 → search_by_term에 핵심 용어만 추출해서 전달 (예: "건폐율")
- 상황 설명, 절차, 조건, 서류 등 일반적인 질문 → search_regulations 사용
- 여러 관점(건축법, 조례, 절차)에서 검색이 필요하면 두 도구를 반복해서 사용하세요.

사전 진단 체크리스트 (사용자가 물어본 범위 내에서, 해당 시 근거와 함께 안내):
1. 주차대수 (부설주차장 설치 기준)
2. 정화조 용량 (좌석 수/인원 기준)
3. 소방시설 (면적별 소방시설 완비증명/신고 여부)
4. 장애인 편의시설 (설치 대상 여부 및 기준)
5. 위생 요구사항 (영업 신고 등 위생 관련 절차)

[전체 답변 구조] (사용자가 전체 정리를 명시적으로 요청했을 때만 사용):
   [상황 분석] - 사용자 상황 요약 + 해당 인허가 유형
   [필요 절차] - 단계별 절차 + 각 단계별 관할 기관
   [필수 서류] - 서류 리스트
   [사전 진단 - 주의사항] - 위 5개 항목 전부 점검
   [예상 소요 기간] - 대략적인 기간

답변 시 반드시 근거 조항을 명시하세요:
- 형식: [출처: 건축법 제OO조] 또는 [출처: 서울시 건축조례 제OO조]

명확하지 않은 정보는 추측하지 말고 "관련 규정에서 확인할 수 없습니다"라고 답하세요.

주의사항:
- 최종 판단은 관할 구청 및 건축사 상담 권장 안내
- 최신 개정 여부는 국가법령정보센터(law.go.kr) 확인 권장
- 개별 사안의 세부 판단은 전문가 상담 필요
"""

# === LLM + 도구 바인딩 ===
llm = ChatGoogleGenerativeAI(
    model = config.GEMINI_MODEL,
    google_api_key = config.GEMINI_API_KEY,
    temperature = config.TEMPERATURE,
    max_output_tokens = config.MAX_OUTPUT_TOKENS,
    thinking_budget = config.THINKING_BUDGET,
    max_retries = 0,  # 할당량 소진은 재시도해도 안 풀리므로 즉시 Cerebras로 폴백
)
llm_with_tools = llm.bind_tools(TOOLS)   # TOOL을 LLM에 바인딩. 제공함

# Cerebras는 실제로 폴백이 발생할 때만 생성 (CEREBRAS_API_KEY가 없으면 생성 시점에 에러남)
_cerebras_llm_with_tools = None

# 프로세스 내에서 Gemini 할당량 소진이 한 번 확인되면 이후 요청은 바로 Cerebras로 보냄
# (매번 Gemini를 재시도하며 429를 반복 유발하지 않기 위함, 재시작 시 초기화)
_gemini_quota_exhausted = False


def _is_quota_error(exc: Exception) -> bool:
    """Gemini 무료 티어 일일 할당량 초과(429 RESOURCE_EXHAUSTED) 여부 판별."""
    msg = str(exc)
    return "RESOURCE_EXHAUSTED" in msg or "429" in msg


def _flatten_content(content):
    """블록/리스트 형태의 메시지 content를 일반 문자열로 변환.

    Gemini는 답변을 문자열 리스트(예: ['텍스트1', '텍스트2'])나
    {'type': 'text', ...} 블록 리스트로 반환하기도 하는데, OpenAI 호환
    API(Cerebras)는 content가 문자열이거나 dict 객체 리스트여야만 해서
    Gemini가 만든 과거 메시지를 그대로 넘기면 400 에러가 남.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return content


def _normalize_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """대화 기록의 AI 메시지 content를 항상 일반 문자열로 통일.

    Gemini↔Cerebras 폴백처럼 한 대화 안에서 서로 다른 제공자가 메시지를
    주고받을 때, 이전 제공자가 만든 content 포맷이 다음 제공자의 API에서
    거부되는 것을 막기 위함.
    """
    normalized = []
    for m in messages:
        if isinstance(m, AIMessage) and isinstance(m.content, list):
            m = m.model_copy(update={"content": _flatten_content(m.content)})
        normalized.append(m)
    return normalized


def _get_cerebras_llm_with_tools():
    global _cerebras_llm_with_tools
    if _cerebras_llm_with_tools is None:
        cerebras_llm = ChatCerebras(
            model = config.CEREBRAS_MODEL,
            api_key = config.CEREBRAS_API_KEY,
            temperature = config.TEMPERATURE,
            max_tokens = config.MAX_OUTPUT_TOKENS,
        )
        _cerebras_llm_with_tools = cerebras_llm.bind_tools(TOOLS)
    return _cerebras_llm_with_tools


# === 노드 정의 ===
def agent_node(state: MessagesState) -> dict:
    """LLM을 호출해서 다음 액션 결정.

    - 도구가 필요하면 tool_calls를 반환
    - 답변 가능하면 최종 답변 반환
    - Gemini 무료 티어 할당량(하루 20회) 소진 시 Cerebras(config.CEREBRAS_MODEL)로 자동 전환
    """
    global _gemini_quota_exhausted
    messages = _normalize_messages([SystemMessage(content=SYSTEM_PROMPT), *state["messages"]])

    if not _gemini_quota_exhausted:
        try:
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}
        except Exception as exc:
            if not _is_quota_error(exc):
                raise
            logger.warning(
                "Gemini 무료 티어 할당량 소진 감지. 이후 요청은 Cerebras(%s)로 전환합니다.",
                config.CEREBRAS_MODEL,
            )
            _gemini_quota_exhausted = True

    response = _get_cerebras_llm_with_tools().invoke(messages)
    return {"messages": [response]}


# === ToolNode 생성 ===
tool_node = ToolNode(tools=TOOLS)   # tool 콜 받아서 실제 함수 실행


# === Graph 구성 ===
def build_graph():
    """ReAct 에이전트 그래프 구성.

    - checkpointer: 대화 이력 (user_id 기준)
    - store: 사용자 정보 저장소 (user_id 기준?, long-term)
    """
    builder = StateGraph(MessagesState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",   # 생략가능
            "__end__": END,     # 생략가능
        }
    )
    builder.add_edge("tools", "agent")   # 순환엣지

    # 체크포인터로 대화 이력 관리
    checkpointer = InMemorySaver()  # short-term memory
    store = InMemoryStore()         # long-term memory
    return builder.compile(checkpointer=checkpointer, store=store)


# === 컴파일된 그래프 (모듈 로드 시 1회) ===
graph = build_graph()

# === Store 접근 헬퍼 (외부에서 저장/조회 가능하도록) ===
def get_store():
    """그래프에 연결된 store 반환.
    
    사용 예:
        from src.agent import get_store
        store = get_store()
        store.put(("user", "jena"), "language", {"value": "ko"})
        result = store.get(("user", "jena"), "language")
    """
    return graph.store