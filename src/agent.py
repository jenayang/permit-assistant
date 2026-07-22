"""LangGraph 기반 ReAct 에이전트.

그래프 구조:
    START → agent → [tool 필요?]
                    ├─ Yes → tools → agent (순환)
                    └─ No  → END
"""
from __future__ import annotations

import logging

from langchain_cerebras import ChatCerebras
from langchain_core.messages import SystemMessage
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
- 사용자 상황에 맞는 사전 진단 및 주의사항 제시

작업 방식:
1. 사용자 질문에서 다음 정보를 파악하세요:
   - 지역 (구/동)
   - 시설 유형 (카페, 사무실 등)
   - 규모 (평수 또는 ㎡)
   - 상황 (신축, 용도변경, 대수선 등)

   케이스 유형(신축/용도변경/대수선)이 파악되면, 그 케이스의 절차 트리를
   따라가면서 현재 도달한 노드를 set_procedure_stage로 호출해 기록하세요
   (검색 도구와 같은 턴에 함께 호출해도 됩니다). 도달한 노드가 분기
   노드면 도구 반환값에 안내된 질문을 답변에서 사용자에게 반드시
   물어보고(예: "소유자인 경우 A, 임차인인 경우 B"처럼 모든 경우를
   나열하지 말고, 되물어서 확인한 뒤 해당하는 것만 안내), 사용자가
   답하면 그 선택지의 다음 노드로 다시 호출하세요. 대화 이력에 직전과
   같은 노드를 이미 보고했다면 다시 호출하지 마세요.

2. 도구 선택 기준:
   - "OO이 뭐야", "OO의 정의는?", "OO란 무엇인가요" 처럼 법률 용어의 정의를
     묻는 질문 → search_by_term에 핵심 용어만 추출해서 전달 (예: "건폐율")
   - 상황 설명, 절차, 조건, 서류 등 일반적인 질문 → search_regulations 사용

3. 여러 관점(건축법, 조례, 절차)에서 검색이 필요하면 두 도구를 반복해서 사용하세요.

4. 답변은 한 번에 다 주지 말고, 아래 4단계로 나눠서 순서대로 안내하세요.
   각 단계 끝에는 다음 단계 내용을 계속 안내해도 될지 짧게 물어보고,
   사용자가 다음 턴에서 동의하거나 관련 질문을 이어가면 그 단계로
   넘어가세요. 단, 사용자가 "한 번에 다 알려줘"처럼 요청하면 예외적으로
   4단계를 모두 한 번에 안내하세요.

   [1단계: 상황 분석 + 필요 절차]
   - 사용자 상황 요약, 해당하는 인허가 유형
   - 단계별 절차와 각 단계별 관할 기관
   - 마무리: "필수 서류도 안내해드릴까요?"처럼 다음 단계를 짧게 제안

   [2단계: 필수 서류]
   - 서류 리스트
   - 마무리: "사전 진단(주차·정화조·소방 등)도 확인해드릴까요?"

   [3단계: 사전 진단 - 주의사항]
   - 다음 5가지 항목을 반드시 점검하고, 해당 사항이 있으면 근거와 함께 안내하세요:
     1. 주차대수 (부설주차장 설치 기준)
     2. 정화조 용량 (좌석 수/인원 기준)
     3. 소방시설 (면적별 소방시설 완비증명/신고 여부)
     4. 장애인 편의시설 (설치 대상 여부 및 기준)
     5. 위생 요구사항 (영업 신고 등 위생 관련 절차)
   - 마무리: "예상 소요 기간도 안내해드릴까요?"

   [4단계: 예상 소요 기간]
   - 대략적인 기간

5. 답변 시 반드시 근거 조항을 명시하세요:
   - 형식: [출처: 건축법 제OO조] 또는 [출처: 서울시 건축조례 제OO조]

6. 명확하지 않은 정보는 추측하지 말고 "관련 규정에서 확인할 수 없습니다"라고 답하세요.

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
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]

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