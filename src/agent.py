"""LangGraph 기반 ReAct 에이전트.

그래프 구조:
    START → agent → [도구 필요?]
                    ├─ 실제 도구 호출 있음 → tools → agent (순환)
                    ├─ case_facts 다 모였고 아직 미분류 → classify → agent (한 번 더,
                    │  분류 결과를 답변에 자연스럽게 반영하도록)
                    └─ 그 외 → END

classify는 LLM이 판단하는 게 아니라 파이썬 규칙 함수(src.tools.classify_case)를
그래프가 강제로 실행하는 노드다 - 허가/신고/기재변경 판정처럼 법령상 객관적 기준으로
정해지는 값을 LLM 재량에 맡기지 않기 위함(LLM이 도구 호출을 빼먹는 문제를 겪은 뒤 도입).
"""
from __future__ import annotations

import logging
import uuid
from typing import Annotated

from langchain_cerebras import ChatCerebras
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from src import config
from src.tools import TOOLS, classify_case, procedure_stage_message, PROCEDURE_TREE

logger = logging.getLogger(__name__)


# === 그래프 상태 ===(MessagesState + 대화 중 파악된 사용자 상황 사실들)
def merge_facts(existing: dict, update: dict) -> dict:
    """case_facts 채널의 reducer. update에서 None이 아닌 값만 덮어써서 누적한다
    (record_case_facts가 매 턴 일부 필드만 채워 보내도 이전 값이 안 지워지도록)."""
    return {**existing, **{k: v for k, v in update.items() if v is not None}}


class AgentState(MessagesState):
    case_facts: Annotated[dict, merge_facts]


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
   - 행위 유형: 신축ᆞ증축ᆞ개축ᆞ재축ᆞ이전ᆞ대수선ᆞ용도변경ᆞ일반수선ᆞ가설건축물 중 어디인지
   - 위 행위 유형 판정에 필요한 사실(규모, 층수, 용도지역, 대수선 해당 여부,
     기존/희망 시설군, 가설건축물 목적ᆞ존치기간 등 - 상황에 맞는 것만)
   - 소유자인지 임차인인지

   **허가 대상인지 신고 대상인지, 용도변경이 허가/신고/기재변경 중 무엇인지는
   법령상 객관적 기준(면적ᆞ층수ᆞ시설군)으로 정해지는 값이라 사용자에게 직접
   묻지 마세요** — 대신 위 사실들이 파악되는 대로 record_case_facts를
   호출해서 기록하면, 정보가 충분히 모이는 순간 시스템이 자동으로 규칙에
   따라 판정합니다(검색 도구와 같은 턴에 함께 호출 가능). 판정 결과는 다음
   턴에 도구 응답으로 전달되니, 그걸 참고해서 답변에 자연스럽게 반영하세요.

   **판정 결과가 아직 도구 응답으로 오지 않았다면, 허가/신고/기재변경 여부를
   당신이 직접 계산하거나 단정하지 마세요.** 특히 다음을 절대 하지 마세요:
   - "연면적 OO㎡ 초과라 허가 대상입니다"처럼 스스로 기준을 계산해서 결론 내리기
   - **85㎡ 기준은 증축ᆞ개축ᆞ재축에만 적용됩니다. 신축에는 적용되지 않으니
     절대 혼동하지 마세요** — 신축의 신고 대상 여부는 용도지역(관리ᆞ농림ᆞ
     자연환경보전지역)ᆞ연면적 200㎡ 미만ᆞ층수 3층 미만을 모두 봐야 하며,
     이 세 값 중 하나라도 모르면 판정 불가능한 것이지 85㎡와는 무관합니다.
   - search_regulations/search_by_term으로 실제 검색하지 않은 조항을 지어내서
     [출처: ...] 형태로 인용하기
   - 사용자가 이미 말한 수치(예: "2층", "150㎡", "관리지역")를 record_case_facts에
     빠뜨리고 넘어가기 — 질문에 등장한 규모ᆞ층수ᆞ용도지역 등은 반드시 그 턴에
     전부 추출해서 기록하세요. 하나라도 놓치면 시스템이 판정을 못 합니다.
   판정에 필요한 정보가 하나라도 비어 있으면, 결론을 내지 말고 부족한 사실을
   채우세요. 단, **용도지역은 예외** - 일반인은 자기 땅 용도지역을 모르는 경우가
   많으니 사용자에게 바로 묻지 말고, 구/동 이상 수준의 주소를 이미 알고 있다면
   먼저 lookup_land_zone을 호출해서 자동으로 확인하세요(성공하면 그 결과를 같은
   턴에 record_case_facts로 기록). 자동 조회가 실패했다는 응답이 왔을 때만
   사용자에게 직접 물어보세요. 용도지역 외의 다른 부족한 사실(층수ᆞ면적 등)은
   원래대로 사용자에게 짧게 되물으세요. "아마 허가 대상일 것 같다"처럼
   애매하게라도 결론을 암시하지 마세요 — 모르면 모른다고 하고 되묻는 것이
   정답입니다.

2. 도구 선택 기준:
   - "OO이 뭐야", "OO의 정의는?", "OO란 무엇인가요" 처럼 법률 용어의 정의를
     묻는 질문 → search_by_term에 핵심 용어만 추출해서 전달 (예: "건폐율")
   - 상황 설명, 절차, 조건, 서류 등 일반적인 질문 → search_regulations 사용
   - 용도지역을 모르는데 주소는 알고 있는 경우 → lookup_land_zone 사용(1번 참고)

3. 여러 관점(건축법, 조례, 절차)에서 검색이 필요하면 두 도구를 반복해서 사용하세요.

4. 답변 분기 - 아래 두 경우를 먼저 구분하고, 섞지 마세요:

   **(A) 판정에 필요한 사실이 아직 부족한 경우** (1번 참고): 부족한 사실을
   확인하는 질문 1~2문장으로만 끝내세요. `[1단계]` 같은 헤더나 절차ᆞ서류
   설명은 이번 턴에 절대 꺼내지 마세요 - 정보 수집과 4단계 안내는 서로 다른
   턴에서 일어나는 별개의 일입니다.

   **(B) 판정이 끝났거나(도구 응답으로 옴) 애초에 판정이 필요 없는 질문**
   (정의 질문 등): 아래 4단계로 나눠서 순서대로 안내하세요. 한 번에 다 주지
   말고, 각 단계 끝에는 다음 단계를 계속 안내해도 될지 짧게 물어보세요.
   사용자가 다음 턴에서 동의하거나 관련 질문을 이어가면 그 단계로 넘어가고,
   "한 번에 다 알려줘"처럼 요청하면 예외적으로 4단계를 모두 한 번에
   안내하세요. **각 단계는 핵심만 간결하게 - 이미 앞 단계나 이전 턴에서 말한
   내용(상황 요약, 판정 결과 등)을 다음 단계에서 다시 설명하지 마세요.**

   [1단계: 상황 분석 + 필요 절차]
   - 사용자 상황 한 줄 요약, 해당하는 인허가 유형
   - 절차는 단계 이름과 관할 기관만 나열(각 단계 설명은 1줄 이내)
   - 마무리: "필수 서류도 안내해드릴까요?"처럼 다음 단계를 짧게 제안

   [2단계: 필수 서류]
   - 서류 리스트만(설명 없이)
   - 마무리: "사전 진단(주차·정화조·소방 등)도 확인해드릴까요?"

   [3단계: 사전 진단 - 주의사항]
   - 다음 5가지 항목 중 실제 해당하는 것만, 항목당 1~2줄로 근거와 함께 안내:
     1. 주차대수 (부설주차장 설치 기준)
     2. 정화조 용량 (좌석 수/인원 기준)
     3. 소방시설 (면적별 소방시설 완비증명/신고 여부)
     4. 장애인 편의시설 (설치 대상 여부 및 기준)
     5. 위생 요구사항 (영업 신고 등 위생 관련 절차)
   - 마무리: "예상 소요 기간도 안내해드릴까요?"

   [4단계: 예상 소요 기간]
   - 대략적인 기간만 한 줄로

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
def agent_node(state: AgentState) -> dict:
    """LLM을 호출해서 다음 액션 결정.

    - 도구가 필요하면 tool_calls를 반환
    - 답변 가능하면 최종 답변 반환
    - Gemini 무료 티어 할당량(하루 20회) 소진 시 Cerebras(config.CEREBRAS_MODEL)로 자동 전환
    - 응답에 record_case_facts 호출이 있으면 그 인자를 case_facts에 병합
    """
    global _gemini_quota_exhausted
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]

    if not _gemini_quota_exhausted:
        try:
            response = llm_with_tools.invoke(messages)
            return _agent_result(response)
        except Exception as exc:
            if not _is_quota_error(exc):
                raise
            logger.warning(
                "Gemini 무료 티어 할당량 소진 감지. 이후 요청은 Cerebras(%s)로 전환합니다.",
                config.CEREBRAS_MODEL,
            )
            _gemini_quota_exhausted = True

    response = _get_cerebras_llm_with_tools().invoke(messages)
    return _agent_result(response)


def _agent_result(response: AIMessage) -> dict:
    """LLM 응답에서 record_case_facts 호출을 찾아 case_facts 갱신분으로 뽑아낸다.

    record_case_facts 자체는 여전히 ToolNode가 정상 실행해서(확인 문자열만
    반환) 도구 호출-응답 짝은 그대로 맞춰지고, 여기서는 그 인자를 그래프
    상태(case_facts)에도 반영하는 부수 작업만 한다.
    """
    facts_update: dict = {}
    for tc in getattr(response, "tool_calls", None) or []:
        if tc["name"] == "record_case_facts":
            facts_update.update(tc["args"])
    result: dict = {"messages": [response]}
    if facts_update:
        result["case_facts"] = facts_update
    return result


def classify_node(state: AgentState) -> dict:
    """case_facts가 충분히 모이면 classify_case()로 결정론적 분류를 실행하고,
    그 결과를 (LLM이 부른 게 아니라 이 노드가 직접 구성한) tool_calls 모양
    메시지로 기록한다. pipeline.py의 tool_calls 추출 로직과 프론트의 로드맵
    렌더링 코드가 "tool: set_procedure_stage" 모양만 보고 반응하므로, 별도
    API/프론트 수정 없이 그대로 재사용된다.
    """
    facts = state.get("case_facts", {})
    result_type = classify_case(facts)
    if result_type is None:
        return {}

    root_node_id = PROCEDURE_TREE[result_type]["root"]
    call_id = f"classify_{uuid.uuid4().hex[:8]}"

    ai_msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "set_procedure_stage",
            "args": {"case_type": result_type, "node_id": root_node_id},
            "id": call_id,
        }],
    )
    tool_msg = ToolMessage(
        content=procedure_stage_message(result_type, root_node_id),
        tool_call_id=call_id,
        name="set_procedure_stage",
    )
    logger.info("[classify] case_facts=%r → %s", facts, result_type)
    return {
        "messages": [ai_msg, tool_msg],
        "case_facts": {"_classified": True},
    }


def route_after_agent(state: AgentState) -> str:
    """agent 다음 어디로 갈지: 실제 도구 호출 > (정보 다 모였고 미분류면) 분류 > 종료."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"

    facts = state.get("case_facts", {})
    if not facts.get("_classified") and classify_case(facts) is not None:
        return "classify"

    return END


# === ToolNode 생성 ===
tool_node = ToolNode(tools=TOOLS)   # tool 콜 받아서 실제 함수 실행


# === Graph 구성 ===
def build_graph():
    """ReAct 에이전트 그래프 구성.

    - checkpointer: 대화 이력 (user_id 기준)
    - store: 사용자 정보 저장소 (user_id 기준?, long-term)
    """
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_node("classify", classify_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "classify": "classify",
            END: END,
        }
    )
    builder.add_edge("tools", "agent")      # 순환엣지
    builder.add_edge("classify", "agent")   # 분류 결과를 LLM이 답변에 반영하도록 한 번 더

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
