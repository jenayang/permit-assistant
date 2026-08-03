"""그래프 라우팅 결정 (agent.py에서 분리, Phase 5).

각 route_after_*가 다음에 갈 노드 이름(문자열)만 결정한다 - 실제 처리는
해당 노드(classify_node/guard_node/finalize_node)가 한다.
"""
from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END

from src.classify import DOMAIN_CONFIGS, _derive_ledger_facility_group

# TYPE_CHECKING 가드를 안 쓰는 이유: 이 파일의 route_after_*는 LangGraph의
# add_conditional_edges에 그대로 등록되는데, 그게 내부적으로 get_type_hints()로
# 타입힌트를 실제로 resolve하려고 시도한다(add_node는 이러지 않음) - TYPE_CHECKING
# 전용으로만 두면 런타임에 AgentState를 못 찾아 NameError가 난다(Phase 5에서
# 실측 확인). agent.py는 이 모듈을 다시 import하지 않으므로 순환참조는 안 생긴다.
from src.agent import AgentState

logger = logging.getLogger(__name__)


def _any_domain_needs_classify(state: AgentState) -> bool:
    """DOMAIN_CONFIGS에 등록된 도메인 중 하나라도 "아직 미분류 + 지금 분류
    가능"이면 True. run_classifier와 판정 조건은 같지만 메시지를 실제로
    만들지는 않는 가벼운 버전 - route_after_tools는 갈 곳만 결정하면 되고
    실제 처리는 classify_node가 한다."""
    for config in DOMAIN_CONFIGS:
        facts = state.get(config.facts_key, {})
        if not facts.get("_classified") and config.classify_fn(facts) is not None:
            return True
    return False


def route_after_tools(state: AgentState) -> str:
    """tools 실행 직후: 등록된 도메인 중 하나라도 방금 완성됐으면(각각
    record_case_facts/record_food_facts/record_fire_facts로 채워짐) agent로
    돌아가기 전에 classify부터 강제한다.

    예전엔 이 체크를 route_after_agent(agent 응답 이후)에서만 했는데, 그러면
    "case_facts는 이미 다 모였지만 LLM이 그 사실을 모른 채 도구 호출 없이
    먼저 답변을 시도 -> 그래프가 그 답변을 무시하고 classify로 강제 전환 ->
    다음 턴 LLM이 '이미 답했나?' 헷갈려하며 부실한 후속 답변을 내는" 문제가
    실제로 발생했다. tools 직후로 당기면 LLM이 답변을 시도하기 전에 분류가
    먼저 끝나서 이 문제가 구조적으로 사라진다. 도메인이 늘어도 이 타이밍
    원칙은 DOMAIN_CONFIGS를 통해 그대로 적용된다.
    """
    if _any_domain_needs_classify(state) or _derive_ledger_facility_group(state) is not None:
        logger.info("[route_after_tools] → classify")
        return "classify"
    logger.info("[route_after_tools] → agent (분류 조건 미충족 또는 이미 분류됨)")
    return "agent"


def route_after_agent(state: AgentState) -> str:
    """agent 다음 어디로 갈지: 실제 도구 호출 > 자유 텍스트 답변은 무조건 guard.

    분류(classify)는 이제 route_after_tools에서 처리되므로, agent가 응답하는
    시점엔 이미 분류가 끝나 있는 게 정상이라 여기선 다시 체크하지 않는다.
    자유 텍스트로 답을 끝내려는 시도는 finalize/END로 바로 보내지 않고 항상
    guard를 거쳐 단계 상한ᆞ인용 근거를 먼저 검증한다.
    """
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        logger.info("[route_after_agent] tool_calls=%r → tools", [tc["name"] for tc in last.tool_calls])
        return "tools"

    logger.info("[route_after_agent] 자유 텍스트 답변 → guard")
    return "guard"


def route_after_guard(state: AgentState) -> str:
    """guard 다음 어디로 갈지: 재시도 유도 메시지가 방금 추가됐으면 agent로
    돌아가고, 아니면 (분류 끝났고 아직 미종합이면) finalize, 그 외엔 종료."""
    last = state["messages"][-1]
    if isinstance(last, ToolMessage) and last.name == "_answer_guard":
        logger.info("[route_after_guard] 재시도 유도 → agent")
        return "agent"

    facts = state.get("case_facts", {})
    if facts.get("_classified") and not facts.get("_finalized"):
        logger.info("[route_after_guard] case_facts=%r → finalize", facts)
        return "finalize"

    logger.info("[route_after_guard] case_facts=%r → END", facts)
    return END
