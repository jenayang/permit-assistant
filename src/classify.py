"""판정 도메인 분류 파이프라인 (agent.py에서 분리, Phase 3).

규칙 엔진 자체(classify_case 등)는 src/agents/*.py에 이미 분리돼 있다 - 여기
있는 건 그 규칙 엔진을 그래프의 tool_calls 모양으로 감싸는 "배관" 코드다
(ClassificationOutput/_DomainConfig/DOMAIN_CONFIGS/run_classifier/classify_node).
Rule Registry 패턴은 `docs/memory/domain-registry-pattern.md` 참고.

merge_facts는 agent.py(AgentState 정의부)에서 함수 안에서 지역 임포트한다 -
agent.py가 이 모듈의 DOMAIN_CONFIGS/_derive_ledger_facility_group을 가져다
쓰므로, 모듈 최상단에서 서로를 import하면 순환참조가 된다.
"""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from langchain_core.messages import AIMessage, ToolMessage

from src.agents.fire_safety import classify_fire_safety, fire_safety_message
from src.agents.food_safety import NO_REPORT_RESULTS, classify_food_business, food_business_message
from src.agents.permit import (
    PROCEDURE_TREE,
    classify_case,
    facility_group_from_use_name,
    procedure_stage_message,
)
from src.rag.regulation import search_regulations
from src.agents.signage import classify_signage, signage_message
from src.message_utils import extract_text
from src.prompts import (
    _FIRE_PROMPT_KEYWORDS,
    _FOOD_PROMPT_KEYWORDS,
    _PERMIT_PROMPT_KEYWORDS,
    _SIGNAGE_PROMPT_KEYWORDS,
)

if TYPE_CHECKING:
    from src.agent import AgentState

logger = logging.getLogger(__name__)


def _auto_search_messages(query: str) -> list:
    """판정 결과를 쿼리로 search_regulations를 그래프가 직접 호출해, LLM이
    검색할지 말지 재량으로 정하는 지점 자체를 없앤다(2026-07-27 도입).

    LLM이 부른 게 아니라 이 함수가 직접 구성한 tool_calls 모양 메시지로
    기록하는 건 위 set_procedure_stage/set_food_business_type과 같은 패턴 -
    guard_node의 _has_grounding_search가 이름(search_regulations)만 보고
    "실제 검색 이력"으로 인정하므로 그쪽 로직도 그대로 재사용된다. 이전엔
    "[2단계: 필수 서류]를 검색해서 기록하라"는 프롬프트 지시에 LLM이 따를지
    말지 맡겼는데, 검색 자체를 생략하고 근거 없이 서류를 지어내는 사례가
    반복 확인돼([[project_tool_call_reliability]]) 판정 시점에 그래프가
    선제적으로 검색해두는 쪽으로 옮겼다 - guard_node는 이 문제를 사후에
    잡아내는 역할이었지, 애초에 검색이 일어나게 만들지는 못했다.
    """
    call_id = f"autosearch_{uuid.uuid4().hex[:8]}"
    result = search_regulations.invoke({"query": query})
    logger.info("[classify] 판정 직후 자동 검색: query=%r", query)
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": "search_regulations", "args": {"query": query}, "id": call_id}],
        ),
        ToolMessage(content=result, tool_call_id=call_id, name="search_regulations"),
    ]


@dataclass(frozen=True)
class ClassificationOutput:
    """도메인 classify_fn 하나가 반환하는 표준 산출물. classify_fn이 None을
    반환하면 "아직 정보 부족"이고, 이 객체를 반환하면(raw_result가 빈 리스트
    같은 falsy 값이어도) 그 자체로 "판정 완료"를 뜻한다 - 완료 여부를 별도
    bool 필드로 안 두는 이유: raw_result 유무 하나로 이미 충분히 표현되는데
    별도 필드를 추가하면 "완료인데 값이 없다"처럼 둘이 어긋나는 상태가
    생길 수 있어(정보 소스가 두 개가 되는 순간 서로 안 맞을 위험이 생김 -
    오늘 잡은 버그들이 전부 이 종류였다). 불변으로 둬서 그래프가 만든 뒤에
    실수로 고쳐 쓰는 것도 막는다.
    """
    raw_result: object          # 상태에 그대로 저장할 값 (str | list[str] 등, 도메인마다 다름)
    tool_name: str               # 합성 tool_call 이름(예: set_procedure_stage)
    tool_args: dict               # 그 tool_call의 args
    tool_message: str             # 대응 ToolMessage 내용(LLM이 답변에 반영할 안내문)
    rag_query: str | None         # 판정 직후 자동 검색할 쿼리. None이면 검색 생략
                                   # (인허가불필요ᆞ해당없음ᆞ빈 소방시설 목록처럼 찾을 근거가 없는 경우)


@dataclass(frozen=True)
class _DomainConfig:
    """classify_node/route_after_tools가 도메인 하나를 다루는 데 필요한 배관
    정보. facts_key/state_key는 그래프(상태 채널 이름)의 책임이라 classify_fn
    자신은 몰라도 된다 - classify_fn은 순수하게 "facts 주면 판정 결과 냄"만
    담당(Strategy 패턴, 다만 메서드가 하나뿐이라 클래스 대신 함수로 충분).
    """
    facts_key: str
    state_key: str | None         # 즉시 저장할 상태 키. None이면 저장 안 함(permit - finalize_node가 나중에 따로 처리)
    classify_fn: Callable[[dict], ClassificationOutput | None]
    record_tool: str              # 이 도메인의 사실을 기록하는 도구 이름(tool_choice 강제 대상)
    keywords: tuple[str, ...]     # 유저가 이 도메인을 꺼냈는지 판별할 키워드(프롬프트 블록 점등과 공유)
    domain: str                   # 로그ᆞ분석용 짧은 이름(facts_key에서 유도하면 case_facts→"case"처럼 어색해짐)


def _classify_permit(facts: dict) -> ClassificationOutput | None:
    """classify_case()(permit.py, 순수 판정 로직)를 감싸 ClassificationOutput
    모양으로 변환하는 배관 전용 래퍼 - classify_case() 자체와 그걸 쓰는
    기존 테스트는 안 건드린다."""
    result_type = classify_case(facts)
    if result_type is None:
        return None
    root_node_id = PROCEDURE_TREE[result_type]["root"]
    return ClassificationOutput(
        raw_result=result_type,
        tool_name="set_procedure_stage",
        tool_args={"case_type": result_type, "node_id": root_node_id},
        tool_message=procedure_stage_message(result_type, root_node_id, facts=facts),
        rag_query=None if result_type == "인허가불필요" else f"{result_type} 절차 및 필요 서류",
    )


def _classify_food(facts: dict) -> ClassificationOutput | None:
    """classify_food_business()(food_safety.py)를 감싸는 배관 전용 래퍼."""
    business_type = classify_food_business(facts)
    if business_type is None:
        return None
    return ClassificationOutput(
        raw_result=business_type,
        tool_name="set_food_business_type",
        tool_args={"business_type": business_type},
        tool_message=food_business_message(business_type),
        rag_query=None if business_type in NO_REPORT_RESULTS else f"{business_type} 영업신고 절차 및 필요 서류",
    )


def _classify_fire(facts: dict) -> ClassificationOutput | None:
    """classify_fire_safety()(fire_safety.py)를 감싸는 배관 전용 래퍼."""
    required = classify_fire_safety(facts)
    if required is None:
        return None
    return ClassificationOutput(
        raw_result=required,
        tool_name="set_fire_safety_result",
        tool_args={"required": required},
        tool_message=fire_safety_message(required),
        rag_query=f"{', '.join(required)} 설치 기준 및 절차" if required else None,
    )


def _classify_signage(facts: dict) -> ClassificationOutput | None:
    """classify_signage()(signage.py)를 감싸는 배관 전용 래퍼. permit/food/fire와
    달리 message 생성에 raw_result(허가/신고/불필요)뿐 아니라 sign_type(facts
    에서 옴)도 같이 필요해서, 이 래퍼가 둘을 합쳐 tool_args/tool_message를
    구성한다."""
    result = classify_signage(facts)
    if result is None:
        return None
    sign_type = facts["sign_type"]
    return ClassificationOutput(
        raw_result=result,
        tool_name="set_signage_result",
        tool_args={"sign_type": sign_type, "result": result},
        tool_message=signage_message(sign_type, result),
        rag_query=None if result == "허가ᆞ신고 불필요" else f"{sign_type} {result} 절차 및 필요 서류",
    )


# 도메인을 추가할 땐 이 목록에 한 줄만 추가하면 된다 - classify_node/
# route_after_tools 둘 다 이 목록만 보고 움직이므로 그래프 쪽은 더 안 고쳐도 됨.
DOMAIN_CONFIGS: list[_DomainConfig] = [
    _DomainConfig(facts_key="case_facts", state_key=None, classify_fn=_classify_permit,
                  record_tool="record_case_facts", keywords=_PERMIT_PROMPT_KEYWORDS,
                  domain="permit"),
    _DomainConfig(facts_key="food_facts", state_key="food_result", classify_fn=_classify_food,
                  record_tool="record_food_facts", keywords=_FOOD_PROMPT_KEYWORDS,
                  domain="food"),
    _DomainConfig(facts_key="fire_facts", state_key="fire_result", classify_fn=_classify_fire,
                  record_tool="record_fire_facts", keywords=_FIRE_PROMPT_KEYWORDS,
                  domain="fire"),
    _DomainConfig(facts_key="signage_facts", state_key="signage_result", classify_fn=_classify_signage,
                  record_tool="record_signage_facts", keywords=_SIGNAGE_PROMPT_KEYWORDS,
                  domain="signage"),
]


def run_classifier(state: "AgentState", config: _DomainConfig, auto_search: bool = True) -> tuple[list, dict] | None:
    """도메인 하나에 대해 "아직 미분류면 classify_fn 실행 → tool_call/
    ToolMessage 구성 → 필요하면 자동 검색까지" 공통 처리를 한다. 이미
    분류됐거나 아직 정보가 부족하면 None을 반환해 classify_node가 건너뛰게
    한다. permit(도메인)처럼 나온 결과를 즉시 저장하지 않는 경우는
    config.state_key가 None이라 자동으로 저장을 생략한다.

    auto_search=False면 rag_query가 있어도 _auto_search_messages를 안 부른다 -
    intake.html 폼 제출(submit_facts)처럼 응답을 즉시 돌려줘야 하는 흐름에서
    rerank 지연을 피하기 위함(2026-08-05 피드백). 근거 검색 자체를 없애는
    게 아니라 시점만 미루는 것 - guard_node의 _has_grounding_search가 대화
    전체를 스캔하므로, 이후 실제 채팅에서 인용이 나오면 그때 검색을 요구한다."""
    facts = state.get(config.facts_key, {})
    if facts.get("_classified"):
        return None
    output = config.classify_fn(facts)
    if output is None:
        return None

    call_id = f"classify_{uuid.uuid4().hex[:8]}"
    messages = [
        AIMessage(content="", tool_calls=[{"name": output.tool_name, "args": output.tool_args, "id": call_id}]),
        ToolMessage(content=output.tool_message, tool_call_id=call_id, name=output.tool_name),
    ]
    state_update: dict = {config.facts_key: {"_classified": True}}
    if config.state_key:
        state_update[config.state_key] = output.raw_result
    logger.info("[classify:%s] facts=%r → %r", config.facts_key, facts, output.raw_result)

    if output.rag_query and auto_search:
        messages.extend(_auto_search_messages(output.rag_query))

    return messages, state_update


_LEDGER_USE_PATTERN = re.compile(r"주용도:\s*(\S+)")


def _derive_ledger_facility_group(state: "AgentState") -> int | None:
    """가장 최근 성공한 lookup_building_ledger 조회의 주용도를 시설군 번호로
    변환해 돌려준다. 이미 current_facility_group이 있거나, 조회 실패ᆞ주용도
    없음ᆞ매핑 불가면 None.

    건축물대장 주용도(예: "제1종근린생활시설")→시설군 매핑은 결정론적이라
    코드가 직접 한다("판정은 코드, LLM은 설명" 원칙). 예전엔 이 기록을 LLM에
    맡겼는데 조회 성공 뒤에도 record_case_facts(current_facility_group)를 자주
    빠뜨려 guard가 매 턴 재시도를 걸었다 - 그 원인을 없앤다."""
    case_facts = state.get("case_facts") or {}
    if case_facts.get("current_facility_group"):
        return None
    for m in reversed(state.get("messages", [])):
        if isinstance(m, ToolMessage) and m.name == "lookup_building_ledger":
            text = extract_text(m.content)
            match = _LEDGER_USE_PATTERN.search(text)
            if not match:
                return None  # 조회 실패ᆞ미등록(주용도 줄 자체가 없음)
            return facility_group_from_use_name(match.group(1))
    return None


def classify_node(state: "AgentState", auto_search: bool = True) -> dict:
    """DOMAIN_CONFIGS에 등록된 도메인마다 run_classifier를 돌려, 그 결과를
    (LLM이 부른 게 아니라 이 노드가 직접 구성한) tool_calls 모양 메시지로
    기록한다. pipeline.py의 tool_calls 추출 로직과 프론트의 로드맵 렌더링
    코드가 "tool: set_procedure_stage" 모양만 보고 반응하므로, 별도
    API/프론트 수정 없이 그대로 재사용된다.

    도메인들은 서로 독립적이라 한 턴에 하나만, 여럿, 혹은
    (route_after_tools가 이미 걸러줘서) 아무것도 새로 분류되지 않을 수
    있다 - 해당하는 도메인만 골라 처리하고 메시지를 이어붙인다.

    auto_search=False면 run_classifier에 그대로 전달돼 판정 직후 자동 rerank
    검색을 전부 생략한다(submit_facts 전용, run_classifier 주석 참고).
    """
    # 순환참조 회피(모듈 docstring 참고) - agent.py가 이 모듈의 DOMAIN_CONFIGS를
    # 가져다 쓰므로, 여기서 agent.py를 최상단에서 import할 수 없다.
    from src.agent import merge_facts

    messages: list = []
    update: dict = {}

    # 건축물대장 주용도 → current_facility_group 자동 기록(LLM 없이 결정론적).
    # 이번 run_classifier가 방금 파생한 값도 반영하도록 로컬 상태에 미리 병합한다
    # (예: 이미 desired_facility_group이 있으면 그 자리에서 용도변경 판정까지 완료).
    derived = _derive_ledger_facility_group(state)
    if derived is not None:
        merged = merge_facts(state.get("case_facts") or {}, {"current_facility_group": derived})
        state = {**state, "case_facts": merged}
        logger.info("[classify] 건축물대장 주용도→시설군 자동 기록: current_facility_group=%d", derived)

    for config in DOMAIN_CONFIGS:
        result = run_classifier(state, config, auto_search=auto_search)
        if result is None:
            continue
        domain_messages, domain_update = result
        messages.extend(domain_messages)
        update.update(domain_update)

    # 파생한 시설군을 case_facts 업데이트에 병합한다 - 도메인 루프가 case_facts를
    # {"_classified": True}로 덮어쓸 수 있어(같은 채널) 반드시 루프 뒤에 얹는다.
    if derived is not None:
        update["case_facts"] = {**update.get("case_facts", {}), "current_facility_group": derived}

    if messages:
        update["messages"] = messages
    return update
