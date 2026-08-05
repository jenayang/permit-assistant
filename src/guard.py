"""답변 신뢰성 사후 검증 (agent.py에서 분리, Phase 4).

프롬프트 지시만으로는 LLM이 근거 없이 앞서나가는 걸 못 막는다는 게 반복
실측됐다(docs/memory/guard-node-pattern.md) - guard_node가 매 턴 자유 텍스트
답변을 정규식+상태로 결정론적으로 검증하고, 위반 시 같은 턴 내 1회 재시도를
유도한다.

2026-08-05: [Step 1-N]/[Step 2-N] 텍스트 마커 개수를 세어 강제하던 검증
(roadmap.py의 _max_allowed_stage 등에 의존)을 제거했다 - 그 마커 체계 자체를
roadmap.py의 current_step_directive로 교체하면서, 페이싱은 강제 장치 없이
순수 프롬프트 지시로만 유도하기로 했다(브레인스토밍에서 사용자가 명시적으로
선택한 트레이드오프). 마커와 무관한 나머지 검증(근거 인용ᆞ건축물대장 조회
후 기록 누락ᆞ소방ᆞ간판 판정 누락ᆞ수치 정정 누락)은 그대로 유지한다.
"""
from __future__ import annotations

import logging
import re
import uuid
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.message_utils import extract_text

if TYPE_CHECKING:
    from src.agent import AgentState

logger = logging.getLogger(__name__)

_CITATION_PATTERN = re.compile(r"\[출처:\s*([^\]]+)\]")

# [출처: ...] 인용의 근거로 인정하는 도구 이름들. search_regulations/
# search_by_term(RAG 검색)뿐 아니라 get_business_registration_guide처럼
# "검색은 안 하지만 코드 안에 원문 대조된 정적 근거를 담고 있는" 도구도
# 포함한다 - 안 넣으면 guard가 실제로 근거 있는 인용을 "지어낸 것"으로
# 오판해 불필요한 재시도를 유도한다(2026-07-27 실측 확인). 위생교육ᆞ
# 간판신고처럼 향후 추가될 정적 콘텐츠 도구도 여기 이름만 추가하면 된다.
# pipeline.py도 "법령 원문 카드" UI에서 어떤 ToolMessage가 진짜 법령 근거인지
# 걸러낼 때 이 목록을 그대로 재사용한다(밑줄 없는 공개 이름 - 단일 소스).
GROUNDING_TOOL_NAMES = (
    "search_regulations", "search_by_term",
    "get_business_registration_guide", "get_hygiene_education_guide",
    "get_opening_checklist", "get_construction_guide",
)


def _has_grounding_search(messages) -> bool:
    """대화 전체에 실제 근거(RAG 검색 또는 원문 대조된 정적 도구) 호출이
    있었는지. 대화 전체를 스캔하므로 훨씬 이전 턴의 근거로 나중 턴의 무관한
    인용까지 통과될 수 있다는 한계는 있지만, "아무 근거 없이 조항을
    지어내는" 가장 심각한 사례는 확실히 잡는다(1차 구현, 2026-07-26)."""
    return any(
        isinstance(m, ToolMessage) and m.name in GROUNDING_TOOL_NAMES
        for m in messages
    )


def _guard_retry_count(messages) -> int:
    """이번 사용자 턴(마지막 HumanMessage 이후) 안에서 guard가 이미 몇 번
    재시도를 유도했는지. 무한 루프 방지용 - 1회를 넘으면 그냥 통과시킨다."""
    count = 0
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, ToolMessage) and m.name == "_answer_guard":
            count += 1
    return count


def _building_ledger_gap_violations(state: "AgentState", messages) -> list[str]:
    """이번 턴에 lookup_building_ledger 조회가 성공했는데 case_facts에
    current_facility_group이 기록되지 않은 경우를 잡는다. SYSTEM_PROMPT에
    "조회 성공 시 같은 턴에 current_facility_group을 기록하라"는 지시가
    이미 있지만, 실측으로 반복 확인된 실패 패턴이다(세션 79430043 -
    lookup_building_ledger가 "제2종근린생활시설"을 정확히 조회했는데도
    current_facility_group이 끝내 기록 안 돼 permit_result가 영영 None으로
    남고 Step1이 멈춘 채 안 움직였음, 2026-07-28 확인ᆞguard로 승격). 조회
    자체가 실패한 경우("조회에 실패했습니다"/"등록된 건축물대장이 없습니다"
    등)는 애초에 기록할 값이 없으니 위반이 아니다 - 이번 턴(마지막
    HumanMessage 이후)만 스캔한다(_guard_retry_count와 동일한 스코프 기준).
    """
    ledger_succeeded = False
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, ToolMessage) and m.name == "lookup_building_ledger":
            content = str(m.content)
            if not any(kw in content for kw in ("실패", "없습니다", "설정되어 있지 않습니다")):
                ledger_succeeded = True
    if not ledger_succeeded:
        return []
    case_facts = state.get("case_facts") or {}
    if case_facts.get("current_facility_group"):
        return []
    return [
        "lookup_building_ledger 조회가 성공했는데(건축물대장상 주용도가 확인됨) "
        "case_facts에 current_facility_group이 기록되지 않았습니다. 지금 바로 "
        "record_case_facts(current_facility_group=...)를 조회된 주용도 기준으로 "
        "호출하고, act_type이 용도변경이면 desired_facility_group도 같은 턴에 "
        "함께 기록한 뒤 답변을 마무리하세요."
    ]


def _fire_signage_gap_violations(state: "AgentState", messages) -> list[str]:
    """사용자가 소방시설ᆞ간판을 직접 물어봤는데 아직 판정(fire_result/
    signage_result)이 안 된 채 일반 지식으로만 답하고 넘어가는 경우를 잡는다.

    SYSTEM_PROMPT에 "식품접객업이면 소방시설도 함께 판정하라"는 지시가 있지만,
    식품 판정이 먼저 끝난 뒤 사용자가 나중에 따로 소방ᆞ간판을 물어보면 그
    지시를 놓치고 record_fire_facts/record_signage_facts를 한 번도 호출하지
    않은 채(fire_facts/signage_facts가 끝까지 빈 채로) "다중이용업소 해당
    여부를 확인하세요" 같은 일반론만 답하는 사례가 실측으로 확인됨(세션
    2cda4d67, 2026-07-28 - 대화 내내 두 도구가 한 번도 안 불림ᆞ새로고침해도
    반영이 안 된다는 사용자 신고로 발견). food_result가 있는(식품접객업으로
    이미 확인된) 케이스에서만 적용한다 - 무관한 업종까지 강제하면 과잉
    개입이 된다.
    """
    if not state.get("food_result"):
        return []
    last_human = None
    tool_names_this_turn = set()
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_human = extract_text(m.content)
            break
        if isinstance(m, AIMessage):
            for tc in (m.tool_calls or []):
                tool_names_this_turn.add(tc["name"])
    if not last_human:
        return []

    violations = []
    if (
        "소방" in last_human
        and state.get("fire_result") is None
        and "record_fire_facts" not in tool_names_this_turn
    ):
        violations.append(
            "사용자가 소방시설에 대해 직접 물었는데 아직 판정(fire_result)이 안 됐습니다. "
            "연면적(size_sqm) 등 필요한 사실을 이미 알고 있으면 지금 record_fire_facts를 "
            "호출해 판정하고, 부족하면 무엇이 더 필요한지 구체적으로 되물으세요 - "
            "일반적인 지식으로 답하고 넘어가지 마세요. 판정 후에는 실행 시점(공사 진행 중 "
            "여부에 따라 지금 진행 가능한지, 아니면 시공 완료 후인지)도 함께 안내하세요 - "
            "이 사실은 [실행 시점 참고] SystemMessage로 별도 제공됩니다."
        )
    if (
        any(kw in last_human for kw in ("간판", "옥외광고"))
        and state.get("signage_result") is None
        and "record_signage_facts" not in tool_names_this_turn
    ):
        violations.append(
            "사용자가 간판ᆞ옥외광고물에 대해 직접 물었는데 아직 판정(signage_result)이 "
            "안 됐습니다. sign_type 등 필요한 사실을 이미 알고 있으면 지금 "
            "record_signage_facts를 호출해 판정하고, 부족하면 무엇이 더 필요한지 "
            "구체적으로 되물으세요 - 일반적인 지식으로 답하고 넘어가지 마세요. 판정 후에는 "
            "실행 시점(공사 진행 중 여부에 따라 지금 설치 가능한지, 아니면 시공 완료 후인지)도 "
            "함께 안내하세요 - 이 사실은 [실행 시점 참고] SystemMessage로 별도 제공됩니다."
        )
    return violations


# record_case_facts의 수치 필드 중 "정정" 감지 대상 - 문자열/불린 필드(land_zone,
# renovation_scope 등)는 숫자 비교 대상이 아니라서 제외.
_NUMERIC_FACT_FIELDS = ("size_sqm", "floors", "extension_size_sqm", "temporary_duration_years")
_CORRECTION_KEYWORDS = ("아니", "정정", "다시 말하면", "아까")
_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


def _correction_omission_violations(state: "AgentState", messages) -> list[str]:
    """유저가 이미 기록된 수치를 정정하는 발화(예: "아까 85㎡라고 했는데 사실
    100㎡야")를 했는데, 그 턴의 AI 응답이 record_case_facts를 다시 호출하지
    않아 옛 값이 그대로 남는 경우를 잡는다.

    merge_facts(agent.py)는 record_case_facts가 재호출되기만 하면 필드
    단위로 정확히 덮어써서 정정을 올바르게 처리한다 - 문제는 그 앞 단계다.
    대화가 길어지면 LLM이 tool-calling 자체를 스킵하는 현상이 반복 관측됐는데
    (docs/idea_notes.md 참고), 하필 정정 발화에서 이게 일어나면 옛 값이 그대로
    남아 잘못된 신고/허가 판정으로 이어질 수 있다 - 다른 위반들(서술만 하고
    도구 미호출)보다 오분류로 직결되는 위험이 커서 우선순위를 높게 본다
    (2026-07-31 검토, 2026-08-01 구현). "아니ᆞ정정ᆞ다시 말하면ᆞ아까" 류
    키워드 + 이미 기록된 값과 다른 숫자가 같이 나왔는지로 정정 발화를
    감지한다. 정정할 기존 수치 자체가 없거나, 언급된 숫자가 전부 이미
    기록된 값과 같으면(재확인일 뿐 정정이 아님) 위반으로 안 본다.
    """
    case_facts = state.get("case_facts") or {}
    numeric_facts = {
        f: case_facts[f] for f in _NUMERIC_FACT_FIELDS if case_facts.get(f) is not None
    }
    if not numeric_facts:
        return []

    last_human = None
    tool_names_this_turn = set()
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_human = extract_text(m.content)
            break
        if isinstance(m, AIMessage):
            for tc in (m.tool_calls or []):
                tool_names_this_turn.add(tc["name"])
    if not last_human or "record_case_facts" in tool_names_this_turn:
        return []
    if not any(kw in last_human for kw in _CORRECTION_KEYWORDS):
        return []

    mentioned_numbers = {float(n) for n in _NUMBER_PATTERN.findall(last_human)}
    recorded_values = {float(v) for v in numeric_facts.values()}
    if not (mentioned_numbers - recorded_values):
        return []

    return [
        f"이번 메시지가 이전에 기록한 수치를 정정하는 것처럼 보이는데(기존 기록: "
        f"{numeric_facts}) record_case_facts를 다시 호출하지 않았습니다. 옛 값이 "
        "그대로 남으면 잘못된 판정으로 이어지니, 지금 바로 정정된 값으로 "
        "record_case_facts를 호출한 뒤 답변을 마무리하세요."
    ]


def guard_node(state: "AgentState") -> dict:
    """agent가 자유 텍스트로 답을 끝내려 할 때, 근거ᆞ판정ᆞ기록 없이 앞서나간
    답변을 한 번 걸러낸다. LLM 판단이 아니라 정규식+상태로 결정론적으로
    감지한다(classify_case와 같은 원칙 - 프롬프트 지시만으로는 못 막는다는 게
    2026-07-26 실사용 세션에서 재현됨).

    막는 위반 네 가지(2026-08-05 - 마커 개수 기반 위반 세 가지는 roadmap.py의
    current_step_directive 전환과 함께 제거했다. 아래는 마커와 무관하게
    독립적으로 유효한 것들만 남긴다):
    1. search_regulations/search_by_term을 한 번도 호출하지 않았는데
       [출처: ...] 인용이 있는 경우 - 근거 없이 조항을 지어낸 것.
    2. lookup_building_ledger 조회가 성공했는데 case_facts에
       current_facility_group을 안 채운 경우 - 판정이 영영 안 나서 Step1이
       멈추는 실제 사례가 반복됨(_building_ledger_gap_violations 참고).
    3. 사용자가 소방시설ᆞ간판을 직접 물었는데 아직 판정이 안 된 채 일반
       지식으로만 답한 경우 - record_fire_facts/record_signage_facts가 끝까지
       한 번도 안 불려서 로드맵에 영영 반영이 안 되는 실제 사례가 확인됨
       (_fire_signage_gap_violations 참고).
    4. 유저가 이미 기록된 수치를 정정하는 발화("아니ᆞ정정ᆞ다시 말하면ᆞ아까" +
       기존 기록과 다른 숫자)를 했는데 record_case_facts가 재호출 안 된 경우 -
       옛 값이 남아 잘못된 판정으로 직결될 수 있어 다른 위반들보다 오분류
       리스크가 크다(_correction_omission_violations 참고, 2026-07-31 검토ᆞ
       2026-08-01 구현).

    같은 사용자 턴 안에서 최대 1회만 재시도를 유도한다(무한 루프 방지) - 그
    이상 반복되면 프롬프트만으로는 못 막는 한계로 보고 그냥 통과시킨다.
    """
    messages = state["messages"]
    last_text = extract_text(messages[-1].content)

    violations = []
    if _CITATION_PATTERN.search(last_text) and not _has_grounding_search(messages):
        violations.append(
            "근거 도구(search_regulations/search_by_term/get_business_registration_guide 등)를 "
            "한 번도 호출하지 않았는데 [출처: ...] 인용이 포함되어 있습니다. 근거 없이 조항을 "
            "지어내지 말고, 먼저 해당 도구를 호출해 실제 근거를 확보한 뒤 답변하세요."
        )
    violations.extend(_building_ledger_gap_violations(state, messages))
    violations.extend(_fire_signage_gap_violations(state, messages))
    violations.extend(_correction_omission_violations(state, messages))

    if violations and _guard_retry_count(messages) < 1:
        logger.info("[guard] 위반 감지, 재시도 유도: %s", violations)
        call_id = f"guard_{uuid.uuid4().hex[:8]}"
        return {
            "messages": [
                AIMessage(content="", tool_calls=[{"name": "_answer_guard", "args": {}, "id": call_id}]),
                ToolMessage(
                    content="답변 재검토 필요:\n- " + "\n- ".join(violations),
                    tool_call_id=call_id,
                    name="_answer_guard",
                ),
            ]
        }

    if violations:
        logger.warning("[guard] 위반이 재시도 후에도 지속됨 - 통과: %s", violations)

    return {}
