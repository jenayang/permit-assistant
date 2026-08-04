"""답변 신뢰성 사후 검증 (agent.py에서 분리, Phase 4).

프롬프트 지시만으로는 LLM이 근거 없이 앞서나가는 걸 못 막는다는 게 반복
실측됐다(docs/memory/guard-node-pattern.md) - guard_node가 매 턴 자유 텍스트
답변을 정규식+상태로 결정론적으로 검증하고, 위반 시 같은 턴 내 1회 재시도를
유도한다.

roadmap.py의 단계 추적(_max_allowed_stage/_mentioned_stages/_STAGE_DOMAIN/
_STEP2_KEYWORDS)에 의존한다(단방향 - roadmap.py는 이 모듈을 모른다).
"""
from __future__ import annotations

import logging
import re
import uuid
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.message_utils import extract_text
from src.roadmap import (
    _STAGE_DOMAIN,
    _STEP2_KEYWORDS,
    _max_allowed_construction_stage,
    _max_allowed_stage,
    _mentioned_construction_stages,
    _mentioned_stages,
)

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


def _construction_guide_shown(messages) -> bool:
    """get_construction_guide가 대화 전체에서 한 번이라도 호출됐는지 -
    disclosed_stage["construction_guide_shown"] 갱신에 쓴다(permit_phase_directive의
    Step 1→2 전환 넛지를 한 번 보여준 뒤엔 매 턴 반복하지 않기 위함). 도구
    이름을 직접 문자열로 비교하지만, 이 호출 이력 → 상태 플래그 변환 지점을
    guard_node 한 곳으로 모아뒀기 때문에 도구가 나중에 개명되거나 대체돼도
    고칠 곳은 여기 한 줄뿐이다."""
    return any(
        isinstance(m, ToolMessage) and m.name == "get_construction_guide"
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


def _synthesis_gap_violations(state: "AgentState", max_mentioned: int) -> list[str]:
    """[Step 1-2]/[Step 1-3]을 언급했는데 record_permit_synthesis로 실제 기록은
    안 한 경우를 잡는다. "도구를 호출했다"는 문장을 텍스트로 지어내고 실제로는
    tool_calls가 비어있던 사례가 실측으로 확인됐다(2026-07-27) - 인용 위반과
    같은 종류의 문제(근거 없이 답만 그럴듯하게 마무리)라 같은 guard에서 같이
    잡는다. 이전 턴에 이미 기록됐으면(permit_synthesis가 누적 상태라) 재확인
    답변에서 또 걸릴 일은 없다 - 그때그때 "이번 턴에 호출했는지"가 아니라
    "지금 상태에 실제로 있는지"만 본다.

    2026-08-03: Step1 순서를 재정렬(사전검토를 서류준비보다 앞으로)하면서
    번호별 필드 매핑도 함께 바꿨다 - [Step 1-2]는 이제 사전검토(pre_diagnosis_items),
    [Step 1-3]은 서류준비(required_documents)를 가리킨다.
    """
    synthesis = state.get("permit_synthesis", {})
    violations = []
    if max_mentioned >= 2 and not synthesis.get("pre_diagnosis_items"):
        violations.append(
            "[Step 1-2: 사전 검토]를 언급했지만 record_permit_synthesis(pre_diagnosis_items=[...])를 "
            "실제로 호출하지 않았습니다. 항목을 텍스트로 나열하는 데서 끝내지 말고, 반드시 "
            "그 도구를 실제로 호출해서 기록한 뒤 답변을 마무리하세요."
        )
    if max_mentioned >= 3 and not synthesis.get("required_documents"):
        violations.append(
            "[Step 1-3: 서류 준비]를 언급했지만 record_permit_synthesis(required_documents=[...])를 "
            "실제로 호출하지 않았습니다. 서류를 텍스트로 나열하는 데서 끝내지 말고, 반드시 "
            "그 도구를 실제로 호출해서 기록한 뒤 답변을 마무리하세요."
        )
    return violations


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
            "일반적인 지식으로 답하고 넘어가지 마세요."
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
            "구체적으로 되물으세요 - 일반적인 지식으로 답하고 넘어가지 마세요."
        )
    return violations


def _construction_guide_gap_violations(state: "AgentState", last_text: str, messages) -> list[str]:
    """Step 1 안내가 다 끝났는데(permit_phase_directive가 get_construction_guide
    호출을 지시하는 시점) 그 지시를 무시하고 Step 2 내용을 텍스트로만 설명한
    경우를 잡는다. permit_phase_directive는 연성 유도(SystemMessage)일 뿐이고
    실제로 안 지켜도 막는 장치가 없었다 - 실사용 세션(5db53ccb, 2026-07-29)에서
    "Step 2(공사 단계)"를 통째로 도구 호출 없이 안내하고 넘어가는 사례가
    확인됨(2026-07-29, RAG 활용도 점검 중 발견).

    disclosed_stage["construction_guide_shown"] 플래그가 아니라 _construction_guide_shown
    (메시지 이력을 직접 스캔)을 쓴다 - 그 플래그는 이 함수가 속한 guard_node
    호출의 결과로 이번 턴에 막 세워지는 값이라, 정상적으로 도구를 호출한 턴
    자체에서 플래그를 참조하면 아직 갱신 전이라 오탐이 난다.

    2026-08-03: application_submitted 확인 게이트가 아직 안 끝났으면(=
    permit_phase_directive가 실제로 지시하는 건 "Step 2 안내"가 아니라
    "신청ᆞ접수 확인 질문") 이 위반도 적용 대상이 아니다 - _should_force_
    construction_guide와 같은 이유로 같은 조건을 봐야 한다.
    """
    if state.get("disclosed_stage", {}).get(_STAGE_DOMAIN, 0) < 4:
        return []
    if not (state.get("task_progress") or {}).get("application_submitted"):
        return []
    if _construction_guide_shown(messages):
        return []
    if not any(kw in last_text for kw in _STEP2_KEYWORDS):
        return []
    return [
        "Step 1 안내가 모두 끝난 뒤 Step 2(공사) 내용을 언급했지만 get_construction_guide를 "
        "한 번도 호출하지 않았습니다. 근거 없이 착공신고ᆞ감리ᆞ사용승인 절차를 서술하지 말고, "
        "지금 그 도구를 호출해 검색된 내용으로 답변을 다시 구성하세요."
    ]


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

    막는 위반 일곱 가지:
    1. 허용된 단계 수([_max_allowed_stage]/[_max_allowed_construction_stage])를
       넘겨 [Step 1-N]/[Step 2-N]을 안내 - 판정 전 절차 안내를 아예 시도한
       경우(허용치 0)와, 판정 후 한 턴에 여러 단계를 몰아서 공개한 경우(사용자가
       "1단계만" 이라고 명시해도 무시하고 전체 단계를 다 준 사례 포함) 둘 다 이
       하나의 규칙으로 잡는다(Step2도 2026-08-04에 동일 원칙으로 확장).
    2. search_regulations/search_by_term을 한 번도 호출하지 않았는데
       [출처: ...] 인용이 있는 경우 - 근거 없이 조항을 지어낸 것.
    3. [Step 1-2]/[Step 1-3]을 언급했는데 record_permit_synthesis로 실제 기록은
       안 한 경우 - "도구를 호출했다"는 문장까지 텍스트로 지어내고 실제로는
       안 부른 사례가 실측으로 확인됨(_synthesis_gap_violations 참고).
    4. lookup_building_ledger 조회가 성공했는데 case_facts에
       current_facility_group을 안 채운 경우 - 판정이 영영 안 나서 Step1이
       멈추는 실제 사례가 반복됨(_building_ledger_gap_violations 참고).
    5. 사용자가 소방시설ᆞ간판을 직접 물었는데 아직 판정이 안 된 채 일반
       지식으로만 답한 경우 - record_fire_facts/record_signage_facts가 끝까지
       한 번도 안 불려서 로드맵에 영영 반영이 안 되는 실제 사례가 확인됨
       (_fire_signage_gap_violations 참고).
    6. Step 1 안내가 다 끝나 permit_phase_directive가 get_construction_guide
       호출을 지시하는 시점인데, 그 지시를 무시하고 "Step 2(공사)" 내용을
       도구 호출 없이 텍스트로만 설명한 경우 - RAG 검색 없이 착공ᆞ감리
       절차를 서술하게 되는 실제 사례가 확인됨(_construction_guide_gap_violations
       참고, 2026-07-29).
    7. 유저가 이미 기록된 수치를 정정하는 발화("아니ᆞ정정ᆞ다시 말하면ᆞ아까" +
       기존 기록과 다른 숫자)를 했는데 record_case_facts가 재호출 안 된 경우 -
       옛 값이 남아 잘못된 판정으로 직결될 수 있어 다른 위반들보다 오분류
       리스크가 크다(_correction_omission_violations 참고, 2026-07-31 검토ᆞ
       2026-08-01 구현).

    같은 사용자 턴 안에서 최대 1회만 재시도를 유도한다(무한 루프 방지) - 그
    이상 반복되면 프롬프트만으로는 못 막는 한계로 보고 그냥 통과시킨다.
    위반이 없거나 재시도가 소진되면, disclosed_stage[_STAGE_DOMAIN]을 갱신해서
    다음 턴의 허용치 계산에 반영한다 - 이때도 실제 언급값(max_mentioned)이
    아니라 허용치(allowed)로 캡을 씌운다. 안 씌우면 재시도가 소진돼 위반이
    그냥 통과되는 경우(예: signage 얘기만 했는데 [Step 1-1]을 잘못 언급) 허용치를
    넘는 값이 그대로 disclosed_stage에 박혀서, 이후 진짜 permit 설명이
    시작될 때 이미 일부 단계가 끝난 것처럼 잘못 판단하게 된다(2026-07-27
    실측 확인 - signage만 다룬 대화에서 disclosed_stage가 1로 오염됨).

    2026-07-28: 이 값을 "지금까지 공개된 적 있는 최댓값"(한 번 오르면 절대
    안 내려가는 래칫)에서 "이번 턴에 실제로 언급된 단계"로 바꿨다 - 래칫
    방식에선 [Step 1-3]까지 간 뒤 사용자가 [Step 1-2]를 다시 물어봐도 값이
    3에 머물러서, 바로 다음 턴에 [Step 1-3]을 건너뛰고 [Step 1-4]로 점프하는
    문제가 있었다(세션 f377454d에서 재현). 이제는 되짚어간 턴 다음엔 값이
    2로 내려가 [Step 1-3]부터 다시 자연스럽게 진행된다. max_mentioned가
    0인 턴(이번 답변에 마커가 아예 없음)은 무시한다 - 안 그러면 무관한
    답변 때문에 매번 0으로 리셋된다. 트레이드오프: 다른 도메인 답변에
    실수로 낮은 번호의 [Step 1-N]이 섞이면 이번엔 값이 잘못 내려갈 수
    있지만, 최악의 결과가 "이미 본 단계를 한 번 더 설명"하는 정도라
    이전의 오염 버그(잘못된 단계로 건너뜀)보다 훨씬 가볍다고 판단해
    지금은 별도 방어를 추가하지 않는다 - 실제로 관측되면 그때 대응한다.
    """
    messages = state["messages"]
    last_text = extract_text(messages[-1].content)

    allowed = _max_allowed_stage(state)
    mentioned = _mentioned_stages(last_text)
    max_mentioned = max(mentioned, default=0)

    mentioned2 = _mentioned_construction_stages(last_text)
    max_mentioned2 = max(mentioned2, default=0)
    allowed2 = _max_allowed_construction_stage(state)

    violations = []
    if max_mentioned > allowed:
        violations.append(
            f"이번 턴엔 [Step 1-{allowed}]까지만 안내할 수 있는데 [Step 1-{max_mentioned}]까지 "
            f"안내했습니다. 허용된 단계까지만 남기고 나머지는 삭제한 뒤, 다음에 계속 "
            f"안내해도 될지 사용자에게 짧게 물어보며 마무리하세요."
        )
    if max_mentioned2 > allowed2:
        violations.append(
            f"이번 턴엔 [Step 2-{allowed2}]까지만 안내할 수 있는데 [Step 2-{max_mentioned2}]까지 "
            f"안내했습니다. 허용된 단계까지만 남기고 나머지는 삭제한 뒤, 다음에 계속 "
            f"안내해도 될지 사용자에게 짧게 물어보며 마무리하세요."
        )
    if _CITATION_PATTERN.search(last_text) and not _has_grounding_search(messages):
        violations.append(
            "근거 도구(search_regulations/search_by_term/get_business_registration_guide 등)를 "
            "한 번도 호출하지 않았는데 [출처: ...] 인용이 포함되어 있습니다. 근거 없이 조항을 "
            "지어내지 말고, 먼저 해당 도구를 호출해 실제 근거를 확보한 뒤 답변하세요."
        )
    violations.extend(_synthesis_gap_violations(state, max_mentioned))
    violations.extend(_building_ledger_gap_violations(state, messages))
    violations.extend(_fire_signage_gap_violations(state, messages))
    violations.extend(_construction_guide_gap_violations(state, last_text, messages))
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

    update: dict = {}
    capped = min(max_mentioned, allowed)
    disclosed_stage = state.get("disclosed_stage", {})
    new_disclosed_stage = dict(disclosed_stage)
    if max_mentioned > 0 and capped != disclosed_stage.get(_STAGE_DOMAIN, 0):
        new_disclosed_stage[_STAGE_DOMAIN] = capped
    if not disclosed_stage.get("construction_guide_shown") and _construction_guide_shown(messages):
        new_disclosed_stage["construction_guide_shown"] = True
    capped2 = min(max_mentioned2, allowed2)
    if max_mentioned2 > 0 and capped2 != disclosed_stage.get("construction_stage", 0):
        new_disclosed_stage["construction_stage"] = capped2
    if new_disclosed_stage != disclosed_stage:
        update["disclosed_stage"] = new_disclosed_stage
    return update
